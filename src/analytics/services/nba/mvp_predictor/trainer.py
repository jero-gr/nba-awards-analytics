import json
import logging
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from django.conf import settings
from sklearn.metrics import ndcg_score

from .features import NBAFeatureEngineer
from .ranker import NBAMVPRanker

logger = logging.getLogger(__name__)

class NBAMVPTrainer:
    def __init__(self, season_start=None, season_end=None, current_season=2026):
        self.engineer = NBAFeatureEngineer()
        self.ranker = NBAMVPRanker()
        self.season_start = season_start
        self.season_end = season_end
        self.current_season = current_season
        self.last_cv_summary = None
        
        # Directorio para guardar reportes de entrenamiento
        self.reports_dir = os.path.join(
            settings.BASE_DIR, 'analytics', 'services', 'nba', 'mvp_predictor', 'reports'
        )
        os.makedirs(self.reports_dir, exist_ok=True)
        self._validate_training_window()

    def _validate_training_window(self):
        if (self.season_start is None) != (self.season_end is None):
            raise ValueError("Debes enviar season_start y season_end juntos, o ninguno.")
        if self.season_start is not None and self.season_end is not None and self.season_start > self.season_end:
            raise ValueError(
                f"Rango de entrenamiento invalido: season_start={self.season_start} > season_end={self.season_end}."
            )

    def _apply_training_window(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.season_start is None:
            return df
        return df[
            (df['season_year'] >= self.season_start) & (df['season_year'] <= self.season_end)
        ].copy()

    def _build_training_df(self) -> pd.DataFrame:
        df = self.engineer.prepare_training_dataset(current_season=self.current_season)
        if df is None or df.empty:
            raise ValueError("No hay datos para entrenar despues del feature engineering.")

        df = self._apply_training_window(df)
        if df.empty:
            raise ValueError(
                f"No hay datos en el rango solicitado: {self.season_start}-{self.season_end}."
            )
        return df

    @staticmethod
    def _map_feature_importance(model, feature_names):
        importances = model.get_booster().get_score(importance_type='gain')
        mapped_importances = {}
        for i, name in enumerate(feature_names):
            key = f'f{i}'
            if key in importances:
                mapped_importances[name] = importances[key]

        if not mapped_importances:
            mapped_importances = {k: v for k, v in importances.items() if k in feature_names}

        return sorted(mapped_importances.items(), key=lambda x: x[1], reverse=True)

    def _prepare_data(self, df):
        """
        Separa X, y y qid del DataFrame de entrenamiento.
        """
        # Identificar columnas que NO son features
        exclude_cols = [
            'player_id', 'player_ext_id', 'team_id', 'season_year', 
            'won_mvp', 'mvp_share', 'mvp_share_weighted', 'all_nba_share'
        ]
        
        # Nos aseguramos de que solo usamos columnas numéricas para X
        X = df.drop(columns=exclude_cols, errors='ignore')
        # Por seguridad, nos quedamos solo con las columnas que terminan en _zscore o inv_conference_rank
        # o que hayamos definido explícitamente como features en el futuro
        X = X.select_dtypes(include=[np.number])
        
        if 'mvp_share_weighted' not in df.columns:
            raise ValueError("Falta la columna mvp_share_weighted en el dataset de entrenamiento.")

        y = (df['mvp_share_weighted'] * 100).astype(int)
        qid = df['season_year']

        if X.empty:
            raise ValueError("La matriz de features quedo vacia tras el preprocesamiento.")
        
        return X, y, qid

    def cross_validate(self):
        """
        Ejecuta Leave-One-Season-Out (LOSO) Cross-Validation y genera un reporte.
        """
        t0 = time.perf_counter()
        df = self._build_training_df()

        # Para mapear IDs a nombres en el reporte
        from sync.models import Player
        player_names = dict(
            Player.objects.filter(id__in=df['player_id'].unique()).values_list('id', 'name')
        )
        df['player_name'] = df['player_id'].map(player_names)

        seasons = sorted(df['season_year'].unique())
        if len(seasons) < 2:
            raise ValueError(
                f"LOSO requiere al menos 2 temporadas. Temporadas disponibles: {len(seasons)}"
            )

        results = []

        logger.info("Iniciando LOSO CV en %s temporadas...", len(seasons))

        for season in seasons:
            # Split
            train_df = df[df['season_year'] != season]
            test_df = df[df['season_year'] == season].copy()
            
            X_train, y_train, qid_train = self._prepare_data(train_df)
            X_test, y_test, qid_test = self._prepare_data(test_df)
            
            # Train (instancia fresca por iteración)
            model = NBAMVPRanker()
            model.build_model()
            model.train(X_train, y_train, qid_train)
            
            # Feature Importance for this fold
            feature_names = X_train.columns.tolist()
            sorted_importance = self._map_feature_importance(model.model, feature_names)
            top_10_features = " | ".join([f"{name} ({score:.2f})" for name, score in sorted_importance[:10]])

            # Predict
            preds = model.predict(X_test)
            test_df['pred_score'] = preds
            
            # Métricas
            # 1. NDCG@5 (sklearn espera arrays 2D [n_samples, n_items])
            n_ndcg = float(ndcg_score([y_test.values], [preds], k=5))
            
            # 2. Rango predicho del MVP real
            # Encontramos al ganador real (el que tiene más share)
            real_winner_id = test_df.sort_values(by='mvp_share_weighted', ascending=False).iloc[0]['player_id']
            # Ordenamos por predicción
            pred_sorted_df = test_df.sort_values(by='pred_score', ascending=False).reset_index(drop=True)
            # Encontramos en qué fila (rank) quedó el ganador real (sumamos 1 para que sea base-1)
            mvp_pred_rank = pred_sorted_df[pred_sorted_df['player_id'] == real_winner_id].index[0] + 1
            
            top1_hit = 1 if mvp_pred_rank == 1 else 0
            
            # Top 10 Arrays para el reporte
            real_top10 = test_df.sort_values(by='mvp_share_weighted', ascending=False).head(10)['player_name'].tolist()
            pred_top10 = pred_sorted_df.head(10)['player_name'].tolist()
            
            results.append({
                'season': season,
                'ndcg_5': n_ndcg,
                'top1_hit': top1_hit,
                'mvp_pred_rank': int(mvp_pred_rank),
                'real_top10': " | ".join([f"{i+1}. {n}" for i, n in enumerate(real_top10)]),
                'pred_top10': " | ".join([f"{i+1}. {n}" for i, n in enumerate(pred_top10)]),
                'top_10_features': top_10_features
            })
            
            status = f"Rank {mvp_pred_rank}"
            logger.info("Temporada %s: NDCG@5=%.4f | MVP Pred: %s", season, n_ndcg, status)

        # Promedios
        avg_ndcg = float(np.mean([r['ndcg_5'] for r in results]))
        avg_top1 = float(np.mean([r['top1_hit'] for r in results]))
        avg_rank = float(np.mean([r['mvp_pred_rank'] for r in results]))
        
        # --- NUEVO: GUARDAR REPORTES ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Guardar CSV detallado
        results_df = pd.DataFrame(results)
        csv_path = os.path.join(self.reports_dir, f'loso_results_{timestamp}.csv')
        results_df.to_csv(csv_path, index=False)
        
        # 2. Guardar JSON de resumen con hiperparámetros
        # Obtenemos los hiperparámetros creando una instancia limpia
        dummy_model = NBAMVPRanker()
        dummy_model.build_model()
        
        summary = {
            "timestamp": timestamp,
            "dataset": {
                "total_rows": len(df),
                "seasons_count": len(seasons),
                "features_count": X_train.shape[1]
            },
            "training_window": {
                "season_start": self.season_start,
                "season_end": self.season_end,
            },
            "metrics": {
                "avg_ndcg_5": avg_ndcg,
                "avg_top1_accuracy": avg_top1,
                "avg_mvp_pred_rank": avg_rank
            },
            "hyperparameters": dummy_model.params
        }
        
        json_path = os.path.join(self.reports_dir, f'summary_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=4)

        self.last_cv_summary = summary
        self.last_cv_summary["paths"] = {
            "csv": csv_path,
            "json": json_path,
        }

        logger.info("RESULTADOS FINALES DE VALIDACION")
        logger.info("Promedio NDCG@5: %.4f", avg_ndcg)
        logger.info("Promedio Top-1 Accuracy: %.2f%%", avg_top1 * 100)
        logger.info("Rango Promedio del MVP Real: %.1f", avg_rank)
        logger.info("Reportes guardados en: %s", self.reports_dir)
        logger.info(" - CSV: %s", os.path.basename(csv_path))
        logger.info(" - JSON: %s", os.path.basename(json_path))
        logger.info("LOSO finalizado en %.2fs", time.perf_counter() - t0)
        
        return results

    def train_final_model(self, save_model=True, model_version=None, cv_summary=None):
        """
        Entrena el modelo con todos los datos históricos y lo guarda para producción.
        """
        t0 = time.perf_counter()
        df = self._build_training_df()
        X, y, qid = self._prepare_data(df)
        
        self.ranker.build_model()
        self.ranker.train(X, y, qid)
        model_artifact = None
        if save_model:
            metadata = {
                "training_window": {
                    "season_start": self.season_start,
                    "season_end": self.season_end,
                },
                "dataset": {
                    "total_rows": int(len(df)),
                    "features_count": int(X.shape[1]),
                    "seasons_count": int(df['season_year'].nunique()),
                },
                "cross_validation": (cv_summary or self.last_cv_summary or {}).get("metrics", {}),
            }
            model_artifact = self.ranker.save(version=model_version, metadata=metadata)
            logger.info("Modelo versionado guardado: %s", model_artifact.get("model_path"))
        else:
            logger.info("Entrenamiento final completado sin guardar modelo (save_model=False).")
        
        # --- NUEVO: Mostrar importancia de variables del modelo GLOBAL ---
        feature_names = X.columns.tolist()
        sorted_importance = self._map_feature_importance(self.ranker.model, feature_names)
        
        # --- Guardar importancia global en JSON ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        importance_dict = {name: float(score) for name, score in sorted_importance}
        json_path = os.path.join(self.reports_dir, f'final_model_features_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump({"timestamp": timestamp, "global_feature_importance_gain": importance_dict}, f, indent=4)
        
        logger.info("TOP 15 VARIABLES - MODELO GLOBAL")
        for name, score in sorted_importance[:15]:
            logger.info("%s: %.4f", name, score)

        logger.info("Modelo final entrenado con %s filas.", len(df))
        logger.info("Importancia de variables guardada en: %s", os.path.basename(json_path))
        logger.info("Entrenamiento final completado en %.2fs", time.perf_counter() - t0)

        return {
            "rows": int(len(df)),
            "saved_model": bool(save_model),
            "model_artifact": model_artifact,
            "feature_importance_path": json_path,
            "training_window": {
                "season_start": self.season_start,
                "season_end": self.season_end,
            },
        }
