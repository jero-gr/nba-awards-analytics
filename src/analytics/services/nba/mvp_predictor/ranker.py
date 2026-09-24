import xgboost as xgb
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class NBAMVPRanker:
    def __init__(self, model_name='mvp_lambda_mart.json'):
        # Standalone adaptation: the original integration derived this directory from
        # Django settings.BASE_DIR; model paths are now relative to this module.
        self.model_dir = Path(__file__).resolve().parent / 'trained_models'
        self.model_name = model_name
        self.model_path = self.model_dir / model_name
        self.latest_manifest_path = self.model_dir / 'mvp_lambda_mart_latest.json'
        self.model = None

    def build_model(self, params=None):
        """
        Inicializa la arquitectura del modelo XGBRanker.
        Estos son los hiperparámetros base óptimos para un dataset de ~900 filas.
        """
        self.params = {
            'objective': 'rank:pairwise',
            'ndcg_exp_gain': False,
            'learning_rate': 0.02,    # Tasa baja para evitar sobreajuste rápido
            'max_depth': 4,           # Profundidad baja porque el dataset es pequeño
            'n_estimators': 300,      # Cantidad de árboles
            'subsample': 0.8,         # Usa 80% de las filas por árbol
            'colsample_bytree': 0.8,  # Usa 80% de las columnas por árbol
            'min_child_weight': 4,
            'reg_lambda': 1.0,
            'tree_method': 'hist',
            'random_state': 42
        }
        
        if params:
            self.params.update(params)
            
        self.model = xgb.XGBRanker(**self.params)
        return self.model

    def train(self, X_train, y_train, qid_train, X_val=None, y_val=None, qid_val=None):
        """
        Entrena el modelo LTR. 
        qid_train es un array con los IDs de grupo (ej. season_year).
        En XGBoost moderno, usar 'qid' es mucho más directo que 'group' cuando
        se tienen identificadores de sesión.
        """
        if self.model is None:
            self.build_model()

        eval_set = None
        eval_qid = None
        
        if X_val is not None and y_val is not None and qid_val is not None:
            eval_set = [(X_val, y_val)]
            eval_qid = [qid_val]

        self.model.fit(
            X_train, y_train,
            qid=qid_train, 
            eval_set=eval_set,
            eval_qid=eval_qid,
            verbose=False
        )

    def predict(self, X):
        """
        Genera los "Ranking Scores". A mayor score, más alto en el Ladder.
        """
        if self.model is None:
            self.load()
        return self.model.predict(X)

    def save(self, version=None, metadata=None):
        """Guarda un artefacto versionado y actualiza una referencia estable de ultimo modelo."""
        if self.model is None:
            raise ValueError("No hay un modelo cargado para guardar.")

        self.model_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        safe_version = ""
        if version:
            safe_version = "_" + str(version).strip().replace(" ", "_")

        versioned_name = f"mvp_lambda_mart{safe_version}_{timestamp}.json"
        versioned_path = self.model_dir / versioned_name

        self.model.save_model(str(versioned_path))
        # Mantiene compatibilidad para consumidores legacy.
        self.model.save_model(str(self.model_path))

        meta = {
            "created_at": datetime.now().isoformat(),
            "model_file": versioned_name,
            "latest_alias": self.model_name,
            "params": getattr(self, 'params', {}),
        }
        if metadata:
            meta.update(metadata)

        metadata_name = versioned_name.replace('.json', '.metadata.json')
        metadata_path = self.model_dir / metadata_name
        with open(metadata_path, 'w') as f:
            json.dump(meta, f, indent=4)

        latest_manifest = {
            "updated_at": datetime.now().isoformat(),
            "model_file": versioned_name,
            "metadata_file": metadata_name,
            # Standalone adaptation: future manifests store portable filenames rather
            # than absolute paths tied to the original runtime.
            "default_model_file": self.model_name,
        }
        with open(self.latest_manifest_path, 'w') as f:
            json.dump(latest_manifest, f, indent=4)

        logger.info("Modelo guardado en %s", versioned_path)
        return {
            "model_path": str(versioned_path),
            "default_model_path": str(self.model_path),
            "metadata_path": str(metadata_path),
            "manifest_path": str(self.latest_manifest_path),
        }

    def _resolve_model_path(self, version=None):
        if version:
            explicit_path = self.model_dir / version
            if explicit_path.exists():
                return explicit_path
            supplied_path = Path(version)
            if supplied_path.exists():
                return supplied_path
            raise FileNotFoundError(f"No se encontro el modelo solicitado: {version}")

        if self.latest_manifest_path.exists():
            with open(self.latest_manifest_path, 'r') as f:
                manifest = json.load(f)
            candidate_name = manifest.get("model_file")
            if candidate_name:
                candidate = self.model_dir / candidate_name
                if candidate.exists():
                    return candidate

        if self.model_path.exists():
            return self.model_path

        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"No existe el directorio de modelos: {self.model_dir}")

        candidates = [
            path
            for path in self.model_dir.iterdir()
            if path.name.startswith("mvp_lambda_mart")
            and path.name.endswith(".json")
            and not path.name.endswith(".metadata.json")
        ]
        if candidates:
            candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            return candidates[0]

        raise FileNotFoundError(f"No se encontro un modelo entrenado en: {self.model_dir}")

    def load(self, version=None):
        """Carga el modelo desde disco. Si no se especifica version, intenta cargar el ultimo disponible."""
        model_path = self._resolve_model_path(version=version)
        
        self.model = xgb.XGBRanker()
        self.model.load_model(str(model_path))
        logger.info("Modelo cargado desde %s", model_path)
        return str(model_path)
