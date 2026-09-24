import pandas as pd
import numpy as np
import logging
from .data_loader import NBADataLoader

logger = logging.getLogger(__name__)

class NBAFeatureEngineer:
    def __init__(self):
        self.loader = NBADataLoader()

    def get_raw_datasets(self):
        """Retorna los tres DataFrames locales sin transformar."""
        df_player_season = self.loader.get_player_season_stats()
        df_team_season = self.loader.get_team_season_stats()
        df_award_stats = self.loader.get_award_stats()
        return df_player_season, df_team_season, df_award_stats

    def clean_and_normalize_player_stats(self, df_player_season, mpg_threshold=25, gp_threshold=0.7, current_season=2026):
        """
        Limpia el dataset de jugadores y aplica normalización Z-score por temporada.
        Incluye la regla de 65 partidos para elegibilidad a partir de 2024.
        """
        df = df_player_season.copy()

        # Rellenar nulos con 0 (asumiendo que si no hay stats, es porque no se logró esa acción)
        df.fillna(0, inplace=True)

        # Convertir temporada a entero (año de finalización)
        df['season_year'] = df['season'].str[:4].astype(int) + 1

        self.columns_to_use = [
            # Standalone adaptation: the original integration resolved names through
            # sync.models.Player; the local dataset carries player_name as display metadata.
            'player_id', 'player_ext_id', 'player_name', 'team_id', 'season_year', 'gp', 'fg_pct', 'fg3_pct', 'fg2_pct', 'efg_pct', 'ft_pct', 'trp_dbl',
            #'fgm_per_poss', 'fga_per_poss', 'fg3m_per_poss', 'fg3a_per_poss', 'fg2m_per_poss', 'fg2a_per_poss', 'ftm_per_poss', 'fta_per_poss',
            #'oreb_per_poss', 'dreb_per_poss', 'reb_per_poss', 'ast_per_poss', 'stl_per_poss', 'blk_per_poss', 'tov_per_poss', 'pf_per_poss', 'pts_per_poss',
            'ortg', 'drtg', 'per', 'ts_pct', 'fg3_attempt_rate', 'ft_attempt_rate', 'oreb_pct', 'dreb_pct', 'reb_pct', 'ast_pct', 'stl_pct', 'blk_pct', 'tov_pct',
            'usg_pct', 'ows', 'dws', 'win_shares', 'win_shares_per_48', 'obpm', 'dbpm', 'bpm', 'vorp', 
            'on_court_pm_per_100', 'net_pm_per_100', 
            'min_total_per_game', 'fgm_per_game', 'fga_per_game', 'fg3m_per_game', 'fg3a_per_game', 'fg2m_per_game', 'fg2a_per_game', 'ftm_per_game', 'fta_per_game',
            'oreb_per_game', 'dreb_per_game', 'reb_per_game', 'ast_per_game', 'stl_per_game', 'blk_per_game', 'tov_per_game', 'pf_per_game', 'pts_per_game',
            'pf_shooting_per_game', 'pf_offball_per_game', 'pts_gen_by_ast_per_game', 'and1_per_game',
            'blocked_attempts_per_game'
        ]
        
        self.columns_to_normalize = [
            'gp', 'fg_pct', 'fg3_pct', 'fg2_pct', 'efg_pct', 'ft_pct', 'trp_dbl',
            #'fgm_per_poss', 'fga_per_poss', 'fg3m_per_poss', 'fg3a_per_poss', 'fg2m_per_poss', 'fg2a_per_poss', 'ftm_per_poss', 'fta_per_poss',
            #'oreb_per_poss', 'dreb_per_poss', 'reb_per_poss', 'ast_per_poss', 'stl_per_poss', 'blk_per_poss', 'tov_per_poss', 'pf_per_poss', 'pts_per_poss',
            'ortg', 'drtg', 'per', 'ts_pct', 'fg3_attempt_rate', 'ft_attempt_rate', 'oreb_pct', 'dreb_pct', 'reb_pct', 'ast_pct', 'stl_pct', 'blk_pct', 'tov_pct',
            'usg_pct', 'ows', 'dws', 'win_shares', 'win_shares_per_48', 'obpm', 'dbpm', 'bpm', 'vorp', 
            'on_court_pm_per_100', 'net_pm_per_100', 
            'min_total_per_game', 'fgm_per_game', 'fga_per_game', 'fg3m_per_game', 'fg3a_per_game', 'fg2m_per_game', 'fg2a_per_game', 'ftm_per_game', 'fta_per_game',
            'oreb_per_game', 'dreb_per_game', 'reb_per_game', 'ast_per_game', 'stl_per_game', 'blk_per_game', 'tov_per_game', 'pf_per_game', 'pts_per_game',
            'pf_shooting_per_game', 'pf_offball_per_game', 'pts_gen_by_ast_per_game', 'and1_per_game',
            'blocked_attempts_per_game'
        ]

        # Filtro de limpieza basado en MPG y % de partidos jugados
        max_gp_per_season = df.groupby('season_year')['gp'].transform('max')
        mask = (df['min_total_per_game'] >= mpg_threshold) & \
               ((df['gp'] / max_gp_per_season) >= gp_threshold)
        
        # Nueva regla: 65 partidos para temporadas >= 2024 (excepto la actual)
        nba_65_mask = (df['season_year'] >= 2024) & (df['season_year'] != current_season)
        mask = mask & ~(nba_65_mask & (df['gp'] < 65))
        
        df_cleaned = df[mask].copy()

        # Cálculo de Z-Score por temporada
        if self.columns_to_normalize:
            for col in self.columns_to_normalize:
                means = df_cleaned.groupby('season_year')[col].transform('mean')
                stds = df_cleaned.groupby('season_year')[col].transform('std')
                df_cleaned[f'{col}_zscore'] = (df_cleaned[col] - means) / stds

        # Selección de columnas: No normalizadas de to_use + las normalizadas con sufijo _zscore
        if self.columns_to_use:
            non_normalized = [c for c in self.columns_to_use if c not in self.columns_to_normalize]
            normalized_zscores = [f'{c}_zscore' for c in self.columns_to_normalize]
            df_cleaned = df_cleaned[non_normalized + normalized_zscores]

        # Ordenamiento final
        df_cleaned = df_cleaned.sort_values(by='season_year', ascending=True)

        return df_cleaned

    def clean_and_normalize_team_stats(self, df_team_season):
        """
        Limpia el dataset de equipos y aplica normalización Z-score por temporada.
        """
        df = df_team_season.copy()

        # Convertir temporada a entero
        df['season_year'] = df['season'].str[:4].astype(int) + 1

        # Invertir conference rank para deslinealizar (rank 1 = mejor equipo, pero queremos que sea un valor alto)
        df['inv_conference_rank'] = 1 / df['conference_rank'].replace(0, 15)

        # Definición de columnas para equipos
        self.team_columns_to_use = [
            'team_id', 'season_year', 'inv_conference_rank', 'wins', 'srs', 'margin_of_victory', 'strength_of_schedule',
            'ortg', 'drtg', 'nrtg', 
        ]
        
        self.team_columns_to_normalize = [
            'wins', 'srs', 'margin_of_victory', 'strength_of_schedule',
            'ortg', 'drtg', 'nrtg',
        ]

        # En equipos no solemos aplicar filtros de MPG/GP, usamos todos los de la temporada
        df_cleaned = df.copy()

        # Cálculo de Z-Score por temporada
        if self.team_columns_to_normalize:
            for col in self.team_columns_to_normalize:
                means = df_cleaned.groupby('season_year')[col].transform('mean')
                stds = df_cleaned.groupby('season_year')[col].transform('std')
                df_cleaned[f'{col}_zscore'] = (df_cleaned[col] - means) / stds

        # Selección de columnas: No normalizadas de to_use + las normalizadas con sufijo _zscore
        if self.team_columns_to_use:
            non_normalized = [c for c in self.team_columns_to_use if c not in self.team_columns_to_normalize]
            normalized_zscores = [f'{c}_zscore' for c in self.team_columns_to_normalize]
            df_cleaned = df_cleaned[non_normalized + normalized_zscores]

        # Ordenamiento final
        df_cleaned = df_cleaned.sort_values(by='season_year', ascending=True)

        return df_cleaned

    def clean_award_stats(self, df_award_stats, scale_factor=1, power=1):
        """
        Limpia el dataset de premios y aplica una transformación de potencia al mvp_share.
        Incluye All-NBA share para selección de candidatos.
        """
        df = df_award_stats.copy()

        # Convertir temporada a entero (año de finalización)
        df['season_year'] = df['season'].str[:4].astype(int) + 1

        # Peso combinado por temporada: proporción de votos 1º lugar + mvp_share
        total_mvp_first_by_season = df.groupby('season_year')['mvp_first'].transform('sum')
        mvp_first_ratio = np.where(total_mvp_first_by_season > 0, df['mvp_first'] / total_mvp_first_by_season, 0.0)
        df['mvp_share_weighted'] = mvp_first_ratio * scale_factor + (df['mvp_share'] ** power) 

        # Retornar solo las columnas necesarias para el merge
        return df[['player_id', 'season_year', 'mvp_share', 'mvp_share_weighted', 'all_nba_share']]

    def get_mvp_winners(self, df_award_stats):
        """
        Identifica quién ganó el MVP en cada temporada basándose en el mayor mvp_share.
        Retorna un DataFrame con player_id, season_year y un booleano won_mvp.
        """
        df = df_award_stats.copy()
        
        # Asegurarnos de tener season_year
        if 'season_year' not in df.columns:
            df['season_year'] = df['season'].str[:4].astype(int) + 1
            
        # Encontrar el índice del jugador con el máximo mvp_share para cada temporada
        idx_winners = df.groupby('season_year')['mvp_share'].idxmax()
        
        # Crear columna booleana
        df['won_mvp'] = False
        df.loc[idx_winners, 'won_mvp'] = True
        
        return df[['player_id', 'season_year', 'won_mvp']]

    def calculate_voting_fatigue(self, df_winners):
        """
        Calcula cuántos MVPs previos y consecutivos ha ganado un jugador hasta antes de la temporada actual.
        """
        df = df_winners.copy()
        
        # 1. Crear un diccionario de {season_year: player_id} con los ganadores históricos
        # Esto permite búsquedas O(1) y maneja perfectamente años donde un jugador no recibió votos
        winners_by_year = df[df['won_mvp'] == True].set_index('season_year')['player_id'].to_dict()
        
        total_previous_mvps = []
        consecutive_mvps = []
        
        for _, row in df.iterrows():
            p_id = row['player_id']
            s_year = row['season_year']
            
            # Contar MVPs ganados en años estrictamente anteriores
            total = sum(1 for y, winner_id in winners_by_year.items() if y < s_year and winner_id == p_id)
            total_previous_mvps.append(total)
            
            # Contar MVPs consecutivos ganados inmediatamente antes de esta temporada
            consecutive = 0
            check_year = s_year - 1
            while winners_by_year.get(check_year) == p_id:
                consecutive += 1
                check_year -= 1
            consecutive_mvps.append(consecutive)
            
        df['total_previous_mvps'] = total_previous_mvps
        df['consecutive_mvps'] = consecutive_mvps
        
        return df[['player_id', 'season_year', 'won_mvp', 'total_previous_mvps', 'consecutive_mvps']]

    def get_final_training_df(self, df_player_cleaned, df_team_cleaned, df_fatigue, df_awards_cleaned):
        """
        Une todos los DataFrames procesados en una matriz final de entrenamiento.
        """
        # 1. Unir jugadores con sus equipos (Left Join)
        df_final = pd.merge(
            df_player_cleaned,
            df_team_cleaned,
            on=['team_id', 'season_year'],
            how='left'
        )

        # 2. Unir con las métricas de fatiga (Left Join)
        df_final = pd.merge(
            df_final,
            df_fatigue,
            on=['player_id', 'season_year'],
            how='left'
        )

        # Rellenar nulos de fatiga (jugadores que nunca ganaron)
        df_final['won_mvp'] = df_final['won_mvp'].fillna(False)
        df_final['total_previous_mvps'] = df_final['total_previous_mvps'].fillna(0).astype(int)
        df_final['consecutive_mvps'] = df_final['consecutive_mvps'].fillna(0).astype(int)

        # 3. Unir con el target de premios (Left Join)
        df_final = pd.merge(
            df_final,
            df_awards_cleaned,
            on=['player_id', 'season_year'],
            how='left'
        )

        # Rellenar los shares con 0 para los jugadores que no recibieron votos
        df_final['mvp_share'] = df_final['mvp_share'].fillna(0.0)
        df_final['mvp_share_weighted'] = df_final['mvp_share_weighted'].fillna(0.0)
        df_final['all_nba_share'] = df_final['all_nba_share'].fillna(0.0)

        # 4. Ordenamiento final: Agrupar por año y poner a los candidatos arriba
        df_final = df_final.sort_values(
            by=['season_year', 'mvp_share'], 
            ascending=[True, False]
        )

        return df_final

    def apply_ltr_top_k_filter(self, df_final, k=30):
        """
        Reduce el dataset a los K mejores candidatos por temporada para estabilizar 
        el algoritmo Learning to Rank (LTR).
        
        Prioridad de selección:
        1. Jugadores con votos al MVP (mvp_share > 0)
        2. Jugadores con votos al All-NBA (all_nba_share > 0)
        3. Relleno con líderes en VORP (vorp_zscore) hasta completar K.
        """
        filtered_dfs = []
        
        # Agrupamos por temporada
        for season, group in df_final.groupby('season_year'):
            # 1. Candidatos con votos al MVP
            mvp_voted = group[group['mvp_share'] > 0]
            
            # 2. Candidatos con votos al All-NBA pero SIN votos al MVP
            all_nba_voted = group[(group['all_nba_share'] > 0) & (group['mvp_share'] == 0)]
            
            # 3. Resto de jugadores (sin votos de ningún tipo)
            no_votes = group[(group['mvp_share'] == 0) & (group['all_nba_share'] == 0)]
            
            # Consolidamos los candidatos prioritarios (Prioridad 1 + 2)
            candidates = pd.concat([mvp_voted, all_nba_voted])
            
            # Si aún no llegamos a K, rellenamos con los mejores VORP de los que no tienen votos
            if len(candidates) < k:
                fill_count = k - len(candidates)
                fillers = no_votes.sort_values(by='vorp_zscore', ascending=False).head(fill_count)
                top_k_group = pd.concat([candidates, fillers])
            else:
                # Si por alguna razón (poco probable) hay más de K entre MVP + All-NBA, 
                # tomamos los mejores K basándonos en mvp_share y luego all_nba_share
                top_k_group = candidates.sort_values(
                    by=['mvp_share', 'all_nba_share'], 
                    ascending=[False, False]
                ).head(k)
            
            filtered_dfs.append(top_k_group)
            
        # Volvemos a unir todas las temporadas y ordenamos
        df_filtered = pd.concat(filtered_dfs)
        return df_filtered.sort_values(by=['season_year', 'mvp_share'], ascending=[True, False])

    def prepare_training_dataset(self, mpg_threshold=25, gp_threshold=0.7, scale_factor=2, power=2, current_season=2026):
        """
        Orquestador principal: ejecuta todo el pipeline desde la carga de datos
        hasta la generación del DataFrame final de entrenamiento.
        Excluye la temporada actual ya que no tiene resultados de votación.
        """
        try:
            # 1. Carga de datos crudos
            df_player_season, df_team_season, df_award_stats = self.get_raw_datasets()

            if df_player_season.empty or df_team_season.empty or df_award_stats.empty:
                logger.warning("Dataset vacio detectado durante prepare_training_dataset.")
                return pd.DataFrame()

            # 2. Procesamiento y normalización
            df_player_cleaned = self.clean_and_normalize_player_stats(df_player_season, mpg_threshold, gp_threshold, current_season)
            df_team_cleaned = self.clean_and_normalize_team_stats(df_team_season)
            df_awards_cleaned = self.clean_award_stats(df_award_stats, scale_factor, power)

            # 3. Análisis de fatiga de voto
            df_winners = self.get_mvp_winners(df_award_stats)
            df_fatigue = self.calculate_voting_fatigue(df_winners)

            # 4. Ensamble final de la matriz
            df_final = self.get_final_training_df(
                df_player_cleaned,
                df_team_cleaned,
                df_fatigue,
                df_awards_cleaned
            )

            # 5. Filtrar temporada actual (Solo queremos datos históricos para entrenamiento)
            df_training = df_final[df_final['season_year'] < current_season].copy()

            # 6. Aplicar corte estricto para LTR (30 jugadores por año)
            df_training = self.apply_ltr_top_k_filter(df_training, k=30)

            logger.info(
                "Dataset de entrenamiento preparado. Filas=%s, temporadas=%s",
                len(df_training),
                df_training['season_year'].nunique() if not df_training.empty else 0,
            )
            return df_training
        except Exception:
            logger.exception("Error preparando dataset de entrenamiento MVP")
            raise
