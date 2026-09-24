import pandas as pd
import logging
from django.db import connection

logger = logging.getLogger(__name__)

class NBADataLoader:
    """
    Carga de datos optimizada mediante SQL directo para entrenamiento de ML.
    """

    @staticmethod
    def _execute_query(query: str, params: list = None) -> pd.DataFrame:
        """Ejecuta SQL y retorna un DataFrame de Pandas."""
        try:
            df = pd.read_sql_query(query, connection, params=params)
            logger.debug("Query ejecutada correctamente. Filas: %s", len(df))
            return df
        except Exception as exc:
            logger.exception("Error ejecutando query SQL para MVP predictor")
            raise RuntimeError("Fallo la carga de datos para entrenamiento MVP") from exc

    def get_player_season_stats(self, season: str = None) -> pd.DataFrame:
        """Obtiene estadísticas de jugadores unidas con su información básica."""
        query = """
            SELECT 
                p.external_id as player_ext_id, p.name as player_name,
                s.*
            FROM bbref_player_season_stats s
            JOIN players p ON s.player_id = p.id
        """
        if season:
            query += " WHERE s.season = %s"
            return self._execute_query(query, [season])
        return self._execute_query(query)

    def get_team_season_stats(self, season: str = None) -> pd.DataFrame:
        """Obtiene estadísticas de equipos."""
        query = """
            SELECT t.name as team_name, t.abbreviation, s.*
            FROM bbref_team_season_stats s
            JOIN teams t ON s.team_id = t.id
        """
        if season:
            query += " WHERE s.season = %s"
            return self._execute_query(query, [season])
        return self._execute_query(query)

    def get_award_stats(self, season: str = None) -> pd.DataFrame:
        """Obtiene las votaciones históricas (nuestro 'Target' para el modelo)."""
        query = "SELECT * FROM bbref_player_award_stats"
        if season:
            query += " WHERE season = %s"
            return self._execute_query(query, [season])
        return self._execute_query(query)

    def get_mvp_dataset(self, season: str = None) -> pd.DataFrame:
        """
        EL GRAN JOIN: Une stats de jugador + stats de su equipo + votos de MVP.
        Este es el dataset listo para el modelo.
        """
        query = """
            SELECT 
                p.name as player_name,
                t.name as team_name,
                ps.*,
                ts.wins as team_wins, ts.win_loss_pct as team_win_pct, 
                ts.conference_rank as team_conf_rank,
                COALESCE(aw.mvp_share, 0) as target_mvp_share
            FROM bbref_player_season_stats ps
            JOIN players p ON ps.player_id = p.id
            JOIN teams t ON ps.team_id = t.id
            JOIN bbref_team_season_stats ts ON ps.team_id = ts.team_id AND ps.season = ts.season
            LEFT JOIN bbref_player_award_stats aw ON ps.player_id = aw.player_id AND ps.season = aw.season
        """
        if season:
            query += " WHERE ps.season = %s"
            return self._execute_query(query, [season])
        return self._execute_query(query)
