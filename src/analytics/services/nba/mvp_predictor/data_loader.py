import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# Standalone adaptation: the original integration read these datasets from private
# PostgreSQL tables through Django; this repository reads equivalent local CSV inputs instead.
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / 'data'


class NBADataLoader:
    """Load the three local datasets required by the MVP feature pipeline."""

    DATASET_FILES = {
        'player_season': 'player_season_stats.csv',
        'team_season': 'team_season_stats.csv',
        'award_stats': 'award_stats.csv',
    }

    REQUIRED_COLUMNS = {
        'player_season': {
            'player_id', 'player_ext_id', 'player_name', 'team_id', 'season', 'gp',
            'fg_pct', 'fg3_pct', 'fg2_pct', 'efg_pct', 'ft_pct', 'trp_dbl', 'ortg',
            'drtg', 'per', 'ts_pct', 'fg3_attempt_rate', 'ft_attempt_rate', 'oreb_pct',
            'dreb_pct', 'reb_pct', 'ast_pct', 'stl_pct', 'blk_pct', 'tov_pct', 'usg_pct',
            'ows', 'dws', 'win_shares', 'win_shares_per_48', 'obpm', 'dbpm', 'bpm',
            'vorp', 'on_court_pm_per_100', 'net_pm_per_100', 'min_total_per_game',
            'fgm_per_game', 'fga_per_game', 'fg3m_per_game', 'fg3a_per_game',
            'fg2m_per_game', 'fg2a_per_game', 'ftm_per_game', 'fta_per_game',
            'oreb_per_game', 'dreb_per_game', 'reb_per_game', 'ast_per_game',
            'stl_per_game', 'blk_per_game', 'tov_per_game', 'pf_per_game', 'pts_per_game',
            'pf_shooting_per_game', 'pf_offball_per_game', 'pts_gen_by_ast_per_game',
            'and1_per_game', 'blocked_attempts_per_game',
        },
        'team_season': {
            'team_id', 'season', 'wins', 'losses', 'conference_rank', 'srs',
            'margin_of_victory', 'strength_of_schedule', 'ortg', 'drtg', 'nrtg',
        },
        'award_stats': {
            'player_id', 'season', 'mvp_first', 'mvp_share', 'all_nba_share',
        },
    }

    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR

    def _load_dataset(self, dataset_name: str, season: str = None) -> pd.DataFrame:
        path = self.data_dir / self.DATASET_FILES[dataset_name]
        if not path.is_file():
            raise FileNotFoundError(
                f"Required local CSV not found: {path}. "
                "Training and inference require the local datasets documented in README.md."
            )

        df = pd.read_csv(path)
        missing_columns = sorted(self.REQUIRED_COLUMNS[dataset_name] - set(df.columns))
        if missing_columns:
            raise ValueError(
                f"Local CSV {path} is missing required columns: {', '.join(missing_columns)}"
            )

        if season is not None:
            df = df[df['season'].astype(str) == str(season)].copy()

        logger.debug("Loaded %s rows from %s", len(df), path)
        return df

    def get_player_season_stats(self, season: str = None) -> pd.DataFrame:
        return self._load_dataset('player_season', season=season)

    def get_team_season_stats(self, season: str = None) -> pd.DataFrame:
        return self._load_dataset('team_season', season=season)

    def get_award_stats(self, season: str = None) -> pd.DataFrame:
        return self._load_dataset('award_stats', season=season)
