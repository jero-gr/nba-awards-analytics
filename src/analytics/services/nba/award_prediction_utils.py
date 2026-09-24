import re

import numpy as np
import pandas as pd


SEASON_STR_RE = re.compile(r"^(\d{4})-(\d{2})$")


def parse_season_end_year(season_value):
    """Accepts 2026, '2026', or '2025-26' and returns end-year int."""
    if isinstance(season_value, int):
        season_year = season_value
    else:
        raw = str(season_value).strip()
        if raw.isdigit() and len(raw) == 4:
            season_year = int(raw)
        else:
            match = SEASON_STR_RE.match(raw)
            if not match:
                raise ValueError(
                    "Formato de season invalido. Usa YYYY (ej: 2026) o YYYY-YY (ej: 2025-26)."
                )
            start_year = int(match.group(1))
            end_two = int(match.group(2))
            expected_two = (start_year + 1) % 100
            if end_two != expected_two:
                raise ValueError(
                    f"Temporada invalida: {raw}. El sufijo esperado es {expected_two:02d}."
                )
            season_year = start_year + 1

    if season_year < 1980 or season_year > 2100:
        raise ValueError(f"Season fuera de rango soportado: {season_year}.")

    return season_year


def apply_projected_65_game_filter(df_player_season, df_team_season, current_season):
    """
    Excludes players in the selected season who can no longer mathematically reach 65 GP.
    Returns (filtered_players_df, excluded_player_ids).
    """
    if df_player_season.empty or df_team_season.empty:
        return df_player_season, []

    player = df_player_season.copy()
    team = df_team_season.copy()

    player['temp_season_year'] = player['season'].str[:4].astype(int) + 1
    team['temp_season_year'] = team['season'].str[:4].astype(int) + 1

    df_p_curr = player[player['temp_season_year'] == current_season].copy()
    df_t_curr = team[team['temp_season_year'] == current_season].copy()

    if df_p_curr.empty or df_t_curr.empty:
        player = player.drop(columns=['temp_season_year'], errors='ignore')
        return player, []

    required_team_cols = {'team_id', 'wins', 'losses'}
    if not required_team_cols.issubset(df_t_curr.columns):
        player = player.drop(columns=['temp_season_year'], errors='ignore')
        return player, []

    df_calc = pd.merge(
        df_p_curr[['player_id', 'team_id', 'gp']],
        df_t_curr[['team_id', 'wins', 'losses']],
        on='team_id',
        how='left',
    )

    df_calc['remaining_games'] = 82 - (df_calc['wins'].fillna(0) + df_calc['losses'].fillna(0))
    df_calc['remaining_games'] = df_calc['remaining_games'].clip(lower=0)
    df_calc['max_possible_gp'] = df_calc['gp'].fillna(0) + df_calc['remaining_games']

    ineligible_player_ids = (
        df_calc[df_calc['max_possible_gp'] < 64]['player_id'].dropna().astype(int).unique().tolist()
    )

    if ineligible_player_ids:
        mask_to_drop = (
            (player['temp_season_year'] == current_season)
            & (player['player_id'].isin(ineligible_player_ids))
        )
        player = player[~mask_to_drop].copy()

    player = player.drop(columns=['temp_season_year'], errors='ignore')
    return player, ineligible_player_ids


def add_player_names(df):
    # Standalone adaptation: the original integration queried sync.models.Player;
    # player_name is now display-only metadata supplied by the local player CSV.
    if 'player_name' not in df.columns:
        raise ValueError(
            "Falta player_name: el CSV local de jugadores debe incluir este metadato."
        )
    out = df.copy()
    return out

def calculate_expected_votes(scores, temperature=0.5):
    scores_array = np.asarray(scores, dtype=float)
    
    # Escalar aplicando la temperatura
    scaled_scores = scores_array / temperature
    
    # Estabilidad numérica real (restar el máximo)
    exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
    
    return exp_scores / exp_scores.sum()
