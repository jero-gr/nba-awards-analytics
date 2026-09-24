from __future__ import annotations

SHAP_GROUPS_BY_AWARD = {
    'mvp': {
        'volumen_ofensivo': [
            'trp_dbl_zscore', 'fg3_attempt_rate_zscore', 'ft_attempt_rate_zscore', 'oreb_pct_zscore', 'usg_pct_zscore',
            'fgm_per_game_zscore', 'fga_per_game_zscore', 'fg3m_per_game_zscore', 'fg3a_per_game_zscore', 'fg2m_per_game_zscore',
            'fg2a_per_game_zscore', 'ftm_per_game_zscore', 'fta_per_game_zscore', 'oreb_per_game_zscore', 'and1_per_game_zscore',
            'blocked_attempts_per_game', 'pts_per_game_zscore',

        ],
        'creacion_juego': [
            'ast_pct_zscore', 'ast_per_game_zscore', 'pts_gen_by_ast_per_game_zscore', 
        ],
        'impacto_defensivo': [
           'drtg_x_zscore', 'dreb_pct_zscore', 'reb_pct_zscore', 'stl_pct_zscore', 'blk_pct_zscore', 'dws_zscore', 'dbpm_zscore',
           'dreb_per_game_zscore', 'reb_per_game_zscore', 'pf_per_game_zscore', 'pf_shooting_per_game_zscore', 'pf_offball_per_game_zscore',
        ],
        'impacto_global_avanzado': [
            'ortg_x_zscore', 'ows_zscore', 'win_shares_zscore', 'win_shares_per_48_zscore', 'obpm_zscore', 'bpm_zscore',
            'vorp_zscore', 'on_court_pm_per_100_zscore', 'net_pm_per_100_zscore', 'per_zscore',

        ],
        'contexto_equipo': [
            'inv_conference_rank', 'wins_zscore', 'srs_zscore', 'margin_of_victory_zscore', 'strength_of_schedule_zscore',
            'ortg_y_zscore', 'drtg_y_zscore', 'nrtg_zscore',
        ],
        'voter_fatigue': [
            'consecutive_mvps', 'total_previous_mvps',
        ],
        'eficiencia_ofensiva': [
            'fg_pct_zscore', 'fg3_pct_zscore', 'fg2_pct_zscore', 'efg_pct_zscore', 'ft_pct_zscore', 'ts_pct_zscore',  'tov_pct_zscore',  'tov_per_game_zscore',
            'min_total_per_game_zscore', 'gp_zscore', 
        ]
    },

    'dpoy': {
        'ancla_defensiva': [
            'blk_per_poss_zscore', 'blk_pct_zscore', 'blk_per_game_zscore', 'pct_blk_zscore', 
        ],
        'playmaking_defensivo': [
            'stl_per_poss_zscore', 'stl_pct_zscore', 'stl_per_game_zscore', 'pct_stl_zscore', 
        ],
        'impacto_avanzado': [
            'pf_per_poss_zscore', 'd_lebron_zscore', 'drtg_rel_zscore', 'dws_per_1000_zscore', 'pf_per_game_zscore',
            'd_fg_pct_zscore', 'pct_plusminus_zscore', 
        ],
        'contexto_equipo': [
            'inv_conference_rank', 'wins_zscore', 'drtg_zscore',
        ],
        'historial_premios': [
            'total_previous_dpoys', 'consecutive_dpoys'
        ],
        'availability': [
            'gp_zscore', 'min_total_per_game_zscore'
        ]
    },

    'mip': {
        'produccion_actual': [
            'CURRENT_pts_per_game_zscore', 'CURRENT_ast_per_game_zscore', 
            'CURRENT_usg_pct_zscore', 'CURRENT_gp_zscore', 'CURRENT_min_total_per_game_zscore', 'CURRENT_reb_per_game_zscore',
            'CURRENT_ts_pct_zscore','CURRENT_win_shares_zscore', 'CURRENT_vorp_zscore', 'CURRENT_bpm_zscore', 'CURRENT_per_zscore'
        ],
        'salto_interanual': [
            'DELTA_pts_per_game_zscore', 'DELTA_ast_per_game_zscore', 
            'DELTA_usg_pct_zscore', 'DELTA_gp_zscore', 'DELTA_min_total_per_game_zscore', 'DELTA_reb_per_game_zscore',
            'DELTA_ts_pct_zscore',  'DELTA_win_shares_zscore', 'DELTA_vorp_zscore', 'DELTA_bpm_zscore', 'DELTA_per_zscore',
        ],
        'contexto_equipo': [
            'inv_conference_rank', 'wins_zscore', 'srs_zscore','ortg_zscore', 'drtg_zscore', 'nrtg_zscore', 
        ],
        'historial_premios': [
            #'PREV_won_mvps', 'PREV_won_dpoys', 'PREV_won_smoys', 'PREV_won_roys', 'PREV_won_mips',
            'PREV_mip_share', #'PREV_mvp_share', 'PREV_all_nba_share'
        ]
    },

    'roy': {
        'produccion_ofensiva': [
            'trp_dbl_zscore', 'usg_pct_zscore', 'min_total_per_game_zscore', 'fgm_per_game_zscore', 'fga_per_game_zscore', 'pts_per_game_zscore',
        ],
        'eficiencia_ofensiva': [
            'fg_pct_zscore', 'fg3_pct_zscore', 'fg2_pct_zscore', 'efg_pct_zscore', 'ft_pct_zscore', 'ts_pct_zscore',
            
        ],
        'impacto_global': [
            'gp_zscore', 'gs_zscore',  'ortg_x_zscore', 'drtg_x_zscore', 'win_shares_zscore', 'bpm_zscore', 'vorp_zscore', 
             'on_court_pm_per_100_zscore', 'net_pm_per_100_zscore', 
             'oreb_per_game_zscore', 'dreb_per_game_zscore', 'reb_per_game_zscore', 'ast_per_game_zscore', 'stl_per_game_zscore', 'blk_per_game_zscore', 'tov_per_game_zscore',
             'pf_per_game_zscore',  'per_zscore',
        ],
        'contexto_equipo': [
            'inv_conference_rank', 'wins_zscore', 'srs_zscore', 'margin_of_victory_zscore', 'strength_of_schedule_zscore',
            'ortg_y_zscore', 'drtg_y_zscore', 'nrtg_zscore',
        ],
    },
 
    'smoy': {
        'anotacion_banca': [
            'usg_pct_zscore', 'fgm_per_game_zscore', 'fga_per_game_zscore', 'pts_per_game_zscore',
        ],
        'eficiencia_ofensiva': [
            'efg_pct_zscore', 'ft_pct_zscore',
        ],
        'impacto_global': [
            'gp_zscore', 'trp_dbl_zscore', 'win_shares_zscore', 'bpm_zscore', 'vorp_zscore', 
            'on_court_pm_per_100_zscore', 'net_pm_per_100_zscore', 'reb_per_game_zscore', 'ast_per_game_zscore', 'stl_per_game_zscore', 'blk_per_game_zscore', 'tov_per_game_zscore',
        ],
        'contexto_equipo': [
            'inv_conference_rank',  'wins_zscore', 'srs_zscore', 'nrtg_zscore',

        ],
        'historial_premios': [
            'total_previous_smoys',
            'consecutive_smoys',
        ],
    },
}


def get_shap_feature_groups(award_slug: str) -> dict[str, list[str]]:
    award_key = str(award_slug or '').strip().lower()
    groups = SHAP_GROUPS_BY_AWARD.get(award_key, {})
    return {group_name: list(features) for group_name, features in groups.items()}


def get_shap_group_names(award_slug: str) -> list[str]:
    return list(get_shap_feature_groups(award_slug).keys())
