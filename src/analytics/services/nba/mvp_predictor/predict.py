from analytics.services.nba.shap_explainability import generate_shap_explanations

from analytics.services.nba.award_prediction_utils import (
    apply_projected_65_game_filter,
    add_player_names,
    calculate_expected_votes,
)

from .ranker import NBAMVPRanker
from .trainer import NBAMVPTrainer


def predict_award_season(season_year=2026, top_k=15, model_version=None):
    trainer = NBAMVPTrainer(current_season=season_year)
    ranker = NBAMVPRanker()
    engineer = trainer.engineer

    try:
        loaded_model_path = ranker.load(version=model_version)
    except FileNotFoundError as exc:
        raise ValueError(
            "No hay modelo MVP guardado. Entrena primero con analytics-train-awards --award mvp."
        ) from exc

    df_player_season, df_team_season, df_award_stats = engineer.get_raw_datasets()
    if df_player_season.empty or df_team_season.empty:
        raise ValueError("No hay datos de jugadores/equipos para generar predicciones MVP.")

    df_player_season, ineligible_ids = apply_projected_65_game_filter(
        df_player_season, df_team_season, season_year
    )

    df_player_cleaned = engineer.clean_and_normalize_player_stats(
        df_player_season, current_season=season_year
    )
    df_team_cleaned = engineer.clean_and_normalize_team_stats(df_team_season)
    df_awards_cleaned = engineer.clean_award_stats(df_award_stats)
    df_winners = engineer.get_mvp_winners(df_award_stats)
    df_fatigue = engineer.calculate_voting_fatigue(df_winners)

    df_final = engineer.get_final_training_df(
        df_player_cleaned,
        df_team_cleaned,
        df_fatigue,
        df_awards_cleaned,
    )

    df_current = df_final[df_final['season_year'] == season_year].copy()
    if df_current.empty:
        raise ValueError(f"No se encontraron candidatos MVP para la temporada {season_year}.")

    df_current = engineer.apply_ltr_top_k_filter(df_current, k=30)
    df_current = add_player_names(df_current)

    X_pred, _, _ = trainer._prepare_data(df_current)
    scores = ranker.predict(X_pred)
    df_current['ai_score'] = scores
    
    # Calculate expected votes using softmax over ALL candidates
    expected_votes = calculate_expected_votes(scores)
    df_current['expected_votes'] = expected_votes

    ladder = df_current.sort_values(by='ai_score', ascending=False).reset_index(drop=True)
    shap_result = generate_shap_explanations(
        award_slug='mvp',
        season_year=season_year,
        model=ranker.model,
        X_pred=X_pred,
        prediction_df=df_current,
        output_dir=trainer.reports_dir,
    )
    shap_group_scores = shap_result.get('grouped_scores_by_player_id', {}) if isinstance(shap_result, dict) else {}

    rows = []
    for idx, row in ladder.head(top_k).iterrows():
        rows.append(
            {
                'rank': int(idx + 1),
                'player_id': int(row['player_id']),
                'player_name': row.get('player_name'),
                'ai_score': float(row.get('ai_score', 0.0)),
                'expected_votes': float(row.get('expected_votes', 0.0)),
                'vorp_zscore': float(row.get('vorp_zscore', 0.0)),
                'wins_zscore': float(row.get('wins_zscore', 0.0)),
                'shap_group_scores': shap_group_scores.get(int(row['player_id']), {}),
            }
        )

    return {
        'award': 'mvp',
        'season_year': int(season_year),
        'loaded_model_path': loaded_model_path,
        'excluded_ineligible_players': int(len(ineligible_ids)),
        'total_candidates': int(len(ladder)),
        'top_k': int(top_k),
        'rows': rows,
    }
