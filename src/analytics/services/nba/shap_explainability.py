import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from analytics.services.nba.shap_group_config import get_shap_feature_groups

try:
    import shap  # type: ignore
except ImportError:  # pragma: no cover - depends on runtime env
    shap = None


def _serialize_contributions(items, limit):
    return [
        {
            'feature': feature_name,
            'value': float(feature_value),
        }
        for feature_name, feature_value in items[:limit]
    ]


def _format_console_contributions(items):
    if not items:
        return '-'

    return ', '.join(f"{item['feature']} ({item['value']:+.2f})" for item in items)


def _format_console_group_contributions(group_scores):
    if not group_scores:
        return '-'

    ordered = sorted(group_scores.items(), key=lambda item: abs(item[1]), reverse=True)
    return ', '.join(f"{group_name} ({float(score):+.2f})" for group_name, score in ordered)


def generate_shap_explanations(
    *,
    award_slug: str,
    season_year: int,
    model: Any,
    X_pred: pd.DataFrame,
    prediction_df: pd.DataFrame,
    output_dir: str,
    player_column: str = 'player_name',
    score_column: str = 'ai_score',
    top_positive_k: int = 5,
    top_negative_k: int = 3,
):
    if X_pred.empty or prediction_df.empty:
        return {
            'enabled': False,
            'reason': 'empty_input',
            'output_path': None,
            'rows_count': 0,
        }

    if shap is None:
        print(
            "[SHAP] Librería 'shap' no instalada. Omitiendo explainability.",
            file=sys.stderr,
        )
        return {
            'enabled': False,
            'reason': 'shap_not_installed',
            'output_path': None,
            'rows_count': 0,
        }

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_pred)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_matrix = np.asarray(shap_values, dtype=float)
    if shap_matrix.ndim == 1:
        shap_matrix = shap_matrix.reshape(1, -1)

    if shap_matrix.shape[0] != len(X_pred):
        raise ValueError(
            f"SHAP row mismatch: shap_rows={shap_matrix.shape[0]} vs X_rows={len(X_pred)}."
        )
    if shap_matrix.shape[1] != X_pred.shape[1]:
        raise ValueError(
            f"SHAP column mismatch: shap_cols={shap_matrix.shape[1]} vs X_cols={X_pred.shape[1]}."
        )

    working_df = prediction_df.copy().reset_index(drop=True)
    if len(working_df) != shap_matrix.shape[0]:
        raise ValueError(
            f"Prediction row mismatch: pred_rows={len(working_df)} vs shap_rows={shap_matrix.shape[0]}."
        )

    working_df['__row_pos'] = np.arange(len(working_df), dtype=int)
    ladder = working_df.sort_values(by=score_column, ascending=False).reset_index(drop=True)
    feature_names = X_pred.columns.tolist()

    feature_groups = get_shap_feature_groups(award_slug)
    feature_to_group = {}
    for group_name, group_features in feature_groups.items():
        for feature_name in group_features:
            if feature_name not in feature_to_group:
                feature_to_group[feature_name] = group_name

    explanations = []
    grouped_scores_by_player_id = {}
    for rank_position, (_, row) in enumerate(ladder.iterrows(), start=1):
        row_pos = int(row['__row_pos'])
        row_shap = shap_matrix[row_pos]
        contributions = list(zip(feature_names, row_shap.tolist()))

        positive_contributions = [item for item in contributions if item[1] > 0]
        negative_contributions = [item for item in contributions if item[1] < 0]

        positive_contributions.sort(key=lambda item: abs(item[1]), reverse=True)
        negative_contributions.sort(key=lambda item: abs(item[1]), reverse=True)

        grouped_scores = {}
        ungrouped_contributions = []
        for feature_name, shap_value in contributions:
            group_name = feature_to_group.get(feature_name)
            if group_name:
                grouped_scores[group_name] = grouped_scores.get(group_name, 0.0) + float(shap_value)
            else:
                ungrouped_contributions.append((feature_name, float(shap_value)))

        player_name = row.get(player_column)
        player_label = str(player_name).strip() if player_name is not None else ''
        if pd.isna(player_name):
            player_label = ''
        if not player_label:
            player_label = f"player_id={row.get('player_id', 'unknown')}"

        explanation_row = {
            'player': player_label,
            'score': float(row.get(score_column, 0.0)),
            'rank': int(rank_position),
            'top_positive': _serialize_contributions(positive_contributions, top_positive_k),
            'top_negative': _serialize_contributions(negative_contributions, top_negative_k),
            'grouped_contributions': [
                {
                    'group': group_name,
                    'value': float(group_value),
                }
                for group_name, group_value in sorted(
                    grouped_scores.items(), key=lambda item: abs(item[1]), reverse=True
                )
            ],
            'ungrouped_atomic': [
                {
                    'feature': feature_name,
                    'value': float(feature_value),
                }
                for feature_name, feature_value in sorted(
                    ungrouped_contributions, key=lambda item: abs(item[1]), reverse=True
                )
            ],
        }
        explanations.append(explanation_row)

        player_id_value = row.get('player_id')
        if player_id_value is not None and not pd.isna(player_id_value):
            grouped_scores_by_player_id[int(player_id_value)] = {
                group_name: float(group_value)
                for group_name, group_value in grouped_scores.items()
            }

    original_output_filename = f"{award_slug}_predictions_{int(season_year)}.json"
    shap_output_filename = f"shap_{original_output_filename}"
    os.makedirs(output_dir, exist_ok=True)
    shap_output_path = os.path.join(output_dir, shap_output_filename)

    with open(shap_output_path, 'w', encoding='utf-8') as shap_file:
        json.dump(explanations, shap_file, ensure_ascii=False, indent=2)

    console_stream = sys.stdout if sys.stdout.isatty() else sys.stderr
    for row in explanations[:3]:
        print(f"Player: {row['player']}", file=console_stream)
        print(f"Top +: {_format_console_contributions(row['top_positive'])}", file=console_stream)
        print(f"Top -: {_format_console_contributions(row['top_negative'])}", file=console_stream)
        print(
            f"Groups: {_format_console_group_contributions({item['group']: item['value'] for item in row['grouped_contributions']})}",
            file=console_stream,
        )

    return {
        'enabled': True,
        'reason': None,
        'output_path': shap_output_path,
        'rows_count': len(explanations),
        'group_names': list(feature_groups.keys()),
        'grouped_scores_by_player_id': grouped_scores_by_player_id,
    }
