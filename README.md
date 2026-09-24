# NBA MVP Learning-to-Rank

This repository is a standalone technical snapshot of an NBA Most Valuable Player ranking module. It focuses on the modeling work—feature engineering, Learning-to-Rank, historical validation, and SHAP explainability—without the surrounding application or data-ingestion system.

The trained model and historical evaluation artifacts are included. The source datasets are not. Training and inference require local CSVs conforming to the documented schema below; this repository does not generate, download, or update them.

## Technical approach

- `XGBRanker` with `objective="rank:pairwise"` and one query group per season.
- 64 model features covering individual production, efficiency, advanced impact, team context, eligibility, and voter fatigue.
- Leave-One-Season-Out validation across 29 seasons, from 1997 through 2025.
- SHAP explanations grouped into offensive volume, playmaking, defensive impact, advanced impact, team context, voter fatigue, and efficiency.
- Candidate selection prioritizes players with MVP votes, then All-NBA votes, then VORP, up to 30 players per season.

## Historical results

These values are preserved from the included validation artifacts; they are not recalculated by this snapshot.

| Metric | Result |
| --- | ---: |
| Training rows | 870 |
| Features | 64 |
| Seasons | 29 |
| NDCG@5 | 0.8936 |
| Top-1 accuracy | 75.86% |
| Average rank of the actual MVP | 1.59 |

## Standalone adaptations

The statistical logic, model parameters, classes, and historical artifacts are preserved. Only integration boundaries were replaced.

| Original integration | Standalone adaptation |
| --- | --- |
| Data access through Django and PostgreSQL | Three equivalent local CSV inputs loaded on demand |
| Paths derived from Django settings | Paths resolved relative to the predictor module |
| Player names resolved through an application model | `player_name` supplied by the player CSV as display-only metadata |
| Absolute imports from the host project | Relative imports within the standalone package |
| Absolute paths in newly generated manifests | Portable artifact filenames |

Each corresponding source change also has a nearby `Standalone adaptation` comment. No application framework, database connection, scraper, API, UI, or automated pipeline is included.

## Local dataset contract

Place the following files in `src/analytics/services/nba/mvp_predictor/data/`. The directory and files are intentionally absent from this repository. Importing the module does not require them; training or inference fails explicitly with the path of the first missing CSV.

All three files use one row per entity and season. `season` must use `YYYY-YY` format, such as `2024-25`. Identifiers must be stable across files.

### `player_season_stats.csv`

Identity and display metadata:

`player_id`, `player_ext_id`, `player_name`, `team_id`, `season`

Availability and box-score/advanced inputs:

`gp`, `fg_pct`, `fg3_pct`, `fg2_pct`, `efg_pct`, `ft_pct`, `trp_dbl`, `ortg`, `drtg`, `per`, `ts_pct`, `fg3_attempt_rate`, `ft_attempt_rate`, `oreb_pct`, `dreb_pct`, `reb_pct`, `ast_pct`, `stl_pct`, `blk_pct`, `tov_pct`, `usg_pct`, `ows`, `dws`, `win_shares`, `win_shares_per_48`, `obpm`, `dbpm`, `bpm`, `vorp`, `on_court_pm_per_100`, `net_pm_per_100`, `min_total_per_game`, `fgm_per_game`, `fga_per_game`, `fg3m_per_game`, `fg3a_per_game`, `fg2m_per_game`, `fg2a_per_game`, `ftm_per_game`, `fta_per_game`, `oreb_per_game`, `dreb_per_game`, `reb_per_game`, `ast_per_game`, `stl_per_game`, `blk_per_game`, `tov_per_game`, `pf_per_game`, `pts_per_game`, `pf_shooting_per_game`, `pf_offball_per_game`, `pts_gen_by_ast_per_game`, `and1_per_game`, `blocked_attempts_per_game`

`player_name` is retained only for reports and prediction display. It is explicitly excluded before the numeric feature matrix is built and does not change the 64-feature model input.

### `team_season_stats.csv`

`team_id`, `season`, `wins`, `losses`, `conference_rank`, `srs`, `margin_of_victory`, `strength_of_schedule`, `ortg`, `drtg`, `nrtg`

`losses` is used by the projected games-played eligibility check for the current season. The remaining statistics provide team context.

### `award_stats.csv`

`player_id`, `season`, `mvp_first`, `mvp_share`, `all_nba_share`

Historical award data supplies the ranking target, candidate-selection signals, and prior-winner history used for voter-fatigue features. For inference, the file still needs the historical rows required to calculate that context; the current season may have no award rows.

The loader validates column presence but deliberately does not fabricate missing files, columns, rows, or values.

## Repository layout

- `features.py`: cleaning, seasonal normalization, eligibility, joins, and voting-fatigue features.
- `ranker.py`: XGBoost ranker definition plus model persistence/loading.
- `trainer.py`: LOSO evaluation and final-model training orchestration.
- `predict.py`: current-season ranking and SHAP orchestration.
- `shap_explainability.py` / `shap_group_config.py`: atomic and grouped explanations.
- `trained_models/`: preserved trained XGBoost model and metadata.
- `reports/`: preserved validation, feature-importance, and 2026 SHAP outputs.

## Environment

Python dependencies are listed in `requirements.txt`: pandas, NumPy, XGBoost, scikit-learn, and SHAP.

There is intentionally no CLI, web interface, service layer, database adapter, or data acquisition code. Without the local CSVs, the included code and artifacts can be inspected and the saved model can be loaded, but training and end-to-end inference are unavailable.
