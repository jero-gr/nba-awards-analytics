# NBA MVP Predictor Pipeline

## Scope and evidence

This document explains the extracted MVP pipeline from input data through training, validation, inference, and SHAP output. It distinguishes between behavior preserved in this repository and integration work that belonged to the original host application.

This repository preserves the modeling code and historical artifacts for technical review. It is not a supported production implementation, and some integration-dependent assumptions—such as current-season award context, upstream validation, artifact promotion, and persistence—remain outside the snapshot.

The snapshot does not include data acquisition, scheduled update jobs, database infrastructure, or the application wrappers that invoked every operation. Where those components are absent, this document describes the required boundary without inventing an implementation or command.

## 1. Data origin and refresh boundary

In the original integrated environment, the predictor reads application-managed relational datasets through Django. Those datasets contain season-level player statistics, season-level team statistics, historical award voting, and player/team identity metadata.

The source data is collected from Basketball Reference by the host platform's ETL during an overnight bulk synchronization. The bulk runs once per day when the NBA season is active, and inference is triggered only when it produces new relevant data. During an active season, the resulting statistics represent cumulative season-to-date values through the latest successful bulk.

The predictor itself does not fetch or update source data. It assumes that the ETL has already refreshed and consolidated the application datasets. A medium-term platform roadmap proposes replacing the scraping integration with premium-provider APIs to improve consistency and pipeline robustness. That migration is not part of this repository.

The standalone adaptation replaces that integration boundary with three local CSV files. Their complete column contract and expected location are documented in [README.md](README.md#local-dataset-contract). The repository intentionally does not generate, download, or silently substitute these inputs.

## 2. Input datasets

### Player-season statistics

One row represents a player in a season and includes:

- Stable player and team identifiers.
- `player_name` as display-only metadata.
- Availability and minutes.
- Shooting volume and efficiency.
- Rebounding, playmaking, defensive, and possession statistics.
- Advanced impact metrics such as PER, Win Shares, BPM, VORP, and plus-minus measures.

For a player traded during the season, the ETL consolidates statistics into a single player-season row.

### Team-season statistics

One row represents a team in a season and supplies wins, losses, conference rank, SRS, margin of victory, strength of schedule, and offensive, defensive, and net ratings.

### Award statistics

One row represents a player-season award record. The MVP pipeline uses first-place votes, MVP vote share, and All-NBA share to construct the historical ranking target, identify previous winners, and select historical candidate pools.

Season strings use `YYYY-YY`. the code converts them to the ending year. For example, `2024-25` becomes `season_year=2025`.

## 3. Loading: `data_loader.py`

`NBADataLoader` is the input boundary.

In the original integration it issues SQL queries through Django and joined statistical records to application-managed player and team metadata. In this standalone snapshot it loads:

- `player_season_stats.csv`.
- `team_season_stats.csv`.
- `award_stats.csv`.

Importing the module does not require those files. Accessing a dataset validates that its CSV exists and contains every required column. otherwise the loader raises an explicit error naming the missing file or columns.

## 4. Feature engineering: `features.py`

`NBAFeatureEngineer` loads and transforms the three datasets.

### Player preparation

`clean_and_normalize_player_stats()`:

1. Derives `season_year` from the season string.
2. Applies a minimum of 25 minutes per game.
3. Requires at least 70% of the maximum games played in that season.
4. For completed seasons from 2024 onward, excludes players below 65 games.
5. Computes player-statistic z-scores independently within each season.
6. Retains only identifiers, display metadata, and the configured normalized variables.

`player_name` remains available for reports and prediction display. `trainer.py` explicitly removes it before constructing the numeric model matrix, so it does not enter or reorder the 64 model features.

### Team preparation

`clean_and_normalize_team_stats()`:

1. Derives `season_year`.
2. Converts conference rank to `1 / conference_rank`.
3. Computes within-season z-scores for team context variables.
4. Retains the team identifier, inverse conference rank, and normalized context.

### MVP target

`clean_award_stats()` derives the ranking relevance target:

```text
first_place_ratio = player first-place votes / season first-place votes

mvp_share_weighted =
    first_place_ratio * scale_factor + mvp_share ** power
```

The normal training pipeline uses `scale_factor=2` and `power=2`. This weighting was a deliberate heuristic intended to emphasize first-place support and dominant candidates. It was not learned or selected through hyperparameter optimization. `trainer.py` multiplies the result by 100 and converts it to an integer relevance label for XGBoost.

### Voter fatigue

`get_mvp_winners()` identifies the highest MVP share in each season, `calculate_voting_fatigue()` then derives:

- total MVP awards won before the row's season.
- consecutive MVP awards immediately preceding the row's season.

Only strictly earlier seasons contribute to these features.

### Final merge and candidate pool

The prepared datasets are left-joined by player/team and season. Missing award shares and voter-fatigue counts are filled with zero.

`apply_ltr_top_k_filter(k=30)` builds a candidate pool for each season in this order:

1. Players with MVP votes.
2. Players with All-NBA votes but no MVP votes.
3. Remaining players ordered by `vorp_zscore` until the pool reaches 30.

Using historical voting outcomes to define this pool was a conscious tradeoff: it stabilized Learning-to-Rank on a small dataset, while making the evaluation cohort partially dependent on ex-post information.

For training, `prepare_training_dataset()` excludes `season_year >= current_season` before returning the historical rows.

## 5. Model input and training: `trainer.py`

`NBAMVPTrainer._prepare_data()` produces:

- `X`: the 64 numeric model features.
- `y`: integer ranking relevance derived from `mvp_share_weighted`.
- `qid`: `season_year`, identifying each season as an independent ranking query.

Player IDs, external IDs, names, team IDs, season, award outcomes, and target columns are excluded from `X`.

An optional `season_start`/`season_end` pair limits the historical training window. Supplying only one boundary is rejected.

## 6. Ranker: `ranker.py`

`NBAMVPRanker` wraps `xgboost.XGBRanker` with the preserved configuration:

```text
objective          rank:pairwise
ndcg_exp_gain      false
learning_rate      0.02
max_depth          4
n_estimators       300
subsample          0.8
colsample_bytree   0.8
min_child_weight   4
reg_lambda         1.0
tree_method        hist
random_state       42
```

Training passes the season identifiers as `qid`. Prediction returns uncalibrated ranking scores: a larger score means a higher position in the MVP ladder.

## 7. Leave-One-Season-Out validation

`NBAMVPTrainer.cross_validate()` performs one fold per historical season:

```text
test  = one complete season
train = every other selected season
```

Each fold creates and trains a fresh ranker, predicts the held-out season, and records:

- NDCG@5.
- Whether the actual MVP ranked first.
- The predicted rank of the actual MVP.
- Actual and predicted top tens.
- The fold's leading features by gain.

The aggregate report contains average NDCG@5, top-1 accuracy, and average actual-MVP rank. The historical results included in this repository were produced over 29 seasons from 1997 through 2025.

This LOSO design intentionally measures generalization across seasons. It was not intended to reproduce a chronological, point-in-time backtest.

## 8. Final model training

`NBAMVPTrainer.train_final_model()` trains one ranker on every historical row in the selected window. It can then save:

- A timestamped model artifact.
- The stable `mvp_lambda_mart.json` alias.
- Model metadata, including parameters, training window, dataset dimensions, and supplied cross-validation metrics.
- A latest-model manifest.
- Global feature importance by gain.

Running final training is separate from refreshing an inference dataset. A current prediction can use the existing model without retraining.

Operationally, training was manual and on demand. The preserved model was refined through repeated training runs before the initial deployment. it was not retrained after that season ended. A post-season retraining cycle is a future consideration, not an automated policy currently implemented.

## 9. Inference: `predict.py`

`predict_award_season()` performs the following steps:

1. Resolves and loads a saved MVP model.
2. Loads player, team, and award datasets.
3. Applies the projected games-played eligibility filter.
4. Repeats the player, team, award, and voter-fatigue transformations.
5. Selects the requested `season_year`.
6. Creates a 30-player candidate pool.
7. Builds the 64-feature matrix in training order.
8. Obtains XGBoost ranking scores.
9. Converts scores into `expected_votes` with a temperature-0.5 softmax.
10. Orders the ladder and calculates SHAP explanations.
11. Returns the requested top `k` rows as a Python dictionary.

For the current season, the eligibility check estimates whether a player can still reach the implemented games threshold:

```text
remaining_games = 82 - (team wins + team losses)
max_possible_gp = current gp + remaining_games
```

The function returns the ladder in memory. It does not persist the main prediction table itself.

Despite its name, `expected_votes` is not a calibrated prediction of literal ballot totals. It is a relative share derived from model scores to make differences among candidates easier to interpret.

In the original integrated environment, successful inference persists the award ranking in the application database. A Go API, independent from Django, serves the stored predictions to downstream consumers. The same pattern is used for the other deployed award predictors.

Production inference is automated after an overnight bulk only when that bulk generates new relevant data. The pipeline has not yet been operated from the opening days of a season. The intended future policy is to wait approximately one month before publishing rankings. That waiting period is an operational plan, not an automatic guard implemented in this snapshot.

### Eligibility is a conservative candidate filter

The projected filter keeps a player when 64 games remain mathematically reachable. This was deliberate: borderline candidates remain in the model rather than having the predictor anticipate a later league eligibility decision.

It is not a complete implementation of the NBA's eligibility rules. In 2025-26, Luka Dončić was declared eligible with 64 qualified games and Cade Cunningham with 63 under the CBA's extraordinary-circumstances provision, while Anthony Edwards' challenge with 60 games was denied ([NBA Communications](https://pr.nba.com/cade-cunningham-luka-doncic-2025-26-nba-awards/)). Other qualification details, such as limited low-minute appearances, are also outside this heuristic ([NBA.com example](https://www.nba.com/news/spurs-victor-wembanyama-stephon-castle-out-trail-blazers)).

## 10. SHAP explainability

`shap_explainability.py` uses `shap.TreeExplainer` to calculate atomic feature contributions for every candidate. It validates that the SHAP matrix matches the prediction rows and feature columns.

For MVP, `shap_group_config.py` aggregates features into:

- Offensive volume.
- Playmaking.
- Defensive impact.
- Advanced overall impact.
- Team context.
- Voter fatigue.
- Offensive efficiency.

The output contains the strongest positive and negative atomic contributions, grouped contributions, and any ungrouped features. SHAP explanations were generated as JSON reports for analysis. they were not stored with the production prediction or served by the API.

## 11. Artifacts

The standalone paths are relative to `src/analytics/services/nba/mvp_predictor/`.

### `trained_models/`

- Timestamped XGBoost models.
- `mvp_lambda_mart.json` stable alias.
- Timestamped metadata.
- Latest-model manifest when generated.

### `reports/`

- `loso_results_<timestamp>.csv`.
- `summary_<timestamp>.json`.
- `final_model_features_<timestamp>.json`.
- `shap_mvp_predictions_<season_year>.json`.

The main prediction response is returned to its caller. Any additional persistence performed by the original host application is outside this snapshot.

The dated artifacts included here were produced by a local run shortly before deployment. They should be read as a preserved technical snapshot, not labeled as production-generated artifacts.

## 12. Generic award components and MVP-specific components

Generic or reusable award infrastructure includes:

- Season parsing.
- Expected-vote softmax.
- Projected games-played eligibility.
- The parameterized SHAP engine.
- SHAP group lookup by award slug.

The SHAP configuration retains group definitions for multiple NBA awards. Those predictors were deployed in the original award system and followed the same operational pattern—manual training, data-triggered inference, database persistence, and Go API serving—but this repository intentionally includes only MVP.

MVP-specific components include:

- The full `mvp_predictor` package.
- Weighted MVP relevance target.
- MVP and All-NBA candidate selection.
- MVP voter-fatigue features.
- The XGBoost ranker configuration and saved model.
- MVP-focused validation metrics and SHAP grouping.

Despite its generic function name, `predict_award_season()` is hard-coded to return `award="mvp"`.

## 13. Temporal safeguards and evaluation limitations

The pipeline contains several safeguards against cross-season leakage:

- Training excludes the configured current season.
- LOSO removes the complete evaluated season from that fold's training rows.
- Voter fatigue only counts winners from strictly earlier seasons.
- Consecutive awards are counted backward from the immediately preceding season.
- All z-scores are calculated within a season rather than across eras.

Two limitations should be kept in mind when interpreting historical validation:

1. LOSO is intentionally not a chronological expanding-window backtest. When an older season is held out, its training fold may contain seasons that occurred later in time. the objective is cross-season generalization.
2. Historical top-30 candidate selection intentionally uses that season's MVP and All-NBA voting outcomes to stabilize ranking on a small dataset. The held-out outcomes do not train the fold, but they partly define its evaluation cohort. A live season without those outcomes relies primarily on the VORP fallback.

LOSO therefore measures separation and generalization by season, but it does not reproduce a strict point-in-time production backtest.

## 14. Execution boundaries

The original extraction preserved one command-shaped reference:

```text
analytics-train-awards --award mvp
```

It appears as guidance when inference cannot find a saved model. Its runner is not included and the exact production syntax is no longer verified, so the string should be treated as historical evidence rather than executable documentation. The data-refresh command and production prediction wrapper are also absent. No other shell command should be inferred from the extracted code.

The confirmed Python entry points are:

```python
trainer = NBAMVPTrainer(
    season_start=1997,
    season_end=2025,
    current_season=2026,
)

trainer.cross_validate()
trainer.train_final_model(cv_summary=trainer.last_cv_summary)
```

and:

```python
predict_award_season(
    season_year=2026,
    top_k=15,
    model_version=None,
)
```

These are library calls, not replacement shell interfaces.

The original operating model is:

- Training: manual and on demand, primarily during pre-deployment model refinement.
- Inference: automatic after an overnight bulk produced new relevant data.
- Prediction persistence: application database.
- Serving: a Go API independent from Django.
- Explainability persistence: JSON reports only.

## Further reading

[Modelos predictivos de los premios individuales de la NBA](https://copero.com.ar/blog/prediccion-premios-nba-2026), by Jerónimo Grinovero (April 2, 2026), presents the public-facing interpretation of the projections and explains the target and relative-score heuristics.

## Flow diagram

```text
External data refresh
        |
        v
player + team + award datasets
        |
        v
NBADataLoader
        |
        v
eligibility + cleaning + within-season z-scores
        |
        +---- award history ----> voter-fatigue features
        |
        +---- team statistics --> team-context features
        v
merged rows + 30-player season candidate pools
        |
        +---- Trainer --> LOSO reports --> final model + metadata
        |
        `---- Predict --> ranking scores --> softmax --> SHAP --> ladder
```

## Recommended reading order

1. `mvp_predictor/data_loader.py`
2. `mvp_predictor/features.py`
3. `mvp_predictor/trainer.py`
4. `mvp_predictor/ranker.py`
5. `mvp_predictor/predict.py`
6. `award_prediction_utils.py`
7. `shap_explainability.py`
8. `shap_group_config.py`
9. `mvp_predictor/reports/summary_*.json`
10. `mvp_predictor/trained_models/*.metadata.json`
