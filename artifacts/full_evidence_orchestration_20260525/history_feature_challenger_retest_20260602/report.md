# History Feature Challenger Retest

## Executive Summary

Final recommendation: `HISTORY_FEATURES_DO_NOT_FIX_BOX_BIAS`.

This was a report-only retest over the reused clean official holdout and reconstructed pre-race history packet. No production model was trained, promoted, registered, or used for betting.

## Data Used

- Clean races: `132`
- Clean snapshot instances: `134`
- Clean runner rows: `943`
- Primary split: `historical_packet_train_to_rolling_packet_eval`
- Train rows/instances: `735` / `105`
- Eval rows/instances: `208` / `29`
- Exclusions: `{'row_exclusion_count': 0, 'reason': 'none; exact clean packet join used'}`

## Feature Packet Provenance

- Packet report: `artifacts/full_evidence_orchestration_20260525/clean_history_feature_packet_20260602/report.md`
- Packet CSV: `artifacts/full_evidence_orchestration_20260525/clean_history_feature_packet_20260602/pre_race_history_feature_packet.csv`
- Feature coverage summary: `{'ambiguous_dog_count': 0, 'explicit_history_feature_columns_now_present': ['prior_start_count', 'days_since_last_start', 'recent_finish_mean_3', 'recent_finish_mean_5', 'recent_finish_best_5', 'recent_win_rate_5', 'recent_place_rate_5', 'recent_avg_margin_5', 'recent_avg_time_5', 'best_time_same_distance', 'avg_time_same_distance', 'starts_same_distance', 'win_rate_same_distance', 'starts_same_venue', 'win_rate_same_venue', 'grade_change_indicator', 'last_start_distance', 'last_start_grade', 'last_start_days', 'last_start_trainer_present', 'last_start_weight', 'recent_avg_weight_5', 'recent_avg_sectional_1st_5', 'db_prior_start_count', 'csv_staging_prior_start_count', 'embedded_form_prior_start_count'], 'matched_dog_count': 632, 'matched_history_pct': 0.695652, 'matched_history_rows': 656, 'previous_required_history_feature_row_coverage': 0, 'race_coverage_1plus_prior_starts': 123, 'race_coverage_1plus_prior_starts_pct': 0.931818, 'races': 132, 'rows_with_1plus_prior_starts': 656, 'rows_with_1plus_prior_starts_pct': 0.695652, 'rows_with_3plus_prior_starts': 583, 'rows_with_3plus_prior_starts_pct': 0.61824, 'rows_with_5plus_prior_starts': 467, 'rows_with_5plus_prior_starts_pct': 0.495228, 'runner_rows': 943, 'snapshot_instances': 134, 'unmatched_dog_count': 0}`
- Join audit: `{'clean_rows': 943, 'packet_rows': 943, 'joined_rows': 943, 'missing_clean_key_count': 0, 'missing_clean_key_examples': [], 'excluded_label_rows': 0, 'join_status': 'PASS'}`
- Selected-feature policy: train features with low coverage or zero variance are excluded and recorded; missing feature values remain `NaN` for native missing-value handling, not fake defaults.

## Leakage Audit

- Status: `PASS`
- Packet leakage status: `PASS`
- Temporal holdout: `{'ok': True, 'train_max_date': '2026-05-21', 'test_min_date': '2026-05-26', 'race_id_overlap': [], 'violations': []}`
- Forbidden feature columns by variant: `{'champion_current_baseline': None, 'history_only_hgb': [], 'no_box_history_hgb': [], 'reduced_box_band_history_hgb': [], 'calibrated_champion_power': None, 'calibrated_history_only_hgb': None, 'calibrated_no_box_history_hgb': None, 'calibrated_reduced_box_band_history_hgb': None}`

## Champion Baseline

- Rolling eval top-pick boxes: `{'1': 27, '2': 1, '7': 1}`
- Rolling eval winner boxes: `{'1': 3, '2': 5, '3': 7, '4': 4, '5': 1, '6': 4, '7': 2, '8': 3}`
- Rolling eval box-1 top-pick share: `0.9310`

## Challenger Comparison Table

| variant | status | races | top1 | top2 | top3 | mean_rank | brier | log_loss | slope | box1_share | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| champion_current_baseline | RUN | 29 | 0.1379 | 0.3103 | 0.4828 | 3.9655 | 0.1221 | 1.9948 | 0.2213 | 0.9310 |  |
| history_only_hgb | RUN | 29 | 0.1379 | 0.2759 | 0.3103 | 3.9310 | 0.1244 | 2.0394 | 0.2041 | 0.1724 |  |
| no_box_history_hgb | RUN | 29 | 0.1724 | 0.3103 | 0.3103 | 3.9655 | 0.1247 | 2.0580 | 0.1236 | 0.1379 |  |
| reduced_box_band_history_hgb | RUN | 29 | 0.1034 | 0.3103 | 0.3793 | 3.8276 | 0.1243 | 2.0462 | 0.1549 | 0.1724 |  |
| calibrated_champion_power | RUN | 29 | 0.1379 | 0.3103 | 0.4828 | 3.9655 | 0.1194 | 1.9551 | 0.5099 | 0.9310 |  |
| calibrated_history_only_hgb | RUN | 29 | 0.1379 | 0.2759 | 0.3103 | 3.9310 | 0.1353 | 2.3171 | 0.0845 | 0.1724 |  |
| calibrated_no_box_history_hgb | RUN | 29 | 0.1724 | 0.3103 | 0.3103 | 3.9655 | 0.1362 | 2.3381 | 0.0350 | 0.1379 |  |
| calibrated_reduced_box_band_history_hgb | RUN | 29 | 0.1034 | 0.3103 | 0.3793 | 3.8276 | 0.1354 | 2.3138 | 0.0517 | 0.1724 |  |
| market_implied | BLOCKED_UNDERPOWERED | DATA_MISSING | DATA_MISSING | DATA_MISSING | DATA_MISSING | DATA_MISSING | DATA_MISSING | DATA_MISSING | DATA_MISSING | DATA_MISSING | complete valid-odds eval races=4; minimum=10 |
| simple_blend_50 | BLOCKED_UNDERPOWERED | DATA_MISSING | DATA_MISSING | DATA_MISSING | DATA_MISSING | DATA_MISSING | DATA_MISSING | DATA_MISSING | DATA_MISSING | DATA_MISSING | complete valid-odds eval races=4; minimum=10 |
| learned_blend | BLOCKED_UNDERPOWERED | DATA_MISSING | DATA_MISSING | DATA_MISSING | DATA_MISSING | DATA_MISSING | DATA_MISSING | DATA_MISSING | DATA_MISSING | DATA_MISSING | complete valid-odds train races=0; minimum=10 |
| calibrated_blend | BLOCKED_UNDERPOWERED | DATA_MISSING | DATA_MISSING | DATA_MISSING | DATA_MISSING | DATA_MISSING | DATA_MISSING | DATA_MISSING | DATA_MISSING | DATA_MISSING | complete valid-odds train races=0; minimum=10 |

## Box-Bias Diagnostics

- Box-bias production gate remains red and was not weakened.
- Full per-variant top-pick, winner-box, and per-box diagnostics are in `box_bias_diagnostics.json`.

## Calibration Diagnostics

- Calibration slope/intercept, Brier, log loss, and reliability bins are in `challenger_metrics.json` and `calibration_diagnostics.json`.
- Paired race-bootstrap deltas are in `statistical_significance.json`.
- Bootstrap summary: `{'champion_current_baseline': {'status': 'RUN', 'method': 'paired_race_bootstrap_500_seed_42', 'paired_snapshot_instances': 29, 'delta_direction': 'positive improves top1/top2/top3; negative improves mean_winner_rank/brier/log_loss', 'top1': {'mean_delta': 0.0, 'lower_95': 0.0, 'upper_95': 0.0}, 'top2': {'mean_delta': 0.0, 'lower_95': 0.0, 'upper_95': 0.0}, 'top3': {'mean_delta': 0.0, 'lower_95': 0.0, 'upper_95': 0.0}, 'mean_winner_rank': {'mean_delta': 0.0, 'lower_95': 0.0, 'upper_95': 0.0}, 'brier': {'mean_delta': 0.0, 'lower_95': 0.0, 'upper_95': 0.0}, 'log_loss': {'mean_delta': 0.0, 'lower_95': 0.0, 'upper_95': 0.0}}, 'history_only_hgb': {'status': 'RUN', 'method': 'paired_race_bootstrap_500_seed_42', 'paired_snapshot_instances': 29, 'delta_direction': 'positive improves top1/top2/top3; negative improves mean_winner_rank/brier/log_loss', 'top1': {'mean_delta': -0.006068965517241381, 'lower_95': -0.20689655172413793, 'upper_95': 0.20689655172413793}, 'top2': {'mean_delta': -0.04110344827586206, 'lower_95': -0.2413793103448276, 'upper_95': 0.1724137931034483}, 'top3': {'mean_delta': -0.17931034482758623, 'lower_95': -0.3793103448275862, 'upper_95': 0.03448275862068967}, 'mean_winner_rank': {'mean_delta': -0.0024137931034482695, 'lower_95': -1.0000000000000002, 'upper_95': 1.068965517241379}, 'brier': {'mean_delta': 0.0024216298782680705, 'lower_95': -0.005337107853442263, 'upper_95': 0.010191051255990518}, 'log_loss': {'mean_delta': 0.049878896431030585, 'lower_95': -0.1366540097063953, 'upper_95': 0.2530890625450925}}, 'no_box_history_hgb': {'status': 'RUN', 'method': 'paired_race_bootstrap_500_seed_42', 'paired_snapshot_instances': 29, 'delta_direction': 'positive improves top1/top2/top3; negative improves mean_winner_rank/brier/log_loss', 'top1': {'mean_delta': 0.030275862068965514, 'lower_95': -0.1724137931034483, 'upper_95': 0.24137931034482757}, 'top2': {'mean_delta': -0.004758620689655171, 'lower_95': -0.20689655172413796, 'upper_95': 0.20689655172413793}, 'top3': {'mean_delta': -0.17517241379310344, 'lower_95': -0.3793103448275862, 'upper_95': 0.018103448275861262}, 'mean_winner_rank': {'mean_delta': 0.02117241379310344, 'lower_95': -0.8620689655172411, 'upper_95': 0.9146551724137925}, 'brier': {'mean_delta': 0.002822255593165796, 'lower_95': -0.004258038282914469, 'upper_95': 0.010344066351196207}, 'log_loss': {'mean_delta': 0.06849161008320741, 'lower_95': -0.1204336853872679, 'upper_95': 0.269875797942538}}, 'reduced_box_band_history_hgb': {'status': 'RUN', 'method': 'paired_race_bootstrap_500_seed_42', 'paired_snapshot_instances': 29, 'delta_direction': 'positive improves top1/top2/top3; negative improves mean_winner_rank/brier/log_loss', 'top1': {'mean_delta': -0.035448275862068966, 'lower_95': -0.1905172413793103, 'upper_95': 0.10344827586206898}, 'top2': {'mean_delta': -0.004758620689655171, 'lower_95': -0.20689655172413796, 'upper_95': 0.20689655172413793}, 'top3': {'mean_delta': -0.10724137931034483, 'lower_95': -0.3103448275862069, 'upper_95': 0.10344827586206895}, 'mean_winner_rank': {'mean_delta': -0.12593103448275864, 'lower_95': -1.0870689655172412, 'upper_95': 0.8275862068965516}, 'brier': {'mean_delta': 0.0023423862920136183, 'lower_95': -0.005105774203194414, 'upper_95': 0.009823453452597596}, 'log_loss': {'mean_delta': 0.0563528580880414, 'lower_95': -0.12365872602715898, 'upper_95': 0.24880523208674785}}, 'calibrated_champion_power': {'status': 'RUN', 'method': 'paired_race_bootstrap_500_seed_42', 'paired_snapshot_instances': 29, 'delta_direction': 'positive improves top1/top2/top3; negative improves mean_winner_rank/brier/log_loss', 'top1': {'mean_delta': 0.0, 'lower_95': 0.0, 'upper_95': 0.0}, 'top2': {'mean_delta': 0.0, 'lower_95': 0.0, 'upper_95': 0.0}, 'top3': {'mean_delta': 0.0, 'lower_95': 0.0, 'upper_95': 0.0}, 'mean_winner_rank': {'mean_delta': 0.0, 'lower_95': 0.0, 'upper_95': 0.0}, 'brier': {'mean_delta': -0.002699134590230485, 'lower_95': -0.006382873213819376, 'upper_95': 0.0011158911845546645}, 'log_loss': {'mean_delta': -0.0378031515712901, 'lower_95': -0.10490194514986555, 'upper_95': 0.041899747517914115}}, 'calibrated_history_only_hgb': {'status': 'RUN', 'method': 'paired_race_bootstrap_500_seed_42', 'paired_snapshot_instances': 29, 'delta_direction': 'positive improves top1/top2/top3; negative improves mean_winner_rank/brier/log_loss', 'top1': {'mean_delta': -0.006068965517241381, 'lower_95': -0.20689655172413793, 'upper_95': 0.20689655172413793}, 'top2': {'mean_delta': -0.04110344827586206, 'lower_95': -0.2413793103448276, 'upper_95': 0.1724137931034483}, 'top3': {'mean_delta': -0.17931034482758623, 'lower_95': -0.3793103448275862, 'upper_95': 0.03448275862068967}, 'mean_winner_rank': {'mean_delta': -0.0024137931034482695, 'lower_95': -1.0000000000000002, 'upper_95': 1.068965517241379}, 'brier': {'mean_delta': 0.013422086532698831, 'lower_95': 0.0014835382900952867, 'upper_95': 0.025786383221873703}, 'log_loss': {'mean_delta': 0.3300392716493093, 'lower_95': -0.00559660455264039, 'upper_95': 0.6929470650079591}}, 'calibrated_no_box_history_hgb': {'status': 'RUN', 'method': 'paired_race_bootstrap_500_seed_42', 'paired_snapshot_instances': 29, 'delta_direction': 'positive improves top1/top2/top3; negative improves mean_winner_rank/brier/log_loss', 'top1': {'mean_delta': 0.030275862068965514, 'lower_95': -0.1724137931034483, 'upper_95': 0.24137931034482757}, 'top2': {'mean_delta': -0.004758620689655171, 'lower_95': -0.20689655172413796, 'upper_95': 0.20689655172413793}, 'top3': {'mean_delta': -0.17517241379310344, 'lower_95': -0.3793103448275862, 'upper_95': 0.018103448275861262}, 'mean_winner_rank': {'mean_delta': 0.02117241379310344, 'lower_95': -0.8620689655172411, 'upper_95': 0.9146551724137925}, 'brier': {'mean_delta': 0.01437641012381234, 'lower_95': 0.0033361549506976925, 'upper_95': 0.026361636246333274}, 'log_loss': {'mean_delta': 0.35173508426841293, 'lower_95': 0.02305870337537083, 'upper_95': 0.7075905810498759}}, 'calibrated_reduced_box_band_history_hgb': {'status': 'RUN', 'method': 'paired_race_bootstrap_500_seed_42', 'paired_snapshot_instances': 29, 'delta_direction': 'positive improves top1/top2/top3; negative improves mean_winner_rank/brier/log_loss', 'top1': {'mean_delta': -0.035448275862068966, 'lower_95': -0.1905172413793103, 'upper_95': 0.10344827586206898}, 'top2': {'mean_delta': -0.004758620689655171, 'lower_95': -0.20689655172413796, 'upper_95': 0.20689655172413793}, 'top3': {'mean_delta': -0.10724137931034483, 'lower_95': -0.3103448275862069, 'upper_95': 0.10344827586206895}, 'mean_winner_rank': {'mean_delta': -0.12593103448275864, 'lower_95': -1.0870689655172412, 'upper_95': 0.8275862068965516}, 'brier': {'mean_delta': 0.013470701316257456, 'lower_95': 0.001792142302258396, 'upper_95': 0.026508422863965066}, 'log_loss': {'mean_delta': 0.3262401851950171, 'lower_95': 0.0031368216056316097, 'upper_95': 0.6848806564530103}}}`

## EV Report-Only Diagnostics

- Odds-derived variants are marked `BLOCKED_UNDERPOWERED` unless at least 10 complete valid-odds races are available.
- EV diagnostics are report-only only and are not evidence of an EV edge.
- Details are in `ev_diagnostics_report_only.json`.

## Endpoint And SQLite Health

- Endpoint health: `{'api_health': 'curl exit 7 connection refused on http://127.0.0.1:5002/api/health', 'port_5002': 'not listening'}`
- SQLite quick_check: `ok`
- Active capture/ingest/promotion/model-registry processes: `none found`

## Commands Run

- `pwd`
- `git branch --show-current`
- `git log -1 --oneline`
- `git diff --cached --name-only`
- `git diff --cached --name-only -- protected paths`
- `git diff --check`
- `sqlite3 greyhound_racing_data_writable.db 'PRAGMA quick_check;'`
- `curl -fsS --max-time 2 http://127.0.0.1:5002/api/health`
- `pgrep -af capture/ingest/promotion/model-registry patterns`
- `sed -n 1,220p ACTIVE_SCRIPTS_GUIDE.md`
- `sed -n 1,260p clean_history_feature_packet/report.md`
- `python3 -m json.tool clean history packet JSONs`
- `.venv/bin/python -m py_compile scripts/run_history_feature_challenger_retest.py`
- `.venv/bin/python -m pytest -q tests/test_run_history_feature_challenger_retest.py --maxfail=1`
- `.venv/bin/python scripts/run_history_feature_challenger_retest.py ...`

## Changed Files

- `scripts/run_history_feature_challenger_retest.py`
- `tests/test_run_history_feature_challenger_retest.py`
- `artifacts/full_evidence_orchestration_20260525/history_feature_challenger_retest_20260602/`

## No-Mutation Confirmation

- No production writes, live result-ingest writes, result label writes, snapshot writes or rewrites, manifest append, model registry mutation, production retrain, production model file changes, model promotion, betting, fake odds, fake EV, mock racing data, `--persist`, `--capture-live-odds`, `--allow-unverified-runner-set`, or `APPROVE_RESULT_LABEL_WRITE` were used.
- The known box-bias regression gate remains intact and red.

## Validation Closeout

- `.venv/bin/python -m py_compile scripts/run_history_feature_challenger_retest.py`: passed (`exit:0`).
- `.venv/bin/python -m pytest -q tests/test_run_history_feature_challenger_retest.py --maxfail=1`: passed, `8 passed`.
- `.venv/bin/python -m pytest -q tests -k 'snapshot or metadata or leakage or runner_set or odds or ev or model_contract or calibration' --maxfail=1`: passed, `229 passed, 656 deselected, 7 warnings`.
- `.venv/bin/python -m pytest -q tests/test_box_bias_regression.py::test_favorite_box1_share_under_threshold --maxfail=1 -vv`: failed as expected, `Box 1 favorites share too high: 90.00% > 50% over 190 files`.
- `git diff --check`: passed.
- `sqlite3 greyhound_racing_data_writable.db 'PRAGMA quick_check;'`: `ok`.
- Final staged files for commit:
  - `artifacts/full_evidence_orchestration_20260525/history_feature_challenger_retest_20260602/report.md`
  - `scripts/run_history_feature_challenger_retest.py`
  - `tests/test_run_history_feature_challenger_retest.py`
- Final protected staged paths: none.
- Final endpoint health: `curl` exit `7`, connection refused on `127.0.0.1:5002`.
- Final matching capture/ingest/promotion/model-registry processes: none found.

## Recommendation

`HISTORY_FEATURES_DO_NOT_FIX_BOX_BIAS`.

## Closeout Addendum

- Closeout branch: `feature/dog-odds-snapshot-readiness`.
- Latest commit before closeout: `e8bea2f8 Fix EV readiness summary reporting`.
- Helper script tracked status before staging: untracked (`git add -N` intent-to-add used for full-addition review only).
- Test file tracked status before staging: untracked (`git add -N` intent-to-add used for full-addition review only).
- Full-addition diff exists and was reviewed: `closeout_validation/history_retest_helper_full_addition.diff`.
- Pre-write validation files:
  - `.venv/bin/python -m py_compile ...`: all passed (`closeout_validation/focused_py_compile.txt`).
  - `.venv/bin/python -m pytest -q tests/test_run_history_feature_challenger_retest.py --maxfail=1`: passed, `8 passed`.
  - `.venv/bin/python -m pytest -q tests -k 'snapshot or metadata or leakage or runner_set or odds or ev or model_contract or calibration' --maxfail=1`: passed, `229 passed, 656 deselected, 7 warnings`.
  - `.venv/bin/python -m pytest -q tests/test_box_bias_regression.py::test_favorite_box1_share_under_threshold --maxfail=1 -vv`: expected failure remains (`Box 1 favorites share too high: 90.00% > 50% over 190 files`), not a closeout blocker.
  - `sqlite3 greyhound_racing_data_writable.db 'PRAGMA quick_check;'`: `ok`.
  - `git diff --check`: passed.
- Endpoint process and health checks:
  - `api_health`: `curl` exit 7 (`connection refused`) on `127.0.0.1:5002/api/health`.
  - Active matching processes: none found for `prejump_prediction_loop`, `capture_prediction_snapshot`, `ingest_results_for_date`, `promote`, `model_registry`.
- Protected path checks:
  - `artifacts/prediction_snapshots/manifest.jsonl` not staged.
  - `model_registry/`, `docs/model_registry/current_production.json`, `ml_models_v4/`, `advanced_models/` not staged.
  - `artifacts/full_evidence_orchestration_20260525/history_feature_challenger_retest_20260602/report.md` staged only after explicit closeout staging step.
  - `reports/agent_jobs/extraction_contract_parity_guard_v1_20260526/diff-check.json` not staged.
- No-mutation confirmation: no production writes, live result-ingest writes, result label writes, snapshot writes/rewrites, manifest append, model registry mutation, production model file writes, promotion, betting, fake odds, fake EV, synthetic racing rows, or production training were introduced.
- Known production-readiness blocker remains:
  - `tests/test_box_bias_regression.py::test_favorite_box1_share_under_threshold` is still red with 90.00% box-1 favorites.
- Final recommendation remains `HISTORY_FEATURES_DO_NOT_FIX_BOX_BIAS`.
- No safe immediate promotion recommendation from this closeout; next safe task:
  - Option B: run a controlled feature-repair study focused on why history features reduce box bias but harm ranking.

## Closeout Commit

- Evidence consistency note: the locally preserved retest closeout is commit `8996b4a4`; the prior `d1c94867` reference was a stale duplicate-subject hash and is corrected here for local closeout consistency.
- Commit created: `8996b4a4`
- Commit subject: `report(challenger): retest reconstructed history features`
