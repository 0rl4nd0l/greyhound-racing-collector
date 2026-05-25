# Shepparton Prediction Evidence Sprint - 2026-05-25

## Scope

- No model retraining or promotion was performed.
- Race 14 was evaluated from the frozen pre-jump snapshot captured at `2026-05-25T13:51:55`.
- Expanded replay used a copied DB at `artifacts/historical_replay_20260525_shepparton_expanded/replay.db`.
- Expanded replay snapshots were generated before attaching any target-race results.
- All evaluated snapshots passed the result-free snapshot guard.

## Race 14 Live Evaluation

- Race: `Race 14 - SHEP - 2026-05-25`
- Result source: `sportsbet_results_top4`
- Label quality: `partial_sportsbet_winner_only`
- TheDogs official fetch: blocked with `thedogs_403_forbidden`
- Sportsbet top-four order: `7,3,5,6`
- Winner: `Ripper Riley` box 7
- Model winner rank: 7
- Top-1/top-2/top-3: `0.0 / 0.0 / 0.0`
- Brier: `0.10262414555555556`
- Log loss: `2.2926347621408776`
- EV/ROI: `DATA_MISSING`, reason `no_valid_pre_jump_dog_level_odds`
- Top pick: `Clause Edward`, box 1, probability `0.2221`

## Expanded Historical Replay

- Corpus: 14 Shepparton races from `2026-05-21`
- Snapshot files: 14
- Dog predictions scored: 104
- Result source: TheDogs official for all 14 races
- Label qualities: 9 `official_or_complete_result`, 5 `winner_name_only_result`
- Top-1/top-2/top-3: `0.21428571428571427 / 0.35714285714285715 / 0.5714285714285714`
- Mean winner rank: `3.357142857142857`
- Winner rank counts: `1:3, 2:2, 3:3, 4:2, 5:2, 6:1, 7:1`
- Brier: `0.11584669990384616`
- Log loss: `1.9768840261794816`
- EV/ROI: `DATA_MISSING`, reason `no_valid_pre_jump_dog_level_odds`

## Failure Modes

- Winner rank by race:
  - `Race 1`: 3
  - `Race 2`: 6
  - `Race 3`: 3
  - `Race 4`: 1
  - `Race 5`: 7
  - `Race 6`: 1
  - `Race 7`: 1
  - `Race 8`: 2
  - `Race 9`: 3
  - `Race 10`: 5
  - `Race 11`: 4
  - `Race 12`: 5
  - `Race 13`: 4
  - `Race 14`: 2
- Wrong top-pick races: 11 of 14
- Average top-pick probability when wrong: `0.25329999999999997`
- Maximum top-pick probability when wrong: `0.281`
- Probability uniformity: average normalized entropy `0.9743138901265744`, average probability spread `0.1511642857142857`
- Field-size effect:
  - 7-runner races: top-1 `0.125`, top-3 `0.5`, mean winner rank `3.625`
  - 8-runner races: top-1 `0.3333333333333333`, top-3 `0.6666666666666666`, mean winner rank `3.0`
- Missing odds/features:
  - 14 of 14 historical races had no valid pre-jump dog-level odds.
  - 104 of 104 historical rows carried `missing_live_odds`.
  - 104 of 104 historical rows carried `single_model_no_ensemble_agreement`.
- Venue breakdown: only `SHEP`, top-1 `0.21428571428571427`, top-3 `0.5714285714285714`
- Distance breakdown: `DATA_MISSING`, no target-race distance metadata in the replay labels.

## Partial vs Official Labels

- Live partial labels, races 13-14 from `2026-05-25`: top-1/top-2/top-3 all `0.0`, mean winner rank `6.5`, Brier `0.10255343166666668`, log loss `2.288689968503296`.
- Expanded official/winner-name labels, `2026-05-21`: top-1 `0.21428571428571427`, top-3 `0.5714285714285714`, mean winner rank `3.357142857142857`.
- Partial labels remain winner-only and are not mixed with official metrics in the label-quality breakdowns.

## Evidence Files

- Race 14 evaluation: `artifacts/prediction_snapshots/2026-05-25/SHEP/race-14_evaluation_after_result.json`
- Live Race 13-14 partial evaluation: `artifacts/prediction_snapshots/2026-05-25/SHEP/race-13-14_partial_evaluation_after_results.json`
- Expanded replay generation report: `artifacts/historical_replay_20260525_shepparton_expanded/generation_report.json`
- Expanded replay evaluation: `artifacts/historical_replay_20260525_shepparton_expanded/evaluation_after_results.json`

## Validation

- `py_compile` passed for changed Python files.
- Focused tests passed: `28 passed in 20.06s`.
- `git diff --check` passed.
- `PRAGMA quick_check` returned `ok` for the live writable DB and expanded replay DB.
- Snapshot leakage check passed for 16 result-free snapshots.
- Port `5002` was clear.
