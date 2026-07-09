# Next Training Prompt - Milestone 6B

Run the next milestone as a training run only after re-checking the 6A design package.

Objective: train and evaluate report-only challenger candidates on the repaired non-TGR feature schema. Do not promote a model, mutate the model registry, rewrite prediction snapshots, write labels, place bets, or enable TGR.

Use these 6A files as hard inputs:

- `outputs/milestone_6a_non_tgr_challenger_training_design/repaired_non_tgr_schema.json`
- `outputs/milestone_6a_non_tgr_challenger_training_design/training_dataset_spec.json`
- `outputs/milestone_6a_non_tgr_challenger_training_design/temporal_validation_plan.json`
- `outputs/milestone_6a_non_tgr_challenger_training_design/leakage_gate_plan.json`
- `outputs/milestone_6a_non_tgr_challenger_training_design/box_bias_gate_plan.json`
- `outputs/milestone_6a_non_tgr_challenger_training_design/calibration_gate_plan.json`

Preflight:

1. Locate or reproduce the 5F evidence checks for `compatible_artifact_scan.json`, `active_model_schema_probe.json`, `acceptance_criteria_audit.json`, and `SUMMARY.md`.
2. Assert the active model is still `V4_ExtraTrees_ExtraTreesClassifier_Calibrated_20260329_212033`.
3. Assert the active fitted artifact still has `n_features_in = 49`.
4. Assert the active schema still includes 18 `tgr_*` columns.
5. Assert the repaired non-TGR candidate schema has exactly 103 columns and zero `tgr_*` columns.
6. Assert TGR remains disabled: `TGR_ENABLED=0`, `TGR_FEATURES_ENABLED=0`, and no `GREYHOUND_ALLOW_TGR=1`.
7. Rebuild or reload the clean official dataset and reproduce at least 132 clean official races and 928 dog rows, or stop with `NOT_READY_LABELS`.

Training design to execute:

1. Build a feature matrix using only `repaired_non_tgr_schema.json`.
2. Use only canonical clean official labels with full finish-position detail.
3. Split strictly by race date, with all dogs in a race in one split and no race_id overlap.
4. Fit these candidate families in memory or in a clearly isolated report-only output directory:
   - ExtraTrees on the repaired non-TGR schema.
   - HistGradientBoosting on the repaired non-TGR schema.
   - Calibration variants as evaluation artifacts only, fitted on train folds only.
5. Score champion and challengers on the exact same temporal holdout.
6. Report Top1, Top3, winner rank, mean winner rank, log loss, Brier, calibration slope/intercept, reliability bins, probability sum error, and box-1 top-pick share.
7. Run the leakage gate, box-bias gate, calibration gate, feature population audit, and no-TGR assertion.

Stop conditions:

- Stop with `NOT_READY_SCHEMA_SPEC` if schema count is not 103, any `tgr_*` column is present, or feature population is insufficient.
- Stop with `NOT_READY_LABELS` if the clean official label cohort cannot be reproduced or identity/label gates fail.
- Stop with `NOT_READY_LEAKAGE_PLAN` if any leakage, future-history, post-outcome, quarantined-row, identity-drift, or TGR violation is found.
- Stop with `READY_FOR_REPORT_ONLY_CHALLENGER_REVIEW` only if all gates pass.

Promotion remains out of scope.
