# Milestone 6A - Non-TGR Repaired Feature Challenger Training Design

Status: **NO TRAINING / NO PROMOTION / NO REGISTRY MUTATION**

Final verdict: **READY_FOR_CHALLENGER_TRAINING_PROMPT**

## Scope

This package defines the training specification for a new challenger artifact on the repaired non-TGR feature schema. It does not train a model, save a model artifact, write labels, mutate the registry, rewrite snapshots, promote a model, enable TGR, or perform EV/betting actions.

The active predictions remain diagnostic only.

## 5F Evidence Intake

The requested exact 5F evidence files were searched for by filename under the current checkout and the offloaded home:

- `compatible_artifact_scan.json`
- `active_model_schema_probe.json`
- `acceptance_criteria_audit.json`
- `SUMMARY.md`

Those exact files were not present locally. This design therefore uses the operator-provided 5F facts as the governing evidence inputs and records a preflight requirement for the next training run to locate or reproduce those checks before fitting.

5F facts used:

- 5F verdict: `BLOCKED_BY_MODEL_ARTIFACT`.
- Active model: `V4_ExtraTrees_ExtraTreesClassifier_Calibrated_20260329_212033`.
- Active fitted artifact has `n_features_in = 49`.
- Active schema includes 18 `tgr_*` fields.
- Repaired non-TGR candidate schema has 103 columns and 0 TGR columns.
- No compatible non-TGR repaired artifact was found after scanning 51 metadata files.
- Canonical clean official baseline is 132 races / 928 dog rows.
- TGR must remain disabled.

Repo corroboration checked during this design:

- `docs/model_contracts/v4_feature_contract.json` names the active model and shows the active 49-column feature contract including the 18 TGR feature names.
- `accuracy_program.evaluation.validate_feature_columns` provides the local forbidden post-result feature check.
- Existing report-only challenger helpers use clean official labels, temporal holdout checks, protected output paths, and no registry writes.

## Why A New Artifact Is Required

The active champion expects the 49-column V4 feature surface. The repaired candidate surface is a 103-column, zero-TGR schema with new same-distance, same-grade, target metadata, reconstructed history, sectional, grade-transition, class-strength, and draw-adjusted history features. That is not shape-compatible with the active fitted artifact. Since no compatible non-TGR repaired artifact was found in 5F, the next step must be a newly trained challenger, not an adapter, shim, or registry promotion.

## Repaired Non-TGR Schema

The schema is defined in `repaired_non_tgr_schema.json`.

Required properties:

- Exactly 103 feature columns.
- Zero feature columns with `tgr_` prefix.
- Includes same-distance features.
- Includes same-grade features.
- Includes `target_distance_safe`.
- Includes `target_grade_safe`.
- Includes repaired historical reconstruction features.
- Includes sectional metrics.
- Includes grade-transition features.
- Excludes post-outcome labels and official result fields.
- Keeps identity fields such as `race_id` and `dog_clean_name` as grouping metadata only, not model features.

Box fields are retained as prediction-time context, but they are not allowed to pass unless the box-bias gate proves box 1 no longer dominates.

## Training Dataset Criteria

The next training run must use only canonical clean official labels.

Baseline expected cohort:

- Clean official races: 132.
- Dog rows: 928.

Inclusion criteria:

- Official or complete result labels only.
- Full finish-position detail for every runner in the race.
- Exactly one winner per race.
- No identity-drift labels.
- No duplicate or ambiguous runner identity.
- No quarantined rows.
- No target-race post-outcome fields in prediction-time features.
- Historical rows strictly before the target race datetime.

Exclusion criteria:

- Winner-name-only or partial Sportsbet labels.
- Any row sourced only from a quarantined or unsafe path.
- Any race with box identity drift or ambiguous runner identity.
- Any race requiring TGR to populate a feature.
- Any feature derived from target-race results, margins, times, winning time, official result text, or post-race scraped fields.

The dataset spec is in `training_dataset_spec.json`.

## Temporal Validation

The validation split is strict temporal holdout by race date.

Required checks:

- Train rows and holdout rows are split by race date, not random dog rows.
- All dogs in a race stay in one split.
- Holdout races occur after train races.
- No `race_id` overlap exists between train and holdout.
- Champion and challengers are scored on the same holdout race set.

Required metrics:

- Top1.
- Top3.
- Winner rank and mean winner rank.
- Log loss.
- Brier.
- Calibration slope/intercept.
- Reliability bins.
- Probability sum error.
- Box-1 top-pick share.
- Top-pick box distribution.

The temporal validation plan is in `temporal_validation_plan.json`.

## Model Candidates

The next training run should evaluate:

- `ExtraTreesClassifier` on the repaired non-TGR schema.
- `HistGradientBoostingClassifier` on the repaired non-TGR schema.
- Calibrated variants only as evaluation artifacts.

Calibration may use sigmoid, isotonic, or cross-fitted calibration only when fit on training folds. Holdout rows cannot be used to fit calibration or choose a production candidate. No calibrator or model artifact is saved during 6A.

## Gates

### Leakage Gate

Defined in `leakage_gate_plan.json`.

Pass requires:

- No forbidden post-result feature columns.
- No `tgr_*` features.
- No identity columns as features.
- No future-history rows.
- No unsafe target metadata promotion.
- No identity drift, quarantined rows, or ambiguous labels.
- Clean temporal split.

### Box-Bias Gate

Defined in `box_bias_gate_plan.json`.

Pass requires:

- Box-1 top-pick share materially below the current champion reference of `0.910569`.
- Primary threshold: box-1 top-pick share <= `0.50`.
- Preferred promotion-review threshold: <= `0.35`.
- Top1 and Top3 above the champion diagnostic baseline on the same holdout.
- Mean winner rank, Brier, and log loss no worse than champion on the same holdout.

### Calibration Gate

Defined in `calibration_gate_plan.json`.

Pass requires:

- Race-normalized probabilities.
- Brier <= `0.18` where sample size supports it.
- Brier and log loss no worse than champion on the same holdout.
- Calibration slope/intercept reported where sample size supports it.
- Holdout never used to fit calibration.

### Feature Population Gate

Pass requires:

- 103 schema columns present.
- 0 TGR columns.
- Train and holdout feature coverage reported separately.
- No repaired non-TGR family entirely all-null in train or holdout.
- All-zero and near-constant features reported.
- Missingness indicators preserved rather than silently fabricating target metadata.

## Rollback And No-Op Plan

Because 6A does not train or mutate production state, rollback is a no-op:

- No registry entry to delete.
- No model artifact to remove.
- No production feature contract to revert.
- No label writes to undo.
- No snapshot or manifest mutation to restore.

If the next training run fails any gate, stop with the relevant not-ready verdict and leave the active champion unchanged.

## Final Verdict

`READY_FOR_CHALLENGER_TRAINING_PROMPT`

Reason: the schema, dataset criteria, temporal validation, leakage plan, box-bias gate, calibration gate, and no-op rollback plan are specified. The next run must still locate or reproduce the exact 5F checks before fitting.
