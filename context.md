# Code Context

## Files Retrieved
1. `artifacts/full_evidence_orchestration_20260525/box_bias_audit/report.md` (lines 1-220) - primary evidence that current champion uses box as dominant signal and live snapshots all picked box 1.
2. `artifacts/full_evidence_orchestration_20260525/box_bias_audit/feature_dominance_report.md` (lines 1-111) - active production model feature-importance evidence: `cat__box_number_1` is top transformed feature in every fold.
3. `artifacts/full_evidence_orchestration_20260525/isolated_challenger_box_bias_study_20260602/report.md` (lines 1-137) - clean official holdout, champion baseline, no-promotion decision, and box-bias regression failure.
4. `artifacts/full_evidence_orchestration_20260525/bounded_calibrated_debiasing_study_20260603/report.md` (lines 1-127) - report-only debiasing sweep and current red box-bias gate.
5. `artifacts/full_evidence_orchestration_20260525/history_feature_failure_diagnosis_20260602/report.md` (lines 1-95) - root cause details for non-box feature weakness and train/eval mismatch.
6. `artifacts/full_evidence_orchestration_20260525/feature_repair_failure_diagnosis_20260603/report.md` (lines 1-73) - repair attempt lowered box bias but did not improve Top3/calibration.
7. `artifacts/full_evidence_orchestration_20260525/target_metadata_capture_audit/report.md` (lines 1-119) - live snapshot feature-missingness evidence for missing target distance/grade and near-duplicate vectors.
8. `artifacts/full_evidence_orchestration_20260525/live_target_metadata_batch/feature_quality_audit/report.md` (lines 1-91) - repeat live feature audit showing target metadata still absent.
9. `model_registry/best_metadata.json` (lines 1-72) - current active model ID/features include `box_number`, `target_distance`, TGR/history features; single non-ensemble active model.
10. `prediction_pipeline_v4.py` (lines 1120-1309) - production response normalization and rank sorting now probability-first.
11. `accuracy_program/bet_readiness.py` (lines 1-242) - current betting-abstain gates.
12. `accuracy_program/snapshots.py` (lines 740-1009) - snapshot readiness gates, result-field exclusion, odds provenance, and report-only calibration contract.
13. `scripts/ingest_results_for_date.py` (lines 1-90, 920-1109, 1540-1838) - official-first result label ingestion, ready-snapshot requirement, dry-run and explicit approval gates.
14. `upcoming_race_browser.py` (lines 1390-1469) - production CSV collection gates for canonical TheDogs normalization and final runner-set alignment.
15. `tests/test_box_bias_regression.py` (lines 1-64) - production regression gate requiring box-1 favorite share <=50% over prediction files.
16. `docs/diagnostics/box_bias_fix/README.md` (lines 1-42) - older UI/order-bias fix note; useful contrast, but current evidence shows a remaining model-probability issue.
17. `CALIBRATION_ROOT_CAUSE_ANALYSIS.md` (lines 1-132) - older empty-training-data diagnosis; superseded in part by current model metadata/artifacts but still relevant to data collection history.

## Key Code

### Current production model surface
`model_registry/best_metadata.json` lines 1-72:
```json
{
  "model_id": "V4_ExtraTrees_ExtraTreesClassifier_Calibrated_20260329_212033",
  "model_type": "ExtraTreesClassifier_Calibrated",
  "training_samples": 1441,
  "test_samples": 354,
  "features_count": 49,
  "feature_names": ["grade", "race_time", "venue", "distance", "weather", "field_size", "box_number", ...],
  "hyperparameters": {"calibration_method": "isotonic", "base_model": "ExtraTreesClassifier"},
  "is_ensemble": false
}
```

### Box-1 dominance evidence
`artifacts/.../box_bias_audit/feature_dominance_report.md` lines 7-32:
- `box_number` in metadata features: `True`.
- Family importance: `box_number` share `0.267393`, larger than `historical_performance` (`0.188490`) and `embedded_form_history` (`0.160341`).
- `cat__box_number_1` is rank 1 transformed feature in every inspected fold.

`artifacts/.../box_bias_audit/report.md` lines 23-30 and 105-118:
- Champion temporal test: Top1 `0.475`, but box1 top-pick rate `0.7625`.
- Live frozen snapshots: champion picked box 1 in `8/8` races; no-box and reduced-box arms picked box 1 in `3/8`.

`artifacts/.../isolated_challenger_box_bias_study_20260602/report.md` lines 14-33:
- Clean races `132`, runner rows `943`.
- Champion historical top-pick boxes `{'1': 105}`; rolling `{'1': 27, '2': 1, '7': 1}`.
- Rolling Top1 only `0.1379`, Top3 `0.4828`, box1_share `0.9310`.

### Data/feature weakness evidence
`artifacts/.../target_metadata_capture_audit/report.md` lines 19-53, 107-119:
- Safe target distance rows: `0/195`; safe target grade rows: `0/195`.
- Near-duplicate non-box vectors >=80% equal peer: `165/195`; top-pick boxes `{'1': 25, '4': 1}`.
- `tgr_all_zero` `195/195`; `target_distance_zero` `178/195`.
- Rejected metadata includes post-result fields `PLC`, `TIME`, `WIN`, `MGN`, etc., showing leakage guards intentionally refuse them.

`artifacts/.../history_feature_failure_diagnosis_20260602/report.md` lines 24-40:
- No-box challenger dropped box1 share from `0.9310` to `0.1379`, but Top3 fell from `0.4828` to `0.3103`, Brier/log loss worsened.
- Same-distance speed features had eval coverage `0.6202` but train coverage `0.0000`, so the model excluded them.
- Target-grade availability on eval was `0.0`.

### Production gates
`prediction_pipeline_v4.py` lines 1134-1284 normalizes probability aliases (`win_prob`, `win_probability`, `win_prob_norm`), computes EV only when odds exist, flags single-model use, and sorts predictions by probability before assigning `predicted_rank`.

`accuracy_program/bet_readiness.py` lines 93-242 adds abstain flags without changing probabilities/ranks. Important gates: missing/stale live odds, thin history, probabilities too uniform, single model, model-vs-market disagreement, and low calibration confidence when model version/metrics are absent or degraded.

`accuracy_program/snapshots.py` lines 770-813 requires pre-jump lifecycle, prediction/feature-freeze timestamps, runner identity/probabilities, source runner-set completeness, odds timestamps/provenance, and explicit missing-odds flags for snapshot readiness. Lines 835-853 reject result-field leakage in prediction rows. Lines 982-1009 enforce report-only calibration: canonical `win_prob_norm` and `predicted_rank` unchanged, no labels/odds at runtime, no registry mutation, no betting.

`scripts/ingest_results_for_date.py` lines 1-13 and 1540-1838: result labels are official-first; Sportsbet fallback is partial and blocked for label writes. Label writes require complete official result positions, clean dry-run report, exact scope match, and explicit `--write-labels-approved` or `APPROVE_RESULT_LABEL_WRITE`.

`tests/test_box_bias_regression.py` lines 1-64: box-bias regression fails if box 1 is top favorite in >50% of parsed prediction files. Recent reports record expected failure: `Box 1 favorites share too high: 90.00% > 50% over 190 files`.

## Architecture

1. **Collection:** `upcoming_race_browser.py` downloads/normalizes TheDogs CSVs and rejects incomplete or unaligned runner sets before accepted files are written.
2. **Prediction:** V4 production model (`model_registry/best_model.joblib` + `best_metadata.json`) uses 49 features including `box_number`, history/TGR, target distance, venue/grade. `prediction_pipeline_v4.py` standardizes probability keys, computes EV only from available odds, flags single-model limitations, and ranks by probability.
3. **Snapshot/EV readiness:** `accuracy_program/snapshots.py` freezes pre-jump predictions, rejects result leakage, records odds provenance and runner completeness, and allows only report-only calibration overlays.
4. **Bet gates:** `accuracy_program/bet_readiness.py` separates prediction availability from betting eligibility; current common blockers are missing odds, single-model/no ensemble, low calibration metrics, thin/uniform history, and market disagreement.
5. **Label collection:** `scripts/ingest_results_for_date.py` loads post-race results only into label surfaces after official-first completeness checks, ready pre-jump snapshot matching, clean dry-run validation, and explicit human approval.
6. **Evaluation gates:** box-bias regression plus artifact studies are currently red/no-promotion. Existing evidence says do not promote, retrain, bet, or infer EV edge from current champion or report-only challengers.

## Root Causes

1. **Not just UI order anymore: active model probability is box-1 dominated.** Older `docs/diagnostics/box_bias_fix/README.md` identified a UI/rank alias issue, but current feature audit proves `cat__box_number_1` is top feature in every active calibrated fold and live champion picks box 1 ~93% on clean rolling evidence.
2. **Non-box feature vectors are too weak/defaulted, making box prior dominate.** Live audits show `0/195` then `0/78` safe target distance/grade rows, `tgr_all_zero` at 100%, high near-duplicate non-box vectors, and target distance mostly zero.
3. **Train/eval feature coverage mismatch blocks useful challenger learning.** Same-distance features exist in eval but not train (`0.6202` vs `0.0000` coverage), so no-box repairs de-bias but lose ranking/calibration.
4. **Clean official labels are still narrow and gated.** Clean holdout has 132 races/943 runners, but rolling evidence only 29 races; odds-provenance-complete evidence was only 4 races in the isolated study. Current gates appropriately block label writes unless official complete result positions and ready snapshots exist.
5. **Production gates are doing their job; they are blockers, not bugs.** Box-bias regression remains red; bet readiness abstains on missing/stale odds, single model, low calibration confidence, thin/uniform data, and market disagreement; label writes require explicit approval and complete official result evidence.

## Concrete Fix Surfaces

1. **Model/training:** reduce or regularize `box_number`, especially `cat__box_number_1`; evaluate softer post-hoc debiasing (`box_temperature_calibration`, `blend_champion_no_box_history`) only after more clean labels. Do not hard-remove box as promotion path yet: prior no-box repairs reduced bias but hurt Top3/calibration.
2. **Feature collection:** repair safe target distance/grade capture in snapshot/source sidecars; improve grade vocabulary; populate same-distance/same-grade and same-venue history consistently across train and live eval.
3. **History/TGR joins:** fix `tgr_all_zero` and near-duplicate non-box vectors; ensure embedded and DB history features are populated pre-race and temporally safe.
4. **Label pipeline:** continue official-first result ingestion, but increase clean completed labels backed by ready pre-jump snapshots; keep Sportsbet partial results out of label writes.
5. **Gates/tests:** keep `tests/test_box_bias_regression.py` threshold red until fresh production prediction corpus is <50% box-1 favorites; keep bet-readiness abstain gates and snapshot readiness gates unchanged while model evidence is weak.

## Start Here

Open `artifacts/full_evidence_orchestration_20260525/box_bias_audit/feature_dominance_report.md` first. It gives the shortest decisive proof that the failure is now an active model-probability/feature-dominance problem, not only a UI sorting issue.

## Supervisor coordination

No supervisor decision requested. Scout was not blocked and made no file modifications beyond this requested report.
