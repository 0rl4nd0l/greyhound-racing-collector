# Greyhound Prediction System Resources

## Knowledge

- [Live shadow model metadata](../greyhound-high-accuracy-refinement-v1-20260610/artifacts/full_evidence_orchestration_20260525/shadow_evaluation_high_accuracy_stage2_20260610T_live_evidence/shadow_model_metadata.json)
  Primary artifact for the scheduled Random Forest model, its 78-column schema, 73 active inputs, training population, calibration, and shadow-only status.
- [On-demand predictor](scripts/predict_race_now.py)
  Primary implementation for exact race selection, pre-jump capture validation, history sealing, feature generation, scoring, and fail-closed output.
- [Frozen market-plus-form model](artifacts/frozen_models/market_form_residual_v1/model.json)
  Primary artifact for the 16 history variables and the mathematical rule that adjusts normalized Sportsbet probabilities.
- [Sportsbet blend implementation](scripts/collect_shadow_odds_snapshots.py)
  Primary implementation of the report-only 70% market / 30% shadow-model probability blend.
- [Feature reconstruction implementation](scripts/run_feature_recovery_execution_v1.py)
  Primary implementation for selecting only prior database history and deriving runner form, time, venue, distance, grade, weight, and sectional features.
- [Race evidence inventory guide](docs/race_evidence_inventory.md)
  Canonical guide for determining actual prediction, odds, and result coverage; use it before making coverage claims.
- [Form/speed residual result](artifacts/form_speed_market_residual_20260818_report_only/report.json)
  Sealed later test whose predeclared conclusion was `NO_INCREMENTAL_FORM_SPEED_SIGNAL`.
- [Betfair teacher result](artifacts/betfair_teacher_distillation_20260818_report_only/report.json)
  Report-only evidence that Betfair-derived teacher signal was promising, without making it a live input.

## Wisdom (Communities)

- Project review and promotion process
  Use for adversarial review of identity, timing, leakage, and deployment boundaries before any predictor change.

## Gaps

- A fresh race-evidence inventory is required before stating how many races currently have complete predictions, odds, and official results.
