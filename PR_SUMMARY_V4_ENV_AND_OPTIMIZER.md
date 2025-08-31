# PR Summary: V4 Environment Pinning and Optimizer Integration

This PR hardens the V4 inference pipeline and environment, and extends the ensemble with a second V4-compatible model.

## Changes

1) Environment stability
- requirements.txt now includes the unified constraints file to pin critical ML libs:
  -r requirements/constraints-unified.txt
- Added scripts/validate_env.py and a CI step to assert scikit-learn == 1.7.1 and print numpy/pandas versions.

2) V4 pipeline safety
- ml_system_v4.py: ColumnTransformer now prefers set_output('pandas') to preserve dtypes and names through transforms.
- temporal_feature_builder.py: Default historical features now include distance_adjusted_time=False and target_distance=0.0 to satisfy the V4 contract even with sparse history.

3) Enhanced Accuracy Optimizer fixes
- Optimizer now builds leakage-safe V4 features first (via MLSystemV4) before predicting, eliminating large NaN gaps and honoring temporal guards.
- Categorical defaults filled and numeric columns coerced to float64 (no pandas.NA Int64) within the optimizer path.
- Preserves dog names across contract alignment.
- Loads only active models from the registry to avoid stale artifact errors.

4) Registry cleanup and ensemble extension
- Deactivated stale registry entries that were missing artifacts or incompatible serialization.
- Added scripts/train_register_v4_gb.py to train a V4-compatible GradientBoosting model on leakage-safe features and register it as V4_STAGING.
- Ensemble now runs with 2 active models (V4_ExtraTrees + V4_GradientBoosting) by default.

## Verification
- Manual prediction through MLSystemV4.predict_race on a minimal upcoming race DataFrame returns success with preserved dog names and no dtype/NA warnings.
- tests/test_v4_transformation.py passed locally.
- Environment validation script reports:
  - scikit-learn: 1.7.1
  - numpy: 1.26.4
  - pandas: 2.3.1
- After registering the new model, the optimizer reports ensemble_models_used: 2 for predictions.

## Notes & Next Steps
- If we want a larger ensemble, prefer retraining additional models under the pinned environment (rather than re-serializing old artifacts) and registering them as V4_STAGING before promotion.
- Optional: follow-up to add feature signature validation in the inference path to reject mismatched artifacts proactively.

