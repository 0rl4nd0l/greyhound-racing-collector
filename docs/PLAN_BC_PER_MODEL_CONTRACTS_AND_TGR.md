Task: Implement Plan B (per-model contracts + TGR compatibility shim) and Plan C (full TGR feature parity) to eliminate missing-column errors for GradientBoosting V4 models and ensure feature parity across the ensemble.

Scope and goals
- Remove missing-column errors logged for V4_GradientBoosting_* models when run via the enhanced_accuracy_optimizer.
- Enforce per-model feature contracts at inference time (not just global V4 ExtraTrees contract).
- Provide a compatibility mapping to backfill tgr_* columns from existing features where feasible until full TGR features are generated.
- Enable DB-only TGR features at inference (no live scraping) and retrain models with full TGR feature sets (later phase).

Work plan (high-level)
1) Per-model contracts
   - Generalize inspect_model_features.py to accept a model path argument and emit a contract JSON (features in order) into docs/model_contracts/<model_stem>.json.
   - Generate contracts for:
     • model_registry/models/V4_GradientBoosting_CalibratedPipeline_20250828_163025_model.joblib
     • model_registry/models/V4_GradientBoosting_CalibratedPipeline_20250828_161925_model.joblib
   - Optionally reuse ModelRegistry metadata.feature_names if populated.

2) Contract enforcement API
   - Add FeatureStore.enforce_contract(features_df, expected_features, mapping=None, log_missing=True) to complement enforce_v4_contract.
   - Add loader: load_contract_by_model_id(model_id) that resolves docs/model_contracts/<model_id or model stem>.json.

3) TGR compatibility mapping (temporary shim)
   - Provide a mapping dict to populate tgr_* columns when TGR integration is disabled or data is missing. Example initial mapping:
     tgr_win_rate <- historical_win_rate
     tgr_place_rate <- historical_place_rate
     tgr_avg_finish_position <- historical_avg_position
     tgr_best_finish_position <- historical_best_position
     tgr_recent_avg_position <- historical_avg_position
     tgr_recent_best_position <- historical_best_position
     tgr_days_since_last_race <- days_since_last_race
     tgr_venues_raced <- venue_experience
     tgr_preferred_distance_avg <- best_distance_avg_position
     tgr_preferred_distance <- target_distance
     tgr_preferred_distance_races <- race_frequency
     tgr_recent_races <- race_frequency
     tgr_consistency <- historical_time_consistency (approximation)
     tgr_form_trend <- historical_form_trend
     tgr_last_race_position <- NaN (imputed downstream)
     tgr_total_races <- 0.0 (imputed downstream)
     tgr_has_comments <- 0.0
     tgr_sentiment_score <- 0.0
   - Remaining missing columns default to 0.0/NaN; rely on model imputers where present.

4) Ensemble inference updates
   - In enhanced_accuracy_optimizer.AdvancedEnsemblePredictor.predict_with_ensemble:
     • Before predicting with each model, load its contract and create a per-model feature frame via enforce_contract + TGR mapping.
     • Do not reuse the global ExtraTrees contract for GB models.
     • Scale and predict using the per-model frame.
     • Keep dog names stable for presentation.

5) Testing
   - Add tests to cover per-model contract alignment and that GradientBoosting models no longer fail with missing columns.
   - Run existing backend tests and any e2e checks to ensure frontend/backend alignment remains intact.

6) Plan C (full TGR parity)
   - Ensure TemporalFeatureBuilder enables DB-only TGR integration during predictions (set TGR_ENABLED=1 in runtime environment) without enabling live scraping.
   - Validate tgr_prediction_integration provides all 18 tgr_* features as per TGR_PREDICTION_INTEGRATION.md.
   - Retrain V4 models (ExtraTrees and GradientBoosting) with TGR features; register and save contracts at training time.
   - Add CI parity tests to assert 100% of required contract columns are produced by the feature builder.

Acceptance criteria
- No missing-column warnings for V4_GradientBoosting_* under the optimizer.
- Per-model contracts loaded correctly; features aligned per model, order preserved, and types cast.
- Predictions succeed for the same inputs that previously failed, with ensemble_models_used ≥ 2.
- After Plan C, feature parity tests pass for all active V4 models; the frontend continues to render predictions.

Key references
- features/feature_store.py: load_v4_model_contract, enforce_v4_contract (extend with generic enforcement)
- enhanced_accuracy_optimizer.py: predict_with_ensemble (apply per-model enforcement and TGR mapping)
- docs/model_contracts/V4_ExtraTrees_20250819.json (contract example)
- inspect_model_features.py (generalize for GB models)
- tgr_prediction_integration.py, temporal_feature_builder.py (TGR feature generation and gating)
- Tests: tests/test_advanced_prediction_systems.py, test_v4_contract_fix.py, test_tgr_prediction_integration.py

