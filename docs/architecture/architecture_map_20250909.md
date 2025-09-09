# Architecture Map (snapshot 2025-09-09)

Overview
- Primary server: Flask app in app.py
- Companion: FastAPI in fastapi_app/main.py (secondary)
- Prediction core: MLSystemV4 (temporal leakage-safe, calibrated, EV, optimizer)
- Orchestration: EnhancedPredictionService or PredictionPipelineV4 → MLSystemV4 → ModelRegistry
- Data semantics: upcoming race CSVs (race data) for prediction; historical form guides for training only; winners scraped from race page per project rules

Key runtime flags
- TESTING, ENABLE_ENDPOINT_DROPDOWNS, DISABLE_ASSET_MINIFY, PREDICTION_IMPORT_MODE, GREYHOUND_DB_PATH
- V4_DISABLE_ACCURACY_OPTIMIZER (disable optimizer integration)
- TGR_ENABLED (feature gating for TGR enrichment)

Prediction call graph (Flask)
1) HTTP → /api/predict_file or /api/predict_single_race_enhanced
2) app.py: run_prediction_for_race_file(race_file_path, tgr_enabled?) with module_guard
3) Pipeline selection:
   - EnhancedPredictionService (if active)
   - else PredictionPipelineV4
4) PredictionPipelineV4 → MLSystemV4.predict_race:
   - Preflight tables
   - TemporalFeatureBuilder features (no leakage)
   - Optional TGR features
   - Calibrated model inference + EV
5) Result enrichment (CSV metadata, odds joins when available) → JSON response

Representative endpoints
- Health/diagnostics: /api/health, /api/model_health, /api/diagnostics/*, /api/server-port
- Prediction: /api/predict_file, /api/predict_single_race_enhanced, /api/predict_all_upcoming_races_enhanced
- Races/Upcoming: /api/upcoming_races_csv, /api/races, /api/races/paginated
- Dogs: /api/dogs/search, /api/dogs/all, /api/dogs/top_performers, /api/dogs/<dog_name>/details
- TGR: /api/tgr/feature_flag, /api/tgr/settings, /api/tgr/status, /api/tgr/jobs*
- Registry/training: /api/model_registry/*, /api/model/training/trigger
- Ingestion/upload: /api/ingest_csv, /upload

Database
- ORM models: models.py (race_metadata, dog_race_data, dogs, ml_model_registry, prediction_history, processed_race_files, db_meta)
- Alembic: baseline + indexes + weather column + merges
- App safeguards: ensure_results_indexes(), _ensure_minimal_ml_schema()

Testing status (this run)
- Schema consistency (alembic autogenerate, FK indexes, integrity): PASSED
- Backend API and prediction suites: PASSED
- V4 CSV transformation: PASSED
- Coverage: coverage.xml emitted (broad repo; many auxiliary modules not under these tests)

Notes
- Align Alembic index for dogs(dog_name) vs dogs(clean_name) via a safe migration/repair script (see migrations/add_dogs_index_safety.py).
- Enforce FK constraints in SQLite sessions; ensure parity in Postgres.
- Disable file watchers in production.

