# Module Loading Policy and Expected States

This document complements the runtime module monitor by documenting expected module states and how alerts are generated.

Overview
- The app logs loaded modules to logs/system_log.jsonl at startup and on every request boundary.
- Any newly loaded modules are classified as allowed, disallowed, or unknown using utils/module_guard policy.
- If disallowed modules are loaded during prediction operations, a CRITICAL alert is emitted.

Expected Module States
- Startup (server boot):
  • Allowed core frameworks: flask, flask_cors, flask_compress, werkzeug, pandas, numpy, sklearn, dotenv, yaml
  • Project-local: features, utils, config, prediction_pipeline_v4, ml_system_v4, model_registry, endpoint_cache (if available)
  • Disallowed: Any scrapers (selenium, playwright, pyppeteer), openai, watchdog, src.collectors, comprehensive_form_data_collector
- Manual prediction (/predict_page):
  • Allowed: prediction_pipeline_v4, ml_system_v4 and their dependencies
  • Unknown: transient stdlib submodules are acceptable; alerts are only raised if a disallowed prefix appears
  • Disallowed (alert): results scrapers and headless browser tooling
- Ingestion endpoints (/api/ingest_csv):
  • Allowed: csv ingestion, pandas, utils.csv_metadata
  • Disallowed: live scraping frameworks unless specifically enabled outside prediction_only mode

Alerts
- The monitor writes events with event=unexpected_module_load and severity=CRITICAL when disallowed modules are loaded during prediction flows.
- The module guard raises ModuleGuardViolation and also logs a module_guard_violation event.

Performance Metrics
- The monitor logs perf_metric events with simple duration measurements via utils.module_monitor.time_block.
- The app also logs request latency in logs/perf_server.log and emits SLOW markers for requests >500ms.

Operational Notes
- Policy can be tuned with environment variables: ALLOWED_MODULE_PREFIXES, DISALLOWED_MODULE_PREFIXES, PREDICTION_IMPORT_MODE, MODULE_GUARD_STRICT.
- Keep scraping and heavy experimental scripts in archive folders to avoid accidental imports.

# Module Loading Policy and Separation of Concerns

This document explains how the repository separates prediction-only modules from historical data modules, and how module loading is controlled to prevent unsafe imports and temporal leakage.

Key goals
- Ensure clean separation between prediction-only code paths and historical/scraping code paths
- Prevent disallowed modules (scrapers, headless browsers, heavy frameworks) from loading during prediction
- Avoid post-race data leakage by keeping training/historical modules out of inference code paths
- Keep the main repo root clean; archive deprecated/legacy import files per project rules

Terminology
- Historical data: data derived from past races (form guides and race results). Per rules, form guides are historical data; race winners must be scraped from the race page itself, not the form guide.
- Race data: the input CSV for a single upcoming race used for prediction.

Directories and roles
- prediction-only modules: PredictionPipelineV4, MLSystemV4, src/parsers/csv_ingestion, utils/* (non-scraping), features/* used for inference.
- historical data modules: scrapers under src/collectors, expert form/download/ingestion tooling, playwright/selenium-based scripts, comprehensive collectors, and test-only scripts.
- archives: archive/, archive_old_apps/, archive_unused_scripts/, etc. Legacy v3 pipelines and deprecated predictors should live here if no longer used in production code paths.

Module guard
- utils/module_guard.py enforces an import policy based on environment variables.
  - PREDICTION_IMPORT_MODE: default prediction_only. In this mode, disallowed prefixes (e.g., selenium, playwright, src.collectors, openai) cannot be loaded.
  - ALLOWED_MODULE_PREFIXES and DISALLOWED_MODULE_PREFIXES allow fine-grained overrides.
- The PredictionPipelineV4 calls module_guard.pre_prediction_sanity_check() at the start of predict_race_file to block disallowed modules from being loaded.

Lazy and conditional imports
- app.py: The primary PredictionPipelineV4 is imported eagerly (safe for inference). All legacy systems (PredictionPipelineV3, UnifiedPredictor, ComprehensivePredictionPipeline) are NOT imported at module scope. They are imported lazily within request handlers only if a fallback is needed, preventing unnecessary loading during normal prediction-only operation.
- app.py: Comprehensive form data collector is loaded via a helper (get_comprehensive_collector_class) that checks feature flags before import, ensuring scrapers do not load in prediction_only mode.
- prediction_pipeline_v4.py: The module guard import is local within the predict method, not at module import time, to avoid false positives and ensure policy is enforced exactly at prediction time.

Why lazy imports?
- Safety: Prevents scrapers, headless browsers, or training components from being imported during inference flows, which could violate the prediction-only policy and lead to temporal leakage.
- Performance: Reduces startup overhead and memory footprint for the web app and batch predictors.
- Clarity: Makes it explicit where non-inference modules are used (fallbacks), easing auditing and testing.

Separation rules
- Prediction-only code must not import or depend on:
  - src.collectors.* (scrapers)
  - playwright, selenium, headless browsers
  - openai (LLM calls are outside the prediction path)
  - training/historical pipelines when not required for inference
- Use CsvIngestion strictly for race data ingestion. Do not pull in broader scraping frameworks.

Environment configuration
- Set PREDICTION_IMPORT_MODE=prediction_only (default) for servers handling inference.
- Optionally set MODULE_GUARD_STRICT=1 to enforce hard failures when disallowed modules are loaded.
- Customize ALLOWED_MODULE_PREFIXES and DISALLOWED_MODULE_PREFIXES via environment if needed for local workflows.

Archival policy
- Move deprecated or redundant import files and legacy pipelines to archive/ subfolders.
- Always check archive/ for existing versions before creating new files.
- Keep tests and manual scripts under tests/ or archive/migrated_test_scripts/, not the repo root.

FAQ
- Q: Why was PredictionPipelineV3 removed from top-level imports?
  A: To avoid loading legacy and possibly training-oriented code into prediction-only processes. It is still available via lazy import if fallback is needed.
- Q: Where do I add a new scraper?
  A: Under src/collectors/, and ensure it is never imported by prediction-only modules. Use feature flags and lazy imports guarded by module_guard.
- Q: How do I allow a new library during prediction?
  A: Add its prefix to ALLOWED_MODULE_PREFIXES or adjust DEFAULT_ALLOWED_PREFIXES in utils/module_guard.py after risk review.

