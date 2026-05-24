# TGR (The Greyhound Recorder) Integration Audit

Date: 2025-08-28
Author: Agent Mode

Overview
- The repository already contains a comprehensive TGR integration stack spanning scraping, database schema, prediction-time enrichment, background enrichment services, dashboards, and tests.
- Integration with the prediction feature builders is present and gated by environment flags.

Key modules discovered
- tgr_prediction_integration.py
  - Provides TGRPredictionIntegrator with on-demand historical lookup, caching (tgr_feature_cache), temporal safety, and feature calculation (win rate, place rate, avg/best position, distance/venue stats, sentiment from comments, recency).
  - Integrates with TemporalFeatureBuilder via integrate_tgr_with_temporal_builder() and direct usage in TemporalFeatureBuilder.build_features_for_race().
- temporal_feature_builder.py and temporal_feature_builder_optimized.py
  - Build leakage-safe features and, when enabled, augment with TGR features per dog.
  - Gate TGR usage via environment: PREDICTION_IMPORT_MODE and ENABLE_RESULTS_SCRAPERS; optimized builder currently initializes the integrator unconditionally if import available.
- src/collectors/the_greyhound_recorder_scraper.py
  - Scraper implementation for TGR pages used by TGRPredictionIntegrator when enable_tgr_lookup is True.
- src/collectors/adapters/the_greyhound_recorder_adapter.py
  - DB adapter to adapt/load TGR meeting-level data into local DB extras tables.
- tgr_enrichment_service.py
  - Background job service (queue) to enrich/calculate TGR-derived data, with jobs, logs, and metrics tables (tgr_enrichment_jobs, tgr_service_log, tgr_service_metrics).
- Dashboards/Utilities
  - tgr_dashboard_server.py, tgr_monitoring_dashboard.py, check_tgr_integration_status.py, test_tgr_* modules, and various scripts for optimization, validation, live tests.

Database schema (TGR-specific)
- sql/create_greyhound_recorder_tables.sql
  - races_gr_extra: stores TGR page URLs and meeting linkage (FK race_metadata.id)
  - gr_race_details: long-form race details (distance, grade, conditions)
  - gr_dog_entries: per-race dog entries from TGR (dog_name, box_number, trainer, etc.)
  - gr_dog_form: historical form entries per dog (race_date, venue, distance, finish, times, comments)
  - Indices provided for performance; triggers maintain updated_at.
- tgr_feature_cache (created at runtime by TGRPredictionIntegrator)
  - Simple cache for precomputed features per dog with timestamps.

Current integration behavior
- At predict time, the TemporalFeatureBuilder computes leakage-safe DB features; if allowed, it augments them with TGR features retrieved by TGRPredictionIntegrator, respecting a 1-year lookback and strict pre-race cutoff.
- TGRPredictionIntegrator first queries local TGR tables (gr_dog_form via gr_dog_entries) by UPPER(dog_name) for the last 365 days; scraping is possible if enabled but integrator primarily uses DB where available.
- Many TGR support scripts/services exist for scraping, backfilling, performance and monitoring.

Environment gating (observed)
- TemporalFeatureBuilder enables TGR only if BOTH:
  - PREDICTION_IMPORT_MODE != 'prediction_only'
  - ENABLE_RESULTS_SCRAPERS not in ('0', 'false', 'False')
- Optimized builder initializes the integrator if importable (no explicit flag check).

Gaps and proposed improvements
- Identity resolution: current query uses UPPER(name) equality (gr_dog_entries.gde.dog_name). A dog_aliases mapping table is not present. Name normalization and alias resolution should be added to improve match rates.
- Unified source-aware ingestion: DB lacks a canonical historical_runs table with source attribution (fasttrack vs tgr). Current TGR data lives in gr_* tables; feature builder currently pulls base histories from dog_race_data/race_metadata; 
  - Consider adding either source columns to dog_race_data or a new unioned historical_runs with source/source_run_id/source_dog_id to combine sources cleanly.
- Feature flag: introduce TGR_ENABLED to allow TGR integration even in prediction_only mode when desired, reducing coupling to results scraper flags.
- API debug fields: consider exposing historical_sources_used/history_runs_used in debug mode to verify TGR contributions per prediction.

Recommended next steps (aligned with todo plan)
1) Add TGR_ENABLED feature flag to the TemporalFeatureBuilder (and consistent handling in the optimized builder) to enable/disable TGR independently of scraper flags.
2) Define the data contract for a unioned historical layer and identity resolution (docs/TGR_INTEGRATION.md). Include proposed dog_aliases, optional track_aliases.
3) Implement identity resolution utilities and (optionally) a minimal alias table to improve TGR-to-canonical dog matching.
4) Ensure the prediction path surfaces debug info (counts, sources used) behind a flag to verify TGR coverage in predictions.
5) Stage rollout with monitoring: coverage metrics like percent_dogs_with_n_plus_runs, missing_history_count_per_race, and source_mix ratios.

Notes
- Extensive TGR-related tests and scripts already exist (test_tgr_*.py), and dashboards/documentation files (TGR_*_README.md) suggest a mature integration footprint.
- This audit recommends consolidating flags and unifying data contracts, not replacing existing modules.

