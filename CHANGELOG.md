# Greyhound Predictor Changelog

## [v3.1.0] - 2025-08-04

### Added

- **Step 4: Strength Index Generation System**: 
  - Implemented `step4_strength_index_generator.py` to generate comparative strength scores for all dogs
  - **Weighted Linear Formula**: Domain knowledge-based scoring with Ballarat-specific enhancements (1.5x multiplier)
  - **Gradient Boosting Regressor**: ML-based approach trained on synthetic performance targets
  - **Score Normalization**: Min-Max scaling to 0-100 range for cross-dog comparison
  - **Model Persistence**: Save/load capability for trained models
  - Generated strength scores for 5 dogs with comprehensive ranking system
  - Enhanced weighting for Ballarat track performance features
  - Feature importance analysis and cross-validation metrics

### Results
- **Linear Weighted Method**: Successfully generated differentiated scores (0-100 range)
- **Top Performer**: HANDOVER (100.00 score) - excellent consistency and time reliability
- **Model Output**: Saved trained gradient boosting model for future predictions
- **Documentation**: Created comprehensive implementation summary with usage examples

## [v3.0.1] - 2025-07-31

### Fixed
-   **Form Guide CSV Scraper**: Fixed regex patterns to correctly recognize race dates and filenames, resolving "Unknown" entries in data processing.

## [v3.0.0] - 2025-07-26

This major update focuses on a comprehensive refactoring of the entire system, from the database to the prediction pipeline and the Flask API. The primary goals were to unify scattered data sources, improve prediction accuracy, enhance system stability, and provide a more robust and developer-friendly platform.

### Added

-   **Unified Database Schema**:
    -   Introduced a single SQLite database (`greyhound_racing_data.db`) to consolidate all historical, race, and dog data.
    -   Created `create_unified_database.py` to build and populate the new schema from legacy data sources.

-   **Unified Prediction System**:
    -   Developed `unified_predictor.py`, a new core prediction engine that intelligently selects the best available prediction method based on available system components.
    -   Implemented `prediction_pipeline_v3.py`, a state-of-the-art machine learning pipeline with advanced feature engineering, data validation, and model management.

-   **Enhanced Flask API (`app.py`)**:
    -   **New Prediction Endpoint**: Added `/api/predict_single_race_enhanced`, which automatically enriches input data and runs the most advanced prediction pipeline available.
    -   **Detailed Data Endpoints**:
        -   `/api/dogs/search`: Search for greyhounds.
        -   `/api/dogs/<dog_name>/details`: Get comprehensive statistics and performance history for a specific dog.
        -   `/api/races/paginated`: A powerful endpoint for browsing historical races with search, sorting, and pagination.
    -   **System Management**: Added endpoints for monitoring logs, managing data processing workflows, and viewing model performance.

-   **Configuration & Stability**:
    -   Introduced `UnifiedPredictorConfig` to centralize all paths, feature names, and system settings.
    -   Implemented caching for prediction results to improve performance.
    -   Added robust error handling and fallback mechanisms across the entire stack.

### Changed

-   **Project Structure**:
    -   Reorganized the project by moving dozens of outdated, redundant, and test-specific scripts into the `archive/` directory to clean up the root folder.
    -   Standardized file naming and module structures for better clarity.

-   **Data Processing**:
    -   The `run.py` script and background processing tasks in `app.py` were updated to work with the new unified database and prediction system.

### Future Improvements

-   **Database Migrations**:
    -   It is highly recommended to integrate **Alembic** to manage future database schema changes. This provides a version-controlled, repeatable, and safe way to evolve the database without manual SQL scripts.

-   **Automated Testing & CI/CD**:
    -   To ensure long-term stability and code quality, setting up **GitHub Actions** for continuous integration is recommended. An automated workflow should be configured to:
        1.  Install Python dependencies.
        2.  Run the `pytest` test suite on every push and pull request.
        3.  (Optional) Deploy the application to a staging environment.

-   **Frontend Enhancements**:
    -   The frontend application can be significantly enhanced by integrating the new detailed API endpoints to provide richer visualizations and deeper insights into dog and race data.


## [audit] - 2026-06-03

- Audited greyhound prediction accuracy blockers and wrote `outputs/greyhound_prediction_system_audit_20260603.md`.
- Recomputed current persisted snapshot box-favorite distribution: box 1 selected in 123/135 snapshots (`0.9111`).
- Re-ran the dedicated box-bias regression; it failed as expected with box 1 favorites at 90.00% over 190 parsed files.
- Verified `sqlite3 greyhound_racing_data_writable.db 'PRAGMA quick_check;'` returned `ok` and `git diff --check` passed.
- Diagnosis: active model probability bias is real, non-box runner signal remains weak/sparse, and dry-run-only pre-jump captures cannot later produce strict ready-snapshot result candidates.

## [audit-tooling] - 2026-06-03

- Added `scripts/audit_prediction_snapshot_readiness.py`, a manifest-backed box-bias and label-readiness audit for result-free pre-jump `prediction_snapshot_v1` artifacts.
- Added focused tests in `tests/test_audit_prediction_snapshot_readiness.py`; `3 passed` locally.
- Ran the new audit on current manifest: gate `FAIL`, latest ready races `119`, box-1 favourite share `0.907563025210084` (`108/119`).
- Ran June 1-2 slice: gate `FAIL`, latest ready races `28`, box-1 favourite share `0.9642857142857143`.
- Preserved legacy box-bias regression failure as expected: `90.00% > 50% over 190 files`.

## [feature-audit] - 2026-06-03

- Ran `scripts/audit_non_box_feature_quality.py` on the 119 latest ready manifest-backed snapshots identified by the new readiness audit.
- Wrote feature-priority note: `outputs/greyhound_feature_repair_priority_20260603.md`.
- Key result: `tgr_all_zero` on 884/884 runner rows, `target_distance_zero` on 843/884 rows, default target distance/grade sources on 687/884 rows, and near-duplicate non-box vectors on 813/884 rows at >=80% equal peer.
- Conclusion: next repair priority is target distance/grade propagation, The Greyhound Record (TGR) source/join repair or quarantine, and richer same-distance/DB-backed runner signal before any challenger promotion.

## [target-distance-feature-propagation] - 2026-06-03

- Patched `temporal_feature_builder.py` and `temporal_feature_builder_optimized.py` so safe, provenance-backed pre-race distance survives into model-facing `target_distance` even when DB timed history is empty.
- Added `tests/test_target_distance_feature_propagation.py` covering safe distance preservation and default-distance refusal.
- Reran manifest-ready feature audit under `artifacts/full_evidence_orchestration_20260525/target_distance_feature_propagation_patch_20260603/`; `target_distance_zero` improved from 843/884 rows to 687/884 rows while default-distance rows stayed blocked.
- Validation: py_compile passed, focused/relevant pytest selector passed (`27 passed`), SQLite quick check returned `ok`, and `git diff --check` passed.
- Remaining blockers: 687/884 rows still have default/missing distance and grade sources, TGR remains all-zero, and box-bias gate remains red.

## [aligned-prediction-sidecar-propagation] - 2026-06-03

- Patched `scripts/capture_prediction_snapshot.py` so canonical-aligned temporary prediction CSVs receive a copy of the original verified `.csv.metadata.json` sidecar before prediction.
- Added `tests/test_capture_prediction_snapshot_sidecar_copy.py` to verify safe sidecar copy and missing-source fail-closed behavior.
- Wrote `outputs/aligned_prediction_sidecar_propagation_patch_20260603.md`.
- Validation: focused selector passed (`26 passed`), broader relevant selector passed (`50 passed`), SQLite quick check returned `ok`, and `git diff --check` passed.
- Expected impact: future aligned captures should preserve safe target distance/grade in the actual prediction input instead of losing provenance before model feature construction.

## [tgr-quarantine] - 2026-06-03

- Freed disk space by removing stale restore/model cache surfaces; preserved current `model_registry/best_model.joblib` target and writable DB.
- Quarantined The Greyhound Record (TGR) source-derived features from default training/prediction feature paths. `TGR_ENABLED=1` alone no longer activates TGR; explicit research override `GREYHOUND_ALLOW_TGR=1` is now required.
- Updated `MLSystemV4.train_model()` to keep TGR disabled by default rather than proactively enabling it.
- Added `tests/test_tgr_quarantine.py` and updated the test stub in `tests/conftest.py`.
- Validation after cleanup: relevant selector `55 passed`, `git diff --check` passed, and SQLite quick check returned `ok`.

## [remediation-delimiter-audit-fix] - 2026-06-03

- Moved `artifacts/` off the nearly full root filesystem to `/mnt/tenn-nvme2/tenn/greyhound_racing_collector_storage/artifacts` and symlinked it back; verified `artifacts/prediction_snapshots/manifest.jsonl` remains readable.
- Verified `sqlite3 greyhound_racing_data_writable.db 'PRAGMA quick_check;'` returned `ok` after the move.
- Diagnosed the fresh `target_distance_zero = 31/31` audit result as an audit reconstruction delimiter bug: `pd.read_csv(..., sep=None)` mis-detected pipe-normalized TheDogs form-guide CSVs, causing parser failures or empty mapped races.
- Patched `scripts/audit_non_box_feature_quality.py` to prefer verified sidecar `normalized_delimiter` and fall back to first-line delimiter counts before pandas sniffing.
- Added `tests/test_audit_non_box_feature_quality_delimiter.py`; focused selector passed (`7 passed`).
- Reran the fresh 4-race feature audit under `artifacts/full_evidence_orchestration_20260525/post_tgr_drop_fresh_capture_20260603_round2/feature_audit_after_delimiter_fix/`; source errors are now empty and `target_distance_feature` is nonzero for `31/31` rows.
- Remaining blockers: active manifest-backed box-1 share still fails at `0.9105691056910569` over `123` latest ready races; TGR remains all-zero/quarantined; fresh four-race top-pick distribution is still box 1 for all races.
- Reran result ingest dry-run with ready snapshots; it found `2` candidates and wrote no labels, so label path is no longer completely starved for the fresh batch, but several older same-day snapshots remain `NOT_READY` because runner probabilities are missing.
- Reran manifest-ready non-box audit after delimiter fix under `artifacts/full_evidence_orchestration_20260525/manifest_ready_non_box_feature_audit_after_delimiter_fix_20260603/`: `target_distance_zero` is now `293/915`, matching default-distance-source rows; parser/source errors are limited to three missing `/tmp/greyhound_snapshot_20260525_midday/...` CSVs.
- Ran report-only box-neutral sensitivity on the four fresh snapshots and saved `artifacts/full_evidence_orchestration_20260525/post_tgr_drop_fresh_capture_20260603_round2/box_neutral_sensitivity_after_delimiter_fix.json`; neutralizing `box_number` changes the top pick away from box 1 in 2/4 races and lowers box-1 normalized probability in all 4, but 2/4 still rank box 1 top from non-box/history signal.
