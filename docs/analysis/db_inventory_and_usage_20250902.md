# DB Inventory and Usage Analysis (20250902)

- Commit: 6d0c0af2
- Generated: 2025-09-02T17:37:35

## Architecture summary
- Dual-DB routing (V4): staging is the writable workspace; analytics is read-only for services/analysis.
- Writers must target staging via routing helpers; analytics is published/snapshotted from staging.
- Env vars: ANALYTICS_DB_PATH, STAGING_DB_PATH; legacy fallback GREYHOUND_DB_PATH.

## Database inventory (on-disk)

| Role | Size | Modified | Path | Valid | Sample table |
|---|---:|---|---|:---:|---|
| other | 24.0 KB | 2025-09-02 16:04:59 | cache/cache/race_times/race_times.db | T | race_times |
| other | 24.0 KB | 2025-09-02 16:09:13 | cache/race_times/race_times.db | T | race_times |
| other | 20.0 KB | 2025-08-31 17:04:53 | database.sqlite | T | db_meta |
| other | 31.2 MB | 2025-08-31 17:17:14 | databases/canonical_greyhound_data.db | T | race_metadata |
| other | 488.0 KB | 2025-09-02 16:54:23 | databases/comprehensive_greyhound_data.db | T | race_metadata |
| other | 0 B | 2025-08-31 17:24:43 | databases/greyhound_racing_data.db | T | None |
| other | 28.0 KB | 2025-08-04 11:44:56 | databases/greyhound_racing.db | T | processed_race_files |
| other | 2.7 MB | 2025-08-03 21:34:12 | databases/race_data.db | T | races |
| other | 0 B | 2025-08-03 21:30:56 | databases/schema_update.db | T | None |
| other | 28.0 KB | 2025-08-03 21:34:12 | databases/unified_data.db | T | processed_race_files |
| other | 28.0 KB | 2025-08-03 21:34:12 | databases/unified_racing.db | T | processed_race_files |
| backup/snapshot | 64.6 MB | 2025-09-01 22:05:48 | docs/analysis/backup_20250901.sqlite | T | sqlite_sequence |
| backup/snapshot | 64.6 MB | 2025-09-01 19:07:28 | docs/analysis/db_snapshot_20250901.sqlite | T | sqlite_sequence |
| backup/snapshot | 64.6 MB | 2025-09-01 22:11:25 | docs/analysis/pre_cleanup_backup_20250901_221125.sqlite | T | sqlite_sequence |
| other | 212.0 KB | 2025-09-01 18:00:13 | greyhound_data.db | T | race_metadata |
| backup/snapshot | 64.6 MB | 2025-09-01 19:12:02 | greyhound_racing_data_backup_20250901_191428.db | T | sqlite_sequence |
| other | 64.6 MB | 2025-09-02 15:47:16 | greyhound_racing_data_staging.db | T | sqlite_sequence |
| analytics | 64.6 MB | 2025-09-02 17:20:48 | greyhound_racing_data.db | T | sqlite_sequence |
| other | 32.0 KB | 2025-08-30 23:38:06 | staging_restore.db | T | tgr_feature_cache |
| other | 152.0 KB | 2025-07-26 11:28:45 | system_backup_20250727_194319/databases/comprehensive_greyhound_data.db | T | race_metadata |
| other | 0 B | 2025-07-25 17:42:03 | system_backup_20250727_194319/databases/greyhound_racing.db | T | None |
| other | 29.8 MB | 2025-07-30 17:40:15 | tests/greyhound_racing_data_test.db | T | race_metadata |
| other | 32.0 KB | 2025-08-03 02:07:06 | tests/greyhound_racing_data.db | T | live_odds |
| other | 0 B | 2025-07-30 14:22:16 | tests/test.db | T | None |

## Configured DBs (effective paths)
- ANALYTICS_DB: /Users/test/Desktop/greyhound_racing_collector/greyhound_racing_data.db (exists=yes)
- STAGING_DB: /Users/test/Desktop/greyhound_racing_collector/greyhound_racing_data_stage.db (exists=no)
- LEGACY_DB: /Users/test/Desktop/greyhound_racing_collector/greyhound_racing_data.db (exists=yes)

## Migration state (Alembic)
### current
```
INFO  [alembic.runtime.migration] Context impl SQLiteImpl.
INFO  [alembic.runtime.migration] Will assume non-transactional DDL.
1419b2b82095 (head) (mergepoint)
```
### heads
```
1419b2b82095 (head)
```
### history (truncated)
```
Rev: 1419b2b82095 (head) (mergepoint)
Merges: 81268533d929, 9f1a2b3c4d5e
Path: /Users/test/Desktop/greyhound_racing_collector/alembic/versions/1419b2b82095_merge_conflicting_migration_heads.py

    merge conflicting migration heads
    
    Revision ID: 1419b2b82095
    Revises: 81268533d929, 9f1a2b3c4d5e
    Create Date: 2025-09-01 19:51:19.136781

Rev: 81268533d929
Parent: 8d202048814f
Path: /Users/test/Desktop/greyhound_racing_collector/alembic/versions/81268533d929_add_weather_column_to_race_metadata.py

    add weather column to race_metadata
    
    Revision ID: 81268533d929
    Revises: 8d202048814f
    Create Date: 2025-08-02 14:39:24.369021

Rev: 8d202048814f (mergepoint)
Merges: 9860d6e5a183, 000000000003
Path: /Users/test/Desktop/greyhound_racing_collector/alembic/versions/8d202048814f_merge_heads.py

    merge heads
    
    Revision ID: 8d202048814f
    Revises: 9860d6e5a183, 000000000003
    Create Date: 2025-08-02 14:39:08.903177

Rev: 9860d6e5a183
Parent: bdd69f3b1271
Path: /Users/test/Desktop/greyhound_racing_collector/alembic/versions/9860d6e5a183_add_indexes_for_better_performance.py

    Add indexes for better performance
    
    Revision ID: 9860d6e5a183
    Revises: bdd69f3b1271
    Create Date: 2025-07-30 20:37:58.150572

Rev: 9f1a2b3c4d5e
Parent: bdd69f3b1271
Path: /Users/test/Desktop/greyhound_racing_collector/alembic/versions/9f1a2b3c4d5e_add_url_columns_to_race_metadata.py

    Ensure url and sportsbet_url columns exist on race_metadata
    
    Revision ID: 9f1a2b3c4d5e
    Revises: bdd69f3b1271
    Create Date: 2025-08-30 08:50:00.000000
    
    This migration is idempotent: it only adds missing columns if they don't exist.

Rev: 000000000003
Parent: 000000000002
Path: /Users/test/Desktop/greyhound_racing_collector/alembic/versions/add_foreign_keys_to_race_id.py

    add_foreign_keys_to_race_id
    
    Revision ID: 000000000003
    Revises: 000000000002
    Create Date: 2025-08-01 00:30:00.000000

Rev: 000000000002
Parent: add_enhancer_modifications
Path: /Users/test/Desktop/greyhound_racing_collector/alembic/versions/add_missing_columns_and_foreign_keys.py

    add_missing_columns_and_foreign_keys.py
    
    Revision ID: 000000000002
    Revises: add_enhancer_modifications
    Create Date: 2025-08-01 00:00:00.000000

Rev: add_enhancer_modifications
Parent: bdd69f3b1271
Path: /Users/test/Desktop/greyhound_racing_collector/alembic/versions/add_enhancer_modifications.py

    Add tables and modify columns for the Greyhound Racing Database

Rev: bdd69f3b1271 (branchpoint)
Parent: <base>
Branches into: 9860d6e5a183, add_enhancer_modifications, 9f1a2b3c4d5e
Path: /Users/test/Desktop/greyhound_racing_collector/alembic/versions/bdd69f3b1271_initial_database_schema.py

    Initial database schema
    
    Revision ID: bdd69f3b1271
    Revises:
    Create Date: 2025-07-30 20:36:29.011084
```

## Health and table inventory
### Analytics DB: /Users/test/Desktop/greyhound_racing_collector/greyhound_racing_data.db
- PRAGMA integrity_check: ok
- PRAGMA quick_check: ok
#### Key table row counts
- race_metadata: 3583
- dog_race_data: 19354
- enhanced_expert_data: 29751
- dogs: 11920
- dog_performances: 8225
- live_odds: 307
- weather_data_v2: 311
#### Freshness signals
- race_metadata.max_race_date: 2025-08-31
- dog_race_data.max_extraction_timestamp: None

### Staging DB: not found
#### Required tables (V4) presence
- ANALYTICS_DB: missing=none
- STAGING_DB: n/a

## Index and FK coverage (analytics)
```sql
idx_db_meta_key|CREATE INDEX idx_db_meta_key ON db_meta (meta_key)
idx_dog_performance_ft_extra_performance_id|CREATE INDEX idx_dog_performance_ft_extra_performance_id ON dog_performance_ft_extra(performance_id)
idx_dog_performances_dog_id|CREATE INDEX idx_dog_performances_dog_id ON dog_performances(dog_id)
idx_dog_race_data_race|CREATE INDEX idx_dog_race_data_race ON dog_race_data (race_id)
idx_dog_race_data_race_id|CREATE INDEX idx_dog_race_data_race_id ON dog_race_data (race_id)
idx_dogs_clean_name|CREATE INDEX idx_dogs_clean_name ON dogs (dog_name)
idx_dogs_ft_extra_dog_id|CREATE INDEX idx_dogs_ft_extra_dog_id ON dogs_ft_extra(dog_id)
idx_dogs_trainer|CREATE INDEX idx_dogs_trainer ON dogs (trainer)
idx_enhanced_expert_data_race_dog|CREATE INDEX idx_enhanced_expert_data_race_dog ON enhanced_expert_data(race_id, dog_clean_name)
idx_enhanced_expert_data_race_id|CREATE INDEX idx_enhanced_expert_data_race_id ON enhanced_expert_data(race_id)
idx_expert_form_analysis_race_id|CREATE INDEX idx_expert_form_analysis_race_id ON expert_form_analysis(race_id)
idx_gr_dog_entries_dog_name|CREATE INDEX idx_gr_dog_entries_dog_name ON gr_dog_entries (dog_name)
idx_gr_dog_entries_race_id|CREATE INDEX idx_gr_dog_entries_race_id ON gr_dog_entries (race_id)
idx_gr_dog_form_dog_entry_id|CREATE INDEX idx_gr_dog_form_dog_entry_id ON gr_dog_form (dog_entry_id)
idx_gr_dog_form_race_date|CREATE INDEX idx_gr_dog_form_race_date ON gr_dog_form (race_date)
idx_gr_race_details_race_id|CREATE INDEX idx_gr_race_details_race_id ON gr_race_details (race_id)
idx_prediction_history_model|CREATE INDEX idx_prediction_history_model ON prediction_history (model_name, model_version)
idx_prediction_history_race_id|CREATE INDEX idx_prediction_history_race_id ON prediction_history (race_id)
idx_processed_files_hash|CREATE INDEX idx_processed_files_hash ON processed_race_files (file_hash)
idx_processed_files_processed_at|CREATE INDEX idx_processed_files_processed_at ON processed_race_files (processed_at)
idx_processed_files_race_key|CREATE INDEX idx_processed_files_race_key ON processed_race_files (race_date, venue, race_no)
idx_race_analytics_race_id|CREATE INDEX idx_race_analytics_race_id ON race_analytics(race_id)
idx_race_metadata_race_id|CREATE INDEX idx_race_metadata_race_id ON race_metadata(race_id)
idx_races_gr_extra_race_id|CREATE INDEX idx_races_gr_extra_race_id ON races_gr_extra (race_id)
idx_unique_box_per_race|CREATE UNIQUE INDEX idx_unique_box_per_race ON dog_race_data(race_id, box_number)
idx_unique_dog_per_race|CREATE UNIQUE INDEX idx_unique_dog_per_race ON dog_race_data(race_id, dog_clean_name)
sqlite_autoindex_alembic_version_1|
sqlite_autoindex_csv_dog_history_staging_1|
sqlite_autoindex_csv_race_metadata_staging_1|
sqlite_autoindex_db_meta_1|
sqlite_autoindex_dog_performance_ft_extra_1|
sqlite_autoindex_dog_race_data_backup_1|
sqlite_autoindex_dogs_1|
sqlite_autoindex_dogs_ft_extra_1|
sqlite_autoindex_race_metadata_1|
sqlite_autoindex_race_notes_1|
sqlite_autoindex_races_ft_extra_1|
sqlite_autoindex_trainers_1|
sqlite_autoindex_trainers_2|
sqlite_autoindex_weather_data_1|
sqlite_autoindex_weather_data_v2_1|
sqlite_autoindex_weather_forecast_cache_1|
sqlite_autoindex_weather_impact_analysis_1|
```

## Code usage map and routing adoption
- Approx. routing adoption: 13.7%
### Analyzer excerpt (first ~80 lines)
```
🔍 Analyzing Database Usage Patterns...
============================================================
📊 DATABASE CONNECTION ANALYSIS REPORT
============================================================
\n🔍 DIRECT SQLITE3.CONNECT
--------------------------------------------------
\n  Pattern: sqlite3\.connect\s*\(\s*["\']([^"\']+)["\']
  Matches: 52
    1. investigate_single_dog_races.py:6 - sqlite3.connect("greyhound_racing_data.db"
    2. demo_tgr_enrichment_system.py:8 - sqlite3.connect("greyhound_racing_data.db"
    3. fix_data_quality.py:1 - sqlite3.connect("greyhound_racing_data.db"
    ... and 49 more matches
\n  Pattern: sqlite3\.connect\s*\(\s*([A-Z_]+)\s*\)
  Matches: 149
    1. temporal_anomaly_investigation.py:1 - sqlite3.connect(db_path)
    2. generate_er_diagram.py:1 - sqlite3.connect(db_path)
    3. tgr_dashboard_server.py:1 - sqlite3.connect(DB_PATH)
    ... and 146 more matches
\n  📈 Category total: 201 matches
\n🔍 DATABASE ROUTING FUNCTIONS
--------------------------------------------------
\n  Pattern: open_sqlite_readonly\s*\(
  Matches: 10
    1. app.py:96 - open_sqlite_readonly(
    2. scripts/monitor_system_health.py:1 - open_sqlite_readonly(
    3. scripts/evaluate_race_level_v4.py:1 - open_sqlite_readonly(
    ... and 7 more matches
\n  Pattern: open_sqlite_writable\s*\(
  Matches: 11
    1. app.py:96 - open_sqlite_writable(
    2. scripts/tgr_backfill_from_longform_by_date.py:1 - open_sqlite_writable(
    3. scripts/verify_and_patch_schema.py:1 - open_sqlite_writable(
    ... and 8 more matches
\n  Pattern: get_analytics_db_path\s*\(
  Matches: 6
    1. app.py:96 - get_analytics_db_path(
    2. app.py:96 - get_analytics_db_path(
    3. app.py:96 - get_analytics_db_path(
    ... and 3 more matches
\n  Pattern: get_staging_db_path\s*\(
  Matches: 5
    1. app.py:96 - get_staging_db_path(
    2. app.py:96 - get_staging_db_path(
    3. app.py:96 - get_staging_db_path(
    ... and 2 more matches
\n  📈 Category total: 32 matches
\n🔍 CONFIG-BASED CONNECTIONS
--------------------------------------------------
\n  Pattern: app\.config\.get\s*\(\s*["\']DATABASE_PATH["\']
  Matches: 4
    1. app.py:96 - app.config.get("DATABASE_PATH"
    2. app.py:96 - app.config.get("DATABASE_PATH"
    3. app.py:96 - app.config.get(
            "DATABASE_PATH"
    ... and 1 more matches
\n  Pattern: DATABASE_PATH
  Matches: 326
    1. check_temporal_integrity.py:1 - database_path
    2. temporal_anomaly_investigation.py:22 - database_path
    3. temporal_anomaly_investigation.py:22 - database_path
    ... and 323 more matches
\n  Pattern: GREYHOUND_DB_PATH
  Matches: 72
    1. config.py:1 - GREYHOUND_DB_PATH
    2. config.py:1 - GREYHOUND_DB_PATH
    3. upcoming_race_predictor.py:1 - GREYHOUND_DB_PATH
    ... and 69 more matches
\n  Pattern: ANALYTICS_DB_PATH
  Matches: 35
    1. check_db_usage.py:1 - analytics_db_path
    2. check_db_usage.py:1 - ANALYTICS_DB_PATH
    3. check_db_usage.py:8 - analytics_db_path
    ... and 32 more matches
\n  Pattern: STAGING_DB_PATH
  Matches: 30
    1. check_db_usage.py:1 - staging_db_path
    2. check_db_usage.py:1 - STAGING_DB_PATH
    3. check_db_usage.py:9 - STAGING_DB_PATH
    ... and 27 more matches
\n  📈 Category total: 467 matches
```

## Violations and risks (highlights)
- Direct sqlite3.connect occurrences may bypass routing and write protections.
- Missing required V4 tables or zero-row tables reduce feature coverage.
- Multiple Alembic heads (if present) can block safe migration advances.
- Missing indexes or FKs can hurt performance and integrity.

## Recommendations (prioritized, no writes)
1) Enforce DB routing across the codebase; ensure writes go to staging only.
2) Resolve migration divergence if present (merge heads); keep models.py aligned with Alembic head.
3) Verify and add critical indexes where missing; plan non-destructive migrations.
4) Ensure analytics DB freshness; publish from staging per docs/DB_WORKFLOW.md.
5) Add CI checks to fail on direct sqlite3.connect or writes against analytics.