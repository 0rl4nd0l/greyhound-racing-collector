# TGR Integration: Data Contract and Feature Flags

Date: 2025-08-28
Author: Agent Mode

Purpose
- Define a clear, source-aware data contract for integrating The Greyhound Recorder (TGR) data with existing histories.
- Document feature flags controlling TGR usage in predictions.

Current state (observed)
- Canonical history tables: dog_race_data (per-dog, per-race), race_metadata (race-level).
- TGR-specific schema lives in gr_* tables (see sql/create_greyhound_recorder_tables.sql):
  - races_gr_extra, gr_race_details, gr_dog_entries, gr_dog_form
- Prediction feature builders (TemporalFeatureBuilder, OptimizedTemporalFeatureBuilder) can augment features with TGRPredictionIntegrator-provided features.

Feature flags
- TGR_ENABLED (new):
  - Values: '1'/'true' to enable; '0'/'false' to disable (default disabled).
  - Behavior: When set, enables TGR integration in both temporal_feature_builder.py and temporal_feature_builder_optimized.py regardless of PREDICTION_IMPORT_MODE or ENABLE_RESULTS_SCRAPERS.
- PREDICTION_IMPORT_MODE (existing):
  - 'prediction_only' (default) or other modes.
- ENABLE_RESULTS_SCRAPERS (existing):
  - '1' to allow scrapers; '0' disables.

Recommended unified historical data contract (proposed)
- Goal: cleanly union histories across sources and avoid duplication while preserving provenance.

Option A: Add source columns to dog_race_data
- Columns to add:
  - source TEXT CHECK(source IN ('fasttrack','tgr')) DEFAULT 'fasttrack'
  - source_run_id TEXT NULL
  - source_dog_id TEXT NULL
  - ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
- Pros: minimal schema sprawl; reuse existing queries with small adjustments.
- Cons: blending source-specific semantics into a single table; dedup logic becomes embedded in queries.

Option B: Introduce historical_runs (recommended for clearer separation)
- historical_runs schema:
  - id INTEGER PRIMARY KEY AUTOINCREMENT
  - dog_id INTEGER (or dog_clean_name TEXT when id unavailable)
  - source TEXT NOT NULL ('fasttrack','tgr',...) 
  - source_run_id TEXT NOT NULL
  - source_dog_id TEXT NULL
  - meeting_date DATE NOT NULL
  - meeting_time TEXT NULL
  - venue TEXT
  - distance_m INTEGER
  - grade TEXT
  - box INTEGER
  - split_time TEXT
  - run_time REAL
  - margin TEXT
  - position INTEGER
  - weight REAL
  - trainer TEXT
  - prize TEXT
  - comments TEXT
  - ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  - UNIQUE(source, source_run_id)
- Indexes: (dog_id), (dog_clean_name), (meeting_date), (venue, meeting_date)
- Pros: source provenance preserved; easier dedup across sources; flexible for future sources.

Identity resolution (proposed)
- Create dog_aliases table to map canonical dog identity across sources and variations:
  - id INTEGER PRIMARY KEY AUTOINCREMENT
  - canonical_dog_name TEXT NOT NULL
  - source TEXT NOT NULL
  - source_dog_id TEXT NULL
  - name_normalized TEXT NOT NULL
  - dob DATE NULL
  - earbrand TEXT NULL
  - microchip TEXT NULL
  - UNIQUE(source, source_dog_id)
  - INDEX(name_normalized)
- Resolution approach:
  1) Exact alias match on (source, source_dog_id)
  2) Exact on (name_normalized, dob)
  3) Exact on (earbrand) or (microchip)
  4) High-confidence fuzzy on name_normalized with track/region hints
- A utilities module (utils/dog_identity.py) should expose:
  - normalize_name(name) -> normalized string
  - score_match(candidate, canonical) -> float
- On confident matches, persist alias; otherwise, queue for manual review.

Deduplication policy
- When merging fasttrack + tgr histories into features:
  - Prefer fasttrack records if near-duplicate is detected by (date±1d, venue, distance±25m, position ±0)
  - Else union both sources

Temporal safety
- Builders must only include runs with race_timestamp strictly before the target race’s timestamp.
- Never use outcomes from form guides of the target race.

APIs and debug surfaces (optional, non-breaking)
- Behind API_INCLUDE_HISTORY_DEBUG=true, include in prediction responses:
  - history_runs_used (integer)
  - historical_sources_used (array/string)
  - missing_history_flag (boolean)

Rollout notes
- Start with TGR_ENABLED=1 in dev, validate coverage improvements (dogs with ≥5 runs), then stage to prod.
- Monitor coverage/latency metrics; provide fast rollback by unsetting TGR_ENABLED.

Appendix
- Existing TGR tables already provide rich form data; this contract focuses on unifying usage and identity resolution for robust, source-aware features.

