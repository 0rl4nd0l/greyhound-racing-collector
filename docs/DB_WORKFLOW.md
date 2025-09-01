# Database Workflow and Publishing Guide

Goal: prevent accidental writes to the analytics DB used by training/analysis, while enabling ingestion to write freely to a staging DB.

Key files
- Analytics (read-only, immutable): greyhound_racing_data.db (or $ANALYTICS_DB_PATH)
- Staging (writeable): greyhound_racing_data_stage.db (or $STAGING_DB_PATH)
- Publish script: scripts/publish_analytics_db.sh

Environment variables
- ANALYTICS_DB_PATH: absolute or relative path to the analytics DB (read-only consumers).
- STAGING_DB_PATH: path to the staging DB (ingestion writers).
- GREYHOUND_DB_PATH: legacy fallback for analytics path (still honored by some scripts).

Rules of engagement
1) Never write directly to the analytics DB. It is protected by:
   - chmod 444 (read-only permissions)
   - chflags uchg (macOS immutable file flag)
2) All ingestion/processing that mutates data must use the staging DB (STAGING_DB_PATH).
3) To publish updates from staging to analytics, use the provided publish script; do NOT manually copy the file.

Publishing flow
1) Prepare staging DB: run all ingestion, transformations, and validations against $STAGING_DB_PATH.
2) Publish to analytics:
   scripts/publish_analytics_db.sh
   - Optionally specify paths:
     STAGING_DB_PATH=/abs/path/stage.db ANALYTICS_DB_PATH=/abs/path/analytics.db scripts/publish_analytics_db.sh
   What it does:
   - Validates staging DB is openable; shows row counts for key tables
   - PRAGMA wal_checkpoint(TRUNCATE); PRAGMA journal_mode=DELETE; VACUUM
   - Unlocks analytics (clears uchg, chmod 644) if present
   - Atomically replaces analytics DB (copy to temp, mv over)
   - Re-applies protections (chmod 444, chflags uchg)
   - Removes any -wal/-shm sidecars and verifies row counts

Read-only consumers (analysis, training)
- All consumer scripts must open the analytics DB with read-only URI mode and PRAGMA query_only=ON.
- Example (Python):
  import os, sqlite3
  db_path = os.getenv("ANALYTICS_DB_PATH", "greyhound_racing_data.db")
  conn = sqlite3.connect(f"file:{os.path.abspath(db_path)}?mode=ro", uri=True)
  conn.execute("PRAGMA query_only=ON")

Writers (ingestion)
- Open the STAGING_DB_PATH normally (writeable). WAL mode is fine during ingestion. The publish script will normalize the journal before promotion.

What if someone needs to hotfix the analytics DB?
- Do not edit in place. Instead, edit the staging DB, then publish via the script. If an emergency requires a one-off unlock:
  chflags nouchg "$ANALYTICS_DB_PATH" && chmod 644 "$ANALYTICS_DB_PATH"
  # Apply fix safely (ideally by promoting from staging)
  chmod 444 "$ANALYTICS_DB_PATH" && chflags uchg "$ANALYTICS_DB_PATH"

Common pitfalls avoided by this workflow
- “attempt to write a readonly database”: occurs if a writer targets the analytics DB while immutable/read-only.
- Stale WAL/SHM files leading to inconsistent views: the publish script clears WAL/SHM and sets journal_mode=DELETE.
- Data races between trainers and ingestion: trainers always read from immutable analytics; ingestion writes only to staging.

Team guidance
- Ingestion jobs must export STAGING_DB_PATH and only write to that DB.
- Analytics/training jobs must export ANALYTICS_DB_PATH (or rely on default) and never attempt writes.
- If you see -wal/-shm files next to analytics DB, remove them; they may reappear if a writer incorrectly connects. Fix the writer to target staging and re-publish.

