#!/usr/bin/env bash
set -euo pipefail

# Reset and rebuild the SQLite database for this project.
# Usage:
#   scripts/db_reset.sh [--collect] [--no-analyze]
#
# Defaults:
#   - analyze runs by default to recreate schema and ingest data from available CSVs
#   - collect is optional; use if you want to fetch CSVs prior to analyze
#
# Notes:
#   - Stop the Flask app or any workers before running to avoid DB locks
#   - A timestamped backup is created if the DB exists

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DB_FILE="greyhound_racing_data.db"
BACKUP_FILE="greyhound_racing_data.backup.$(date +%Y%m%d%H%M%S).db"
DO_COLLECT=0
DO_ANALYZE=1

for arg in "$@"; do
  case "$arg" in
    --collect)
      DO_COLLECT=1
      shift
      ;;
    --no-analyze)
      DO_ANALYZE=0
      shift
      ;;
    *)
      ;;
  esac
done

echo "[db-reset] Project root: $PROJECT_ROOT"

# 1) Backup if present
if [[ -f "$DB_FILE" ]]; then
  echo "[db-reset] Backing up $DB_FILE -> $BACKUP_FILE"
  cp "$DB_FILE" "$BACKUP_FILE"
else
  echo "[db-reset] No existing DB found; skipping backup"
fi

# 2) Remove DB
if [[ -f "$DB_FILE" ]]; then
  echo "[db-reset] Removing $DB_FILE"
  rm -f "$DB_FILE"
fi

# 3) Optional: collect CSVs
if [[ "$DO_COLLECT" -eq 1 ]]; then
  echo "[db-reset] Collecting CSVs (run.py collect)"
  python3 run.py collect || {
    echo "[db-reset] WARNING: collect step failed, continuing..." >&2
  }
fi

# 4) Analyze to recreate schema/data
if [[ "$DO_ANALYZE" -eq 1 ]]; then
  echo "[db-reset] Rebuilding schema/data (run.py analyze)"
  python3 run.py analyze
else
  echo "[db-reset] Skipping analyze per flag"
fi

echo "[db-reset] Done"

