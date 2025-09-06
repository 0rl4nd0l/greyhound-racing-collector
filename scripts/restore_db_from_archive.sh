#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE_DIR="$PROJECT_ROOT/archive"
RESTORE_LOG_DIR="$ARCHIVE_DIR/db_restores"
TS="${TS:-$(date -u +"%Y%m%dT%H%M%SZ")}" 
DEST_DB="${GREYHOUND_DB_PATH:-$PROJECT_ROOT/greyhound_racing_data.db}"
mkdir -p "$RESTORE_LOG_DIR/$TS"

# Locate SQL dump to restore from (prefer full backup)
SQL_FILE=""
if ls "$ARCHIVE_DIR"/database_full_backup_*.sql >/dev/null 2>&1; then
  SQL_FILE="$(ls -t "$ARCHIVE_DIR"/database_full_backup_*.sql | head -n 1)"
else
  # fallback to newest .sql in archive root
  SQL_FILE="$(find "$ARCHIVE_DIR" -maxdepth 1 -type f -name "*.sql" -print0 | xargs -0 ls -t 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "${SQL_FILE:-}" || ! -f "$SQL_FILE" ]]; then
  echo "[restore] No SQL dump found in $ARCHIVE_DIR" >&2
  exit 1
fi

echo "[restore] Using SQL dump: $SQL_FILE"
echo "[restore] Destination DB: $DEST_DB"

# Archive current DB if present
if [[ -f "$DEST_DB" ]]; then
  BACKUP_PATH="$RESTORE_LOG_DIR/$TS/current.backup.sqlite"
  echo "[restore] Backing up existing DB -> $BACKUP_PATH"
  if command -v sqlite3 >/dev/null 2>&1; then
    sqlite3 "$DEST_DB" ".backup '$BACKUP_PATH'" || cp -f "$DEST_DB" "$BACKUP_PATH"
  else
    cp -f "$DEST_DB" "$BACKUP_PATH"
  fi
  shasum -a 256 "$BACKUP_PATH" > "$BACKUP_PATH.sha256" || true
fi

# Remove destination DB to avoid conflicts
rm -f "$DEST_DB"

# Restore
if ! command -v sqlite3 >/dev/null 2>&1; then
  echo "[restore] sqlite3 not found on PATH" >&2
  exit 1
fi

echo "[restore] Restoring from SQL dump..."
FILTERED_SQL="$RESTORE_LOG_DIR/$TS/filtered.sql"
# Filter out sqlite_stat* lines which may not be present in this SQLite build
if command -v grep >/dev/null 2>&1; then
  grep -vi 'sqlite_stat' "$SQL_FILE" > "$FILTERED_SQL" || cp -f "$SQL_FILE" "$FILTERED_SQL"
else
  cp -f "$SQL_FILE" "$FILTERED_SQL"
fi
sqlite3 "$DEST_DB" ".read '$FILTERED_SQL'"

# Integrity check
echo "[restore] Running integrity check..."
sqlite3 "$DEST_DB" "PRAGMA integrity_check;" | tee "$RESTORE_LOG_DIR/$TS/integrity_check.txt"

# Basic counts (best-effort)
{
  echo "[restore] Table counts:"
  echo -n "race_metadata: "; sqlite3 "$DEST_DB" "SELECT COUNT(*) FROM race_metadata;" || true
  echo -n "dog_race_data: "; sqlite3 "$DEST_DB" "SELECT COUNT(*) FROM dog_race_data;" || true
  echo -n "dogs: "; sqlite3 "$DEST_DB" "SELECT COUNT(*) FROM dogs;" || true
} | tee "$RESTORE_LOG_DIR/$TS/counts.txt"

# Save schema
sqlite3 "$DEST_DB" ".schema" > "$RESTORE_LOG_DIR/$TS/post_restore.schema.sql" || true

echo "[restore] Done. Logs at $RESTORE_LOG_DIR/$TS"

