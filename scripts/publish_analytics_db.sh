#!/usr/bin/env bash
set -euo pipefail

# Publish the staging SQLite DB to the immutable analytics DB.
# - Validates staging DB appears healthy
# - Forces WAL checkpoint and sets journal_mode=DELETE
# - Atomically replaces analytics DB
# - Applies read-only perms and macOS immutable flag (uchg)
# - Cleans up WAL/SHM sidecars
#
# Env vars (with defaults):
#   STAGING_DB_PATH=greyhound_racing_data_stage.db
#   ANALYTICS_DB_PATH=greyhound_racing_data.db
#   DRY_RUN=0  # set to 1 to print steps only
#
# Usage:
#   scripts/publish_analytics_db.sh
#   STAGING_DB_PATH=/path/to/stage.db ANALYTICS_DB_PATH=/path/to/analytics.db scripts/publish_analytics_db.sh

STAGING_DB_PATH=${STAGING_DB_PATH:-greyhound_racing_data_stage.db}
ANALYTICS_DB_PATH=${ANALYTICS_DB_PATH:-greyhound_racing_data.db}
DRY_RUN=${DRY_RUN:-0}

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "+ $*"
  else
    eval "$@"
  fi
}

if [[ ! -f "$STAGING_DB_PATH" ]]; then
  echo "❌ Staging DB not found: $STAGING_DB_PATH" >&2
  exit 1
fi

# Quick sanity check on staging DB
if ! sqlite3 "$STAGING_DB_PATH" "SELECT 1;" >/dev/null 2>&1; then
  echo "❌ Cannot open staging DB: $STAGING_DB_PATH" >&2
  exit 1
fi

# Summarize staging DB counts (best-effort)
echo "🔍 Validating staging DB..."
run "sqlite3 -cmd \".mode column\" -cmd \".headers on\" $STAGING_DB_PATH \"SELECT 'dog_race_data' AS table, COUNT(*) AS rows FROM dog_race_data; SELECT 'race_metadata' AS table, COUNT(*) AS rows FROM race_metadata;\""

# Flush WAL and normalize journal mode on staging DB for a clean copy
echo "🧹 Normalizing staging DB (checkpoint WAL + set journal_mode=DELETE)..."
run "sqlite3 $STAGING_DB_PATH \"PRAGMA wal_checkpoint(TRUNCATE); PRAGMA journal_mode=DELETE; VACUUM;\""

# Unlock analytics DB if it exists (clear uchg), then prepare atomic replace
if [[ -f "$ANALYTICS_DB_PATH" ]]; then
  echo "🔓 Unlocking analytics DB (if immutable)..."
  run "chflags nouchg \"$ANALYTICS_DB_PATH\" 2>/dev/null || true"
  run "chmod 644 \"$ANALYTICS_DB_PATH\" 2>/dev/null || true"
fi

# Remove stale sidecars for analytics DB
run "rm -f \"$ANALYTICS_DB_PATH-wal\" \"$ANALYTICS_DB_PATH-shm\""

# Atomic replace: copy then move
TMP_TARGET="${ANALYTICS_DB_PATH}.tmp_publish_$$"
echo "📦 Publishing: $STAGING_DB_PATH -> $ANALYTICS_DB_PATH"
run "cp -p \"$STAGING_DB_PATH\" \"$TMP_TARGET\""
run "mv -f \"$TMP_TARGET\" \"$ANALYTICS_DB_PATH\""

# Apply protections
echo "🔐 Applying read-only and immutable flags to analytics DB..."
run "chmod 444 \"$ANALYTICS_DB_PATH\""
# macOS immutable flag; ignore failure on non-macOS
run "chflags uchg \"$ANALYTICS_DB_PATH\" 2>/dev/null || true"

# Final verification
echo "✅ Verifying analytics DB after publish..."
run "sqlite3 -cmd \".mode column\" -cmd \".headers on\" $ANALYTICS_DB_PATH \"SELECT 'dog_race_data' AS table, COUNT(*) AS rows FROM dog_race_data; SELECT 'race_metadata' AS table, COUNT(*) AS rows FROM race_metadata;\""

# Remove sidecars again if any were created during verification
run "rm -f \"$ANALYTICS_DB_PATH-wal\" \"$ANALYTICS_DB_PATH-shm\""

echo "🎉 Publish complete: $ANALYTICS_DB_PATH is locked and ready."

