#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="${GREYHOUND_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG_DIR="$REPO_DIR/logs/data_quality"
mkdir -p "$LOG_DIR"
cd "$REPO_DIR"
# Prefer the local analytics DB unless the caller explicitly supplies a path.
export GREYHOUND_DB_PATH="${GREYHOUND_DB_PATH:-$REPO_DIR/greyhound_racing_data.db}"
export ANALYTICS_DB_PATH="${ANALYTICS_DB_PATH:-$GREYHOUND_DB_PATH}"
export STAGING_DB_PATH="${STAGING_DB_PATH:-$REPO_DIR/greyhound_racing_data_stage.db}"
export PYTHONPATH=.
/usr/bin/env python3 scripts/weight_completeness_audit.py >> "$LOG_DIR/weight_audit_cron.log" 2>&1
