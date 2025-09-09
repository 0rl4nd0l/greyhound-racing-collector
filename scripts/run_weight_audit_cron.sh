#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="/Users/test/Desktop/greyhound_racing_collector"
LOG_DIR="$REPO_DIR/logs/data_quality"
mkdir -p "$LOG_DIR"
cd "$REPO_DIR"
# Prefer writable DB if present
if [[ -f "$REPO_DIR/greyhound_racing_data_writable.db" ]]; then
  export GREYHOUND_DB_PATH="$REPO_DIR/greyhound_racing_data_writable.db"
fi
export PYTHONPATH=.
/usr/bin/env python3 scripts/weight_completeness_audit.py >> "$LOG_DIR/weight_audit_cron.log" 2>&1
