#!/usr/bin/env bash
# Setup a cron job to run the upcoming housekeeping script periodically.
# Usage: ./scripts/setup_upcoming_housekeeping_cron.sh [minute] [hour]
# Defaults: every hour at minute 15 (15 * * * *)
# Note: This writes to the user's crontab. Review before applying in production.

set -euo pipefail

MINUTE="${1:-15}"
HOUR="${2:-*}"

# Resolve repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python3}"
HOUSEKEEPING_SCRIPT="${REPO_ROOT}/scripts/archive_past_upcoming.py"
LOG_DIR="${REPO_ROOT}/logs/qa"
LOG_FILE="${LOG_DIR}/upcoming_housekeeping.cron.log"

mkdir -p "${LOG_DIR}"

# Prepare cron line
CRON_LINE="${MINUTE} ${HOUR} * * * cd ${REPO_ROOT} && DRY_RUN=0 ${PYTHON} ${HOUSEKEEPING_SCRIPT} >> ${LOG_FILE} 2>&1"

# Install or update crontab entry
( crontab -l 2>/dev/null | grep -v "archive_past_upcoming.py"; echo "${CRON_LINE}" ) | crontab -

echo "Installed cron entry: ${CRON_LINE}"
echo "Logs will be written to: ${LOG_FILE}"

