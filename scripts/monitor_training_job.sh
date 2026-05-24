#!/usr/bin/env bash
set -euo pipefail

JOB_ID="${1:-}"
if [ -z "$JOB_ID" ]; then
  echo "Usage: $0 JOB_ID" >&2
  exit 2
fi
BASE_URL="${BASE_URL:-http://127.0.0.1:5002}"
LOG="logs/training_monitor_${JOB_ID}.log"

mkdir -p logs
echo "Monitoring job $JOB_ID at $(date -Iseconds)" | tee -a "$LOG"

while true; do
  TS=$(date -Iseconds)
  RES=$(curl -sf "${BASE_URL}/api/model/registry/status?job_id=${JOB_ID}" || echo '{}')
  STATUS=$(printf '%s' "$RES" | ./.venv/bin/python - <<'PY'
import sys, json
try:
    j=json.loads(sys.stdin.read() or '{}')
    print(j.get('status',''))
except Exception:
    print('')
PY
)
  PROG=$(printf '%s' "$RES" | ./.venv/bin/python - <<'PY'
import sys, json
try:
    j=json.loads(sys.stdin.read() or '{}')
    p=j.get('progress', '')
    print(p if p is not None else '')
except Exception:
    print('')
PY
)
  echo "$TS status=$STATUS progress=${PROG}%" | tee -a "$LOG"
  if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ] || [ "$STATUS" = "canceled" ]; then
    exit 0
  fi
  sleep 30
done

