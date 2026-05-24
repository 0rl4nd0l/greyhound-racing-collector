#!/usr/bin/env bash
set -euo pipefail

# Launch Gunicorn for the Greyhound app
ROOT="${GREYHOUND_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export GREYHOUND_DB_PATH="${GREYHOUND_DB_PATH:-$ROOT/greyhound_racing_data.db}"
export ANALYTICS_DB_PATH="${ANALYTICS_DB_PATH:-$GREYHOUND_DB_PATH}"
export STAGING_DB_PATH="${STAGING_DB_PATH:-$ROOT/greyhound_racing_data_stage.db}"
export PORT="${PORT:-5002}"

# Auto-tune workers/threads if not provided
# Prefer physical cores for workers and 2 threads per worker by default
_phys=$(sysctl -n hw.physicalcpu 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 2)
_logi=$(sysctl -n hw.logicalcpu 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo "$_phys")
if [ -z "${_logi}" ]; then _logi=${_phys}; fi
if [ "${_phys}" -lt 1 ]; then _phys=1; fi
_threads_per_worker=$(( _logi / _phys ))
if [ "${_threads_per_worker}" -lt 2 ]; then _threads_per_worker=2; fi
if [ "${_threads_per_worker}" -gt 4 ]; then _threads_per_worker=4; fi

export GUNI_WORKERS="${GUNI_WORKERS:-${_phys}}"
export GUNI_THREADS="${GUNI_THREADS:-${_threads_per_worker}}"
export GUNI_TIMEOUT="${GUNI_TIMEOUT:-60}"
export GUNI_KEEPALIVE="${GUNI_KEEPALIVE:-30}"
export GUNI_GRACEFUL="${GUNI_GRACEFUL:-30}"

# Conservative PATH (include common Homebrew + system paths)
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

cd "$ROOT"
# Write a small banner into logs so we can see restarts
printf "[run_gunicorn] Starting app on port %s with DB %s (workers=%s, threads=%s)\n" "$PORT" "$GREYHOUND_DB_PATH" "$GUNI_WORKERS" "$GUNI_THREADS" >> server.out 2>> server.err || true

exec python3 -m gunicorn \
  --worker-class gthread \
  --workers "$GUNI_WORKERS" \
  --threads "$GUNI_THREADS" \
  --timeout "$GUNI_TIMEOUT" \
  --graceful-timeout "$GUNI_GRACEFUL" \
  --keep-alive "$GUNI_KEEPALIVE" \
  --bind "0.0.0.0:$PORT" \
  app:app
