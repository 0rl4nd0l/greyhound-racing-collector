#!/usr/bin/env bash
set -euo pipefail

# Launch Gunicorn for the Greyhound app
ROOT="/Users/test/Desktop/greyhound_racing_collector"
export GREYHOUND_DB_PATH="${GREYHOUND_DB_PATH:-/Users/test/Desktop/greyhound_racing_collector/databases/canonical_greyhound_data.db}"
export PORT="${PORT:-5002}"

# Auto-tune workers/threads if not provided
# Prefer physical cores for workers and 2 threads per worker by default
_phys=$(sysctl -n hw.physicalcpu 2>/dev/null || true)
_logi=$(sysctl -n hw.logicalcpu 2>/dev/null || true)
if [ -z "${_phys}" ]; then _phys=$(sysctl -n hw.ncpu 2>/dev/null || echo 2); fi
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

