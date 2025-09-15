#!/usr/bin/env bash
set -euo pipefail

# Persistent live server launcher for local dev
# - Ensures live scraping flags are enabled
# - Disables testing mode
# - Uses project venv gunicorn when available

PORT="${PORT:-5002}"

export FLASK_ENV="${FLASK_ENV:-development}"
export ENABLE_LIVE_SCRAPING=1
export ENABLE_RESULTS_SCRAPERS=1
export TESTING=0
export DISABLE_ASSET_MINIFY="${DISABLE_ASSET_MINIFY:-1}"
export ENABLE_ENDPOINT_DROPDOWNS="${ENABLE_ENDPOINT_DROPDOWNS:-0}"
export V4_DISABLE_ACCURACY_OPTIMIZER="${V4_DISABLE_ACCURACY_OPTIMIZER:-0}"
export LOG_LEVEL="${LOG_LEVEL:-DEBUG}"
export GUNICORN_LOGLEVEL="${GUNICORN_LOGLEVEL:-debug}"
export GUNICORN_ACCESSLOG="${GUNICORN_ACCESSLOG:--}"
export GUNICORN_ERRORLOG="${GUNICORN_ERRORLOG:--}"

# Prefer project venv gunicorn
if [[ -x ".venv/bin/gunicorn" ]]; then
  GUNICORN=".venv/bin/gunicorn"
elif command -v gunicorn >/dev/null 2>&1; then
  GUNICORN="gunicorn"
else
  echo "gunicorn not found. Create a venv and install deps: make install" >&2
  exit 1
fi

echo "Starting Gunicorn (LIVE) on :$PORT (FLASK_ENV=$FLASK_ENV, TESTING=$TESTING, ENABLE_LIVE_SCRAPING=$ENABLE_LIVE_SCRAPING)"
exec "$GUNICORN" -c gunicorn.conf.py app:app

