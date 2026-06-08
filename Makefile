# Makefile for the Greyhound Racing Collector project
# Updated for unified environment structure

.PHONY: help init deps lock test lint format e2e perf security schema-tests schema-baseline schema-monitor contract-validate contract-validate-api install-hooks clean check-preflight check-v4-sanity train-win train-place calibrate-win calibrate-place backtest-win backtest-place simulate-anomalies prototype-model-upgrades

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
REQUIREMENTS_DIR := requirements

help:
	@echo "Available targets:"
	@echo "  init                 - Create virtual environment and install dependencies"
	@echo "  deps                 - Reinstall dependencies from lock file"
	@echo "  lock                 - Recompile requirements lock file from .in files"
	@echo "  install              - Legacy target (use 'deps' instead)"
	@echo "  test                 - Run test suite"
	@echo "  lint                 - Run linting checks"
	@echo "  format               - Format code with black and isort"
	@echo "  security             - Run security scans"
	@echo "  e2e                  - Run end-to-end tests"
	@echo "  perf                 - Run performance tests"
	@echo "  schema-*             - Schema monitoring commands"
	@echo "  contract-validate    - Validate feature contract (python mode, strict)"
	@echo "  contract-regenerate  - Regenerate V4 feature contract JSON"
	@echo "  contract-validate-api- Validate feature contract via API at CONTRACT_API_URL (strict)"
	@echo "  promote-gate         - Run V4 promotion gate (fails if metrics exceed thresholds)"
	@echo "  run-api-gunicorn     - Run Flask app via Gunicorn (threaded workers; SSE-friendly)"
	@echo "  run-api-gunicorn-verbose - Run Gunicorn with DEBUG logs (access/error to stdout)"
	@echo "  install-hooks        - Install git hooks (pre-push contract validation)"
	@echo "  clean                - Remove virtual environment"
	@echo "  train-win            - Train win model"
	@echo "  train-place          - Train place model (TopN=$${TOPN_PLACE:-3})"
	@echo "  calibrate-win        - Verify calibration (win); retrain calibrators"
	@echo "  calibrate-place      - Verify calibration (place); retrain calibrators"
	@echo "  backtest-win         - Backtest win model(s) -> $${BACKTEST_OUT_DIR:-backtests}/win_report.json"
	@echo "  backtest-place       - Backtest place model(s) (TopN=$${TOPN_PLACE:-3}) -> $${BACKTEST_OUT_DIR:-backtests}/place_report.json"
	@echo "  simulate-anomalies   - Write synthetic place EV anomalies to predictions/"
	@echo "  persist-predictions  - Import predictions/*.json into DB (maps to standardized race_id when possible)"
	@echo "  predict-upcoming     - Predict all CSVs in UPCOMING_RACES_DIR -> predictions/*.json"
	@echo "  predict-and-persist  - Run predict-upcoming, then persist-predictions (last 72h)"
	@echo "  prototype-model-upgrades - Run throwaway TUI for future model upgrade decisions"

$(VENV)/bin/python:
	python3.11 -m venv $(VENV)

init: $(VENV)/bin/python
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PYTHON) -m pip install -r $(REQUIREMENTS_DIR)/requirements.lock
	$(VENV)/bin/playwright install

# Install and update dependencies (legacy compatibility)
install: deps

deps:
	$(PIP) install -r $(REQUIREMENTS_DIR)/requirements.lock

lock:
	$(PYTHON) -m pip install pip-tools
	$(VENV)/bin/pip-compile --resolver=backtracking --strip-extras -q \
		-o $(REQUIREMENTS_DIR)/requirements.lock \
		-c $(REQUIREMENTS_DIR)/constraints-unified.txt \
		$(REQUIREMENTS_DIR)/all.in

# Linting and formatting
lint:
	$(VENV)/bin/black --check --diff .
	$(VENV)/bin/flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	$(VENV)/bin/isort --check-only --diff .

format:
	$(VENV)/bin/black .
	$(VENV)/bin/isort .

# Run all tests
test:
	$(PYTEST) tests/unit/ tests/integration/ --cov=.

# Run database schema consistency tests
# DESTRUCTIVE: schema-prepare archives and recreates the DB. Guarded by env checks.
schema-prepare:
	@echo "Bootstrapping clean development database from models.py (DESTRUCTIVE)..."
	@if [ "$$ALLOW_DB_RESET" != "1" ]; then \
		echo "ERROR: ALLOW_DB_RESET=1 is required to run schema-prepare"; \
		exit 1; \
	fi
	@if [ "$${ENVIRONMENT:-development}" = "production" ]; then \
		echo "ERROR: ENVIRONMENT=production is not allowed for reset"; \
		exit 1; \
	fi
	@if [ "$$CONFIRM" != "RESET_DB" ]; then \
		echo "ERROR: set CONFIRM=RESET_DB to proceed"; \
		exit 1; \
	fi
	ALLOW_DB_RESET=1 FORCE=1 $(PYTHON) scripts/bootstrap_test_db.py

# Safe schema tests (non-destructive)
schema-tests:
	@echo "Running database schema consistency tests (safe, non-destructive)..."
	$(PYTHON) tests/test_database_schema_consistency.py

# Explicitly destructive schema tests that reset the DB first
schema-tests-reset: schema-prepare schema-tests
	@echo "Completed destructive schema test run."

# Create baseline schema snapshot
schema-baseline:
	@echo "Creating baseline schema snapshot..."
	python scripts/schema_drift_monitor.py --prod-db-url="sqlite:///greyhound_racing_data.db" --create-baseline

# Run schema drift monitoring manually
schema-monitor:
	@echo "Running schema drift monitoring..."
	python scripts/schema_drift_monitor.py --prod-db-url="sqlite:///greyhound_racing_data.db"

# Run end-to-end tests
e2e:
	pytest tests/e2e/

# Run performance tests
perf:
	locust --headless -u 10 -r 1 -f load_tests/locustfile.py --run-time 2m --csv=perf-test-report

# Run security tests
# Default: focus on runtime surfaces (app, src, services, utils). Include scripts by setting BANDIT_INCLUDE_SCRIPTS=1
security: security-app
	@if [ "$$BANDIT_INCLUDE_SCRIPTS" = "1" ]; then $(MAKE) security-scripts; else echo "Skipping scripts/ scan (set BANDIT_INCLUDE_SCRIPTS=1 to include)"; fi
	$(VENV)/bin/safety scan

# Security (runtime surfaces only)
security-app:
	$(VENV)/bin/bandit -r app.py src services utils -x 'archive,archive_old_apps,archive_unused_scripts,feature_importance_backups' -q

# Security (scripts only; noisy by design)
security-scripts:
	$(VENV)/bin/bandit -r scripts -x 'archive,archive_old_apps,archive_unused_scripts,feature_importance_backups' -q

persist-predictions:
	$(PYTHON) scripts/persist_predictions_from_json.py --hours $${HOURS:-72}

predict-upcoming:
	UPCOMING_RACES_DIR=$${UPCOMING_RACES_DIR:-./upcoming_races_temp} $(PYTHON) upcoming_race_predictor_wrapper.py

predict-and-persist: predict-upcoming persist-predictions

prototype-model-upgrades:
	$(PYTHON) scripts/prototypes/model_upgrade_path_tui.py $${ARGS:-}

# Generate prediction coverage report (last HOURS, default 24)
# Usage: make report-coverage HOURS=24 DATABASE_PATH=./greyhound_racing_data.db
report-coverage:
	@echo "Generating prediction coverage report (last $${HOURS:-24}h)"
	@mkdir -p docs/analysis
	$(PYTHON) scripts/report_prediction_coverage.py --hours $${HOURS:-24} \
		--db $${DATABASE_PATH:-$${STAGING_DB_PATH:-$${GREYHOUND_DB_PATH:-./greyhound_racing_data.db}}} \
		--save docs/analysis/prediction_coverage_report_$$(date +%Y%m%d_%H%M%S).json

.PHONY: normalize-odds
normalize-odds:
	@echo "Normalizing live_odds race_id to canonical form..."
	$(PYTHON) scripts/normalize_live_odds_race_ids.py --db $${DATABASE_PATH:-$${STAGING_DB_PATH:-$${GREYHOUND_DB_PATH:-./greyhound_racing_data.db}}}

# Contract validation (python mode, no server)
contract-validate:
	@echo "Validating feature contract (python mode, strict)..."
	$(PYTHON) scripts/verify_feature_contract.py --refresh --strict --json

contract-regenerate:
	@echo "Regenerating V4 feature contract JSON..."
	$(PYTHON) scripts/regenerate_feature_contract_v4.py

# Contract validation via API (requires running server)
# Use CONTRACT_API_URL to override base URL (default http://localhost:$(PORT))
CONTRACT_API_URL ?= http://localhost:$(PORT)
contract-validate-api:
	@echo "Validating feature contract via API at $(CONTRACT_API_URL) (strict)..."
	$(PYTHON) scripts/verify_feature_contract.py --mode api --url $(CONTRACT_API_URL) --strict --json

# Install git hooks (pre-push validation)
install-hooks:
	@echo "Installing pre-push git hook for contract validation..."
	@mkdir -p .git/hooks
	@cp scripts/git-hooks/pre-push .git/hooks/pre-push
	@chmod +x .git/hooks/pre-push
	@echo "Installed .git/hooks/pre-push"

promote-gate:
	@echo "Running V4 promotion gate (optimizer OFF, simple normalization)..."
	V4_DISABLE_ACCURACY_OPTIMIZER=1 \
	V4_NORMALIZATION_MODE=$${V4_NORMALIZATION_MODE:-simple} \
	BRIER_MAX=$${BRIER_MAX:-0.125} \
	LOGLOSS_MAX=$${LOGLOSS_MAX:-0.41} \
	TOP1_MIN=$${TOP1_MIN:-0.30} \
	$(PYTHON) scripts/ci_promote_gate_v4.py

e2e-prepare:
	docker-compose -f docker-compose.test.yml run --rm playwright npx playwright install-deps

# Docker image configuration
DOCKER_IMAGE ?= greyhound-predictor
DOCKER_PORT ?= 5002
DOCKER_RACES_DIR ?= $(shell pwd)/upcoming_races_temp

# Build Docker image
.PHONY: docker-build
docker-build:
	@echo "Building Docker image: $(DOCKER_IMAGE)"
	docker build -t $(DOCKER_IMAGE) .

# Run the API in Docker (toolbar off by default)
.PHONY: run-docker-api
run-docker-api: docker-build
	@echo "Running $(DOCKER_IMAGE) on http://localhost:$(DOCKER_PORT) (toolbar off)"
	docker run --rm -it \
		-p $(DOCKER_PORT):5002 \
		-e PORT=5002 \
		-e UPCOMING_RACES_DIR=/app/upcoming_races_temp \
		-e ENABLE_ENDPOINT_DROPDOWNS=0 \
		-e DISABLE_ASSET_MINIFY=$${DISABLE_ASSET_MINIFY:-1} \
		-e TESTING=$${TESTING:-false} \
		-e V4_DISABLE_ACCURACY_OPTIMIZER=$${V4_DISABLE_ACCURACY_OPTIMIZER:-0} \
		-e GUNICORN_LOGLEVEL=$${GUNICORN_LOGLEVEL:-debug} \
		-e GUNICORN_ACCESSLOG=$${GUNICORN_ACCESSLOG:--} \
		-e GUNICORN_ERRORLOG=$${GUNICORN_ERRORLOG:--} \
		-e LOG_LEVEL=$${LOG_LEVEL:-DEBUG} \
		-v "$(DOCKER_RACES_DIR):/app/upcoming_races_temp" \
		$(DOCKER_IMAGE)

# Run the API in Docker with dev toolbar enabled
.PHONY: run-docker-api-dev-toolbar
run-docker-api-dev-toolbar: docker-build
	@echo "Running $(DOCKER_IMAGE) on http://localhost:$(DOCKER_PORT) with dev toolbar (ENABLE_ENDPOINT_DROPDOWNS=1, TESTING=true)"
	docker run --rm -it \
		-p $(DOCKER_PORT):5002 \
		-e PORT=5002 \
		-e UPCOMING_RACES_DIR=/app/upcoming_races_temp \
		-e ENABLE_ENDPOINT_DROPDOWNS=1 \
		-e DISABLE_ASSET_MINIFY=$${DISABLE_ASSET_MINIFY:-1} \
		-e TESTING=true \
		-e V4_DISABLE_ACCURACY_OPTIMIZER=$${V4_DISABLE_ACCURACY_OPTIMIZER:-0} \
		-e GUNICORN_LOGLEVEL=$${GUNICORN_LOGLEVEL:-debug} \
		-e GUNICORN_ACCESSLOG=$${GUNICORN_ACCESSLOG:--} \
		-e GUNICORN_ERRORLOG=$${GUNICORN_ERRORLOG:--} \
		-e LOG_LEVEL=$${LOG_LEVEL:-DEBUG} \
		-v "$(DOCKER_RACES_DIR):/app/upcoming_races_temp" \
		$(DOCKER_IMAGE)

# Run the Flask API normally (toolbar off by default)
# Note: Enhanced accuracy optimizer is DISABLED by default. To enable for dev runs:
#   export V4_DISABLE_ACCURACY_OPTIMIZER=0 && make run-api
.PHONY: run-api
run-api:
	@echo "Starting Flask app on port $${PORT:-5002} (toolbar off)"
	PORT=$${PORT:-5002} \
	ENABLE_ENDPOINT_DROPDOWNS=$${ENABLE_ENDPOINT_DROPDOWNS:-0} \
	DISABLE_ASSET_MINIFY=$${DISABLE_ASSET_MINIFY:-1} \
	TESTING=$${TESTING:-false} \
	LOG_LEVEL=$${LOG_LEVEL:-DEBUG} \
	ANALYTICS_DB_PATH=$${ANALYTICS_DB_PATH:-$(shell pwd)/greyhound_racing_data.db} \
	STAGING_DB_PATH=$${STAGING_DB_PATH:-$(shell pwd)/greyhound_racing_data_stage.db} \
	GREYHOUND_DB_PATH=$${GREYHOUND_DB_PATH:-$(shell pwd)/greyhound_racing_data.db} \
	V4_DISABLE_ACCURACY_OPTIMIZER=$${V4_DISABLE_ACCURACY_OPTIMIZER:-0} \
	$(PYTHON) app.py

# Run the Flask API with the dev endpoints toolbar enabled (QA convenience)
# Note: Enhanced accuracy optimizer is DISABLED by default. To enable for dev runs:
#   export V4_DISABLE_ACCURACY_OPTIMIZER=0 && make run-api-dev-toolbar
.PHONY: run-api-dev-toolbar
run-api-dev-toolbar:
	@echo "Starting Flask app with dev toolbar (ENABLE_ENDPOINT_DROPDOWNS=1, TESTING=true) on port $${PORT:-5002}"
	PORT=$${PORT:-5002} \
	ENABLE_ENDPOINT_DROPDOWNS=1 \
	TESTING=true \
	DISABLE_ASSET_MINIFY=$${DISABLE_ASSET_MINIFY:-1} \
	LOG_LEVEL=$${LOG_LEVEL:-DEBUG} \
	ANALYTICS_DB_PATH=$${ANALYTICS_DB_PATH:-$(shell pwd)/greyhound_racing_data.db} \
	STAGING_DB_PATH=$${STAGING_DB_PATH:-$(shell pwd)/greyhound_racing_data_stage.db} \
	GREYHOUND_DB_PATH=$${GREYHOUND_DB_PATH:-$(shell pwd)/greyhound_racing_data.db} \
	V4_DISABLE_ACCURACY_OPTIMIZER=$${V4_DISABLE_ACCURACY_OPTIMIZER:-0} \
	$(PYTHON) app.py

# Run the Flask API via Gunicorn (threaded workers; SSE-friendly)
# Note: By default, the enhanced accuracy optimizer is DISABLED here. To enable it for dev runs,
# export V4_DISABLE_ACCURACY_OPTIMIZER=0 before invoking this target, or use run-api-gunicorn-opt.
.PHONY: run-api-gunicorn
run-api-gunicorn:
	@echo "Starting Gunicorn on port $${PORT:-5002} ($${GUNICORN_WORKERS:-2} workers x $${GUNICORN_THREADS:-4} threads; class=$${GUNICORN_WORKER_CLASS:-gthread})"
	PORT=$${PORT:-5002} \
	ENABLE_ENDPOINT_DROPDOWNS=$${ENABLE_ENDPOINT_DROPDOWNS:-0} \
	DISABLE_ASSET_MINIFY=$${DISABLE_ASSET_MINIFY:-1} \
	TESTING=0 \
	LOG_LEVEL=$${LOG_LEVEL:-DEBUG} \
	GUNICORN_LOGLEVEL=$${GUNICORN_LOGLEVEL:-debug} \
	GUNICORN_ACCESSLOG=$${GUNICORN_ACCESSLOG:--} \
	GUNICORN_ERRORLOG=$${GUNICORN_ERRORLOG:--} \
	ANALYTICS_DB_PATH=$${ANALYTICS_DB_PATH:-$(shell pwd)/greyhound_racing_data.db} \
	STAGING_DB_PATH=$${STAGING_DB_PATH:-$(shell pwd)/greyhound_racing_data_stage.db} \
	GREYHOUND_DB_PATH=$${GREYHOUND_DB_PATH:-$(shell pwd)/greyhound_racing_data.db} \
	V4_DISABLE_ACCURACY_OPTIMIZER=$${V4_DISABLE_ACCURACY_OPTIMIZER:-0} \
	$(VENV)/bin/gunicorn -c gunicorn.conf.py app:app

# Convenience target: run API with optimizer ENABLED (V4_DISABLE_ACCURACY_OPTIMIZER=0)
.PHONY: run-api-gunicorn-opt
run-api-gunicorn-opt:
	@echo "Starting Gunicorn (optimizer ON) on port $${PORT:-5002} ($${GUNICORN_WORKERS:-2} workers x $${GUNICORN_THREADS:-4} threads; class=$${GUNICORN_WORKER_CLASS:-gthread})"
	PORT=$${PORT:-5002} \
	ENABLE_ENDPOINT_DROPDOWNS=$${ENABLE_ENDPOINT_DROPDOWNS:-0} \
	DISABLE_ASSET_MINIFY=$${DISABLE_ASSET_MINIFY:-1} \
	TESTING=$${TESTING:-false} \
	ANALYTICS_DB_PATH=$${ANALYTICS_DB_PATH:-$(shell pwd)/greyhound_racing_data.db} \
	STAGING_DB_PATH=$${STAGING_DB_PATH:-$(shell pwd)/greyhound_racing_data_stage.db} \
	GREYHOUND_DB_PATH=$${GREYHOUND_DB_PATH:-$(shell pwd)/greyhound_racing_data.db} \
	V4_DISABLE_ACCURACY_OPTIMIZER=0 \
	$(VENV)/bin/gunicorn -c gunicorn.conf.py app:app

# Convenience target: Gunicorn with DEBUG logs and unbuffered Python output
.PHONY: run-api-gunicorn-verbose
run-api-gunicorn-verbose:
	@echo "Starting Gunicorn (VERBOSE) on port $${PORT:-5002} ($${GUNICORN_WORKERS:-2} workers x $${GUNICORN_THREADS:-4} threads; class=$${GUNICORN_WORKER_CLASS:-gthread})"
	PORT=$${PORT:-5002} \
	ENABLE_ENDPOINT_DROPDOWNS=$${ENABLE_ENDPOINT_DROPDOWNS:-0} \
	DISABLE_ASSET_MINIFY=$${DISABLE_ASSET_MINIFY:-1} \
	TESTING=$${TESTING:-false} \
	LOG_LEVEL=$${LOG_LEVEL:-DEBUG} \
	ANALYTICS_DB_PATH=$${ANALYTICS_DB_PATH:-$(shell pwd)/greyhound_racing_data.db} \
	STAGING_DB_PATH=$${STAGING_DB_PATH:-$(shell pwd)/greyhound_racing_data_stage.db} \
	GREYHOUND_DB_PATH=$${GREYHOUND_DB_PATH:-$(shell pwd)/greyhound_racing_data.db} \
	V4_DISABLE_ACCURACY_OPTIMIZER=$${V4_DISABLE_ACCURACY_OPTIMIZER:-0} \
	PYTHONUNBUFFERED=1 DEBUG=$${DEBUG:-1} \
	GUNICORN_LOGLEVEL=$${GUNICORN_LOGLEVEL:-debug} \
	GUNICORN_ACCESSLOG=$${GUNICORN_ACCESSLOG:--} \
	GUNICORN_ERRORLOG=$${GUNICORN_ERRORLOG:--} \
	$(VENV)/bin/gunicorn -c gunicorn.conf.py app:app

# Run the Flask API via Gunicorn with live scraping enabled and testing disabled
.PHONY: run-api-live-gunicorn
run-api-live-gunicorn:
	@echo "Starting Gunicorn (LIVE) on port $${PORT:-5002} ($${GUNICORN_WORKERS:-2} workers x $${GUNICORN_THREADS:-4} threads; class=$${GUNICORN_WORKER_CLASS:-gthread})"
	PORT=$${PORT:-5002} \
	FLASK_ENV=$${FLASK_ENV:-development} \
	ENABLE_LIVE_SCRAPING=1 \
	ENABLE_RESULTS_SCRAPERS=1 \
	TESTING=0 \
	ENABLE_ENDPOINT_DROPDOWNS=$${ENABLE_ENDPOINT_DROPDOWNS:-0} \
	DISABLE_ASSET_MINIFY=$${DISABLE_ASSET_MINIFY:-1} \
	LOG_LEVEL=$${LOG_LEVEL:-DEBUG} \
	GUNICORN_LOGLEVEL=$${GUNICORN_LOGLEVEL:-debug} \
	GUNICORN_ACCESSLOG=$${GUNICORN_ACCESSLOG:--} \
	GUNICORN_ERRORLOG=$${GUNICORN_ERRORLOG:--} \
	ANALYTICS_DB_PATH=$${ANALYTICS_DB_PATH:-$(shell pwd)/greyhound_racing_data.db} \
	STAGING_DB_PATH=$${STAGING_DB_PATH:-$(shell pwd)/greyhound_racing_data_stage.db} \
	GREYHOUND_DB_PATH=$${GREYHOUND_DB_PATH:-$(shell pwd)/greyhound_racing_data.db} \
	V4_DISABLE_ACCURACY_OPTIMIZER=$${V4_DISABLE_ACCURACY_OPTIMIZER:-0} \
	$(VENV)/bin/gunicorn -c gunicorn.conf.py app:app

# Clean up environment
clean:
	rm -rf $(VENV)

# Database restore from latest archive SQL dump
.PHONY: db-restore-latest
db-restore-latest:
	@echo "Restoring DB from latest archive SQL dump..."
	bash scripts/restore_db_from_archive.sh

# Quick DB verification: integrity and row counts
.PHONY: db-verify
db-verify:
	@echo "Running DB integrity and row count checks..."
	@sqlite3 greyhound_racing_data.db "PRAGMA integrity_check;"
	@sqlite3 greyhound_racing_data.db "SELECT 'race_metadata', COUNT(*) FROM race_metadata;"
	@sqlite3 greyhound_racing_data.db "SELECT 'dog_race_data', COUNT(*) FROM dog_race_data;"

# Patch/verify schema columns and indexes
.PHONY: db-patch-schema
db-patch-schema:
	@echo "Verifying and patching DB schema..."
	$(PYTHON) -m scripts.verify_and_patch_schema

# App smoke test (safe, no scraping)
.PHONY: smoke-test
smoke-test:
	@echo "Running app smoke test (safe, non-network)..."
	TESTING=1 ENABLE_LIVE_SCRAPING=0 ENABLE_RESULTS_SCRAPERS=0 $(VENV)/bin/python scripts/smoke_test_app.py

# DB maintenance (non-destructive)
.PHONY: db-analyze db-vacuum guard-run

db-analyze:
	@echo "Analyzing and optimizing DB..."
	@sqlite3 greyhound_racing_data.db "PRAGMA analysis_limit=400; ANALYZE; PRAGMA optimize;"

db-vacuum:
	@echo "Vacuuming, analyzing and optimizing DB..."
	@sqlite3 greyhound_racing_data.db "VACUUM; ANALYZE; PRAGMA analysis_limit=400; PRAGMA optimize;"

# Run any writer command under DB guard (backup + integrity + optional optimize)
# Usage: make guard-run CMD='python scripts/register_latest_v4_model.py' [DB=path] [LABEL=name]
# Optional: DB_GUARD_OPTIMIZE=analyze|vacuum to enable post-op optimization
# Example: DB_GUARD_OPTIMIZE=analyze make guard-run CMD='python scripts/ingest_csv_history.py --csv "Race 7 - ... .csv"'

guard-run:
	@if [ -z "$(CMD)" ]; then echo "Usage: make guard-run CMD='python your_script.py args' [DB=path] [LABEL=name]"; exit 2; fi; \
	DB_PATH="$(DB)"; LABEL="$(LABEL)"; \
	if [ -z "$$DB_PATH" ]; then DB_PATH="greyhound_racing_data.db"; fi; \
	$(PYTHON) scripts/run_with_db_guard.py --db "$$DB_PATH" --label "$$LABEL" -- $(CMD)

# Quick ML v4 checks
.PHONY: check-preflight check-v4-sanity
check-preflight:
	@echo "Running V4 DB preflight checks..."
	$(PYTHON) scripts/dev/check_preflight.py

check-v4-sanity:
	@echo "Running V4 data preparation sanity check..."
	$(PYTHON) scripts/dev/check_v4_sanity.py --max-races $${MAX_RACES:-200}

# ------------------------------
# Dual-model (win/place) helpers
# ------------------------------
MODEL_DIR ?= models
WIN_MODEL_GLOB ?= $(MODEL_DIR)/win/*
PLACE_MODEL_GLOB ?= $(MODEL_DIR)/place/*
BACKTEST_OUT_DIR ?= backtests
TOPN_PLACE ?= 3
RACES ?= 6

train-win:
	@echo "Training win (Top1) model..."
	$(PYTHON) run_training.py --mode win

train-place:
	@echo "Training place (Top$(TOPN_PLACE)) model..."
	$(PYTHON) run_training.py --mode place --topN $(TOPN_PLACE)

calibrate-win:
	@echo "Calibrating/verifying win model(s) with isotonic calibrator (will retrain calibrators)..."
	$(PYTHON) run_calibration.py --model "$(WIN_MODEL_GLOB)" --retrain-calibrators

calibrate-place:
	@echo "Calibrating/verifying place model(s) with isotonic calibrator (will retrain calibrators)..."
	$(PYTHON) run_calibration.py --model "$(PLACE_MODEL_GLOB)" --retrain-calibrators

backtest-win:
	@echo "Backtesting win model(s) -> $(BACKTEST_OUT_DIR)/win_report.json"
	$(PYTHON) run_backtesting.py --model "$(WIN_MODEL_GLOB)" --output "$(BACKTEST_OUT_DIR)/win_report.json"

backtest-place:
	@echo "Backtesting place model(s) (Top$(TOPN_PLACE)) -> $(BACKTEST_OUT_DIR)/place_report.json"
	$(PYTHON) run_backtesting.py --model "$(PLACE_MODEL_GLOB)" --topN $(TOPN_PLACE) --output "$(BACKTEST_OUT_DIR)/place_report.json"

simulate-anomalies:
	@echo "Simulating anomalous place EV predictions (RACES=$(RACES)) into predictions/"
	$(PYTHON) scripts/simulate_place_ev_anomalies.py --races $(RACES)
