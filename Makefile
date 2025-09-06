# Makefile for the Greyhound Racing Collector project
# Updated for unified environment structure

.PHONY: help init deps lock test lint format e2e perf security schema-tests schema-baseline schema-monitor contract-validate contract-validate-api install-hooks clean check-preflight check-v4-sanity

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
	@echo "  contract-validate-api- Validate feature contract via API at CONTRACT_API_URL (strict)"
	@echo "  install-hooks        - Install git hooks (pre-push contract validation)"
	@echo "  clean                - Remove virtual environment"

$(VENV)/bin/python:
	python3.11 -m venv $(VENV)

init: $(VENV)/bin/python
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r $(REQUIREMENTS_DIR)/requirements.lock
	$(VENV)/bin/playwright install

# Install and update dependencies (legacy compatibility)
install: deps

deps:
	$(PIP) install -r $(REQUIREMENTS_DIR)/requirements.lock

lock:
	cd $(REQUIREMENTS_DIR) && $(PIP) install pip-tools
	cd $(REQUIREMENTS_DIR) && pip-compile --resolver=backtracking --strip-extras -q -o requirements.lock -c constraints-unified.txt all.in

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
security:
	bandit -r .
	safety check

# Contract validation (python mode, no server)
contract-validate:
	@echo "Validating feature contract (python mode, strict)..."
	$(PYTHON) scripts/verify_feature_contract.py --refresh --strict --json

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
		-v "$(DOCKER_RACES_DIR):/app/upcoming_races_temp" \
		$(DOCKER_IMAGE)

# Run the Flask API normally (toolbar off by default)
.PHONY: run-api
run-api:
	@echo "Starting Flask app on port $${PORT:-5002} (toolbar off)"
	PORT=$${PORT:-5002} \
	ENABLE_ENDPOINT_DROPDOWNS=$${ENABLE_ENDPOINT_DROPDOWNS:-0} \
	DISABLE_ASSET_MINIFY=$${DISABLE_ASSET_MINIFY:-1} \
	TESTING=$${TESTING:-false} \
	$(PYTHON) app.py

# Run the Flask API with the dev endpoints toolbar enabled (QA convenience)
.PHONY: run-api-dev-toolbar
run-api-dev-toolbar:
	@echo "Starting Flask app with dev toolbar (ENABLE_ENDPOINT_DROPDOWNS=1, TESTING=true) on port $${PORT:-5002}"
	PORT=$${PORT:-5002} \
	ENABLE_ENDPOINT_DROPDOWNS=1 \
	TESTING=true \
	DISABLE_ASSET_MINIFY=$${DISABLE_ASSET_MINIFY:-1} \
	$(PYTHON) app.py

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
	python3 scripts/verify_and_patch_schema.py

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
