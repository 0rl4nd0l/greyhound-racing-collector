# Makefile for the Greyhound Racing Collector project
# Updated for unified environment structure

.PHONY: help init deps lock test lint format e2e perf security schema-tests schema-baseline schema-monitor clean

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
REQUIREMENTS_DIR := requirements

help:
	@echo "Available targets:"
	@echo "  init           - Create virtual environment and install dependencies"
	@echo "  deps           - Reinstall dependencies from lock file"
	@echo "  lock           - Recompile requirements lock file from .in files"
	@echo "  install        - Legacy target (use 'deps' instead)"
	@echo "  test           - Run test suite"
	@echo "  lint           - Run linting checks"
	@echo "  format         - Format code with black and isort"
	@echo "  security       - Run security scans"
	@echo "  e2e            - Run end-to-end tests"
	@echo "  perf           - Run performance tests"
	@echo "  schema-*       - Schema monitoring commands"
	@echo "  clean          - Remove virtual environment"

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
schema-tests:
	@echo "Running database schema consistency tests..."
	pytest tests/test_database_schema_consistency.py -v

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
