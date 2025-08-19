# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

- Primary stack: Python 3.9–3.11 (backend, data/ML), with some Node-based browser tests/utilities
- Source-of-truth for commands: Makefile targets, CI workflows under .github/workflows, and docs under docs/
- Data semantics and archive-first policy are critical in this repo (see Data semantics below)

Quickstart
- Create and activate a virtualenv, then install deps
  - python -m venv .venv && source .venv/bin/activate
  - pip install -r requirements.txt
  - pip install -r requirements-test.txt
- Optional: Node utilities for browser tests
  - npm install
  - npx playwright install --with-deps
- Environment variables commonly used (see README.md for details)
  - export UPCOMING_RACES_DIR=./upcoming_races_temp
  - export DATABASE_URL=sqlite:///greyhound_racing_data.db
  - export TESTING=true

Common commands
Backend install and setup
- Install Python deps: make install
- Train a lightweight test model (as in CI): python scripts/train_test_model.py

Lint and format (mirrors CI)
- Syntax/quality (flake8): flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
- Formatting check (black): black --check --diff .
- Import ordering (isort): isort --check-only --diff .

Tests (pytest)
- Run all tests quickly: pytest -q
- Verbose with coverage: pytest -v --cov=. --cov-report=term-missing
- Run a specific file: pytest tests/test_flask_api.py -v
- Run a single test by node: pytest tests/test_flask_api.py::TestAPI::test_health_endpoint -v
- Filter by pattern: pytest -k "temporal or leakage" -v
- Backend suites used in CI:
  - pytest tests/test_backend_suite.py tests/test_flask_api.py tests/test_backend.py --cov=app -v --tb=short

Browser/E2E tests
- Playwright (Node)
  - Install browsers: npx playwright install --with-deps
  - Run all: npm run test:playwright
  - Example targeted run: npx playwright test tests/playwright/e2e/workflow-*.spec.js
  - Debug UI: npm run test:playwright:ui
- Cypress (if needed): npm run test:e2e or npm run cypress:run

Performance and security
- Load testing (Locust): locust --headless -u 10 -r 1 -f load_tests/locustfile.py --run-time 2m --csv=perf-test-report
- Security checks (Makefile target): make security  # runs bandit -r . and safety check

Utilities
- Schema drift baseline: make schema-baseline
- Schema drift monitor: make schema-monitor
- Database schema consistency tests: make schema-tests

Data semantics (project rules to respect)
- Historical data = form guide CSVs and their derivatives. Used for model training/analysis; never used to infer the winner of the same race.
- Race data = the data for the race itself (e.g., weather, winners). The winner must be scraped from the corresponding race webpage, not the form guide.
- Form guide CSV format note: only 10 unique dogs; blank rows continue the dog above.
- Archive-first policy: before creating a new file, check archive/ for an existing one. Move outdated/redundant files to archive/ to keep the repo root clean; keep ad hoc test scripts in tests/.

High-level architecture overview
- Ingestion and scraping
  - src/collectors and src/collectors/adapters: scrapers for sources (e.g., FastTrack, The Greyhound Recorder)
  - ingestion/ingest_race_csv.py and tools/form_guide_validator.py: CSV ingestion and validation utilities
  - Separation of concerns: upcoming race CSVs (race data) live under UPCOMING_RACES_DIR; historical form guides flow through unprocessed/ → processed/ → database
- Parsing and normalization
  - src/parsers/csv_ingestion.py and utils/csv_metadata.py, utils/file_naming.py, utils/race_file_utils.py: normalize filenames/CSV structure; enforce schemas and naming standards
  - Strong emphasis on schema conformance and key consistency (see docs/schema* and tests/*schema* and tests/*key_consistency*)
- Feature engineering and prediction
  - src/predictor/feature_builders/* (e.g., fasttrack_features.py) and features/feature_store.py encapsulate feature generation
  - ML prediction pipelines (current V4 summarized in docs/ML_SYSTEM_V4_README.md) with leakage protection and probability calibration; legacy v3 components are archived under archive/
- Services and API
  - fastapi_app/main.py provides an API surface (there are also Flask-oriented tests; the API layer may be in transition—consult tests and README endpoints)
  - services/guardian_service.py and monitoring/prometheus_exporter.py provide health/monitoring interfaces
- Orchestration and tooling
  - scripts/ and tools/ contain operational scripts (validation, performance guardrails, docs updating, etc.)
  - GitHub Actions workflows run linting, tests, schema checks, Playwright installation, and load tests
- Storage and migrations
  - migrations/ and alembic/ track schema evolution; Alembic is used for DB migrations (see docs/migrations and CI steps)

Notes for running and developing
- Upcoming races: The app enumerates UPCOMING_RACES_DIR; files are immediately discoverable by the UI/API. Keep outcome fields out of upcoming CSVs.
- Chrome/Playwright: CI installs Playwright browsers and ChromeDriver. Locally, run npx playwright install --with-deps before browser tests.
- Environment parity: Use the same environment variables as in CI when reproducing failures (DATABASE_URL, REDIS_URL, TESTING, UPCOMING_RACES_DIR).

Docker (optional)
- Build the image:
  - docker build -t greyhound-predictor .
- Run the API (maps port 5002 and mounts upcoming races directory):
  - docker run --rm -it \
      -p 5002:5002 \
      -e PORT=5002 \
      -e UPCOMING_RACES_DIR=/app/upcoming_races_temp \
      -v "$(pwd)/upcoming_races_temp:/app/upcoming_races_temp" \
      greyhound-predictor
- Health check inside the container is configured; once running, check:
  - curl http://localhost:5002/api/health
- Notes:
  - The container defaults UPCOMING_RACES_DIR=/app/upcoming_races_temp (see Dockerfile). Mount your host folder to that path for live CSV discovery.
  - For browser/E2E tests, prefer running them on the host per CI parity unless you maintain a compose setup.

Key references
- README.md: end-to-end usage patterns, API endpoints, upcoming races conventions, environment variables
- docs/
  - SYSTEM_ARCHITECTURE.md, architecture/*: system and data flow overviews
  - ML_SYSTEM_V4_README.md: current ML pipeline high level
  - BACKEND_TESTING.md: backend testing approach and targets
  - FASTTRACK_INTEGRATION_GUIDE.md and FORM_GUIDE_SPEC.md: data source details and CSV specs
  - PIPELINE_SUMMARY.md and data_dictionary/*: pipeline and feature definitions
- .github/workflows/*.yml: authoritative CI commands for linting, testing, migrations, and performance checks
- Warp workflows: .warp/workflows/common.yaml (quick actions for pytest, schema checks, security, and Playwright)

