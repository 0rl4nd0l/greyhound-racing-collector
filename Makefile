# Makefile for the Greyhound Racing Collector project

.PHONY: test e2e perf security

# Install and update dependencies
install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-test.txt

# Run all tests
test:
	pytest tests/unit/ tests/integration/ --cov=.

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
