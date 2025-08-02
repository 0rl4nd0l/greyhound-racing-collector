# End-to-End Integration Testing Guide

## Overview

This document describes the comprehensive end-to-end (E2E) testing suite for the Greyhound Racing Predictor system. The tests verify complete workflows including frontend interactions, API calls, background task processing with Celery/RQ workers, and database operations.

## Test Architecture

### Test Components

1. **Playwright E2E Tests**: Browser-based tests that simulate user interactions
2. **Background Worker Integration**: Tests for Celery and RQ task processing
3. **Docker Test Environment**: Isolated testing with all dependencies
4. **API Integration Tests**: Direct API endpoint testing

### Test Suites

#### 1. Workflow Tests (`tests/playwright/e2e/workflow-*.spec.js`)

- **Workflow 1**: Download and Processing
  - Login → Download race CSV → Trigger file processing → Wait for Celery/RQ task → Verify database entries
  
- **Workflow 2**: ML Predictions
  - Navigate to ML dashboard → Request predictions → Verify probabilities displayed → Confirm storage

- **Workflow 3**: Race Notes Editing
  - Edit race notes → Confirm persistence via GET requests → Test concurrent editing

#### 2. Background Integration Tests (`tests/playwright/e2e/background-integration.spec.js`)

- **Full workflow testing**: CSV download → processing → database storage
- **Concurrent task handling**: Multiple background tasks running simultaneously  
- **Task failure scenarios**: Graceful error handling and recovery
- **Task history and cleanup**: Tracking completed tasks and cleanup procedures
- **Worker health monitoring**: Health checks and metrics collection
- **Worker restart resilience**: Handling worker restarts during task execution

## Running Tests

### Prerequisites

1. **Docker and Docker Compose** installed
2. **Node.js** (v16+) with npm
3. **Python** dependencies installed
4. **Playwright** browsers installed: `npx playwright install`

### Quick Start

```bash
# Run complete E2E integration test suite with Celery
npm run test:e2e:integration

# Run with RQ instead of Celery
npm run test:e2e:integration:rq

# Run with browser visible (non-headless)
npm run test:e2e:integration:headed

# Run only workflow tests (without full docker setup)
npm run test:e2e:workflows

# Run only background worker integration tests
npm run test:e2e:background
```

### Direct Script Execution

```bash
# Full integration test with all options
./run-e2e-tests.sh

# With environment variables
BACKEND_TYPE=rq HEADLESS=false TIMEOUT=120 ./run-e2e-tests.sh
```

### Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKEND_TYPE` | `celery` | Background task processor (`celery` or `rq`) |
| `HEADLESS` | `true` | Run browser tests in headless mode |
| `TIMEOUT` | `60` | Test timeout in seconds |
| `TEST_PROFILE` | `e2e` | Docker compose profile to use |

## Test Environment

### Docker Services

The test environment includes:

- **Flask App**: Main application server
- **PostgreSQL**: Test database
- **Redis**: Caching and task queue
- **Celery Worker**: Background task processing (when `BACKEND_TYPE=celery`)
- **RQ Worker**: Alternative background task processing (when `BACKEND_TYPE=rq`)

### Service Health Checks

All services include health checks to ensure they're ready before tests run:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
  interval: 10s
  timeout: 5s
  retries: 5
```

## Test Scenarios

### 1. Download and Processing Workflow

```javascript
// Test flow:
1. Trigger CSV download via API call
2. Monitor task status via polling
3. Verify file download completion
4. Trigger file processing task
5. Wait for processing completion
6. Verify database entries created
7. Confirm task tracking works
```

### 2. ML Predictions Workflow  

```javascript
// Test flow:
1. Navigate to ML dashboard
2. Request predictions for race
3. Verify UI shows prediction probabilities
4. Confirm predictions stored in database/cache
5. Test prediction consistency across requests
6. Handle invalid prediction requests
```

### 3. Race Notes Editing

```javascript
// Test flow:
1. Navigate to race notes interface
2. Set initial notes via API
3. Edit notes through UI or API
4. Verify persistence across page loads
5. Test concurrent editing scenarios
6. Validate notes sanitization
```

### 4. Background Worker Integration

```javascript
// Test scenarios:
- Full workflow: CSV → processing → database storage
- Concurrent task execution and management
- Task failure handling and error recovery
- Task history tracking and cleanup
- Worker health monitoring and metrics
- Worker restart resilience testing
```

## Monitoring and Debugging

### Test Reports

- **HTML Report**: Generated at `playwright-report/index.html`
- **JSON Results**: Saved as `playwright-results.json`
- **Live Report**: Available at `http://localhost:9323` during test runs

### Log Collection

Test logs are automatically collected:

- `test-logs-flask.txt`: Flask application logs
- `test-logs-postgres.txt`: Database logs  
- `test-logs-redis.txt`: Redis logs
- `test-logs-celery.txt`: Celery worker logs (if using Celery)
- `test-logs-rq.txt`: RQ worker logs (if using RQ)

### Debug Mode

```bash
# Run with visible browser and debug info
HEADLESS=false npm run test:playwright:debug

# Or use Playwright's built-in debug mode
npx playwright test --debug tests/playwright/e2e/workflow-1-download-process.spec.js
```

## Test Data Management

### Test Database

- Uses separate PostgreSQL database for isolation
- Automatically created and cleaned up
- Test data fixtures loaded before each test suite

### Test Files

- Sample CSV files in `tests/fixtures/`
- Generated test data for race information
- Cleanup performed after each test run

## Continuous Integration

### GitHub Actions Integration

```yaml
name: E2E Integration Tests
on: [push, pull_request]

jobs:
  e2e-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        backend: [celery, rq]
    
    steps:
      - uses: actions/checkout@v3
      - name: Setup Node.js
        uses: actions/setup-node@v3
      - name: Install dependencies
        run: npm ci && npx playwright install
      - name: Run E2E tests
        run: BACKEND_TYPE=${{ matrix.backend }} npm run test:e2e:integration
```

### Test Performance

- **Average test suite runtime**: 5-10 minutes
- **Individual test timeout**: 60 seconds (configurable)
- **Parallel execution**: Supported across different browser projects
- **Resource usage**: ~2GB RAM, ~1GB disk space

## Troubleshooting

### Common Issues

1. **Services not starting**: Check Docker daemon and port availability
2. **Test timeouts**: Increase `TIMEOUT` environment variable
3. **Database connection errors**: Verify PostgreSQL container health
4. **Worker not processing tasks**: Check Redis connection and worker logs

### Debug Commands

```bash
# Check service status
docker-compose -f docker-compose.test.yml ps

# View service logs
docker-compose -f docker-compose.test.yml logs flask-app

# Connect to test database
docker-compose -f docker-compose.test.yml exec postgres-test psql -U testuser -d testdb

# Monitor Redis queue
docker-compose -f docker-compose.test.yml exec redis-test redis-cli monitor
```

## Best Practices

### Writing E2E Tests

1. **Use descriptive test names** that explain the scenario
2. **Include proper wait conditions** for async operations
3. **Verify both UI and API responses** where applicable
4. **Clean up test data** after each test
5. **Use environment variables** for configuration

### Test Maintenance

1. **Regular test updates** as features evolve
2. **Monitor test performance** and optimize slow tests
3. **Update dependencies** regularly
4. **Review test coverage** and add missing scenarios

## Contributing

When adding new E2E tests:

1. Follow existing test patterns and structure
2. Add tests to appropriate workflow file or create new one
3. Update this documentation with new test scenarios
4. Ensure tests work with both Celery and RQ backends
5. Add proper error handling and cleanup

## Support

For issues with E2E testing:

1. Check the troubleshooting section
2. Review test logs in generated log files
3. Run tests in headed mode for visual debugging
4. Consult Playwright documentation for browser automation issues
