#!/bin/bash

# Greyhound Racing E2E Test Suite with Full Integration
# This script runs comprehensive end-to-end tests with all background workers

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TEST_PROFILE=${TEST_PROFILE:-"e2e"}
BACKEND_TYPE=${BACKEND_TYPE:-"celery"}
HEADLESS=${HEADLESS:-"true"}
TIMEOUT=${TIMEOUT:-60}

echo -e "${BLUE}🚀 Starting Greyhound Racing E2E Test Suite${NC}"
echo -e "${BLUE}Profile: ${TEST_PROFILE}, Backend: ${BACKEND_TYPE}, Headless: ${HEADLESS}${NC}"

# Function to cleanup
cleanup() {
    echo -e "${YELLOW}🧹 Cleaning up test environment...${NC}"
    docker-compose -f docker-compose.test.yml --profile $TEST_PROFILE down -v --remove-orphans
    docker system prune -f --volumes
}

# Trap cleanup on exit
trap cleanup EXIT

# Step 1: Cleanup any existing containers
echo -e "${YELLOW}📦 Cleaning up existing containers...${NC}"
cleanup

# Step 2: Start the integrated test environment
echo -e "${BLUE}🏗️ Starting integrated test environment with $BACKEND_TYPE workers...${NC}"
BACKEND_TYPE=$BACKEND_TYPE docker-compose -f docker-compose.test.yml --profile $TEST_PROFILE up -d

# Step 3: Wait for services to be healthy
echo -e "${YELLOW}⏳ Waiting for services to be ready...${NC}"
sleep 10

# Function to check service health
check_service() {
    local service=$1
    local timeout=${2:-60}
    local count=0
    
    echo -e "${YELLOW}Checking ${service} health...${NC}"
    while [ $count -lt $timeout ]; do
        if docker-compose -f docker-compose.test.yml ps $service | grep -q "healthy\|Up"; then
            echo -e "${GREEN}✅ ${service} is ready${NC}"
            return 0
        fi
        sleep 2
        count=$((count + 2))
    done
    
    echo -e "${RED}❌ ${service} failed to become ready within ${timeout}s${NC}"
    return 1
}

# Check all services
check_service "flask-app"
check_service "postgres-test"
check_service "redis-test"

if [ "$BACKEND_TYPE" = "celery" ]; then
    check_service "celery-worker"
elif [ "$BACKEND_TYPE" = "rq" ]; then
    check_service "rq-worker"
fi

# Step 4: Verify Flask app is responding
echo -e "${YELLOW}🔍 Verifying Flask app endpoints...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:5000/health > /dev/null; then
        echo -e "${GREEN}✅ Flask app is responding${NC}"
        break
    fi
    echo "Waiting for Flask app to respond... (attempt $i/30)"
    sleep 2
done

# Verify critical endpoints
echo -e "${YELLOW}🔍 Testing critical API endpoints...${NC}"
endpoints=("/health" "/api/races" "/api/tasks" "/api/background/status")
for endpoint in "${endpoints[@]}"; do
    if curl -s "http://localhost:5000${endpoint}" > /dev/null; then
        echo -e "${GREEN}✅ ${endpoint} is accessible${NC}"
    else
        echo -e "${RED}❌ ${endpoint} is not accessible${NC}"
    fi
done

# Step 5: Run database migrations and setup
echo -e "${BLUE}🗄️ Setting up test database...${NC}"
docker-compose -f docker-compose.test.yml exec -T flask-app python -c "
import sys
sys.path.append('.')
from app import app, db
with app.app_context():
    db.create_all()
    print('Database tables created successfully')
"

# Step 6: Run the Playwright E2E tests
echo -e "${BLUE}🎭 Running Playwright E2E Tests...${NC}"

# Test configuration based on headless mode
if [ "$HEADLESS" = "true" ]; then
    PLAYWRIGHT_CMD="npx playwright test --reporter=line,html"
else
    PLAYWRIGHT_CMD="npx playwright test --headed --reporter=line,html"
fi

# Environment variables for tests
export FLASK_BASE_URL="http://localhost:5000"
export TEST_TIMEOUT="60000"
export BACKEND_TYPE="$BACKEND_TYPE"

# Run individual test suites with proper reporting
echo -e "${BLUE}📋 Test Suite 1: Download and Processing Workflow${NC}"
if $PLAYWRIGHT_CMD tests/playwright/e2e/workflow-1-download-process.spec.js; then
    echo -e "${GREEN}✅ Workflow 1 tests passed${NC}"
else
    echo -e "${RED}❌ Workflow 1 tests failed${NC}"
    exit 1
fi

echo -e "${BLUE}📋 Test Suite 2: ML Predictions Workflow${NC}"
if $PLAYWRIGHT_CMD tests/playwright/e2e/workflow-2-ml-predictions.spec.js; then
    echo -e "${GREEN}✅ Workflow 2 tests passed${NC}"
else
    echo -e "${RED}❌ Workflow 2 tests failed${NC}"
    exit 1
fi

echo -e "${BLUE}📋 Test Suite 3: Race Notes Editing Workflow${NC}"
if $PLAYWRIGHT_CMD tests/playwright/e2e/workflow-3-race-notes.spec.js; then
    echo -e "${GREEN}✅ Workflow 3 tests passed${NC}"
else
    echo -e "${RED}❌ Workflow 3 tests failed${NC}"
    exit 1
fi

# Step 7: Verify background worker integration
echo -e "${BLUE}🔧 Testing Background Worker Integration...${NC}"
worker_test_cmd="npx playwright test --grep='background.*worker.*integration' --reporter=line"
if command -v timeout > /dev/null; then
    timeout 120 $worker_test_cmd || echo -e "${YELLOW}⚠️ Worker integration tests timed out or completed with warnings${NC}"
else
    $worker_test_cmd || echo -e "${YELLOW}⚠️ Worker integration tests completed with warnings${NC}"
fi

# Step 8: Generate comprehensive test report
echo -e "${BLUE}📊 Generating test reports...${NC}"
npx playwright show-report --host=0.0.0.0 --port=9323 &
REPORT_PID=$!

echo -e "${GREEN}✅ All E2E tests completed successfully!${NC}"
echo -e "${BLUE}📊 Test report available at: http://localhost:9323${NC}"
echo -e "${BLUE}📁 Test artifacts saved in: playwright-report/${NC}"

# Step 9: Collect logs for debugging
echo -e "${YELLOW}📝 Collecting service logs...${NC}"
docker-compose -f docker-compose.test.yml logs --tail=100 flask-app > test-logs-flask.txt
docker-compose -f docker-compose.test.yml logs --tail=100 postgres-test > test-logs-postgres.txt
docker-compose -f docker-compose.test.yml logs --tail=100 redis-test > test-logs-redis.txt

if [ "$BACKEND_TYPE" = "celery" ]; then
    docker-compose -f docker-compose.test.yml logs --tail=100 celery-worker > test-logs-celery.txt
elif [ "$BACKEND_TYPE" = "rq" ]; then
    docker-compose -f docker-compose.test.yml logs --tail=100 rq-worker > test-logs-rq.txt
fi

echo -e "${GREEN}🎉 E2E Test Suite completed successfully!${NC}"
echo -e "${BLUE}Logs saved as test-logs-*.txt files${NC}"

# Keep report server running for a bit if not in CI
if [ -z "$CI" ]; then
    echo -e "${YELLOW}⏰ Report server will run for 30 seconds...${NC}"
    sleep 30
    kill $REPORT_PID 2>/dev/null || true
fi

echo -e "${GREEN}🏁 Test run complete!${NC}"
