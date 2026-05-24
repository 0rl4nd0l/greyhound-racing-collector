#!/bin/bash

# Test runner script for Greyhound Racing Dashboard
# This script starts the Flask app in testing mode and runs tests

set -e  # Exit on any error

# Default values
TEST_TYPE="all"
PORT=${PORT:-5002}
HOST=${HOST:-localhost}
HEADLESS=${HEADLESS:-true}
FLASK_PID=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to cleanup processes
cleanup() {
    if [ ! -z "$FLASK_PID" ]; then
        print_status "Stopping Flask app (PID: $FLASK_PID)..."
        kill $FLASK_PID 2>/dev/null || true
        wait $FLASK_PID 2>/dev/null || true
    fi
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Function to check if port is available
check_port() {
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
        print_error "Port $PORT is already in use. Please stop the service or use a different port."
        exit 1
    fi
}

# Function to start Flask app in testing mode
start_flask() {
    print_status "Starting Flask app in testing mode on $HOST:$PORT..."
    
    # Set testing environment variables
    export TESTING=true
    export FLASK_ENV=testing
    export MODULE_GUARD_STRICT=0
    export PREDICTION_IMPORT_MODE=relaxed
    
    # Check if virtual environment exists and activate it
    if [ -d ".venv" ]; then
        print_status "Activating virtual environment..."
        source .venv/bin/activate
    elif [ -d "venv" ]; then
        print_status "Activating virtual environment..."
        source venv/bin/activate
    else
        print_warning "No virtual environment found. Using system Python."
    fi
    
    # Start Flask app in background
    python app.py --host $HOST --port $PORT &
    FLASK_PID=$!
    
    print_status "Flask app started with PID: $FLASK_PID"
    print_status "Waiting for Flask app to be ready..."
    
    # Wait for Flask app to start (up to 30 seconds)
    for i in {1..30}; do
        if curl -s "http://$HOST:$PORT/ping" >/dev/null 2>&1; then
            print_success "Flask app is ready!"
            break
        fi
        if [ $i -eq 30 ]; then
            print_error "Flask app failed to start within 30 seconds"
            exit 1
        fi
        sleep 1
    done
}

# Function to install dependencies
install_deps() {
    print_status "Installing Node.js dependencies..."
    npm install
    
    if command -v playwright &> /dev/null; then
        print_status "Installing Playwright browsers..."
        npx playwright install
    fi
}

# Function to run Cypress tests
run_cypress() {
    print_status "Running Cypress tests..."
    if [ "$HEADLESS" = "true" ]; then
        npm run cypress:run
    else
        npm run cypress:open
    fi
}

# Function to run Playwright tests
run_playwright() {
    print_status "Running Playwright tests..."
    if [ "$HEADLESS" = "true" ]; then
        npm run test:playwright
    else
        npm run test:playwright:headed
    fi
}

# Function to run specific test file
run_specific_test() {
    local test_file=$1
    local framework=$2
    
    print_status "Running specific test: $test_file with $framework"
    
    if [ "$framework" = "cypress" ]; then
        npx cypress run --spec "$test_file"
    elif [ "$framework" = "playwright" ]; then
        npx playwright test "$test_file"
    fi
}

# Main function
main() {
    print_status "=== Greyhound Racing Dashboard Test Runner ==="
    print_status "Test Type: $TEST_TYPE"
    print_status "Port: $PORT"
    print_status "Host: $HOST"
    print_status "Headless: $HEADLESS"
    echo
    
    # Check if port is available
    check_port
    
    # Install dependencies if needed
    if [ ! -d "node_modules" ]; then
        install_deps
    fi
    
    # Start Flask app
    start_flask
    
    # Run tests based on type
    case $TEST_TYPE in
        "cypress")
            run_cypress
            ;;
        "playwright")
            run_playwright
            ;;
        "helper-routes")
            print_status "Running helper routes tests..."
            run_specific_test "cypress/e2e/test-helper-routes.cy.js" "cypress"
            run_specific_test "tests/playwright/test-helper-routes.spec.js" "playwright"
            ;;
        "all")
            print_status "Running all tests..."
            run_cypress
            run_playwright
            ;;
        *)
            print_error "Unknown test type: $TEST_TYPE"
            print_error "Valid options: cypress, playwright, helper-routes, all"
            exit 1
            ;;
    esac
    
    print_success "All tests completed successfully!"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            TEST_TYPE="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -h|--host)
            HOST="$2"
            shift 2
            ;;
        --headed)
            HEADLESS="false"
            shift
            ;;
        --install)
            install_deps
            exit 0
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -t, --type TYPE     Test type (cypress, playwright, helper-routes, all)"
            echo "  -p, --port PORT     Port for Flask app (default: 5002)"
            echo "  -h, --host HOST     Host for Flask app (default: localhost)"
            echo "      --headed        Run tests in headed mode (with browser UI)"
            echo "      --install       Only install dependencies and exit"
            echo "      --help          Show this help message"
            echo
            echo "Examples:"
            echo "  $0                           # Run all tests"
            echo "  $0 -t cypress               # Run only Cypress tests"
            echo "  $0 -t helper-routes         # Run only helper routes tests"
            echo "  $0 --headed                 # Run tests with browser UI"
            echo "  $0 -p 5003                  # Use port 5003"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main
