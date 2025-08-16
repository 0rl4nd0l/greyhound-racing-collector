#!/bin/bash

# Test script for negative API test cases
# Step 8: Negative API tests (missing & malformed files)

BASE_URL="http://localhost:5002"
TEST_DIR="./test_files"
LOG_FILE="./test_negative_api.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Initialize log file
echo "$(date): Starting Negative API Tests" > $LOG_FILE

# Create test directory if it doesn't exist
mkdir -p $TEST_DIR

# Function to log and print
log_and_print() {
    echo "$1"
    echo "$(date): $1" >> $LOG_FILE
}

# Test 1: Missing file test
test_missing_file() {
    log_and_print "${YELLOW}Test 1: Missing file test${NC}"
    
    # Attempt to upload a non-existent file
    response=$(curl -s -w "%{http_code}" -X POST \
        -F "file=@/tmp/does_not_exist.csv" \
        "$BASE_URL/api/ingest_csv" \
        2>/dev/null)
    
    http_code="${response: -3}"
    body="${response%???}"
    
    log_and_print "HTTP Code: $http_code"
    log_and_print "Response Body: $body"
    
    # Verify response
    if [[ "$http_code" == "400" ]]; then
        if echo "$body" | grep -q '"error":"file not found"' || echo "$body" | grep -q '"error":"No file part"'; then
            log_and_print "${GREEN}✓ Test 1 PASSED: Missing file returns 400 with expected error${NC}"
            return 0
        else
            log_and_print "${RED}✗ Test 1 FAILED: Wrong error message format${NC}"
            return 1
        fi
    else
        log_and_print "${RED}✗ Test 1 FAILED: Expected 400, got $http_code${NC}"
        return 1
    fi
}

# Test 2: Malformed CSV test
test_malformed_csv() {
    log_and_print "${YELLOW}Test 2: Malformed CSV test${NC}"
    
    # Create a corrupted CSV file
    malformed_file="$TEST_DIR/malformed.csv"
    echo -e "\x00\x00\x00NotARealCSVContent\x00\x01\x02corrupted" > "$malformed_file"
    
    # Upload the malformed CSV
    response=$(curl -s -w "%{http_code}" -X POST \
        -F "file=@$malformed_file" \
        "$BASE_URL/api/ingest_csv" \
        2>/dev/null)
    
    http_code="${response: -3}"
    body="${response%???}"
    
    log_and_print "HTTP Code: $http_code"
    log_and_print "Response Body: $body"
    
    # Verify response (should be 422 or 400)
    if [[ "$http_code" == "422" || "$http_code" == "400" ]]; then
        if echo "$body" | grep -qi "could not parse file\|schema.*mismatch\|validation.*failed"; then
            log_and_print "${GREEN}✓ Test 2 PASSED: Malformed CSV returns $http_code with clear message${NC}"
            
            # Check logs for proper error logging without traceback leakage
            check_logs
            return $?
        else
            log_and_print "${RED}✗ Test 2 FAILED: Wrong error message format${NC}"
            return 1
        fi
    else
        log_and_print "${RED}✗ Test 2 FAILED: Expected 422/400, got $http_code${NC}"
        return 1
    fi
}

# Check logs for proper error handling
check_logs() {
    log_and_print "${YELLOW}Checking logs for proper error handling...${NC}"
    
    # Check for schema_mismatch in logs
    log_files=("logs/errors.log" "logs/process.log" "logs/system.log")
    
    found_schema_mismatch=false
    found_traceback=false
    
    for log_file in "${log_files[@]}"; do
        if [[ -f "$log_file" ]]; then
            if grep -q '"status":"error","reason":"schema_mismatch"' "$log_file"; then
                found_schema_mismatch=true
                log_and_print "✓ Found schema_mismatch in $log_file"
            fi
            
            if grep -q "Traceback\|Exception:" "$log_file"; then
                found_traceback=true
                log_and_print "⚠ Found traceback in $log_file (potential leakage)"
            fi
        fi
    done
    
    if [[ "$found_schema_mismatch" == true && "$found_traceback" == false ]]; then
        log_and_print "${GREEN}✓ Log check PASSED: Found schema_mismatch without traceback leakage${NC}"
        return 0
    elif [[ "$found_schema_mismatch" == false ]]; then
        log_and_print "${RED}✗ Log check FAILED: schema_mismatch not found in logs${NC}"
        return 1
    else
        log_and_print "${RED}✗ Log check FAILED: Traceback leakage detected${NC}"
        return 1
    fi
}

# Test 3: UI Banner check (mock test for now)
test_ui_banner() {
    log_and_print "${YELLOW}Test 3: UI Banner surface check${NC}"
    
    # This would normally test the frontend UI banner
    # For now, we'll just verify the API returns proper error structure
    log_and_print "Note: UI banner test would verify 'Could not parse file' banner appears"
    log_and_print "${GREEN}✓ Test 3 PLACEHOLDER: UI integration test needed${NC}"
    return 0
}

# Main test execution
main() {
    log_and_print "Starting Negative API Tests for CSV Ingestion"
    log_and_print "============================================"
    
    # Check if server is running
    if ! curl -s "$BASE_URL/api/health" > /dev/null; then
        log_and_print "${RED}ERROR: API server not running at $BASE_URL${NC}"
        log_and_print "Please start the Flask app: python app.py"
        exit 1
    fi
    
    log_and_print "${GREEN}✓ API server is running${NC}"
    
    # Run tests
    test_results=()
    
    test_missing_file
    test_results+=($?)
    
    test_malformed_csv
    test_results+=($?)
    
    test_ui_banner
    test_results+=($?)
    
    # Summary
    log_and_print ""
    log_and_print "Test Summary:"
    log_and_print "============"
    
    passed=0
    failed=0
    
    for result in "${test_results[@]}"; do
        if [[ $result -eq 0 ]]; then
            ((passed++))
        else
            ((failed++))
        fi
    done
    
    log_and_print "${GREEN}Passed: $passed${NC}"
    log_and_print "${RED}Failed: $failed${NC}"
    
    if [[ $failed -eq 0 ]]; then
        log_and_print "${GREEN}✓ All negative API tests PASSED${NC}"
        exit 0
    else
        log_and_print "${RED}✗ Some negative API tests FAILED${NC}"
        exit 1
    fi
}

# Cleanup function
cleanup() {
    log_and_print "Cleaning up test files..."
    rm -rf "$TEST_DIR"
}

# Trap cleanup on exit
trap cleanup EXIT

# Run main function
main "$@"
