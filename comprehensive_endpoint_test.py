#!/usr/bin/env python3
"""
Comprehensive Endpoint Testing Script
=====================================

Tests all API endpoints and frontend routes to ensure full functionality.
"""

import json
import sys
import time
from datetime import datetime
from urllib.parse import urljoin

import requests

BASE_URL = "http://127.0.0.1:5002"
TIMEOUT = 10

# Test results tracking
test_results = {"passed": [], "failed": [], "warnings": []}


def log_test(endpoint, status, message=""):
    """Log test result"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    result_str = f"[{timestamp}] {endpoint} - {status}"
    if message:
        result_str += f": {message}"

    print(result_str)

    if status == "PASS":
        test_results["passed"].append((endpoint, message))
    elif status == "FAIL":
        test_results["failed"].append((endpoint, message))
    else:
        test_results["warnings"].append((endpoint, message))


def test_endpoint(method, endpoint, expected_status=200, data=None, headers=None):
    """Test a single endpoint"""
    url = urljoin(BASE_URL, endpoint)

    try:
        if method.upper() == "GET":
            response = requests.get(url, timeout=TIMEOUT, headers=headers)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, timeout=TIMEOUT, headers=headers)
        elif method.upper() == "PUT":
            response = requests.put(url, json=data, timeout=TIMEOUT, headers=headers)
        elif method.upper() == "DELETE":
            response = requests.delete(url, timeout=TIMEOUT, headers=headers)
        else:
            log_test(endpoint, "FAIL", f"Unsupported method: {method}")
            return False

        if response.status_code == expected_status:
            log_test(endpoint, "PASS", f"{method} {response.status_code}")
            return True
        else:
            log_test(
                endpoint,
                "FAIL",
                f"{method} Expected {expected_status}, got {response.status_code}",
            )
            return False

    except requests.exceptions.RequestException as e:
        log_test(endpoint, "FAIL", f"{method} Request error: {str(e)}")
        return False
    except Exception as e:
        log_test(endpoint, "FAIL", f"{method} Unexpected error: {str(e)}")
        return False


def test_core_api_endpoints():
    """Test core API endpoints"""
    print("\n=== Testing Core API Endpoints ===")

    # Health and status endpoints
    test_endpoint("GET", "/api/health")
    test_endpoint("GET", "/api/model_health")
    test_endpoint("GET", "/api/status")
    test_endpoint("GET", "/api/server_port")

    # Dog-related endpoints
    test_endpoint("GET", "/api/dogs")
    test_endpoint("GET", "/api/dogs/search?q=test")
    test_endpoint("GET", "/api/all_dogs")
    test_endpoint("GET", "/api/top_performers")
    test_endpoint("GET", "/api/dogs/1", expected_status=[200, 404])  # May not exist

    # Race-related endpoints
    test_endpoint("GET", "/api/races")
    test_endpoint("GET", "/api/races/paginated")
    test_endpoint("GET", "/api/recent_races")
    test_endpoint("GET", "/api/upcoming_races")
    test_endpoint("GET", "/api/upcoming_races_csv")

    # Prediction endpoints
    test_endpoint("GET", "/api/predict")  # May need race file
    test_endpoint("GET", "/api/prediction_results")
    test_endpoint("GET", "/api/prediction_insights")


def test_workflow_endpoints():
    """Test workflow and task endpoints"""
    print("\n=== Testing Workflow Endpoints ===")

    test_endpoint("GET", "/api/workflow/data")
    test_endpoint("GET", "/api/workflow/scraper_status")
    test_endpoint("GET", "/api/workflow/processing_status")
    test_endpoint("GET", "/api/tasks/status")


def test_diagnostics_endpoints():
    """Test diagnostics and monitoring endpoints"""
    print("\n=== Testing Diagnostics Endpoints ===")

    test_endpoint("GET", "/api/diagnostics/jobs")
    test_endpoint("GET", "/api/diagnostics/summary")
    test_endpoint("GET", "/api/logs")
    test_endpoint("GET", "/api/logs/application")
    test_endpoint("GET", "/api/logs/model_registry")


def test_batch_prediction_endpoints():
    """Test batch prediction endpoints"""
    print("\n=== Testing Batch Prediction Endpoints ===")

    test_endpoint("GET", "/api/batch/jobs")
    test_endpoint("GET", "/api/batch/status")


def test_model_registry_endpoints():
    """Test model registry endpoints"""
    print("\n=== Testing Model Registry Endpoints ===")

    test_endpoint("GET", "/api/models")
    test_endpoint("GET", "/api/models/metadata")
    test_endpoint("GET", "/api/models/performance")
    test_endpoint("GET", "/api/models/best")


def test_enhanced_prediction_endpoints():
    """Test enhanced prediction service endpoints"""
    print("\n=== Testing Enhanced Prediction Endpoints ===")

    test_endpoint("GET", "/api/predict_single_race_enhanced")
    test_endpoint("GET", "/api/enhanced_prediction_quality")


def test_frontend_routes():
    """Test frontend HTML routes"""
    print("\n=== Testing Frontend Routes ===")

    frontend_routes = [
        "/",
        "/index",
        "/races",
        "/predict",
        "/monitoring",
        "/scraping_status",
        "/logs_viewer",
        "/model_registry",
        "/uploads",
        "/interactive_races",
        "/ml_dashboard",
        "/ml_training",
    ]

    for route in frontend_routes:
        test_endpoint("GET", route, expected_status=[200, 302])  # May redirect


def test_file_upload_endpoints():
    """Test file upload functionality (without actual files)"""
    print("\n=== Testing File Upload Endpoints ===")

    # These will fail without actual files, but we can test they exist
    test_endpoint(
        "POST", "/upload_race_file", expected_status=[400, 422]
    )  # No file provided
    test_endpoint("GET", "/api/file_stats", expected_status=[200, 404])


def test_scraping_endpoints():
    """Test scraping and data collection endpoints"""
    print("\n=== Testing Scraping Endpoints ===")

    test_endpoint("GET", "/api/scraping/status")
    test_endpoint("GET", "/api/scraping/config")


def run_all_tests():
    """Run all endpoint tests"""
    print(f"Starting comprehensive endpoint testing at {datetime.now()}")
    print(f"Testing against: {BASE_URL}")

    # Test core functionality first
    test_core_api_endpoints()

    # Test workflow and tasks
    test_workflow_endpoints()

    # Test diagnostics
    test_diagnostics_endpoints()

    # Test batch predictions
    test_batch_prediction_endpoints()

    # Test model registry
    test_model_registry_endpoints()

    # Test enhanced predictions
    test_enhanced_prediction_endpoints()

    # Test scraping
    test_scraping_endpoints()

    # Test file uploads
    test_file_upload_endpoints()

    # Test frontend routes
    test_frontend_routes()

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"✅ PASSED: {len(test_results['passed'])}")
    print(f"❌ FAILED: {len(test_results['failed'])}")
    print(f"⚠️  WARNINGS: {len(test_results['warnings'])}")

    if test_results["failed"]:
        print("\nFAILED TESTS:")
        for endpoint, message in test_results["failed"]:
            print(f"  ❌ {endpoint}: {message}")

    total_tests = len(test_results["passed"]) + len(test_results["failed"])
    success_rate = (
        (len(test_results["passed"]) / total_tests * 100) if total_tests > 0 else 0
    )
    print(f"\nSuccess Rate: {success_rate:.1f}%")

    return len(test_results["failed"]) == 0


if __name__ == "__main__":
    # Wait for app to be ready
    print("Waiting for Flask app to be ready...")
    time.sleep(3)

    # Run tests
    success = run_all_tests()

    # Exit with proper code
    sys.exit(0 if success else 1)
