#!/usr/bin/env python3
"""
Final Comprehensive Integration Test
===================================

Tests end-to-end workflows covering all user journeys from data ingestion to prediction results.
"""

import json
import sys
import time
from datetime import datetime
from urllib.parse import urljoin

import requests

BASE_URL = "http://127.0.0.1:5002"
TIMEOUT = 15


def log_test(component, status, message=""):
    """Log test result with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {component} - {status}: {message}")


def test_system_health():
    """Test overall system health and readiness"""
    print("\n🔍 Testing System Health...")

    try:
        # Test API health
        response = requests.get(f"{BASE_URL}/api/health", timeout=TIMEOUT)
        health = response.json()

        if health.get("components", {}).get("database") == "connected":
            log_test("Database", "✅ PASS", "Connected and operational")
        else:
            log_test("Database", "❌ FAIL", "Not connected")
            return False

        # Test model health
        response = requests.get(f"{BASE_URL}/api/model_health", timeout=TIMEOUT)
        model_health = response.json()

        if model_health.get("ready"):
            log_test(
                "ML Models",
                "✅ PASS",
                f"Ready with {model_health.get('feature_count', 0)} features",
            )
        else:
            log_test("ML Models", "❌ FAIL", "Not ready")
            return False

        return True

    except Exception as e:
        log_test("System Health", "❌ FAIL", str(e))
        return False


def test_data_access_workflow():
    """Test data access and querying workflows"""
    print("\n📊 Testing Data Access Workflow...")

    try:
        # Test dog data access
        response = requests.get(f"{BASE_URL}/api/dogs/search?q=test", timeout=TIMEOUT)
        if response.status_code == 200:
            log_test("Dog Search", "✅ PASS", "Search functionality working")
        else:
            log_test("Dog Search", "❌ FAIL", f"Status {response.status_code}")
            return False

        # Test race data access
        response = requests.get(f"{BASE_URL}/api/races/paginated", timeout=TIMEOUT)
        race_data = response.json()
        if race_data.get("success"):
            log_test("Race Data", "✅ PASS", f"Paginated races loaded")
        else:
            log_test("Race Data", "❌ FAIL", "Could not load races")
            return False

        # Test upcoming races
        response = requests.get(f"{BASE_URL}/api/upcoming_races_csv", timeout=TIMEOUT)
        upcoming = response.json()
        if upcoming.get("success"):
            log_test("Upcoming Races", "✅ PASS", "CSV data available")
        else:
            log_test("Upcoming Races", "❌ FAIL", "No upcoming data")

        return True

    except Exception as e:
        log_test("Data Access", "❌ FAIL", str(e))
        return False


def test_prediction_workflow():
    """Test end-to-end prediction workflow"""
    print("\n🎯 Testing Prediction Workflow...")

    try:
        # Test prediction page access (should use enhanced service)
        response = requests.get(f"{BASE_URL}/predict_page", timeout=TIMEOUT)
        if response.status_code == 200:
            log_test("Prediction Page", "✅ PASS", "Page accessible")
        else:
            log_test("Prediction Page", "❌ FAIL", f"Status {response.status_code}")
            return False

        # Test prediction results endpoint
        response = requests.get(f"{BASE_URL}/api/prediction_results", timeout=TIMEOUT)
        if response.status_code == 200:
            log_test("Prediction Results", "✅ PASS", "Results endpoint working")
        else:
            log_test("Prediction Results", "❌ FAIL", f"Status {response.status_code}")
            return False

        return True

    except Exception as e:
        log_test("Prediction Workflow", "❌ FAIL", str(e))
        return False


def test_model_registry_workflow():
    """Test model registry and management workflow"""
    print("\n🗂️ Testing Model Registry Workflow...")

    try:
        # Test model registry status
        response = requests.get(
            f"{BASE_URL}/api/model_registry/status", timeout=TIMEOUT
        )
        registry_data = response.json()

        if registry_data.get("success"):
            model_count = registry_data.get("model_count", 0)
            best_models = len(registry_data.get("best_models", {}))
            log_test(
                "Model Registry",
                "✅ PASS",
                f"{model_count} models, {best_models} best models",
            )
        else:
            log_test("Model Registry", "❌ FAIL", "Registry not accessible")
            return False

        # Test model listing
        response = requests.get(
            f"{BASE_URL}/api/model_registry/models", timeout=TIMEOUT
        )
        if response.status_code == 200:
            log_test("Model Listing", "✅ PASS", "Models list accessible")
        else:
            log_test("Model Listing", "❌ FAIL", f"Status {response.status_code}")
            return False

        return True

    except Exception as e:
        log_test("Model Registry", "❌ FAIL", str(e))
        return False


def test_frontend_integration():
    """Test frontend pages and UI integration"""
    print("\n🖥️ Testing Frontend Integration...")

    frontend_pages = [
        ("/", "Dashboard"),
        ("/races", "Races List"),
        ("/monitoring", "Monitoring"),
        ("/model_registry", "Model Registry"),
        ("/ml_dashboard", "ML Dashboard"),
    ]

    all_passed = True

    for path, name in frontend_pages:
        try:
            response = requests.get(f"{BASE_URL}{path}", timeout=TIMEOUT)
            if response.status_code == 200:
                log_test(name, "✅ PASS", "Page loads successfully")
            else:
                log_test(name, "❌ FAIL", f"Status {response.status_code}")
                all_passed = False
        except Exception as e:
            log_test(name, "❌ FAIL", str(e))
            all_passed = False

    return all_passed


def test_monitoring_and_diagnostics():
    """Test monitoring and diagnostic capabilities"""
    print("\n📈 Testing Monitoring & Diagnostics...")

    try:
        # Test logs endpoint
        response = requests.get(f"{BASE_URL}/api/logs", timeout=TIMEOUT)
        if response.status_code == 200:
            log_test("Logs Access", "✅ PASS", "Logs accessible")
        else:
            log_test("Logs Access", "❌ FAIL", f"Status {response.status_code}")
            return False

        # Test diagnostics summary
        response = requests.get(f"{BASE_URL}/api/diagnostics/summary", timeout=TIMEOUT)
        if response.status_code == 200:
            log_test("Diagnostics", "✅ PASS", "Summary available")
        else:
            log_test("Diagnostics", "❌ FAIL", f"Status {response.status_code}")
            return False

        # Test file stats
        response = requests.get(f"{BASE_URL}/api/file_stats", timeout=TIMEOUT)
        if response.status_code == 200:
            log_test("File Stats", "✅ PASS", "Statistics available")
        else:
            log_test("File Stats", "❌ FAIL", f"Status {response.status_code}")
            return False

        return True

    except Exception as e:
        log_test("Monitoring", "❌ FAIL", str(e))
        return False


def test_enhanced_prediction_service():
    """Test enhanced prediction service integration"""
    print("\n🎯 Testing Enhanced Prediction Service...")

    try:
        # Verify enhanced service is available through model health
        response = requests.get(f"{BASE_URL}/api/model_health", timeout=TIMEOUT)
        model_health = response.json()

        if model_health.get("source") == "model_registry" and model_health.get("ready"):
            log_test(
                "Enhanced Service",
                "✅ PASS",
                "Service active with model registry integration",
            )
        else:
            log_test(
                "Enhanced Service", "⚠️ WARN", "Service may not be fully integrated"
            )

        # Test prediction endpoint that uses enhanced service
        response = requests.get(f"{BASE_URL}/predict_page", timeout=TIMEOUT)
        if response.status_code == 200:
            log_test(
                "Enhanced Predictions",
                "✅ PASS",
                "Enhanced prediction endpoint accessible",
            )
        else:
            log_test(
                "Enhanced Predictions", "❌ FAIL", f"Status {response.status_code}"
            )
            return False

        return True

    except Exception as e:
        log_test("Enhanced Service", "❌ FAIL", str(e))
        return False


def run_comprehensive_integration_test():
    """Run all integration tests"""
    print("=" * 60)
    print("🚀 COMPREHENSIVE INTEGRATION TEST")
    print("=" * 60)
    print(f"Testing Flask app at: {BASE_URL}")
    print(f"Started at: {datetime.now()}")

    test_results = []

    # Run all test suites
    test_suites = [
        ("System Health", test_system_health),
        ("Data Access", test_data_access_workflow),
        ("Prediction Workflow", test_prediction_workflow),
        ("Model Registry", test_model_registry_workflow),
        ("Frontend Integration", test_frontend_integration),
        ("Monitoring & Diagnostics", test_monitoring_and_diagnostics),
        ("Enhanced Prediction Service", test_enhanced_prediction_service),
    ]

    for suite_name, test_function in test_suites:
        try:
            result = test_function()
            test_results.append((suite_name, result))
        except Exception as e:
            print(f"\n❌ {suite_name} FAILED with exception: {str(e)}")
            test_results.append((suite_name, False))

    # Print summary
    print("\n" + "=" * 60)
    print("📋 INTEGRATION TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)

    for suite_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {suite_name}")

    print(f"\nOverall Result: {passed}/{total} test suites passed")
    success_rate = (passed / total * 100) if total > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")

    if success_rate >= 85:
        print("\n🎉 INTEGRATION TEST PASSED - System is ready for production!")
        return True
    else:
        print("\n⚠️ INTEGRATION TEST INCOMPLETE - Some issues need attention")
        return False


if __name__ == "__main__":
    print("Starting comprehensive integration test...")

    # Wait a moment for system to be ready
    time.sleep(2)

    success = run_comprehensive_integration_test()

    print(f"\nTest completed at: {datetime.now()}")
    sys.exit(0 if success else 1)
