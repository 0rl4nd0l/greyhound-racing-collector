#!/usr/bin/env python3
"""
Key Consistency Test Runner
==========================

Dedicated test runner for key consistency regression tests.
This script ensures that all prediction layers handle keys consistently
and provides detailed reporting for CI integration.
"""

import sys
import os
import subprocess
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_key_consistency_tests():
    """Run key consistency tests with detailed reporting"""
    print("🔧 Key Consistency Test Runner - Step 6 Implementation")
    print("=" * 60)
    
    # Test configuration
    test_file = "tests/test_key_consistency.py"
    results = {
        "timestamp": datetime.now().isoformat(),
        "test_results": {},
        "summary": {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "key_errors_detected": 0
        }
    }
    
    # Define test categories
    test_categories = [
        {
            "name": "Basic CSV Structure Tests",
            "tests": [
                "test_test_race_csv_exists",
                "test_csv_structure_and_key_consistency"
            ]
        },
        {
            "name": "Parametrized Layer Tests",
            "tests": [
                "test_prediction_layers_key_consistency",
                "test_loaders_accept_constant_keys"
            ]
        },
        {
            "name": "Pipeline Integration Tests", 
            "tests": [
                "test_prediction_pipeline_no_key_errors",
                "test_weather_enhanced_predictor_key_handling"
            ]
        },
        {
            "name": "Error Handling & Fallback Tests",
            "tests": [
                "test_error_handling_and_fallback_logic"
            ]
        },
        {
            "name": "Full Integration Tests",
            "tests": [
                "test_integration_all_layers_consistent_keys"
            ]
        }
    ]
    
    # Run each category of tests
    total_passed = 0
    total_failed = 0
    
    for category in test_categories:
        print(f"\n📋 Running {category['name']}...")
        print("-" * 40)
        
        category_results = {
            "passed": 0,
            "failed": 0,
            "errors": []
        }
        
        for test_name in category['tests']:
            test_cmd = [
                "python", "-m", "pytest", 
                f"{test_file}::{test_name}",
                "-v", "--tb=short", "--no-header"
            ]
            
            try:
                result = subprocess.run(
                    test_cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=300  # 5 minute timeout per test
                )
                
                if result.returncode == 0:
                    print(f"  ✅ {test_name}")
                    category_results["passed"] += 1
                    total_passed += 1
                else:
                    print(f"  ❌ {test_name}")
                    category_results["failed"] += 1
                    total_failed += 1
                    
                    # Check for KeyError in output
                    if "KeyError" in result.stdout or "KeyError" in result.stderr:
                        results["summary"]["key_errors_detected"] += 1
                        category_results["errors"].append({
                            "test": test_name,
                            "type": "KeyError",
                            "output": result.stderr[-500:]  # Last 500 chars
                        })
                        print(f"    🚨 KeyError detected in {test_name}!")
                    else:
                        category_results["errors"].append({
                            "test": test_name,
                            "type": "Other",
                            "output": result.stderr[-500:]
                        })
                
            except subprocess.TimeoutExpired:
                print(f"  ⏰ {test_name} (TIMEOUT)")
                category_results["failed"] += 1
                total_failed += 1
                category_results["errors"].append({
                    "test": test_name,
                    "type": "Timeout",
                    "output": "Test exceeded 5 minute timeout"
                })
        
        results["test_results"][category["name"]] = category_results
        print(f"Category Summary: {category_results['passed']} passed, {category_results['failed']} failed")
        
    # Overall summary
    results["summary"]["total_tests"] = total_passed + total_failed
    results["summary"]["passed"] = total_passed
    results["summary"]["failed"] = total_failed
    
    print("\n" + "=" * 60)
    print("🏁 Key Consistency Test Summary")
    print("=" * 60)
    print(f"Total Tests: {results['summary']['total_tests']}")
    print(f"Passed: ✅ {results['summary']['passed']}")
    print(f"Failed: ❌ {results['summary']['failed']}")
    print(f"KeyErrors Detected: 🚨 {results['summary']['key_errors_detected']}")
    
    # Save results for CI
    results_file = Path(__file__).parent / "key_consistency_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Exit with appropriate code
    if results['summary']['failed'] == 0:
        print("\n🎉 All key consistency tests passed!")
        if results['summary']['key_errors_detected'] == 0:
            print("✅ No KeyErrors detected - prediction layers are handling keys consistently")
        return 0
    else:
        print(f"\n❌ {results['summary']['failed']} tests failed")
        if results['summary']['key_errors_detected'] > 0:
            print(f"🚨 CRITICAL: {results['summary']['key_errors_detected']} KeyErrors detected!")
            print("This indicates regression in key handling that must be fixed before merge.")
        return 1

if __name__ == "__main__":
    sys.exit(run_key_consistency_tests())
