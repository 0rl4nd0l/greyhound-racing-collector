#!/usr/bin/env python3
"""
Flask API Endpoint Testing Script
===============================

This script performs comprehensive testing of Flask API endpoints to validate:
1. Endpoint availability and response formats
2. Error handling and status codes
3. Database connectivity through endpoints
4. Prediction system integration
5. File management endpoints
6. Data quality validation endpoints

Author: AI Assistant
Date: January 2025
"""

import sys
import os
import json
import time
import requests
from datetime import datetime
from pathlib import Path

class FlaskAPITester:
    """Comprehensive Flask API endpoint testing class"""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.test_results = []
        self.passed_tests = 0
        self.failed_tests = 0
        
        print("🧪 FLASK API ENDPOINT TESTING")
        print("=" * 50)
        print(f"Base URL: {self.base_url}")
        print()
    
    def test_endpoint(self, endpoint_name, method, path, data=None, expected_status=200):
        """Test a single API endpoint"""
        print(f"🔍 Testing {method} {path}")
        
        try:
            url = f"{self.base_url}{path}"
            start_time = time.time()
            
            if method.upper() == 'GET':
                response = requests.get(url, timeout=10)
            elif method.upper() == 'POST':
                response = requests.post(url, json=data, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Check status code
            status_ok = response.status_code == expected_status
            
            # Try to parse JSON
            try:
                json_data = response.json()
                json_valid = True
            except:
                json_data = None
                json_valid = False
            
            # Record result
            result = {
                'endpoint': endpoint_name,
                'method': method,
                'path': path,
                'status_code': response.status_code,
                'expected_status': expected_status,
                'status_ok': status_ok,
                'json_valid': json_valid,
                'response_time': response_time,
                'response_data': json_data,
                'success': status_ok and json_valid
            }
            
            self.test_results.append(result)
            
            if result['success']:
                self.passed_tests += 1
                print(f"  ✅ PASS ({response_time:.2f}s) - Status: {response.status_code}")
                if json_data and isinstance(json_data, dict):
                    if 'message' in json_data:
                        print(f"     Message: {json_data['message']}")
                    elif 'count' in json_data:
                        print(f"     Count: {json_data['count']}")
            else:
                self.failed_tests += 1
                print(f"  ❌ FAIL ({response_time:.2f}s) - Status: {response.status_code}")
                if not json_valid:
                    print(f"     Invalid JSON response")
                print(f"     Response: {response.text[:200]}...")
                
        except requests.exceptions.RequestException as e:
            self.failed_tests += 1
            result = {
                'endpoint': endpoint_name,
                'method': method,
                'path': path,
                'success': False,
                'error': str(e),
                'response_time': 0
            }
            self.test_results.append(result)
            print(f"  ❌ ERROR - {str(e)}")
        
        print()
    
    def test_core_endpoints(self):
        """Test core API endpoints"""
        print("🏠 Testing Core Endpoints")
        print("-" * 30)
        
        # Basic status endpoints
        self.test_endpoint("Stats API", "GET", "/api/stats")
        self.test_endpoint("Recent Races API", "GET", "/api/recent_races")
        self.test_endpoint("Processing Status", "GET", "/api/processing_status")
        
    def test_dog_search_endpoints(self):
        """Test dog search and data endpoints"""
        print("🐕 Testing Dog Search Endpoints")
        print("-" * 35)
        
        # Dog search endpoints
        self.test_endpoint("Dog Search", "GET", "/api/dogs/search")
        self.test_endpoint("Dog Search with Query", "GET", "/api/dogs/search?query=test")
        
        # These might not exist but we'll test them
        self.test_endpoint("Top Performers", "GET", "/api/dogs/top_performers")
    
    def test_prediction_endpoints(self):
        """Test prediction-related endpoints"""
        print("🎯 Testing Prediction Endpoints")
        print("-" * 35)
        
        # Prediction endpoints - these require POST data so we expect some to fail gracefully
        self.test_endpoint("Basic Predict", "POST", "/predict", 
                         data={"race_id": "test_race"}, expected_status=400)
        
        self.test_endpoint("API Predict", "POST", "/api/predict", 
                         data={}, expected_status=200)
        
        self.test_endpoint("Race Files Status", "GET", "/api/race_files_status")
    
    def test_file_management_endpoints(self):
        """Test file management endpoints"""
        print("📁 Testing File Management Endpoints")
        print("-" * 40)
        
        # File processing endpoints
        self.test_endpoint("Process Files", "POST", "/api/process_files", expected_status=200)
        self.test_endpoint("Fetch CSV", "POST", "/api/fetch_csv", expected_status=200)
        self.test_endpoint("Process Data", "POST", "/api/process_data", expected_status=200)
        
        # Upcoming races
        self.test_endpoint("Upcoming Races", "GET", "/api/upcoming_races")
    
    def test_training_endpoints(self):
        """Test ML training endpoints"""
        print("🧠 Testing ML Training Endpoints")
        print("-" * 35)
        
        # Training status
        self.test_endpoint("Training Status", "GET", "/api/training_status")
        
        # Model registry
        self.test_endpoint("Model Registry Models", "GET", "/api/model_registry/models")
        self.test_endpoint("Model Registry Performance", "GET", "/api/model_registry/performance")
    
    def test_data_quality_endpoints(self):
        """Test data quality and monitoring endpoints"""
        print("📊 Testing Data Quality Endpoints")
        print("-" * 40)
        
        # Data quality checks
        self.test_endpoint("Check Data Quality", "GET", "/api/check_data_quality")
        self.test_endpoint("Check Performance Drift", "GET", "/api/check_performance_drift")
        
        # Logs
        self.test_endpoint("API Logs", "GET", "/api/logs")
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("⚠️ Testing Edge Cases")
        print("-" * 25)
        
        # Non-existent endpoints
        self.test_endpoint("Non-existent Endpoint", "GET", "/api/nonexistent", expected_status=404)
        
        # Invalid POST data
        self.test_endpoint("Invalid Predict Data", "POST", "/predict", 
                         data={"invalid": "data"}, expected_status=400)
    
    def run_all_tests(self):
        """Run all API endpoint tests"""
        print(f"Starting Flask API testing at {datetime.now()}")
        print()
        
        # Test different categories
        self.test_core_endpoints()
        self.test_dog_search_endpoints()
        self.test_prediction_endpoints()
        self.test_file_management_endpoints()
        self.test_training_endpoints()
        self.test_data_quality_endpoints()
        self.test_edge_cases()
        
        # Print summary
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        total_tests = len(self.test_results)
        success_rate = (self.passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print("=" * 60)
        print("🏁 FLASK API TESTING SUMMARY")
        print("=" * 60)
        
        print(f"📊 Test Results:")
        print(f"   Total Tests: {total_tests}")
        print(f"   ✅ Passed: {self.passed_tests}")
        print(f"   ❌ Failed: {self.failed_tests}")
        print(f"   📈 Success Rate: {success_rate:.1f}%")
        
        print(f"\n📋 Detailed Results:")
        for result in self.test_results:
            status_emoji = "✅" if result.get('success', False) else "❌"
            endpoint_name = result.get('endpoint', 'Unknown')
            method = result.get('method', 'GET')
            path = result.get('path', '')
            response_time = result.get('response_time', 0)
            
            print(f"   {status_emoji} {endpoint_name:<25} {method} {path:<30} ({response_time:.2f}s)")
            
            if not result.get('success', False) and 'error' in result:
                print(f"      Error: {result['error']}")
        
        # Overall assessment
        print(f"\n🎯 Overall Assessment:")
        if success_rate >= 80:
            print("   🟢 EXCELLENT - Flask API is highly functional")
        elif success_rate >= 60:
            print("   🟡 GOOD - Flask API is mostly functional with some issues")
        elif success_rate >= 40:
            print("   🟠 FAIR - Flask API has significant issues but core functionality works")
        else:
            print("   🔴 POOR - Flask API has major functionality problems")
        
        print(f"\n💡 Recommendations:")
        failed_endpoints = [r for r in self.test_results if not r.get('success', False)]
        if failed_endpoints:
            print("   • Review and fix failed endpoints")
            print("   • Check database connectivity for data endpoints")
            print("   • Verify prediction system integration")
            print("   • Ensure proper error handling for edge cases")
        else:
            print("   • All endpoints are functioning correctly!")
            print("   • Consider adding more comprehensive endpoint testing")
            print("   • Monitor endpoint performance over time")
    
    def save_results(self, filename="flask_api_test_results.json"):
        """Save test results to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'total_tests': len(self.test_results),
                    'passed_tests': self.passed_tests,
                    'failed_tests': self.failed_tests,
                    'success_rate': (self.passed_tests / len(self.test_results) * 100) if self.test_results else 0,
                    'detailed_results': self.test_results
                }, f, indent=2)
            print(f"\n💾 Test results saved to {filename}")
        except Exception as e:
            print(f"\n❌ Failed to save results: {e}")

def check_flask_app_running(base_url="http://localhost:5000"):
    """Check if Flask app is running"""
    try:
        response = requests.get(f"{base_url}/api/stats", timeout=5)
        return True
    except:
        return False

def main():
    """Main testing function"""
    print("🚀 Flask API Endpoint Testing System")
    print("=" * 50)
    
    # Check if app is running
    base_url = "http://localhost:5002"
    if not check_flask_app_running(base_url):
        print(f"❌ Flask app is not running at {base_url}")
        print("\n💡 To start the Flask app, run:")
        print("   python app.py")
        print("\nAlternatively, try different ports:")
        for port in [5001, 5002, 5003]:
            test_url = f"http://localhost:{port}"
            if check_flask_app_running(test_url):
                print(f"✅ Flask app found running at {test_url}")
                base_url = test_url
                break
        else:
            print("❌ No Flask app found on common ports")
            return
    
    print(f"✅ Flask app is running at {base_url}")
    print()
    
    # Run tests
    tester = FlaskAPITester(base_url)
    tester.run_all_tests()
    
    # Save results
    tester.save_results()

if __name__ == "__main__":
    main()
