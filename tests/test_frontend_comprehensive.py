#!/usr/bin/env python3
"""
Comprehensive Frontend Testing Script for Greyhound Racing Dashboard
Tests all major frontend functionality and API endpoints
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Any

class FrontendTester:
    def __init__(self, base_url="http://localhost:5002"):
        self.base_url = base_url
        self.results = []
        
    def log_result(self, test_name: str, success: bool, details: str = "", response_data: Any = None):
        """Log test result"""
        result = {
            "test_name": test_name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat(),
            "response_data": response_data
        }
        self.results.append(result)
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} | {test_name}: {details}")
        
    def test_health_endpoint(self):
        """Test the health API endpoint"""
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.log_result(
                    "Health Endpoint", 
                    True, 
                    f"Status: {data.get('status', 'unknown')}, Components: {len(data.get('components', {}))}", 
                    data
                )
            else:
                self.log_result("Health Endpoint", False, f"HTTP {response.status_code}")
        except Exception as e:
            self.log_result("Health Endpoint", False, str(e))
    
    def test_page_accessibility(self):
        """Test accessibility of main pages"""
        pages = [
            ("/", "Home Page"),
            ("/predict", "Prediction Page"),
            ("/monitoring", "Monitoring Page"),
            ("/upload", "Upload Page"),
        ]
        
        for url, name in pages:
            try:
                response = requests.get(f"{self.base_url}{url}", timeout=10)
                if response.status_code == 200:
                    has_title = "<title>" in response.text
                    has_content = len(response.text) > 1000
                    self.log_result(
                        f"Page: {name}", 
                        True, 
                        f"Accessible, has_title={has_title}, has_content={has_content}"
                    )
                else:
                    self.log_result(f"Page: {name}", False, f"HTTP {response.status_code}")
            except Exception as e:
                self.log_result(f"Page: {name}", False, str(e))
    
    def test_single_prediction(self):
        """Test single race prediction"""
        try:
            data = {
                'race_files': 'Race 1 - AP_K - 2025-08-04.csv',
                'action': 'single'
            }
            response = requests.post(f"{self.base_url}/predict", data=data, timeout=30)
            
            if response.status_code == 200:
                # Extract JSON from HTML response
                html_content = response.text
                if 'prediction-json' in html_content:
                    # Try to extract the JSON
                    import re
                    json_match = re.search(r'<pre id="prediction-json">(.*?)</pre>', html_content, re.DOTALL)
                    if json_match:
                        try:
                            prediction_data = json.loads(json_match.group(1))
                            num_predictions = len(prediction_data.get('predictions', []))
                            winner = prediction_data['predictions'][0]['dog_name'] if num_predictions > 0 else 'None'
                            
                            self.log_result(
                                "Single Prediction", 
                                True, 
                                f"Predicted {num_predictions} dogs, Winner: {winner}",
                                prediction_data
                            )
                        except json.JSONDecodeError as e:
                            self.log_result("Single Prediction", False, f"JSON decode error: {e}")
                    else:
                        self.log_result("Single Prediction", False, "Could not extract JSON from HTML")
                else:
                    self.log_result("Single Prediction", False, "No prediction JSON found in response")
            else:
                self.log_result("Single Prediction", False, f"HTTP {response.status_code}")
        except Exception as e:
            self.log_result("Single Prediction", False, str(e))
    
    def test_multiple_predictions(self):
        """Test predictions for multiple races"""
        races = [
            'Race 1 - AP_K - 2025-08-04.csv',
            'Race 2 - AP_K - 2025-08-04.csv'
        ]
        
        successful_predictions = 0
        
        for race in races:
            try:
                data = {
                    'race_files': race,
                    'action': 'single'
                }
                response = requests.post(f"{self.base_url}/predict", data=data, timeout=30)
                
                if response.status_code == 200 and 'prediction-json' in response.text:
                    successful_predictions += 1
                    
            except Exception as e:
                pass  # Continue with other races
        
        success_rate = successful_predictions / len(races)
        self.log_result(
            "Multiple Predictions", 
            success_rate >= 0.5,  # Consider success if at least 50% work
            f"{successful_predictions}/{len(races)} races predicted successfully ({success_rate:.1%})"
        )
    
    def test_batch_prediction_api(self):
        """Test batch prediction API endpoint"""
        try:
            payload = {
                'race_files': ['Race 1 - AP_K - 2025-08-04.csv'],
                'batch_size': 1,
                'max_workers': 1
            }
            
            response = requests.post(
                f"{self.base_url}/predict_batch", 
                json=payload, 
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    job_id = data.get('job_id', 'unknown')
                    self.log_result(
                        "Batch Prediction API", 
                        True, 
                        f"Started batch job: {job_id}",
                        data
                    )
                else:
                    self.log_result("Batch Prediction API", False, data.get('message', 'Unknown error'))
            else:
                self.log_result("Batch Prediction API", False, f"HTTP {response.status_code}")
        except Exception as e:
            self.log_result("Batch Prediction API", False, str(e))
    
    def test_api_endpoints(self):
        """Test various API endpoints"""
        api_endpoints = [
            ("/api/health", "Health API"),
        ]
        
        for endpoint, name in api_endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                if response.status_code == 200:
                    try:
                        data = response.json()
                        self.log_result(f"API: {name}", True, "Valid JSON response", data)
                    except json.JSONDecodeError:
                        self.log_result(f"API: {name}", False, "Invalid JSON response")
                else:
                    self.log_result(f"API: {name}", False, f"HTTP {response.status_code}")
            except Exception as e:
                self.log_result(f"API: {name}", False, str(e))
    
    def run_all_tests(self):
        """Run all frontend tests"""
        print("🚀 Starting Comprehensive Frontend Testing")
        print("=" * 50)
        
        # Test basic connectivity
        self.test_health_endpoint()
        
        # Test page accessibility
        self.test_page_accessibility()
        
        # Test prediction functionality
        self.test_single_prediction()
        self.test_multiple_predictions()
        
        # Test API endpoints
        self.test_batch_prediction_api()
        self.test_api_endpoints()
        
        # Generate summary
        self.generate_summary()
    
    def generate_summary(self):
        """Generate test summary"""
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {passed_tests/total_tests:.1%}")
        
        if failed_tests > 0:
            print(f"\n🔍 FAILED TESTS:")
            for result in self.results:
                if not result['success']:
                    print(f"  - {result['test_name']}: {result['details']}")
        
        print(f"\n📋 WORKING FUNCTIONALITY:")
        for result in self.results:
            if result['success']:
                print(f"  ✅ {result['test_name']}")
        
        # Save detailed results
        with open('frontend_test_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n📁 Detailed results saved to: frontend_test_results.json")

def main():
    tester = FrontendTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
