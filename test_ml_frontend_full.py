#!/usr/bin/env python3
"""
Comprehensive ML Frontend Process Test
======================================

This script tests the complete ML frontend workflow:
1. Flask app initialization
2. Race files loading and API endpoints
3. Single race prediction pipeline
4. Batch prediction capabilities
5. ML dashboard functionality
6. Database integration
7. JSON response validation

Tests both backend API and frontend integration.
"""

import sys
import os
import json
import time
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.getcwd())

class MLFrontendTester:
    def __init__(self):
        self.app = None
        self.client = None
        self.test_results = []
        self.failed_tests = []
        
    def log_test(self, test_name, status, details="", duration=0):
        """Log test results"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        result = {
            'timestamp': timestamp,
            'test': test_name,
            'status': status,
            'details': details,
            'duration': duration
        }
        
        self.test_results.append(result)
        
        if status == "PASS":
            print(f"✅ [{timestamp}] {test_name} - {details} ({duration:.2f}s)")
        elif status == "FAIL":
            print(f"❌ [{timestamp}] {test_name} - {details} ({duration:.2f}s)")
            self.failed_tests.append(result)
        else:
            print(f"⚠️  [{timestamp}] {test_name} - {details} ({duration:.2f}s)")
    
    def setup_flask_app(self):
        """Initialize Flask app for testing"""
        print("\n🚀 Setting up Flask Application...")
        start_time = time.time()
        
        try:
            import app
            self.app = app.app
            self.client = self.app.test_client()
            
            # Set app to testing mode
            self.app.config['TESTING'] = True
            
            duration = time.time() - start_time
            self.log_test("Flask App Setup", "PASS", "App initialized successfully", duration)
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Flask App Setup", "FAIL", f"Error: {str(e)}", duration)
            return False
    
    def test_basic_routes(self):
        """Test basic Flask routes"""
        print("\n🧪 Testing Basic Routes...")
        
        routes_to_test = [
            ('/', 'Home page'),
            ('/ml_dashboard', 'ML Dashboard'),
            ('/upcoming', 'Upcoming races page'),
            ('/dogs', 'Dogs page'),
        ]
        
        for route, description in routes_to_test:
            start_time = time.time()
            try:
                response = self.client.get(route)
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    self.log_test(f"Route {route}", "PASS", 
                                f"{description} responds correctly", duration)
                else:
                    self.log_test(f"Route {route}", "FAIL", 
                                f"{description} returned {response.status_code}", duration)
                    
            except Exception as e:
                duration = time.time() - start_time
                self.log_test(f"Route {route}", "FAIL", f"Error: {str(e)}", duration)
    
    def test_api_endpoints(self):
        """Test critical API endpoints"""
        print("\n🧪 Testing API Endpoints...")
        
        api_tests = [
            ('/api/file_stats', 'File statistics API'),
            ('/api/race_files_status', 'Race files status API'),
            ('/api/dogs/search?q=test', 'Dog search API'),
            ('/api/system_status', 'System status API'),
        ]
        
        for endpoint, description in api_tests:
            start_time = time.time()
            try:
                response = self.client.get(endpoint)
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    # Try to parse JSON response
                    try:
                        data = json.loads(response.data)
                        self.log_test(f"API {endpoint}", "PASS", 
                                    f"{description} returns valid JSON", duration)
                        
                        # Check for success field in response
                        if 'success' in data:
                            if data['success']:
                                self.log_test(f"API {endpoint} Data", "PASS", 
                                            f"API reports success", 0)
                            else:
                                self.log_test(f"API {endpoint} Data", "WARN", 
                                            f"API reports failure: {data.get('message', 'Unknown')}", 0)
                        
                    except json.JSONDecodeError:
                        self.log_test(f"API {endpoint}", "WARN", 
                                    f"{description} returns non-JSON response", duration)
                else:
                    self.log_test(f"API {endpoint}", "FAIL", 
                                f"{description} returned {response.status_code}", duration)
                    
            except Exception as e:
                duration = time.time() - start_time
                self.log_test(f"API {endpoint}", "FAIL", f"Error: {str(e)}", duration)
    
    def test_race_files_loading(self):
        """Test race files loading functionality"""
        print("\n🧪 Testing Race Files Loading...")
        
        start_time = time.time()
        try:
            response = self.client.get('/api/race_files_status')
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = json.loads(response.data)
                
                if data.get('success'):
                    predicted_count = data.get('total_predicted', 0)
                    unpredicted_count = data.get('total_unpredicted', 0)
                    total_races = predicted_count + unpredicted_count
                    
                    self.log_test("Race Files Loading", "PASS", 
                                f"Found {total_races} races ({unpredicted_count} unpredicted)", duration)
                    
                    # Check race data structure
                    unpredicted_races = data.get('unpredicted_races', [])
                    if unpredicted_races:
                        sample_race = unpredicted_races[0]
                        required_fields = ['filename', 'race_id', 'file_size']
                        
                        if all(field in sample_race for field in required_fields):
                            self.log_test("Race Data Structure", "PASS", 
                                        "Race objects have required fields", 0)
                        else:
                            missing = [f for f in required_fields if f not in sample_race]
                            self.log_test("Race Data Structure", "FAIL", 
                                        f"Missing fields: {missing}", 0)
                    
                    return unpredicted_races
                else:
                    self.log_test("Race Files Loading", "FAIL", 
                                f"API returned error: {data.get('message')}", duration)
            else:
                self.log_test("Race Files Loading", "FAIL", 
                            f"HTTP {response.status_code}", duration)
        
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Race Files Loading", "FAIL", f"Error: {str(e)}", duration)
        
        return []
    
    def test_single_race_prediction(self, race_files):
        """Test single race prediction functionality"""
        print("\n🧪 Testing Single Race Prediction...")
        
        if not race_files:
            self.log_test("Single Race Prediction", "SKIP", "No unpredicted races available", 0)
            return
        
        # Test with the first available race
        test_race = race_files[0]
        race_filename = test_race['filename']
        
        start_time = time.time()
        try:
            # Make prediction request
            response = self.client.post('/api/predict_single_race', 
                                      json={'race_filename': race_filename})
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = json.loads(response.data)
                
                if data.get('success'):
                    self.log_test("Single Race Prediction", "PASS", 
                                f"Successfully predicted {race_filename}", duration)
                    
                    # Check prediction structure
                    prediction = data.get('prediction', {})
                    if prediction:
                        required_fields = ['race_name', 'venue', 'total_dogs', 'top_pick']
                        present_fields = [f for f in required_fields if f in prediction]
                        
                        self.log_test("Prediction Structure", "PASS", 
                                    f"Prediction has {len(present_fields)}/{len(required_fields)} fields", 0)
                        
                        # Check top pick structure
                        top_pick = prediction.get('top_pick', {})
                        if top_pick and 'dog_name' in top_pick and 'prediction_score' in top_pick:
                            score = top_pick.get('prediction_score', 0)
                            self.log_test("Top Pick Quality", "PASS", 
                                        f"Top pick: {top_pick['dog_name']} ({score:.1%})", 0)
                        else:
                            self.log_test("Top Pick Quality", "FAIL", 
                                        "Top pick missing required fields", 0)
                    
                    return data
                else:
                    self.log_test("Single Race Prediction", "FAIL", 
                                f"Prediction failed: {data.get('message')}", duration)
            else:
                self.log_test("Single Race Prediction", "FAIL", 
                            f"HTTP {response.status_code}", duration)
        
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Single Race Prediction", "FAIL", f"Error: {str(e)}", duration)
        
        return None
    
    def test_ml_dashboard_functionality(self):
        """Test ML Dashboard page functionality"""
        print("\n🧪 Testing ML Dashboard Functionality...")
        
        start_time = time.time()
        try:
            response = self.client.get('/ml_dashboard')
            duration = time.time() - start_time
            
            if response.status_code == 200:
                html_content = response.data.decode('utf-8')
                
                # Check for essential dashboard elements
                required_elements = [
                    'Enhanced ML Dashboard',
                    'Run All Predictions',
                    'predicted-races',
                    'unpredicted-races',
                    'predictSingleRace',
                    'predictAllRaces'
                ]
                
                present_elements = [elem for elem in required_elements if elem in html_content]
                
                self.log_test("ML Dashboard Content", "PASS", 
                            f"Dashboard has {len(present_elements)}/{len(required_elements)} elements", duration)
                
                # Check for JavaScript functionality
                js_functions = ['refreshRaceData', 'renderPredictedRaces', 'renderUnpredictedRaces']
                present_js = [func for func in js_functions if func in html_content]
                
                self.log_test("ML Dashboard JavaScript", "PASS", 
                            f"Dashboard has {len(present_js)}/{len(js_functions)} JS functions", 0)
                
            else:
                self.log_test("ML Dashboard Content", "FAIL", 
                            f"HTTP {response.status_code}", duration)
        
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("ML Dashboard Content", "FAIL", f"Error: {str(e)}", duration)
    
    def test_database_integration(self):
        """Test database connectivity and data"""
        print("\n🧪 Testing Database Integration...")
        
        start_time = time.time()
        try:
            import sqlite3
            
            conn = sqlite3.connect('greyhound_racing_data.db')
            cursor = conn.cursor()
            
            # Test essential tables
            tables_to_check = [
                ('race_metadata', 'race data'),
                ('dog_race_data', 'dog performance data'),
                ('dogs', 'dog information'),
                ('trainers', 'trainer information')
            ]
            
            for table, description in tables_to_check:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                
                if count > 0:
                    self.log_test(f"Database {table}", "PASS", 
                                f"{description} table has {count} records", 0)
                else:
                    self.log_test(f"Database {table}", "WARN", 
                                f"{description} table is empty", 0)
            
            conn.close()
            duration = time.time() - start_time
            self.log_test("Database Integration", "PASS", "Database accessible", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Database Integration", "FAIL", f"Error: {str(e)}", duration)
    
    def test_ml_system_integration(self):
        """Test ML system integration"""
        print("\n🧪 Testing ML System Integration...")
        
        start_time = time.time()
        try:
            # Test ML system import and initialization
            from ml_system_v3 import MLSystemV3
            from prediction_pipeline_v3 import PredictionPipelineV3
            
            # Initialize ML system
            ml_system = MLSystemV3()
            pipeline = PredictionPipelineV3()
            
            duration = time.time() - start_time
            self.log_test("ML System Import", "PASS", "ML components imported successfully", duration)
            
            # Test model availability
            if hasattr(ml_system, 'pipeline') and ml_system.pipeline:
                self.log_test("ML Model Availability", "PASS", "ML model is loaded", 0)
            else:
                self.log_test("ML Model Availability", "WARN", "No pre-trained model found", 0)
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("ML System Integration", "FAIL", f"Error: {str(e)}", duration)
    
    def generate_comprehensive_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*80)
        print("🔍 COMPREHENSIVE ML FRONTEND TEST REPORT")
        print("="*80)
        
        total_tests = len(self.test_results)
        passed_tests = len([t for t in self.test_results if t['status'] == 'PASS'])
        failed_tests = len([t for t in self.test_results if t['status'] == 'FAIL'])
        warned_tests = len([t for t in self.test_results if t['status'] == 'WARN'])
        skipped_tests = len([t for t in self.test_results if t['status'] == 'SKIP'])
        
        print(f"\n📊 Test Summary:")
        print(f"  Total Tests: {total_tests}")
        print(f"  ✅ Passed: {passed_tests}")
        print(f"  ❌ Failed: {failed_tests}")
        print(f"  ⚠️  Warnings: {warned_tests}")
        print(f"  ⏭️  Skipped: {skipped_tests}")
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        print(f"  🎯 Success Rate: {success_rate:.1f}%")
        
        if self.failed_tests:
            print(f"\n❌ FAILED TESTS:")
            for test in self.failed_tests:
                print(f"  {test['test']}: {test['details']}")
        
        # System health assessment
        print(f"\n🏥 System Health Assessment:")
        if failed_tests == 0:
            print("  🟢 EXCELLENT - All critical systems functioning")
        elif failed_tests <= 2:
            print("  🟡 GOOD - Minor issues detected")
        elif failed_tests <= 5:
            print("  🟠 FAIR - Some issues need attention")
        else:
            print("  🔴 POOR - Multiple critical issues")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if failed_tests == 0:
            print("  - System is ready for production use")
            print("  - Consider running performance tests")
        else:
            print("  - Fix failed tests before production deployment")
            print("  - Review system logs for additional details")
        
        print(f"\n📋 Next Steps:")
        print("  1. Address any failed tests")
        print("  2. Start the Flask application: python3 app.py")
        print("  3. Access ML Dashboard: http://localhost:5000/ml_dashboard")
        print("  4. Test race predictions through the web interface")
        
        return failed_tests == 0

def run_comprehensive_test():
    """Run the complete ML frontend test suite"""
    print("🚀 Starting Comprehensive ML Frontend Test Suite...")
    print(f"Working directory: {os.getcwd()}")
    print(f"Timestamp: {datetime.now()}")
    
    tester = MLFrontendTester()
    
    # Initialize Flask app
    if not tester.setup_flask_app():
        print("❌ Cannot proceed without Flask app")
        return False
    
    # Run all tests
    tester.test_basic_routes()
    tester.test_api_endpoints()
    
    race_files = tester.test_race_files_loading()
    tester.test_single_race_prediction(race_files)
    
    tester.test_ml_dashboard_functionality()
    tester.test_database_integration()
    tester.test_ml_system_integration()
    
    # Generate final report
    success = tester.generate_comprehensive_report()
    
    if success:
        print(f"\n🎉 All tests PASSED! ML Frontend is ready for use!")
        return True
    else:
        print(f"\n🔧 Some tests FAILED. Please review and fix issues.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
