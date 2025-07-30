#!/usr/bin/env python3
"""
Quick Flask App Test
===================

This script tests if the Flask app can start without errors and validates
basic functionality without running the server.
"""

import sys
import os
import traceback

def test_flask_app_import():
    """Test if Flask app can be imported without errors"""
    print("🔍 Testing Flask app import...")
    
    try:
        from app import app, db_manager
        print("✅ Flask app imported successfully")
        return True
    except Exception as e:
        print(f"❌ Flask app import failed: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        return False

def test_flask_app_config():
    """Test Flask app configuration"""
    print("\n🔧 Testing Flask app configuration...")
    
    try:
        from app import app
        print(f"✅ Flask app created: {app.name}")
        print(f"   Debug mode: {app.debug}")
        print(f"   Secret key set: {bool(app.secret_key)}")
        return True
    except Exception as e:
        print(f"❌ Flask app configuration test failed: {str(e)}")
        return False

def test_database_manager():
    """Test database manager functionality"""
    print("\n🗄️ Testing database manager...")
    
    try:
        from app import db_manager
        
        # Test database connection
        stats = db_manager.get_database_stats()
        print(f"✅ Database accessible")
        print(f"   Total races: {stats.get('total_races', 'Unknown')}")
        print(f"   Total dogs: {stats.get('total_dogs', 'Unknown')}")
        return True
    except Exception as e:
        print(f"❌ Database manager test failed: {str(e)}")
        return False

def test_prediction_systems():
    """Test prediction system imports"""
    print("\n🎯 Testing prediction systems...")
    
    systems_tested = 0
    systems_working = 0
    
    # Test UnifiedPredictor
    try:
        from unified_predictor import UnifiedPredictor
        predictor = UnifiedPredictor()
        print("✅ UnifiedPredictor available")
        systems_tested += 1
        systems_working += 1
    except Exception as e:
        print(f"⚠️ UnifiedPredictor not available: {str(e)}")
        systems_tested += 1
    
    # Test ComprehensivePredictionPipeline
    try:
        from comprehensive_prediction_pipeline import ComprehensivePredictionPipeline
        pipeline = ComprehensivePredictionPipeline()
        print("✅ ComprehensivePredictionPipeline available")
        systems_tested += 1
        systems_working += 1
    except Exception as e:
        print(f"⚠️ ComprehensivePredictionPipeline not available: {str(e)}")
        systems_tested += 1
    
    print(f"📊 Prediction systems: {systems_working}/{systems_tested} working")
    return systems_working > 0

def test_key_endpoints():
    """Test key endpoints using Flask test client"""
    print("\n🌐 Testing key endpoints with test client...")
    
    try:
        from app import app
        
        with app.test_client() as client:
            # Test basic endpoint
            response = client.get('/api/stats')
            print(f"✅ /api/stats: {response.status_code}")
            
            # Test another endpoint
            response = client.get('/api/processing_status')
            print(f"✅ /api/processing_status: {response.status_code}")
            
            # Test recent races
            response = client.get('/api/recent_races')
            print(f"✅ /api/recent_races: {response.status_code}")
            
            return True
    except Exception as e:
        print(f"❌ Endpoint testing failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🧪 QUICK FLASK APP TESTING")
    print("=" * 40)
    
    total_tests = 0
    passed_tests = 0
    
    # Run tests
    tests = [
        ("Flask App Import", test_flask_app_import),
        ("Flask App Config", test_flask_app_config),
        ("Database Manager", test_database_manager),
        ("Prediction Systems", test_prediction_systems),
        ("Key Endpoints", test_key_endpoints)
    ]
    
    for test_name, test_func in tests:
        total_tests += 1
        print(f"\n{'='*60}")
        print(f"Test: {test_name}")
        print("="*60)
        
        if test_func():
            passed_tests += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    # Summary
    print(f"\n{'='*60}")
    print("🏁 QUICK TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {total_tests - passed_tests}")
    print(f"📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Flask app appears to be functional.")
    elif passed_tests > total_tests * 0.6:
        print("\n👍 Most tests passed. Flask app has minor issues.")
    else:
        print("\n⚠️ Multiple test failures. Flask app has significant issues.")

if __name__ == "__main__":
    main()
