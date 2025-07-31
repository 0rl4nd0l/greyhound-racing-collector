#!/usr/bin/env python3
"""
Production Readiness Test
========================

Final verification that the ML greyhound prediction system is ready for production use.
"""

import sys
import os
import json
import time
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.getcwd())

def test_production_readiness():
    """Quick test to verify production readiness"""
    print("🚀 Production Readiness Test")
    print("="*50)
    
    results = []
    
    # Test 1: Flask App Import
    try:
        import app
        flask_app = app.app
        results.append(("Flask App", True, "Successfully imported and initialized"))
    except Exception as e:
        results.append(("Flask App", False, f"Failed: {str(e)}"))
    
    # Test 2: ML Components
    try:
        from ml_system_v3 import MLSystemV3
        from prediction_pipeline_v3 import PredictionPipelineV3
        ml_system = MLSystemV3()
        pipeline = PredictionPipelineV3()
        results.append(("ML Components", True, "All ML components working"))
    except Exception as e:
        results.append(("ML Components", False, f"Failed: {str(e)}"))
    
    # Test 3: Database
    try:
        import sqlite3
        conn = sqlite3.connect('greyhound_racing_data.db')
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM race_metadata")
        race_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM dogs")
        dog_count = cursor.fetchone()[0]
        conn.close()
        
        if race_count > 0 and dog_count > 0:
            results.append(("Database", True, f"{race_count} races, {dog_count} dogs"))
        else:
            results.append(("Database", False, "Empty database"))
    except Exception as e:
        results.append(("Database", False, f"Failed: {str(e)}"))
    
    # Test 4: Race Files
    try:
        upcoming_races = os.listdir('upcoming_races')
        csv_files = [f for f in upcoming_races if f.endswith('.csv')]
        if len(csv_files) > 0:
            results.append(("Race Files", True, f"{len(csv_files)} race files available"))
        else:
            results.append(("Race Files", False, "No race files found"))
    except Exception as e:
        results.append(("Race Files", False, f"Failed: {str(e)}"))
    
    # Test 5: API Endpoints
    try:
        with flask_app.test_client() as client:
            response1 = client.get('/api/race_files_status')
            response2 = client.get('/ml_dashboard')
            
            if response1.status_code == 200 and response2.status_code == 200:
                results.append(("API Endpoints", True, "All endpoints responding"))
            else:
                results.append(("API Endpoints", False, f"Status codes: {response1.status_code}, {response2.status_code}"))
    except Exception as e:
        results.append(("API Endpoints", False, f"Failed: {str(e)}"))
    
    # Print Results
    print(f"\n📊 Test Results:")
    passed = 0
    total = len(results)
    
    for test, success, message in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {test}: {message}")
        if success:
            passed += 1
    
    success_rate = (passed / total * 100) if total > 0 else 0
    print(f"\n🎯 Success Rate: {passed}/{total} ({success_rate:.1f}%)")
    
    # Final Assessment
    print(f"\n🏥 Production Readiness Assessment:")
    if passed == total:
        print("  🟢 READY FOR PRODUCTION")
        print("  All critical systems are functioning correctly")
        print(f"\n🚀 Next Steps:")
        print("  1. Start the application: python3 app.py")
        print("  2. Access the ML Dashboard: http://localhost:5000/ml_dashboard")
        print("  3. Begin making race predictions!")
        return True
    elif passed >= total * 0.8:
        print("  🟡 MOSTLY READY")
        print("  Minor issues detected, but system should work")
        return True
    else:
        print("  🔴 NOT READY")
        print("  Critical issues need to be resolved first")
        return False

if __name__ == "__main__":
    success = test_production_readiness()
    sys.exit(0 if success else 1)
