#!/usr/bin/env python3
"""
Quick Flask App Startup Test
============================

Test if the Flask application can start successfully without actually running the server.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

def test_app_startup():
    """Test if the Flask app can be imported and initialized"""
    print("🧪 Testing Flask App Startup...")
    
    try:
        # Import the app
        print("  1. Importing app module...")
        import app
        print("  ✅ App module imported successfully")
        
        # Check if Flask app is created
        if hasattr(app, 'app'):
            print("  2. Checking Flask app instance...")
            flask_app = app.app
            print(f"  ✅ Flask app instance found: {type(flask_app)}")
            
            # Test that we can create a test client
            print("  3. Creating test client...")
            with flask_app.test_client() as client:
                print("  ✅ Test client created successfully")
                
                # Test a simple route
                print("  4. Testing basic route...")
                response = client.get('/')
                print(f"  ✅ Basic route responds with status: {response.status_code}")
                
                # Test API route
                print("  5. Testing API route...")
                response = client.get('/api/race_files_status')
                print(f"  ✅ API route responds with status: {response.status_code}")
                
            print("\n🎉 Flask app startup test PASSED!")
            return True
        else:
            print("  ❌ No Flask app instance found in app module")
            return False
            
    except Exception as e:
        print(f"  ❌ Flask app startup test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_app_startup()
    sys.exit(0 if success else 1)
