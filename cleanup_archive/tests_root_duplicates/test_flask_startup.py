#!/usr/bin/env python3
"""
Test Flask app startup to debug any issues
"""

import sys
import os
import traceback

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test critical imports"""
    try:
        print("Testing Flask import...")
        from flask import Flask
        print("✅ Flask imported successfully")
        
        print("Testing logger import...")
        from logger import logger
        print("✅ Logger imported successfully")
        
        print("Testing database connection...")
        import sqlite3
        conn = sqlite3.connect("greyhound_racing_data.db")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
        result = cursor.fetchone()
        conn.close()
        print("✅ Database connection successful")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_flask_app():
    """Test basic Flask app creation"""
    try:
        print("Creating Flask app...")
        from flask import Flask
        from flask_cors import CORS
        
        app = Flask(__name__)
        app.secret_key = "test_key"
        CORS(app)
        
        @app.route('/health')
        def health():
            return {"status": "ok"}
        
        print("✅ Flask app created successfully")
        return app
    except Exception as e:
        print(f"❌ Flask app creation failed: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🚀 Testing Flask app startup components...")
    
    # Get PORT from environment or default to 5002
    import os
    PORT = int(os.environ.get('PORT', 5002))
    
    if test_imports():
        app = test_flask_app()
        if app:
            print(f"🎉 All tests passed! Starting server on port {PORT}...")
            try:
                app.run(host="0.0.0.0", port=PORT, use_reloader=False)
            except Exception as e:
                print(f"❌ Server startup failed: {e}")
                traceback.print_exc()
        else:
            print("❌ Flask app creation failed")
    else:
        print("❌ Import tests failed")
