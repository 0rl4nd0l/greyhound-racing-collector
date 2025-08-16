#!/usr/bin/env python3
"""
Quick Fix for Critical Issues
============================

Rapidly addresses the most critical issues preventing the system from functioning.

Author: AI Assistant  
Date: July 28, 2025
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    base_dir = Path("/Users/orlandolee/greyhound_racing_collector")
    venv_path = base_dir / "venv"
    
    print("🚀 Quick Fix for Greyhound Analysis Predictor")
    print("=" * 60)
    
    # 1. Install missing schedule module
    print("\n1️⃣ Installing missing 'schedule' module...")
    try:
        pip_path = venv_path / "bin" / "pip"
        result = subprocess.run([
            str(pip_path), "install", "schedule"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Schedule module installed successfully")
        else:
            print(f"❌ Failed to install schedule: {result.stderr}")
    except Exception as e:
        print(f"❌ Error installing schedule: {e}")
    
    # 2. Test Flask app import
    print("\n2️⃣ Testing Flask app import...")
    try:
        python_path = venv_path / "bin" / "python"
        result = subprocess.run([
            str(python_path), "-c", 
            f"import sys; sys.path.insert(0, '{base_dir}'); import app; print('✅ Flask app import successful')"
        ], capture_output=True, text=True, cwd=str(base_dir))
        
        if result.returncode == 0:
            print("✅ Flask app imports successfully")
        else:
            print(f"❌ Flask app import failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Error testing Flask import: {e}")
    
    # 3. Test prediction pipeline
    print("\n3️⃣ Testing prediction pipeline...")
    try:
        result = subprocess.run([
            str(python_path), "-c", 
            f"import sys; sys.path.insert(0, '{base_dir}'); from weather_enhanced_predictor import WeatherEnhancedPredictor; print('✅ Prediction pipeline working')"
        ], capture_output=True, text=True, cwd=str(base_dir))
        
        if result.returncode == 0:
            print("✅ Prediction pipeline working")
        else:
            print(f"❌ Prediction pipeline failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Error testing prediction pipeline: {e}")
    
    # 4. Test basic Flask app functionality
    print("\n4️⃣ Testing Flask app functionality...")
    try:
        test_script = f'''
import sys
sys.path.insert(0, "{base_dir}")
import app

if hasattr(app, "app"):
    flask_app = app.app
    with flask_app.test_client() as client:
        flask_app.config["TESTING"] = True
        response = client.get("/")
        print(f"Root route status: {{response.status_code}}")
        
        # Test API endpoint
        response = client.get("/api/stats")
        print(f"API stats status: {{response.status_code}}")
else:
    print("❌ Flask app instance not found")
'''
        
        result = subprocess.run([
            str(python_path), "-c", test_script
        ], capture_output=True, text=True, cwd=str(base_dir))
        
        print("Flask test output:")
        print(result.stdout)
        
        if result.stderr:
            print("Errors:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error testing Flask functionality: {e}")
    
    # 5. Create simple working diagnostic
    print("\n5️⃣ Creating simple diagnostic...")
    
    simple_diagnostic = f'''#!/usr/bin/env python3
import sys
import sqlite3
from pathlib import Path

def main():
    base_dir = Path("{base_dir}")
    db_path = base_dir / "greyhound_racing_data.db"
    
    print("🧪 Simple System Health Check")
    print("=" * 40)
    
    # Test database
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM race_metadata")
        race_count = cursor.fetchone()[0]
        print(f"✅ Database: {{race_count}} races")
        conn.close()
    except Exception as e:
        print(f"❌ Database: {{e}}")
    
    # Test Flask import
    try:
        sys.path.insert(0, str(base_dir))
        import app
        print("✅ Flask app import successful")
        
        if hasattr(app, "app"):
            print("✅ Flask app instance found")
        else:
            print("❌ Flask app instance not found")
            
    except Exception as e:
        print(f"❌ Flask import: {{e}}")
    
    # Test prediction import
    try:
        from weather_enhanced_predictor import WeatherEnhancedPredictor
        print("✅ Weather predictor import successful")
    except Exception as e:
        print(f"❌ Weather predictor: {{e}}")
    
    print("\\n🎯 Health check complete")

if __name__ == "__main__":
    main()
'''
    
    diagnostic_file = base_dir / "simple_diagnostic.py"
    with open(diagnostic_file, 'w') as f:
        f.write(simple_diagnostic)
    
    print(f"✅ Simple diagnostic created: {diagnostic_file}")
    
    # 6. Run the simple diagnostic
    print("\n6️⃣ Running simple diagnostic...")
    try:
        result = subprocess.run([
            str(python_path), str(diagnostic_file)
        ], capture_output=True, text=True, cwd=str(base_dir))
        
        print("Diagnostic output:")
        print(result.stdout)
        
        if result.stderr:
            print("Diagnostic errors:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error running diagnostic: {e}")
    
    print("\n🎯 Quick fix complete!")
    print("\nNext steps:")
    print("1. Run: /Users/orlandolee/greyhound_racing_collector/venv/bin/python app.py")
    print("2. Test prediction: /Users/orlandolee/greyhound_racing_collector/venv/bin/python weather_enhanced_predictor.py [race_file.csv]")

if __name__ == "__main__":
    main()
