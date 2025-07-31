#!/usr/bin/env python3
"""
Simple Flask App Startup Test
============================

Test basic Flask app functionality to ensure it can start and respond to requests.
"""

import os
import sys
import requests
import subprocess
import time
import threading
from datetime import datetime

def test_flask_startup():
    """Test that Flask app can start and respond to basic requests"""
    print("🧪 Testing Flask App Startup...")
    
    # Start Flask app in background
    flask_process = subprocess.Popen(
        [sys.executable, 'app.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Give the app time to start
    print("⏳ Waiting for Flask app to start...")
    time.sleep(10)
    
    try:
        # Test basic endpoints
        base_url = "http://localhost:5002"
        
        # Test 1: API stats endpoint
        print("📊 Testing /api/stats endpoint...")
        try:
            response = requests.get(f"{base_url}/api/stats", timeout=10)
            if response.status_code == 200:
                print("✅ /api/stats - OK")
                data = response.json()
                print(f"   Database stats: {data.get('database', {})}")
            else:
                print(f"❌ /api/stats - Failed ({response.status_code})")
        except Exception as e:
            print(f"❌ /api/stats - Error: {e}")
        
        # Test 2: API system status
        print("🔍 Testing /api/system_status endpoint...")
        try:
            response = requests.get(f"{base_url}/api/system_status", timeout=10)
            if response.status_code == 200:
                print("✅ /api/system_status - OK")
                data = response.json()
                print(f"   System status: {data.get('success', False)}")
            else:
                print(f"❌ /api/system_status - Failed ({response.status_code})")
        except Exception as e:
            print(f"❌ /api/system_status - Error: {e}")
        
        # Test 3: File stats
        print("📁 Testing /api/file_stats endpoint...")
        try:
            response = requests.get(f"{base_url}/api/file_stats", timeout=10)
            if response.status_code == 200:
                print("✅ /api/file_stats - OK")
                data = response.json()
                stats = data.get('stats', {})
                print(f"   File stats: {stats}")
            else:
                print(f"❌ /api/file_stats - Failed ({response.status_code})")
        except Exception as e:
            print(f"❌ /api/file_stats - Error: {e}")
        
        print("\n🎉 Flask startup test completed!")
        
    finally:
        # Clean up - terminate Flask process
        print("🔄 Shutting down Flask app...")
        flask_process.terminate()
        try:
            flask_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            flask_process.kill()
        print("✅ Flask app stopped")

def run_quick_validation():
    """Run quick validation of key system components"""
    print("\n🔬 Running Quick System Validation...")
    
    # Check database
    if os.path.exists('greyhound_racing_data.db'):
        print("✅ Database file exists")
        
        # Quick database check
        try:
            import sqlite3
            conn = sqlite3.connect('greyhound_racing_data.db')
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM race_metadata")
            race_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM dog_race_data")
            dog_count = cursor.fetchone()[0]
            conn.close()
            
            print(f"✅ Database accessible: {race_count} races, {dog_count} dog entries")
        except Exception as e:
            print(f"❌ Database error: {e}")
    else:
        print("❌ Database file not found")
    
    # Check key directories
    directories = ['./unprocessed', './processed', './upcoming_races', './predictions']
    for directory in directories:
        if os.path.exists(directory):
            file_count = len([f for f in os.listdir(directory) if f.endswith('.csv') or f.endswith('.json')])
            print(f"✅ {directory}: {file_count} files")
        else:
            print(f"⚠️ {directory}: Directory missing")
    
    # Check model files
    model_dirs = ['./comprehensive_trained_models', './models']
    model_found = False
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib') or f.endswith('.pkl')]
            if model_files:
                print(f"✅ Models found: {len(model_files)} files in {model_dir}")
                model_found = True
                break
    
    if not model_found:
        print("⚠️ No trained model files found")
    
    print("✅ Quick validation completed!\n")

if __name__ == '__main__':
    print("🚀 Starting Flask App Validation Tests")
    print("=" * 50)
    
    # Run quick validation first
    run_quick_validation()
    
    # Test Flask startup
    test_flask_startup()
    
    print("\n📝 Summary:")
    print("- Flask app startup test completed")
    print("- Check console output above for any errors")
    print("- If tests pass, the Flask app should work correctly")
