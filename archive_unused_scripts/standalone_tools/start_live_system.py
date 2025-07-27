#!/usr/bin/env python3
"""
Live Odds System Startup Script
==============================

This script starts both the data collection system and the web dashboard.
"""

import subprocess
import threading
import time
import sys
import os
from pathlib import Path

def start_data_collection():
    """Start the odds collection system in background"""
    print("🚀 Starting odds collection system...")
    try:
        # Run the odds integrator with continuous monitoring
        result = subprocess.run([
            sys.executable, "sportsbet_odds_integrator.py"
        ], cwd=Path(__file__).parent)
    except KeyboardInterrupt:
        print("🛑 Stopping odds collection...")

def start_dashboard():
    """Start the Streamlit dashboard"""
    print("🌐 Starting web dashboard...")
    time.sleep(5)  # Give data collection a head start
    try:
        # Run the Streamlit dashboard
        result = subprocess.run([
            "streamlit", "run", "live_odds_dashboard.py", 
            "--server.port", "8501",
            "--server.headless", "false"
        ], cwd=Path(__file__).parent)
    except KeyboardInterrupt:
        print("🛑 Stopping dashboard...")

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['streamlit', 'plotly', 'selenium', 'pandas']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("⚠️  Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    return True

def main():
    """Main startup function"""
    print("🐕 Live Greyhound Odds System")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        return
    
    print("✅ All requirements satisfied")
    print("🎯 Starting live odds system...")
    
    # Start data collection in background thread
    collection_thread = threading.Thread(target=start_data_collection, daemon=True)
    collection_thread.start()
    
    # Start dashboard (this will block)
    try:
        start_dashboard()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down live odds system...")
    
    print("👋 Live odds system stopped")

if __name__ == "__main__":
    main()
