#!/usr/bin/env python3
"""
Sportsbet Dashboard Launcher
============================

This script launches the Sportsbet Live Odds Dashboard with proper error handling
and setup instructions.
"""

import os
import subprocess
import sys
from pathlib import Path


def check_streamlit():
    """Check if Streamlit is installed"""
    try:
        import streamlit as st

        return True
    except ImportError:
        return False


def install_streamlit():
    """Install Streamlit"""
    print("📦 Installing Streamlit...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        print("✅ Streamlit installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Streamlit: {e}")
        return False


def launch_dashboard():
    """Launch the Sportsbet dashboard"""
    dashboard_path = Path(
        "archive_unused_scripts/standalone_tools/live_odds_dashboard.py"
    )

    if not dashboard_path.exists():
        print(f"❌ Dashboard file not found: {dashboard_path}")
        return False

    print("🚀 Launching Sportsbet Live Odds Dashboard...")
    print("📱 The dashboard will open in your browser at http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the dashboard")

    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])
    except KeyboardInterrupt:
        print("🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")
        return False

    return True


def main():
    print("🐕 Greyhound Racing - Sportsbet Dashboard Launcher")
    print("=" * 50)

    # Check if Streamlit is available
    if not check_streamlit():
        print("⚠️ Streamlit is not installed")
        response = input("Would you like to install Streamlit? (y/n): ").lower().strip()
        if response in ["y", "yes"]:
            if not install_streamlit():
                print(
                    "❌ Installation failed. Please install manually: pip install streamlit"
                )
                return
            print("✅ Installation complete!")
        else:
            print("❌ Streamlit is required to run the Sportsbet dashboard")
            print("💡 Install it manually with: pip install streamlit")
            return

    # Launch dashboard
    if not launch_dashboard():
        print("❌ Failed to launch dashboard")
        return

    print("👋 Dashboard session ended")


if __name__ == "__main__":
    main()
