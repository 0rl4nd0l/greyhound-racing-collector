#!/usr/bin/env python3
"""
TGR Dashboard Startup Script
===========================

Simple script to start the TGR Dashboard server with all dependencies.
"""

import os
import platform
import subprocess
import sys
import time


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        "flask",
        "flask-cors",
        "flask-socketio",
        "python-socketio",
        "python-engineio",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")

    return missing_packages


def install_dependencies(packages):
    """Install missing dependencies."""
    if not packages:
        return True

    print(f"\n📦 Installing missing packages: {', '.join(packages)}")

    try:
        cmd = [sys.executable, "-m", "pip", "install"] + packages
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Failed to install dependencies: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False


def check_database():
    """Check if the database exists."""
    db_path = "greyhound_racing_data.db"
    if os.path.exists(db_path):
        size_mb = os.path.getsize(db_path) / (1024 * 1024)
        print(f"✅ Database found: {db_path} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"⚠️ Database not found: {db_path}")
        return False


def check_frontend_files():
    """Check if frontend files exist."""
    required_files = [
        "frontend/index.html",
        "frontend/css/dashboard.css",
        "frontend/js/dashboard.js",
        "frontend/js/charts.js",
        "frontend/js/api.js",
    ]

    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} found")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path} missing")

    return len(missing_files) == 0


def get_local_ip():
    """Get local IP address."""
    try:
        import socket

        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        return local_ip
    except:
        return "localhost"


def start_dashboard_server():
    """Start the dashboard server."""
    print("\n🚀 Starting TGR Dashboard Server...")
    print("=" * 50)

    try:
        # Set environment variables
        os.environ["FLASK_ENV"] = "development"
        os.environ["FLASK_DEBUG"] = "0"

        # Start the server
        from tgr_dashboard_server import app, socketio

        local_ip = get_local_ip()
        port = 5003

        print(f"🌐 Dashboard will be available at:")
        print(f"   • Local:   http://localhost:{port}")
        print(f"   • Network: http://{local_ip}:{port}")
        print(f"📊 API endpoints: http://localhost:{port}/api/v1/*")
        print(f"⚡ WebSocket: ws://localhost:{port}/socket.io/")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 50)

        # Run the server
        socketio.run(
            app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True
        )

    except ImportError as e:
        print(f"❌ Failed to import dashboard server: {e}")
        print("Make sure tgr_dashboard_server.py is in the current directory")
        return False
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
        return True
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False


def main():
    """Main startup function."""
    print("🎯 TGR Dashboard Startup")
    print("=" * 30)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Check and install dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        if not install_dependencies(missing_deps):
            print("\n❌ Failed to install required dependencies")
            print("Please install them manually with:")
            print(f"   pip install {' '.join(missing_deps)}")
            sys.exit(1)

    # Check database
    check_database()

    # Check frontend files
    if not check_frontend_files():
        print("\n⚠️ Some frontend files are missing")
        print("The dashboard may not function correctly")

        response = input("Continue anyway? (y/N): ").lower()
        if response != "y":
            print("Startup cancelled")
            sys.exit(1)

    # Start the server
    if not start_dashboard_server():
        sys.exit(1)


if __name__ == "__main__":
    main()
