#!/usr/bin/env python3
"""
Scraping System Setup Script
============================

Installs and configures all dependencies for the greyhound racing scraping system.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a shell command with error handling."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def check_python_packages():
    """Check and install required Python packages."""
    required_packages = [
        'requests',
        'beautifulsoup4',
        'selenium',
        'python-dotenv',
        'lxml',
        'html5lib',
        'urllib3'
    ]
    
    print("📦 Checking Python packages...")
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} - already installed")
        except ImportError:
            print(f"📦 Installing {package}...")
            if not run_command(f"pip install {package}", f"Installing {package}"):
                print(f"⚠️ Failed to install {package} - you may need to install it manually")

def check_chrome_driver():
    """Check if Chrome and ChromeDriver are available."""
    print("🌐 Checking Chrome/ChromeDriver...")
    
    # Check if Chrome is installed
    chrome_paths = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium-browser"
    ]
    
    chrome_found = any(Path(path).exists() for path in chrome_paths)
    if chrome_found:
        print("✅ Chrome browser found")
    else:
        print("⚠️ Chrome browser not found - some scrapers may not work")
        print("   Install Chrome from: https://www.google.com/chrome/")
    
    # Check ChromeDriver
    try:
        result = subprocess.run(['chromedriver', '--version'], capture_output=True, text=True)
        print(f"✅ ChromeDriver found: {result.stdout.strip()}")
    except FileNotFoundError:
        print("⚠️ ChromeDriver not found")
        print("   Install with: brew install chromedriver (macOS)")
        print("   Or download from: https://chromedriver.chromium.org/")

def setup_directories():
    """Create necessary directories."""
    directories = [
        'logs',
        'upcoming_races',
        'processed',
        'unprocessed',
        '.scraping_cache',
        '.scraping_cache/fasttrack',
        '.scraping_cache/tgr',
        'quarantine'
    ]
    
    print("📁 Setting up directories...")
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Created/verified: {dir_name}")

def main():
    print("="*80)
    print("🚀 GREYHOUND RACING SCRAPING SYSTEM SETUP")
    print("="*80)
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("⚠️ Python 3.8+ recommended for best compatibility")
    else:
        print("✅ Python version is compatible")
    
    # Setup components
    setup_directories()
    check_python_packages()
    check_chrome_driver()
    
    # Verify environment file
    env_file = Path('.env')
    if env_file.exists():
        print("✅ Environment file (.env) exists")
        
        # Check key settings
        with open(env_file) as f:
            env_content = f.read()
            
        if 'ENABLE_RESULTS_SCRAPERS=1' in env_content:
            print("✅ Results scrapers enabled")
        else:
            print("⚠️ Results scrapers may not be enabled")
            
        if 'ENABLE_LIVE_SCRAPING=1' in env_content:
            print("✅ Live scraping enabled")
        else:
            print("⚠️ Live scraping may not be enabled")
    else:
        print("⚠️ Environment file (.env) not found - some features may not work")
    
    print("\n" + "="*80)
    print("🎯 SETUP COMPLETE")
    print("="*80)
    
    print("\n📋 Next steps:")
    print("1. Test the scraping system:")
    print("   python scraping_manager.py --mode status")
    print("\n2. Scrape upcoming races:")
    print("   python scraping_manager.py --mode upcoming --days 1")
    print("\n3. Test FastTrack scraper:")
    print("   python scraping_manager.py --mode fasttrack --dogs 5")
    print("\n4. Full system status:")
    print("   python scraping_manager.py --mode status")
    
    print("\n🌐 Available scrapers:")
    print("   • FastTrack (GRV Official)")
    print("   • The Greyhound Recorder")
    print("   • Live Upcoming Races Browser")
    print("   • Comprehensive Form Data Collector")
    
    print(f"\n📁 All data will be saved to: {Path.cwd()}")
    print(f"📊 Logs will be saved to: {Path.cwd() / 'logs'}")
    print(f"🗄️ Cache will be stored in: {Path.cwd() / '.scraping_cache'}")

if __name__ == '__main__':
    main()
