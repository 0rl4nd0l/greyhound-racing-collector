#!/usr/bin/env python3
"""
Test Chrome WebDriver setup
This script verifies that the Chrome WebDriver can be properly initialized.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_chrome_setup():
    """Test Chrome WebDriver setup"""
    try:
        print("🧪 Testing Chrome WebDriver setup...")
        
        # Import the driver setup function
        from drivers import get_chrome_driver
        
        # Try to create a Chrome driver instance
        print("🚀 Initializing Chrome WebDriver...")
        driver = get_chrome_driver(headless=True)
        
        # Test basic functionality
        print("🌐 Testing basic WebDriver functionality...")
        driver.get("https://www.google.com")
        title = driver.title
        print(f"✅ Successfully loaded page: {title}")
        
        # Check Chrome and ChromeDriver versions
        chrome_version = driver.capabilities['browserVersion']
        chromedriver_version = driver.capabilities['chrome']['chromedriverVersion'].split(' ')[0]
        
        print(f"🔍 Chrome version: {chrome_version}")
        print(f"🔍 ChromeDriver version: {chromedriver_version}")
        
        # Close the driver
        driver.quit()
        print("✅ Chrome WebDriver setup test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Chrome WebDriver setup test FAILED: {e}")
        print("\n💡 Troubleshooting suggestions:")
        print("1. Update Chrome browser: brew upgrade google-chrome")
        print("2. Update ChromeDriver: brew upgrade chromedriver")
        print("3. Clear webdriver-manager cache: rm -rf ~/.wdm")
        print("4. Check Chrome installation: /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --version")
        return False

def check_chrome_versions():
    """Check Chrome and ChromeDriver versions for compatibility"""
    try:
        import subprocess
        
        # Check Chrome version
        chrome_result = subprocess.run([
            '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', 
            '--version'
        ], capture_output=True, text=True)
        
        if chrome_result.returncode == 0:
            chrome_version = chrome_result.stdout.strip()
            print(f"🌐 Chrome Browser: {chrome_version}")
        else:
            print("⚠️ Could not determine Chrome browser version")
        
        # Check ChromeDriver version
        chromedriver_result = subprocess.run([
            'chromedriver', '--version'
        ], capture_output=True, text=True, timeout=5)
        
        if chromedriver_result.returncode == 0:
            chromedriver_version = chromedriver_result.stdout.strip()
            print(f"🔧 ChromeDriver: {chromedriver_version}")
        else:
            print("⚠️ Could not determine ChromeDriver version")
            
    except subprocess.TimeoutExpired:
        print("⚠️ ChromeDriver version check timed out (possible compatibility issue)")
    except Exception as e:
        print(f"⚠️ Error checking versions: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 CHROME WEBDRIVER SETUP TEST")
    print("=" * 60)
    
    # Check versions first
    check_chrome_versions()
    print("-" * 60)
    
    # Run the main test
    success = test_chrome_setup()
    
    print("-" * 60)
    if success:
        print("🎉 All tests passed! WebDriver setup is working correctly.")
        sys.exit(0)
    else:
        print("💥 Tests failed! Please check the troubleshooting suggestions above.")
        sys.exit(1)
