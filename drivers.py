from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import os
import shutil


def get_chrome_driver(headless=True):
    """
    Set up Chrome WebDriver with options and improved error handling

    Args:
        headless (bool): Run Chrome in headless mode if True

    Returns:
        WebDriver: Configured Chrome WebDriver instance
    """
    options = Options()
    
    # Basic options
    if headless:
        options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    
    # Additional options for better compatibility and performance
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-plugins')
    options.add_argument('--disable-images')
    options.add_argument('--disable-javascript')
    options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36')
    
    # Add additional stability options for macOS
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    try:
        # First try using system ChromeDriver if available and compatible
        system_chromedriver = shutil.which('chromedriver')
        if system_chromedriver:
            print(f"🔧 Using system ChromeDriver: {system_chromedriver}")
            service = Service(system_chromedriver)
            try:
                driver = webdriver.Chrome(service=service, options=options)
                return driver
            except Exception as e:
                print(f"⚠️ System ChromeDriver failed: {e}")
                print("🔄 Falling back to webdriver-manager...")
        
        # Fallback to webdriver-manager
        print("📥 Using webdriver-manager to get ChromeDriver...")
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        return driver
        
    except Exception as e:
        print(f"❌ Chrome driver setup failed: {e}")
        print("💡 Try updating Chrome browser and ChromeDriver:")
        print("   brew upgrade chromedriver")
        print("   or manually download from: https://chromedriver.chromium.org/")
        raise


def setup_selenium_driver_path():
    """
    Set SELENIUM_DRIVER_PATH environment variable
    """
    driver_path = ChromeDriverManager().install()
    os.environ['SELENIUM_DRIVER_PATH'] = driver_path
    return driver_path
