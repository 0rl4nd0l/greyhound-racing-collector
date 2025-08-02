from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import os


def get_chrome_driver(headless=True):
    """
    Set up Chrome WebDriver with options

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
    options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    # Use webdriver-manager to download and set up the ChromeDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver


def setup_selenium_driver_path():
    """
    Set SELENIUM_DRIVER_PATH environment variable
    """
    driver_path = ChromeDriverManager().install()
    os.environ['SELENIUM_DRIVER_PATH'] = driver_path
    return driver_path
