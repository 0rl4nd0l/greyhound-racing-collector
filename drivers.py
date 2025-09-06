import os
import shutil

# Respect DISABLE_SELENIUM without breaking test collection.
# When disabled, we avoid importing selenium at module import time and provide
# stubs that raise at call-time instead of raising ImportError here.
_SELENIUM_DISABLED = str(os.environ.get("DISABLE_SELENIUM", "0")).strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

if not _SELENIUM_DISABLED:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
else:
    webdriver = None
    Service = None
    ChromeDriverManager = None
    Options = None


def get_chrome_driver(headless=True):
    """
    Set up Chrome WebDriver with options and improved error handling.
    If DISABLE_SELENIUM=1, raise a clear RuntimeError so tests can catch/skip.

    Args:
        headless (bool): Run Chrome in headless mode if True

    Returns:
        WebDriver: Configured Chrome WebDriver instance
    """
    if _SELENIUM_DISABLED:
        try:
            import pytest  # type: ignore

            pytest.skip("Selenium disabled via DISABLE_SELENIUM=1")
        except Exception:
            # Fallback if pytest not available: raise a clear runtime error
            raise RuntimeError("Selenium disabled via DISABLE_SELENIUM=1")

    options = Options()

    # Basic options
    if headless:
        options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")

    # Additional options for better compatibility and performance
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-plugins")
    options.add_argument("--disable-images")
    # Allow JS by default; can be disabled via DISABLE_JS=1 for specialized tests
    _js_disabled = str(os.environ.get("DISABLE_JS", "0")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if _js_disabled:
        options.add_argument("--disable-javascript")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/********* Safari/537.36"
    )

    # Add additional stability options
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    try:
        # Prefer system ChromeDriver if available
        system_chromedriver = shutil.which("chromedriver")
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
    """Set SELENIUM_DRIVER_PATH environment variable.
    When selenium is disabled, set to empty string and return ''.
    """
    if _SELENIUM_DISABLED:
        os.environ["SELENIUM_DRIVER_PATH"] = ""
        return ""
    driver_path = ChromeDriverManager().install()
    os.environ["SELENIUM_DRIVER_PATH"] = driver_path
    return driver_path
