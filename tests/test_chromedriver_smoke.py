#!/usr/bin/env python3
"""
ChromeDriver Smoke Test

This script tests that the ChromeDriver is properly set up and can open a webpage.
It verifies that Selenium with webdriver-manager is working correctly.
"""

import logging
import os
import sys

# Add parent directory to path so we can import drivers module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drivers import get_chrome_driver, setup_selenium_driver_path


def test_chromedriver_smoke():
    """
    Smoke test to verify ChromeDriver is working properly.
    Opens about:blank page and checks basic functionality.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting ChromeDriver smoke test...")

    # Set up driver path
    try:
        driver_path = setup_selenium_driver_path()
        logger.info(f"ChromeDriver path: {driver_path}")
    except Exception as e:
        logger.error(f"Failed to set up ChromeDriver path: {e}")
        return False

    # Initialize driver
    driver = None
    try:
        logger.info("Initializing Chrome WebDriver...")
        driver = get_chrome_driver(headless=True)
        logger.info("✅ Chrome WebDriver initialized successfully")

        # Test opening a simple page
        logger.info("Opening about:blank page...")
        driver.get("about:blank")

        # Verify page title
        title = driver.title
        logger.info(f"Page title: '{title}'")

        # Verify we can get page source
        page_source = driver.page_source
        assert len(page_source) > 0, "Page source should not be empty"
        logger.info(f"Page source length: {len(page_source)} characters")

        # Test getting current URL
        current_url = driver.current_url
        logger.info(f"Current URL: {current_url}")
        assert (
            "about:blank" in current_url
        ), f"Expected about:blank in URL, got: {current_url}"

        logger.info("✅ ChromeDriver smoke test PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ ChromeDriver smoke test FAILED: {e}")
        return False

    finally:
        if driver:
            try:
                driver.quit()
                logger.info("Chrome WebDriver closed successfully")
            except Exception as e:
                logger.error(f"Error closing driver: {e}")


def run_comprehensive_test():
    """Run comprehensive ChromeDriver tests"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("ChromeDriver + Selenium Stack Verification")
    logger.info("=" * 60)

    # Test 1: Basic smoke test
    logger.info("\n1. Running basic smoke test...")
    basic_test_passed = test_chromedriver_smoke()

    # Test 2: Test with different options
    logger.info("\n2. Testing with different Chrome options...")
    driver = None
    try:
        driver = get_chrome_driver(headless=False)  # Test non-headless mode
        driver.set_window_size(1920, 1080)
        driver.get("about:blank")
        logger.info("✅ Non-headless mode test passed")
        options_test_passed = True
    except Exception as e:
        logger.error(f"❌ Non-headless mode test failed: {e}")
        options_test_passed = False
    finally:
        if driver:
            driver.quit()

    # Test 3: Environment variable check
    logger.info("\n3. Checking environment variables...")
    import os

    selenium_path = os.environ.get("SELENIUM_DRIVER_PATH")
    if selenium_path:
        logger.info(f"✅ SELENIUM_DRIVER_PATH set to: {selenium_path}")
        env_test_passed = True
    else:
        logger.error("❌ SELENIUM_DRIVER_PATH not set")
        env_test_passed = False

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(
        f"Basic smoke test:      {'✅ PASSED' if basic_test_passed else '❌ FAILED'}"
    )
    logger.info(
        f"Chrome options test:   {'✅ PASSED' if options_test_passed else '❌ FAILED'}"
    )
    logger.info(
        f"Environment variables: {'✅ PASSED' if env_test_passed else '❌ FAILED'}"
    )

    all_passed = basic_test_passed and options_test_passed and env_test_passed
    logger.info(
        f"\nOverall result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}"
    )

    return all_passed


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
