#!/usr/bin/env python3
"""
Debug script to examine race page content
"""

import re
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


def setup_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")

    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        return driver
    except Exception as e:
        print(f"Error setting up driver: {e}")
        return None


def debug_race_page():
    driver = setup_driver()
    if not driver:
        return

    try:
        # Navigate to a race page
        url = "https://www.sportsbet.com.au/greyhound-racing/australia-nz/geelong"
        print(f"🔍 Navigating to: {url}")
        driver.get(url)

        # Wait for page to load
        time.sleep(5)

        print(f"📄 Page title: {driver.title}")
        print(f"📄 Current URL: {driver.current_url}")

        # Look for various elements that might contain odds
        selectors_to_check = [
            "button",
            "[class*='odds']",
            "[class*='price']",
            "[class*='selection']",
            "[class*='runner']",
            "[class*='bet']",
            "[data-automation-id]",
            "table tr",
            "[role='button']",
        ]

        for selector in selectors_to_check:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    print(
                        f"\n🔍 Found {len(elements)} elements with selector: {selector}"
                    )

                    # Show first few elements' text content
                    for i, elem in enumerate(elements[:3]):
                        try:
                            text = elem.text.strip()
                            if text and len(text) < 100:  # Don't show very long text
                                print(f"  [{i}]: {text}")
                        except:
                            pass
            except:
                pass

        # Look for specific betting patterns in page source
        page_source = driver.page_source

        # Look for decimal odds patterns
        decimal_pattern = r"[1-9]\d*\.\d{2}"
        decimal_matches = re.findall(decimal_pattern, page_source)
        if decimal_matches:
            unique_odds = list(set(decimal_matches))[:10]  # Show unique values
            print(f"\n💰 Found decimal odds patterns: {unique_odds}")

        # Look for fractional odds patterns
        fractional_pattern = r"[1-9]\d*/[1-9]\d*"
        fractional_matches = re.findall(fractional_pattern, page_source)
        if fractional_matches:
            unique_fractions = list(set(fractional_matches))[:10]
            print(f"💰 Found fractional odds patterns: {unique_fractions}")

        # Look for automation IDs
        automation_pattern = r'data-automation-id="[^"]*"'
        automation_matches = re.findall(automation_pattern, page_source)
        if automation_matches:
            unique_automation = list(set(automation_matches))[:10]
            print(f"\n🤖 Found automation IDs: {unique_automation}")

        # Look for race-specific content
        if "race" in page_source.lower():
            print("✅ Page contains race-related content")
        else:
            print("⚠️  Page does not contain race-related content")

        if "greyhound" in page_source.lower():
            print("✅ Page contains greyhound-related content")
        else:
            print("⚠️  Page does not contain greyhound-related content")

        # Save a sample of the page source for manual inspection
        with open(
            "/Users/orlandolee/greyhound_racing_collector/debug_page_sample.txt", "w"
        ) as f:
            # Get first 5000 characters of page source
            f.write(page_source[:5000])
        print("💾 Saved page source sample to debug_page_sample.txt")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        driver.quit()


if __name__ == "__main__":
    debug_race_page()
