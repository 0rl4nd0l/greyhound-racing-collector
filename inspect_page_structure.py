import csv
import re
import sqlite3
import time

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def inspect_page_structure():
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")

    try:
        driver = webdriver.Chrome(options=chrome_options)
        print("✅ Chrome driver initialized successfully")

        # Test URL formation and scraping
        test_url = "https://www.thedogs.com.au/racing/2024-07-28/warrnambool/race-12"
        print(f"🌐 Testing URL: {test_url}")

        driver.get(test_url)
        time.sleep(5)  # Wait longer for page to load

        # Get page title
        title = driver.title
        print(f"📄 Page title: {title}")

        # Check if page loaded correctly
        page_source = driver.page_source
        print(f"📏 Page source length: {len(page_source)} characters")

        # Look for common race result indicators
        result_indicators = [
            "result",
            "winner",
            "first",
            "position",
            "place",
            "finish",
            "race-result",
            "results",
            "placing",
            "pos-1",
            "position-1",
        ]

        found_indicators = []
        for indicator in result_indicators:
            if indicator.lower() in page_source.lower():
                found_indicators.append(indicator)

        print(f"🔍 Found result indicators: {found_indicators}")

        # Look for specific elements that might contain results
        selectors_to_test = [
            ".race-result",
            ".results",
            ".result",
            ".winner",
            ".first-place",
            '[data-position="1"]',
            ".pos-1",
            ".position-1",
            ".placing",
            "table tr",
            ".runner",
            ".dog",
            ".greyhound",
        ]

        print("\n🎯 TESTING SELECTORS:")
        for selector in selectors_to_test:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    print(f"  ✅ {selector}: {len(elements)} elements found")
                    # Show first element text
                    if elements[0].text.strip():
                        print(f"     Sample text: {elements[0].text.strip()[:100]}...")
                else:
                    print(f"  ❌ {selector}: No elements found")
            except Exception as e:
                print(f"  ⚠️  {selector}: Error - {e}")

        # Look for tables which often contain race results
        tables = driver.find_elements(By.TAG_NAME, "table")
        print(f"\n📊 Found {len(tables)} tables on page")

        for i, table in enumerate(tables[:3]):  # Check first 3 tables
            try:
                rows = table.find_elements(By.TAG_NAME, "tr")
                print(f"  Table {i+1}: {len(rows)} rows")
                if rows:
                    first_row_text = rows[0].text.strip()
                    print(f"    First row: {first_row_text[:100]}...")
            except Exception as e:
                print(f"    Error reading table {i+1}: {e}")

        # Save a snippet of the page source for manual inspection
        with open("page_source_sample.html", "w", encoding="utf-8") as f:
            f.write(page_source[:10000])  # First 10k characters
        print("\n💾 Saved page source sample to page_source_sample.html")

        driver.quit()
        return True

    except Exception as e:
        print(f"❌ Inspection failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🔍 INSPECTING PAGE STRUCTURE FOR RACE RESULTS")
    print("=" * 60)
    success = inspect_page_structure()
    print(f'\nInspection result: {"COMPLETED" if success else "FAILED"}')
