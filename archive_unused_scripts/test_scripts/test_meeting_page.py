#!/usr/bin/env python3
"""
Test meeting page structure to find individual races
"""

import re
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager


def setup_driver():
    """Setup Chrome driver"""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        return driver
    except Exception as e:
        print(f"Failed to setup driver: {e}")
        return None


def test_meeting_page():
    """Test what's on a meeting page"""
    driver = setup_driver()
    if not driver:
        return

    try:
        # Test a meeting page that shows countdown (Warragul with 11m)
        test_url = "https://www.sportsbet.com.au/greyhound-racing/australia-nz/warragul/meeting-9436742"
        print(f"🎯 Testing meeting page: {test_url}")

        driver.get(test_url)
        time.sleep(5)  # Give more time for dynamic content

        print(f"📄 Page title: {driver.title}")

        # Look for individual race cards or sections
        print("\n🏁 Looking for individual races...")

        race_selectors = [
            "[class*='race-card']",
            "[class*='race']",
            "[data-automation-id*='race']",
            "section",
            "article",
            "[class*='event']",
        ]

        for selector in race_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements and len(elements) > 2:
                    print(
                        f"  📋 Found {len(elements)} elements with selector: {selector}"
                    )

                    # Check first few for race-like content
                    for i, elem in enumerate(elements[:5]):
                        try:
                            text = elem.text.strip()
                            if (
                                "race" in text.lower()
                                or "r1" in text.lower()
                                or "r2" in text.lower()
                            ) and len(text) > 20:
                                print(f"    {i+1}. Race-like content: {text[:200]}...")
                        except:
                            continue
                    break
            except:
                continue

        # Look for runners within the page
        print("\n🐕 Looking for runners/dogs...")
        runner_selectors = [
            "[class*='runner']",
            "[class*='selection']",
            "[data-automation-id*='runner']",
            "[class*='participant']",
            "[class*='competitor']",
        ]

        found_runners = False
        for selector in runner_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    print(
                        f"  📋 Found {len(elements)} runners with selector: {selector}"
                    )
                    found_runners = True

                    # Show first few runners
                    for i, elem in enumerate(elements[:8]):
                        try:
                            text = elem.text.strip()[:100]
                            print(f"    Runner {i+1}: {text}...")
                        except:
                            continue
                    break
            except:
                continue

        if not found_runners:
            print("  ❌ No runners found")

        # Look for odds buttons
        print("\n💰 Looking for odds buttons...")
        buttons = driver.find_elements(By.CSS_SELECTOR, "button")
        odds_buttons = []

        for button in buttons:
            try:
                text = button.text.strip()
                if re.match(r"^\d+\.\d{2}$", text):  # Looks like decimal odds
                    odds_buttons.append(text)
            except:
                continue

        print(f"  💸 Found {len(odds_buttons)} odds-like buttons: {odds_buttons[:10]}")

        # Look for race navigation or tabs
        print("\n📑 Looking for race navigation...")
        nav_selectors = [
            "[class*='tab']",
            "[class*='nav']",
            "a[href*='race']",
            "[role='tab']",
            "[class*='menu']",
        ]

        for selector in nav_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                race_nav = []
                for elem in elements:
                    try:
                        text = elem.text.strip()
                        href = (
                            elem.get_attribute("href") if elem.tag_name == "a" else None
                        )
                        if (
                            "r1" in text.lower()
                            or "r2" in text.lower()
                            or "race" in text.lower()
                        ) and len(text) < 50:
                            race_nav.append((text, href))
                    except:
                        continue

                if race_nav:
                    print(f"  📋 Found {len(race_nav)} race navigation items:")
                    for text, href in race_nav[:5]:
                        print(f"    {text} -> {href}")
                    break
            except:
                continue

        # Check if this is a "next race" page that automatically shows the next upcoming race
        print(f"\n📊 Page content analysis...")
        page_text = driver.page_source.lower()

        keywords = ["next race", "upcoming", "race 1", "race 2", "greyhound", "odds"]
        for keyword in keywords:
            count = page_text.count(keyword)
            print(f"  '{keyword}': {count} occurrences")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        driver.quit()


if __name__ == "__main__":
    test_meeting_page()
