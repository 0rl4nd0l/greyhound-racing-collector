#!/usr/bin/env python3
"""
Test individual race page structure
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


def test_race_page():
    """Test what's on a race page"""
    driver = setup_driver()
    if not driver:
        return

    try:
        # Test the main schedule page first
        print("🔍 Testing main schedule page...")
        driver.get("https://www.sportsbet.com.au/racing-schedule/greyhound/today")
        time.sleep(3)

        # Look specifically for race links with countdown timers
        print("\n📋 Looking for race links...")

        # Find all links that might be race links
        links = driver.find_elements(By.CSS_SELECTOR, "a[href*='greyhound']")

        race_links = []
        for link in links[:20]:  # Check first 20 links
            try:
                href = link.get_attribute("href")
                text = link.text.strip()

                # Look for links that might be specific races
                if ("race-" in href or "/race/" in href) and len(text) > 5:
                    race_links.append((href, text))
                    print(f"  🔗 Race link: {href}")
                    print(f"      Text: {text[:100]}")

            except:
                continue

        if race_links:
            # Test the first race link
            test_url, test_text = race_links[0]
            print(f"\n🎯 Testing race page: {test_url}")

            driver.get(test_url)
            time.sleep(3)

            print(f"📄 Page title: {driver.title}")

            # Look for runners/dogs
            print("\n🐕 Looking for runners...")
            runner_selectors = [
                "[class*='runner']",
                "[class*='selection']",
                "[data-automation-id*='runner']",
                "[class*='participant']",
            ]

            for selector in runner_selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        print(
                            f"  📋 Found {len(elements)} elements with selector: {selector}"
                        )

                        # Show first few elements
                        for i, elem in enumerate(elements[:3]):
                            text = elem.text.strip()[:150]
                            print(f"    {i+1}. {text}...")
                        break
                except:
                    continue

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

            print(
                f"  💸 Found {len(odds_buttons)} odds-like buttons: {odds_buttons[:10]}"
            )

        else:
            print("❌ No specific race links found")

            # Show what links we did find
            print("\n🔍 All greyhound links found:")
            for link in links[:10]:
                try:
                    href = link.get_attribute("href")
                    text = link.text.strip()[:50]
                    print(f"  {href} - {text}")
                except:
                    continue

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        driver.quit()


if __name__ == "__main__":
    test_race_page()
