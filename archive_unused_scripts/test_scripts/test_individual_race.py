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
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        return driver
    except Exception as e:
        print(f"Failed to setup driver: {e}")
        return None

def test_individual_race():
    """Test what's on an individual race page"""
    driver = setup_driver()
    if not driver:
        return
    
    try:
        # Test the first race from Warragul (with 9m countdown)
        test_url = "https://www.sportsbet.com.au/greyhound-racing/australia-nz/warragul/race-1-9436742"
        print(f"🎯 Testing individual race page: {test_url}")
        
        driver.get(test_url)
        time.sleep(5)  # Give more time for dynamic content
        
        print(f"📄 Page title: {driver.title}")
        
        # Look for runners/dogs
        print("\n🐕 Looking for runners/dogs...")
        runner_selectors = [
            "[class*='runner']",
            "[class*='selection']", 
            "[data-automation-id*='runner']",
            "[class*='participant']",
            "[class*='competitor']"
        ]
        
        found_runners = False
        for selector in runner_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    print(f"  📋 Found {len(elements)} runners with selector: {selector}")
                    found_runners = True
                    
                    # Show all runners with their text
                    for i, elem in enumerate(elements[:8]):
                        try:
                            text = elem.text.strip()
                            print(f"    Runner {i+1}: {text}")
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
                if re.match(r'^\d+\.\d{2}$', text):  # Looks like decimal odds
                    odds_buttons.append(text)
            except:
                continue
        
        print(f"  💸 Found {len(odds_buttons)} odds-like buttons: {odds_buttons}")
        
        # Look for any text that might contain dog names and odds
        print("\n🔍 Looking for dog names and odds patterns...")
        page_text = driver.page_source
        
        # Look for patterns like "Dog Name 2.50" or "1. Dog Name (1) 3.25"
        patterns = [
            r'(\w+(?:\s+\w+)*)\s+(\d+\.\d{2})',  # Dog Name 2.50
            r'(\d+)\.\s*([A-Za-z\s\']+)\s*\(\d+\)\s*(\d+\.\d{2})',  # 1. Dog Name (1) 2.50
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, page_text)
            if matches:
                print(f"  🎯 Found {len(matches)} matches with pattern: {pattern}")
                for i, match in enumerate(matches[:5]):  # Show first 5
                    print(f"    {i+1}. {match}")
                break
        
        # Check race info
        print(f"\n📊 Race information...")
        if 'warragul' in test_url.lower():
            print("  🏟️ Venue: Warragul")
        
        race_match = re.search(r'/race-(\d+)-', test_url)
        if race_match:
            print(f"  🏁 Race number: {race_match.group(1)}")
        
        # Look for race status
        status_keywords = ['upcoming', 'next', 'live', 'finished', 'result']
        page_lower = driver.page_source.lower()
        for keyword in status_keywords:
            if keyword in page_lower:
                print(f"  📊 Status keyword '{keyword}' found")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    test_individual_race()
