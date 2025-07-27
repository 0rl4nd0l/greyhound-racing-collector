#!/usr/bin/env python3
"""
Quick debug script to test Sportsbet race extraction
"""

import requests
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import re
import json

def setup_driver():
    """Setup Chrome driver"""
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        return driver
    except Exception as e:
        print(f"Failed to setup driver: {e}")
        return None

def debug_sportsbet():
    """Debug what's happening with Sportsbet"""
    url = "https://www.sportsbet.com.au/racing-schedule/greyhound/today"
    
    print(f"🔍 Debugging Sportsbet at: {url}")
    print(f"⏰ Current time: {datetime.now()}")
    print("-" * 60)
    
    driver = setup_driver()
    if not driver:
        return
    
    try:
        # Load the page
        print("📄 Loading page...")
        driver.get(url)
        time.sleep(5)
        
        # Get basic page info
        title = driver.title
        print(f"📋 Page title: {title}")
        
        # Check if page loaded properly
        page_source = driver.page_source.lower()
        print(f"🔍 Page contains 'greyhound': {'greyhound' in page_source}")
        print(f"🔍 Page contains 'race': {'race' in page_source}")
        print(f"🔍 Page length: {len(page_source)} characters")
        
        # Try to find JSON-LD data
        print("\n🔍 Looking for JSON-LD structured data...")
        json_pattern = r'\[\{"@context":"https://schema\.org".*?"@type":"SportsEvent".*?\}\]'
        matches = re.search(json_pattern, driver.page_source, re.DOTALL)
        
        if matches:
            print("✅ Found JSON-LD data!")
            json_data_str = matches.group(0)
            print(f"📊 JSON length: {len(json_data_str)} characters")
            
            try:
                events = json.loads(json_data_str)
                print(f"🏁 Found {len(events)} events in JSON-LD")
                
                current_time = datetime.now()
                upcoming_count = 0
                
                for i, event in enumerate(events[:10]):  # Show first 10
                    if event.get('@type') == 'SportsEvent':
                        venue_name = event.get('name', 'Unknown')
                        start_date = event.get('startDate', '')
                        
                        try:
                            if 'T' in start_date:
                                race_datetime = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                            else:
                                race_datetime = datetime.strptime(start_date, '%d %b %Y %H:%M:%S')
                            
                            time_diff = race_datetime - current_time
                            is_upcoming = race_datetime > current_time
                            
                            if is_upcoming:
                                upcoming_count += 1
                            
                            print(f"  {i+1}. {venue_name} - {race_datetime.strftime('%H:%M')} ({'UPCOMING' if is_upcoming else 'FINISHED'}) [{time_diff.total_seconds()/60:.0f}min]")
                            
                        except Exception as e:
                            print(f"  {i+1}. {venue_name} - Error parsing time: {e}")
                
                print(f"\n📊 Summary: {upcoming_count} upcoming races out of {len(events)} total")
                
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON-LD: {e}")
        else:
            print("❌ No JSON-LD structured data found")
        
        # Try to find race elements in DOM
        print("\n🔍 Looking for race elements in DOM...")
        selectors_to_try = [
            "a[href*='greyhound']",
            "[class*='race']",
            "[data-automation-id='race-card']",
            ".race-card"
        ]
        
        for selector in selectors_to_try:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    print(f"  ✅ Found {len(elements)} elements with selector: {selector}")
                    
                    # Show first few elements
                    for i, elem in enumerate(elements[:5]):
                        try:
                            text = elem.text.strip()[:100]
                            href = elem.get_attribute('href') if elem.tag_name == 'a' else 'N/A'
                            print(f"    {i+1}. Text: {text}... | Href: {href}")
                        except:
                            print(f"    {i+1}. Error getting element info")
                    break
            except Exception as e:
                print(f"  ❌ Error with selector {selector}: {e}")
        
    except Exception as e:
        print(f"❌ Error debugging: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    debug_sportsbet()
