#!/usr/bin/env python3
"""
Debug script to test track condition extraction from race pages
"""

import sys
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import re

def setup_driver():
    """Setup Chrome driver for testing"""
    try:
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36')
        
        driver = webdriver.Chrome(options=options)
        print("✅ Chrome driver initialized")
        return driver
    except Exception as e:
        print(f"❌ Chrome driver setup failed: {e}")
        return None

def extract_track_conditions(soup):
    """Extract track conditions from race page - using same logic as processor"""
    print("🔍 Starting track condition extraction...")
    try:
        conditions = {}
        
        # Look for track condition information with expanded selectors
        condition_selectors = [
            '.track-condition',
            '.conditions',
            '[data-condition]',
            '.race-conditions .condition',
            '.meeting-conditions',
            '.track-details .condition',
            '.race-info .condition',
            '.conditions-panel .condition',
            '.race-header .condition',
            '.meeting-info .track-condition',
            '.track-info',
            '.meeting-info',
            '.race-details',
            '.track-data',
            '.race-card-header',
            '.meeting-header',
            '.race-meeting-info'
        ]
        
        for selector in condition_selectors:
            elem = soup.select_one(selector)
            if elem:
                text = elem.get_text(strip=True)
                # Filter out navigation/venue lists and other irrelevant text
                if text and len(text) > 2 and len(text) < 100:  # Not too long (likely navigation)
                    # Check if it's actually track condition info, not venue listings
                    if not any(word in text for word in ['TAS', 'NSW', 'VIC', 'SA', 'QLD', 'WA', 'Fields', 'races', 'R1', 'R2', 'R3', 'R4', 'R5', 'Ladbrokes']):
                        conditions['condition'] = text
                        print(f"   🏁 Found track condition via selector '{selector}': {text}")
                        break
        
        # Alternative: look for text patterns in the page
        if not conditions.get('condition'):
            page_text = soup.get_text()
            # Clean up page text - remove excessive whitespace
            page_text = re.sub(r'\s+', ' ', page_text)
            print(f"   📝 Searching in {len(page_text)} characters of page text...")
            
            # Look for common track condition terms with improved patterns
            condition_patterns = [
                r'track[\s:]*([\w\s]+?)(?:weather|temperature|\.|\n|$)',
                r'condition[\s:]*([\w\s]+?)(?:weather|temperature|\.|\n|$)',
                r'(fast|good|slow|heavy|dead)\s*track',
                r'track\s*(fast|good|slow|heavy|dead)',
                r'(fast|good|slow|heavy|dead)(?=\s|$)',
                r'track condition[\s:]*([\w\s]+?)(?:weather|temperature|\.|\n|$)',
                r'conditions?[\s:]*([\w\s]+?)(?:weather|temperature|\.|\n|$)',
                r'(?:track|condition)\s*(?:is|was)?\s*([\w\s]+?)(?:weather|temperature|\.|\n|$)',
                r'rail\s*(?:position)?[\s:]*([\w\s]+?)(?:weather|temperature|\.|\n|$)',
                r'(good|fast|slow|heavy|dead|firm|soft)\s*(?:track|condition)',
                r'(?:meeting|race)\s*conditions?[\s:]*([\w\s]+?)(?:weather|temperature|\.|\n|$)'
            ]
            
            for i, pattern in enumerate(condition_patterns):
                print(f"   🎯 Trying pattern {i+1}: {pattern}")
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    condition = match.group(1).strip()
                    # Filter out non-condition text and validate
                    if condition and len(condition) > 1 and not condition.isdigit():
                        # Check if it contains common track conditions
                        condition_words = ['fast', 'good', 'slow', 'heavy', 'dead', 'firm', 'soft', 'true', 'out']
                        if any(word in condition.lower() for word in condition_words):
                            conditions['condition'] = condition
                            print(f"   ✅ Found track condition via pattern: {condition}")
                            break
                        else:
                            print(f"   ⚠️ Pattern matched but no valid condition words: {condition}")
            
            # If still no condition found, look for specific track condition keywords
            if not conditions.get('condition'):
                print(f"   🔍 Falling back to keyword search...")
                track_keywords = ['fast', 'good', 'slow', 'heavy', 'dead', 'firm', 'soft']
                for keyword in track_keywords:
                    if keyword in page_text.lower():
                        # Extract surrounding context
                        pattern = rf'\b({keyword})\b'
                        match = re.search(pattern, page_text, re.IGNORECASE)
                        if match:
                            conditions['condition'] = match.group(1).title()
                            print(f"   ✅ Found track condition keyword: {match.group(1)}")
                            break
        
        return conditions if conditions else None
        
    except Exception as e:
        print(f"   ❌ Error extracting track conditions: {e}")
    
    return None

def test_track_condition_extraction(race_url):
    """Test track condition extraction on a specific race URL"""
    print(f"\n🧪 Testing track condition extraction on: {race_url}")
    
    driver = setup_driver()
    if not driver:
        return False
    
    try:
        # Load the race page
        print("🌐 Loading race page...")
        driver.get(race_url)
        time.sleep(3)  # Wait for page to load
        
        # Get page source and parse with BeautifulSoup
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        
        print(f"📄 Page loaded, source length: {len(page_source)} characters")
        
        # Extract track conditions
        track_info = extract_track_conditions(soup)
        
        print(f"\n📊 Results:")
        if track_info:
            print(f"   Track Condition: {track_info.get('condition', 'Not found')}")
            print(f"   Weather: {track_info.get('weather', 'Not found')}")
        else:
            print(f"   Track Condition: Not found")
            print(f"   Weather: Not found")
        
        # Debug: Show some relevant page content
        print(f"\n📝 Page content sample (first 1000 chars):")
        print(page_source[:1000])
        
        # Search for track-related keywords in the page
        track_keywords = ['track', 'condition', 'fast', 'good', 'slow', 'heavy', 'weather']
        print(f"\n🔍 Searching for track-related keywords:")
        page_text = soup.get_text()
        page_text_lower = page_text.lower()
        for keyword in track_keywords:
            count = page_text_lower.count(keyword)
            if count > 0:
                print(f"   '{keyword}': found {count} times")
                # Show context around each occurrence
                import re
                for match in re.finditer(re.escape(keyword), page_text_lower):
                    start = max(0, match.start() - 50)
                    end = min(len(page_text), match.end() + 50)
                    context = page_text[start:end].replace('\n', ' ').strip()
                    print(f"      Context: ...{context}...")
        
        return track_info and track_info.get('condition') is not None
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False
    finally:
        driver.quit()

if __name__ == "__main__":
    # Test with the race that should have track condition data (from database)
    test_url = "https://www.thedogs.com.au/racing/richmond/2025-07-16/8/ladbrokes-fast-withdrawals-2-4-win?trial=false"
    
    success = test_track_condition_extraction(test_url)
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}: Track condition extraction test")
