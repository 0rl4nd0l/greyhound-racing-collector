
import sqlite3
import csv
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re
import random

def test_with_evasion():
    # Enhanced Chrome options to avoid detection
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        
        # Execute script to remove webdriver property
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        print('✅ Chrome driver initialized with evasion techniques')
        
        # Test with a more recent date that might exist
        test_urls = [
            'https://www.thedogs.com.au/racing/2024-12-01/warrnambool/race-1',
            'https://www.thedogs.com.au/racing/2024-11-15/warrnambool/race-1',
            'https://www.thedogs.com.au/racing/2024-10-01/warrnambool/race-1'
        ]
        
        for test_url in test_urls:
            print(f'\n🌐 Testing URL: {test_url}')
            
            # Random delay to appear more human-like
            time.sleep(random.uniform(2, 4))
            
            driver.get(test_url)
            time.sleep(random.uniform(3, 6))
            
            # Get page title and basic info
            title = driver.title
            page_source = driver.page_source
            
            print(f'📄 Page title: {title}')
            print(f'📏 Page source length: {len(page_source)} characters')
            
            if '403' in title or '403' in page_source:
                print('❌ Still getting 403 Forbidden')
                continue
            elif 'race' in title.lower() or 'warrnambool' in title.lower():
                print('✅ Successfully accessed race page!')
                
                # Look for race results
                result_indicators = ['winner', 'result', 'first', 'position']
                found = []
                for indicator in result_indicators:
                    if indicator.lower() in page_source.lower():
                        found.append(indicator)
                
                print(f'🔍 Found indicators: {found}')
                
                # Try to find any dog names or results
                dog_elements = driver.find_elements(By.CSS_SELECTOR, '*[class*="dog"], *[class*="runner"], *[class*="greyhound"]')
                if dog_elements:
                    print(f'🐕 Found {len(dog_elements)} potential dog elements')
                
                # Save successful page for analysis
                if len(page_source) > 1000:  # Only if we got substantial content
                    with open(f'successful_page_{test_url.split("/")[-2]}_{test_url.split("/")[-1]}.html', 'w', encoding='utf-8') as f:
                        f.write(page_source)
                    print('💾 Saved successful page source')
                
                break
            else:
                print('⚠️  Unknown page type')
        
        driver.quit()
        return True
        
    except Exception as e:
        print(f'❌ Test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print('🕵️ TESTING WITH BOT DETECTION EVASION')
    print('=' * 60)
    success = test_with_evasion()
    print(f'\nTest result: {"COMPLETED" if success else "FAILED"}')
