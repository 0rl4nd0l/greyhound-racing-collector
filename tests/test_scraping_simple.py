
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

def test_single_race_scraping():
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        print('✅ Chrome driver initialized successfully')
        
        # Test URL formation and scraping
        test_url = 'https://www.thedogs.com.au/racing/2024-07-28/warrnambool/race-12'
        print(f'🌐 Testing URL: {test_url}')
        
        driver.get(test_url)
        time.sleep(3)
        
        # Look for winner information
        winner_elements = driver.find_elements(By.CSS_SELECTOR, '.winner, .first-place, [data-position="1"], .pos-1')
        if winner_elements:
            winner_text = winner_elements[0].text
            print(f'🏆 Found winner element: {winner_text[:50]}...')
        else:
            print('⚠️  No winner element found with standard selectors')
            
        # Check page content
        page_source = driver.page_source
        if 'winner' in page_source.lower() or 'first' in page_source.lower():
            print('✅ Page contains winner-related content')
        else:
            print('⚠️  Page may not contain race results')
            
        driver.quit()
        return True
        
    except Exception as e:
        print(f'❌ Scraping test failed: {e}')
        return False

if __name__ == "__main__":
    print('🧪 TESTING RACE RESULT SCRAPING')
    print('=' * 50)
    success = test_single_race_scraping()
    print(f'Test result: {"PASSED" if success else "FAILED"}')
