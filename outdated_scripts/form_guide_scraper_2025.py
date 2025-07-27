#!/usr/bin/env python3
"""
2025 Form Guide Scraper for thedogs.com.au
Downloads CSV form guides for 2025 races and saves them to the unprocessed folder
"""

import os
import re
import pandas as pd
import asyncio
import json
import requests
from datetime import datetime, timedelta
import sys
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
import time
import random

class FormGuideScraper2025:
    def __init__(self):
        self.base_url = "https://www.thedogs.com.au"
        self.download_dir = "./form_guides/unprocessed"
        self.processed_dir = "./form_guides/processed"
        
        # Ensure directories exist
        os.makedirs(self.download_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Track what we've already downloaded
        self.existing_files = set()
        self.load_existing_files()
        
        # Common venue mappings
        self.venue_map = {
            'wentworth-park': 'W_PK',
            'albion-park': 'APWE',
            'angle-park': 'AP_K',
            'sandown': 'SAN',
            'the-meadows': 'MEA',
            'ballarat': 'BAL',
            'bendigo': 'BEN',
            'cannington': 'CANN',
            'dapto': 'DAPT',
            'geelong': 'GEE',
            'gawler': 'GAWL',
            'horsham': 'HOR',
            'richmond': 'RICH',
            'sale': 'SAL',
            'traralgon': 'TRA',
            'warrnambool': 'WAR',
            'northam': 'NOR',
            'temora': 'TEMA',
            'hobart': 'HOBT',
            'rockhampton': 'ROCK',
            'darwin': 'DARW',
            'murray-bridge': 'MURR',
            'mount-gambier': 'MOUNT',
            'mandurah': 'MAND',
            'shepparton': 'SHEP',
            'warragul': 'WARR',
            'casino': 'CASO',
            'gunnedah': 'GUNN',
            'capalaba': 'CAPA',
            'bathurst': 'BATT',
            'healesville': 'HEA',
            'the-gardens': 'GRDN',
            'ipswich': 'IPFR'
        }
        
        self.driver = None
        
    def load_existing_files(self):
        """Load list of existing files to avoid duplicates"""
        for directory in [self.download_dir, self.processed_dir]:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    if file.endswith('.csv'):
                        self.existing_files.add(file)
        print(f"📋 Found {len(self.existing_files)} existing form guide files")
        
    def setup_driver(self):
        """Setup Chrome driver with download preferences"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
        
        # Set download preferences
        prefs = {
            "download.default_directory": os.path.abspath(self.download_dir),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            print("✅ Chrome driver initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize Chrome driver: {e}")
            return False
    
    def get_2025_race_dates(self):
        """Get list of 2025 dates to check for races - prioritize recent dates"""
        dates = []
        
        # Start from current date and go backwards, then forwards
        today = datetime.now()
        
        # Check last 30 days first (most likely to have races)
        for i in range(30):
            check_date = today - timedelta(days=i)
            if check_date.year == 2025:
                dates.append(check_date)
        
        # Then check next 30 days
        for i in range(1, 31):
            check_date = today + timedelta(days=i)
            if check_date.year == 2025:
                dates.append(check_date)
        
        # Finally add remaining dates from 2025
        start_date = datetime(2025, 1, 1)
        end_date = min(datetime(2025, 12, 31), datetime.now() + timedelta(days=365))
        
        current_date = start_date
        while current_date <= end_date:
            if current_date not in dates:
                dates.append(current_date)
            current_date += timedelta(days=1)
            
        # Remove duplicates and sort by most recent first
        dates = sorted(list(set(dates)), reverse=True)
        
        print(f"📅 Will check {len(dates)} dates for 2025 races (starting with most recent)")
        return dates
    
    def scrape_date_page(self, date):
        """Scrape a specific date page to find race meetings"""
        date_str = date.strftime('%Y-%m-%d')
        url = f"{self.base_url}/racing/{date_str}"
        
        print(f"🔍 Checking {date_str}...")
        
        try:
            self.driver.get(url)
            time.sleep(random.uniform(1, 2))  # Shorter delay
            
            # Check page source for race links
            page_source = self.driver.page_source
            
            # Look for race URLs in page source
            race_pattern = rf'/racing/[^/]+/{date_str}/\d+/?"'
            race_matches = re.findall(race_pattern, page_source)
            
            if race_matches:
                # Clean up the URLs
                race_pages = []
                for match in race_matches:
                    clean_url = match.replace('"', '')
                    if not clean_url.startswith('http'):
                        clean_url = f"{self.base_url}{clean_url}"
                    race_pages.append(clean_url)
                
                # Remove duplicates
                race_pages = list(set(race_pages))
                print(f"   ✅ Found {len(race_pages)} race pages for {date_str}")
                return race_pages
            
            # Fallback: try selenium selectors
            race_links = []
            selectors = [
                "a[href*='/racing/']",
                "a[href*='" + date_str + "']"
            ]
            
            for selector in selectors:
                try:
                    links = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for link in links:
                        href = link.get_attribute('href')
                        if href and '/racing/' in href and date_str in href and re.search(r'/\d+/?$', href):
                            race_links.append(href)
                except:
                    continue
            
            if race_links:
                race_pages = list(set(race_links))
                print(f"   ✅ Found {len(race_pages)} race pages for {date_str} (fallback)")
                return race_pages
            else:
                print(f"   ⚪ No races found for {date_str}")
                return []
                
        except Exception as e:
            print(f"   ❌ Error checking {date_str}: {e}")
            return []
    
    def extract_race_info(self, race_url):
        """Extract race information from URL"""
        # Pattern: /racing/{venue}/{date}/{race_number}/
        pattern = r'/racing/([^/]+)/(\d{4}-\d{2}-\d{2})/(\d+)/?'
        match = re.search(pattern, race_url)
        
        if match:
            venue = match.group(1)
            date_str = match.group(2)
            race_number = match.group(3)
            
            # Convert venue to our code
            venue_code = self.venue_map.get(venue, venue.upper())
            
            # Convert date
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            date_formatted = date_obj.strftime('%d %B %Y')
            
            return {
                'venue': venue,
                'venue_code': venue_code,
                'date': date_formatted,
                'race_number': race_number,
                'filename': f"Race {race_number} - {venue_code} - {date_formatted}.csv"
            }
        
        return None
    
    def download_form_guide(self, race_url):
        """Download form guide CSV for a specific race"""
        race_info = self.extract_race_info(race_url)
        if not race_info:
            print(f"   ❌ Could not extract race info from {race_url}")
            return False
        
        filename = race_info['filename']
        
        # Check if we already have this file
        if filename in self.existing_files:
            print(f"   ⚪ Already have {filename}")
            return True
        
        print(f"   🔄 Downloading {filename}...")
        
        try:
            # Navigate to race page
            self.driver.get(race_url)
            time.sleep(random.uniform(2, 4))
            
            # Look for form guide download link
            download_selectors = [
                "a[href*='form-guide']",
                "a[href*='csv']",
                "a[download*='csv']",
                ".form-guide-download",
                ".csv-download",
                "a[href*='download']"
            ]
            
            downloaded = False
            for selector in download_selectors:
                try:
                    download_links = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for link in download_links:
                        href = link.get_attribute('href')
                        if href and ('csv' in href.lower() or 'form' in href.lower()):
                            print(f"      🔗 Found download link: {href}")
                            
                            # Click the download link
                            self.driver.execute_script("arguments[0].click();", link)
                            time.sleep(3)  # Wait for download to start
                            
                            # Check if file was downloaded
                            if self.wait_for_download(filename):
                                self.existing_files.add(filename)
                                downloaded = True
                                print(f"      ✅ Downloaded {filename}")
                                break
                            
                except Exception as e:
                    continue
                    
                if downloaded:
                    break
            
            if not downloaded:
                # Try direct CSV download URL construction
                csv_url = f"{race_url}form-guide.csv"
                try:
                    response = requests.get(csv_url, timeout=10)
                    if response.status_code == 200 and 'csv' in response.headers.get('content-type', '').lower():
                        filepath = os.path.join(self.download_dir, filename)
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                        self.existing_files.add(filename)
                        print(f"      ✅ Downloaded {filename} via direct URL")
                        return True
                except:
                    pass
                
                print(f"      ❌ Could not download {filename}")
                return False
            
            return downloaded
            
        except Exception as e:
            print(f"   ❌ Error downloading {filename}: {e}")
            return False
    
    def wait_for_download(self, expected_filename, timeout=30):
        """Wait for file to be downloaded"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check if file exists
            filepath = os.path.join(self.download_dir, expected_filename)
            if os.path.exists(filepath):
                return True
            
            # Check for any new CSV files
            for file in os.listdir(self.download_dir):
                if file.endswith('.csv') and file not in self.existing_files:
                    # Rename to expected filename
                    old_path = os.path.join(self.download_dir, file)
                    new_path = os.path.join(self.download_dir, expected_filename)
                    try:
                        os.rename(old_path, new_path)
                        return True
                    except:
                        pass
            
            time.sleep(1)
        
        return False
    
    def scrape_2025_form_guides(self):
        """Main method to scrape all 2025 form guides"""
        print("🚀 Starting 2025 Form Guide Scraper")
        print("=" * 60)
        
        if not self.setup_driver():
            return
        
        try:
            dates = self.get_2025_race_dates()
            total_downloaded = 0
            dates_checked = 0
            
            for date in dates:
                dates_checked += 1
                race_pages = self.scrape_date_page(date)
                
                if race_pages:
                    for race_url in race_pages:
                        try:
                            if self.download_form_guide(race_url):
                                total_downloaded += 1
                            
                            # Small delay between downloads
                            time.sleep(random.uniform(1, 2))
                            
                        except Exception as e:
                            print(f"   ❌ Error with {race_url}: {e}")
                            continue
                
                # Progress update
                if dates_checked % 10 == 0:
                    print(f"📊 Progress: {dates_checked}/{len(dates)} dates checked, {total_downloaded} files downloaded")
                
                # Shorter delay between dates
                time.sleep(random.uniform(1, 2))
            
            print(f"\n🎉 Scraping complete!")
            print(f"📊 Total new files downloaded: {total_downloaded}")
            print(f"📁 Files saved to: {self.download_dir}")
            
        except Exception as e:
            print(f"❌ Critical error: {e}")
        finally:
            if self.driver:
                self.driver.quit()
                print("🔄 Chrome driver closed")

if __name__ == "__main__":
    scraper = FormGuideScraper2025()
    scraper.scrape_2025_form_guides()
