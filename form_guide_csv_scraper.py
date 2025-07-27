#!/usr/bin/env python3
"""
Form Guide CSV Scraper for thedogs.com.au
=========================================

This script downloads CSV form guides from race pages on thedogs.com.au
and saves them to the unprocessed folder for further analysis.

The CSV files contain individual greyhound form data including:
- Dog Name, Sex, Placing, Box, Weight, Distance, Date, Track, Grade
- Time, Win Time, Bonus, First Split, Margin, PIR, Starting Price

Usage: python3 form_guide_csv_scraper.py

Author: AI Assistant
Date: July 11, 2025
"""

import os
import sys
import requests
import time
import random
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import re
from pathlib import Path
import sqlite3

class FormGuideCsvScraper:
    def __init__(self):
        self.base_url = "https://www.thedogs.com.au"
        self.unprocessed_dir = "./unprocessed"
        self.download_dir = "./form_guides/downloaded"
        self.database_path = "./databases/greyhound_racing.db"
        
        # Create directories
        os.makedirs(self.unprocessed_dir, exist_ok=True)
        os.makedirs(self.download_dir, exist_ok=True)
        
        # Track collected races across all directories
        self.collected_races = set()  # Stores (date, venue, race_number) tuples
        self.completed_dates = set()  # Stores dates that have been fully downloaded
        self.existing_files = set()   # Stores filenames for backup checking
        self.load_collected_races()
        
        # Note: We collect all historical races (previous day or earlier) for training data
        
        # Setup session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Venue mapping
        self.venue_map = {
            'angle-park': 'AP_K',
            'sandown': 'SAN',
            'warrnambool': 'WAR',
            'bendigo': 'BEN',
            'geelong': 'GEE',
            'ballarat': 'BAL',
            'horsham': 'HOR',
            'traralgon': 'TRA',
            'dapto': 'DAPT',
            'wentworth-park': 'W_PK',
            'albion-park': 'APWE',
            'cannington': 'CANN',
            'the-meadows': 'MEA',
            'healesville': 'HEA',
            'sale': 'SAL',
            'richmond': 'RICH',
            'murray-bridge': 'MURR',
            'gawler': 'GAWL',
            'mount-gambier': 'MOUNT',
            'northam': 'NOR',
            'mandurah': 'MAND'
        }
        
        print("🏁 Form Guide CSV Scraper initialized")
        print(f"📂 Unprocessed directory: {self.unprocessed_dir}")
        print(f"📂 Download directory: {self.download_dir}")
        print(f"🎯 Target: Historical races (previous day or earlier) for training data")
    
    def load_collected_races(self):
        """Load all collected races from all directories to avoid re-downloading"""
        self.collected_races.clear()
        self.existing_files.clear()
        
        # Check all directories where race files might exist
        directories = [
            "./unprocessed",
            "./form_guides/downloaded", 
            "./form_guides/processed",
            "./historical_races",
            "./processed"
        ]
        
        total_files = 0
        
        for directory in directories:
            if os.path.exists(directory):
                files = [f for f in os.listdir(directory) if f.endswith('.csv')]
                total_files += len(files)
                
                for filename in files:
                    self.existing_files.add(filename)
                    
                    # Parse race info from filename: "Race {num} - {venue} - {date}.csv"
                    match = re.match(r'Race (\d+) - ([A-Z_]+) - (\d{1,2} \w+ \d{4})\.csv', filename)
                    if match:
                        race_number, venue, date_str = match.groups()
                        try:
                            # Convert date to standard format
                            date_obj = datetime.strptime(date_str, '%d %B %Y')
                            date_formatted = date_obj.strftime('%Y-%m-%d')
                            
                            # Store race identifier
                            race_id = (date_formatted, venue, race_number)
                            self.collected_races.add(race_id)
                            
                        except ValueError:
                            # Skip files with unparseable dates
                            continue
        
        print(f"📋 Loaded {len(self.collected_races)} unique races from {total_files} files across all directories")

        # Analyze collected races to identify fully collected dates
        race_dates = {}
        for (date, venue, race_number) in self.collected_races:
            if date not in race_dates:
                race_dates[date] = {'races': set(), 'venues': set()}
            race_dates[date]['races'].add((venue, race_number))
            race_dates[date]['venues'].add(venue)

        # A date is considered "completed" if it has substantial coverage
        # We'll be conservative: dates with 8+ races from multiple venues are likely complete
        self.completed_dates = set()
        for date, data in race_dates.items():
            race_count = len(data['races'])
            venue_count = len(data['venues'])
            
            # Conservative criteria for completion:
            # - At least 8 races collected
            # - At least 3 different venues (indicating broad coverage)
            # OR
            # - At least 15 races collected (likely comprehensive)
            if (race_count >= 8 and venue_count >= 3) or race_count >= 15:
                self.completed_dates.add(date)
        
        if self.completed_dates:
            print(f"📅 Completed dates ({len(self.completed_dates)}): {sorted(list(self.completed_dates))}")
        else:
            print(f"📅 No fully completed dates identified yet")
        print(f"   • Directories checked: {', '.join([d for d in directories if os.path.exists(d)])}")
    
    def is_race_already_collected(self, race_date, venue, race_number):
        """Check if a race has already been collected in any directory"""
        race_id = (race_date, venue, str(race_number))
        return race_id in self.collected_races
    
    def load_processed_races(self):
        """Load processed races from the database"""
        processed_races = set()
        
        if not os.path.exists(self.database_path):
            print(f"⚠️ Database not found at {self.database_path}")
            return processed_races
        
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Check if race_metadata table exists (enhanced database)
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='race_metadata'")
            if cursor.fetchone():
                # Enhanced database - get races from race_metadata table
                cursor.execute("""
                    SELECT DISTINCT race_date, venue, race_number 
                    FROM race_metadata
                    WHERE race_date IS NOT NULL AND venue IS NOT NULL AND race_number IS NOT NULL
                """)
                
                for row in cursor.fetchall():
                    race_date, venue, race_number = row
                    # Convert date format for comparison
                    try:
                        date_obj = datetime.strptime(race_date, '%Y-%m-%d')
                        formatted_date = date_obj.strftime('%Y-%m-%d')
                        processed_races.add((formatted_date, venue, str(race_number)))
                    except ValueError:
                        # Try alternative date format
                        try:
                            date_obj = datetime.strptime(race_date, '%d %B %Y')
                            formatted_date = date_obj.strftime('%Y-%m-%d')
                            processed_races.add((formatted_date, venue, str(race_number)))
                        except ValueError:
                            continue
            else:
                # Original database - get races from race_results table
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='race_results'")
                if cursor.fetchone():
                    cursor.execute("""
                        SELECT DISTINCT race_date, venue, race_number 
                        FROM race_results
                        WHERE race_date IS NOT NULL AND venue IS NOT NULL AND race_number IS NOT NULL
                    """)
                    
                    for row in cursor.fetchall():
                        race_date, venue, race_number = row
                        # Convert date format for comparison
                        try:
                            date_obj = datetime.strptime(race_date, '%Y-%m-%d')
                            formatted_date = date_obj.strftime('%Y-%m-%d')
                            processed_races.add((formatted_date, venue, str(race_number)))
                        except ValueError:
                            continue
            
            conn.close()
            
        except Exception as e:
            print(f"⚠️ Error loading processed races from database: {e}")
        
        return processed_races
    
    def refresh_existing_files(self):
        """Refresh list of existing files to avoid duplicates"""
        self.existing_files.clear()
        directories = [self.unprocessed_dir, self.download_dir, "./form_guides/processed"]
        
        for directory in directories:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    if file.endswith('.csv'):
                        self.existing_files.add(file)
        
        print(f"📋 Refreshed list: {len(self.existing_files)} existing form guide files")

    def load_existing_files(self):
        """Initial load of existing files to avoid duplicates"""
        self.refresh_existing_files()
    
    def file_already_exists(self, filename, race_info):
        """Check if file already exists with various filename patterns"""
        # Check exact filename match
        if filename in self.existing_files:
            return True
        
        # Check alternative filename patterns that might exist
        race_num = race_info['race_number']
        venue = race_info['venue']
        date = race_info['date']
        
        # Alternative patterns to check
        alternatives = [
            f"Race {race_num} - {venue} - {date}.csv",
            f"Race {race_num} - {venue} - {date}_*.csv",  # Files with timestamps
            f"Race{race_num}_{venue}_{date}.csv",
            f"R{race_num}_{venue}_{date}.csv",
        ]
        
        # Check if any alternative pattern matches existing files
        for existing_file in self.existing_files:
            for pattern in alternatives:
                if pattern.replace('*', '') in existing_file:
                    return True
        
        return False
    
    def handle_file_exists_interaction(self, filename, race_info):
        """Handle user interaction when file already exists"""
        race_num = race_info['race_number']
        venue = race_info['venue']
        date = race_info['date']
        
        print(f"\n⚠️  File already exists: {filename}")
        print(f"   Race: {venue} Race {race_num} on {date}")
        print(f"\n   What would you like to do?")
        print(f"   [s] Skip this race and continue with next")
        print(f"   [o] Overwrite the existing file")
        print(f"   [q] Quit the scraping process")
        print(f"   [a] Skip all remaining duplicates automatically")
        
        while True:
            try:
                choice = input("\n   Enter your choice (s/o/q/a): ").lower().strip()
                
                if choice in ['s', 'skip']:
                    print(f"   ✅ Skipping race: {venue} Race {race_num}")
                    return 'skip'
                elif choice in ['o', 'overwrite']:
                    print(f"   🔄 Will overwrite: {filename}")
                    return 'overwrite'
                elif choice in ['q', 'quit', 'exit']:
                    print(f"   🛑 Exiting scraping process...")
                    return 'quit'
                elif choice in ['a', 'auto', 'all']:
                    print(f"   ⏭️  Will skip all remaining duplicates automatically")
                    return 'auto_skip'
                else:
                    print(f"   ❌ Invalid choice. Please enter 's' (skip), 'o' (overwrite), 'q' (quit), or 'a' (auto-skip)")
            except KeyboardInterrupt:
                print(f"\n   🛑 Process interrupted by user")
                return 'quit'
            except EOFError:
                print(f"\n   🛑 Input stream closed")
                return 'quit'
    
    def move_downloaded_to_unprocessed(self):
        """Move files from download directory to unprocessed for analysis"""
        if not os.path.exists(self.download_dir):
            return 0
        
        moved_count = 0
        download_files = [f for f in os.listdir(self.download_dir) if f.endswith('.csv')]
        
        for filename in download_files:
            download_path = os.path.join(self.download_dir, filename)
            unprocessed_path = os.path.join(self.unprocessed_dir, filename)
            
            # Check if file doesn't already exist in unprocessed
            if not os.path.exists(unprocessed_path):
                try:
                    # Copy file to unprocessed directory
                    with open(download_path, 'r', encoding='utf-8') as src:
                        content = src.read()
                    
                    with open(unprocessed_path, 'w', encoding='utf-8') as dst:
                        dst.write(content)
                    
                    moved_count += 1
                    print(f"   📞 Moved to unprocessed: {filename}")
                    
                except Exception as e:
                    print(f"   ⚠️ Error moving {filename}: {e}")
        
        if moved_count > 0:
            print(f"✅ Moved {moved_count} files from download to unprocessed directory")
        
        return moved_count
    
    def is_race_processed(self, race_date, track, race_number):
        """Check if a race has been processed and is in the database"""
        # Convert track name to match database format
        track_code = None
        
        # Try to find matching track code
        for venue_key, venue_code in self.venue_map.items():
            if venue_key.replace('-', ' ').lower() in track.lower() or venue_code.lower() == track.lower():
                track_code = venue_code
                break
        
        if not track_code:
            track_code = track  # Use original track name as fallback
        
        # Check if race is in processed races
        return (race_date, track_code, str(race_number)) in self.processed_races
    
    def get_race_dates(self, days_back=30, days_forward=0):
        """Get list of dates to check for races - historical races only (previous day or earlier)"""
        dates = []
        today = datetime.now().date()
        
        # Only check past dates (previous day or earlier) for training data
        for i in range(1, days_back + 1):  # Start from 1 to exclude today
            check_date = today - timedelta(days=i)
            dates.append(check_date)
        
        # No future dates - we only want historical races for training
        
        return sorted(dates, reverse=True)
    
    def find_race_urls(self, date):
        """Find race URLs for a specific date"""
        date_str = date.strftime('%Y-%m-%d')
        base_url = f"{self.base_url}/racing/{date_str}"
        
        print(f"🔍 Checking races for {date_str}...")
        
        try:
            response = self.session.get(base_url, timeout=30)
            
            if response.status_code != 200:
                print(f"   ❌ Failed to access {base_url}: {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find race links
            race_links = []
            
            # Look for race links in various formats
            selectors = [
                'a[href*="/racing/"]',
                'a[href*="{date_str}"]'.format(date_str=date_str),
                '.race-link',
                '.race-card a'
            ]
            
            for selector in selectors:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href')
                    if href and date_str in href and href.count('/') >= 4:
                        # Extract race number from URL
                        parts = href.split('/')
                        if len(parts) >= 5 and parts[-2].isdigit():
                            full_url = href if href.startswith('http') else f"{self.base_url}{href}"
                            race_links.append(full_url)
            
            # Remove duplicates
            unique_links = list(set(race_links))
            
            if unique_links:
                print(f"   ✅ Found {len(unique_links)} race URLs for {date_str}")
            else:
                print(f"   ⚪ No races found for {date_str}")
            
            return unique_links
            
        except Exception as e:
            print(f"   ❌ Error checking {date_str}: {e}")
            return []
    
    def download_csv_from_race_page(self, race_url):
        """Download CSV form guide from a race page"""
        try:
            print(f"🔄 Processing: {race_url}")
            
            # Get race page
            response = self.session.get(race_url, timeout=30)
            
            if response.status_code != 200:
                print(f"   ❌ Failed to access race page: {response.status_code}")
                return False
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract race information for filename
            race_info = self.extract_race_info(soup, race_url)
            
            if not race_info:
                print(f"   ❌ Could not extract race information")
                return False
            
            # Check if this is a historical race (previous day or earlier)
            race_date_formatted = datetime.strptime(race_info['date'], '%d %B %Y').strftime('%Y-%m-%d')
            race_date_obj = datetime.strptime(race_date_formatted, '%Y-%m-%d').date()
            today = datetime.now().date()
            
            if race_date_obj >= today:
                print(f"   ⚪ Skipping future/today race: {race_info['venue']} Race {race_info['race_number']} on {race_info['date']}")
                return False
            
            print(f"   ✅ Historical race found: {race_info['venue']} Race {race_info['race_number']} on {race_info['date']}")
            
            # Check if race has already been collected (primary check)
            if self.is_race_already_collected(race_date_formatted, race_info['venue'], race_info['race_number']):
                print(f"   ⭕ Race already collected: {race_info['venue']} Race {race_info['race_number']} on {race_info['date']}")
                return False
            
            # Generate filename
            filename = f"Race {race_info['race_number']} - {race_info['venue']} - {race_info['date']}.csv"
            
            # Check if already exists (backup check for files with different naming)
            if self.file_already_exists(filename, race_info):
                # Handle user interaction for existing files
                if not hasattr(self, 'auto_skip_duplicates'):
                    self.auto_skip_duplicates = False
                
                if self.auto_skip_duplicates:
                    print(f"   ⏭️  Auto-skipping duplicate: {filename}")
                    return False
                
                choice = self.handle_file_exists_interaction(filename, race_info)
                
                if choice == 'skip':
                    return False
                elif choice == 'overwrite':
                    print(f"   🔄 Proceeding to overwrite: {filename}")
                    # Continue with download to overwrite
                elif choice == 'quit':
                    print(f"   🛑 User chose to quit. Exiting scraper...")
                    exit(0)
                elif choice == 'auto_skip':
                    self.auto_skip_duplicates = True
                    print(f"   ⏭️  Auto-skip mode enabled for remaining duplicates")
                    return False
            
            # Look for CSV download link
            csv_url = self.find_csv_download_link(soup, race_url)
            
            if not csv_url:
                print(f"   ❌ No CSV download link found")
                return False
            
            # Download CSV
            return self.download_csv_file(csv_url, filename)
            
        except Exception as e:
            print(f"   ❌ Error processing race page: {e}")
            return False
    
    def extract_race_info(self, soup, race_url):
        """Extract race information from the page"""
        try:
            # Extract race number from URL
            url_parts = race_url.split('/')
            race_number = None
            venue = None
            date = None
            
            # Find race number in URL
            for i, part in enumerate(url_parts):
                if part.isdigit() and i > 0:
                    race_number = part
                    break
            
            # Extract venue from URL
            for venue_key, venue_code in self.venue_map.items():
                if venue_key in race_url:
                    venue = venue_code
                    break
            
            # Extract date from URL
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', race_url)
            if date_match:
                date_obj = datetime.strptime(date_match.group(1), '%Y-%m-%d')
                date = date_obj.strftime('%d %B %Y')
            
            # Try to extract from page content if not found in URL
            if not venue:
                venue_selectors = ['.venue-name', '.track-name', 'h1', '.race-header']
                for selector in venue_selectors:
                    element = soup.select_one(selector)
                    if element:
                        text = element.get_text(strip=True)
                        for venue_key, venue_code in self.venue_map.items():
                            if venue_key.replace('-', ' ').lower() in text.lower():
                                venue = venue_code
                                break
                        if venue:
                            break
            
            if race_number and venue and date:
                return {
                    'race_number': race_number,
                    'venue': venue,
                    'date': date
                }
            
            return None
            
        except Exception as e:
            print(f"   ❌ Error extracting race info: {e}")
            return None
    
    def find_csv_download_link(self, soup, race_url):
        """Find CSV download link on the page using expert-form method"""
        # Try the expert-form page method
        # Remove query parameters and add expert-form
        base_race_url = race_url.split('?')[0]  # Remove ?trial=false etc.
        expert_form_url = f"{base_race_url}/expert-form"
        print(f"   🔍 Checking expert-form page: {expert_form_url}")
        
        try:
            # Step 1: Get the expert-form page
            response = self.session.get(expert_form_url, timeout=10)
            if response.status_code != 200:
                print(f"   ⚠️ Expert-form page not accessible: {response.status_code}")
                return None
                
            expert_soup = BeautifulSoup(response.content, 'html.parser')
            
            # Step 2: Find the form with CSV export capability
            form = None
            for f in expert_soup.find_all('form'):
                if f.find('input', {'name': 'export_csv'}) or f.find('button', {'name': 'export_csv'}):
                    form = f
                    break
            
            if not form:
                print(f"   ⚠️ No CSV export form found on expert-form page")
                return None
                
            print(f"   ✅ Found CSV export form")
            
            # Step 3: Extract form data
            form_action = form.get('action')
            form_method = form.get('method', 'GET').upper()
            
            # Build form data
            form_data = {}
            
            # Get all form inputs
            for input_elem in form.find_all(['input', 'select', 'textarea']):
                name = input_elem.get('name')
                if name:
                    input_type = input_elem.get('type', 'text')
                    
                    if input_type == 'checkbox':
                        if input_elem.get('checked'):
                            form_data[name] = input_elem.get('value', 'on')
                    elif input_type == 'radio':
                        if input_elem.get('checked'):
                            form_data[name] = input_elem.get('value', '')
                    elif input_type == 'submit':
                        pass
                    elif input_type == 'hidden':
                        form_data[name] = input_elem.get('value', '')
                    else:
                        form_data[name] = input_elem.get('value', '')
            
            # Add the CSV export parameter
            form_data['export_csv'] = 'true'
            
            # Determine the target URL
            if form_action:
                if form_action.startswith('/'):
                    target_url = f"{self.base_url}{form_action}"
                elif form_action.startswith('http'):
                    target_url = form_action
                else:
                    target_url = f"{self.base_url}/{form_action}"
            else:
                target_url = expert_form_url
            
            print(f"   🔗 Submitting form to: {target_url}")
            
            # Step 4: Submit the form to get the download URL
            if form_method == 'POST':
                form_response = self.session.post(target_url, data=form_data, timeout=10)
            else:
                form_response = self.session.get(target_url, params=form_data, timeout=10)
            
            if form_response.status_code == 200:
                # The response should contain the actual download URL
                download_url = form_response.text.strip()
                
                if download_url.startswith('http'):
                    print(f"   ✅ Got CSV download URL: {download_url}")
                    return download_url
                else:
                    print(f"   ⚠️ Unexpected response: {download_url[:100]}")
                    return None
            else:
                print(f"   ❌ Form submission failed: {form_response.status_code}")
                return None
                
        except Exception as e:
            print(f"   ⚠️ Error with expert-form method: {e}")
        
        # Fallback to regular page CSV search (original method)
        print(f"   🔍 Fallback: Checking regular page for CSV links")
        
        # Look for common CSV download patterns
        csv_selectors = [
            'a[href*="csv"]',
            'a[href*="export"]',
            'a[href*="download"]',
            '.csv-download',
            '.export-csv',
            '.download-csv'
        ]
        
        for selector in csv_selectors:
            elements = soup.select(selector)
            for element in elements:
                href = element.get('href')
                if href:
                    # Make URL absolute
                    if href.startswith('/'):
                        href = f"{self.base_url}{href}"
                    elif not href.startswith('http'):
                        href = f"{self.base_url}/{href}"
                    
                    return href
        
        # Try constructing common CSV URLs
        common_patterns = [
            f"{race_url}/csv",
            f"{race_url}/export",
            f"{race_url}/download",
            f"{race_url}.csv",
            f"{race_url}/form-guide.csv"
        ]
        
        for pattern in common_patterns:
            try:
                response = self.session.head(pattern, timeout=10)
                if response.status_code == 200:
                    content_type = response.headers.get('content-type', '').lower()
                    if 'csv' in content_type or 'text' in content_type:
                        return pattern
            except:
                continue
        
        return None
    
    def download_csv_file(self, csv_url, filename):
        """Download the CSV file"""
        try:
            print(f"   📥 Downloading: {filename}")
            
            response = self.session.get(csv_url, timeout=30)
            
            if response.status_code != 200:
                print(f"   ❌ Failed to download CSV: {response.status_code}")
                return False
            
            # Check if content looks like CSV
            content = response.text
            if not content.strip():
                print(f"   ❌ Empty CSV content")
                return False
            
            # Basic CSV validation
            lines = content.strip().split('\n')
            if len(lines) < 2:
                print(f"   ❌ CSV has insufficient data")
                return False
            
            # Check for expected headers
            first_line = lines[0].lower()
            if not any(header in first_line for header in ['dog name', 'dog', 'runner', 'name']):
                print(f"   ❌ CSV doesn't appear to be a form guide")
                return False
            
            # Save to download directory first (for backup/tracking)
            download_filepath = os.path.join(self.download_dir, filename)
            with open(download_filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Move to unprocessed directory for analysis
            unprocessed_filepath = os.path.join(self.unprocessed_dir, filename)
            with open(unprocessed_filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Add to existing files list to prevent future duplicates
            self.existing_files.add(filename)
            
            # Extract race info from filename and add to collected races
            match = re.match(r'Race (\d+) - ([A-Z_]+) - (\d{1,2} \w+ \d{4})\.csv', filename)
            if match:
                race_number, venue, date_str = match.groups()
                try:
                    date_obj = datetime.strptime(date_str, '%d %B %Y')
                    date_formatted = date_obj.strftime('%Y-%m-%d')
                    race_id = (date_formatted, venue, race_number)
                    self.collected_races.add(race_id)
                except ValueError:
                    pass  # Skip if date parsing fails
            
            print(f"   ✅ Successfully downloaded and moved to unprocessed: {filename}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error downloading CSV: {e}")
            return False
    
    def run_scraper(self):
        """Run the main scraping process"""
        print("🚀 STARTING FORM GUIDE CSV SCRAPER (HISTORICAL RACES)")
        print("=" * 60)
        
        # Show current status
        unprocessed_count = len([f for f in os.listdir(self.unprocessed_dir) if f.endswith('.csv')]) if os.path.exists(self.unprocessed_dir) else 0
        download_count = len([f for f in os.listdir(self.download_dir) if f.endswith('.csv')]) if os.path.exists(self.download_dir) else 0
        print(f"🧹 Clean Status: {unprocessed_count} files in unprocessed queue, {download_count} files in download backup")
        print(f"🎯 Target: Historical races (previous day or earlier) for training data")
        
        # DISABLED: First, move any existing files from download to unprocessed
        # This was causing an endless loop by repeatedly moving the same 847 files
        # print("📞 Moving existing downloaded files to unprocessed...")
        # existing_moved = self.move_downloaded_to_unprocessed()
        # if existing_moved > 0:
        #     print(f"✅ Moved {existing_moved} existing files")
        # else:
        #     print("⭕ No existing files to move")
        print("⚠️ File moving from downloaded to unprocessed has been disabled to prevent loops")
        
        dates = self.get_race_dates()
        total_downloaded = 0
        
        for date in dates:
            if date.strftime('%Y-%m-%d') in self.completed_dates:
                print(f"⭕ Date already fully collected: {date.strftime('%Y-%m-%d')}")
                continue

            race_urls = self.find_race_urls(date)
            
            for race_url in race_urls:
                if self.download_csv_from_race_page(race_url):
                    total_downloaded += 1
                
                # Add delay between requests
                time.sleep(random.uniform(1, 3))
            
            # Refresh existing files after each date to avoid duplicates
            self.refresh_existing_files()
        
        print(f"\n🎯 SCRAPING COMPLETE")
        print("=" * 60)
        print(f"📊 Total CSV files downloaded: {total_downloaded}")
        print(f"📂 Files ready for analysis in: {self.unprocessed_dir}")
        print(f"📂 Files backed up in: {self.download_dir}")
        
        # Show current file counts
        unprocessed_count = len([f for f in os.listdir(self.unprocessed_dir) if f.endswith('.csv')]) if os.path.exists(self.unprocessed_dir) else 0
        download_count = len([f for f in os.listdir(self.download_dir) if f.endswith('.csv')]) if os.path.exists(self.download_dir) else 0
        
        print(f"📁 Current file counts:")
        print(f"   • Unprocessed (ready for analysis): {unprocessed_count} files")
        print(f"   • Downloaded (backup): {download_count} files")
        
        # Final clean status
        print(f"\n🧹 Final Clean Status: {unprocessed_count} files ready for processing")
        
        if total_downloaded > 0:
            print(f"\n💡 Next steps:")
            print(f"   • Run analysis on files in {self.unprocessed_dir}")
            print(f"   • Files are automatically moved to unprocessed for analysis")


def main():
    """Main function"""
    scraper = FormGuideCsvScraper()
    scraper.run_scraper()


if __name__ == "__main__":
    main()
