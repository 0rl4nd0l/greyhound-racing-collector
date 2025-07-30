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
        
        # Comprehensive venue mapping for all Australian greyhound tracks
        self.venue_map = {
            # Major metropolitan tracks
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
            'meadows': 'MEA',
            'healesville': 'HEA',
            'sale': 'SAL',
            'richmond': 'RICH',
            'richmond-straight': 'RICH_S',
            'murray-bridge': 'MURR',
            'gawler': 'GAWL',
            'mount-gambier': 'MOUNT',
            'northam': 'NOR',
            'mandurah': 'MAND',
            
            # NSW tracks
            'the-gardens': 'GARD',
            'casino': 'CAS',
            'wagga': 'WAG',
            'goulburn': 'GOUL',
            'taree': 'TAR',
            'dubbo': 'DUB',
            'grafton': 'GRAF',
            'broken-hill': 'BH',
            'lismore': 'LIS',
            'nowra': 'NOW',
            'temora': 'TEM',
            'young': 'YOU',
            'orange': 'ORA',
            'mudgee': 'MUD',
            'cowra': 'COW',
            'bathurst': 'BAT',
            'katoomba': 'KAT',
            'wollongong': 'WOL',
            'ingle-farm': 'INF',
            'bulli': 'BUL',
            'raymond-terrace': 'RAY',
            
            # QLD tracks
            'ladbrokes-q1-lakeside': 'Q1L',
            'ladbrokes-q-straight': 'QST',
            'townsville': 'TWN',
            'capalaba': 'CAP',
            'ipswich': 'IPS',
            'rockhampton': 'ROCK',
            'bundaberg': 'BUN',
            'cairns': 'CAI',
            'mackay': 'MAC',
            'toowoomba': 'TOO',
            'gold-coast': 'GC',
            'caloundra': 'CAL',
            'maroochy': 'MAR',
            
            # VIC tracks
            'shepparton': 'SHEP',
            'warragul': 'WRGL',
            'cranbourne': 'CRAN',
            'moe': 'MOE',
            'pakenham': 'PAK',
            'colac': 'COL',
            'hamilton': 'HAM',
            'portland': 'PORT',
            'ararat': 'ARA',
            'stawell': 'STA',
            'swan-hill': 'SH',
            'mildura': 'MIL',
            'echuca': 'ECH',
            'seymour': 'SEY',
            'kilmore': 'KIL',
            'wodonga': 'WOD',
            'wodonga-gvgrc': 'WOD_G',
            
            # SA tracks
            'virginia': 'VIR',
            'strathalbyn': 'STR',
            'whyalla': 'WHY',
            'port-augusta': 'PA',
            'port-pirie': 'PP',
            'glenelg': 'GLE',
            
            # WA tracks
            'albany': 'ALB',
            'geraldton': 'GER',
            'kalgoorlie': 'KAL',
            'bunbury': 'BUNB',
            'esperance': 'ESP',
            'broome': 'BRO',
            'karratha': 'KAR',
            'port-hedland': 'PH',
            'kununurra': 'KUN',
            
            # TAS tracks
            'hobart': 'HOB',
            'launceston': 'LAU',
            'devonport': 'DEV',
            
            # NT tracks
            'darwin': 'DAR',
            'alice-springs': 'AS',
            
            # ACT tracks
            'canberra': 'CANB'
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
                    
                    # Try to parse from multiple known filename formats
                    match = re.match(r'Race (\\d+) - ([A-Z_]+) - (\\d{1,2} \\w+ \\d{4})\\.csv', filename)
                    if not match:
                        match = re.match(r'\\w+_Race_(\\d+)_([A-Z_]+)_([\\d-]+)\\.csv', filename)
                    if not match:
                        match = re.match(r'Race_(\\d+)_-_([A-Z_]+)_-_([\\d_A-Za-z]+)\\.csv', filename)
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
    
    def is_valid_race_url(self, url, date_str):
        """Validate if a URL is a proper race URL with extractable race number"""
        if not url or date_str not in url:
            return False
        
        # Check URL structure
        parts = url.split('/')
        if len(parts) < 7:  # Need at least: https://www.thedogs.com.au/racing/venue/date/race_num/
            return False
        
        # Find numeric race number in URL parts
        race_number = None
        for i, part in enumerate(parts):
            if part.isdigit() and i > 4:  # Skip domain parts
                race_number = part
                break
        
        if not race_number:
            return False
        
        # Check if venue is in our mapping
        venue_found = False
        for venue_key in self.venue_map.keys():
            if venue_key in url:
                venue_found = True
                break
        
        if not venue_found:
            # Still valid if we can't map venue, but less reliable
            pass
        
        # Exclude trial URLs unless they have proper race structure
        if '?trial=true' in url and '/racing/' not in url:
            return False
        
        return True

    def find_race_urls(self, date):
        """Find race URLs for a specific date with improved filtering"""
        date_str = date.strftime('%Y-%m-%d')
        base_url = f"{self.base_url}/racing/{date_str}"
        
        print(f"🔍 Checking races for {date_str}...")
        
        try:
            response = self.session.get(base_url, timeout=30)
            
            if response.status_code != 200:
                print(f"   ❌ Failed to access {base_url}: {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all links that might be race links
            all_links = soup.find_all('a', href=True)
            race_links = []
            
            for link in all_links:
                href = link.get('href')
                if not href:
                    continue
                
                # Make URL absolute if needed
                if href.startswith('/'):
                    full_url = f"{self.base_url}{href}"
                elif href.startswith('http'):
                    full_url = href
                else:
                    continue
                
                # Apply validation filters
                if self.is_valid_race_url(full_url, date_str):
                    race_links.append(full_url)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_links = []
            for url in race_links:
                if url not in seen:
                    seen.add(url)
                    unique_links.append(url)
            
            # Filter out non-race URLs (venue-only URLs)
            filtered_links = []
            for url in unique_links:
                # Skip URLs that don't have a race number (venue-only pages)
                if '?trial=' in url and url.split('?')[0].endswith(date_str):
                    continue  # Skip venue-only trial pages
                filtered_links.append(url)
            
            if filtered_links:
                print(f"   ✅ Found {len(filtered_links)} valid race URLs for {date_str}")
                # Show sample URLs for debugging
                if len(filtered_links) > 3:
                    print(f"   📋 Sample URLs: {filtered_links[:3]}")
                else:
                    print(f"   📋 All URLs: {filtered_links}")
            else:
                print(f"   ⚪ No valid race URLs found for {date_str}")
            
            return filtered_links
            
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
        """Find CSV download link on the page using improved expert-form method"""
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
            
            # Step 2: Find the form with CSV export capability (enhanced detection)
            form = None
            csv_button_found = False
            
            # Look for forms with CSV export capability
            for f in expert_soup.find_all('form'):
                # Check for export_csv input/button
                csv_input = f.find('input', {'name': 'export_csv'})
                csv_button = f.find('button', {'name': 'export_csv'})
                
                # Also check for buttons with CSV text
                csv_text_buttons = f.find_all('button', string=lambda text: text and 'csv' in text.lower())
                csv_value_buttons = f.find_all('button', {'value': lambda val: val and 'csv' in val.lower()})
                
                if csv_input or csv_button or csv_text_buttons or csv_value_buttons:
                    form = f
                    csv_button_found = True
                    print(f"   ✅ Found CSV export form with button/input")
                    break
            
            if not form:
                print(f"   ⚠️ No CSV export form found on expert-form page")
                return None
            
            # Step 3: Extract form data and action
            form_action = form.get('action')
            form_method = form.get('method', 'GET').upper()
            
            # Determine the target URL
            if form_action:
                if form_action.startswith('/'):
                    target_url = f"{self.base_url}{form_action}"
                elif form_action.startswith('http'):
                    target_url = form_action
                else:
                    # Relative URL - construct based on expert form page
                    target_url = f"{'/'.join(expert_form_url.split('/')[:-1])}/{form_action}"
            else:
                # No action specified - use the expert form URL itself
                target_url = expert_form_url
            
            print(f"   🎯 Form action: {form_action or 'None (using current URL)'}")
            print(f"   🔗 Target URL: {target_url}")
            
            # Build form data
            form_data = {}
            
            # Get all form inputs with their current values
            for input_elem in form.find_all(['input', 'select', 'textarea']):
                name = input_elem.get('name')
                if not name:
                    continue
                    
                input_type = input_elem.get('type', 'text').lower()
                
                if input_type == 'checkbox':
                    if input_elem.get('checked'):
                        form_data[name] = input_elem.get('value', 'on')
                elif input_type == 'radio':
                    if input_elem.get('checked'):
                        form_data[name] = input_elem.get('value', '')
                elif input_type in ['submit', 'button']:
                    # Skip submit buttons unless they're the CSV export button
                    if name == 'export_csv':
                        form_data[name] = input_elem.get('value', 'Export CSV')
                elif input_type == 'hidden':
                    form_data[name] = input_elem.get('value', '')
                else:
                    # Text, email, etc. - use existing value or empty string
                    form_data[name] = input_elem.get('value', '')
            
            # Ensure CSV export is requested
            if 'export_csv' not in form_data:
                form_data['export_csv'] = 'Export CSV'
            
            print(f"   📋 Form data keys: {list(form_data.keys())}")
            
            # Step 4: Submit the form to get CSV data or download URL
            print(f"   📤 Submitting form ({form_method}) to: {target_url}")
            
            if form_method == 'POST':
                form_response = self.session.post(target_url, data=form_data, timeout=15)
            else:
                form_response = self.session.get(target_url, params=form_data, timeout=15)
            
            if form_response.status_code == 200:
                content_type = form_response.headers.get('content-type', '').lower()
                content = form_response.text.strip()
                
                # Check if we got CSV data directly
                if 'csv' in content_type or 'text/plain' in content_type:
                    # Response is CSV data - check if it looks valid
                    if content and len(content.split('\n')) > 1:
                        lines = content.split('\n')
                        first_line = lines[0].lower()
                        if any(header in first_line for header in ['dog', 'name', 'runner', 'placing']):
                            print(f"   ✅ Got CSV data directly ({len(lines)} lines)")
                            return f"data:{content}"  # Special marker for direct data
                
                # Check if response contains a download URL
                if content.startswith('http'):
                    print(f"   ✅ Got CSV download URL: {content}")
                    return content
                
                # Check if response is HTML with a download link
                if '<' in content and '>' in content:
                    response_soup = BeautifulSoup(content, 'html.parser')
                    csv_links = response_soup.find_all('a', href=True)
                    for link in csv_links:
                        href = link.get('href')
                        if href and ('csv' in href.lower() or 'export' in href.lower()):
                            if href.startswith('/'):
                                href = f"{self.base_url}{href}"
                            elif not href.startswith('http'):
                                href = f"{self.base_url}/{href}"
                            print(f"   ✅ Found CSV link in response: {href}")
                            return href
                
                print(f"   ⚠️ Unexpected response format ({len(content)} chars): {content[:100]}")
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
                    
                    print(f"   🔍 Found potential CSV link: {href}")
                    return href
        
        return None
    
    def download_csv_file(self, csv_url_or_data, filename):
        """Download the CSV file or handle direct CSV data"""
        try:
            print(f"   📥 Processing CSV for: {filename}")
            
            # Check if we have direct CSV data (marked with 'data:' prefix)
            if csv_url_or_data.startswith('data:'):
                content = csv_url_or_data[5:]  # Remove 'data:' prefix
                print(f"   📋 Using direct CSV data ({len(content.split('\n'))} lines)")
            else:
                # Download from URL
                print(f"   🌍 Downloading from URL: {csv_url_or_data}")
                response = self.session.get(csv_url_or_data, timeout=30)
                
                if response.status_code != 200:
                    print(f"   ❌ Failed to download CSV: {response.status_code}")
                    return False
                
                content = response.text
            
            # Validate content
            if not content.strip():
                print(f"   ❌ Empty CSV content")
                return False
            
            # Basic CSV validation
            lines = content.strip().split('\n')
            if len(lines) < 2:
                print(f"   ❌ CSV has insufficient data ({len(lines)} lines)")
                return False
            
            # Check for expected headers
            first_line = lines[0].lower()
            expected_headers = ['dog name', 'dog', 'runner', 'name', 'placing', 'box']
            if not any(header in first_line for header in expected_headers):
                print(f"   ⚠️ CSV may not be a form guide (first line: {first_line[:100]})")
                # Continue anyway as format might be different but still valid
            else:
                print(f"   ✅ CSV appears to be valid form guide data")
            
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
            
            print(f"   ✅ Successfully saved CSV data to: {filename} ({len(lines)} lines)")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error processing CSV data: {e}")
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
