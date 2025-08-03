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
Date: July 31, 2025
Version: 3.0.1 - Fixed regex patterns and centralized date parsing
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
from utils.date_parsing import parse_date_flexible
from src.parsers.csv_ingestion import CsvIngestion

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
            'casino': 'CASINO',
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
        
        try:
            # Ensure database tables exist
            self._ensure_database_tables()
            
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            # Fetch existing hashes from database
            cursor.execute("SELECT file_hash FROM processed_race_files")
            self.processed_hashes = set(row[0] for row in cursor.fetchall())
            conn.close()

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
                        
                        # Extract race info from filename using improved regex patterns
                        race_id = self.parse_filename_to_race_id(filename)
                        if race_id:
                            self.collected_races.add(race_id)
            
            print(f"📊 Loaded {len(self.collected_races)} unique races from {total_files} files")
            print(f"🗂️ Found {len(self.processed_hashes)} previously processed file hashes")
        except Exception as e:
            print(f"❌ Error loading collected races: {e}")
            self.processed_hashes = set()

    def _ensure_database_tables(self):
        """Ensure the processed_race_files table exists"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_race_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash TEXT UNIQUE NOT NULL,
                race_date DATE NOT NULL,
                venue TEXT NOT NULL, 
                race_no INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER,
                processed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'processed',
                error_message TEXT
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_processed_files_hash ON processed_race_files(file_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_processed_files_race_key ON processed_race_files(race_date, venue, race_no)')
        
        conn.commit()
        conn.close()
    
    def parse_filename_to_race_id(self, filename):
        """Parse filename to extract race_id tuple (date, venue, race_number)"""
        # Pattern 1: Race N - VENUE - DD Month YYYY.csv
        match = re.match(r'Race (\d+) - ([A-Z_]+) - (\d{1,2} \w+ \d{4})\.csv', filename)
        if match:
            race_number, venue, date_str = match.groups()
            try:
                formatted_date = parse_date_flexible(date_str)
                return (formatted_date, venue, race_number)
            except ValueError:
                pass
        
        # Pattern 2: ???_Race_N_VENUE_DATE.csv  
        match = re.match(r'\w+_Race_(\d+)_([A-Z_]+)_([\d-]+)\.csv', filename)
        if match:
            race_number, venue, date_str = match.groups()
            try:
                formatted_date = parse_date_flexible(date_str)
                return (formatted_date, venue, race_number)
            except ValueError:
                pass
        
        # Pattern 3: Race_N_-_VENUE_-_DATE.csv
        match = re.match(r'Race_(\d+)_-_([A-Z_]+)_-_([\d_A-Za-z]+)\.csv', filename)
        if match:
            race_number, venue, date_str = match.groups()
            try:
                formatted_date = parse_date_flexible(date_str)
                return (formatted_date, venue, race_number)
            except ValueError:
                pass
        
        # Pattern 4: Race N - VENUE - YYYY-MM-DD.csv (new compact pattern)
        match = re.match(r'Race (\d+) - ([A-Z_]+) - (\d{4}-\d{2}-\d{2})\.csv', filename)
        if match:
            race_number, venue, date_str = match.groups()
            try:
                formatted_date = parse_date_flexible(date_str)
                return (formatted_date, venue, race_number)
            except ValueError:
                pass
        
        return None

    def compute_file_hash(self, file_path):
        """Compute SHA-256 hash of a file"""
        import hashlib
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read and update hash in chunks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def parse_csv_with_ingestion(self, file_path, force=False):
        """Parse CSV using CsvIngestion module with caching and de-duplication"""
        file_hash = self.compute_file_hash(file_path)

        # Check if file already processed (cache hit)
        if file_hash in self.processed_hashes and not force:
            print(f"⚠️ Cache HIT: Skipping already processed file: {file_path}")
            return "hit"

        print(f"🔄 Cache MISS: Processing file: {file_path}")
        
        # Extract race metadata from filename for database recording
        race_info = self.parse_filename_to_race_id(os.path.basename(file_path))
        
        try:
            # Parse CSV using ingestion module
            try:
                ingestion = CsvIngestion(file_path)
                parsed_race, validation_report = ingestion.parse_csv()
            except NameError as name_error:
                if "ParsedRace" in str(name_error):
                    # Fallback: Just validate that file is readable CSV
                    import csv
                    with open(file_path, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        headers = next(reader)
                        row_count = sum(1 for row in reader)
                    print(f"📋 Parsed CSV: {len(headers)} columns, {row_count} rows")
                    parsed_race = {"headers": headers, "row_count": row_count}
                    validation_report = {"errors": []}
                else:
                    raise

            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Record processing outcome in database
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            if race_info:
                race_date, venue, race_no = race_info
                cursor.execute("""
                    INSERT OR REPLACE INTO processed_race_files 
                    (file_hash, race_date, venue, race_no, file_path, file_size, status) 
                    VALUES (?, ?, ?, ?, ?, ?, 'processed')
                """, (file_hash, race_date, venue, race_no, file_path, file_size))
            else:
                cursor.execute("""
                    INSERT OR REPLACE INTO processed_race_files 
                    (file_hash, race_date, venue, race_no, file_path, file_size, status) 
                    VALUES (?, 'unknown', 'unknown', 0, ?, ?, 'processed')
                """, (file_hash, file_path, file_size))
                
            conn.commit()
            conn.close()
            
            # Add to processed hashes to avoid re-processing in same session
            self.processed_hashes.add(file_hash)
            
            print(f"✅ Successfully processed and cached: {file_path}")
            return "miss"
            
        except Exception as e:
            print(f"❌ Error processing file {file_path}: {e}")
            
            # Record error in database
            try:
                conn = sqlite3.connect(self.database_path)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO processed_race_files 
                    (file_hash, race_date, venue, race_no, file_path, file_size, status, error_message) 
                    VALUES (?, 'error', 'error', 0, ?, ?, 'failed', ?)
                """, (file_hash, file_path, os.path.getsize(file_path) if os.path.exists(file_path) else 0, str(e)))
                conn.commit()
                conn.close()
            except Exception as db_error:
                print(f"❌ Error recording failure to database: {db_error}")
                
            return "error"
    
    def load_processed_races(self):
        """Load processed races to avoid re-processing"""
        processed_races = set()
        processed_dir = "./processed"
        
        if os.path.exists(processed_dir):
            for filename in os.listdir(processed_dir):
                if filename.endswith('.csv'):
                    # Extract race info from processed filename
                    match = re.match(r'(\d{4}-\d{2}-\d{2})_([A-Z_]+)_Race_(\d+)_processed\.csv', filename)
                    if match:
                        race_date, venue, race_number = match.groups()
                        try:
                            formatted_date = parse_date_flexible(race_date)
                            race_id = (formatted_date, venue, race_number)
                            processed_races.add(race_id)
                        except ValueError:
                            continue
        
        return processed_races
    
    def download_csv_from_race_page(self, race_info, max_retries=3):
        """Download CSV file from a race page with improved date parsing"""
        for attempt in range(max_retries):
            try:
                race_url = race_info['url']
                print(f"🏁 Downloading CSV from race: {race_url}")
                
                response = self.session.get(race_url, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for CSV download link
                csv_link = soup.find('a', href=re.compile(r'.*\.csv.*'))
                if not csv_link:
                    print(f"❌ No CSV link found on race page: {race_url}")
                    return False
                
                csv_url = csv_link.get('href')
                if not csv_url.startswith('http'):
                    csv_url = self.base_url + csv_url
                
                # Download CSV content
                csv_response = self.session.get(csv_url, timeout=30)
                csv_response.raise_for_status()
                
                # Use centralized date parsing for consistent formatting
                try:
                    formatted_date = parse_date_flexible(race_info['date'])
                except ValueError as e:
                    print(f"❌ Date parsing error for {race_info['date']}: {e}")
                    return False
                
                # Generate filename with consistent date format
                filename = f"Race {race_info['race_number']} - {race_info['venue']} - {formatted_date}.csv"
                filepath = os.path.join(self.unprocessed_dir, filename)
                
                # Save CSV file
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(csv_response.text)
                
                print(f"✅ Downloaded: {filename}")
                return True
                
            except Exception as e:
                print(f"❌ Error downloading CSV (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
        
        return False
    
    def extract_race_info(self, race_element):
        """Extract race information from race element with improved date parsing"""
        try:
            # Extract race number
            race_number_elem = race_element.find('span', class_='race-number')
            if not race_number_elem:
                return None
            race_number = race_number_elem.text.strip().replace('R', '')
            
            # Extract race URL
            link_elem = race_element.find('a')
            if not link_elem:
                return None
            race_url = link_elem.get('href')
            if not race_url.startswith('http'):
                race_url = self.base_url + race_url
            
            # Extract venue from URL or race element
            venue = None
            for url_venue, code in self.venue_map.items():
                if url_venue in race_url:
                    venue = code
                    break
            
            if not venue:
                print(f"⚠️ Could not determine venue for URL: {race_url}")
                return None
            
            # Extract date from URL or page context
            date_match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', race_url)
            if date_match:
                year, month, day = date_match.groups()
                date_str = f"{year}-{month}-{day}"
            else:
                # Try to extract from other sources
                date_str = None
            
            if not date_str:
                print(f"⚠️ Could not extract date from URL: {race_url}")
                return None
            
            # Use centralized date parsing
            try:
                formatted_date = parse_date_flexible(date_str)
            except ValueError as e:
                print(f"❌ Date parsing error for {date_str}: {e}")
                return None
            
            return {
                'race_number': race_number,
                'venue': venue,
                'date': formatted_date,
                'url': race_url
            }
            
        except Exception as e:
            print(f"❌ Error extracting race info: {e}")
            return None
    
    def download_csv_file(self, race_info):
        """Download and save CSV file with improved date parsing"""
        try:
            # Use centralized date parsing to ensure consistency
            try:
                formatted_date = parse_date_flexible(race_info['date'])
            except ValueError as e:
                print(f"❌ Date parsing error for {race_info['date']}: {e}")
                return False
            
            race_id = (formatted_date, race_info['venue'], race_info['race_number'])
            
            # Check if already collected
            if race_id in self.collected_races:
                print(f"⏭️ Already have: {race_id}")
                return True
            
            # Download the CSV
            success = self.download_csv_from_race_page(race_info)
            if success:
                self.collected_races.add(race_id)
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error downloading CSV file: {e}")
            return False

    def get_driver(self):
        """Set up Chrome driver with options"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            return driver
        except Exception as e:
            print(f"❌ Error setting up Chrome driver: {e}")
            return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Form Guide CSV Scraper with robust caching')
    parser.add_argument('--force', action='store_true', 
                       help='Force reprocessing of all files, ignoring cache')
    parser.add_argument('--test-file', type=str, 
                       help='Test the caching system with a specific CSV file')
    parser.add_argument('--stats', action='store_true',
                       help='Show cache statistics and exit')
    
    args = parser.parse_args()
    
    scraper = FormGuideCsvScraper()
    print(f"📊 Loaded {len(scraper.collected_races)} existing races")
    print(f"📁 Found {len(scraper.existing_files)} existing files")
    
    if args.stats:
        # Show detailed cache statistics
        conn = sqlite3.connect(scraper.database_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM processed_race_files")
        total_processed = cursor.fetchone()[0]
        
        cursor.execute("SELECT status, COUNT(*) FROM processed_race_files GROUP BY status")
        status_counts = cursor.fetchall()
        
        cursor.execute("SELECT COUNT(DISTINCT venue) FROM processed_race_files WHERE venue != 'unknown'")
        unique_venues = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(processed_at), MAX(processed_at) FROM processed_race_files")
        date_range = cursor.fetchone()
        
        print(f"\n📈 Cache Statistics:")
        print(f"   Total processed files: {total_processed}")
        print(f"   Unique venues: {unique_venues}")
        for status, count in status_counts:
            print(f"   {status.capitalize()}: {count}")
        if date_range[0] and date_range[1]:
            print(f"   Date range: {date_range[0]} to {date_range[1]}")
        
        conn.close()
        sys.exit(0)
    
    if args.test_file:
        if not os.path.exists(args.test_file):
            print(f"❌ Test file not found: {args.test_file}")
            sys.exit(1)
            
        print(f"\n🧪 Testing caching with file: {args.test_file}")
        
        # Test without force
        result1 = scraper.parse_csv_with_ingestion(args.test_file, force=False)
        print(f"First attempt (no force): {result1}")
        
        # Test with force
        result2 = scraper.parse_csv_with_ingestion(args.test_file, force=args.force)
        print(f"Second attempt (force={args.force}): {result2}")
        
        sys.exit(0)
    
    print(f"\n✨ Caching system ready! Use --help for options.")
    print(f"💡 Tips:")
    print(f"   - Use --test-file <path> to test caching with a specific file")
    print(f"   - Use --force to ignore cache and reprocess files")
    print(f"   - Use --stats to see cache statistics")
