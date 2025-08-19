#!/usr/bin/env python3
"""
Expert Form CSV Scraper for thedogs.com.au
==========================================

This script downloads CSV form guides from expert-form pages on thedogs.com.au
using the proper expert-form URL pattern and form submission approach.

Based on the documentation and archive analysis, CSV files are available on:
https://www.thedogs.com.au/racing/venue/YYYY-MM-DD/race_number/race-name/expert-form

The CSV is downloaded by submitting a form with export_csv parameter.

Author: AI Assistant
Date: August 4, 2025
Version: 1.0.0 - Expert form approach implementation
"""

import os
import sys
import requests
import time
import random
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import re
from pathlib import Path
import sqlite3
from utils.date_parsing import parse_date_flexible
from utils.race_file_utils import RaceFileManager
import threading
from concurrent.futures import ThreadPoolExecutor
import json

class ExpertFormCsvScraper:
    def __init__(self, max_workers=3, verbose=False):
        self.base_url = "https://www.thedogs.com.au"
        self.unprocessed_dir = "./unprocessed"
        self.download_dir = "./form_guides/downloaded"
        # Output directory used by API
        self.output_dir = os.environ.get('UPCOMING_RACES_DIR', './upcoming_races')
        self.database_path = "./databases/greyhound_racing.db"
        self.max_workers = max_workers
        self.verbose = verbose
        
        # Create directories
        os.makedirs(self.unprocessed_dir, exist_ok=True)
        os.makedirs(self.download_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize race file manager for caching
        self.race_file_manager = RaceFileManager(self.database_path)
        self.collected_races = self.race_file_manager.collected_races
        self.existing_files = self.race_file_manager.existing_files
        
        # Setup session with proper headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Statistics tracking
        self.stats = {
            'races_requested': 0,
            'cache_hits': 0,
            'fetches_attempted': 0,
            'fetches_failed': 0,
            'successful_saves': 0,
            'expert_form_found': 0,
            'csv_forms_found': 0,
            'direct_csv_downloads': 0
        }
        
        # Thread-safe logging
        self.log_lock = threading.Lock()
        
        print("🏁 Expert Form CSV Scraper initialized")
        print(f"📂 Unprocessed directory: {self.unprocessed_dir}")
        print(f"📂 Download directory: {self.download_dir}")
        print(f"⚡ Max workers: {self.max_workers}")
        print(f"🔍 Verbose logging: {self.verbose}")
    
    def safe_log(self, message, level="INFO"):
        """Thread-safe logging"""
        with self.log_lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            if level == "ERROR":
                print(f"[{timestamp}] ❌ {message}")
            elif level == "WARNING":
                print(f"[{timestamp}] ⚠️ {message}")
            elif level == "SUCCESS":
                print(f"[{timestamp}] ✅ {message}")
            elif self.verbose or level == "INFO":
                print(f"[{timestamp}] {message}")
    
    def get_expert_form_url(self, race_url):
        """Convert a race URL to its expert-form URL"""
        # Remove query parameters and trailing slashes
        base_race_url = race_url.split('?')[0].rstrip('/')
        expert_form_url = f"{base_race_url}/expert-form"
        return expert_form_url
    
    def find_csv_download_form(self, soup, expert_form_url):
        """Find and extract CSV download form data from expert-form page"""
        try:
            # Look for forms with CSV export capability
            csv_form = None
            for form in soup.find_all('form'):
                # Check for export_csv input/button
                csv_input = form.find('input', {'name': 'export_csv'})
                csv_button = form.find('button', {'name': 'export_csv'})
                
                # Also check for buttons with CSV text
                csv_text_buttons = form.find_all('button', string=lambda text: text and 'csv' in text.lower())
                csv_value_buttons = form.find_all('button', {'value': lambda val: val and 'csv' in val.lower()})
                submit_buttons = form.find_all('input', {'type': 'submit'})
                
                if csv_input or csv_button or csv_text_buttons or csv_value_buttons or submit_buttons:
                    csv_form = form
                    self.stats['csv_forms_found'] += 1
                    if self.verbose:
                        self.safe_log(f"Found CSV export form")
                    break
            
            if not csv_form:
                if self.verbose:
                    self.safe_log("No CSV export form found on expert-form page", "WARNING")
                return None
            
            # Extract form data and action
            form_action = csv_form.get('action', '')
            form_method = csv_form.get('method', 'GET').upper()
            
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
            
            # Build form data
            form_data = {}
            
            # Get all form inputs with their current values
            for input_elem in csv_form.find_all(['input', 'select', 'textarea']):
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
            
            return {
                'action': target_url,
                'method': form_method,
                'data': form_data
            }
            
        except Exception as e:
            self.safe_log(f"Error finding CSV form: {e}", "ERROR")
            return None
    
    def download_csv_from_expert_form(self, race_url, filename):
        """Download CSV using the expert-form method"""
        try:
            expert_form_url = self.get_expert_form_url(race_url)
            self.safe_log(f"Accessing expert-form page: {expert_form_url}")
            
            # Step 1: Get the expert-form page
            response = self.session.get(expert_form_url, timeout=30)
            if response.status_code != 200:
                self.safe_log(f"Expert-form page not accessible: {response.status_code}", "WARNING")
                return False
            
            self.stats['expert_form_found'] += 1
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Step 2: Find the CSV download form
            form_info = self.find_csv_download_form(soup, expert_form_url)
            if not form_info:
                return False
            
            # Step 3: Submit the form to get CSV data
            self.safe_log(f"Submitting form ({form_info['method']}) to: {form_info['action']}")
            
            if form_info['method'] == 'POST':
                form_response = self.session.post(form_info['action'], data=form_info['data'], timeout=30)
            else:
                form_response = self.session.get(form_info['action'], params=form_info['data'], timeout=30)
            
            if form_response.status_code != 200:
                self.safe_log(f"Form submission failed: {form_response.status_code}", "ERROR")
                return False
            
            content_type = form_response.headers.get('content-type', '').lower()
            content = form_response.text.strip()
            
            # Check if we got CSV data directly
            if 'csv' in content_type or 'text/plain' in content_type:
                if content and len(content.split('\n')) > 1:
                    lines = content.split('\n')
                    first_line = lines[0].lower()
                    if any(header in first_line for header in ['dog', 'name', 'runner', 'placing', 'box']):
                        self.safe_log(f"Got CSV data directly ({len(lines)} lines)")
                        return self.save_csv_content(content, filename)
            
            # Check if response contains a download URL
            if content.startswith('http'):
                self.safe_log(f"Got CSV download URL: {content}")
                csv_response = self.session.get(content, timeout=30)
                if csv_response.status_code == 200:
                    return self.save_csv_content(csv_response.text, filename)
            
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
                        
                        self.safe_log(f"Found CSV link in response: {href}")
                        csv_response = self.session.get(href, timeout=30)
                        if csv_response.status_code == 200:
                            return self.save_csv_content(csv_response.text, filename)
            
            self.safe_log(f"Unexpected response format ({len(content)} chars)", "WARNING")
            return False
            
        except Exception as e:
            self.safe_log(f"Error with expert-form method: {e}", "ERROR")
            return False
    
    def save_csv_content(self, content, filename):
        """Save CSV content to files and API's upcoming directory"""
        try:
            # Validate content
            if not content.strip():
                self.safe_log("Empty CSV content", "ERROR")
                return False
            
            # Basic CSV validation
            lines = content.strip().split('\n')
            if len(lines) < 2:
                self.safe_log(f"CSV has insufficient data ({len(lines)} lines)", "ERROR")
                return False
            
            # Check for expected headers
            first_line = lines[0].lower()
            expected_headers = ['dog name', 'dog', 'runner', 'name', 'placing', 'box']
            if not any(header in first_line for header in expected_headers):
                self.safe_log(f"CSV may not be valid form guide data", "WARNING")
                # Continue anyway as format might be different but still valid
            else:
                self.safe_log("CSV appears to be valid form guide data")
            
            # Save to download directory first (for backup/tracking)
            download_filepath = os.path.join(self.download_dir, filename)
            with open(download_filepath, 'w', encoding='utf-8', newline='') as f:
                f.write(content)
            
            # Move to unprocessed directory for analysis
            unprocessed_filepath = os.path.join(self.unprocessed_dir, filename)
            with open(unprocessed_filepath, 'w', encoding='utf-8', newline='') as f:
                f.write(content)
            
            # Also save into API's upcoming races dir with compliant name
            upcoming_filepath = os.path.join(self.output_dir, filename)
            with open(upcoming_filepath, 'w', encoding='utf-8', newline='') as f:
                f.write(content)
            
            # Add to existing files list to prevent future duplicates
            self.existing_files.add(filename)
            
            # Extract race info from filename and add to collected races
            match = re.match(r'Race (\d+) - ([A-Z_]+) - (\d{4}-\d{2}-\d{2})\.csv', filename)
            if match:
                race_number, venue, date_str = match.groups()
                try:
                    race_id = (date_str, venue, race_number)
                    self.collected_races.add(race_id)
                except ValueError:
                    pass  # Skip if date parsing fails
            
            self.safe_log(f"Successfully saved CSV: {filename} ({len(lines)} lines)", "SUCCESS")
            self.stats['direct_csv_downloads'] += 1
            return True
            
        except Exception as e:
            self.safe_log(f"Error saving CSV content: {e}", "ERROR")
            return False
    
    def download_race_csv(self, race_info):
        """Download CSV for a single race"""
        try:
            from utils.file_naming import build_upcoming_csv_filename
        except Exception:
            # Lazy import fallback if utils path not available
            build_upcoming_csv_filename = lambda n, v, d: f"Race {n} - {v} - {d}.csv"
        
        try:
            self.stats['races_requested'] += 1
            
            # Use centralized date parsing
            try:
                formatted_date = parse_date_flexible(race_info['date'])
            except ValueError as e:
                self.stats['fetches_failed'] += 1
                self.safe_log(f"Date parsing error for {race_info['date']}: {e}", "ERROR")
                return False
            
            race_id = (formatted_date, race_info['venue'], str(race_info['race_number']))
            
            # Check if already collected (cache hit)
            if race_id in self.collected_races:
                self.stats['cache_hits'] += 1
                if self.verbose:
                    self.safe_log(f"Race already collected: {race_id}")
                return True
            
            # Generate compliant filename using deterministic builder
            filename = build_upcoming_csv_filename(race_info['race_number'], race_info['venue'], formatted_date)
            
            # Check if file already exists in output dir
            upcoming_path = os.path.join(self.output_dir, filename)
            if os.path.exists(upcoming_path):
                self.stats['cache_hits'] += 1
                if self.verbose:
                    self.safe_log(f"File already exists in upcoming dir: {filename}")
                return True
            
            self.stats['fetches_attempted'] += 1
            self.safe_log(f"Downloading: {race_info['venue']} Race {race_info['race_number']} ({formatted_date})")
            
            # Try expert-form method
            success = self.download_csv_from_expert_form(race_info['url'], filename)
            
            if success:
                self.collected_races.add(race_id)
                self.stats['successful_saves'] += 1
                self.safe_log(f"Successfully downloaded: {filename}", "SUCCESS")
                return True
            else:
                self.stats['fetches_failed'] += 1
                self.safe_log(f"Failed to download: {filename}", "ERROR")
                return False
                
        except Exception as e:
            self.stats['fetches_failed'] += 1
            self.safe_log(f"Error downloading race CSV: {e}", "ERROR")
            return False
    
    def get_upcoming_races(self, days_ahead=1):
        """Get upcoming races using the existing upcoming race browser"""
        try:
            from upcoming_race_browser import UpcomingRaceBrowser
            
            browser = UpcomingRaceBrowser()
            upcoming_races = []
            
            # Get races for the next few days
            for days in range(days_ahead + 1):
                target_date = datetime.now().date() + timedelta(days=days)
                race_data_list = browser.get_races_for_date(target_date)
                
                for race_data in race_data_list:
                    race_info = {
                        'race_number': race_data.get('race_number', '1'),
                        'venue': race_data.get('venue', 'UNKNOWN'),
                        'date': target_date.strftime('%Y-%m-%d'),
                        'url': race_data.get('url', '')
                    }
                    upcoming_races.append(race_info)
            
            self.safe_log(f"Found {len(upcoming_races)} upcoming races")
            return upcoming_races
            
        except Exception as e:
            self.safe_log(f"Error getting upcoming races: {e}", "ERROR")
            return []
    
    def run_batch_download(self, days_ahead=1):
        """Run batch download of upcoming race CSVs"""
        self.safe_log("🚀 Starting batch CSV download...")
        
        # Get upcoming races
        upcoming_races = self.get_upcoming_races(days_ahead)
        
        if not upcoming_races:
            self.safe_log("No upcoming races found", "WARNING")
            return
        
        self.safe_log(f"📋 Processing {len(upcoming_races)} races with {self.max_workers} workers")
        
        # Use thread pool for concurrent downloads
        successful_downloads = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all download tasks
            future_to_race = {
                executor.submit(self.download_race_csv, race_info): race_info 
                for race_info in upcoming_races
            }
            
            # Process completed downloads
            for future in future_to_race:
                try:
                    success = future.result(timeout=60)  # 60 second timeout per race
                    if success:
                        successful_downloads += 1
                except Exception as e:
                    race_info = future_to_race[future]
                    self.safe_log(f"Download failed for {race_info['venue']} Race {race_info['race_number']}: {e}", "ERROR")
        
        # Print final statistics
        self.print_statistics()
        self.safe_log(f"🏁 Batch download completed: {successful_downloads}/{len(upcoming_races)} successful", "SUCCESS")
    
    def print_statistics(self):
        """Print download statistics"""
        print("\n📊 Download Statistics:")
        print(f"   🏁 Races requested: {self.stats['races_requested']}")
        print(f"   ⚡ Cache hits: {self.stats['cache_hits']}")
        print(f"   🌐 Fetches attempted: {self.stats['fetches_attempted']}")
        print(f"   ❌ Fetches failed: {self.stats['fetches_failed']}")
        print(f"   ✅ Successful saves: {self.stats['successful_saves']}")
        print(f"   📋 Expert forms found: {self.stats['expert_form_found']}")
        print(f"   📝 CSV forms found: {self.stats['csv_forms_found']}")
        print(f"   📥 Direct CSV downloads: {self.stats['direct_csv_downloads']}")
        
        # Calculate success rate
        if self.stats['fetches_attempted'] > 0:
            success_rate = (self.stats['successful_saves'] / self.stats['fetches_attempted']) * 100
            print(f"   📈 Success rate: {success_rate:.1f}%")
        
        # Calculate cache hit rate
        if self.stats['races_requested'] > 0:
            cache_hit_rate = (self.stats['cache_hits'] / self.stats['races_requested']) * 100
            print(f"   ⚡ Cache hit rate: {cache_hit_rate:.1f}%")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Expert Form CSV Scraper for thedogs.com.au')
    parser.add_argument('--days-ahead', type=int, default=1,
                       help='Number of days ahead to scrape (default: 1)')
    parser.add_argument('--max-workers', type=int, default=3,
                       help='Maximum number of concurrent workers (default: 3)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--test-url', type=str,
                       help='Test download from a specific race URL')
    
    args = parser.parse_args()
    
    scraper = ExpertFormCsvScraper(
        max_workers=args.max_workers,
        verbose=args.verbose
    )
    
    if args.test_url:
        # Test mode - download from single URL
        print(f"🧪 Test Mode: Downloading from {args.test_url}")
        
        # Extract race info from URL
        try:
            # Basic URL parsing - you might need to enhance this
            url_parts = args.test_url.split('/')
            if len(url_parts) >= 7:
                venue_slug = url_parts[4]
                date_parts = url_parts[5].split('-')
                race_number = url_parts[6]
                
                race_info = {
                    'race_number': race_number,
                    'venue': venue_slug.upper().replace('-', '_'),
                    'date': f"{date_parts[0]}-{date_parts[1]}-{date_parts[2]}",
                    'url': args.test_url
                }
                
                success = scraper.download_race_csv(race_info)
                scraper.print_statistics()
                
                if success:
                    print("✅ Test completed successfully!")
                else:
                    print("❌ Test failed!")
                    sys.exit(1)
            else:
                print("❌ Invalid URL format")
                sys.exit(1)
                
        except Exception as e:
            print(f"❌ Error in test mode: {e}")
            sys.exit(1)
    else:
        # Normal batch mode
        scraper.run_batch_download(days_ahead=args.days_ahead)

if __name__ == "__main__":
    main()
