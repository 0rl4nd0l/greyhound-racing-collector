#!/usr/bin/env python3
"""
Debug Race Extraction Script
============================

This script diagnoses issues with race information extraction from thedogs.com.au
to identify why the CSV scraper is failing with "could not extract" errors.
"""

import requests
import re
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import random
import time

class RaceExtractionDebugger:
    def __init__(self):
        self.base_url = "https://www.thedogs.com.au"
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
    
    def test_date_page(self, date):
        """Test accessing a date page and finding race URLs"""
        date_str = date.strftime('%Y-%m-%d')
        base_url = f"{self.base_url}/racing/{date_str}"
        
        print(f"\n🔍 Testing date page: {date_str}")
        print(f"URL: {base_url}")
        
        try:
            response = self.session.get(base_url, timeout=30)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ Failed to access page")
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Debug: Show page title and main content structure
            title = soup.find('title')
            if title:
                print(f"Page Title: {title.get_text(strip=True)}")
            
            # Look for various link patterns
            print(f"\n📋 Analyzing page structure:")
            
            # Method 1: All links containing racing
            racing_links = soup.find_all('a', href=re.compile(r'/racing/'))
            print(f"Links containing '/racing/': {len(racing_links)}")
            
            # Method 2: All links containing the date
            date_links = soup.find_all('a', href=re.compile(f'{date_str}'))
            print(f"Links containing '{date_str}': {len(date_links)}")
            
            # Method 3: Look for common class patterns
            race_card_links = soup.select('.race-card a, .race-link, .race-item a')
            print(f"Race card/link elements: {len(race_card_links)}")
            
            # Collect potential race URLs
            potential_urls = []
            
            for link in racing_links + date_links + race_card_links:
                href = link.get('href')
                if href and date_str in href:
                    full_url = href if href.startswith('http') else f"{self.base_url}{href}"
                    potential_urls.append(full_url)
            
            # Remove duplicates
            unique_urls = list(set(potential_urls))
            
            print(f"\n🎯 Found {len(unique_urls)} potential race URLs:")
            for i, url in enumerate(unique_urls[:10]):  # Show first 10
                print(f"  {i+1}. {url}")
                
            if len(unique_urls) > 10:
                print(f"  ... and {len(unique_urls) - 10} more")
            
            # Test URL pattern analysis
            print(f"\n🔍 URL Pattern Analysis:")
            valid_race_urls = []
            
            for url in unique_urls:
                parts = url.split('/')
                print(f"\nURL: {url}")
                print(f"Parts: {parts}")
                
                # Check if it looks like a race URL
                if len(parts) >= 5:
                    # Look for numeric race number
                    race_num_candidates = [p for p in parts if p.isdigit()]
                    if race_num_candidates:
                        print(f"  Race number candidates: {race_num_candidates}")
                        valid_race_urls.append(url)
                    else:
                        print(f"  ❌ No numeric race number found")
                else:
                    print(f"  ❌ URL too short (less than 5 parts)")
            
            print(f"\n✅ Valid race URLs: {len(valid_race_urls)}")
            return valid_race_urls[:5]  # Return first 5 for testing
            
        except Exception as e:
            print(f"❌ Error testing date page: {e}")
            return []
    
    def test_race_info_extraction(self, race_url):
        """Test extracting race information from a specific race URL"""
        print(f"\n🔍 Testing race info extraction for: {race_url}")
        
        try:
            response = self.session.get(race_url, timeout=30)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ Failed to access race page")
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Debug extraction process
            print(f"\n📋 Extracting race information...")
            
            # Extract race number from URL
            url_parts = race_url.split('/')
            race_number = None
            venue = None
            date = None
            
            print(f"URL parts: {url_parts}")
            
            # Find race number in URL
            for i, part in enumerate(url_parts):
                if part.isdigit() and i > 0:
                    race_number = part
                    print(f"✅ Race number from URL: {race_number}")
                    break
            
            if not race_number:
                print(f"❌ Could not find race number in URL")
            
            # Extract venue from URL
            print(f"\n🏁 Venue extraction:")
            for venue_key, venue_code in self.venue_map.items():
                if venue_key in race_url:
                    venue = venue_code
                    print(f"✅ Venue from URL: {venue_key} -> {venue_code}")
                    break
            
            if not venue:
                print(f"❌ Could not find venue in URL")
                print(f"Available venues: {list(self.venue_map.keys())}")
            
            # Extract date from URL
            print(f"\n📅 Date extraction:")
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', race_url)
            if date_match:
                date_obj = datetime.strptime(date_match.group(1), '%Y-%m-%d')
                date = date_obj.strftime('%d %B %Y')
                print(f"✅ Date from URL: {date_match.group(1)} -> {date}")
            else:
                print(f"❌ Could not find date in URL")
            
            # Try to extract venue from page content if not found in URL
            if not venue:
                print(f"\n🔍 Trying to extract venue from page content...")
                venue_selectors = ['.venue-name', '.track-name', 'h1', '.race-header', 'title']
                for selector in venue_selectors:
                    elements = soup.select(selector)
                    for element in elements:
                        text = element.get_text(strip=True)
                        print(f"  {selector}: '{text}'")
                        for venue_key, venue_code in self.venue_map.items():
                            if venue_key.replace('-', ' ').lower() in text.lower():
                                venue = venue_code
                                print(f"  ✅ Found venue match: {venue_key} -> {venue_code}")
                                break
                        if venue:
                            break
                    if venue:
                        break
            
            # Show page structure for debugging
            print(f"\n📋 Page structure analysis:")
            title = soup.find('title')
            if title:
                print(f"Title: {title.get_text(strip=True)}")
            
            h1_tags = soup.find_all('h1')
            print(f"H1 tags: {[h.get_text(strip=True) for h in h1_tags]}")
            
            # Look for common racing page elements
            common_selectors = ['.race-info', '.race-details', '.race-header', '.venue', '.track']
            for selector in common_selectors:
                elements = soup.select(selector)
                if elements:
                    print(f"{selector}: {[e.get_text(strip=True)[:50] for e in elements[:3]]}")
            
            # Final result
            print(f"\n🎯 Extraction Results:")
            print(f"Race Number: {race_number}")
            print(f"Venue: {venue}")
            print(f"Date: {date}")
            
            if race_number and venue and date:
                result = {
                    'race_number': race_number,
                    'venue': venue,
                    'date': date
                }
                print(f"✅ Successfully extracted race info: {result}")
                return result
            else:
                print(f"❌ Missing required information")
                return None
                
        except Exception as e:
            print(f"❌ Error extracting race info: {e}")
            return None
    
    def test_csv_link_finding(self, race_url):
        """Test finding CSV download links"""
        print(f"\n🔍 Testing CSV link finding for: {race_url}")
        
        try:
            # Test expert-form method
            base_race_url = race_url.split('?')[0]
            expert_form_url = f"{base_race_url}/expert-form"
            
            print(f"Expert-form URL: {expert_form_url}")
            
            response = self.session.get(expert_form_url, timeout=10)
            print(f"Expert-form status: {response.status_code}")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for forms
                forms = soup.find_all('form')
                print(f"Forms found: {len(forms)}")
                
                for i, form in enumerate(forms):
                    print(f"\nForm {i+1}:")
                    print(f"  Action: {form.get('action')}")
                    print(f"  Method: {form.get('method')}")
                    
                    # Look for CSV-related inputs
                    csv_inputs = form.find_all('input', {'name': re.compile(r'csv|export')})
                    csv_buttons = form.find_all('button', {'name': re.compile(r'csv|export')})
                    
                    if csv_inputs or csv_buttons:
                        print(f"  ✅ CSV export capability found!")
                        for inp in csv_inputs:
                            print(f"    Input: {inp.get('name')} = {inp.get('value')}")
                        for btn in csv_buttons:
                            print(f"    Button: {btn.get('name')} = {btn.get_text(strip=True)}")
                    else:
                        print(f"  ❌ No CSV export capability")
            
            # Test fallback methods
            print(f"\n🔍 Testing fallback CSV link methods...")
            
            # Get main race page
            response = self.session.get(race_url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for CSV links
                csv_links = soup.find_all('a', href=re.compile(r'csv|export|download'))
                print(f"CSV/Export/Download links: {len(csv_links)}")
                
                for link in csv_links:
                    href = link.get('href')
                    text = link.get_text(strip=True)
                    print(f"  Link: {text} -> {href}")
            
        except Exception as e:
            print(f"❌ Error testing CSV link finding: {e}")
    
    def run_diagnosis(self):
        """Run comprehensive diagnosis"""
        print("🚀 RACE EXTRACTION DIAGNOSIS")
        print("=" * 50)
        
        # Test with recent dates
        today = datetime.now().date()
        test_dates = [today - timedelta(days=i) for i in range(1, 4)]  # Last 3 days
        
        for date in test_dates:
            print(f"\n{'='*50}")
            race_urls = self.test_date_page(date)
            
            # Test extraction on first valid URL
            if race_urls:
                test_url = race_urls[0]
                race_info = self.test_race_info_extraction(test_url)
                
                if race_info:
                    self.test_csv_link_finding(test_url)
                
                # Add delay between requests
                time.sleep(2)
            else:
                print(f"❌ No race URLs found for {date}")
        
        print(f"\n{'='*50}")
        print("🎯 DIAGNOSIS COMPLETE")
        print("\nCommon issues that could cause 'could not extract' errors:")
        print("1. URL structure changes on the website")
        print("2. Venue mapping not matching current URL patterns")
        print("3. Date format extraction issues")
        print("4. Race number extraction failing")
        print("5. Page structure changes affecting selectors")

def main():
    debugger = RaceExtractionDebugger()
    debugger.run_diagnosis()

if __name__ == "__main__":
    main()
