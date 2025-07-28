#!/usr/bin/env python3
"""
Sportsbet Recent Races Scraper
===============================

This script scrapes recent greyhound race results from Sportsbet for 
populating the database with completed race information.

Usage: python3 sportsbet_recent_races_scraper.py
"""

import os
import re
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import sqlite3

class SportsbetRecentRacesScraper:
    """Scraper for getting recent race results from Sportsbet"""
    
    def __init__(self, db_path="greyhound_data.db"):
        self.base_url = "https://www.sportsbet.com.au"
        self.greyhound_url = f"{self.base_url}/racing/greyhound-racing"
        self.results_url = f"{self.base_url}/racing/results/greyhound-racing"
        self.driver = None
        self.db_path = db_path
        
        # Venue mapping for consistency
        self.venue_map = {
            'angle park': 'AP_K',
            'sandown': 'SAN', 
            'warrnambool': 'WAR',
            'bendigo': 'BEN',
            'geelong': 'GEE',
            'ballarat': 'BAL',
            'healesville': 'HEA',
            'sale': 'SAL',
            'richmond': 'RICH',
            'murray bridge': 'MURR',
            'the meadows': 'MEA',
            'wentworth park': 'WPK',
            'dapto': 'DAPT',
            'albion park': 'ALBION',
            'capalaba': 'CAPALABA',
            'rockhampton': 'ROCK',
            'broken hill': 'BROKEN-HILL',
            'grafton': 'GRAF',
            'darwin': 'DARW',
            'cannington': 'CANN',
            'northam': 'NOR',
            'mandurah': 'MAND',
            'gosford': 'GOSF',
            'hobart': 'HOBT',
            'the gardens': 'GRDN',
            'taree': 'TAREE'
        }
        
    def setup_driver(self):
        """Setup Chrome driver with stealth mode for web scraping"""
        if self.driver:
            return True
            
        options = Options()
        
        # Use headless mode with window size
        options.add_argument('--headless=new')
        options.add_argument('--window-size=1920,1080')
        
        # Stealth mode settings
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--enable-features=NetworkService,NetworkServiceInProcess')
        options.add_argument('--force-color-profile=srgb')
        options.add_argument('--disable-web-security')
        options.add_argument('--allow-running-insecure-content')
        options.add_argument('--disable-site-isolation-trials')
        options.add_argument('--disable-application-cache')
        options.add_argument('--disable-features=IsolateOrigins,site-per-process')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--no-sandbox')
        
        # Browser fingerprinting evasion
        options.add_argument('--disable-webgl')
        options.add_argument('--disable-notifications')
        options.add_argument('--disable-popup-blocking')
        options.add_argument('--ignore-certificate-errors')
        options.add_argument('--no-default-browser-check')
        options.add_argument('--no-first-run')
        options.add_argument('--disable-gpu')
        
        # Set a realistic user agent
        options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 11_0_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36')
        
        # Additional preferences to appear more human-like
        options.add_experimental_option('excludeSwitches', ['enable-automation', 'enable-logging'])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_experimental_option('prefs', {
            'profile.default_content_setting_values.notifications': 2,
            'credentials_enable_service': False,
            'profile.password_manager_enabled': False,
            'plugins.always_open_pdf_externally': True,
            'download_restrictions': 3,
            'profile.default_content_settings.popups': 0,
            'profile.managed_default_content_settings.images': 1
        })
        
        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
            
            # Additional stealth script injections
            stealth_script = """
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            window.chrome = { runtime: {} };
            Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
            """
            self.driver.execute_script(stealth_script)
            
            print("✅ Chrome driver setup successful with stealth mode")
            return True
        except Exception as e:
            print(f"❌ Chrome driver setup failed: {e}")
            self.driver = None
            return False
    
    def close_driver(self):
        """Close the web driver"""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def scrape_recent_races(self, days_back=3) -> List[Dict]:
        """Scrape recent race results from Sportsbet"""
        if not self.setup_driver():
            return []
        
        all_races = []
        
        try:
            print(f"🏁 Scraping recent race results from Sportsbet ({days_back} days back)...")
            
            # First navigate to the main page to establish a session
            print("🌐 Establishing session...")
            self.driver.get(self.base_url)
            time.sleep(2)
            
            # Then go to the greyhound racing section
            print("🌐 Navigating to greyhound racing section...")
            self.driver.get(self.greyhound_url)
            time.sleep(2)
            
            # Finally navigate to results
            print(f"🌐 Navigating to: {self.results_url}")
            self.driver.get(self.results_url)
            
            # Wait for page to load
            try:
                WebDriverWait(self.driver, 15).until(
                    lambda driver: driver.execute_script("return document.readyState") == "complete"
                )
                print("📄 Results page loaded successfully")
            except TimeoutException:
                print("⚠️ Timeout waiting for results page to load")
            
            time.sleep(3)
            
            # Look for recent race results
            recent_races = self._extract_recent_race_results()
            if recent_races:
                all_races.extend(recent_races)
                print(f"✅ Found {len(recent_races)} recent races from results page")
                
                # Debug output for the first few races
                print("\n🔍 Sample race content:")
                for race in recent_races[:3]:
                    print("---")
                    print(f"Raw text: {race['raw_text']}")
            
            # If results page doesn't have enough, try main greyhound page
            if len(all_races) < 20:
                print(f"🌐 Also checking main greyhound page: {self.greyhound_url}")
                self.driver.get(self.greyhound_url)
                
                try:
                    WebDriverWait(self.driver, 10).until(
                        lambda driver: driver.execute_script("return document.readyState") == "complete"
                    )
                except TimeoutException:
                    pass
                
                time.sleep(3)
                
                # Look for completed races on main page
                main_page_races = self._extract_completed_races_from_main_page()
                if main_page_races:
                    # Avoid duplicates
                    existing_race_keys = set()
                    for race in all_races:
                        race_key = f"{race['venue']}_{race['race_number']}_{race['date']}"
                        existing_race_keys.add(race_key)
                    
                    new_races = []
                    for race in main_page_races:
                        race_key = f"{race['venue']}_{race['race_number']}_{race['date']}"
                        if race_key not in existing_race_keys:
                            new_races.append(race)
                    
                    all_races.extend(new_races)
                    print(f"✅ Added {len(new_races)} additional races from main page")
            
            print(f"📊 Total races found: {len(all_races)}")
            return all_races
            
        except Exception as e:
            print(f"❌ Error scraping recent races: {e}")
            return all_races
        finally:
            self.close_driver()
    
    def _extract_recent_race_results(self) -> List[Dict]:
        """Extract recent race results from the results page"""
        races = []
        
        try:
            print(f"🔍 Looking for recent race result elements...")
            
            # First, let's inspect the page structure
            print("\n🔍 Page structure analysis:")
            page_source = self.driver.page_source
            print(f"Page title: {self.driver.title}")
            print(f"Page URL: {self.driver.current_url}")
            print(f"Page source length: {len(page_source)} characters")
            
            # Look for key indicators in the page source
            if 'final' in page_source.lower():
                print("✅ Found 'FINAL' text in page source")
            if 'result' in page_source.lower():
                print("✅ Found 'result' text in page source")
            if 'winner' in page_source.lower():
                print("✅ Found 'winner' text in page source")
            
            # Save a snippet of the page source for debugging
            with open('sportsbet_page_debug.html', 'w') as f:
                f.write(page_source)
            print("💾 Saved page source to sportsbet_page_debug.html")
            
            # Target Sportsbet's specific structure
            result_selectors = [
                # Look for venue rows and race cells
                "tr",  # Table rows for venues
                "td",  # Table cells for individual races
                "[class*='meeting']",
                "[class*='venue']",
                "[class*='race']",
                # Look for elements containing "FINAL" (completed races)
                "*:contains('FINAL')",
                "*[title*='FINAL']",
                "*[alt*='FINAL']"
            ]
            
            result_elements = []
            for selector in result_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        result_elements.extend(elements)
                        print(f"  📊 Found {len(elements)} elements with selector: {selector}")
                except:
                    continue
            
            if not result_elements:
                print("⚠️ No specific result elements found, trying broader search...")
                result_elements = self._find_results_by_content()
            
            print(f"📋 Processing {len(result_elements)} potential result elements...")
            
            # Process each element
            for i, element in enumerate(result_elements[:50]):  # Limit to first 50
                try:
                    race_result = self._extract_race_result_from_element(element, i)
                    if race_result:
                        races.append(race_result)
                        print(f"  ✅ Added: {race_result['venue']} Race {race_result['race_number']} - Winner: {race_result.get('winner_name', 'Unknown')}")
                except Exception as e:
                    print(f"  ⚠️ Error processing result element {i}: {e}")
                    continue
            
            return races
            
        except Exception as e:
            print(f"❌ Error extracting race results: {e}")
            return []
    
    def _extract_completed_races_from_main_page(self) -> List[Dict]:
        """Extract completed races from main greyhound page"""
        races = []
        
        try:
            print("🔍 Looking for completed races on main page...")
            
            # Look for elements that might contain completed race info
            race_selectors = [
                "[class*='race']",
                "[class*='meeting']", 
                "[class*='result']",
                "a[href*='greyhound-racing']"
            ]
            
            race_elements = []
            for selector in race_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        race_elements.extend(elements)
                except:
                    continue
            
            print(f"📋 Processing {len(race_elements)} potential race elements...")
            
            # Process elements looking for completed races
            for i, element in enumerate(race_elements[:30]):  # Limit for performance
                try:
                    text = element.text.strip().lower()
                    
                    # Skip elements with time countdown (upcoming races)
                    if re.search(r'\d+[ms]\s*(?:\d+[ms])?$', text):
                        continue
                    
                    # Look for indicators of completed races
                    if any(indicator in text for indicator in ['winner', 'result', 'finished', 'won', 'paid', '1st']):
                        race_result = self._extract_race_result_from_element(element, i)
                        if race_result:
                            races.append(race_result)
                            print(f"  ✅ Added completed race: {race_result['venue']} Race {race_result['race_number']}")
                            
                except Exception as e:
                    continue
            
            return races
            
        except Exception as e:
            print(f"❌ Error extracting completed races: {e}")
            return []
    
    def _find_results_by_content(self) -> List:
        """Find result elements by analyzing page content"""
        try:
            all_elements = self.driver.find_elements(By.CSS_SELECTOR, "div, section, article, li")
            result_elements = []
            
            for element in all_elements:
                try:
                    text = element.text.strip().lower()
                    # Look for elements that might contain race results
                    # Focus on completed races (FINAL) and venues
                    if (('final' in text or 'result' in text or 'finished' in text or 'winner' in text) and 
                        len(text) > 10 and len(text) < 500):
                        result_elements.append(element)
                        if len(result_elements) >= 50:
                            break
                    # Also look for venue names with race info
                    elif (any(venue in text for venue in ['sandown', 'ballarat', 'warrnambool', 'shepparton', 'angle park', 'dubbo', 'nowra']) and
                          len(text) > 5):
                        result_elements.append(element)
                        if len(result_elements) >= 50:
                            break
                except:
                    continue
            
            print(f"📊 Found {len(result_elements)} elements by content analysis")
            return result_elements
            
        except Exception as e:
            print(f"⚠️ Error in content-based search: {e}")
            return []
    
    def _extract_race_result_from_element(self, element, index: int) -> Optional[Dict]:
        """Extract race result information from a single element"""
        try:
            text = element.text.strip()
            if not text or len(text) < 10:
                return None
            
            # Try to get link URL for more context
            race_url = None
            try:
                if element.tag_name == 'a':
                    race_url = element.get_attribute('href')
                else:
                    link = element.find_element(By.CSS_SELECTOR, 'a')
                    race_url = link.get_attribute('href')
            except:
                pass
            
            # Skip elements with countdown timers (upcoming races)
            if re.search(r'\d+[ms]\s*$', text) or re.search(r'-?\d+m\s+\d+s', text):
                return None
            
            # Skip if it doesn't contain "FINAL" or result indicators
            if not any(indicator in text.lower() for indicator in ['final', 'result', 'winner', 'won']):
                # Unless it's a venue name with race structure
                if not any(venue in text.lower() for venue in ['sandown', 'ballarat', 'warrnambool', 'shepparton', 'angle park', 'dubbo', 'nowra']):
                    return None
            
            # Extract venue name
            venue = self._extract_venue_from_text(text, race_url)
            if not venue:
                return None
            
            # Extract race number - look in surrounding elements too
            race_number = self._extract_race_number_from_element(element)
            if not race_number:
                return None
            
            # Extract race date (look for recent dates)
            race_date = self._extract_race_date_from_text(text)
            if not race_date:
                # Default to today if no date found
                race_date = datetime.now().strftime('%Y-%m-%d')
            
            # Extract winner information
            winner_info = self._extract_winner_from_text(text)
            
            # Extract race time if available
            race_time = self._extract_race_time_from_text(text)
            
            # Extract distance and grade
            distance = self._extract_distance_from_text(text)
            grade = self._extract_grade_from_text(text)
            
            # Create race result
            race_result = {
                'venue': venue,
                'race_number': race_number,
                'date': race_date,
                'race_time': race_time,
                'distance': distance,
                'grade': grade,
                'winner_name': winner_info.get('name'),
                'winner_odds': winner_info.get('odds'),
                'winner_margin': winner_info.get('margin'),
                'source': 'sportsbet',
                'raw_text': text[:200],  # First 200 chars for debugging
                'url': race_url,
                'extraction_timestamp': datetime.now().isoformat()
            }
            
            return race_result
            
        except Exception as e:
            print(f"    ⚠️ Error extracting result from element: {e}")
            return None
    
    def _extract_venue_from_text(self, text: str, url: Optional[str] = None) -> Optional[str]:
        """Extract venue name from text or URL"""
        text_lower = text.lower()
        
        # Try to find venue in text
        for venue_name, venue_code in self.venue_map.items():
            if venue_name in text_lower:
                return venue_code
        
        # Try to extract from URL if available
        if url:
            for venue_name, venue_code in self.venue_map.items():
                venue_slug = venue_name.replace(' ', '-')
                if venue_slug in url.lower():
                    return venue_code
        
        return None
    
    def _extract_race_number_from_element(self, element) -> Optional[int]:
        """Extract race number from element and its context"""
        try:
            # First try the element's own text
            text = element.text.strip()
            race_num = self._extract_race_number_from_text(text)
            if race_num:
                return race_num
            
            # Try parent elements
            parent = element
            for _ in range(3):  # Check up to 3 levels up
                try:
                    parent = parent.find_element(By.XPATH, "..")
                    parent_text = parent.text.strip()
                    race_num = self._extract_race_number_from_text(parent_text)
                    if race_num:
                        return race_num
                except:
                    break
            
            # Try sibling elements
            try:
                siblings = element.find_elements(By.XPATH, "..//*")
                for sibling in siblings[:10]:  # Check first 10 siblings
                    sibling_text = sibling.text.strip()
                    race_num = self._extract_race_number_from_text(sibling_text)
                    if race_num:
                        return race_num
            except:
                pass
            
            return None
            
        except Exception as e:
            return None
    
    def _extract_race_number_from_text(self, text: str, url: Optional[str] = None) -> Optional[int]:
        """Extract race number from text or URL"""
        # Try URL first
        if url:
            race_match = re.search(r'/race-(\d+)-', url)
            if race_match:
                return int(race_match.group(1))
        
        # Try text patterns
        race_patterns = [
            r'R(\d+)',           # R1, R2, etc.
            r'Race\s*(\d+)',     # Race 1, Race 2
            r'#(\d+)',           # #1, #2
        ]
        
        for pattern in race_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                race_num = int(match.group(1))
                if 1 <= race_num <= 20:  # Reasonable race number
                    return race_num
        
        return None
    
    def _extract_race_date_from_text(self, text: str) -> Optional[str]:
        """Extract race date from text"""
        # Look for date patterns
        date_patterns = [
            r'(\d{1,2}/\d{1,2}/\d{4})',      # DD/MM/YYYY
            r'(\d{4}-\d{2}-\d{2})',          # YYYY-MM-DD
            r'(\d{1,2}\s+\w+\s+\d{4})',      # DD Month YYYY
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                date_str = match.group(1)
                try:
                    if '/' in date_str:
                        date_obj = datetime.strptime(date_str, '%d/%m/%Y')
                    elif '-' in date_str:
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    else:
                        date_obj = datetime.strptime(date_str, '%d %B %Y')
                    
                    return date_obj.strftime('%Y-%m-%d')
                except:
                    continue
        
        # Look for relative dates
        if 'today' in text.lower():
            return datetime.now().strftime('%Y-%m-%d')
        elif 'yesterday' in text.lower():
            return (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        return None
    
    def _extract_race_time_from_text(self, text: str) -> Optional[str]:
        """Extract race time from text"""
        # Look for time patterns
        time_patterns = [
            r'(\d{1,2}:\d{2}\s*(?:AM|PM))',  # 7:45 PM
            r'(\d{1,2}:\d{2})',              # 19:45
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                time_str = match.group(1)
                try:
                    if 'AM' in time_str.upper() or 'PM' in time_str.upper():
                        return time_str.upper()
                    elif ':' in time_str:
                        # Convert 24-hour to 12-hour format
                        hour, minute = time_str.split(':')
                        hour = int(hour)
                        minute = int(minute)
                        if 0 <= hour <= 23 and 0 <= minute <= 59:
                            if hour > 12:
                                return f"{hour-12}:{minute:02d} PM"
                            elif hour == 12:
                                return f"12:{minute:02d} PM"
                            elif hour == 0:
                                return f"12:{minute:02d} AM"
                            else:
                                return f"{hour}:{minute:02d} AM"
                except:
                    continue
        
        return None
    
    def _extract_distance_from_text(self, text: str) -> Optional[str]:
        """Extract race distance from text"""
        distance_patterns = [
            r'(\d{3,4})m',
            r'(\d{3,4})\s*metre',
            r'(\d{3,4})\s*meter'
        ]
        
        for pattern in distance_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_grade_from_text(self, text: str) -> Optional[str]:
        """Extract race grade from text"""
        grade_patterns = [
            r'(Grade\s*\d+)',
            r'(G\d+)',
            r'(Maiden)',
            r'(Open)',
            r'(Novice)',
            r'(Final)',
            r'(Heat)',
            r'(Restricted)',
            r'(Mixed)',
            r'(Free For All)',
        ]
        
        for pattern in grade_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_winner_from_text(self, text: str) -> Dict:
        """Extract winner information from text"""
        winner_info = {'name': None, 'odds': None, 'margin': None}
        
        try:
        # Look for winner patterns (enhanced for Sportsbet format)
            winner_patterns = [
                # Direct winner indicators
                r'\b1\. ([A-Za-z\s\-\']+)',  # "1. Dog Name"
                r'winner[:\s]+([A-Za-z\s\-\']+)',
                r'first[:\s]+([A-Za-z\s\-\']+)',
                r'1st[:\s]+([A-Za-z\s\-\']+)',
                r'\b1\.?\s*\$?\d+(?:\.\d+)?\s+([A-Za-z\s\-\']+)',  # "1. $3.50 Dog Name"
                
                # Box + Winner patterns
                r'(?:Box|B)\s*\d+\s*\$?\d+(?:\.\d+)?\s+([A-Za-z\s\-\']+)',  # "Box 1 $3.50 Dog Name"
                r'(?:Box|B)\s*\d+\s+([A-Za-z\s\-\']+)',  # "Box 1 Dog Name"
                
                # Results with odds
                r'\$\d+(?:\.\d+)?\s+([A-Za-z\s\-\']+)',  # "$3.50 Dog Name"
                r'Paid\s+\$\d+(?:\.\d+)?\s+([A-Za-z\s\-\']+)',  # "Paid $3.50 Dog Name"
                
                # Won/Winner patterns
                r'([A-Za-z\s\-\']+)\s+won\b',
                r'([A-Za-z\s\-\']+)\s+wins\b',
                r'\bwon[:\s]+([A-Za-z\s\-\']+)',
                
                # Result-based patterns
                r'result[:\s]+([A-Za-z\s\-\']+)',
                r'finished[:\s]+([A-Za-z\s\-\']+)',
                r'placed[:\s]+([A-Za-z\s\-\']+)',
            ]
            
            for pattern in winner_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    winner_name = match.group(1).strip().title()
                    if len(winner_name) > 2 and len(winner_name) < 30:
                        winner_info['name'] = winner_name
                        break
            
            # Look for odds (enhanced)
            odds_patterns = [
                r'\$(\d+\.\d+)',         # $3.50
                r'(\d+(?:\.\d+)?)/1',    # 3/1, 5.5/1
                r'(\d+\.\d+)',          # 3.50 (decimal odds)
                r'\$(\d+)',             # $3 (whole number)
                r'(\d+(?:\.\d+)?)',     # 3.5 (decimal without $)
                r'(\d+)\s*to\s*1',      # 3 to 1
                r'(\d+)\s*-\s*1',       # 3-1
            ]
            
            for pattern in odds_patterns:
                match = re.search(pattern, text)
                if match:
                    winner_info['odds'] = match.group(1)
                    break
            
            # Look for margin (enhanced)
            margin_patterns = [
                r'(\d*\.?\d+)\s*(?:length|lengths|len|L)s?',  # 2.5 lengths, 2L
                r'by\s+(\d*\.?\d+)(?:\s*(?:length|lengths|len|L)s?)?',  # by 2.5, by 2.5L
                r'margin[:\s]+(\d*\.?\d+)',  # margin: 2.5
                r'won\s+by\s+(\d*\.?\d+)',  # won by 2.5
                r'(\d*\.?\d+)\s*m(?:argin)?',  # 2.5m
                r'(\d*\.?\d+)\s*(?=\s*(?:length|lengths|len|L)s?)',  # Number followed by length word
            ]
            
            for pattern in margin_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    winner_info['margin'] = match.group(1)
                    break
            
        except Exception as e:
            print(f"    ⚠️ Error extracting winner info: {e}")
        
        return winner_info
    
    def save_to_database(self, races: List[Dict]):
        """Save scraped races to database"""
        if not races:
            print("⚠️ No races to save")
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            saved_count = 0
            updated_count = 0
            
            for race in races:
                try:
                    # Generate race_id
                    race_id = f"{race['venue']}_{race['date']}_{race['race_number']}"
                    
                    # Check if race already exists
                    cursor.execute("SELECT race_id FROM race_metadata WHERE race_id = ?", (race_id,))
                    existing = cursor.fetchone()
                    
                    if existing:
                        # Update existing record with new information
                        cursor.execute("""
                            UPDATE race_metadata SET
                                race_name = COALESCE(?, race_name),
                                grade = COALESCE(?, grade),
                                distance = COALESCE(?, distance),
                                winner_name = COALESCE(?, winner_name),
                                winner_odds = COALESCE(?, winner_odds),
                                winner_margin = COALESCE(?, winner_margin),
                                url = COALESCE(?, url),
                                extraction_timestamp = ?
                            WHERE race_id = ?
                        """, (
                            race.get('race_name'),
                            race.get('grade'), 
                            race.get('distance'),
                            race.get('winner_name'),
                            race.get('winner_odds'),
                            race.get('winner_margin'),
                            race.get('url'),
                            race.get('extraction_timestamp'),
                            race_id
                        ))
                        updated_count += 1
                        print(f"  📝 Updated: {race_id}")
                    else:
                        # Insert new record
                        cursor.execute("""
                            INSERT INTO race_metadata (
                                race_id, venue, race_number, race_date, race_name,
                                grade, distance, winner_name, winner_odds, winner_margin,
                                url, extraction_timestamp
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            race_id,
                            race['venue'],
                            race['race_number'],
                            race['date'],
                            race.get('race_name'),
                            race.get('grade'),
                            race.get('distance'),
                            race.get('winner_name'),
                            race.get('winner_odds'),
                            race.get('winner_margin'),
                            race.get('url'),
                            race.get('extraction_timestamp')
                        ))
                        saved_count += 1
                        print(f"  ✅ Saved: {race_id}")
                        
                except Exception as e:
                    print(f"  ❌ Error saving race {race.get('venue', 'Unknown')} Race {race.get('race_number', '?')}: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            print(f"💾 Database updated: {saved_count} new races, {updated_count} updated races")
            
        except Exception as e:
            print(f"❌ Database error: {e}")

def main():
    """Main function"""
    try:
        scraper = SportsbetRecentRacesScraper()
        
        print("🏁 Starting Sportsbet recent races scraper...")
        recent_races = scraper.scrape_recent_races(days_back=3)
        
        if recent_races:
            print(f"\n📊 Found {len(recent_races)} recent races")
            
            # Save to database
            scraper.save_to_database(recent_races)
            
            # Display summary
            print(f"\n📋 Recent races summary:")
            print("=" * 60)
            
            for i, race in enumerate(recent_races[:10], 1):  # Show first 10
                winner_info = f" - Winner: {race['winner_name']}" if race.get('winner_name') else ""
                print(f"{i:2d}. {race['venue']} Race {race['race_number']} ({race['date']}){winner_info}")
                if race.get('grade'):
                    print(f"     Grade: {race['grade']}")
                if race.get('distance'):
                    print(f"     Distance: {race['distance']}m")
                print()
            
            if len(recent_races) > 10:
                print(f"... and {len(recent_races) - 10} more races")
        else:
            print("⚠️ No recent races found")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
