#!/usr/bin/env python3
"""
Sportsbet Odds Integration System
=================================

This system integrates live odds from Sportsbet with the greyhound racing prediction system.
It provides real-time market data to enhance prediction accuracy and value betting opportunities.

Features:
- Live odds collection from Sportsbet
- Integration with existing race predictions
- Value betting opportunity detection
- Market movement tracking
- Automated odds updates for upcoming races
"""

import requests
import json
import time
import re
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None
try:
    import redis
except ImportError:
    redis = None
import pandas as pd
from urllib.parse import urljoin, urlparse
import threading
import schedule
from pathlib import Path

class SportsbetOddsIntegrator:
    """Comprehensive Sportsbet odds integration system"""
    
    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.base_url = "https://www.sportsbet.com.au"
        if redis:
            self.redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
        else:
            self.redis_client = None
        self.session = requests.Session()
        self.odds_cache = {}
        self.update_interval = 30  # seconds
        self.setup_session()
        self.setup_database()
        
    def setup_session(self):
        """Setup requests session with proper headers"""
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
    def setup_database(self):
        """Setup database tables for odds storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Live odds table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS live_odds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT,
                venue TEXT,
                race_number INTEGER,
                race_date DATE,
                race_time TEXT,
                dog_name TEXT,
                dog_clean_name TEXT,
                box_number INTEGER,
                odds_decimal REAL,
                odds_fractional TEXT,
                market_type TEXT DEFAULT 'win',
                source TEXT DEFAULT 'sportsbet',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_current BOOLEAN DEFAULT TRUE
            )
        ''')
        
        # Odds history table for tracking movements
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS odds_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT,
                dog_clean_name TEXT,
                odds_decimal REAL,
                odds_change REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                source TEXT DEFAULT 'sportsbet'
            )
        ''')
        
        # Value betting opportunities table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS value_bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT,
                dog_clean_name TEXT,
                predicted_probability REAL,
                market_odds REAL,
                implied_probability REAL,
                value_percentage REAL,
                confidence_level TEXT,
                bet_recommendation TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create predictions table for value betting (optional)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT,
                dog_clean_name TEXT,
                predicted_probability REAL,
                confidence_level TEXT,
                prediction_source TEXT DEFAULT 'ml_model',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create race_metadata table for storing race times and Sportsbet URLs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS race_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT UNIQUE,
                venue TEXT,
                race_number INTEGER,
                race_date DATE,
                race_time TEXT,
                race_datetime TEXT,
                sportsbet_url TEXT,
                venue_slug TEXT,
                start_datetime TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def fetch_odds(self):
        """Fetch odds using headless Playwright"""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(self.greyhound_url)
            # Implement logic to extract odds here using page.query_selector_all or similar
            print("✅ Fetched odds successfully")
            browser.close()

    def publish_to_redis(self, key, data):
        if self.redis_client:
            self.redis_client.set(key, data)
            self.redis_client.publish('odds_updates', data)
        else:
            print("⚠️ Redis not available, skipping publish")
            
    def close_driver(self):
        """Close the web driver"""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def find_race_elements(self):
        """Find race elements using multiple strategies"""
        race_elements = []
        
        # Strategy 1: Look for standard automation IDs
        try:
            elements = self.driver.find_elements(By.CSS_SELECTOR, "[data-automation-id='race-card']")
            if elements:
                race_elements.extend(elements)
                print(f"Found {len(elements)} races using automation ID")
        except:
            pass
        
        # Strategy 2: Look for class-based selectors
        if not race_elements:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, ".race-card, [class*='race-card'], [class*='racecard']")
                if elements:
                    race_elements.extend(elements)
                    print(f"Found {len(elements)} races using class selectors")
            except:
                pass
        
        # Strategy 3: Look for links containing race information
        if not race_elements:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='/race/'], a[href*='greyhound']")
                if elements:
                    # Filter for elements that might contain race info
                    filtered_elements = []
                    for elem in elements:
                        text = elem.text.strip().lower()
                        if any(keyword in text for keyword in ['race', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9']):
                            filtered_elements.append(elem)
                    race_elements = filtered_elements
                    print(f"Found {len(race_elements)} potential races using link analysis")
            except:
                pass
        
        # Strategy 4: Generic search for elements with racing-related content
        if not race_elements:
            try:
                all_elements = self.driver.find_elements(By.CSS_SELECTOR, "div, section, article")
                for elem in all_elements:
                    try:
                        text = elem.text.strip().lower()
                        if ('race' in text and any(time in text for time in ['pm', 'am', ':']) and 
                            len(text) > 10 and len(text) < 500):
                            race_elements.append(elem)
                            if len(race_elements) >= 10:  # Limit to avoid too many false positives
                                break
                    except:
                        continue
                print(f"Found {len(race_elements)} potential races using content analysis")
            except:
                pass
        
        return race_elements
    
    def extract_race_info_from_json_ld(self):
        """Extract race info from JSON-LD structured data in page source"""
        try:
            page_source = self.driver.page_source
            
            # Find JSON-LD script tags that contain SportsEvent data
            import re
            json_pattern = r'\[\{"@context":"https://schema\.org".*?"@type":"SportsEvent".*?\}\]'
            matches = re.search(json_pattern, page_source, re.DOTALL)
            
            if not matches:
                print("⚠️  No JSON-LD SportsEvent data found")
                return []
            
            json_data_str = matches.group(0)
            
            try:
                import json
                events = json.loads(json_data_str)
                
                races = []
                for event in events:
                    if event.get('@type') == 'SportsEvent':
                        venue_name = event.get('name', 'Unknown')
                        venue_url = event.get('url', '')
                        start_date = event.get('startDate', '')
                        
                        # Parse the date and extract race info
                        if start_date and venue_url:
                            try:
                                from datetime import datetime
                                # Handle different datetime formats
                                if 'T' in start_date:
                                    race_datetime = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                                else:
                                    # Handle format like "25 Jan 2025 07:05:00"
                                    race_datetime = datetime.strptime(start_date, '%d %b %Y %H:%M:%S')
                                
                                # Extract venue slug from URL for race ID
                                venue_slug = venue_url.split('/')[-1] if '/' in venue_url else venue_name.lower().replace(' ', '-')
                                
                                race_id = f"{venue_slug}_{race_datetime.strftime('%Y%m%d')}"
                                
                                race_info = {
                                    'race_id': race_id,
                                    'venue': venue_name,
                                    'venue_slug': venue_slug,
                                    'race_date': race_datetime.date(),
                                    'race_time': race_datetime.strftime('%H:%M'),
                                    'start_datetime': race_datetime,
                                    'venue_url': venue_url,
                                    'odds_data': []  # Will be populated by scraping individual race pages
                                }
                                
                                races.append(race_info)
                                
                            except Exception as e:
                                print(f"⚠️  Error parsing race datetime for {venue_name}: {e}")
                                continue
                
                print(f"✅ Extracted {len(races)} races from JSON-LD data")
                return races
                
            except json.JSONDecodeError as e:
                print(f"⚠️  Error parsing JSON-LD data: {e}")
                return []
                
        except Exception as e:
            print(f"⚠️  Error extracting JSON-LD race info: {e}")
            return []
    
    def extract_race_info_flexible(self, element, index):
        """Extract race info using flexible methods - legacy fallback"""
        try:
            # Try to extract basic info from the element
            text = element.text.strip()
            
            # Parse race info from element text (fallback method)
            # Example: "R7 Townsville\n1m 16s"
            lines = text.split('\n')
            if len(lines) >= 2:
                first_line = lines[0].strip()
                second_line = lines[1].strip()
                
                # Extract race number and venue
                race_match = re.match(r'R(\d+)\s+(.+)', first_line)
                if race_match:
                    race_number = race_match.group(1)
                    venue_name = race_match.group(2).strip()
                    
                    # Create race ID
                    venue_slug = venue_name.lower().replace(' ', '-')
                    race_id = f"{venue_slug}_{race_number}_{datetime.now().strftime('%Y%m%d')}"
                    
                    # Parse time info (e.g., "1m 16s" means 1 minute 16 seconds from now)
                    race_time = "Unknown"
                    try:
                        if 'm' in second_line and 's' in second_line:
                            # Calculate actual race time from countdown
                            time_match = re.match(r'(\d+)m\s*(\d+)s?', second_line)
                            if time_match:
                                minutes = int(time_match.group(1))
                                seconds = int(time_match.group(2)) if time_match.group(2) else 0
                                
                                race_datetime = datetime.now() + timedelta(minutes=minutes, seconds=seconds)
                                race_time = race_datetime.strftime('%H:%M')
                    except:
                        pass
                    
                    return {
                        'race_id': race_id,
                        'venue': venue_name,
                        'race_number': race_number,
                        'race_date': datetime.now().date(),
                        'race_time': race_time,
                        'odds_data': []  # No odds available from this method
                    }
            
            return None
            
        except Exception as e:
            print(f"⚠️  Error in flexible extraction: {e}")
            return None
    
    def extract_races_from_dom(self) -> List[Dict]:
        """Extract upcoming races from DOM table with countdown timers"""
        races = []
        
        try:
            print("🔍 Looking for race table rows...")
            
            # Find venue rows in the racing table
            venue_selectors = [
                "tr",  # Table rows
                "[class*='venue']",
                "[class*='race-row']"
            ]
            
            rows = []
            for selector in venue_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if len(elements) > 5:  # Need reasonable number of rows
                        rows = elements
                        print(f"  📊 Found {len(rows)} rows using selector: {selector}")
                        break
                except:
                    continue
            
            if not rows:
                print("  ⚠️  No table rows found")
                return []
            
            current_time = datetime.now()
            
            for row in rows[:20]:  # Check first 20 rows
                try:
                    row_text = row.text.strip()
                    if len(row_text) < 10:  # Skip empty or very short rows
                        continue
                    
                    # Look for venue names (should be at start of row)
                    venue_match = re.match(r'^([A-Za-z\s]+)\s+Australia', row_text)
                    if not venue_match:
                        continue
                    
                    venue_name = venue_match.group(1).strip()
                    print(f"  🏟️  Processing venue: {venue_name}")
                    
                    # Look for countdown timers in the row (like "17m", "32m")
                    countdown_matches = re.findall(r'(\d+)m(?:\s|$)', row_text)
                    
                    if countdown_matches:
                        print(f"    ⏰ Found countdowns: {countdown_matches}")
                        
                        # For each countdown, create a race entry
                        for i, minutes_str in enumerate(countdown_matches[:3]):  # Max 3 races per venue
                            try:
                                minutes = int(minutes_str)
                                if minutes > 60:  # Skip if too far in future
                                    continue
                                
                                # Calculate race time
                                race_datetime = current_time + timedelta(minutes=minutes)
                                
                                # Try to find the specific race link - look for meeting links first
                                meeting_selectors = [
                                    "a[href*='meeting-']",
                                    "a[href*='/meeting/']",
                                    "a[href*='race-']",
                                    "a[href*='/race/']"
                                ]
                                
                                race_url = None
                                print(f"      🔍 Looking for race/meeting links in row...")
                                
                                for selector in meeting_selectors:
                                    try:
                                        race_links = row.find_elements(By.CSS_SELECTOR, selector)
                                        if race_links:
                                            print(f"        📋 Found {len(race_links)} links with selector: {selector}")
                                            # Debug: Print first few links
                                            for j, link in enumerate(race_links[:3]):
                                                href = link.get_attribute('href')
                                                text = link.text.strip()[:50]
                                                print(f"          {j+1}. {href} ('{text}')")
                                            
                                            # Try to match the countdown with the appropriate race link
                                            selected_url = None
                                            selected_race_number = None
                                            
                                            # Look for race link that matches this countdown
                                            for link in race_links:
                                                try:
                                                    href = link.get_attribute('href')
                                                    link_text = link.text.strip().lower()
                                                    
                                                    # Check if this link text contains our countdown
                                                    if f'{minutes}m' in link_text or f'{minutes} m' in link_text:
                                                        selected_url = href
                                                        # Extract race number from URL
                                                        race_match = re.search(r'/race-(\d+)-', href)
                                                        if race_match:
                                                            selected_race_number = race_match.group(1)
                                                        print(f"        🎯 Matched countdown {minutes}m with race: {href}")
                                                        break
                                                except:
                                                    continue
                                            
                                            # Fallback: use race link by position (i-th countdown gets i-th race)
                                            if not selected_url and i < len(race_links):
                                                selected_url = race_links[i].get_attribute('href')
                                                race_match = re.search(r'/race-(\d+)-', selected_url)
                                                if race_match:
                                                    selected_race_number = race_match.group(1)
                                                print(f"        📍 Using position-based selection: {selected_url}")
                                            
                                            # Final fallback: use first race link
                                            if not selected_url:
                                                selected_url = race_links[0].get_attribute('href')
                                                race_match = re.search(r'/race-(\d+)-', selected_url)
                                                if race_match:
                                                    selected_race_number = race_match.group(1)
                                                print(f"        🔄 Using first race as fallback: {selected_url}")
                                            
                                            race_url = selected_url
                                            race_number = selected_race_number or str(i + 1)
                                            print(f"        ✅ Selected URL: {race_url} (Race {race_number})")
                                            break
                                    except Exception as e:
                                        print(f"        ⚠️  Error with selector {selector}: {e}")
                                        continue
                                
                                if not race_url:
                                    print(f"      ⚠️  No meeting/race links found, using fallback venue URL...")
                                    # Fallback: Create venue URL
                                    venue_slug = venue_name.lower().replace(' ', '-')
                                    race_url = f"{self.base_url}/greyhound-racing/australia-nz/{venue_slug}"
                                    race_number = str(i + 1)
                                    print(f"        🔄 Fallback URL: {race_url}")
                                
                                # Create race info
                                venue_slug = venue_name.lower().replace(' ', '-')
                                race_id = f"{venue_slug}_{current_time.strftime('%Y%m%d')}"
                                
                                race_info = {
                                    'race_id': race_id,
                                    'venue': venue_name,
                                    'venue_slug': venue_slug,
                                    'race_number': race_number,
                                    'race_date': current_time.date(),
                                    'race_time': race_datetime.strftime('%H:%M'),
                                    'start_datetime': race_datetime,
                                    'venue_url': race_url,
                                    'odds_data': [],
                                    'countdown_minutes': minutes
                                }
                                
                                races.append(race_info)
                                print(f"    ✅ Added: {venue_name} in {minutes}min at {race_datetime.strftime('%H:%M')}")
                                
                            except Exception as e:
                                print(f"    ⚠️  Error processing countdown {minutes_str}: {e}")
                                continue
                    
                except Exception as e:
                    print(f"  ⚠️  Error processing row: {e}")
                    continue
            
            print(f"📊 Extracted {len(races)} upcoming races from DOM")
            return races
            
        except Exception as e:
            print(f"⚠️  Error extracting races from DOM: {e}")
            return []
            
    def get_race_odds_from_page(self, race_info: Dict) -> Dict:
        """Navigate to individual race page and extract live odds"""
        try:
            venue_url = race_info.get('venue_url')
            if not venue_url:
                print(f"⚠️  No venue URL for race {race_info['race_id']}, skipping odds extraction")
                return race_info
            
            # Construct full race URL
            if venue_url.startswith('/'):  # Relative URL
                full_url = urljoin(self.base_url, venue_url)
            else:
                full_url = venue_url
            
            print(f"  📄 Navigating to race page: {full_url}")
            
            # Check if this is a meeting page that needs further navigation
            if '/meeting-' in full_url:
                # Navigate to meeting page first to find individual race
                individual_race_url = self.find_next_race_from_meeting(full_url)
                if individual_race_url:
                    full_url = individual_race_url
                    print(f"  🎯 Found individual race URL: {full_url}")
                else:
                    print(f"  ⚠️  Could not find individual race from meeting page")
                    return race_info
            
            # Navigate to the race page
            self.driver.get(full_url)
            
            # Wait for page to load
            try:
                WebDriverWait(self.driver, 10).until(
                    lambda driver: driver.execute_script("return document.readyState") == "complete"
                )
            except TimeoutException:
                print(f"  ⚠️  Timeout waiting for page to load")
            
            # Give additional time for dynamic content to load
            print(f"  ⏳ Waiting for dynamic content to load...")
            time.sleep(5)
            
            # Debug: Print page info
            print(f"  🔍 Page loaded: {self.driver.title}")
            print(f"  📊 Page has {len(self.driver.find_elements(By.CSS_SELECTOR, 'button'))} buttons total")
            
            # Try multiple strategies to extract live odds
            odds_data = []
            
            # Strategy 1: Look for runner cards with odds
            odds_data = self.extract_odds_strategy_runner_cards()
            
            if not odds_data:
                # Strategy 2: Try to match runners with separate odds buttons
                odds_data = self.extract_odds_strategy_separate_buttons()
            
            if not odds_data:
                # Strategy 3: Look for table-based odds display
                odds_data = self.extract_odds_strategy_table()
            
            if not odds_data:
                # Strategy 4: Look for any elements with odds-like patterns
                odds_data = self.extract_odds_strategy_generic()
            
            if odds_data:
                print(f"  ✅ Extracted {len(odds_data)} live odds")
                race_info['odds_data'] = odds_data
                
                # Try to extract race number for better identification
                race_number = self.extract_race_number_from_page(race_info['venue'])
                if race_number:
                    race_info['race_number'] = race_number
                    # Update race_id to include race number
                    venue_slug = race_info.get('venue_slug', race_info['venue'].lower().replace(' ', '-'))
                    race_info['race_id'] = f"{venue_slug}_r{race_number}_{race_info['start_datetime'].strftime('%Y%m%d')}"
                    print(f"  📍 Updated race ID to: {race_info['race_id']}")
            else:
                print(f"  ⚠️  No live odds found on race page")
                # Try to get race information from the page even without odds
                race_number = self.extract_race_number_from_page(race_info['venue'])
                if race_number:
                    race_info['race_number'] = race_number
            
            return race_info
            
        except Exception as e:
            print(f"  ⚠️  Error extracting odds from race page: {e}")
            return race_info
    
    def extract_odds_strategy_runner_cards(self) -> List[Dict]:
        """Extract odds using robust Sportsbet-specific DOM selectors with comprehensive fallbacks"""
        odds_data = []
        
        try:
            print(f"  🔍 Looking for Sportsbet runner containers...")
            
            # Enhanced wait for dynamic content with multiple selectors
            print(f"  ⏳ Waiting for dynamic content to load...")
            try:
                # Wait for price elements to appear (indicates odds are loaded)
                WebDriverWait(self.driver, 12).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "[data-automation-id='price-text']"))
                )
                print(f"    ✅ Price elements loaded successfully")
            except TimeoutException:
                print(f"    ⚠️  Timeout waiting for price elements, trying alternative selectors...")
                # Try alternative wait selectors
                try:
                    WebDriverWait(self.driver, 5).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "button.price-button, span[class*='priceText'], .priceContainer"))
                    )
                    print(f"    ✅ Alternative price elements found")
                except TimeoutException:
                    print(f"    ⚠️  No price elements found, proceeding anyway...")
            
            # Additional wait time for complex loading
            time.sleep(2)
            
            # Find all runner containers using the specific Sportsbet selector
            cards = self.driver.find_elements(
                By.CSS_SELECTOR, 
                "div[data-automation-id^='racecard-outcome-']"
            )
            
            if not cards:
                print(f"  ⚠️  No Sportsbet runner containers found, falling back to broader approach...")
                return self._extract_from_broader_containers()
            
            print(f"  📊 Found {len(cards)} Sportsbet runner containers")
            
            # Process each card individually with comprehensive error handling
            successful_extractions = 0
            
            for i, card in enumerate(cards[:8]):  # Max 8 runners
                print(f"  🐕 Processing runner card {i+1}/{min(len(cards), 8)}...")
                
                # Extract dog name with multiple fallback strategies
                dog_name = self._extract_dog_name_with_fallbacks(card, i+1)
                
                # Extract odds with multiple fallback strategies  
                odds_decimal, odds_text = self._extract_odds_with_fallbacks(card, i+1)
                
                # Add to results if we have both name and odds
                if dog_name and odds_decimal > 0:
                    odds_data.append({
                        'dog_name': dog_name,
                        'dog_clean_name': self.clean_dog_name(dog_name),
                        'box_number': i + 1,
                        'odds_decimal': odds_decimal,
                        'odds_fractional': odds_text
                    })
                    successful_extractions += 1
                    print(f"    ✅ Successfully extracted: {dog_name} - ${odds_decimal:.2f}")
                elif dog_name:
                    print(f"    ⚠️  Found dog name '{dog_name}' but no odds for card {i+1}")
                elif odds_decimal > 0:
                    print(f"    ⚠️  Found odds ${odds_decimal:.2f} but no dog name for card {i+1}")
                else:
                    print(f"    ❌ No data extracted from card {i+1} (likely scratched or loading issue)")
            
            print(f"  📊 Extraction summary: {successful_extractions}/{len(cards)} cards successfully processed")
            
            # Debug: Save screenshot/source if insufficient data found
            if successful_extractions < 4:
                print(f"  🚨 WARNING: Only found {successful_extractions} complete runners (expected 4+)")
                self._save_debug_info(successful_extractions)
            
            if odds_data:
                print(f"  🎯 Sportsbet extraction SUCCESS: Found {len(odds_data)} complete runner odds")
                
                # Format output as requested JSON structure
                formatted_output = []
                for runner in odds_data:
                    formatted_output.append({
                        "dog": runner['dog_name'],
                        "odds": f"{runner['odds_decimal']:.2f}"
                    })
                
                print(f"  📋 Final JSON output:")
                import json
                print(f"    {json.dumps(formatted_output, indent=2)}")
                
                return odds_data
            else:
                print(f"  ⚠️  No complete runner data extracted, trying fallback strategies...")
                return self._extract_from_broader_containers()
                
        except Exception as e:
            print(f"  ⚠️  Error in robust Sportsbet runner card strategy: {e}")
            # Fallback to broader container approach
            return self._extract_from_broader_containers()
        
        return odds_data
    
    def _extract_dog_name_with_fallbacks(self, card, card_number: int) -> str:
        """Extract dog name with comprehensive fallback strategies"""
        import re  # Import re at the beginning of the function
        dog_name = ""
        
        # Strategy 1: Primary selector
        try:
            name_element = card.find_element(
                By.CSS_SELECTOR, 
                "div[data-automation-id='racecard-outcome-name'] span"
            )
            raw_name = name_element.text.strip()
            if raw_name:
                # Clean the name - remove number prefix like "2. My Roadster Boy" -> "My Roadster Boy"
                cleaned_name = re.sub(r'^\d+\.\s*', '', raw_name)
                dog_name = cleaned_name
                print(f"    📝 Found dog name (primary): '{raw_name}' -> '{dog_name}'")
                return dog_name
        except Exception as e:
            print(f"    ⚠️  Primary dog name selector failed for card {card_number}: {e}")
        
        # Strategy 2: Fallback to .runnerInfo class
        try:
            runner_info = card.find_element(By.CSS_SELECTOR, ".runnerInfo")
            name_spans = runner_info.find_elements(By.CSS_SELECTOR, "span")
            for span in name_spans:
                text = span.text.strip()
                if text and len(text) > 2 and not re.match(r'^[\d\.\s/]+$', text):
                    cleaned_name = re.sub(r'^\d+\.\s*', '', text)
                    if cleaned_name != text:
                        dog_name = cleaned_name
                        print(f"    📝 Found dog name (runnerInfo): '{text}' -> '{dog_name}'")
                        return dog_name
        except Exception as e:
            print(f"    ⚠️  runnerInfo fallback failed for card {card_number}: {e}")
        
        # Strategy 3: Fallback to .outcomeName class
        try:
            outcome_name = card.find_element(By.CSS_SELECTOR, ".outcomeName")
            text = outcome_name.text.strip()
            if text:
                cleaned_name = re.sub(r'^\d+\.\s*', '', text)
                dog_name = cleaned_name
                print(f"    📝 Found dog name (outcomeName): '{text}' -> '{dog_name}'")
                return dog_name
        except Exception as e:
            print(f"    ⚠️  outcomeName fallback failed for card {card_number}: {e}")
        
        # Strategy 4: Look for any data-automation-id containing 'name'
        try:
            name_elements = card.find_elements(By.CSS_SELECTOR, "[data-automation-id*='name']")
            for elem in name_elements:
                text = elem.text.strip()
                if text and len(text) > 3 and len(text) < 30:
                    cleaned_name = re.sub(r'^\d+\.\s*', '', text)
                    if re.match(r'^[A-Z]', cleaned_name):
                        dog_name = cleaned_name
                        print(f"    📝 Found dog name (automation-id name): '{text}' -> '{dog_name}'")
                        return dog_name
        except Exception as e:
            print(f"    ⚠️  automation-id name fallback failed for card {card_number}: {e}")
        
        # Strategy 5: Deep search in any span with reasonable text
        try:
            all_spans = card.find_elements(By.CSS_SELECTOR, "span")
            for span in all_spans:
                text = span.text.strip()
                # Look for text that could be a dog name (not odds, not numbers only)
                if (text and 
                    len(text) > 3 and len(text) < 30 and
                    not re.match(r'^[\d\.\s/]+$', text) and  # Not just numbers/odds
                    not text.lower() in ['bet', 'win', 'place', 'show', 'each way', 'ew'] and
                    re.search(r'[a-zA-Z]', text)):  # Contains letters
                    
                    cleaned_name = re.sub(r'^\d+\.\s*', '', text)
                    # Basic validation - should start with uppercase letter
                    if re.match(r'^[A-Z]', cleaned_name):
                        dog_name = cleaned_name
                        print(f"    📝 Found dog name (deep search): '{text}' -> '{dog_name}'")
                        return dog_name
        except Exception as e:
            print(f"    ⚠️  Deep search fallback failed for card {card_number}: {e}")
        
        # Strategy 6: Get card text and analyze it
        try:
            card_text = card.text.strip()
            if card_text:
                print(f"    🔍 Analyzing card text for names: '{card_text[:100]}...'")
                lines = card_text.split('\n')
                for line in lines:
                    line = line.strip()
                    # Look for lines that look like dog names
                    if (line and len(line) > 3 and len(line) < 30 and
                        not re.match(r'^[\d\.\s/$]+$', line) and  # Not just numbers/odds/symbols
                        re.search(r'[A-Za-z]{3,}', line)):  # Contains at least 3 letters
                        
                        # Clean potential dog name
                        cleaned_name = re.sub(r'^\d+\.\s*', '', line)
                        cleaned_name = re.sub(r'\s*\([^)]*\)\s*', '', cleaned_name)  # Remove parentheses content
                        cleaned_name = cleaned_name.strip()
                        
                        # Validate it looks like a dog name
                        if (cleaned_name and len(cleaned_name) > 3 and
                            re.match(r'^[A-Z][a-zA-Z\s]+$', cleaned_name) and
                            not any(word in cleaned_name.lower() for word in 
                                   ['barrier', 'weight', 'trainer', 'jockey', 'form', 'speed', 'time'])):
                            dog_name = cleaned_name
                            print(f"    📝 Found dog name (text analysis): '{line}' -> '{dog_name}'")
                            return dog_name
        except Exception as e:
            print(f"    ⚠️  Text analysis fallback failed for card {card_number}: {e}")
        
        print(f"    ❌ No dog name found for card {card_number}")
        return ""
    
    def _extract_odds_with_fallbacks(self, card, card_number: int) -> tuple[float, str]:
        """Extract odds with comprehensive fallback strategies"""
        odds_decimal = 0.0
        odds_text = ""
        
        # Strategy 1: Primary selector
        try:
            odds_element = card.find_element(
                By.CSS_SELECTOR, 
                "div[data-automation-id='price-text'] span"
            )
            odds_text = odds_element.text.strip()
            if odds_text:
                odds_decimal = self.parse_odds_to_decimal(odds_text)
                if odds_decimal > 0:
                    print(f"    💰 Found odds (primary): '{odds_text}' -> ${odds_decimal:.2f}")
                    return odds_decimal, odds_text
        except Exception as e:
            print(f"    ⚠️  Primary odds selector failed for card {card_number}: {e}")
        
        # Strategy 2: Fallback to button.price-button
        try:
            price_button = card.find_element(By.CSS_SELECTOR, "button.price-button")
            odds_text = price_button.text.strip()
            if odds_text:
                odds_decimal = self.parse_odds_to_decimal(odds_text)
                if odds_decimal > 0:
                    print(f"    💰 Found odds (price-button): '{odds_text}' -> ${odds_decimal:.2f}")
                    return odds_decimal, odds_text
        except Exception as e:
            print(f"    ⚠️  price-button fallback failed for card {card_number}: {e}")
        
        # Strategy 3: Fallback to span[class*='priceText']
        try:
            price_spans = card.find_elements(By.CSS_SELECTOR, "span[class*='priceText']")
            for span in price_spans:
                odds_text = span.text.strip()
                if odds_text:
                    odds_decimal = self.parse_odds_to_decimal(odds_text)
                    if odds_decimal > 0:
                        print(f"    💰 Found odds (priceText): '{odds_text}' -> ${odds_decimal:.2f}")
                        return odds_decimal, odds_text
        except Exception as e:
            print(f"    ⚠️  priceText fallback failed for card {card_number}: {e}")
        
        # Strategy 4: Fallback to .priceContainer children
        try:
            price_container = card.find_element(By.CSS_SELECTOR, ".priceContainer")
            price_elements = price_container.find_elements(By.CSS_SELECTOR, "span, button, div")
            for elem in price_elements:
                odds_text = elem.text.strip()
                if odds_text and re.match(r'^\d+\.\d{1,2}$', odds_text):
                    potential_odds = float(odds_text)
                    if 1.01 <= potential_odds <= 50.0:
                        odds_decimal = potential_odds
                        print(f"    💰 Found odds (priceContainer): '{odds_text}' -> ${odds_decimal:.2f}")
                        return odds_decimal, odds_text
        except Exception as e:
            print(f"    ⚠️  priceContainer fallback failed for card {card_number}: {e}")
        
        # Strategy 5: Generic button search with odds pattern
        try:
            buttons = card.find_elements(By.CSS_SELECTOR, "button")
            for button in buttons:
                button_text = button.text.strip()
                if button_text and re.match(r'^\d+\.\d{1,2}$', button_text):
                    potential_odds = float(button_text)
                    if 1.01 <= potential_odds <= 50.0:
                        odds_decimal = potential_odds
                        odds_text = button_text
                        print(f"    💰 Found odds (generic button): '{odds_text}' -> ${odds_decimal:.2f}")
                        return odds_decimal, odds_text
        except Exception as e:
            print(f"    ⚠️  Generic button search failed for card {card_number}: {e}")
        
        # Strategy 6: Deep search for any element with odds pattern
        try:
            all_elements = card.find_elements(By.CSS_SELECTOR, "span, div, button")
            for elem in all_elements:
                text = elem.text.strip()
                if text and re.match(r'^\d+\.\d{1,2}$', text):
                    potential_odds = float(text)
                    if 1.01 <= potential_odds <= 50.0:
                        odds_decimal = potential_odds
                        odds_text = text
                        print(f"    💰 Found odds (deep search): '{odds_text}' -> ${odds_decimal:.2f}")
                        return odds_decimal, odds_text
        except Exception as e:
            print(f"    ⚠️  Deep search for odds failed for card {card_number}: {e}")
        
        print(f"    ❌ No odds found for card {card_number}")
        return 0.0, ""
    
    def _save_debug_info(self, successful_extractions: int):
        """Save debug information when insufficient data is found"""
        try:
            import os
            from datetime import datetime
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            debug_dir = "debug_screenshots"
            
            # Create debug directory if it doesn't exist
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            
            # Save screenshot
            screenshot_path = f"{debug_dir}/sportsbet_debug_{timestamp}_{successful_extractions}odds.png"
            try:
                self.driver.save_screenshot(screenshot_path)
                print(f"    📸 Debug screenshot saved: {screenshot_path}")
            except Exception as e:
                print(f"    ⚠️  Could not save screenshot: {e}")
            
            # Save page source
            source_path = f"{debug_dir}/sportsbet_debug_{timestamp}_{successful_extractions}odds.html"
            try:
                with open(source_path, 'w', encoding='utf-8') as f:
                    f.write(self.driver.page_source)
                print(f"    📄 Debug page source saved: {source_path}")
            except Exception as e:
                print(f"    ⚠️  Could not save page source: {e}")
            
            # Save URL for reference
            print(f"    🔗 Current URL: {self.driver.current_url}")
            
        except Exception as e:
            print(f"    ⚠️  Error saving debug info: {e}")
    
    def _extract_from_broader_containers(self) -> List[Dict]:
        """Extract from broader containers when individual runners don't work"""
        odds_data = []
        try:
            broader_selectors = [
                "tr",  # Table rows are most likely to contain both
                "[class*='market-row']",  # Market row containers
                "[class*='outcome-row']",  # Outcome row containers  
                "[class*='selection-row']",  # Selection row containers
                "[class*='runner-row']",  # Runner row containers
                "[class*='grid-item']",  # Grid item containers
                "[class*='card']",  # Card containers
                "[role='row']",  # ARIA row elements
            ]
            
            print(f"  🔍 Looking for broader containers that include both runner info and odds...")
            
            potential_containers = []
            for selector in broader_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    print(f"    📊 Found {len(elements)} elements with selector: {selector}")
                    
                    # Test each element to see if it contains both runner info and odds
                    for elem in elements:
                        try:
                            # Check if it contains runner info
                            has_runner_info = bool(elem.find_elements(By.CSS_SELECTOR, "[class*='runner'], [data-automation-id*='name']"))
                            
                            # Check if it contains odds buttons
                            buttons = elem.find_elements(By.CSS_SELECTOR, "button")
                            has_odds = False
                            for btn in buttons:
                                btn_text = btn.text.strip()
                                if btn_text and re.match(r'^\d+\.\d{1,2}$', btn_text):
                                    potential_odds = float(btn_text)
                                    if 1.01 <= potential_odds <= 50.0:
                                        has_odds = True
                                        break
                            
                            if has_runner_info and has_odds:
                                potential_containers.append(elem)
                                print(f"      ✅ Found container with both runner info and odds")
                                
                        except:
                            continue
                    
                    if potential_containers:
                        print(f"    🎯 Found {len(potential_containers)} viable containers with selector: {selector}")
                        break
                        
                except:
                    continue
            
            if potential_containers:
                print(f"  📋 Processing {len(potential_containers)} containers with both runner info and odds")
                
                # IMPROVED: Extract ALL runners and their odds, not just unique ones
                all_valid_containers = []
                
                for container in potential_containers:
                    try:
                        # Quick check to extract dog name from this container
                        runner_elements = container.find_elements(By.CSS_SELECTOR, "[class*='runner'], [data-automation-id*='name']")
                        temp_dog_name = ""
                        
                        for runner_elem in runner_elements:
                            runner_text = runner_elem.text.strip()
                            extracted_name = self.extract_dog_name_from_text(runner_text)
                            if extracted_name:
                                temp_dog_name = extracted_name
                                break
                        
                        # Add all containers with valid dog names (allow duplicates for now)
                        if temp_dog_name:
                            all_valid_containers.append((container, temp_dog_name))
                            
                    except:
                        continue
                
                # Sort containers by their position on page to maintain runner order
                try:
                    all_valid_containers.sort(key=lambda x: x[0].location['y'])
                except:
                    pass  # If location fails, keep original order
                
                # Now deduplicate while preserving order
                seen_dogs = set()
                unique_containers = []
                for container, dog_name in all_valid_containers:
                    if dog_name not in seen_dogs:
                        seen_dogs.add(dog_name)
                        unique_containers.append((container, dog_name))
                
                print(f"  🔍 Found {len(unique_containers)} unique dog containers after deduplication (from {len(all_valid_containers)} total)")
                
                for i, (container, expected_dog_name) in enumerate(unique_containers[:8]):  # Max 8 runners
                    try:
                        # Extract runner info from this container
                        runner_elements = container.find_elements(By.CSS_SELECTOR, "[class*='runner'], [data-automation-id*='name']")
                        dog_name = ""
                        
                        for runner_elem in runner_elements:
                            runner_text = runner_elem.text.strip()
                            extracted_name = self.extract_dog_name_from_text(runner_text)
                            if extracted_name:
                                dog_name = extracted_name
                                break
                        
                        # Extract odds from this container
                        odds_decimal = 0.0
                        odds_text = ""
                        
                        buttons = container.find_elements(By.CSS_SELECTOR, "button")
                        for button in buttons:
                            button_text = button.text.strip()
                            if re.match(r'^\d+\.\d{1,2}$', button_text):
                                potential_odds = float(button_text)
                                if 1.01 <= potential_odds <= 50.0:
                                    # Verify this isn't a box number by checking parent context
                                    try:
                                        parent_class = button.get_attribute('class') or ''
                                        parent_element = button.find_element(By.XPATH, '..')
                                        parent_parent_class = parent_element.get_attribute('class') or ''
                                        
                                        # Skip if it looks like box numbers (in tab containers)
                                        if 'tab' in parent_parent_class.lower() or 'circle' in parent_parent_class.lower():
                                            continue
                                            
                                        odds_decimal = potential_odds
                                        odds_text = button_text
                                        print(f"      🎯 Found odds button in container: {button_text}")
                                        break
                                    except:
                                        odds_decimal = potential_odds
                                        odds_text = button_text
                                        print(f"      🎯 Found odds button in container: {button_text}")
                                        break
                        
                        if dog_name and odds_decimal > 0:
                            print(f"    ✅ Container {i+1}: Found {dog_name} - ${odds_decimal:.2f}")
                            odds_data.append({
                                'dog_name': dog_name,
                                'dog_clean_name': self.clean_dog_name(dog_name),
                                'box_number': i + 1,
                                'odds_decimal': odds_decimal,
                                'odds_fractional': odds_text
                            })
                        else:
                            print(f"    ⚠️  Container {i+1}: Missing data - name='{dog_name}', odds={odds_decimal}")
                            
                    except Exception as e:
                        print(f"    ⚠️  Error processing container {i+1}: {e}")
                        continue
                
                if odds_data:
                    print(f"  🎯 Strategy 1 SUCCESS: Extracted {len(odds_data)} odds from broader containers")
                    return odds_data
            
            # FALLBACK: Original approach with individual runner elements
            print(f"  🔄 No broad containers found, falling back to individual runner elements...")
            
            # Look for runner/dog cards with various selectors
            selectors_to_try = [
                "[class*='runner']",  # Try this first based on debug output
                "[data-automation-id*='runner']",
                "[data-automation-id*='selection']", 
                "[class*='selection']",
                "[class*='participant']",
                "[class*='competitor']",
                "[class*='dog']",  # Add dog-specific selector
                "[class*='entry']",  # Generic entry selector
                "tr[class*='row']",  # Table row approach for some venues
                "div[class*='item']",  # Generic item containers
            ]
            
            runners = []
            for selector in selectors_to_try:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements and len(elements) >= 4:  # At least 4 for a valid race
                        # MUCH better filtering for runner elements
                        filtered_elements = []
                        seen_names = set()
                        
                        for elem in elements:
                            try:
                                elem_text = elem.text.strip()
                                elem_text_lower = elem_text.lower()
                                
                                # Skip if too short or too long
                                if len(elem_text) < 15 or len(elem_text) > 500:
                                    continue
                                
                                # Skip obvious non-runner elements
                                if any(skip_word in elem_text_lower for skip_word in 
                                      ['header', 'footer', 'navigation', 'menu', 'advertisement', 'sponsored', 
                                       'betting', 'login', 'signup', 'place bet', 'live streaming']):
                                    continue
                                
                                # Must contain dog-related indicators
                                has_dog_indicators = (
                                    re.search(r'\d+\.\s*[A-Z][a-z]+', elem_text) or  # "1. DogName"
                                    ('trainer' in elem_text_lower and ('form' in elem_text_lower or 'speed' in elem_text_lower)) or
                                    (re.search(r'\([1-8]\)', elem_text) and len([line for line in elem_text.split('\n') if line.strip()]) >= 3)
                                )
                                
                                if not has_dog_indicators:
                                    continue
                                
                                # Extract potential dog name to avoid duplicates
                                potential_name = self.extract_dog_name_from_text(elem_text)
                                if potential_name and potential_name not in seen_names:
                                    seen_names.add(potential_name)
                                    filtered_elements.append(elem)
                                
                            except:
                                continue
                        
                        if len(filtered_elements) >= 4:
                            runners = filtered_elements[:8]  # Limit to max 8
                            print(f"  📋 Found {len(runners)} unique runners using selector: {selector}")
                            break
                except:
                    continue
            
            if not runners:
                return []
            
            for i, runner in enumerate(runners):
                try:
                    # Extract dog name from runner element text
                    runner_text = runner.text.strip()
                    print(f"  🐕 Debug - Runner {i+1} text: {runner_text[:100]}...")
                    
                    # Extract dog name with improved logic
                    dog_name = self.extract_dog_name_from_text(runner_text)
                    
                    # Try to find odds within this runner element
                    odds_decimal = 0.0
                    odds_text = ""
                    
                    try:
                        # Strategy 1: Look for buttons with odds within the runner element
                        buttons = runner.find_elements(By.CSS_SELECTOR, "button")
                        print(f"    🔍 Found {len(buttons)} buttons in runner element")
                        for j, button in enumerate(buttons):
                            button_text = button.text.strip()
                            print(f"      Button {j+1}: '{button_text}'")
                            if re.match(r'^\d+\.\d{1,2}$', button_text):  # Decimal odds like "2.50" or "2.5"
                                odds_decimal = float(button_text)
                                odds_text = button_text
                                print(f"      ✅ Found odds in button: {odds_text}")
                                break
                            elif '/' in button_text and re.match(r'^\d+/\d+$', button_text):  # Fractional like "5/2"
                                parts = button_text.split('/')
                                if len(parts) == 2:
                                    odds_decimal = (float(parts[0]) / float(parts[1])) + 1.0
                                    odds_text = button_text
                                    print(f"      ✅ Found fractional odds in button: {odds_text} = {odds_decimal:.2f}")
                                    break
                        
                        # Strategy 2: Look for odds in span/div elements within runner
                        if odds_decimal == 0.0:
                            odds_elements = runner.find_elements(By.CSS_SELECTOR, "span, div, [class*='odds'], [class*='price']")
                            print(f"    🔍 Found {len(odds_elements)} potential odds elements in runner")
                            for j, elem in enumerate(odds_elements[:10]):  # Check first 10
                                elem_text = elem.text.strip()
                                if elem_text and re.match(r'^\d+\.\d{1,2}$', elem_text):
                                    potential_odds = float(elem_text)
                                    if 1.01 <= potential_odds <= 50.0:  # Realistic odds range
                                        odds_decimal = potential_odds
                                        odds_text = elem_text
                                        print(f"      ✅ Found odds in element {j+1}: {odds_text}")
                                        break
                        
                        # Strategy 3: Look for odds in the runner text itself
                        if odds_decimal == 0.0:
                            print(f"    🔍 Searching for odds patterns in runner text...")
                            lines = runner_text.split('\n')
                            for line_num, line in enumerate(lines):
                                line = line.strip()
                                # Look for decimal odds patterns
                                odds_matches = re.findall(r'(\d+\.\d{1,2})', line)
                                for odds_match in odds_matches:
                                    potential_odds = float(odds_match)
                                    if 1.01 <= potential_odds <= 50.0:  # Realistic odds range
                                        odds_decimal = potential_odds
                                        odds_text = odds_match
                                        print(f"      ✅ Found odds in text line {line_num+1}: '{line}' -> {odds_text}")
                                        break
                                if odds_decimal > 0:
                                    break
                                        
                    except Exception as e:
                        print(f"    ⚠️  Error extracting odds from runner element: {e}")
                    
                    if dog_name and odds_decimal > 0:
                        print(f"    ✅ Extracted: {dog_name} - ${odds_decimal:.2f}")
                        odds_data.append({
                            'dog_name': dog_name,
                            'dog_clean_name': self.clean_dog_name(dog_name),
                            'box_number': i + 1,
                            'odds_decimal': odds_decimal,
                            'odds_fractional': odds_text
                        })
                    else:
                        print(f"    ⚠️  Could not extract odds for runner {i+1}: name='{dog_name}', odds={odds_decimal}")
                        
                except Exception as e:
                    print(f"  ⚠️  Error extracting runner {i}: {e}")
                    continue
                    
        except Exception as e:
            print(f"  ⚠️  Error in runner card strategy: {e}")
        
        return odds_data
    
    def _extract_dog_name_enhanced(self, runner_text: str) -> str:
        """Enhanced dog name extraction with multiple strategies"""
        lines = runner_text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Strategy 1: Look for pattern like "1. Dog Name (1)" 
            name_match = re.match(r'\d+\.\s*([A-Za-z\s\'\-]+)\s*\(\d+\)', line)
            if name_match:
                return name_match.group(1).strip()
            
            # Strategy 2: Look for pattern like "1. Dog Name"
            name_match = re.match(r'\d+\.\s*([A-Za-z\s\'\-]+)$', line)
            if name_match:
                return name_match.group(1).strip()
            
            # Strategy 3: Look for dog names without numbers (but must be proper case)
            if (3 < len(line) < 30 and 
                re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]*)*$', line) and
                not any(word in line.lower() for word in 
                        ['speed', 'trainer', 'form', 'weight', 'barrier', 'early', 'jockey', 'time'])):
                return line
        
        return ""
    
    def _find_odds_for_runner(self, runner) -> Tuple[float, str]:
        """Find odds for a specific runner element"""
        odds_decimal = 0.0
        odds_text = ""
        
        try:
            # Strategy 1: Look for buttons with odds within the runner element
            buttons = runner.find_elements(By.CSS_SELECTOR, "button")
            for button in buttons:
                button_text = button.text.strip()
                if re.match(r'^\d+\.\d{1,2}$', button_text):  # Decimal odds like "2.50" or "2.5"
                    potential_odds = float(button_text)
                    if 1.01 <= potential_odds <= 50.0:
                        odds_decimal = potential_odds
                        odds_text = button_text
                        break
                elif '/' in button_text and re.match(r'^\d+/\d+$', button_text):  # Fractional like "5/2"
                    parts = button_text.split('/')
                    if len(parts) == 2:
                        odds_decimal = (float(parts[0]) / float(parts[1])) + 1.0
                        odds_text = button_text
                        break
            
            # Strategy 2: Look for odds in span/div elements within runner
            if odds_decimal == 0.0:
                odds_elements = runner.find_elements(By.CSS_SELECTOR, "span, div, [class*='odds'], [class*='price']")
                for elem in odds_elements:
                    elem_text = elem.text.strip()
                    if elem_text and re.match(r'^\d+\.\d{1,2}$', elem_text):
                        potential_odds = float(elem_text)
                        if 1.01 <= potential_odds <= 50.0:  # Realistic odds range
                            odds_decimal = potential_odds
                            odds_text = elem_text
                            break
            
            # Strategy 3: Look for odds in the runner text itself
            if odds_decimal == 0.0:
                runner_text = runner.text.strip()
                lines = runner_text.split('\n')
                for line in lines:
                    line = line.strip()
                    # Look for decimal odds patterns
                    odds_matches = re.findall(r'(\d+\.\d{1,2})', line)
                    for odds_match in odds_matches:
                        potential_odds = float(odds_match)
                        if 1.01 <= potential_odds <= 50.0:  # Realistic odds range
                            odds_decimal = potential_odds
                            odds_text = odds_match
                            break
                    if odds_decimal > 0:
                        break
                        
        except Exception as e:
            print(f"    ⚠️  Error extracting odds from runner element: {e}")
        
        return odds_decimal, odds_text
    
    def extract_dog_name_from_text(self, text: str) -> str:
        """Extract dog name from text with multiple strategies"""
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Strategy 1: Look for pattern like "1. Dog Name (1)" 
            name_match = re.match(r'\d+\.\s*([A-Za-z\s\'\-]+)\s*\(\d+\)', line)
            if name_match:
                return name_match.group(1).strip()
            
            # Strategy 2: Look for pattern like "1. Dog Name"
            name_match = re.match(r'\d+\.\s*([A-Za-z\s\'\-]+)$', line)
            if name_match:
                return name_match.group(1).strip()
            
            # Strategy 3: Look for dog names without numbers (but must be proper case)
            if (3 < len(line) < 30 and 
                re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]*)*$', line) and
                not any(word in line.lower() for word in 
                        ['speed', 'trainer', 'form', 'weight', 'barrier', 'early', 'jockey', 'time'])):
                return line
        
        return ""
    
    def extract_odds_strategy_separate_buttons(self) -> List[Dict]:
        """Try to match runners with separate odds buttons by position/order"""
        odds_data = []
        
        try:
            # Get runner names from runner elements
            runners = self.driver.find_elements(By.CSS_SELECTOR, "[class*='runner']")
            if not runners or len(runners) < 4:  # Reduced from 6 to 4 - some races have only 5 runners
                print(f"  ⚠️  Only found {len(runners)} runners, need at least 4 for button matching")
                return []
            
            print(f"  🎯 Found {len(runners)} runners for button matching")
            
            # Extract dog names
            dog_names = []
            for i, runner in enumerate(runners[:8]):
                runner_text = runner.text.strip()
                lines = runner_text.split('\n')
                dog_name = None
                
                for line in lines:
                    line = line.strip()
                    # Look for pattern like "1. Dog Name (1)"
                    name_match = re.match(r'\d+\.\s*([A-Za-z\s\']+)\s*\(\d+\)', line)
                    if name_match:
                        dog_name = name_match.group(1).strip()
                        break
                
                if dog_name:
                    dog_names.append(dog_name)
                    print(f"    Dog {i+1}: {dog_name}")
                else:
                    # Try alternative extraction for this runner
                    alt_name = self._extract_dog_name_enhanced(runner_text)
                    if alt_name:
                        dog_names.append(alt_name)
                        print(f"    Dog {i+1}: {alt_name} (alternative extraction)")
            
            # Now find all buttons with decimal odds patterns
            all_buttons = self.driver.find_elements(By.CSS_SELECTOR, "button")
            print(f"  🔍 Debug - Total buttons on page: {len(all_buttons)}")
            
            odds_buttons = []
            sample_button_texts = []
            
            for i, button in enumerate(all_buttons):
                try:
                    button_text = button.text.strip()
                    if i < 25:  # Sample first 25 buttons for debugging
                        sample_button_texts.append(f"'{button_text}'")
                    
                    # Skip obvious non-odds buttons first
                    if button_text in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 'EW', 'Runner', 'Filter', 'Join Now', 'Log In', 'Exotics']:
                        continue
                    
                    # Look for decimal odds pattern - be more flexible but avoid single digits
                    # Accept patterns like "1.60", "12.50", "2.5" but NOT "1", "2", etc.
                    odds_match = re.match(r'^(\d+\.\d{1,2})$', button_text)
                    if odds_match:
                        odds_decimal = float(odds_match.group(1))
                        if 1.01 <= odds_decimal <= 50.0:  # Realistic odds range
                            # Additional check: ensure it's not in a box number container
                            try:
                                parent_class = button.get_attribute('class') or ''
                                parent_element = button.find_element(By.XPATH, '..')
                                parent_parent_class = parent_element.get_attribute('class') or ''
                                
                                # Skip if it looks like box numbers (in tab containers)
                                if 'tab' in parent_parent_class.lower() or 'circle' in parent_parent_class.lower():
                                    continue
                                    
                                odds_buttons.append((button_text, odds_decimal))
                                print(f"      🎯 Found odds button: {button_text} (button #{i+1})")
                            except:
                                # If we can't check the parent, still add it if it looks like odds
                                odds_buttons.append((button_text, odds_decimal))
                                print(f"      🎯 Found odds button: {button_text} (button #{i+1})")
                    
                    # Also try fractional odds like "3/2", "5/1"
                    elif re.match(r'^\d+/\d+$', button_text):
                        try:
                            parts = button_text.split('/')
                            if len(parts) == 2:
                                numerator, denominator = float(parts[0]), float(parts[1])
                                odds_decimal = (numerator / denominator) + 1.0
                                if 1.01 <= odds_decimal <= 50.0:
                                    odds_buttons.append((button_text, odds_decimal))
                                    print(f"      🎯 Found fractional odds button: {button_text} = {odds_decimal:.2f} (button #{i+1})")
                        except:
                            continue
                except:
                    continue
            
            print(f"  🔍 Debug - Sample button texts (first 20): {', '.join(sample_button_texts[:10])}...")
            print(f"  💰 Found {len(odds_buttons)} odds buttons: {[f'${b[1]:.2f}' for b in odds_buttons[:8]]}")
            
            # Match dog names with odds buttons by position (assuming same order)
            min_count = min(len(dog_names), len(odds_buttons))
            
            for i in range(min_count):
                if i < len(dog_names) and i < len(odds_buttons):
                    dog_name = dog_names[i]
                    odds_text, odds_decimal = odds_buttons[i]
                    
                    odds_data.append({
                        'dog_name': dog_name,
                        'dog_clean_name': self.clean_dog_name(dog_name),
                        'box_number': i + 1,
                        'odds_decimal': odds_decimal,
                        'odds_fractional': odds_text
                    })
                    
                    print(f"    ✅ Matched: {dog_name} - ${odds_decimal:.2f}")
            
        except Exception as e:
            print(f"  ⚠️  Error in separate buttons strategy: {e}")
        
        return odds_data
    
    def extract_odds_strategy_table(self) -> List[Dict]:
        """Extract odds using table-based strategy"""
        odds_data = []
        
        try:
            # Look for table rows containing race data
            table_selectors = [
                "table tr",
                "[role='row']", 
                ".table-row",
                "[class*='row']"
            ]
            
            rows = []
            for selector in table_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if len(elements) > 1:  # Need at least header + data rows
                        rows = elements[1:]  # Skip header
                        print(f"  📊 Found {len(rows)} table rows using selector: {selector}")
                        break
                except:
                    continue
            
            if not rows:
                return []
            
            for i, row in enumerate(rows[:8]):
                try:
                    # Extract data from table cells
                    cells = row.find_elements(By.CSS_SELECTOR, "td, [role='cell'], [class*='cell']")
                    
                    if len(cells) >= 2:  # Need at least name and odds
                        dog_name = None
                        odds_text = None
                        
                        # Look for dog name and odds in cells
                        for cell in cells:
                            cell_text = cell.text.strip()
                            
                            # Check if this cell contains a dog name (usually text without numbers)
                            if cell_text and not re.match(r'^[\d./]+$', cell_text) and len(cell_text) > 2:
                                if not dog_name:  # Take first text cell as dog name
                                    dog_name = cell_text
                            
                            # Check if this cell contains odds (numbers with decimal or fraction)
                            if cell_text and re.search(r'\d+[\.\d/]+', cell_text):
                                odds_text = cell_text
                        
                        if dog_name and odds_text:
                            odds_decimal = self.parse_odds_to_decimal(odds_text)
                            if odds_decimal > 0:
                                odds_data.append({
                                    'dog_name': dog_name,
                                    'dog_clean_name': self.clean_dog_name(dog_name),
                                    'box_number': i + 1,
                                    'odds_decimal': odds_decimal,
                                    'odds_fractional': odds_text
                                })
                        
                except Exception as e:
                    print(f"  ⚠️  Error extracting table row {i}: {e}")
                    continue
                    
        except Exception as e:
            print(f"  ⚠️  Error in table strategy: {e}")
        
        return odds_data
    
    def extract_odds_strategy_generic(self) -> List[Dict]:
        """Extract odds using generic text pattern matching"""
        odds_data = []
        
        try:
            # Get page source and look for patterns that might contain odds
            page_source = self.driver.page_source
            
            # More specific pattern to avoid CSS and other webpage artifacts
            # Look for dog names that are likely to be real names followed by realistic odds
            pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+([1-9]\d*\.\d{1,2}|[1-9]\d*/[1-9]\d*|[1-9]\d+)'
            matches = re.findall(pattern, page_source)
            
            if matches:
                print(f"  🔍 Found {len(matches)} potential dog/odds matches")
                
                # Filter and validate matches
                valid_matches = []
                for dog_name, odds_text in matches:
                    # Skip obvious webpage artifacts
                    if (len(dog_name) < 3 or len(dog_name) > 25 or 
                        any(word in dog_name.lower() for word in 
                            ['click', 'bet', 'odds', 'race', 'time', 'place', 'login', 'sign', 'account', 'menu', 
                             'live', 'home', 'sport', 'market', 'price', 'value', 'next', 'back', 'view', 'more']) or
                        re.search(r'[0-9]', dog_name) or  # Names shouldn't contain numbers
                        dog_name.isupper() or dog_name.islower() or  # Should be proper case
                        len(re.findall(r'[AEIOU]', dog_name)) < 2):  # Should have vowels
                        continue
                    
                    odds_decimal = self.parse_odds_to_decimal(odds_text)
                    
                    # Only accept realistic greyhound racing odds (1.01 to 50.00)
                    if 1.01 <= odds_decimal <= 50.0:
                        valid_matches.append((dog_name, odds_text, odds_decimal))
                
                print(f"  ✅ Filtered to {len(valid_matches)} valid matches")
                
                # Take up to 8 unique dog names
                seen_names = set()
                for dog_name, odds_text, odds_decimal in valid_matches[:20]:  # Check more but take fewer
                    if len(odds_data) >= 8:  
                        break
                        
                    clean_name = self.clean_dog_name(dog_name)
                    if clean_name not in seen_names:
                        seen_names.add(clean_name)
                        odds_data.append({
                            'dog_name': dog_name,
                            'dog_clean_name': clean_name,
                            'box_number': len(odds_data) + 1,
                            'odds_decimal': odds_decimal,
                            'odds_fractional': odds_text
                        })
                        
        except Exception as e:
            print(f"  ⚠️  Error in generic strategy: {e}")
        
        return odds_data
    
    def extract_dog_name_from_element(self, element) -> str:
        """Extract dog name from a runner element"""
        name_selectors = [
            "[data-automation-id*='name']",
            "[data-automation-id*='runner']",
            ".name",
            ".runner-name",
            ".selection-name",
            "[class*='name']"
        ]
        
        for selector in name_selectors:
            try:
                name_elem = element.find_element(By.CSS_SELECTOR, selector)
                name = name_elem.text.strip()
                if name and len(name) > 2:
                    return name
            except:
                continue
        
        # Fallback: use element text and try to extract name
        element_text = element.text.strip()
        lines = element_text.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for line that looks like a dog name (not odds, not numbers)
            if line and not re.match(r'^[\d\./:\s]+$', line) and len(line) > 2:
                return line
        
        return ""
    
    def extract_odds_from_element(self, element) -> Tuple[str, float]:
        """Extract odds from a runner element"""
        odds_selectors = [
            "[data-automation-id*='odds']",
            "[data-automation-id*='price']",
            ".odds",
            ".price",
            "[class*='odds']",
            "[class*='price']"
        ]
        
        for selector in odds_selectors:
            try:
                odds_elem = element.find_element(By.CSS_SELECTOR, selector)
                odds_text = odds_elem.text.strip()
                if odds_text:
                    odds_decimal = self.parse_odds_to_decimal(odds_text)
                    if odds_decimal > 0:
                        return odds_text, odds_decimal
            except:
                continue
        
        # Fallback: look for odds patterns in element text
        element_text = element.text.strip()
        lines = element_text.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for odds patterns
            if re.search(r'\d+[\.\d/]+|\d+/\d+', line):
                odds_decimal = self.parse_odds_to_decimal(line)
                if odds_decimal > 0:
                    return line, odds_decimal
        
        return "", 0.0
    
    def find_next_race_from_meeting(self, meeting_url: str) -> Optional[str]:
        """Navigate to meeting page and find the next available race URL"""
        try:
            print(f"  🏟️  Navigating to meeting page: {meeting_url}")
            self.driver.get(meeting_url)
            
            # Wait for page to load
            try:
                WebDriverWait(self.driver, 10).until(
                    lambda driver: driver.execute_script("return document.readyState") == "complete"
                )
            except TimeoutException:
                print(f"  ⚠️  Timeout waiting for meeting page to load")
            
            time.sleep(2)  # Let dynamic content load
            
            # Look for race navigation links with countdown timers
            race_selectors = [
                "a[href*='/race-']",  # Direct race links
                "a[href*='race'][class*='countdown']",  # Race links with countdown class
                "[class*='race-nav'] a",  # Race navigation links
                "[class*='countdown'] a",  # Links within countdown elements
                "a[href*='meeting'][href*='race']",  # Meeting-race links
            ]
            
            race_links = []
            for selector in race_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        race_links.extend(elements)
                        print(f"    📋 Found {len(elements)} race links with selector: {selector}")
                        break
                except:
                    continue
            
            if not race_links:
                print(f"  ⚠️  No race links found on meeting page")
                return None
            
            # Find the next race based on countdown or timing
            best_race_url = None
            shortest_countdown = float('inf')
            
            for link in race_links[:10]:  # Check first 10 links
                try:
                    href = link.get_attribute('href')
                    if not href or '/race-' not in href:
                        continue
                    
                    # Get text around the link to find countdown info
                    link_text = link.text.strip()
                    parent_text = link.find_element(By.XPATH, "..").text.strip()
                    
                    # Look for countdown patterns like "5m" or "Next race"
                    countdown_text = f"{link_text} {parent_text}".lower()
                    
                    # Check if this looks like the next/current race
                    if any(keyword in countdown_text for keyword in ['next', 'live', 'now', 'current']):
                        print(f"    🎯 Found next/current race: {href}")
                        return href
                    
                    # Extract countdown minutes if available
                    countdown_match = re.search(r'(\d+)m(?:\s|$)', countdown_text)
                    if countdown_match:
                        minutes = int(countdown_match.group(1))
                        if minutes < shortest_countdown and minutes < 30:  # Within 30 minutes
                            shortest_countdown = minutes
                            best_race_url = href
                            print(f"    ⏰ Found race in {minutes}min: {href}")
                    
                except Exception as e:
                    print(f"    ⚠️  Error processing race link: {e}")
                    continue
            
            if best_race_url:
                print(f"  ✅ Selected race with {shortest_countdown}min countdown: {best_race_url}")
                return best_race_url
            
            # Fallback: take the first race link if no countdown found
            if race_links:
                first_race_href = race_links[0].get_attribute('href')
                if first_race_href and '/race-' in first_race_href:
                    print(f"  📋 Fallback to first race link: {first_race_href}")
                    return first_race_href
            
            print(f"  ⚠️  Could not find suitable race URL from meeting page")
            return None
            
        except Exception as e:
            print(f"  ⚠️  Error finding race from meeting page: {e}")
            return None
    
    def extract_race_number_from_page(self, expected_venue: str = None) -> Optional[int]:
        """Extract race number from the current page, venue-specific"""
        try:
            url = self.driver.current_url
            print(f"  🔗 Current URL: {url}")
            
            # Strategy 1: Extract from URL path - most reliable
            url_matches = re.findall(r'/race[\s\-_]*(\d+)|/r(\d+)/', url.lower())
            for match in url_matches:
                race_num = int(match[0] or match[1])
                if 1 <= race_num <= 20:
                    print(f"  📍 Found race number {race_num} in URL")
                    return race_num
            
            # Strategy 2: Look in page title - prioritize venue-specific
            title = self.driver.title
            print(f"  📄 Page title: {title}")
            
            if expected_venue:
                # Look for venue name followed by race number in title
                venue_pattern = expected_venue.lower().replace(' ', r'\s*')
                title_pattern = rf'{venue_pattern}.*?(?:race\s*#?\s*|r\s*)(\d+)'
                title_matches = re.findall(title_pattern, title.lower())
                for match in title_matches:
                    race_num = int(match)
                    if 1 <= race_num <= 20:
                        print(f"  📍 Found race number {race_num} in venue-specific title")
                        return race_num
            
            # Strategy 3: Look for main race heading on the page
            main_selectors = [
                "h1",
                "[class*='main-title']",
                "[class*='page-title']",
                "[class*='race-title']",
                ".title-main"
            ]
            
            for selector in main_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for elem in elements:
                        text = elem.text.strip()
                        if expected_venue:
                            # Only consider headings that contain the expected venue
                            if expected_venue.lower() not in text.lower():
                                continue
                            
                            # Look for race number specifically after venue name
                            venue_pattern = expected_venue.lower().replace(' ', r'\s*')
                            pattern = rf'{venue_pattern}.*?(?:race\s*#?\s*|r\s*)(\d+)'
                            matches = re.findall(pattern, text.lower())
                            for match in matches:
                                race_num = int(match)
                                if 1 <= race_num <= 20:
                                    print(f"  📍 Found race number {race_num} in main heading: {text[:50]}...")
                                    return race_num
                except:
                    continue
            
            # Strategy 4: Look for race-specific content in page source
            page_source = self.driver.page_source
            
            if expected_venue:
                venue_pattern = expected_venue.lower().replace(' ', r'[\s\-_]*')
                json_pattern = rf'.*{venue_pattern}.*"race(?:Number|Id|_number)?"\s*:\s*(\d+)'
                source_matches = re.findall(json_pattern, page_source, re.IGNORECASE | re.DOTALL)
                for match in source_matches:
                    race_num = int(match)
                    if 1 <= race_num <= 20:
                        print(f"  📍 Found race number {race_num} in venue-specific JSON")
                        return race_num
            
            # Strategy 5: Look for current race indicator
            current_race_selectors = [
                "[class*='current']",
                "[class*='next']",
                "[class*='active']",
                "[class*='live']"
            ]
            
            for selector in current_race_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for elem in elements:
                        text = elem.text.strip()
                        if expected_venue and expected_venue.lower() in text.lower():
                            matches = re.findall(r'(?:race\s*#?\s*|r\s*)(\d+)', text.lower())
                            for match in matches:
                                race_num = int(match)
                                if 1 <= race_num <= 20:
                                    print(f"  📍 Found race number {race_num} in current race indicator: {text[:50]}...")
                                    return race_num
                except:
                    continue
            
            # Strategy 6: Find race info in specific race content
            race_content_selectors = [
                "[class*='race-info']",
                "[class*='race-header']",
                "[class*='race-details']",
                ".race-card header",
                "[data-automation-id*='race']"
            ]
            
            for selector in race_content_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for elem in elements:
                        text = elem.text.strip()
                        if expected_venue and expected_venue.lower() in text.lower():
                            matches = re.findall(r'(?:race\s*#?\s*|r\s*)(\d+)', text.lower())
                            for match in matches:
                                race_num = int(match)
                                if 1 <= race_num <= 20:
                                    print(f"  📍 Found race number {race_num} in race content: {text[:30]}...")
                                    return race_num
                except:
                    continue
            
            print(f"  ⚠️  Could not extract race number from page for venue {expected_venue}")
            return None
            
        except Exception as e:
            print(f"  ⚠️  Error extracting race number: {e}")
            return None
            
    def get_today_races(self) -> List[Dict]:
        """Get today's greyhound races from Sportsbet with live odds"""
        self.setup_driver()
        
        if not self.driver:
            return []
            
        try:
            print("🔄 Loading Sportsbet greyhound races...")
            self.driver.get(self.greyhound_url)
            
            # Wait for page to load - try multiple selectors
            print("⏳ Waiting for page to load...")
            
            # Try different selectors that might indicate races are loaded
            selectors_to_try = [
                "[data-automation-id='race-card']",
                ".race-card",
                "[class*='race']",
                "[class*='card']",
                "a[href*='greyhound']",
                "a[href*='/race/']"
            ]
            
            page_loaded = False
            for selector in selectors_to_try:
                try:
                    WebDriverWait(self.driver, 3).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    print(f"✅ Page loaded, found elements with selector: {selector}")
                    page_loaded = True
                    break
                except TimeoutException:
                    continue
            
            if not page_loaded:
                print("⚠️  No race elements found, trying manual inspection...")
                # Get page title and some content for debugging
                title = self.driver.title
                print(f"Page title: {title}")
                
                # Look for any greyhound-related content
                page_source = self.driver.page_source.lower()
                if 'greyhound' in page_source:
                    print("✅ Page contains greyhound content")
                else:
                    print("⚠️  No greyhound content found")
                    return []
            
            races = []
            
            # Primary method: Extract from JSON-LD structured data
            print("🔍 Attempting to extract races from JSON-LD structured data...")
            races = self.extract_race_info_from_json_ld()
            
            if races:
                print(f"✅ Successfully extracted {len(races)} races from JSON-LD")
                
                # Filter for upcoming races only
                now = datetime.now()
                upcoming_races = []
                for race in races:
                    race_time = race.get('start_datetime')
                    if race_time and race_time > now:
                        time_diff = race_time - now
                        if time_diff.total_seconds() < 3600:  # Within next hour
                            upcoming_races.append(race)
                            print(f"  ⏰ Upcoming: {race['venue']} at {race_time.strftime('%H:%M')} ({time_diff.total_seconds()/60:.0f}min)")
                
                if not upcoming_races:
                    print("ℹ️  No upcoming races found in JSON-LD data, trying DOM extraction...")
                    
                    # Fallback: Extract from DOM elements with countdown timers
                    dom_races = self.extract_races_from_dom()
                    if dom_races:
                        print(f"✅ Found {len(dom_races)} upcoming races from DOM")
                        
                        # ENHANCED: Process multiple races with better scheduling
                        print(f"🎯 Fetching live odds for {len(dom_races)} upcoming races...")
                        races_with_odds = []
                        
                        # Sort races by countdown time (soonest first)
                        sorted_races = sorted(dom_races, key=lambda x: x.get('countdown_minutes', 999))
                        
                        # Process up to 8 races with priority for soonest ones
                        max_races = min(8, len(sorted_races))
                        
                        # Group races: immediate (< 10min), soon (10-30min), later (30min+)
                        immediate_races = [r for r in sorted_races if r.get('countdown_minutes', 999) < 10]
                        soon_races = [r for r in sorted_races if 10 <= r.get('countdown_minutes', 999) < 30]
                        later_races = [r for r in sorted_races if r.get('countdown_minutes', 999) >= 30]
                        
                        print(f"  📋 Race scheduling: {len(immediate_races)} immediate, {len(soon_races)} soon, {len(later_races)} later")
                        
                        # Process immediate races first with minimal delay
                        for i, race in enumerate(immediate_races[:3]):  # Max 3 immediate
                            print(f"🚨 URGENT - Processing race {i+1}: {race['venue']} in {race.get('countdown_minutes', '?')}min")
                            enhanced_race = self.get_race_odds_from_page(race)
                            races_with_odds.append(enhanced_race)
                            
                            self._debug_print_odds(enhanced_race, "URGENT")
                            time.sleep(1)  # Minimal delay for urgent races
                        
                        # Process soon races with moderate delay
                        remaining_slots = max_races - len(races_with_odds)
                        for i, race in enumerate(soon_races[:remaining_slots]):
                            print(f"⏰ SOON - Processing race {i+1}: {race['venue']} in {race.get('countdown_minutes', '?')}min")
                            enhanced_race = self.get_race_odds_from_page(race)
                            races_with_odds.append(enhanced_race)
                            
                            self._debug_print_odds(enhanced_race, "SOON")
                            time.sleep(2)  # Moderate delay
                        
                        # Process later races with normal delay if slots remain
                        remaining_slots = max_races - len(races_with_odds)
                        for i, race in enumerate(later_races[:remaining_slots]):
                            print(f"📅 LATER - Processing race {i+1}: {race['venue']} in {race.get('countdown_minutes', '?')}min")
                            enhanced_race = self.get_race_odds_from_page(race)
                            races_with_odds.append(enhanced_race)
                            
                            self._debug_print_odds(enhanced_race, "LATER")
                            time.sleep(3)  # Normal delay
                        
                        # Summary
                        total_dogs_with_odds = sum(len(r.get('odds_data', [])) for r in races_with_odds)
                        print(f"🎯 ENHANCED COVERAGE: Processed {len(races_with_odds)} races with {total_dogs_with_odds} total dog odds")
                        
                        return races_with_odds
                    else:
                        print("ℹ️  No upcoming races found")
                        return []
                
                # Now enhance each race with live odds by visiting individual race pages
                print(f"🎯 Fetching live odds for {len(upcoming_races)} upcoming races...")
                races_with_odds = []
                
                for i, race in enumerate(upcoming_races[:3]):  # Limit to first 3 upcoming races
                    print(f"🔄 Processing race {i+1}/{min(3, len(upcoming_races))}: {race['venue']}")
                    enhanced_race = self.get_race_odds_from_page(race)
                    races_with_odds.append(enhanced_race)
                    
                    # Debug: Print what we extracted
                    odds_data = enhanced_race.get('odds_data', [])
                    if odds_data:
                        print(f"  🐕 Debug - Extracted odds:")
                        for dog in odds_data:
                            print(f"    {dog['dog_name']}: ${dog['odds_decimal']:.2f}")
                    
                    # Rate limiting
                    time.sleep(3)
                
                return races_with_odds
            
            # Fallback method: Try to find race elements in the DOM
            print("🔄 Falling back to DOM element extraction...")
            race_cards = self.find_race_elements()
            print(f"📊 Found {len(race_cards)} race elements")
            
            if not race_cards:
                print("ℹ️  No live races found, checking if this is expected...")
                current_hour = datetime.now().hour
                if current_hour < 6 or current_hour > 23:
                    print("ℹ️  Outside typical racing hours - this is normal")
                else:
                    print("⚠️  Expected to find races during racing hours")
                return []
            
            # Process DOM elements as fallback
            for i, card in enumerate(race_cards[:10]):
                try:
                    race_info = self.extract_race_info_flexible(card, i)
                    if race_info:  # Remove the odds_data requirement for fallback
                        races.append(race_info)
                except Exception as e:
                    print(f"⚠️  Error extracting race info from card {i}: {e}")
                    continue
                    
            return races
            
        except Exception as e:
            print(f"⚠️  Error getting races: {e}")
            return []
        finally:
            # Don't close driver here if we need to visit individual race pages
            pass
            
    def extract_race_info(self, race_card) -> Optional[Dict]:
        """Extract race information from a race card"""
        try:
            # Get race details
            venue_elem = race_card.find_element(By.CSS_SELECTOR, "[data-automation-id='race-venue']")
            venue = venue_elem.text.strip()
            
            race_number_elem = race_card.find_element(By.CSS_SELECTOR, "[data-automation-id='race-number']")
            race_number = race_number_elem.text.strip()
            
            race_time_elem = race_card.find_element(By.CSS_SELECTOR, "[data-automation-id='race-time']")
            race_time = race_time_elem.text.strip()
            
            # Generate race ID
            race_id = f"{venue}_{race_number}_{datetime.now().strftime('%Y%m%d')}"
            
            # Get odds
            odds_data = self.extract_race_odds(race_card)
            
            return {
                'race_id': race_id,
                'venue': venue,
                'race_number': race_number,
                'race_date': datetime.now().date(),
                'race_time': race_time,
                'odds_data': odds_data
            }
            
        except Exception as e:
            print(f"⚠️  Error extracting race info: {e}")
            return None
            
    def extract_race_odds(self, race_card) -> List[Dict]:
        """Extract odds for all dogs in a race"""
        odds_data = []
        
        try:
            # Find all runner cards
            runners = race_card.find_elements(By.CSS_SELECTOR, "[data-automation-id='race-runner']")
            
            for i, runner in enumerate(runners):
                try:
                    # Get dog name
                    dog_name_elem = runner.find_element(By.CSS_SELECTOR, "[data-automation-id='runner-name']")
                    dog_name = dog_name_elem.text.strip()
                    
                    # Get odds
                    odds_elem = runner.find_element(By.CSS_SELECTOR, "[data-automation-id='runner-odds']")
                    odds_text = odds_elem.text.strip()
                    
                    # Parse odds
                    odds_decimal = self.parse_odds_to_decimal(odds_text)
                    
                    odds_data.append({
                        'dog_name': dog_name,
                        'dog_clean_name': self.clean_dog_name(dog_name),
                        'box_number': i + 1,
                        'odds_decimal': odds_decimal,
                        'odds_fractional': odds_text
                    })
                    
                except Exception as e:
                    print(f"⚠️  Error extracting runner odds: {e}")
                    continue
                    
        except Exception as e:
            print(f"⚠️  Error extracting race odds: {e}")
            
        return odds_data
        
    def parse_odds_to_decimal(self, odds_text: str) -> float:
        """Convert various odds formats to decimal"""
        try:
            # Remove any currency symbols or extra characters
            odds_text = re.sub(r'[^\d./:\-+]', '', odds_text)
            
            # Handle decimal odds (e.g., "2.50")
            if '.' in odds_text and '/' not in odds_text:
                return float(odds_text)
                
            # Handle fractional odds (e.g., "3/2", "5/1")
            if '/' in odds_text:
                parts = odds_text.split('/')
                if len(parts) == 2:
                    numerator = float(parts[0])
                    denominator = float(parts[1])
                    return (numerator / denominator) + 1.0
                    
            # Handle odds with colon (e.g., "3:2")
            if ':' in odds_text:
                parts = odds_text.split(':')
                if len(parts) == 2:
                    numerator = float(parts[0])
                    denominator = float(parts[1])
                    return (numerator / denominator) + 1.0
                    
            # If it's just a number, assume it's decimal
            return float(odds_text)
            
        except:
            return 0.0
            
    def clean_dog_name(self, name: str) -> str:
        """Clean dog name for database consistency"""
        return re.sub(r'[^\w\s]', '', name.upper().strip())
        
    def _debug_print_odds(self, race_info: Dict, priority: str = ""):
        """Debug helper to print extracted odds data"""
        try:
            odds_data = race_info.get('odds_data', [])
            venue = race_info.get('venue', 'Unknown')
            countdown = race_info.get('countdown_minutes', '?')
            
            if odds_data:
                print(f"  🐕 {priority} - Extracted {len(odds_data)} odds for {venue} (in {countdown}min):")
                for i, dog in enumerate(odds_data[:8]):
                    print(f"    {i+1}. {dog['dog_name']}: ${dog['odds_decimal']:.2f}")
            else:
                print(f"  ⚠️ {priority} - No odds extracted for {venue} (in {countdown}min)")
        except Exception as e:
            print(f"  ⚠️ Error in debug print: {e}")
        
    def save_odds_to_database(self, race_info: Dict):
        """Save odds data and race metadata to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Always save race metadata even if no odds data available
            race_datetime_str = None
            if race_info.get('start_datetime'):
                try:
                    race_datetime_str = race_info['start_datetime'].isoformat()
                except:
                    pass
            
            # Save or update race metadata
            cursor.execute('''
                INSERT OR REPLACE INTO race_metadata 
                (race_id, venue, race_number, race_date, race_time, 
                 sportsbet_url, venue_slug, start_datetime)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                race_info['race_id'],
                race_info['venue'],
                race_info.get('race_number', 0),
                race_info['race_date'],
                race_info['race_time'],
                race_info.get('venue_url', ''),
                race_info.get('venue_slug', ''),
                race_datetime_str
            ))
            
            # Skip saving odds if no odds data available
            odds_data = race_info.get('odds_data', [])
            if not odds_data:
                print(f"ℹ️  Saved race metadata for {race_info['race_id']}, but no odds data available")
                conn.commit()
                return
            
            # Mark previous odds as not current
            cursor.execute('''
                UPDATE live_odds 
                SET is_current = FALSE 
                WHERE race_id = ? AND is_current = TRUE
            ''', (race_info['race_id'],))
            
            # Insert new odds
            for dog_odds in odds_data:
                cursor.execute('''
                    INSERT INTO live_odds 
                    (race_id, venue, race_number, race_date, race_time, 
                     dog_name, dog_clean_name, box_number, odds_decimal, odds_fractional)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    race_info['race_id'],
                    race_info['venue'],
                    race_info.get('race_number', 0),
                    race_info['race_date'],
                    race_info['race_time'],
                    dog_odds['dog_name'],
                    dog_odds['dog_clean_name'],
                    dog_odds['box_number'],
                    dog_odds['odds_decimal'],
                    dog_odds['odds_fractional']
                ))
                
            conn.commit()
            print(f"✅ Saved race metadata and odds for {race_info['race_id']} ({len(odds_data)} dogs)")
            
        except Exception as e:
            print(f"⚠️  Error saving race data: {e}")
        finally:
            conn.close()
            
    def track_odds_movement(self, race_id: str, dog_name: str, new_odds: float):
        """Track odds movements for a specific dog"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get previous odds
            cursor.execute('''
                SELECT odds_decimal FROM odds_history 
                WHERE race_id = ? AND dog_clean_name = ? 
                ORDER BY timestamp DESC LIMIT 1
            ''', (race_id, dog_name))
            
            result = cursor.fetchone()
            previous_odds = result[0] if result else new_odds
            
            # Calculate change
            odds_change = new_odds - previous_odds
            
            # Insert movement record
            cursor.execute('''
                INSERT INTO odds_history (race_id, dog_clean_name, odds_decimal, odds_change)
                VALUES (?, ?, ?, ?)
            ''', (race_id, dog_name, new_odds, odds_change))
            
            conn.commit()
            
            if abs(odds_change) > 0.5:  # Significant movement
                print(f"📈 Significant odds movement: {dog_name} {previous_odds:.2f} → {new_odds:.2f}")
                
        except Exception as e:
            print(f"⚠️  Error tracking odds movement: {e}")
        finally:
            conn.close()
            
    def identify_value_bets(self):
        """Identify value betting opportunities by comparing predictions with market odds"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Get recent predictions with corresponding live odds
            query = '''
                SELECT 
                    p.race_id,
                    p.dog_clean_name,
                    p.predicted_probability,
                    p.confidence_level,
                    o.odds_decimal as market_odds
                FROM (
                    SELECT DISTINCT race_id, dog_clean_name, 
                           predicted_probability, confidence_level
                    FROM predictions 
                    WHERE timestamp > datetime('now', '-24 hours')
                ) p
                JOIN live_odds o ON p.race_id = o.race_id 
                    AND p.dog_clean_name = o.dog_clean_name
                WHERE o.is_current = TRUE AND o.odds_decimal > 0
            '''
            
            df = pd.read_sql_query(query, conn)
            
            if df.empty:
                print("ℹ️  No matching predictions and odds found")
                return []
                
            value_bets = []
            
            for _, row in df.iterrows():
                predicted_prob = row['predicted_probability']
                market_odds = row['market_odds']
                
                # Calculate implied probability from market odds
                implied_prob = 1.0 / market_odds if market_odds > 0 else 0
                
                # Calculate value (predicted probability vs implied probability)
                if implied_prob > 0:
                    value_percentage = ((predicted_prob - implied_prob) / implied_prob) * 100
                    
                    # Identify value bets (predicted probability > implied probability)
                    if value_percentage > 10:  # At least 10% value
                        bet_recommendation = self.generate_bet_recommendation(
                            value_percentage, row['confidence_level'], market_odds
                        )
                        
                        value_bet = {
                            'race_id': row['race_id'],
                            'dog_clean_name': row['dog_clean_name'],
                            'predicted_probability': predicted_prob,
                            'market_odds': market_odds,
                            'implied_probability': implied_prob,
                            'value_percentage': value_percentage,
                            'confidence_level': row['confidence_level'],
                            'bet_recommendation': bet_recommendation
                        }
                        
                        value_bets.append(value_bet)
                        
            # Save value bets to database
            self.save_value_bets(value_bets)
            
            return value_bets
            
        except Exception as e:
            print(f"⚠️  Error identifying value bets: {e}")
            return []
        finally:
            conn.close()
            
    def generate_bet_recommendation(self, value_percentage: float, confidence: str, odds: float) -> str:
        """Generate betting recommendation based on value and confidence"""
        if value_percentage > 30 and confidence == 'HIGH':
            return f"STRONG BET - {value_percentage:.1f}% value at {odds:.2f}"
        elif value_percentage > 20 and confidence in ['HIGH', 'MEDIUM']:
            return f"GOOD BET - {value_percentage:.1f}% value at {odds:.2f}"
        elif value_percentage > 10:
            return f"SMALL BET - {value_percentage:.1f}% value at {odds:.2f}"
        else:
            return f"MONITOR - {value_percentage:.1f}% value at {odds:.2f}"
            
    def save_value_bets(self, value_bets: List[Dict]):
        """Save value betting opportunities to database"""
        if not value_bets:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for bet in value_bets:
                cursor.execute('''
                    INSERT INTO value_bets 
                    (race_id, dog_clean_name, predicted_probability, market_odds, 
                     implied_probability, value_percentage, confidence_level, bet_recommendation)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    bet['race_id'],
                    bet['dog_clean_name'],
                    bet['predicted_probability'],
                    bet['market_odds'],
                    bet['implied_probability'],
                    bet['value_percentage'],
                    bet['confidence_level'],
                    bet['bet_recommendation']
                ))
                
            conn.commit()
            print(f"✅ Saved {len(value_bets)} value betting opportunities")
            
        except Exception as e:
            print(f"⚠️  Error saving value bets: {e}")
        finally:
            conn.close()
            
    def update_all_odds(self):
        """Update odds for all today's races"""
        print("🔄 Starting odds update...")
        
        races = self.get_today_races()
        
        for race in races:
            self.save_odds_to_database(race)
            time.sleep(2)  # Rate limiting
            
        print(f"✅ Updated odds for {len(races)} races")
        
        # Identify value bets after updating odds
        value_bets = self.identify_value_bets()
        if value_bets:
            print(f"💰 Found {len(value_bets)} value betting opportunities")
            
    def start_continuous_monitoring(self):
        """Start continuous odds monitoring"""
        print("🚀 Starting continuous odds monitoring...")
        
        # Schedule regular updates
        schedule.every(self.update_interval).seconds.do(self.update_all_odds)
        
        # Initial update
        self.update_all_odds()
        
        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(1)
            
    def get_live_odds_summary(self) -> Dict:
        """Get summary of current live odds with race times and Sportsbet links"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            query = '''
                SELECT 
                    lo.venue,
                    lo.race_number,
                    lo.race_id,
                    COUNT(*) as dog_count,
                    AVG(lo.odds_decimal) as avg_odds,
                    MIN(lo.odds_decimal) as favorite_odds,
                    MAX(lo.odds_decimal) as longest_odds,
                    lo.timestamp,
                    rm.race_time,
                    rm.race_date,
                    rm.url as sportsbet_url
                FROM live_odds lo
                LEFT JOIN race_metadata rm ON lo.race_id = rm.race_id
                WHERE lo.is_current = TRUE 
                GROUP BY lo.race_id
                ORDER BY 
                    CASE 
                        WHEN rm.race_date IS NOT NULL AND rm.race_time IS NOT NULL 
                        THEN rm.race_date || ' ' || rm.race_time
                        ELSE lo.timestamp
                    END ASC
            '''
            
            df = pd.read_sql_query(query, conn)
            
            # Convert race_datetime strings back to datetime objects for sorting
            records = df.to_dict('records')
            
            # Add time until race and format display time
            for record in records:
                race_date = record.get('race_date')
                race_time = record.get('race_time')
                
                if race_date and race_time:
                    try:
                        # Combine race_date and race_time to create datetime
                        if isinstance(race_date, str):
                            date_str = race_date
                        else:
                            date_str = str(race_date)
                        
                        datetime_str = f"{date_str} {race_time}"
                        race_dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M')
                        now = datetime.now()
                        
                        time_diff = race_dt - now
                        minutes_until = int(time_diff.total_seconds() / 60)
                        
                        record['minutes_until_race'] = minutes_until
                        record['formatted_race_time'] = race_dt.strftime('%H:%M')
                        record['time_status'] = (
                            'SOON' if 0 <= minutes_until <= 15 else
                            'UPCOMING' if 15 < minutes_until <= 60 else
                            'LATER' if minutes_until > 60 else
                            'PAST'
                        )
                    except (ValueError, TypeError) as e:
                        print(f"Error parsing race date/time {race_date} {race_time}: {e}")
                        record['minutes_until_race'] = 999
                        record['formatted_race_time'] = record.get('race_time', 'Unknown')
                        record['time_status'] = 'UNKNOWN'
                else:
                    record['minutes_until_race'] = 999
                    record['formatted_race_time'] = record.get('race_time', 'Unknown')
                    record['time_status'] = 'UNKNOWN'
                
                # Ensure Sportsbet URL is properly formatted
                if not record.get('sportsbet_url'):
                    # Try to construct a URL if we have venue and race number
                    venue = record.get('venue', '').lower().replace(' ', '-')
                    race_num = record.get('race_number')
                    if venue and race_num:
                        # This is a fallback URL construction - may not always work
                        record['sportsbet_url'] = f"https://www.sportsbet.com.au/betting/racing/greyhound/{venue}"
            
            # Sort by time until race (soonest first)
            records.sort(key=lambda x: x.get('minutes_until_race', 999))
            
            return records
            
        except Exception as e:
            print(f"⚠️  Error getting odds summary: {e}")
            return []
        finally:
            conn.close()
            
    def get_value_bets_summary(self) -> List[Dict]:
        """Get current value betting opportunities"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            query = '''
                SELECT 
                    race_id,
                    dog_clean_name,
                    predicted_probability,
                    market_odds,
                    value_percentage,
                    confidence_level,
                    bet_recommendation,
                    timestamp
                FROM value_bets 
                WHERE timestamp > datetime('now', '-6 hours')
                ORDER BY value_percentage DESC
                LIMIT 20
            '''
            
            df = pd.read_sql_query(query, conn)
            return df.to_dict('records')
            
        except Exception as e:
            print(f"⚠️  Error getting value bets: {e}")
            return []
        finally:
            conn.close()

def main():
    """Main execution function for testing"""
    integrator = SportsbetOddsIntegrator()
    
    try:
        print("🏁 Sportsbet Odds Integrator")
        print("=" * 50)
        
        # Get today's races
        races = integrator.get_today_races()
        print(f"📊 Found {len(races)} races")
        
        # Save odds
        for race in races:
            integrator.save_odds_to_database(race)
            
        # Identify value bets
        value_bets = integrator.identify_value_bets()
        print(f"💰 Found {len(value_bets)} value opportunities")
        
        # Show summaries
        print("\n📈 Live Odds Summary:")
        odds_summary = integrator.get_live_odds_summary()
        for race in odds_summary[:5]:
            print(f"{race['venue']} R{race['race_number']}: {race['dog_count']} dogs, favorite ${race['favorite_odds']:.2f}")
            
        print("\n💎 Top Value Bets:")
        value_summary = integrator.get_value_bets_summary()
        for bet in value_summary[:3]:
            print(f"{bet['dog_clean_name']}: {bet['bet_recommendation']}")
            
    finally:
        # Ensure the driver is closed
        integrator.close_driver()
        print("\n🏁 Odds integrator session complete")
        
if __name__ == "__main__":
    main()
