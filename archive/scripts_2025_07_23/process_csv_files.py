#!/usr/bin/env python3
"""
Enhanced CSV File Processor with Scraped Data Integration
========================================================

This script processes CSV files from the unprocessed directory:
1. Reads CSV files from unprocessed/
2. Extracts comprehensive race and dog data from all 17 columns
3. Combines with scraped data (weather, track conditions, odds)
4. Inserts enhanced data into the database
5. Moves processed files to processed/

Author: AI Assistant
Date: July 11, 2025
"""

import os
import sqlite3
import pandas as pd
import shutil
from datetime import datetime
from pathlib import Path
import re
import requests
import json
from typing import Dict, List, Optional, Any
import time

# Selenium for web scraping
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# BeautifulSoup for HTML parsing
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

class EnhancedCSVProcessor:
    def __init__(self):
        self.db_path = "./databases/comprehensive_greyhound_data.db"
        self.unprocessed_dir = "./unprocessed"
        self.processed_dir = "./processed"
        self.upcoming_dir = "./upcoming_races"
        
        # Ensure directories exist
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.upcoming_dir, exist_ok=True)
        
        # Initialize web driver for scraping
        self.driver = None
        if SELENIUM_AVAILABLE:
            self.setup_driver()
        
        # Weather API key (free from openweathermap.org)
        self.weather_api_key = None  # Set your API key here
        
        print("📊 Enhanced CSV Processor initialized")
        print(f"✅ Selenium Available: {SELENIUM_AVAILABLE}")
        print(f"✅ BeautifulSoup Available: {BS4_AVAILABLE}")
    
    def setup_driver(self):
        """Setup Chrome driver for web scraping"""
        try:
            import shutil
            chromedriver_path = shutil.which('chromedriver')
            
            if not chromedriver_path:
                print("⚠️ ChromeDriver not found, web scraping disabled")
                return
                
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--window-size=1920,1080')
            
            self.driver = webdriver.Chrome(options=options)
            print("✅ Chrome driver initialized")
        except Exception as e:
            print(f"⚠️ Chrome driver setup failed: {e}")
            self.driver = None
    
    def get_weather_data(self, venue: str, race_date) -> Optional[Dict[str, Any]]:
        """Get weather data for the race venue and date"""
        if not self.weather_api_key:
            return None
        
        try:
            # Map venue codes to cities
            venue_to_city = {
                'MAND': 'melbourne',
                'SAL': 'adelaide', 
                'MOUNT': 'melbourne',
                'HEA': 'melbourne',
                'SAN': 'adelaide',
                'WAR': 'melbourne'
            }
            
            city = venue_to_city.get(venue, 'melbourne')
            
            # Get weather data from OpenWeatherMap
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.weather_api_key}&units=metric"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'weather': data['weather'][0]['description'],
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'wind_speed': data['wind']['speed'],
                    'wind_direction': data['wind']['deg']
                }
        except Exception as e:
            print(f"   ⚠️ Weather API error: {e}")
        
        return None
    
    def scrape_race_details(self, race_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Scrape additional race details from the web"""
        if not self.driver or not BS4_AVAILABLE:
            return None
        
        try:
            # Construct URL for race details
            url = self.construct_race_url(race_info)
            if not url:
                return None
            
            print(f"   🌐 Scraping race details from: {url}")
            self.driver.get(url)
            time.sleep(2)  # Wait for page to load
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            scraped_data = {}
            
            # Try to extract track conditions
            track_condition_elem = soup.find(['span', 'div'], class_=re.compile(r'track.*condition', re.I))
            if track_condition_elem:
                scraped_data['track_condition'] = track_condition_elem.get_text(strip=True)
            
            # Try to extract additional odds data
            odds_elements = soup.find_all(['span', 'div'], class_=re.compile(r'odds|price', re.I))
            if odds_elements:
                scraped_data['additional_odds'] = [elem.get_text(strip=True) for elem in odds_elements[:5]]
            
            return scraped_data if scraped_data else None
            
        except Exception as e:
            print(f"   ⚠️ Scraping error: {e}")
            return None
    
    def construct_race_url(self, race_info: Dict[str, Any]) -> Optional[str]:
        """Construct URL for race details page"""
        try:
            venue_mapping = {
                'MAND': 'mandurah',
                'SAL': 'sandown-lakeside',
                'MOUNT': 'the-meadows',
                'HEA': 'healesville',
                'SAN': 'sandown-lakeside',
                'WAR': 'warrnambool'
            }
            
            venue_url = venue_mapping.get(race_info['venue'])
            if not venue_url:
                return None
            
            race_date = race_info['race_date']
            if isinstance(race_date, str):
                race_date = datetime.strptime(race_date, '%Y-%m-%d').date()
            
            date_str = race_date.strftime('%Y-%m-%d')
            race_number = race_info['race_number']
            
            return f"https://www.thedogs.com.au/racing/{venue_url}/{date_str}/{race_number}"
            
        except Exception as e:
            print(f"   ❌ Error constructing URL: {e}")
            return None
    
    def get_enhanced_race_data(self, race_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get enhanced race data by combining scraped and weather data"""
        enhanced_data = {}
        
        # Get weather data
        weather_data = self.get_weather_data(race_info['venue'], race_info['race_date'])
        if weather_data:
            enhanced_data.update(weather_data)
            print(f"   ☀️ Weather: {weather_data.get('weather', 'N/A')} ({weather_data.get('temperature', 'N/A')}°C)")
        
        # Get scraped race details
        scraped_data = self.scrape_race_details(race_info)
        if scraped_data:
            enhanced_data.update(scraped_data)
            print(f"   🌐 Scraped track condition: {scraped_data.get('track_condition', 'N/A')}")
        
        return enhanced_data
    
    def extract_race_info(self, filename):
        """Extract race information from filename"""
        # Example: "Race 1 - MAND - 10 July 2025.csv"
        pattern = r"Race (\d+) - ([A-Z_]+) - (\d{1,2} [A-Za-z]+ \d{4})\.csv"
        match = re.match(pattern, filename)
        
        if match:
            race_number = int(match.group(1))
            venue = match.group(2)
            date_str = match.group(3)
            
            # Parse date
            try:
                race_date = datetime.strptime(date_str, "%d %B %Y").date()
            except ValueError:
                try:
                    race_date = datetime.strptime(date_str, "%d %b %Y").date()
                except ValueError:
                    race_date = None
            
            # Create race_id
            if race_date:
                race_id = f"{venue.lower()}_{race_date}_{race_number}"
            else:
                race_id = f"{venue.lower()}_{race_number}"
            
            return {
                'race_id': race_id,
                'venue': venue,
                'race_number': race_number,
                'race_date': race_date,
                'filename': filename
            }
        
        return None
    
    def process_csv_file(self, filepath):
        """Process a single CSV file"""
        filename = os.path.basename(filepath)
        print(f"📄 Processing: {filename}")
        
        # Extract race info
        race_info = self.extract_race_info(filename)
        if not race_info:
            print(f"   ⚠️  Could not extract race info from filename")
            return False
        
        try:
            # Read CSV file
            df = pd.read_csv(filepath)
            
            if df.empty:
                print(f"   ⚠️  Empty CSV file")
                return False
            
            print(f"   📊 Found {len(df)} dogs with columns: {list(df.columns)}")
            
            # Check if this is a future race (no results yet)
            has_results = False
            if 'Finish Position' in df.columns:
                has_results = df['Finish Position'].notna().any()
            elif 'finish_position' in df.columns:
                has_results = df['finish_position'].notna().any()
            
            # Get enhanced data (weather, track conditions, etc.)
            enhanced_data = self.get_enhanced_race_data(race_info)
            
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Propagate last known dog name downward in the DataFrame
            df['Dog Name'] = df['Dog Name'].ffill()

            # Now filter after propagation
            real_dog_data = df.dropna(subset=['Dog Name'])
            actual_field_size = real_dog_data['Dog Name'].nunique()
            print(f"   🐕 Correctly found {actual_field_size} unique dogs in the race")

            # Insert enhanced race metadata with correct field size
            cursor.execute('''
                INSERT OR REPLACE INTO race_metadata (
                    race_id, venue, race_number, race_date, race_name, 
                    field_size, extraction_timestamp, data_source, race_status,
                    weather, temperature, track_condition, grade, distance
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                race_info['race_id'],
                race_info['venue'],
                race_info['race_number'],
                race_info['race_date'],
                f"Race {race_info['race_number']}",
                actual_field_size,
                datetime.now(),
                'enhanced_csv_processor',
                'completed' if has_results else 'upcoming',
                enhanced_data.get('weather'),
                enhanced_data.get('temperature'),
                enhanced_data.get('track_condition'),
                enhanced_data.get('grade'),
                enhanced_data.get('distance')
            ))
            
            # Extract race metadata from the data
            race_grade = None
            race_distance = None
            if not df.empty:
                # Get grade and distance from first row with data
                first_row = df.iloc[0]
                if 'G' in first_row and pd.notna(first_row['G']):
                    race_grade = str(first_row['G'])
                if 'DIST' in first_row and pd.notna(first_row['DIST']):
                    race_distance = str(first_row['DIST'])
            
            # Update race metadata with extracted info
            cursor.execute('''
                UPDATE race_metadata 
                SET grade = ?, distance = ?
                WHERE race_id = ?
            ''', (race_grade, race_distance, race_info['race_id']))
            
            # Convert the current dog's info into a list of dictionaries, one per row per dog
            dogs_with_names = real_dog_data.groupby('Dog Name').first().reset_index()  # Only use first occurrence as main row
            
            dogs_inserted = 0
            for index, row in dogs_with_names.iterrows():
                # Extract dog data with flexible column names
                dog_data = self.extract_dog_data(row, race_info['race_id'])
                
                if dog_data:
                    cursor.execute('''
                        INSERT OR REPLACE INTO dog_race_data (
                            race_id, dog_name, box_number, finish_position, 
                            weight, starting_price, extraction_timestamp, data_source,
                            individual_time, sectional_1st, margin, running_style
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        dog_data['race_id'],
                        dog_data['dog_name'],
                        dog_data['box_number'],
                        dog_data['finish_position'],
                        dog_data['weight'],
                        dog_data['starting_price'],
                        datetime.now(),
                        'enhanced_csv_processor',
                        dog_data.get('time'),
                        dog_data.get('sectional_1st'),
                        dog_data.get('margin'),
                        dog_data.get('grade')
                    ))
                    dogs_inserted += 1
                    
                    # Log comprehensive data extraction
                    print(f"      🐕 {dog_data['dog_name']}: Box {dog_data['box_number']}, SP {dog_data.get('starting_price', 'N/A')}, Time {dog_data.get('time', 'N/A')}")
            
            conn.commit()
            conn.close()
            
            print(f"   ✅ Processed {dogs_inserted} dogs")
            
            # Move file to appropriate directory
            if has_results:
                dest_path = os.path.join(self.processed_dir, filename)
                print(f"   📁 Moving to processed/")
            else:
                dest_path = os.path.join(self.upcoming_dir, filename)
                print(f"   📁 Moving to upcoming_races/")
            
            shutil.move(filepath, dest_path)
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error processing file: {e}")
            return False
    
    def extract_dog_data(self, row, race_id):
        """Extract comprehensive dog data from CSV row with all 17 columns"""
        try:
            # Initialize all fields
            dog_name = None
            sex = None
            plc = None
            box_number = None
            weight = None
            distance = None
            date = None
            track = None
            grade = None
            time = None
            win = None
            bon = None
            sectional_1st = None
            margin = None
            w_2g = None
            pir = None
            starting_price = None
            
            # Extract Dog Name
            for col in ['Dog Name', 'dog_name', 'Dog', 'name']:
                if col in row and pd.notna(row[col]):
                    dog_name = str(row[col]).strip()
                    break
            
            # Extract Sex
            if 'Sex' in row and pd.notna(row['Sex']):
                sex = str(row['Sex']).strip()
            
            # Extract PLC (Place)
            if 'PLC' in row and pd.notna(row['PLC']):
                plc = str(row['PLC']).strip()
            
            # Extract BOX
            if 'BOX' in row and pd.notna(row['BOX']):
                try:
                    box_number = int(row['BOX'])
                except:
                    pass
            
            # Extract WGT (Weight)
            if 'WGT' in row and pd.notna(row['WGT']):
                try:
                    weight = float(row['WGT'])
                except:
                    pass
            
            # Extract DIST (Distance)
            if 'DIST' in row and pd.notna(row['DIST']):
                try:
                    distance = int(row['DIST'])
                except:
                    pass
            
            # Extract DATE
            if 'DATE' in row and pd.notna(row['DATE']):
                date = str(row['DATE']).strip()
            
            # Extract TRACK
            if 'TRACK' in row and pd.notna(row['TRACK']):
                track = str(row['TRACK']).strip()
            
            # Extract G (Grade)
            if 'G' in row and pd.notna(row['G']):
                grade = str(row['G']).strip()
            
            # Extract TIME
            if 'TIME' in row and pd.notna(row['TIME']):
                time = str(row['TIME']).strip()
            
            # Extract WIN
            if 'WIN' in row and pd.notna(row['WIN']):
                win = str(row['WIN']).strip()
            
            # Extract BON
            if 'BON' in row and pd.notna(row['BON']):
                bon = str(row['BON']).strip()
            
            # Extract 1 SEC (First Sectional)
            if '1 SEC' in row and pd.notna(row['1 SEC']):
                sectional_1st = str(row['1 SEC']).strip()
            
            # Extract MGN (Margin)
            if 'MGN' in row and pd.notna(row['MGN']):
                margin = str(row['MGN']).strip()
            
            # Extract W/2G (Winner/Second in Grade)
            if 'W/2G' in row and pd.notna(row['W/2G']):
                w_2g = str(row['W/2G']).strip()
            
            # Extract PIR (Performance Index Rating)
            if 'PIR' in row and pd.notna(row['PIR']):
                pir = str(row['PIR']).strip()
            
            # Extract SP (Starting Price)
            if 'SP' in row and pd.notna(row['SP']):
                try:
                    starting_price = float(row['SP'])
                except:
                    pass
            
            # Return comprehensive dog data if we have essential fields
            if dog_name and box_number is not None:
                return {
                    'race_id': race_id,
                    'dog_name': dog_name,
                    'sex': sex,
                    'plc': plc,
                    'box_number': box_number,
                    'weight': weight,
                    'distance': distance,
                    'date': date,
                    'track': track,
                    'grade': grade,
                    'time': time,
                    'win': win,
                    'bon': bon,
                    'sectional_1st': sectional_1st,
                    'margin': margin,
                    'w_2g': w_2g,
                    'pir': pir,
                    'starting_price': starting_price,
                    'finish_position': plc  # Use PLC as finish position
                }
            
            return None
            
        except Exception as e:
            print(f"   ⚠️  Error extracting dog data: {e}")
            return None
    
    def process_all_files(self):
        """Process all CSV files in unprocessed directory"""
        if not os.path.exists(self.unprocessed_dir):
            print("❌ Unprocessed directory not found")
            return
        
        csv_files = [f for f in os.listdir(self.unprocessed_dir) if f.endswith('.csv')]
        
        if not csv_files:
            print("📭 No CSV files to process")
            return
        
        print(f"🔄 Processing {len(csv_files)} CSV files...")
        
        processed_count = 0
        error_count = 0
        
        for filename in csv_files:
            filepath = os.path.join(self.unprocessed_dir, filename)
            
            if self.process_csv_file(filepath):
                processed_count += 1
            else:
                error_count += 1
        
        print(f"\n📊 Processing Summary:")
        print(f"   ✅ Successfully processed: {processed_count}")
        print(f"   ❌ Errors: {error_count}")
        
        # Show database stats
        self.show_database_stats()
    
    def show_database_stats(self):
        """Show current database statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM race_metadata")
            total_races = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM dog_race_data")
            total_dogs = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT venue) FROM race_metadata")
            venues = cursor.fetchone()[0]
            
            print(f"\n🏆 Database Statistics:")
            print(f"   🏁 Total races: {total_races}")
            print(f"   🐕 Total dog entries: {total_dogs}")
            print(f"   🏟️  Venues: {venues}")
            
            conn.close()
            
        except Exception as e:
            print(f"   ⚠️  Error getting database stats: {e}")

def main():
    processor = EnhancedCSVProcessor()
    processor.process_all_files()

if __name__ == "__main__":
    main()
