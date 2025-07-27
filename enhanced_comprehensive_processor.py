#!/usr/bin/env python3
"""
Enhanced Comprehensive Greyhound Racing Data Processor
====================================================

This processor combines all advanced features into a single, comprehensive pipeline:
- Advanced CSV processing with weather, track conditions, and odds
- Real-time enhanced data collection
- Database population with advanced metadata
- AI-powered analysis and insights
- Winner prediction and analysis

Author: AI Assistant
Date: July 11, 2025
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple, Any
import time
import random
import threading

# Selenium for advanced scraping
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

# Advanced analytics
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Global processing status variables (imported from app.py if available)
processing_lock = threading.Lock()
processing_status = {
    'running': True,  # Default to running if not externally controlled
    'progress': 0,
    'current_task': ''
}

class EnhancedComprehensiveProcessor:
    """
    Enhanced comprehensive processor that handles all advanced data collection,
    processing, and analysis in a single unified pipeline.
    """
    
    def __init__(self, db_path="greyhound_racing_data.db", processing_mode="full", batch_size=50):
        self.db_path = db_path
        self.unprocessed_dir = "./unprocessed"
        self.processed_dir = "./processed"
        self.results_dir = "./logs"
        
        # Performance optimization settings
        self.processing_mode = processing_mode  # "full", "fast", "minimal"
        self.batch_size = batch_size  # Process files in batches
        self.enable_web_scraping = True
        self.scraping_timeout = 10  # Seconds to wait for scraping
        self.max_retries = 2
        
        # Create directories
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize database
        self.init_database()
        
        # Initialize web driver for enhanced scraping (only if needed)
        self.driver = None
        if self.processing_mode != "minimal":
            print("🔧 Attempting to initialize ChromeDriver...")
            self.setup_driver()
        else:
            print("⚡ Minimal mode: Skipping web driver initialization")
        
        # Initialize weather service integration
        try:
            from weather_service_open_meteo import OpenMeteoWeatherService
            self.weather_service = OpenMeteoWeatherService()
            print("✅ Weather service initialized successfully")
        except ImportError as e:
            print(f"⚠️ Weather service not available: {e}")
            self.weather_service = None
        
        print(f"🚀 Enhanced Comprehensive Processor Initialized ({processing_mode} mode)")
        print(f"✅ Selenium Available: {SELENIUM_AVAILABLE}")
        print(f"✅ BeautifulSoup Available: {BS4_AVAILABLE}")
        print(f"✅ Scikit-learn Available: {SKLEARN_AVAILABLE}")
        print(f"⚙️ Processing Mode: {processing_mode}")
        print(f"📦 Batch Size: {batch_size}")
    
    def setup_driver(self):
        """Setup Chrome driver for enhanced scraping"""
        try:
            # Try to find ChromeDriver in common locations
            import shutil
            chromedriver_path = shutil.which('chromedriver')
            
            if not chromedriver_path:
                print("⚠️ ChromeDriver not found in PATH, skipping web scraping")
                self.driver = None
                return
                
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--window-size=1920,1080')
            options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36')
            
            self.driver = webdriver.Chrome(options=options)
            print("✅ Chrome driver initialized")
        except Exception as e:
            print(f"⚠️ Chrome driver setup failed: {e}")
            print("⚠️ Web scraping features disabled")
            self.driver = None
    
    def init_database(self):
        """Initialize comprehensive database with all advanced tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enhanced race metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS race_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT UNIQUE,
                venue TEXT,
                race_number INTEGER,
                race_date DATE,
                race_name TEXT,
                grade TEXT,
                distance TEXT,
                track_condition TEXT,
                weather TEXT,
                temperature REAL,
                humidity REAL,
                wind_speed REAL,
                wind_direction TEXT,
                track_record TEXT,
                prize_money_total REAL,
                prize_money_breakdown TEXT,
                race_time TEXT,
                field_size INTEGER,
                url TEXT,
                extraction_timestamp DATETIME,
                data_source TEXT,
                winner_name TEXT,
                winner_odds REAL,
                winner_margin REAL,
                race_status TEXT,
                data_quality_note TEXT,
                actual_field_size INTEGER,
                scratched_count INTEGER,
                scratch_rate REAL,
                box_analysis TEXT,
                UNIQUE(race_id)
            )
        ''')
        
        # Enhanced dog race data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dog_race_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT,
                dog_name TEXT,
                dog_clean_name TEXT,
                dog_id INTEGER,
                box_number INTEGER,
                finish_position TEXT,
                trainer_name TEXT,
                trainer_id INTEGER,
                weight REAL,
                running_style TEXT,
                odds_decimal REAL,
                odds_fractional TEXT,
                starting_price REAL,
                individual_time TEXT,
                sectional_1st TEXT,
                sectional_2nd TEXT,
                sectional_3rd TEXT,
                margin TEXT,
                beaten_margin REAL,
                was_scratched BOOLEAN DEFAULT FALSE,
                blackbook_link TEXT,
                extraction_timestamp DATETIME,
                data_source TEXT,
                form_guide_json TEXT,
                historical_records TEXT,
                performance_rating REAL,
                speed_rating REAL,
                class_rating REAL,
                recent_form TEXT,
                win_probability REAL,
                place_probability REAL,
                scraped_trainer_name TEXT,
                scraped_reaction_time TEXT,
                scraped_nbtt TEXT,
                scraped_race_classification TEXT,
                scraped_raw_result TEXT,
                scraped_finish_position TEXT,
                best_time REAL,
                FOREIGN KEY (race_id) REFERENCES race_metadata (race_id),
                UNIQUE(race_id, dog_clean_name, box_number)
            )
        ''')
        
        # Advanced analytics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS race_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT,
                analysis_type TEXT,
                analysis_data TEXT,
                confidence_score REAL,
                predicted_winner TEXT,
                predicted_odds REAL,
                analysis_timestamp DATETIME,
                model_version TEXT,
                FOREIGN KEY (race_id) REFERENCES race_metadata (race_id)
            )
        ''')
        
        # Track conditions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS track_conditions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                venue TEXT,
                date DATE,
                condition TEXT,
                rail_position TEXT,
                track_rating REAL,
                weather_conditions TEXT,
                temperature REAL,
                humidity REAL,
                wind_conditions TEXT,
                track_bias TEXT,
                extraction_timestamp DATETIME
            )
        ''')
        
        conn.commit()
        
        # Add missing columns to existing tables (migration)
        self._migrate_database_schema(cursor)
        
        conn.commit()
        print("Database initialized successfully.")
        conn.close()
        print("✅ Enhanced database initialized")
    
    def _migrate_database_schema(self, cursor):
        """Migrate database schema to add missing columns"""
        try:
            # Check if new columns exist and add them if missing
            new_columns = [
                ('actual_field_size', 'INTEGER'),
                ('scratched_count', 'INTEGER'),
                ('scratch_rate', 'REAL'),
                ('box_analysis', 'TEXT'),
                ('weather_condition', 'TEXT'),
                ('precipitation', 'REAL'),
                ('pressure', 'REAL'),
                ('visibility', 'REAL'),
                ('weather_location', 'TEXT'),
                ('weather_timestamp', 'DATETIME'),
                ('weather_adjustment_factor', 'REAL')
            ]
            
            # Get existing columns
            cursor.execute("PRAGMA table_info(race_metadata)")
            existing_columns = [row[1] for row in cursor.fetchall()]
            
            # Add missing columns
            for column_name, column_type in new_columns:
                if column_name not in existing_columns:
                    try:
                        cursor.execute(f"ALTER TABLE race_metadata ADD COLUMN {column_name} {column_type}")
                        print(f"   ✅ Added column: {column_name}")
                    except Exception as e:
                        print(f"   ⚠️ Could not add column {column_name}: {e}")
            
        except Exception as e:
            print(f"   ⚠️ Database migration error: {e}")
    
    def group_dog_data(self, df: pd.DataFrame) -> List[List[Any]]:
        """Group dog data by dog name, associating blank rows with previous dog name"""
        dog_groups = []
        current_group = []

        for index, row in df.iterrows():
            if pd.notna(row['Dog Name']):
                if current_group:
                    dog_groups.append(current_group)
                current_group = [row]
            else:
                current_group.append(row)

        if current_group:
            dog_groups.append(current_group)

        return dog_groups

    def process_grouped_dog_data(self, dog_group: List[pd.Series], race_info: Dict[str, Any], race_results: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Process a group of rows belonging to the same dog, handling historical data"""
        try:
            # Find the current race row - it should match the race date and have the most recent data
            current_race_row = None
            race_date_str = race_info['race_date'].strftime('%Y-%m-%d') if hasattr(race_info['race_date'], 'strftime') else str(race_info['race_date'])
            
            # Look for the row that matches our race date or is the most recent
            for row in dog_group:
                row_date = str(row.get('DATE', ''))
                # Try to match the race date (convert formats if needed)
                if race_date_str in row_date or row_date in race_date_str:
                    current_race_row = row
                    break
            
            # If no exact date match, use the first row (most recent/current race)
            if current_race_row is None:
                current_race_row = dog_group[0]
            
            # Extract the dog number from the first column (which should be the box number for this race)
            dog_number = None
            if hasattr(current_race_row, 'name') and current_race_row.name is not None:
                # Try to extract from the first column/index
                first_col_value = str(current_race_row.iloc[0]) if len(current_race_row) > 0 else ''
                # Look for pattern like "1." or "2." etc.
                import re
                match = re.match(r'^(\d+)\.', first_col_value)
                if match:
                    dog_number = int(match.group(1))
            
            dog_data = self.process_dog_data(current_race_row, race_info, race_results)
            if dog_data:
                dog_data['box_number'] = dog_number
                # All rows are historical races for this dog (including the current one)
                dog_data['historical_records'] = json.dumps([
                    {
                        'date': str(row.get('DATE', '')),
                        'time': str(row.get('TIME', '')),
                        'margin': str(row.get('MGN', '')),
                        'position': str(row.get('PLC', '')),
                        'box': str(row.get('BOX', '')),
                        'track': str(row.get('TRACK', '')),
                        'grade': str(row.get('G', ''))
                    }
                    for row in dog_group
                ])
                
                # Calculate best time and recent form from historical records
                try:
                    historical_data = json.loads(dog_data['historical_records'])
                    if historical_data:
                        # Calculate best time (fastest time from historical records)
                        times = []
                        recent_positions = []
                        for record in historical_data:  # Process ALL historical records
                            time_str = record.get('time', '')
                            position = record.get('position', '')
                            if time_str and time_str.replace('.', '').isdigit():
                                times.append(float(time_str))
                            if position and position.isdigit():
                                recent_positions.append(position)
                        
                        if times:
                            dog_data['best_time'] = min(times)
                        else:
                            dog_data['best_time'] = 0.0
                        
                        # Recent form (last 5 positions)
                        dog_data['recent_form'] = ''.join(recent_positions[:5])
                        
                        # Additional analytics from historical data
                        dog_data['total_races'] = len(historical_data)
                        dog_data['avg_time'] = sum(times) / len(times) if times else 0.0
                        win_count = sum(1 for pos in recent_positions if pos == '1')
                        dog_data['win_rate'] = win_count / len(recent_positions) if recent_positions else 0.0
                        
                except:
                    dog_data['best_time'] = 0.0
                    dog_data['recent_form'] = ''
                    dog_data['total_races'] = 0
                    dog_data['avg_time'] = 0.0
                    dog_data['win_rate'] = 0.0

            return dog_data

        except Exception as e:
            print(f"⚠️ Error processing grouped dog data: {e}")
            return None

    def process_csv_file(self, csv_file_path: str) -> Dict[str, Any]:
        """Process a single CSV file with enhanced data extraction and race results scraping"""
        print(f"📈 Processing: {os.path.basename(csv_file_path)}")

        try:
            # Read CSV file (form guide data)
            df = pd.read_csv(csv_file_path)

            # Extract race information from filename and CSV data
            race_info = self.extract_race_info_from_filename_and_csv(csv_file_path, df)
            if not race_info:
                print(f"⚠️ Could not extract race info from filename")
                return {'status': 'error', 'error': 'Could not extract race info'}

            print(f"   📍 Race: {race_info['venue']} Race {race_info['race_number']} on {race_info['race_date']}")

            # STEP 1: Scrape race results and winners from website (mode-dependent)
            race_results = {}
            if self.processing_mode == "minimal":
                print(f"   ⚡ Minimal mode: Skipping web scraping")
                race_results = {
                    'winner': None,
                    'scraped_successfully': False,
                    'all_results': [],
                    'race_url': None
                }
            else:
                print(f"   🔍 Scraping race results and winners...")
                race_results = self.scrape_race_results(race_info)

            # STEP 2: Combine form guide data with race results
            enhanced_race_info = self.combine_race_data(race_info, race_results)

            # STEP 3: Group and process dog data properly
            print(f"   🐕 Grouping dog data by name...")
            dog_groups = self.group_dog_data(df)

            processed_dogs = []
            for dog_group in dog_groups:
                dog_data = self.process_grouped_dog_data(dog_group, race_info, race_results)
                if dog_data:
                    processed_dogs.append(dog_data)
            
            # STEP 4: Validate and fix winner data consistency
            print(f"   🔧 Validating winner data consistency...")
            self.validate_winner_consistency(enhanced_race_info, processed_dogs)
            
            # STEP 5: Assign finish positions from scraped results
            print(f"   🏁 Assigning finish positions from scraped results...")
            self.assign_finish_positions_from_scraped_results(race_results, processed_dogs)
            
            # STEP 6: Winner must come from web scraping - form guides only contain historical data
            if not enhanced_race_info.get('winner_name'):
                print(f"   ⚠️ No winner found from web scraping - form guides contain only historical data")
                print(f"   📋 Form guide structure: Each dog has multiple historical race rows")
                print(f"   🚫 Cannot determine actual race winner from historical performance data")
                # Do not attempt to determine winner from form guide - it's invalid
            elif enhanced_race_info.get('data_quality_note') and 'mismatch' in enhanced_race_info.get('data_quality_note', ''):
                print(f"   🎯 Handling data mismatch - prioritizing scraped result...")
                # If there's a mismatch, trust the scraped result over form guide
                # The scraped winner is already set, so we'll keep it

            print(f"   📊 Processed {len(processed_dogs)} dogs (grouped from {len(df)} rows)")

            # STEP 7: Check if race should be excluded - strict criteria
            has_scraped_data = enhanced_race_info.get('scraped_successfully', False)
            has_winner = enhanced_race_info.get('winner_name', '') != ''
            has_url = enhanced_race_info.get('url', '') != ''
            has_positioned_dogs = any(dog.get('finish_position') not in ['N/A', '', None] for dog in processed_dogs)
            
            # Enhanced data consistency checks
            total_dogs = len(processed_dogs)
            positioned_dogs_count = sum(1 for dog in processed_dogs if dog.get('finish_position') not in ['N/A', '', None])
            positioning_ratio = positioned_dogs_count / total_dogs if total_dogs > 0 else 0
            
            # Check if winner is in the positioned dogs
            winner_name = enhanced_race_info.get('winner_name', '')
            winner_in_data = any(
                dog.get('finish_position') == '1' or 
                (winner_name and winner_name.lower() in dog.get('dog_clean_name', '').lower())
                for dog in processed_dogs
            )
            
            # Check for duplicate positions and handle dead heats
            positions = [dog.get('finish_position') for dog in processed_dogs if dog.get('finish_position') not in ['N/A', '', None]]
            
            # Detect and handle dead heats (duplicate positions)
            position_counts = {}
            for pos in positions:
                clean_pos = pos.rstrip('=')
                position_counts[clean_pos] = position_counts.get(clean_pos, 0) + 1
            
            # Find positions with duplicates (dead heats)
            dead_heat_positions = [pos for pos, count in position_counts.items() if count > 1]
            
            if dead_heat_positions:
                print(f"   🏁 Dead heats detected at positions: {dead_heat_positions}")
                # Fix dead heat notation
                for dog in processed_dogs:
                    pos = dog.get('finish_position', '')
                    if pos and pos not in ['N/A', '', None]:
                        clean_pos = pos.rstrip('=')
                        if clean_pos in dead_heat_positions and not pos.endswith('='):
                            dog['finish_position'] = f"{clean_pos}="
                            print(f"   🔄 Fixed dead heat: {dog.get('dog_clean_name')} -> {clean_pos}=")
                
                # Recalculate positions after dead heat fix
                positions = [dog.get('finish_position') for dog in processed_dogs if dog.get('finish_position') not in ['N/A', '', None]]
            
            # Check for duplicates after dead heat handling
            # Dead heats (positions ending with =) are NOT duplicates - they're legitimate ties
            non_dead_heat_positions = [pos for pos in positions if not pos.endswith('=')]
            dead_heat_positions_unique = [pos.rstrip('=') for pos in positions if pos.endswith('=')]
            
            # Check for duplicates in non-dead heat positions
            has_non_deadheat_duplicates = len(non_dead_heat_positions) != len(set(non_dead_heat_positions))
            
            # Check for invalid dead heat positions (shouldn't duplicate with non-dead heat)
            invalid_deadheat = any(
                pos.rstrip('=') in [p for p in non_dead_heat_positions] 
                for pos in positions if pos.endswith('=')
            )
            
            has_duplicates = has_non_deadheat_duplicates or invalid_deadheat
            
            # Debug output for duplicate detection
            if has_duplicates or positions:
                print(f"   🔍 Debug - Position analysis:")
                print(f"      All positions: {[dog.get('finish_position') for dog in processed_dogs]}")
                print(f"      Filtered positions: {positions}")
                print(f"      Unique positions: {list(set(positions))}")
                print(f"      Dead heats: {dead_heat_positions}")
                print(f"      Has duplicates: {has_duplicates}")
                for i, dog in enumerate(processed_dogs):
                    print(f"      Dog {i+1}: {dog.get('dog_clean_name', 'Unknown')} - Position: {dog.get('finish_position')}")
            
            # Enhanced field size analysis for box win rate correlation
            actual_field_size = len([dog for dog in processed_dogs if dog.get('finish_position') not in ['N/A', '', None]])
            scratched_count = total_dogs - actual_field_size
            
            if scratched_count > 0:
                print(f"   📊 Field analysis: {actual_field_size} runners, {scratched_count} scratched")
            
            # Box position analysis for win rate correlation
            box_analysis = {}
            winner_box = None
            for dog in processed_dogs:
                box_num = dog.get('box_number')
                finish_pos = dog.get('finish_position')
                if box_num and finish_pos not in ['N/A', '', None]:
                    box_analysis[f'box_{box_num}'] = {
                        'dog_name': dog.get('dog_clean_name', ''),
                        'finish_position': finish_pos,
                        'was_winner': finish_pos == '1'
                    }
                    if finish_pos == '1':
                        winner_box = box_num
            
            # Store field size and box analysis
            enhanced_race_info['field_size'] = total_dogs
            enhanced_race_info['actual_field_size'] = actual_field_size
            enhanced_race_info['scratched_count'] = scratched_count
            enhanced_race_info['scratch_rate'] = scratched_count / total_dogs if total_dogs > 0 else 0
            enhanced_race_info['winner_box'] = winner_box
            enhanced_race_info['box_analysis'] = json.dumps(box_analysis)
            
            if winner_box:
                print(f"   🏆 Winner from box {winner_box} ({actual_field_size} field)")
            
            # Only include races that have ALL of:
            # 1. Successfully scraped data
            # 2. Clear winner
            # 3. Race URL
            # 4. Winner in positioned dogs
            # 5. No duplicate positions
            # Note: Positioning ratio removed - scratched dogs don't invalidate race data
            if not (has_scraped_data and has_winner and has_url and 
                    winner_in_data and not has_duplicates):
                print(f"   🚫 Excluding race {enhanced_race_info['race_id']} - data quality issues")
                print(f"      Scraped: {has_scraped_data}, Winner: {has_winner}, URL: {has_url}")
                print(f"      Positioning: {positioning_ratio:.1%}, Winner in data: {winner_in_data}, Duplicates: {has_duplicates}")
                
                # Move excluded race to processed folder to avoid reprocessing
                self.move_to_processed(csv_file_path, status='excluded')
                return {'status': 'excluded', 'reason': 'Data quality issues'}

            # STEP 8: Collect and store weather data for this race
            print(f"   🌤️ Collecting weather data for {race_info['venue']} on {race_info['race_date']}...")
            weather_data = self.collect_weather_data_for_race(race_info)
            if weather_data:
                # Merge weather data into enhanced race info
                enhanced_race_info.update(weather_data)
                print(f"   ✅ Weather data collected: {weather_data.get('weather_condition', 'Unknown')} conditions")
            else:
                print(f"   ⚠️ Weather data not available for this race")

            # STEP 9: Save combined data to database
            self.save_to_database(enhanced_race_info, processed_dogs)

            # STEP 9: Move file to processed directory
            self.move_to_processed(csv_file_path, status='success')

            return {
                'race_info': enhanced_race_info,
                'dogs': processed_dogs,
                'status': 'success'
            }

        except Exception as e:
            print(f"❌ Error processing {csv_file_path}: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def extract_race_info_from_filename_and_csv(self, filepath: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Extract race information from CSV filename and data"""
        filename = os.path.basename(filepath)
        
        # Try multiple filename patterns
        patterns = [
            # Pattern 1: "Race_XX_VENUE_YYYY-MM-DD.csv" (current format)
            (r'Race_(\d+)_([A-Z_]+)_(\d{4}-\d{2}-\d{2})\.csv', '%Y-%m-%d'),
            # Pattern 2: "Race X - VENUE - DATE.csv" (legacy format)
            (r'Race (\d+) - ([A-Z_]+) - (\d{1,2} \w+ \d{4})\.csv', '%d %B %Y')
        ]
        
        race_number = None
        venue = None
        race_date = None
        
        for pattern, date_format in patterns:
            match = re.match(pattern, filename)
            if match:
                race_number = int(match.group(1))
                venue = match.group(2)
                date_str = match.group(3)
                
                # Parse date
                try:
                    race_date = datetime.strptime(date_str, date_format).date()
                    break
                except ValueError:
                    continue
        
        if race_number is not None and venue is not None and race_date is not None:
            # Generate race ID
            race_id = f"{venue.lower()}_{race_date}_{race_number}"
            
            # Extract grade and distance from CSV data
            grade = ''
            distance = ''
            
            if not df.empty and len(df) > 0:
                # Get the first row to extract race details
                first_row = df.iloc[0]
                
                # Extract grade from 'G' column
                if 'G' in df.columns:
                    grade_value = first_row['G']
                    if pd.notna(grade_value):
                        grade = str(grade_value)
                
                # Extract distance from 'DIST' column
                if 'DIST' in df.columns:
                    dist_value = first_row['DIST']
                    if pd.notna(dist_value):
                        distance = str(int(dist_value)) + 'm'  # Convert to string with 'm' suffix
            
            return {
                'race_id': race_id,
                'race_number': race_number,
                'venue': venue,
                'race_date': race_date,
                'filename': filename,
                'grade': grade,  # Add grade from CSV
                'distance': distance  # Add distance from CSV
            }
        
        return None
    
    def scrape_race_results(self, race_info: Dict[str, Any]) -> Dict[str, Any]:
        """Scrape race results including winners, odds, and track conditions from thedogs.com.au"""
        results = {
            'winner': None,
            'winner_odds': None,
            'winner_time': None,
            'track_condition': None,
            'weather': None,
            'field_size': None,
            'race_time': None,
            'all_results': [],
            'scraped_successfully': False,
            'race_url': None
        }
        
        if not self.driver or not BS4_AVAILABLE:
            print("   ⚠️ No web driver available - skipping race results scraping")
            return results
        
        print("   🎯 Attempting broader search strategy...")
        try:
            # Step 1: Find the actual race URL by searching the date page
            race_url = self.find_race_url_from_date_page(race_info)
            if not race_url:
                print("   ⚠️ Could not find race URL for this race")
                return results
            
            print(f"   🌐 Accessing: {race_url}")
            
            # Store the race URL in results and instance for enhanced extraction
            results['race_url'] = race_url
            self._current_race_url = race_url  # Store for enhanced extractor
            
            # Load the race page
            self.driver.get(race_url)
            time.sleep(2)  # Wait for page to load
            
            # Get page source and parse with BeautifulSoup
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Extract race results
            winner_info = self.extract_winner_info(soup)
            if winner_info:
                results['winner'] = winner_info.get('name')
                results['winner_odds'] = winner_info.get('odds')
                results['winner_time'] = winner_info.get('time')
                results['scraped_successfully'] = True
                print(f"   ✅ Winner found: {results['winner']}")
            
            # Extract track conditions
            track_info = self.extract_track_conditions(soup)
            if track_info:
                results['track_condition'] = track_info.get('condition')
                results['weather'] = track_info.get('weather')
                results['race_time'] = track_info.get('race_time')
            
            # Extract all finishing positions
            all_results = self.extract_all_results(soup)
            if all_results:
                results['all_results'] = all_results
                results['field_size'] = len(all_results)
            
            print(f"   ✅ Race results scraped successfully")
            
        except Exception as e:
            print(f"   ❌ Error scraping race results: {e}")
        
        return results
    
    def find_race_url_from_date_page(self, race_info: Dict[str, Any]) -> Optional[str]:
        """Find the actual race URL by searching the date page for race links"""
        try:
            # Format date for URL
            race_date = race_info['race_date']
            if isinstance(race_date, str):
                race_date = datetime.strptime(race_date, '%Y-%m-%d').date()
            
            date_str = race_date.strftime('%Y-%m-%d')
            race_number = race_info['race_number']
            venue = race_info['venue']
            
            # Comprehensive venue mapping (code -> URL format)
            venue_map = {
                # Major metropolitan tracks
                'AP_K': 'angle-park',
                'SAN': 'sandown',
                'WAR': 'warrnambool',
                'BEN': 'bendigo',
                'GEE': 'geelong',
                'BAL': 'ballarat',
                'HOR': 'horsham',
                'TRA': 'traralgon',
                'DAPT': 'dapto',
                'W_PK': 'wentworth-park',
                'APWE': 'albion-park',
                'APTH': 'albion-park',  # Alternative code
                'CANN': 'cannington',
                'MEA': 'the-meadows',
                'HEA': 'healesville',
                'SAL': 'sale',
                'RICH': 'richmond',
                'RICH_S': 'richmond-straight',
                'MURR': 'murray-bridge',
                'GAWL': 'gawler',
                'MOUNT': 'mount-gambier',
                'NOR': 'northam',
                'MAND': 'mandurah',
                
                # NSW tracks
                'GARD': 'the-gardens',
                'GRDN': 'the-gardens',  # Alternative code
                'CAS': 'casino',
                'CASO': 'casino',  # Alternative code
                'WAG': 'wagga',
                'GOUL': 'goulburn',
                'TAR': 'taree',
                'DUB': 'dubbo',
                'GRAF': 'grafton',
                'BH': 'broken-hill',
                'LIS': 'lismore',
                'NOW': 'nowra',
                'TEM': 'temora',
                'TEMA': 'temora',  # Alternative code
                'YOU': 'young',
                'ORA': 'orange',
                'MUD': 'mudgee',
                'COW': 'cowra',
                'BAT': 'bathurst',
                'KAT': 'katoomba',
                'WOL': 'wollongong',
                'INF': 'ingle-farm',
                'BUL': 'bulli',
                'RAY': 'raymond-terrace',
                
                # QLD tracks
                'Q1L': 'ladbrokes-q1-lakeside',
                'QST': 'ladbrokes-q-straight',
                'TWN': 'townsville',
                'CAP': 'capalaba',
                'CAPA': 'capalaba',  # Alternative code
                'IPS': 'ipswich',
                'ROCK': 'rockhampton',
                'BUN': 'bundaberg',
                'CAI': 'cairns',
                'MAC': 'mackay',
                'TOO': 'toowoomba',
                'GC': 'gold-coast',
                'CAL': 'caloundra',
                'MAR': 'maroochy',
                
                # VIC tracks
                'SHEP': 'shepparton',
                'WRGL': 'warragul',
                'WARR': 'warragul',  # Alternative code
                'CRAN': 'cranbourne',
                'MOE': 'moe',
                'PAK': 'pakenham',
                'COL': 'colac',
                'HAM': 'hamilton',
                'PORT': 'portland',
                'ARA': 'ararat',
                'STA': 'stawell',
                'SH': 'swan-hill',
                'MIL': 'mildura',
                'ECH': 'echuca',
                'SEY': 'seymour',
                'KIL': 'kilmore',
                'WOD': 'wodonga',
                'WOD_G': 'wodonga-gvgrc',
                
                # SA tracks
                'VIR': 'virginia',
                'STR': 'strathalbyn',
                'WHY': 'whyalla',
                'PA': 'port-augusta',
                'PP': 'port-pirie',
                'GLE': 'glenelg',
                
                # WA tracks
                'ALB': 'albany',
                'GER': 'geraldton',
                'KAL': 'kalgoorlie',
                'BUNB': 'bunbury',
                'ESP': 'esperance',
                'BRO': 'broome',
                'KAR': 'karratha',
                'PH': 'port-hedland',
                'KUN': 'kununurra',
                
                # TAS tracks
                'HOB': 'hobart',
                'HOBT': 'hobart',  # Alternative code
                'LAU': 'launceston',
                'DEV': 'devonport',
                
                # NT tracks
                'DAR': 'darwin',
                'DARW': 'darwin',  # Alternative code
                'AS': 'alice-springs',
                
                # ACT tracks
                'CANB': 'canberra',
                
                # Legacy codes for compatibility
                'GOSF': 'gosford',
                'GUNN': 'gunnedah'
            }
            
            venue_url = venue_map.get(venue)
            if not venue_url:
                print(f"   ⚠️ Unknown venue code: {venue}")
                return None
            
            # Step 1: Access the date page to find race links
            date_page_url = f"https://www.thedogs.com.au/racing/{date_str}"
            print(f"   🔍 Searching for race links on: {date_page_url}")
            
            self.driver.get(date_page_url)
            time.sleep(2)  # Wait for page to load
            
            # Get page source and parse with BeautifulSoup
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Find race links that match our criteria
            race_links = []
            
            # Look for all links that contain racing URLs
            all_links = soup.find_all('a', href=True)
            
            print(f"   🔍 Scanning {len(all_links)} links on the date page...")
            
            for link in all_links:
                href = link.get('href')
                if not href:
                    continue
                    
                # Make sure it's a full URL
                if not href.startswith('http'):
                    href = f"https://www.thedogs.com.au{href}"
                
                # Check if this looks like a race URL
                if ('/racing/' in href and date_str in href and 
                    venue_url in href and f'/{race_number}/' in href):
                    race_links.append(href)
                    print(f"   🎯 Found potential race URL: {href}")
            
            # Remove duplicates
            unique_links = list(set(race_links))
            
            if unique_links:
                print(f"   ✅ Found {len(unique_links)} matching race URLs")
                return unique_links[0]  # Return the first match
            else:
                # If no exact match found, let's try more comprehensive search
                print(f"   🔍 No exact match found, scanning every link...")
                broader_links = []

                for link in all_links:
                    href = link.get('href')
                    if not href:
                        continue
                        
                    # Make sure it's a full URL
                    if not href.startswith('http'):
                        href = f"https://www.thedogs.com.au{href}"
                    
                    # Adjust search criteria
                    if ('/racing/' in href and venue_url in href):
                        broader_links.append(href)
                        print(f"   🎯 Potential match found: {href}")
                
                unique_broader_links = list(set(broader_links))
                unique_broader_links.sort()  # Sort to get a consistent selection

                if unique_broader_links:
                    print(f"   ✅ Found {len(unique_broader_links)} potential matches")
                    return unique_broader_links[0]
                else:
                    print(f"   ⚪ No matching race URLs found for {venue} Race {race_number}")
                    # Debug: show some sample links
                    sample_links = [link.get('href') for link in all_links[:10] if link.get('href') and '/racing/' in link.get('href', '')]
                    if sample_links:
                        print(f"   🔍 Sample racing links found: {sample_links[:3]}")
                    return None
            
        except Exception as e:
            print(f"   ❌ Error finding race URL: {e}")
            return None
    
    def construct_race_url(self, race_info: Dict[str, Any]) -> Optional[str]:
        """Construct the race URL from race information - DEPRECATED: Use find_race_url_from_date_page instead"""
        try:
            # Map venue codes to URL format
            venue_map = {
                'HEA': 'healesville',
                'SAL': 'sale',
                'SAN': 'sandown',
                'MEA': 'the-meadows',
                'WAR': 'warrnambool',
                'AP_K': 'angle-park',
                'APTH': 'albion-park',
                'APWE': 'albion-park',
                'BAL': 'ballarat',
                'BEN': 'bendigo',
                'CANN': 'cannington',
                'CASO': 'casino',
                'DAPT': 'dapto',
                'GEE': 'geelong',
                'GAWL': 'gawler',
                'HOR': 'horsham',
                'MOUNT': 'mount-gambier',
                'MURR': 'murray-bridge',
                'RICH': 'richmond',
                'TRA': 'traralgon',
                'W_PK': 'wentworth-park',
                'MAND': 'mandurah',
                'HOBT': 'hobart',
                'SHEP': 'shepparton',
                'WARR': 'warragul',
                'NOR': 'northam',
                'TEMA': 'temora',
                'GUNN': 'gunnedah',
                'CAPA': 'capalaba',
                'ROCK': 'rockhampton',
                'DARW': 'darwin',
                'GRDN': 'the-gardens'
            }
            
            venue_url = venue_map.get(race_info['venue'])
            if not venue_url:
                return None
            
            # Format date for URL
            race_date = race_info['race_date']
            if isinstance(race_date, str):
                race_date = datetime.strptime(race_date, '%Y-%m-%d').date()
            
            date_str = race_date.strftime('%Y-%m-%d')
            race_number = race_info['race_number']
            
            # Try different URL patterns - add trailing slash as per website structure
            base_url = f"https://www.thedogs.com.au/racing/{venue_url}/{date_str}/{race_number}/"
            
            return base_url
            
        except Exception as e:
            print(f"   ❌ Error constructing URL: {e}")
            return None
    
    def extract_winner_info(self, soup) -> Optional[Dict[str, Any]]:
        """Extract winner information from race results page"""
        try:
            # Look for winner information in various selectors - updated for current HTML structure
            winner_selectors = [
                '.race-result .winner',
                '.result-row[data-position="1"]',
                '.finishing-position-1',
                '.first-place',
                '[data-finish="1"]',
                '.runner[data-position="1"]',
                '.result-table tbody tr:first-child',
                'table.results tr:first-child',
                '.race-results .winner',
                '.results-grid .position-1'
            ]
            
            for selector in winner_selectors:
                winner_element = soup.select_one(selector)
                if winner_element:
                    # Extract dog name with multiple methods
                    name = None
                    name_selectors = [
                        '.runner-name',
                        '.dog-name', 
                        '.name',
                        'h3',
                        'h4',
                        'span.runner-name',
                        '.greyhound-name',
                        'a[href*="greyhound"]'
                    ]
                    
                    for name_sel in name_selectors:
                        name_elem = winner_element.select_one(name_sel)
                        if name_elem:
                            name = name_elem.get_text(strip=True)
                            break
                    
                    # Extract odds with multiple methods
                    odds = None
                    odds_selectors = [
                        '.odds',
                        '.price',
                        '.starting-price',
                        '.sp',
                        '[data-odds]',
                        '.odds-value'
                    ]
                    
                    for odds_sel in odds_selectors:
                        odds_elem = winner_element.select_one(odds_sel)
                        if odds_elem:
                            odds = odds_elem.get_text(strip=True)
                            break
                    
                    # Extract time
                    time_val = None
                    time_selectors = [
                        '.time',
                        '.race-time',
                        '.finish-time',
                        '[data-time]',
                        '.result-time'
                    ]
                    
                    for time_sel in time_selectors:
                        time_elem = winner_element.select_one(time_sel)
                        if time_elem:
                            time_val = time_elem.get_text(strip=True)
                            break
                    
                    if name:
                        # Clean up the name - remove time stamps, trainer info, etc.
                        clean_name = re.sub(r'[\d\.]+[A-Z]*:?\s*', '', name)  # Remove timestamps like "19.22T:"
                        clean_name = re.sub(r'R/T:?\s*$', '', clean_name)  # Remove "R/T:" at end
                        clean_name = re.sub(r'T:?\s*[\w\s]*$', '', clean_name)  # Remove trainer info
                        clean_name = clean_name.strip()
                        
                        # Parse odds if found
                        odds_value = None
                        if odds:
                            odds_match = re.search(r'([\d\.]+)', odds)
                            if odds_match:
                                try:
                                    odds_value = float(odds_match.group(1))
                                except:
                                    pass
                        
                        # Parse time if found
                        time_value = None
                        if time_val:
                            time_match = re.search(r'([\d\.]+)', time_val)
                            if time_match:
                                try:
                                    time_value = float(time_match.group(1))
                                except:
                                    pass
                        
                        return {
                            'name': clean_name,
                            'odds': odds_value,
                            'time': time_value
                        }
            
            # Fallback: look for any results table and get first position
            table_selectors = [
                'table.results',
                'table.result-table',
                'table[class*="result"]',
                '.results-table table',
                'table'
            ]
            
            for table_sel in table_selectors:
                results_table = soup.select_one(table_sel)
                if results_table:
                    # Look for first data row (skip header)
                    rows = results_table.find_all('tr')
                    if len(rows) > 1:
                        first_row = rows[1]  # Skip header row
                        cells = first_row.find_all(['td', 'th'])
                        if len(cells) >= 2:
                            # Try to extract name from various cell positions
                            for i, cell in enumerate(cells):
                                text = cell.get_text(strip=True)
                                # Look for dog name patterns (not just numbers or short text)
                                if text and len(text) > 3 and not text.isdigit() and not text in ['1st', '2nd', '3rd', '1', '2', '3']:
                                    return {'name': text, 'odds': None, 'time': None}
            
            # Last resort: look for any text containing '1st' or position indicators
            position_texts = soup.find_all(text=re.compile(r'1st|winner|won', re.IGNORECASE))
            for text in position_texts:
                parent = text.parent
                if parent:
                    # Look for dog name near the winner indicator
                    siblings = parent.find_next_siblings()
                    for sibling in siblings:
                        if sibling.name and sibling.get_text(strip=True):
                            potential_name = sibling.get_text(strip=True)
                            if len(potential_name) > 3 and not potential_name.isdigit():
                                return {'name': potential_name, 'odds': None, 'time': None}
            
        except Exception as e:
            print(f"   ❌ Error extracting winner info: {e}")
        
        return None
    
    def extract_track_conditions(self, soup) -> Optional[Dict[str, Any]]:
        """Enhanced track condition extraction with false positive prevention"""
        try:
            from enhanced_track_condition_extractor import EnhancedTrackConditionExtractor
            
            # Use enhanced extractor
            extractor = EnhancedTrackConditionExtractor()
            race_url = getattr(self, '_current_race_url', '')
            
            # Try enhanced extraction first
            enhanced_result = extractor.extract_track_conditions_enhanced(soup, race_url)
            if enhanced_result and enhanced_result.get('condition'):
                return enhanced_result
            
            # Fallback to basic extraction with anti-sponsorship filters
            return self._extract_track_conditions_basic_filtered(soup, race_url)
            
        except ImportError:
            print("   ⚠️ Enhanced extractor not available, using basic extraction")
            return self._extract_track_conditions_basic_filtered(soup, getattr(self, '_current_race_url', ''))
        except Exception as e:
            print(f"   ❌ Error in enhanced track condition extraction: {e}")
            return self._extract_track_conditions_basic_filtered(soup, getattr(self, '_current_race_url', ''))
    
    def _extract_track_conditions_basic_filtered(self, soup, race_url: str = '') -> Optional[Dict[str, Any]]:
        """Basic track condition extraction with sponsorship filtering"""
        try:
            conditions = {}
            
            # Sponsorship patterns to avoid
            sponsorship_patterns = [
                'ladbrokes-fast-withdrawals',
                'sportsbet-fast-form', 
                'tab-fast-',
                'bet365-fast-',
                'pointsbet-fast-'
            ]
            
            # Look for track condition information with expanded selectors
            condition_selectors = [
                '.track-condition',
                '.conditions',
                '[data-condition]',
                '.race-conditions .condition',
                '.meeting-conditions',
                '.track-details .condition',
                '.race-info .condition',
                '.conditions-panel .condition',
                '.race-header .condition',
                '.meeting-info .track-condition',
                '.track-info',
                '.meeting-info',
                '.race-details',
                '.track-data',
                '.race-card-header',
                '.meeting-header',
                '.race-meeting-info'
            ]
            
            for selector in condition_selectors:
                elem = soup.select_one(selector)
                if elem:
                    text = elem.get_text(strip=True)
                    # Filter out navigation/venue lists and other irrelevant text
                    if text and len(text) > 2 and len(text) < 100:  # Not too long (likely navigation)
                        # Check if it's actually track condition info, not venue listings or sponsorship
                        if not any(word in text for word in ['TAS', 'NSW', 'VIC', 'SA', 'QLD', 'WA', 'Fields', 'races', 'R1', 'R2', 'R3', 'R4', 'R5', 'Ladbrokes', 'Sportsbet']):
                            # Additional check: avoid if condition appears in race URL (sponsorship)
                            if not (race_url and text.lower() in race_url.lower()):
                                conditions['condition'] = text
                                print(f"   🏁 Found track condition via selector '{selector}': {text}")
                                break
            
            # Look for weather information
            weather_selectors = [
                '.weather',
                '.weather-info',
                '[data-weather]',
                '.race-conditions .weather',
                '.meeting-conditions .weather',
                '.track-details .weather',
                '.race-info .weather',
                '.conditions-panel .weather',
                '.race-header .weather',
                '.meeting-info .weather'
            ]
            
            for selector in weather_selectors:
                elem = soup.select_one(selector)
                if elem:
                    conditions['weather'] = elem.get_text(strip=True)
                    break
            
            # Alternative: look for text patterns with sponsorship filtering
            if not conditions.get('condition'):
                page_text = soup.get_text()
                page_text = re.sub(r'\s+', ' ', page_text)
                
                # Enhanced patterns that avoid sponsorship context
                condition_patterns = [
                    r'track\s+condition[:\s]+([a-z]+)',
                    r'surface[:\s]+([a-z]+)',
                    r'meeting\s+condition[:\s]+([a-z]+)',
                    # Very specific context patterns only
                    r'(?:today\'s|race\s+day)\s+track\s+condition[:\s]+([a-z]+)'
                ]
                
                for pattern in condition_patterns:
                    matches = re.finditer(pattern, page_text, re.IGNORECASE)
                    for match in matches:
                        condition_text = match.group(1).strip().lower()
                        
                        # Skip if this appears to be from sponsorship
                        if race_url and any(sponsor in race_url.lower() for sponsor in sponsorship_patterns):
                            if condition_text in race_url.lower():
                                continue
                        
                        # Check surrounding context for sponsorship indicators
                        start = max(0, match.start() - 100)
                        end = min(len(page_text), match.end() + 100)
                        context = page_text[start:end].lower()
                        
                        if any(sponsor in context for sponsor in ['ladbrokes', 'sportsbet', 'withdrawals', 'form']):
                            continue
                        
                        # Normalize condition
                        if condition_text in ['fast', 'good', 'slow', 'heavy', 'dead', 'firm', 'soft']:
                            conditions['condition'] = condition_text.title()
                            print(f"   🏁 Found track condition via filtered pattern: {condition_text.title()}")
                            break
                    
                    if conditions.get('condition'):
                        break
            
            # Look for weather patterns
            if not conditions.get('weather'):
                weather_patterns = [
                    r'weather[\s:]*([\w\s]+?)(?:track|condition|temperature|\.|$)',
                    r'(fine|sunny|cloudy|overcast|rainy|wet|drizzle|shower)\s*weather',
                    r'weather\s*(fine|sunny|cloudy|overcast|rainy|wet|drizzle|shower)'
                ]
                
                for pattern in weather_patterns:
                    match = re.search(pattern, page_text, re.IGNORECASE)
                    if match:
                        conditions['weather'] = match.group(1).strip()
                        break
            
            # Look for temperature information
            temp_selectors = [
                '.temperature',
                '.temp',
                '[data-temperature]',
                '.race-conditions .temperature',
                '.meeting-conditions .temperature'
            ]
            
            for selector in temp_selectors:
                elem = soup.select_one(selector)
                if elem:
                    conditions['temperature'] = elem.get_text(strip=True)
                    break
            
            # Look for temperature in text patterns
            if not conditions.get('temperature'):
                temp_patterns = [
                    r'temperature[\s:]*([\d\.]+)\s*°?[cf]?',
                    r'([\d\.]+)\s*°[cf]',
                    r'temp[\s:]*([\d\.]+)\s*°?[cf]?'
                ]
                
                for pattern in temp_patterns:
                    match = re.search(pattern, page_text, re.IGNORECASE)
                    if match:
                        conditions['temperature'] = match.group(1).strip()
                        break
            
            return conditions if conditions else None
            
        except Exception as e:
            print(f"   ❌ Error extracting track conditions: {e}")
        
        return None
    
    def extract_all_results(self, soup) -> List[Dict[str, Any]]:
        """Extract all race results from the page"""
        try:
            results = []
            
            # First try to find results table with various selectors
            results_selectors = [
                'table.results',
                'table.result-table',
                'table[class*="result"]',
                '.results-table table',
                '.race-results table',
                'table',
                '.results-grid',
                '.race-result',
                '.finishing-order',
                '.results-container'
            ]
            
            results_container = None
            for selector in results_selectors:
                results_container = soup.select_one(selector)
                if results_container:
                    print(f"   📊 Found results using selector: {selector}")
                    break
            
            if results_container:
                rows = results_container.find_all('tr')
                if len(rows) > 1:  # Has header and data rows
                    for row in rows[1:]:  # Skip header
                        cells = row.find_all(['td', 'th'])
                        if len(cells) >= 2:
                            try:
                                position = cells[0].get_text(strip=True)
                                name = cells[1].get_text(strip=True)
                                
                                # Skip if position is not a number (likely header)
                                if not position or not position[0].isdigit():
                                    continue
                                
                                # Parse the name field which might contain dog name + trainer info
                                full_info = name if name else ''
                                if len(cells) > 2:
                                    full_info = cells[2].get_text(strip=True)
                                
                                # Extract dog name and trainer from the full info
                                parsed_info = self.parse_scraped_result_components(full_info)
                                
                                result = {
                                    'position': position,
                                    'name': parsed_info['clean_name'] or full_info.split('T:')[0].strip(),
                                    'trainer': parsed_info['trainer_name'],
                                    'full_info': full_info
                                }
                                
                                # Try to extract additional info
                                if len(cells) > 3:
                                    result['time'] = cells[3].get_text(strip=True)
                                
                                results.append(result)
                                print(f"   📊 Found result: {position} - {name}")
                            except Exception as e:
                                print(f"   ⚠️ Error processing result row: {e}")
                                continue
                else:
                    # Try div-based results
                    result_divs = results_container.find_all('div', class_=re.compile(r'result|runner|position'))
                    for i, div in enumerate(result_divs):
                        try:
                            text = div.get_text(strip=True)
                            if text and any(char.isdigit() for char in text):
                                # Try to extract position and name
                                position_match = re.search(r'^(\d+)', text)
                                if position_match:
                                    position = position_match.group(1)
                                    name = text.replace(position_match.group(0), '').strip()
                                    if name:
                                        results.append({
                                            'position': position,
                                            'name': name
                                        })
                                        print(f"   📊 Found div result: {position} - {name}")
                        except:
                            continue
            
            # If no results found, try to extract from the winner info we already have
            if not results:
                print("   ⚠️ No results table found, trying to extract from page text")
                page_text = soup.get_text()
                # Look for position patterns in text
                position_patterns = [
                    r'1st\s*([A-Za-z\s]+)',
                    r'2nd\s*([A-Za-z\s]+)',
                    r'3rd\s*([A-Za-z\s]+)',
                    r'Position\s*(\d+)\s*([A-Za-z\s]+)',
                    r'(\d+)\s*([A-Za-z\s]+?)\s*(?:won|finished|placed)'
                ]
                
                for pattern in position_patterns:
                    matches = re.findall(pattern, page_text, re.IGNORECASE)
                    for match in matches:
                        if isinstance(match, tuple) and len(match) >= 2:
                            position = match[0] if match[0].isdigit() else '1'
                            name = match[1].strip()
                        else:
                            position = '1'
                            name = match.strip()
                        
                        if name and len(name) > 2:
                            results.append({
                                'position': position,
                                'name': name
                            })
                            print(f"   📊 Found text result: {position} - {name}")
                            break
            
            print(f"   📊 Total results extracted: {len(results)}")
            return results
            
        except Exception as e:
            print(f"   ❌ Error extracting all results: {e}")
        
        return []
    
    def combine_race_data(self, race_info: Dict[str, Any], race_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine race info from filename with scraped race results"""
        combined = race_info.copy()
        
        # Add scraped data with parsed components
        winner_name = race_results.get('winner', '')
        if winner_name:
            # Parse the winner result to extract all components
            winner_components = self.parse_scraped_result_components(winner_name)
            combined['winner_name'] = winner_components['clean_name'] or winner_name
            combined['winner_scraped_trainer'] = winner_components['trainer_name']
            combined['winner_reaction_time'] = winner_components['reaction_time']
            combined['winner_nbtt'] = winner_components['nbtt']
            combined['winner_race_classification'] = winner_components['race_classification']
            combined['winner_raw_result'] = winner_components['raw_result']
        else:
            combined['winner_name'] = ''
            combined['winner_scraped_trainer'] = ''
            combined['winner_reaction_time'] = ''
            combined['winner_nbtt'] = ''
            combined['winner_race_classification'] = ''
            combined['winner_raw_result'] = ''
        
        combined['winner_odds'] = race_results.get('winner_odds')
        combined['winner_time'] = race_results.get('winner_time')
        combined['track_condition'] = race_results.get('track_condition')
        combined['weather'] = race_results.get('weather')
        combined['field_size'] = race_results.get('field_size')
        combined['race_time'] = race_results.get('race_time')
        combined['url'] = race_results.get('race_url')  # Add the race URL
        combined['extraction_timestamp'] = datetime.now()
        combined['data_source'] = 'enhanced_processor_with_scraping'
        
        # Set scraped_successfully flag based on whether we have scraped data
        combined['scraped_successfully'] = race_results.get('scraped_successfully', False)
        
        return combined
    
    def parse_scraped_result_components(self, result_string: str) -> Dict[str, str]:
        """Parse scraped race result to extract individual components"""
        components = {
            'clean_name': '',
            'trainer_name': '',
            'reaction_time': '',
            'nbtt': '',
            'race_classification': '',
            'raw_result': result_string
        }
        
        if not result_string:
            return components
        
        # Store the raw result
        components['raw_result'] = result_string
        
        # Extract time (pattern: "19.64T:")
        time_match = re.search(r'(\d{1,2}\.\d{2})T:', result_string)
        if time_match:
            components['reaction_time'] = time_match.group(1)
        
        # Extract NBTT
        if 'NBTT:' in result_string:
            components['nbtt'] = 'NBTT'
        
        # Extract race classification (after R/T: or at end)
        rt_match = re.search(r'R/T:\s*([A-Z]{1,3})\s*$', result_string)
        if rt_match:
            components['race_classification'] = rt_match.group(1)
        
        # Extract dog name and trainer name
        work_string = result_string
        
        # Remove time stamps and indicators to isolate name section
        work_string = re.sub(r'\d{1,2}\.\d{2}T:', '', work_string).strip()
        work_string = re.sub(r'R/T:.*$', '', work_string).strip()
        work_string = re.sub(r'NBTT:', '', work_string).strip()
        
        # Common patterns for dog names + trainer names:
        # "Dog Name" + "TrainerFirstName TrainerLastName"
        # "Dog Name" + "TrainerName"
        
        # Try to match: "Dog Name" + "TrainerFirstName TrainerLastName"
        pattern1 = r'^(.+?)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\s*$'
        match1 = re.match(pattern1, work_string)
        if match1:
            components['clean_name'] = match1.group(1).strip()
            components['trainer_name'] = match1.group(2).strip()
        else:
            # Try to match: "Dog Name" + "TrainerName"
            pattern2 = r'^(.+?)\s+([A-Z][a-z]+)\s*$'
            match2 = re.match(pattern2, work_string)
            if match2 and len(match2.group(1).split()) >= 2:  # Dog name should have at least 2 words
                components['clean_name'] = match2.group(1).strip()
                components['trainer_name'] = match2.group(2).strip()
            else:
                # If no clear trainer pattern, use the whole string as dog name
                components['clean_name'] = work_string.strip()
                components['trainer_name'] = ''
        
        return components
    
    def enhance_race_metadata(self, race_info: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance race metadata with weather, track conditions, and other data"""
        enhanced_info = race_info.copy()
        
        # Get weather data
        if self.weather_api_key:
            weather_data = self.get_weather_data(race_info['venue'], race_info['race_date'])
            if weather_data:
                enhanced_info.update(weather_data)
        
        # Get track conditions if available
        track_conditions = self.get_track_conditions(race_info['venue'], race_info['race_date'])
        if track_conditions:
            enhanced_info.update(track_conditions)
        
        # Try to scrape additional race information
        if self.driver and BS4_AVAILABLE:
            scraped_data = self.scrape_race_details(race_info)
            if scraped_data:
                enhanced_info.update(scraped_data)
        
        # Add processing timestamp
        enhanced_info['processing_timestamp'] = datetime.now()
        
        return enhanced_info
    
    def get_weather_data(self, venue: str, race_date) -> Optional[Dict[str, Any]]:
        """Get weather data for the race venue and date"""
        if not self.weather_api_key:
            return None
        
        # Venue coordinates (approximate)
        venue_coords = {
            'HEA': (-37.6, 145.2),  # Healesville
            'SAL': (-38.1, 147.1),  # Sale
            'SAN': (-37.9, 145.1),  # Sandown
            'MEA': (-37.7, 144.7),  # The Meadows
            'WAR': (-38.4, 142.5),  # Warrnambool
            'AP_K': (-34.7, 138.5), # Angle Park
            'MOUNT': (-37.8, 140.8), # Mount Gambier
            'MURR': (-35.1, 139.3), # Murray Bridge
            'RICH': (-37.8, 144.9), # Richmond
            'TRA': (-38.2, 146.5),  # Traralgon
        }
        
        coords = venue_coords.get(venue)
        if not coords:
            return None
        
        try:
            lat, lon = coords
            url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={self.weather_api_key}&units=metric"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                return {
                    'weather': data['weather'][0]['description'],
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'wind_speed': data['wind']['speed'],
                    'wind_direction': data['wind'].get('deg', 0)
                }
        except Exception as e:
            print(f"⚠️ Weather API error: {e}")
        
        return None
    
    def get_track_conditions(self, venue: str, race_date) -> Optional[Dict[str, Any]]:
        """Get track conditions (placeholder - would need venue-specific APIs)"""
        # This would typically scrape from venue websites or APIs
        # For now, return some default conditions
        return {
            'track_condition': 'Good',
            'rail_position': 'True',
            'track_rating': 5.0
        }
    
    def scrape_race_details(self, race_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Scrape additional race details from thedogs.com.au"""
        if not self.driver:
            return None
        
        try:
            # Construct URL (this would need to be adapted based on actual URL structure)
            venue_map = {
                'HEA': 'healesville',
                'SAL': 'sale',
                'SAN': 'sandown',
                'MEA': 'the-meadows',
                'WAR': 'warrnambool',
                'AP_K': 'angle-park',
                'MOUNT': 'mount-gambier',
                'MURR': 'murray-bridge',
                'RICH': 'richmond',
                'TRA': 'traralgon'
            }
            
            venue_slug = venue_map.get(race_info['venue'])
            if not venue_slug:
                return None
            
            race_date_str = race_info['race_date'].strftime('%Y-%m-%d')
            race_url = f"https://www.thedogs.com.au/racing/{race_date_str}/{venue_slug}/{race_info['race_number']}"
            
            self.driver.get(race_url)
            time.sleep(2)  # Wait for page to load
            
            # Extract additional data
            scraped_data = {}
            
            # Try to get prize money
            try:
                prize_element = self.driver.find_element(By.CSS_SELECTOR, '.prize-money, .stake')
                if prize_element:
                    prize_text = prize_element.text
                    prize_match = re.search(r'\$?([\d,]+)', prize_text)
                    if prize_match:
                        scraped_data['prize_money_total'] = float(prize_match.group(1).replace(',', ''))
            except:
                pass
            
            # Try to get race grade/class
            try:
                grade_element = self.driver.find_element(By.CSS_SELECTOR, '.grade, .class')
                if grade_element:
                    scraped_data['grade'] = grade_element.text.strip()
            except:
                pass
            
            # Try to get distance
            try:
                distance_element = self.driver.find_element(By.CSS_SELECTOR, '.distance')
                if distance_element:
                    distance_text = distance_element.text
                    distance_match = re.search(r'(\d+)m', distance_text)
                    if distance_match:
                        scraped_data['distance'] = distance_match.group(1)
            except:
                pass
            
            return scraped_data if scraped_data else None
            
        except Exception as e:
            print(f"⚠️ Scraping error: {e}")
            return None
    
    def process_dog_data(self, row, race_info: Dict[str, Any], race_results: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Process individual dog data with enhanced analysis and race results"""
        try:
            # Extract basic data from form guide
            # CRITICAL: Form guides contain historical data, not current race results
            # PLC column contains historical race positions, NOT finish position for current race
            dog_data = {
                'race_id': race_info['race_id'],
                'dog_name': str(row.get('Dog Name', '')).strip(),
                'box_number': self.safe_int(row.get('BOX', 0)),
                'finish_position': 'N/A',  # Form guides don't contain current race results
                'trainer_name': '',  # Will be filled from scraped data if available
                'weight': self.safe_float(row.get('WGT', 0)),
                'starting_price': self.safe_float(row.get('SP', 0)),
                'individual_time': str(row.get('TIME', '')).strip(),
                'margin': str(row.get('MGN', '')).strip(),
                'sectional_1st': str(row.get('1 SEC', '')).strip(),
                'sectional_2nd': str(row.get('2 SEC', '')).strip(),
                'sectional_3rd': str(row.get('3 SEC', '')).strip(),
                'beaten_margin': self.safe_float(row.get('MGN', 0)),
                'extraction_timestamp': datetime.now(),
                'data_source': 'enhanced_processor_with_results',
                # Store historical PLC data separately for analysis
                'historical_plc': str(row.get('PLC', '')).strip()
            }

            # Clean dog name
            dog_data['dog_clean_name'] = self.clean_dog_name(dog_data['dog_name'])

            # Add race results data if available
            if race_results:
                # Check if this dog was the winner
                winner_name = race_results.get('winner')
                winner_parsed = self.parse_scraped_result_components(winner_name)
                if winner_parsed['clean_name'] and self.dogs_match(dog_data['dog_clean_name'], winner_parsed['clean_name']):
                    dog_data['was_winner'] = True
                    dog_data['winner_time'] = race_results.get('winner_time')
                    dog_data['winner_odds'] = race_results.get('winner_odds')
                    dog_data['finish_position'] = '1'  # Winner gets position 1
                else:
                    dog_data['was_winner'] = False

                # Try to find this dog in the complete race results
                for result in race_results.get('all_results', []):
                    if self.dogs_match(dog_data['dog_clean_name'], result.get('name', '')):
                        dog_data['actual_finish_position'] = result.get('position') or dog_data['finish_position']
                        dog_data['actual_odds'] = result.get('odds')
                        dog_data['actual_time'] = result.get('time')
                        # Use the scraped finish position instead of form guide
                        dog_data['finish_position'] = result.get('position', '') or dog_data['finish_position']

                        # Parse scraped race data components for this dog
                        scraped_name = result.get('name', '')
                        if scraped_name:
                            scraped_components = self.parse_scraped_result_components(scraped_name)
                            dog_data['scraped_trainer_name'] = scraped_components['trainer_name']
                            dog_data['scraped_reaction_time'] = scraped_components['reaction_time']
                            dog_data['scraped_nbtt'] = scraped_components['nbtt']
                            dog_data['scraped_race_classification'] = scraped_components['race_classification']
                            dog_data['scraped_raw_result'] = scraped_components['raw_result']
                            # Use scraped trainer name if available
                            if scraped_components['trainer_name']:
                                dog_data['trainer_name'] = scraped_components['trainer_name']
                        break

                # If no race results found, set finish position to default from form guide
                if not any(self.dogs_match(dog_data['dog_clean_name'], result.get('name', '')) 
                          for result in race_results.get('all_results', [])):
                    dog_data['finish_position'] = dog_data['finish_position'] or 'N/A'
                    dog_data['scraped_trainer_name'] = ''
                    dog_data['scraped_reaction_time'] = ''
                    dog_data['scraped_nbtt'] = ''
                    dog_data['scraped_race_classification'] = ''
                    dog_data['scraped_raw_result'] = ''
            else:
                # No race results available, set scraped fields to empty
                dog_data['scraped_trainer_name'] = ''
                dog_data['scraped_reaction_time'] = ''
                dog_data['scraped_nbtt'] = ''
                dog_data['scraped_race_classification'] = ''
                dog_data['scraped_raw_result'] = ''
            
            # Calculate enhanced metrics
            dog_data['performance_rating'] = self.calculate_performance_rating(row)
            dog_data['speed_rating'] = self.calculate_speed_rating(row)
            dog_data['class_rating'] = self.calculate_class_rating(row)
            
            # Calculate win/place probabilities
            probabilities = self.calculate_probabilities(row, race_info)
            dog_data.update(probabilities)
            
            # Store original form guide data as JSON
            form_guide_data = {
                'sex': str(row.get('Sex', '')),
                'track': str(row.get('TRACK', '')),
                'distance': str(row.get('DIST', '')),
                'date': str(row.get('DATE', '')),
                'grade': str(row.get('G', '')),
                'bonus': str(row.get('BON', '')),
                'pir': str(row.get('PIR', '')),
                'w_2g': str(row.get('W/2G', ''))
            }
            dog_data['form_guide_json'] = json.dumps(form_guide_data)
            
            # Initialize best_time and recent_form (will be calculated later in process_grouped_dog_data)
            dog_data['best_time'] = 0.0
            dog_data['recent_form'] = ''
            
            return dog_data
            
        except Exception as e:
            print(f"⚠️ Error processing dog data: {e}")
            return None
    
    def validate_winner_consistency(self, race_info: Dict[str, Any], processed_dogs: List[Dict[str, Any]]):
        """Validate and fix winner data consistency between scraped results and form guide"""
        try:
            scraped_winner_name = race_info.get('winner_name', '')
            
            if not scraped_winner_name:
                print("   ⚠️ No scraped winner found")
                return
            
            # Find dogs with finish_position = 1
            form_guide_winners = [dog for dog in processed_dogs if dog.get('finish_position') == '1']
            
            # Check if scraped winner exists in form guide dogs
            scraped_winner_in_form_guide = None
            for dog in processed_dogs:
                if self.dogs_match(dog.get('dog_clean_name', ''), scraped_winner_name):
                    scraped_winner_in_form_guide = dog
                    break
            
            if scraped_winner_in_form_guide:
                # Scraped winner exists in form guide - make sure it has position 1
                if scraped_winner_in_form_guide.get('finish_position') != '1':
                    print(f"   🔧 Fixing winner position for {scraped_winner_name}")
                    scraped_winner_in_form_guide['finish_position'] = '1'
                    scraped_winner_in_form_guide['was_winner'] = True
                    
                    # Reset other dogs with position 1
                    for dog in processed_dogs:
                        if dog != scraped_winner_in_form_guide and dog.get('finish_position') == '1':
                            dog['finish_position'] = 'N/A'
                            dog['was_winner'] = False
                            
                print(f"   ✅ Winner consistency validated: {scraped_winner_name}")
            else:
                # Scraped winner not found in form guide - this indicates CSV/scraping mismatch
                print(f"   ⚠️ CSV/Scraping mismatch: Winner '{scraped_winner_name}' not found in form guide")
                print(f"   📋 Form guide dogs: {[dog.get('dog_clean_name', '') for dog in processed_dogs]}")
                
                # If there are form guide winners, note the discrepancy
                if form_guide_winners:
                    form_guide_winner_names = [dog.get('dog_clean_name', '') for dog in form_guide_winners]
                    print(f"   📋 Form guide winners: {form_guide_winner_names}")
                    
                    # Option 1: Trust the scraped winner and add a note
                    # Option 2: Trust the form guide winner and add a note
                    # For now, we'll trust the scraped winner and add a note
                    race_info['data_quality_note'] = f"Winner mismatch: Scraped='{scraped_winner_name}', Form guide={form_guide_winner_names}"
                    
        except Exception as e:
            print(f"   ⚠️ Error validating winner consistency: {e}")
    
    def assign_finish_positions_from_scraped_results(self, race_results: Dict[str, Any], processed_dogs: List[Dict[str, Any]]):
        """Assign finish positions from scraped race results to avoid duplicates"""
        try:
            all_results = race_results.get('all_results', [])
            
            if not all_results:
                print("   ⚠️ No scraped race results available for position assignment")
                return
            
            # Create a mapping of dog names to positions and trainers from scraped results
            scraped_positions = {}
            scraped_trainers = {}
            for result in all_results:
                result_name = result.get('name', '')
                position = result.get('position', '')
                trainer = result.get('trainer', '')
                
                if result_name and position:
                    # Clean the name to match with form guide names
                    cleaned_name = self.clean_dog_name(result_name)
                    scraped_positions[cleaned_name.lower()] = str(position).replace('st', '').replace('nd', '').replace('rd', '').replace('th', '')
                    scraped_trainers[cleaned_name.lower()] = trainer
            
            # Assign positions to dogs based on scraped results
            assigned_count = 0
            for dog in processed_dogs:
                dog_name = dog.get('dog_clean_name', '')
                if dog_name:
                    # Try to find matching position in scraped results
                    cleaned_dog_name = self.clean_dog_name(dog_name).lower()
                    
                    # Look for exact match first
                    found_position = None
                    found_trainer = None
                    for scraped_name, position in scraped_positions.items():
                        if self.dogs_match(dog_name, scraped_name):
                            found_position = position
                            found_trainer = scraped_trainers.get(scraped_name, '')
                            break
                    
                    if found_position:
                        old_position = dog.get('finish_position', '')
                        dog['finish_position'] = found_position
                        dog['scraped_finish_position'] = found_position
                        if found_trainer:
                            dog['scraped_trainer_name'] = found_trainer
                        assigned_count += 1
                        
                        if old_position != found_position:
                            print(f"   🔄 Updated {dog_name}: {old_position} -> {found_position}")
                    else:
                        # If no scraped position found, mark as unknown
                        dog['finish_position'] = 'N/A'
                        dog['scraped_finish_position'] = 'N/A'
            
            print(f"   ✅ Assigned {assigned_count}/{len(processed_dogs)} finish positions from scraped results")
            
        except Exception as e:
            print(f"   ⚠️ Error assigning finish positions: {e}")
    
    def dogs_match(self, name1: str, name2: str) -> bool:
        """Check if two dog names match (handles variations in naming)"""
        if not name1 or not name2:
            return False
        
        # Clean and normalize names
        clean1 = self.clean_dog_name(name1).lower()
        clean2 = self.clean_dog_name(name2).lower()
        
        # Exact match
        if clean1 == clean2:
            return True
        
        # Remove common prefixes/suffixes and try again
        prefixes = ['1. ', '2. ', '3. ', '4. ', '5. ', '6. ', '7. ', '8. ']
        for prefix in prefixes:
            clean1 = clean1.replace(prefix, '')
            clean2 = clean2.replace(prefix, '')
        
        # Check if one name is contained in the other
        if clean1 in clean2 or clean2 in clean1:
            return True
        
        # Word-by-word matching for compound names
        words1 = [w for w in clean1.split() if len(w) > 1]
        words2 = [w for w in clean2.split() if len(w) > 1]
        
        if len(words1) > 0 and len(words2) > 0:
            # Check if significant portion of words match
            matches = sum(1 for w in words1 if w in words2)
            match_ratio = matches / max(len(words1), len(words2))
            return match_ratio > 0.6  # 60% threshold
        
        return clean1 == clean2
    
    def calculate_performance_rating(self, row) -> float:
        """Calculate performance rating based on recent form"""
        # Simplified performance rating calculation
        try:
            time_str = str(row.get('TIME', ''))
            if time_str and time_str.replace('.', '').isdigit():
                time_val = float(time_str)
                # Simple rating: lower time = higher rating
                return max(0, 100 - (time_val - 20) * 5)
        except:
            pass
        return 50.0  # Default rating
    
    def calculate_speed_rating(self, row) -> float:
        """Calculate speed rating"""
        try:
            time_str = str(row.get('TIME', ''))
            win_time_str = str(row.get('WIN', ''))
            
            if time_str and win_time_str and time_str.replace('.', '').isdigit() and win_time_str.replace('.', '').isdigit():
                time_val = float(time_str)
                win_time_val = float(win_time_str)
                
                if win_time_val > 0:
                    # Speed rating relative to winner
                    return max(0, 100 - ((time_val - win_time_val) * 10))
        except:
            pass
        return 50.0
    
    def calculate_class_rating(self, row) -> float:
        """Calculate class rating based on grade"""
        try:
            grade_str = str(row.get('G', '')).upper()
            if 'MAIDEN' in grade_str:
                return 30.0
            elif 'GRADE 7' in grade_str:
                return 40.0
            elif 'GRADE 6' in grade_str:
                return 50.0
            elif 'GRADE 5' in grade_str:
                return 60.0
            elif 'GRADE 4' in grade_str:
                return 70.0
            elif 'GRADE 3' in grade_str:
                return 80.0
            elif 'GRADE 2' in grade_str:
                return 90.0
            elif 'GRADE 1' in grade_str:
                return 100.0
        except:
            pass
        return 50.0
    
    def calculate_probabilities(self, row, race_info: Dict[str, Any]) -> Dict[str, float]:
        """Calculate win and place probabilities"""
        try:
            starting_price = self.safe_float(row.get('SP', 0))
            if starting_price > 0:
                # Convert odds to probability
                win_probability = 1 / starting_price
                place_probability = win_probability * 3  # Simplified place probability
                
                return {
                    'win_probability': min(win_probability, 1.0),
                    'place_probability': min(place_probability, 1.0)
                }
        except:
            pass
        
        return {'win_probability': 0.1, 'place_probability': 0.3}
    
    def generate_race_analytics(self, race_info: Dict[str, Any], dogs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate advanced race analytics"""
        analytics = {
            'race_id': race_info['race_id'],
            'analysis_timestamp': datetime.now(),
            'field_size': len(dogs),
            'analysis_type': 'comprehensive'
        }
        
        if dogs:
            # Sort dogs by win probability
            dogs_sorted = sorted(dogs, key=lambda x: x.get('win_probability', 0), reverse=True)
            
            # Predicted winner
            predicted_winner = dogs_sorted[0]
            analytics['predicted_winner'] = predicted_winner['dog_clean_name']
            analytics['predicted_winner_probability'] = predicted_winner.get('win_probability', 0)
            
            # Top 3 selections
            analytics['top_3_selections'] = [
                {
                    'dog_name': dog['dog_clean_name'],
                    'box': dog['box_number'],
                    'win_probability': dog.get('win_probability', 0),
                    'odds': dog.get('starting_price', 0)
                }
                for dog in dogs_sorted[:3]
            ]
            
            # Race competitiveness
            probabilities = [dog.get('win_probability', 0) for dog in dogs]
            analytics['race_competitiveness'] = np.std(probabilities) if probabilities else 0
            
            # Average class rating
            class_ratings = [dog.get('class_rating', 50) for dog in dogs]
            analytics['average_class_rating'] = np.mean(class_ratings) if class_ratings else 50
        
        return analytics
    
    def determine_winner_from_form_guide(self, race_info: Dict[str, Any], dogs: List[Dict[str, Any]]):
        """Determine winner from form guide data when scraping fails"""
        try:
            # Find the dog with the best (lowest) finish position
            valid_dogs = [dog for dog in dogs if dog.get('finish_position') not in ['N/A', '', None]]
            
            if not valid_dogs:
                return
            
            # Convert positions to integers for comparison
            for dog in valid_dogs:
                try:
                    pos = dog.get('finish_position')
                    if isinstance(pos, str) and pos.isdigit():
                        dog['_numeric_position'] = int(pos)
                    else:
                        dog['_numeric_position'] = 999  # High number for invalid positions
                except:
                    dog['_numeric_position'] = 999
            
            # Sort by position and get the winner
            valid_dogs.sort(key=lambda x: (x.get('_numeric_position', 999), x.get('dog_name', '')) ) 
            
            if valid_dogs:
                winner = valid_dogs[0]
                race_info['winner_name'] = winner.get('dog_clean_name', winner.get('dog_name', ''))
                race_info['data_quality_note'] = 'Winner determined from form guide data (scraping failed)'
                print(f"   ✅ Winner determined from form guide: {race_info['winner_name']}")
                
                # Clean up temporary field
                for dog in valid_dogs:
                    if '_numeric_position' in dog:
                        del dog['_numeric_position']
                        
        except Exception as e:
            print(f"   ⚠️ Error determining winner from form guide: {e}")
    
    def validate_race_data(self, race_info: Dict[str, Any], dogs: List[Dict[str, Any]]) -> bool:
        """Validate race data to prevent corrupted entries"""
        try:
            # Validate field size (must be reasonable for greyhound racing)
            num_dogs = len(dogs)
            if num_dogs > 12:
                print(f"   ❌ Data validation failed: Too many dogs ({num_dogs}) - maximum is 12")
                return False
            
            if num_dogs == 0:
                print(f"   ❌ Data validation failed: No dogs in race")
                return False
            
            # Validate race ID format (no special characters that could indicate corruption)
            race_id = race_info.get('race_id', '')
            if not race_id or race_id in ['', 'UNK_0_UNKNOWN', 'UNKNOWN']:
                print(f"   ❌ Data validation failed: Invalid race ID: {race_id}")
                return False
            
            # Check for duplicate dog names (indicates data corruption)
            dog_names = [dog.get('dog_name', '').strip() for dog in dogs if dog.get('dog_name', '').strip()]
            unique_dog_names = set(dog_names)
            if len(dog_names) != len(unique_dog_names):
                print(f"   ❌ Data validation failed: Duplicate dog names detected (corruption indicator)")
                return False
            
            # Validate finish positions are reasonable
            valid_positions = []
            for dog in dogs:
                pos = dog.get('finish_position', '')
                if pos and str(pos).isdigit():
                    pos_num = int(pos)
                    if pos_num <= 0 or pos_num > 12:
                        print(f"   ❌ Data validation failed: Invalid finish position {pos_num} for {dog.get('dog_name', 'unknown')}")
                        return False
                    valid_positions.append(pos_num)
            
            # Check for unreasonable number of unique dogs (corruption indicator)
            if len(unique_dog_names) > 12:
                print(f"   ❌ Data validation failed: Too many unique dogs ({len(unique_dog_names)}) indicates data corruption")
                return False
            
            # Validate venue is not 'Unknown' (indicates corrupted data)
            venue = race_info.get('venue', '')
            if venue in ['Unknown', 'UNKNOWN', 'UNK', '']:
                print(f"   ❌ Data validation failed: Invalid venue: {venue}")
                return False
            
            # Set proper field size based on actual dog count
            race_info['field_size'] = min(num_dogs, 12)
            race_info['actual_field_size'] = num_dogs
            
            print(f"   ✅ Data validation passed: {num_dogs} dogs, venue: {venue}, race_id: {race_id}")
            return True
            
        except Exception as e:
            print(f"   ❌ Data validation error: {str(e)}")
            return False
    
    def save_to_database(self, race_info: Dict[str, Any], dogs: List[Dict[str, Any]]):
        """Save enhanced data to database with validation"""
        # Validate data before saving
        if not self.validate_race_data(race_info, dogs):
            print(f"   🚫 Skipping race {race_info.get('race_id', 'unknown')} - failed validation")
            return False
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Save race metadata with all enhanced fields including box analysis and weather data
            cursor.execute('''
                INSERT OR REPLACE INTO race_metadata 
                (race_id, venue, race_number, race_date, race_name, grade, distance, 
                 track_condition, weather, temperature, humidity, wind_speed, wind_direction, 
                 field_size, actual_field_size, scratched_count, scratch_rate, box_analysis, 
                 extraction_timestamp, data_source, winner_name, winner_odds, winner_margin, 
                 url, data_quality_note, weather_condition, precipitation, pressure, 
                 visibility, weather_location, weather_timestamp, weather_adjustment_factor)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                race_info['race_id'],
                race_info['venue'],
                race_info['race_number'],
                race_info['race_date'],
                race_info.get('race_name', ''),
                race_info.get('grade', ''),
                race_info.get('distance', ''),
                race_info.get('track_condition', ''),
                race_info.get('weather', ''),
                race_info.get('temperature', ''),
                race_info.get('humidity', ''),
                race_info.get('wind_speed', ''),
                race_info.get('wind_direction', ''),
                race_info.get('field_size'),
                race_info.get('actual_field_size'),
                race_info.get('scratched_count', 0),
                race_info.get('scratch_rate', 0.0),
                race_info.get('box_analysis', '{}'),
                race_info.get('extraction_timestamp', datetime.now()),
                'enhanced_processor_with_scraping',
                race_info.get('winner_name', ''),
                race_info.get('winner_odds', ''),
                race_info.get('winner_margin', ''),
                race_info.get('url', ''),
                race_info.get('data_quality_note', ''),
                race_info.get('weather_condition', ''),
                race_info.get('precipitation', ''),
                race_info.get('pressure', ''),
                race_info.get('visibility', ''),
                race_info.get('weather_location', ''),
                race_info.get('weather_timestamp', ''),
                race_info.get('weather_adjustment_factor', '')
            ))
            
            # Save dog data (matching existing schema) - filter out N/A finish positions
            valid_dogs = []
            for dog in dogs:
                finish_position = dog.get('finish_position', '')
                # Skip dogs with N/A, empty, or null finish positions
                if not finish_position or finish_position in ['N/A', 'n/a', 'NA', '']:
                    print(f"   ⚠️ Skipping {dog.get('dog_clean_name', 'unknown')} - No valid finish position ({finish_position})")
                    continue
                valid_dogs.append(dog)
            
            for dog in valid_dogs:
                cursor.execute('''
                    INSERT OR REPLACE INTO dog_race_data (
                     race_id, dog_name, dog_clean_name, box_number, finish_position, 
                     trainer_name, weight, starting_price, individual_time, margin, 
                     sectional_1st, sectional_2nd, sectional_3rd, beaten_margin, 
                     form_guide_json, historical_records, extraction_timestamp, data_source,
                     scraped_trainer_name, scraped_reaction_time, scraped_nbtt, 
                     scraped_race_classification, scraped_raw_result, scraped_finish_position, 
                     recent_form, best_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    dog['race_id'],
                    dog['dog_name'],
                    dog['dog_clean_name'],
                    dog['box_number'],
                    dog['finish_position'],
                    dog.get('trainer_name', ''),
                    dog['weight'],
                    dog['starting_price'],
                    dog['individual_time'],
                    dog['margin'],
                    dog['sectional_1st'],
                    dog.get('sectional_2nd', ''),
                    dog.get('sectional_3rd', ''),
                    dog.get('beaten_margin', 0.0),
                    dog.get('form_guide_json', ''),
                    dog.get('historical_records', ''),
                    dog['extraction_timestamp'],
                    dog['data_source'],
                    dog.get('scraped_trainer_name', ''),
                    dog.get('scraped_reaction_time', ''),
                    dog.get('scraped_nbtt', ''),
                    dog.get('scraped_race_classification', ''),
                    dog.get('scraped_raw_result', ''),
                    dog.get('scraped_finish_position', ''),
                    dog.get('recent_form', ''),
                    dog.get('best_time', 0.0)
                ))
            
            conn.commit()
            print(f"✅ Saved {len(valid_dogs)} valid dogs to database for race {race_info['race_id']} (filtered out {len(dogs) - len(valid_dogs)} with N/A positions)")
            
        except Exception as e:
            print(f"❌ Database error: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def move_to_processed(self, csv_file_path: str, status: str = 'success'):
        """Move processed file to processed directory with status-based subfolder"""
        try:
            filename = os.path.basename(csv_file_path)
            
            # Create status-based subfolder
            if status == 'excluded':
                status_dir = os.path.join(self.processed_dir, 'excluded')
            elif status == 'success':
                status_dir = os.path.join(self.processed_dir, 'completed')
            else:
                status_dir = os.path.join(self.processed_dir, 'other')
            
            # Create directory if it doesn't exist
            os.makedirs(status_dir, exist_ok=True)
            
            processed_path = os.path.join(status_dir, filename)
            
            if not os.path.exists(processed_path):
                import shutil
                shutil.move(csv_file_path, processed_path)
                print(f"📁 Moved {filename} to processed/{status} directory")
        except Exception as e:
            print(f"⚠️ Error moving file: {e}")
    
    def check_if_already_processed(self, filename: str) -> bool:
        """Check if file has already been processed by looking in database"""
        try:
            # Extract race info from filename to generate race_id
            file_path = os.path.join(self.unprocessed_dir, filename)
            # Read CSV file to extract race info including grade and distance
            df = pd.read_csv(file_path)
            race_info = self.extract_race_info_from_filename_and_csv(file_path, df)
            if not race_info:
                return False
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if race exists in database
            cursor.execute(
                "SELECT COUNT(*) FROM race_metadata WHERE race_id = ?",
                (race_info['race_id'],)
            )
            count = cursor.fetchone()[0]
            conn.close()
            
            return count > 0
        except Exception as e:
            print(f"   ⚠️ Error checking if processed: {e}")
            return False
    
    def process_all_unprocessed(self) -> Dict[str, Any]:
        """Process all unprocessed CSV files (skip already processed ones)"""
        if not os.path.exists(self.unprocessed_dir):
            return {'status': 'error', 'message': 'Unprocessed directory not found'}
        
        csv_files = [f for f in os.listdir(self.unprocessed_dir) if f.endswith('.csv')]
        
        if not csv_files:
            return {'status': 'success', 'message': 'No unprocessed files found', 'processed_count': 0}
        
        results = {
            'status': 'success',
            'processed_count': 0,
            'failed_count': 0,
            'skipped_count': 0,
            'results': []
        }
        
        print(f"\n📊 Found {len(csv_files)} CSV files to check...")
        
        for filename in csv_files:
            # Check if we should stop processing
            with processing_lock:
                if not processing_status['running']:
                    print("\n🛑 Processing stopped by user")
                    return {'status': 'stopped', 'message': 'Processing was stopped by user', 'processed_count': results['processed_count'], 'failed_count': results['failed_count'], 'skipped_count': results['skipped_count'], 'results': results['results']}
            
            file_path = os.path.join(self.unprocessed_dir, filename)
            print(f"\n🔍 Checking: {filename}")
            
            # Check if this file has already been processed
            if self.check_if_already_processed(filename):
                print(f"   ⏭️ Skipping {filename} - already processed")
                # Move already processed file to avoid future iteration
                self.move_to_processed(file_path, status='already_processed')
                results['skipped_count'] += 1
                results['results'].append({
                    'filename': filename,
                    'result': {'status': 'skipped', 'reason': 'Already processed'}
                })
                continue
            
            print(f"   🔄 Processing: {filename}")
            result = self.process_csv_file(file_path)
            if result and result.get('status') == 'success':
                results['processed_count'] += 1
            else:
                results['failed_count'] += 1
            
            results['results'].append({
                'filename': filename,
                'result': result
            })
        
        print(f"\n📈 Processing Summary:")
        print(f"   ✅ Processed: {results['processed_count']}")
        print(f"   ❌ Failed: {results['failed_count']}")
        print(f"   ⏭️ Skipped: {results['skipped_count']}")
        
        return results
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive analysis report"""

        print("Initializing database...")
        conn = sqlite3.connect(self.db_path)
        try:
            # Get recent races
            recent_races = pd.read_sql_query('''
                SELECT * FROM race_metadata 
                WHERE race_date >= date('now', '-30 days')
                ORDER BY race_date DESC, race_number DESC
            ''', conn)
            
            # Get recent dog data
            recent_dogs = pd.read_sql_query('''
                SELECT * FROM dog_race_data 
                WHERE race_id IN (
                    SELECT race_id FROM race_metadata
                    WHERE race_date >= date('now', '-30 days')
                )
            ''', conn)
            
            # Generate report
            report = f"""
# Enhanced Greyhound Racing Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Recent Activity Summary
- **Recent Races**: {len(recent_races)} races in the last 30 days
- **Total Entries**: {len(recent_dogs)} dog entries processed
- **Venues**: {', '.join(recent_races['venue'].unique()) if len(recent_races) > 0 else 'None'}

## Top Performing Dogs (by win probability)
"""
            
            if len(recent_dogs) > 0:
                # Convert win_probability to numeric, handling NaN and string values
                recent_dogs['win_probability_numeric'] = pd.to_numeric(recent_dogs['win_probability'], errors='coerce')
                
                # Filter out rows with NaN win_probability and get top dogs
                dogs_with_prob = recent_dogs[recent_dogs['win_probability_numeric'].notna()]
                
                if len(dogs_with_prob) > 0:
                    top_dogs = dogs_with_prob.nlargest(10, 'win_probability_numeric')
                    for idx, dog in top_dogs.iterrows():
                        prob_value = dog['win_probability_numeric']
                        if prob_value > 0:
                            report += f"- **{dog['dog_clean_name']}** (Box {dog['box_number']}) - {prob_value:.2%} win probability\n"
                else:
                    report += "- No dogs with valid win probability data found\n"
            
            report += f"""

## Venue Performance Analysis
"""
            
            if len(recent_races) > 0:
                venue_stats = recent_races.groupby('venue').agg({
                    'race_id': 'count'
                }).round(0)
                
                for venue, stats in venue_stats.iterrows():
                    report += f"- **{venue}**: {int(stats['race_id'])} races\n"
            
            # Save report
            report_path = os.path.join(self.results_dir, f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
            with open(report_path, 'w') as f:
                f.write(report)
            
            return report_path
            
        except Exception as e:
            print(f"❌ Report generation error: {e}")
            return None
        finally:
            conn.close()
    
    def cleanup(self):
        """Cleanup resources"""
        if self.driver:
            self.driver.quit()
    
    # Helper methods
    def safe_int(self, value, default=0):
        """Safely convert to int"""
        try:
            return int(float(str(value).strip())) if value else default
        except:
            return default
    
    def safe_float(self, value, default=0.0):
        """Safely convert to float"""
        try:
            return float(str(value).strip()) if value else default
        except:
            return default
    
    def clean_dog_name(self, name: str) -> str:
        """Clean dog name for consistent identification"""
        if not name:
            return ""
        
        # Remove quotes, numbers, and extra spaces
        cleaned = re.sub(r'^["\d\.\s]+', '', str(name))
        cleaned = re.sub(r'["\s]+$', '', cleaned)
        return cleaned.strip().upper()
    
    def collect_weather_data_for_race(self, race_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Collect weather data for a specific race using OpenMeteo service"""
        if not self.weather_service:
            return None
            
        try:
            venue_code = race_info['venue']
            race_date = race_info['race_date']
            
            # Convert date to string format if needed
            if hasattr(race_date, 'strftime'):
                date_str = race_date.strftime('%Y-%m-%d')
            else:
                date_str = str(race_date)
            
            print(f"     🌤️ Fetching weather for {venue_code} on {date_str}...")
            
            # Get weather data from OpenMeteo service
            weather_data = self.weather_service.get_weather_for_race(venue_code, date_str)
            
            if weather_data:
                # Convert WeatherData object to dictionary for database storage
                weather_dict = {
                    'weather_condition': weather_data.condition.value,
                    'temperature': weather_data.temperature,
                    'humidity': weather_data.humidity,
                    'wind_speed': weather_data.wind_speed,
                    'wind_direction': weather_data.wind_direction,
                    'pressure': weather_data.pressure,
                    'precipitation': weather_data.precipitation,
                    'visibility': weather_data.visibility,
                    'weather_location': weather_data.location,
                    'weather_timestamp': weather_data.timestamp
                }
                
                # Calculate weather adjustment factor
                adjustment_factor = self.weather_service.calculate_weather_adjustment_factor(
                    weather_data, venue_code
                )
                weather_dict['weather_adjustment_factor'] = adjustment_factor
                
                print(f"     ✅ Weather collected: {weather_data.condition.value}, {weather_data.temperature:.1f}°C")
                print(f"     🎯 Weather adjustment factor: {adjustment_factor:.3f}")
                
                return weather_dict
            else:
                print(f"     ⚠️ No weather data available for {venue_code} on {date_str}")
                return None
                
        except Exception as e:
            print(f"     ❌ Error collecting weather data: {e}")
            return None


def main():
    """Main function to run the enhanced processor"""
    print("🚀 ENHANCED COMPREHENSIVE GREYHOUND RACING PROCESSOR")
    print("=" * 70)
    
    # Initialize processor
    processor = EnhancedComprehensiveProcessor()
    
    try:
        # Process all unprocessed files
        results = processor.process_all_unprocessed()
        
        print(f"\n📊 PROCESSING COMPLETE")
        print("=" * 70)
        print(f"✅ Successfully processed: {results.get('processed_count', 0)} files")
        print(f"❌ Failed to process: {results.get('failed_count', 0)} files")
        
        if results.get('processed_count', 0) > 0:
            # Generate comprehensive report
            print(f"\n📋 Generating comprehensive report...")
            report_path = processor.generate_comprehensive_report()
            if report_path:
                print(f"✅ Report saved to: {report_path}")
        
        print(f"\n💡 Enhanced processing complete!")
        print("Advanced features included:")
        print("- ✅ Weather data collection")
        print("- ✅ Track condition analysis")
        print("- ✅ Performance rating calculations")
        print("- ✅ Win/place probability predictions")
        print("- ✅ Advanced database schema")
        print("- ✅ Comprehensive analytics")
        
    except Exception as e:
        print(f"❌ Processing error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        processor.cleanup()


if __name__ == "__main__":
    main()
