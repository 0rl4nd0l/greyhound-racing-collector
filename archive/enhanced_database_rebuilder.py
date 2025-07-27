#!/usr/bin/env python3
"""
Enhanced Database Rebuilder
============================

This script rebuilds the database with proper extraction of all enhanced data
from the CSV files, including grades, trainers, and other detailed information.

Author: AI Assistant
Date: July 11, 2025
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, date
import re
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedDatabaseRebuilder:
    def __init__(self, db_path='./databases/greyhound_racing.db'):
        self.db_path = db_path
        self.processed_dir = Path('./form_guides/processed')
        
    def backup_database(self):
        """Create a backup of the current database"""
        import shutil
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"./databases/backup_enhanced_rebuild_{timestamp}.db"
        if os.path.exists(self.db_path):
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backed up to: {backup_path}")
        return backup_path
    
    def extract_race_info_from_filename(self, filename):
        """Extract race information from filename"""
        try:
            # Pattern: "Race X - VENUE - DD Month YYYY.csv"
            pattern = r'Race\s+(\d+)\s+-\s+([A-Za-z_]+)\s+-\s+(\d{1,2})\s+(\w+)\s+(\d{4})\.csv'
            match = re.search(pattern, filename)
            
            if match:
                race_number, venue, day, month, year = match.groups()
                
                # Convert month name to number
                month_map = {
                    'january': 1, 'february': 2, 'march': 3, 'april': 4,
                    'may': 5, 'june': 6, 'july': 7, 'august': 8,
                    'september': 9, 'october': 10, 'november': 11, 'december': 12
                }
                
                month_num = month_map.get(month.lower())
                if month_num:
                    race_date = f"{year}-{month_num:02d}-{int(day):02d}"
                    race_id = f"{venue.lower()}_{race_date}_{race_number}"
                    
                    return {
                        'race_id': race_id,
                        'venue': venue.upper(),
                        'race_number': int(race_number),
                        'race_date': race_date,
                        'filename': filename
                    }
            
        except Exception as e:
            logger.error(f"Error parsing filename {filename}: {e}")
        
        return None
    
    def analyze_csv_structure(self, file_path):
        """Analyze the CSV structure to understand available columns"""
        try:
            df = pd.read_csv(file_path, nrows=5)
            logger.info(f"CSV columns in {file_path.name}: {list(df.columns)}")
            return df.columns.tolist()
        except Exception as e:
            logger.error(f"Error analyzing CSV structure: {e}")
            return []
    
    def extract_enhanced_race_data(self, file_path, race_info):
        """Extract enhanced race and dog data from CSV"""
        try:
            df = pd.read_csv(file_path)
            
            if df.empty:
                logger.warning(f"Empty file: {file_path.name}")
                return None, []
            
            # Analyze first few rows to extract race-level information
            race_data = race_info.copy()
            
            # Extract race details from first valid row
            first_row = df.iloc[0]
            
            # Distance - look for DIST column
            if 'DIST' in df.columns:
                distances = df['DIST'].dropna().unique()
                if len(distances) > 0:
                    race_data['distance'] = f"{distances[0]}m"
            
            # Track - look for TRACK column
            if 'TRACK' in df.columns:
                tracks = df['TRACK'].dropna().unique()
                if len(tracks) > 0:
                    race_data['track_condition'] = tracks[0]
            
            # Grade - look for G column
            if 'G' in df.columns:
                grades = df['G'].dropna().unique()
                if len(grades) > 0:
                    race_data['grade'] = grades[0]
            
            # Race name - construct from venue and grade
            if 'grade' in race_data and 'distance' in race_data:
                race_data['race_name'] = f"{race_data['venue']} R{race_data['race_number']} {race_data['grade']} {race_data['distance']}"
            
            # Extract dog data
            dogs = []
            processed_dogs = set()
            
            for _, row in df.iterrows():
                # Clean dog name
                dog_name = self.clean_dog_name(row.get('Dog Name', ''))
                if not dog_name:
                    continue
                
                # Get box number
                box_number = self.safe_int(row.get('BOX', ''))
                
                # Create unique key to prevent duplicates
                dog_key = (dog_name, box_number)
                if dog_key in processed_dogs:
                    continue
                processed_dogs.add(dog_key)
                
                # Extract comprehensive dog data
                dog_data = {
                    'race_id': race_info['race_id'],
                    'dog_name': dog_name,
                    'dog_clean_name': dog_name,
                    'box_number': box_number,
                    'finish_position': self.safe_get(row, 'PLC', ''),
                    'weight': self.safe_float(row.get('WGT', '')),
                    'starting_price': self.safe_float(row.get('SP', '')),
                    'individual_time': self.safe_get(row, 'TIME', ''),
                    'margin': self.safe_get(row, 'MGN', ''),
                    'sectional_1st': self.safe_get(row, '1 SEC', ''),
                    'sectional_2nd': self.safe_get(row, '2 SEC', ''),
                    'sectional_3rd': self.safe_get(row, '3 SEC', ''),
                    
                    # Enhanced fields
                    'trainer_name': self.safe_get(row, 'WIN', ''),  # WIN column often contains trainer
                    'sex': self.safe_get(row, 'Sex', ''),
                    'distance': self.safe_get(row, 'DIST', ''),
                    'track': self.safe_get(row, 'TRACK', ''),
                    'grade': self.safe_get(row, 'G', ''),
                    'bonus_point': self.safe_get(row, 'BON', ''),
                    'pir': self.safe_get(row, 'PIR', ''),  # Performance Index Rating
                    'beaten_margin': self.safe_float(row.get('MGN', '')),
                    
                    # Form data
                    'form_guide_json': self.extract_form_data(row),
                    
                    # Metadata
                    'extraction_timestamp': datetime.now(),
                    'data_source': 'enhanced_rebuild'
                }
                
                dogs.append(dog_data)
            
            # Update race info with actual field size
            race_data['field_size'] = len(dogs)
            
            logger.info(f"Processed {len(dogs)} dogs from {file_path.name}")
            return race_data, dogs
            
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            return None, []
    
    def extract_form_data(self, row):
        """Extract form guide data as JSON"""
        form_data = {}
        
        # Extract relevant columns for form guide
        form_columns = ['DATE', 'TRACK', 'DIST', 'G', 'TIME', 'PLC', 'MGN', 'SP', 'WIN', 'BON', '1 SEC', 'PIR']
        
        for col in form_columns:
            if col in row and pd.notna(row[col]) and row[col] != '':
                form_data[col] = str(row[col])
        
        return str(form_data) if form_data else None
    
    def clean_dog_name(self, name):
        """Clean dog name for consistency"""
        if not name or pd.isna(name):
            return ""
        
        name_str = str(name).strip()
        
        # Skip if it's obviously not a dog name
        if (name_str.lower() in ['nan', 'null', '""', ''] or 
            name_str.isdigit() or 
            len(name_str) < 2):
            return ""
        
        # Remove leading numbers and quotes
        cleaned = re.sub(r'^["\d\.\s]+', '', name_str)
        cleaned = re.sub(r'["\s]+$', '', cleaned)
        
        # Remove trailing dots that might be race numbering
        cleaned = re.sub(r'^\d+\.\s*', '', cleaned)
        
        return cleaned.strip().upper()
    
    def safe_get(self, row, key, default=''):
        """Safely get value from row"""
        try:
            value = row.get(key, default)
            if pd.isna(value) or value == 'nan' or value == '""':
                return default
            return str(value).strip()
        except:
            return default
    
    def safe_float(self, value, default=None):
        """Safely convert to float"""
        try:
            if pd.isna(value) or value == 'nan' or value == '' or value == '""':
                return default
            return float(value)
        except:
            return default
    
    def safe_int(self, value, default=None):
        """Safely convert to int"""
        try:
            if pd.isna(value) or value == 'nan' or value == '' or value == '""':
                return default
            return int(float(value))
        except:
            return default
    
    def save_enhanced_data(self, race_data, dogs):
        """Save enhanced race and dog data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Insert enhanced race metadata
            cursor.execute('''
                INSERT OR REPLACE INTO race_metadata 
                (race_id, venue, race_number, race_date, race_name, grade, distance, 
                 track_condition, field_size, extraction_timestamp, data_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                race_data['race_id'],
                race_data['venue'],
                race_data['race_number'],
                race_data['race_date'],
                race_data.get('race_name', ''),
                race_data.get('grade', ''),
                race_data.get('distance', ''),
                race_data.get('track_condition', ''),
                race_data['field_size'],
                datetime.now(),
                'enhanced_rebuild'
            ))
            
            # Insert enhanced dog data
            for dog in dogs:
                cursor.execute('''
                    INSERT OR REPLACE INTO dog_race_data 
                    (race_id, dog_name, dog_clean_name, box_number, finish_position,
                     trainer_name, weight, starting_price, individual_time, margin,
                     sectional_1st, sectional_2nd, sectional_3rd, beaten_margin,
                     form_guide_json, extraction_timestamp, data_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    dog['race_id'],
                    dog['dog_name'],
                    dog['dog_clean_name'],
                    dog['box_number'],
                    dog['finish_position'],
                    dog['trainer_name'],
                    dog['weight'],
                    dog['starting_price'],
                    dog['individual_time'],
                    dog['margin'],
                    dog['sectional_1st'],
                    dog['sectional_2nd'],
                    dog['sectional_3rd'],
                    dog['beaten_margin'],
                    dog['form_guide_json'],
                    dog['extraction_timestamp'],
                    dog['data_source']
                ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Database error: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def create_database_schema(self):
        """Create the database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Create race_metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS race_metadata (
                    race_id TEXT PRIMARY KEY,
                    venue TEXT,
                    race_number INTEGER,
                    race_date TEXT,
                    race_name TEXT,
                    grade TEXT,
                    distance TEXT,
                    track_condition TEXT,
                    field_size INTEGER,
                    extraction_timestamp TEXT,
                    data_source TEXT
                )
            ''')
            
            # Create dog_race_data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS dog_race_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    race_id TEXT,
                    dog_name TEXT,
                    dog_clean_name TEXT,
                    box_number INTEGER,
                    finish_position TEXT,
                    trainer_name TEXT,
                    weight REAL,
                    starting_price REAL,
                    individual_time TEXT,
                    margin TEXT,
                    sectional_1st TEXT,
                    sectional_2nd TEXT,
                    sectional_3rd TEXT,
                    beaten_margin REAL,
                    form_guide_json TEXT,
                    extraction_timestamp TEXT,
                    data_source TEXT,
                    FOREIGN KEY (race_id) REFERENCES race_metadata(race_id)
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_race_id ON dog_race_data(race_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_dog_name ON dog_race_data(dog_clean_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_venue_date ON race_metadata(venue, race_date)')
            
            conn.commit()
            logger.info("✅ Database schema created successfully")
            
        except Exception as e:
            logger.error(f"Error creating database schema: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def rebuild_with_enhanced_data(self):
        """Rebuild database with enhanced data extraction"""
        logger.info("🚀 Starting enhanced database rebuild...")
        
        # Backup database
        backup_path = self.backup_database()
        
        # Create database schema
        self.create_database_schema()
        
        if not self.processed_dir.exists():
            logger.error(f"Processed directory not found: {self.processed_dir}")
            return
        
        csv_files = list(self.processed_dir.glob('*.csv'))
        
        if not csv_files:
            logger.warning("No CSV files found in processed directory")
            return
        
        logger.info(f"📁 Found {len(csv_files)} files to process")
        
        processed_count = 0
        failed_count = 0
        total_dogs = 0
        
        # Analyze first file to understand structure
        if csv_files:
            sample_columns = self.analyze_csv_structure(csv_files[0])
        
        for file_path in csv_files:
            logger.info(f"🔄 Processing: {file_path.name}")
            
            # Extract race info from filename
            race_info = self.extract_race_info_from_filename(file_path.name)
            
            if not race_info:
                logger.error(f"Could not parse race info from filename: {file_path.name}")
                failed_count += 1
                continue
            
            # Extract enhanced data
            race_data, dogs = self.extract_enhanced_race_data(file_path, race_info)
            
            if race_data and dogs:
                # Save to database
                self.save_enhanced_data(race_data, dogs)
                processed_count += 1
                total_dogs += len(dogs)
                logger.info(f"✅ Saved {len(dogs)} dogs for race {race_data['race_id']}")
            else:
                failed_count += 1
                logger.error(f"❌ Failed to process {file_path.name}")
        
        logger.info("\n" + "="*60)
        logger.info("📊 ENHANCED REBUILD SUMMARY")
        logger.info("="*60)
        logger.info(f"✅ Successfully processed: {processed_count} races")
        logger.info(f"❌ Failed: {failed_count} races")
        logger.info(f"🐕 Total dogs: {total_dogs}")
        logger.info(f"💾 Backup saved to: {backup_path}")
        
        # Get final statistics
        self.show_enhanced_stats()
    
    def show_enhanced_stats(self):
        """Show enhanced database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get enhanced data completeness
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_races,
                    COUNT(CASE WHEN race_name IS NOT NULL AND race_name != '' THEN 1 END) as has_race_name,
                    COUNT(CASE WHEN grade IS NOT NULL AND grade != '' THEN 1 END) as has_grade,
                    COUNT(CASE WHEN distance IS NOT NULL AND distance != '' THEN 1 END) as has_distance
                FROM race_metadata
            ''')
            
            race_stats = cursor.fetchone()
            
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_dogs,
                    COUNT(CASE WHEN trainer_name IS NOT NULL AND trainer_name != '' THEN 1 END) as has_trainer,
                    COUNT(CASE WHEN weight IS NOT NULL THEN 1 END) as has_weight,
                    COUNT(CASE WHEN form_guide_json IS NOT NULL AND form_guide_json != '' THEN 1 END) as has_form_data
                FROM dog_race_data
            ''')
            
            dog_stats = cursor.fetchone()
            
            logger.info("\n📈 ENHANCED DATA STATISTICS:")
            logger.info(f"  Total races: {race_stats[0]}")
            logger.info(f"  Races with names: {race_stats[1]} ({race_stats[1]/race_stats[0]*100:.1f}%)")
            logger.info(f"  Races with grades: {race_stats[2]} ({race_stats[2]/race_stats[0]*100:.1f}%)")
            logger.info(f"  Races with distances: {race_stats[3]} ({race_stats[3]/race_stats[0]*100:.1f}%)")
            
            logger.info(f"\n🐕 DOG DATA STATISTICS:")
            logger.info(f"  Total dogs: {dog_stats[0]}")
            logger.info(f"  Dogs with trainers: {dog_stats[1]} ({dog_stats[1]/dog_stats[0]*100:.1f}%)")
            logger.info(f"  Dogs with weights: {dog_stats[2]} ({dog_stats[2]/dog_stats[0]*100:.1f}%)")
            logger.info(f"  Dogs with form data: {dog_stats[3]} ({dog_stats[3]/dog_stats[0]*100:.1f}%)")
            
        except Exception as e:
            logger.error(f"Error getting enhanced stats: {e}")
        finally:
            conn.close()

def main():
    """Main function"""
    rebuilder = EnhancedDatabaseRebuilder()
    rebuilder.rebuild_with_enhanced_data()
    
    print("\n🎉 Enhanced database rebuild completed!")
    print("The database now contains properly extracted enhanced data.")

if __name__ == "__main__":
    main()
