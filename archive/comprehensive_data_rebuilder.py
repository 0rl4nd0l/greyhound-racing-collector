#!/usr/bin/env python3
"""
Comprehensive Data Rebuilder and Cleaner

This script completely rebuilds the database from scratch using only clean CSV files,
addressing the severe data quality issues found in the analysis.
"""

import os
import pandas as pd
import sqlite3
from datetime import datetime
import logging
from pathlib import Path
import shutil
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveDataRebuilder:
    def __init__(self, processed_dir, database_path):
        self.processed_dir = processed_dir
        self.database_path = database_path
        self.stats = {
            'files_processed': 0,
            'files_skipped': 0,
            'races_created': 0,
            'dogs_inserted': 0,
            'errors': []
        }
        
    def backup_database(self):
        """Create backup of current database"""
        backup_path = f"{self.database_path}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        if os.path.exists(self.database_path):
            shutil.copy2(self.database_path, backup_path)
            logger.info(f"Database backed up to: {backup_path}")
        return backup_path
    
    def create_clean_database(self):
        """Create a fresh database with proper schema"""
        logger.info("Creating fresh database with clean schema...")
        
        # Remove old database if it exists
        if os.path.exists(self.database_path):
            os.remove(self.database_path)
        
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Create race_metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS race_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT UNIQUE NOT NULL,
                venue TEXT NOT NULL,
                race_number INTEGER NOT NULL,
                race_date DATE NOT NULL,
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
                extraction_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                data_source TEXT DEFAULT 'csv_rebuild',
                winner_name TEXT,
                winner_odds REAL,
                winner_margin REAL,
                race_status TEXT DEFAULT 'completed'
            )
        ''')
        
        # Create dog_race_data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dog_race_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT NOT NULL,
                dog_name TEXT NOT NULL,
                dog_clean_name TEXT,
                dog_id INTEGER,
                box_number INTEGER NOT NULL,
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
                extraction_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                data_source TEXT DEFAULT 'csv_rebuild',
                form_guide_json TEXT,
                performance_rating REAL,
                speed_rating REAL,
                class_rating REAL,
                recent_form TEXT,
                win_probability REAL,
                place_probability REAL,
                FOREIGN KEY (race_id) REFERENCES race_metadata (race_id),
                UNIQUE(race_id, box_number)
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_race_date ON race_metadata(race_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_venue ON race_metadata(venue)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_dog_race_id ON dog_race_data(race_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_dog_name ON dog_race_data(dog_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_box_number ON dog_race_data(box_number)')
        
        conn.commit()
        conn.close()
        logger.info("✅ Clean database schema created")
    
    def extract_race_info_from_filename(self, filename):
        """Extract race information from filename"""
        # Expected format: "Race X - VENUE - DATE.csv"
        pattern = r'Race\s+(\d+)\s+-\s+([A-Z_]+)\s+-\s+(\d{1,2}\s+\w+\s+\d{4})\.csv'
        match = re.match(pattern, filename)
        
        if match:
            race_number = int(match.group(1))
            venue = match.group(2)
            date_str = match.group(3)
            
            try:
                # Parse date
                race_date = datetime.strptime(date_str, '%d %B %Y').date()
                race_id = f"{venue.lower()}_{race_date.strftime('%Y-%m-%d')}_{race_number}"
                
                return {
                    'race_id': race_id,
                    'venue': venue,
                    'race_number': race_number,
                    'race_date': race_date.strftime('%Y-%m-%d'),
                    'race_name': f"Race {race_number}"
                }
            except ValueError as e:
                logger.error(f"Could not parse date from {filename}: {e}")
                return None
        
        return None
    
    def clean_dog_name(self, dog_name):
        """Clean and standardize dog names"""
        if pd.isna(dog_name) or str(dog_name).strip() in ['', 'nan', 'NaN']:
            return None
            
        # Remove leading numbers and dots (e.g., "1. Dog Name" -> "Dog Name")
        clean_name = re.sub(r'^\d+\.\s*', '', str(dog_name).strip())
        
        # Remove quotes
        clean_name = clean_name.replace('"', '')
        
        # Convert to title case
        clean_name = clean_name.title()
        
        return clean_name if clean_name else None
    
    def process_csv_file(self, file_path):
        """Process a single CSV file and extract clean data"""
        filename = os.path.basename(file_path)
        logger.info(f"Processing {filename}")
        
        # Extract race info from filename
        race_info = self.extract_race_info_from_filename(filename)
        if not race_info:
            logger.warning(f"Could not extract race info from {filename}")
            self.stats['files_skipped'] += 1
            return None
            
        try:
            # Read CSV file
            try:
                df = pd.read_csv(file_path, sep=',', header=0)
            except pd.errors.ParserError:
                self.stats['errors'].append(f"{filename}: ParserError")
                self.stats['files_skipped'] += 1
                logger.warning(f"Skipping {filename}: Parser error")
                return None
            except Exception as e:
                self.stats['errors'].append(f"{filename}: {str(e)}")
                self.stats['files_skipped'] += 1
                logger.warning(f"Skipping {filename}: {str(e)}")
                return None
            
            # Clean the data
            df['Dog Name'] = df['Dog Name'].apply(self.clean_dog_name)
            
            # Remove rows with missing essential data
            df = df.dropna(subset=['Dog Name', 'BOX'])
            
            # Remove rows where dog name is still invalid
            df = df[df['Dog Name'].notna()]
            df = df[df['Dog Name'] != '']
            
            # Ensure box numbers are integers
            df['BOX'] = pd.to_numeric(df['BOX'], errors='coerce')
            df = df.dropna(subset=['BOX'])
            df['BOX'] = df['BOX'].astype(int)
            
            # Remove duplicate boxes (keep first occurrence)
            df = df.drop_duplicates(subset=['BOX'], keep='first')
            
            # Check for reasonable field size
            if len(df) < 4 or len(df) > 12:
                logger.warning(f"Unusual field size in {filename}: {len(df)} dogs")
                if len(df) > 12:
                    # Keep only first 12 dogs
                    df = df.head(12)
                elif len(df) < 4:
                    # Skip files with too few dogs
                    logger.warning(f"Skipping {filename} - too few dogs ({len(df)})")
                    self.stats['files_skipped'] += 1
                    return None
            
            # Convert finish positions to integers where possible
            if 'PLC' in df.columns:
                df['PLC'] = pd.to_numeric(df['PLC'], errors='coerce')
            
            # Clean other numeric columns
            numeric_cols = ['WGT', 'SP', 'MGN']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Add race information
            df['race_id'] = race_info['race_id']
            df['venue'] = race_info['venue']
            df['race_date'] = race_info['race_date']
            df['race_number'] = race_info['race_number']
            
            self.stats['files_processed'] += 1
            return df, race_info
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            self.stats['errors'].append(f"{filename}: {str(e)}")
            self.stats['files_skipped'] += 1
            return None
    
    def rebuild_database(self):
        """Rebuild the entire database from clean CSV files"""
        logger.info("Starting database rebuild from CSV files...")
        
        # Get all CSV files
        csv_files = [f for f in os.listdir(self.processed_dir) if f.endswith('.csv')]
        logger.info(f"Found {len(csv_files)} CSV files to process")
        
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        processed_races = set()
        
        for filename in csv_files:
            file_path = os.path.join(self.processed_dir, filename)
            
            result = self.process_csv_file(file_path)
            if not result:
                continue
                
            df, race_info = result
            
            # Skip if we've already processed this race
            race_id = race_info['race_id']
            if race_id in processed_races:
                logger.warning(f"Duplicate race {race_id} - skipping")
                continue
                
            processed_races.add(race_id)
            
            # Insert race metadata
            cursor.execute('''
                INSERT OR REPLACE INTO race_metadata (
                    race_id, venue, race_number, race_date, race_name, 
                    field_size, data_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                race_id,
                race_info['venue'],
                race_info['race_number'],
                race_info['race_date'],
                race_info['race_name'],
                len(df),
                'csv_rebuild'
            ))
            
            self.stats['races_created'] += 1
            
            # Insert dog data
            for _, row in df.iterrows():
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO dog_race_data (
                            race_id, dog_name, dog_clean_name, box_number,
                            finish_position, weight, starting_price, margin,
                            individual_time, data_source
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        race_id,
                        row['Dog Name'],
                    self.clean_dog_name(row['Dog Name']),
                        int(row['BOX']),
                        str(row.get('PLC', '')) if pd.notna(row.get('PLC')) else None,
                        float(row['WGT']) if pd.notna(row.get('WGT')) else None,
                        float(row['SP']) if pd.notna(row.get('SP')) else None,
                        str(row.get('MGN', '')) if pd.notna(row.get('MGN')) else None,
                        str(row.get('TIME', '')) if pd.notna(row.get('TIME')) else None,
                        'csv_rebuild'
                    ))
                    
                    self.stats['dogs_inserted'] += 1
                    
                except Exception as e:
                    logger.error(f"Error inserting dog data for {race_id}: {e}")
                    self.stats['errors'].append(f"{race_id} dog insert: {str(e)}")
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Database rebuild completed")
    
    def validate_rebuilt_database(self):
        """Validate the rebuilt database"""
        logger.info("Validating rebuilt database...")
        
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Check for duplicate box assignments
        cursor.execute('''
            SELECT race_id, box_number, COUNT(*) as count
            FROM dog_race_data
            GROUP BY race_id, box_number
            HAVING COUNT(*) > 1
        ''')
        
        duplicates = cursor.fetchall()
        if duplicates:
            logger.warning(f"Found {len(duplicates)} duplicate box assignments")
            for race_id, box_num, count in duplicates:
                logger.warning(f"  Race {race_id}, Box {box_num}: {count} dogs")
        else:
            logger.info("✅ No duplicate box assignments found")
        
        # Check race field sizes
        cursor.execute('''
            SELECT race_id, COUNT(*) as dogs
            FROM dog_race_data
            GROUP BY race_id
            HAVING COUNT(*) > 12 OR COUNT(*) < 4
        ''')
        
        unusual_races = cursor.fetchall()
        if unusual_races:
            logger.warning(f"Found {len(unusual_races)} races with unusual field sizes")
            for race_id, dogs in unusual_races:
                logger.warning(f"  Race {race_id}: {dogs} dogs")
        else:
            logger.info("✅ All races have reasonable field sizes")
        
        conn.close()
    
    def get_final_stats(self):
        """Get final statistics"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM race_metadata')
        total_races = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM dog_race_data')
        total_dogs = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT dog_name) FROM dog_race_data')
        unique_dogs = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT venue) FROM race_metadata')
        venues = cursor.fetchone()[0]
        
        cursor.execute('SELECT MIN(race_date), MAX(race_date) FROM race_metadata')
        date_range = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_races': total_races,
            'total_dogs': total_dogs,
            'unique_dogs': unique_dogs,
            'venues': venues,
            'date_range': f"{date_range[0]} to {date_range[1]}"
        }
    
    def run_comprehensive_rebuild(self):
        """Run the complete rebuild process"""
        logger.info("🚀 Starting comprehensive database rebuild")
        
        # Backup existing database
        backup_path = self.backup_database()
        
        # Create clean database
        self.create_clean_database()
        
        # Rebuild from CSV files
        self.rebuild_database()
        
        # Validate results
        self.validate_rebuilt_database()
        
        # Get final statistics
        final_stats = self.get_final_stats()
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("📊 REBUILD SUMMARY")
        logger.info("="*60)
        logger.info(f"Files processed: {self.stats['files_processed']}")
        logger.info(f"Files skipped: {self.stats['files_skipped']}")
        logger.info(f"Races created: {self.stats['races_created']}")
        logger.info(f"Dogs inserted: {self.stats['dogs_inserted']}")
        logger.info(f"Errors: {len(self.stats['errors'])}")
        
        if self.stats['errors']:
            logger.warning("First 5 errors:")
            for error in self.stats['errors'][:5]:
                logger.warning(f"  {error}")
        
        logger.info("\n📈 FINAL DATABASE STATS:")
        logger.info(f"Total races: {final_stats['total_races']}")
        logger.info(f"Total dog entries: {final_stats['total_dogs']}")
        logger.info(f"Unique dogs: {final_stats['unique_dogs']}")
        logger.info(f"Venues: {final_stats['venues']}")
        logger.info(f"Date range: {final_stats['date_range']}")
        
        if backup_path:
            logger.info(f"\n💾 Database backup: {backup_path}")
        logger.info("✅ Comprehensive rebuild completed successfully!")
        
        return final_stats

def main():
    """Main function"""
    processed_dir = '/Users/orlandolee/greyhound_racing_collector/form_guides/processed'
    database_path = '/Users/orlandolee/greyhound_racing_collector/databases/comprehensive_greyhound_data.db'
    
    rebuilder = ComprehensiveDataRebuilder(processed_dir, database_path)
    final_stats = rebuilder.run_comprehensive_rebuild()
    
    print("\n🎉 Database rebuild completed!")
    print("The database has been completely rebuilt from clean CSV data.")

if __name__ == "__main__":
    main()
