#!/usr/bin/env python3
"""
Comprehensive Database Fix Script
================================

This script:
1. Removes all duplicate entries
2. Rebuilds the database from processed files
3. Fixes data quality issues
4. Provides proper race counts

Author: AI Assistant
Date: July 11, 2025
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
import re

class ComprehensiveDatabaseFix:
    def __init__(self):
        self.db_path = './databases/comprehensive_greyhound_data.db'
        self.processed_dir = Path('./form_guides/processed')
        self.backup_db_path = f'./databases/backup_comprehensive_greyhound_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db'
        
    def backup_database(self):
        """Create backup of current database"""
        print("💾 Creating database backup...")
        try:
            import shutil
            shutil.copy2(self.db_path, self.backup_db_path)
            print(f"✅ Database backed up to: {self.backup_db_path}")
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return False
        return True
    
    def clear_database(self):
        """Clear all data from database tables"""
        print("🧹 Clearing database...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Clear all data but keep table structure
            cursor.execute("DELETE FROM dog_race_data")
            cursor.execute("DELETE FROM race_metadata")
            cursor.execute("DELETE FROM race_analytics")
            
            conn.commit()
            print("✅ Database cleared successfully")
        except Exception as e:
            print(f"❌ Error clearing database: {e}")
            conn.rollback()
        finally:
            conn.close()
    
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
            print(f"Error parsing filename {filename}: {e}")
        
        return None
    
    def clean_dog_name(self, name):
        """Clean dog name for consistency"""
        if not name or pd.isna(name):
            return ""
        
        name_str = str(name).strip()
        
        # Skip if it's obviously not a dog name
        if (name_str.lower() in ['nan', 'null', ''] or 
            name_str.isdigit() or 
            len(name_str) < 2):
            return ""
        
        # Remove leading numbers and quotes
        cleaned = re.sub(r'^["\d\.\s]+', '', name_str)
        cleaned = re.sub(r'["\s]+$', '', cleaned)
        
        return cleaned.strip().upper()
    
    def process_csv_file(self, file_path, race_info):
        """Process a single CSV file and extract race data"""
        try:
            df = pd.read_csv(file_path)
            
            if df.empty:
                print(f"   ⚠️  Empty file: {file_path.name}")
                return None, []
            
            # Extract dogs data
            dogs = []
            processed_dogs = set()  # Track processed dogs to avoid duplicates
            
            for _, row in df.iterrows():
                # Get dog name - try multiple possible column names
                dog_name = self.clean_dog_name(row.get('Dog Name', '') or row.get('DOG', ''))
                
                if not dog_name:
                    continue
                
                # Get box number
                box_number = row.get('BOX', '')
                try:
                    box_number = int(box_number) if box_number else None
                except:
                    box_number = None
                
                # Create unique key to prevent duplicates
                dog_key = (dog_name, box_number)
                if dog_key in processed_dogs:
                    continue
                processed_dogs.add(dog_key)
                
                # Extract other data with correct column mappings
                dog_data = {
                    'race_id': race_info['race_id'],
                    'dog_name': dog_name,
                    'dog_clean_name': dog_name,
                    'box_number': box_number,
                    'finish_position': self.safe_get(row, 'PLC', '') or self.safe_get(row, 'FINISH_POSITION', ''),
                    'trainer_name': self.safe_get(row, 'TRAINER', ''),
                    'weight': self.safe_float(row.get('WGT', '') or row.get('WEIGHT', '')),
                    'starting_price': self.safe_float(row.get('SP', '')),
                    'individual_time': self.safe_get(row, 'TIME', ''),
                    'margin': self.safe_get(row, 'MGN', '') or self.safe_get(row, 'MARGIN', ''),
                    'sectional_1st': self.safe_get(row, '1 SEC', '') or self.safe_get(row, 'SECTIONAL_1ST', ''),
                    'extraction_timestamp': datetime.now(),
                    'data_source': 'comprehensive_rebuild'
                }
                
                dogs.append(dog_data)
            
            # Update race info with actual field size
            race_info['field_size'] = len(dogs)
            
            print(f"   ✅ Processed {len(dogs)} dogs from {file_path.name}")
            return race_info, dogs
            
        except Exception as e:
            print(f"   ❌ Error processing {file_path.name}: {e}")
            return None, []
    
    def safe_get(self, row, key, default=''):
        """Safely get value from row"""
        try:
            value = row.get(key, default)
            if pd.isna(value) or value == 'nan':
                return default
            return str(value).strip()
        except:
            return default
    
    def safe_float(self, value, default=None):
        """Safely convert to float"""
        try:
            if pd.isna(value) or value == 'nan' or value == '':
                return default
            return float(value)
        except:
            return default
    
    def save_to_database(self, race_info, dogs):
        """Save race and dog data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Insert race metadata
            cursor.execute('''
                INSERT OR REPLACE INTO race_metadata 
                (race_id, venue, race_number, race_date, field_size, extraction_timestamp, data_source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                race_info['race_id'],
                race_info['venue'],
                race_info['race_number'],
                race_info['race_date'],
                race_info['field_size'],
                datetime.now(),
                'comprehensive_rebuild'
            ))
            
            # Insert dog data
            for dog in dogs:
                cursor.execute('''
                    INSERT OR REPLACE INTO dog_race_data 
                    (race_id, dog_name, dog_clean_name, box_number, finish_position, 
                     trainer_name, weight, starting_price, individual_time, margin, 
                     sectional_1st, extraction_timestamp, data_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    dog['extraction_timestamp'],
                    dog['data_source']
                ))
            
            conn.commit()
            
        except Exception as e:
            print(f"❌ Database error: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def rebuild_database(self):
        """Rebuild database from processed files"""
        print("🔄 Rebuilding database from processed files...")
        
        if not self.processed_dir.exists():
            print(f"❌ Processed directory not found: {self.processed_dir}")
            return
        
        csv_files = list(self.processed_dir.glob('*.csv'))
        
        if not csv_files:
            print("⚠️  No CSV files found in processed directory")
            return
        
        print(f"📁 Found {len(csv_files)} files to process")
        
        processed_count = 0
        failed_count = 0
        total_dogs = 0
        
        for file_path in csv_files:
            print(f"\n🔄 Processing: {file_path.name}")
            
            # Extract race info from filename
            race_info = self.extract_race_info_from_filename(file_path.name)
            
            if not race_info:
                print(f"   ❌ Could not parse race info from filename")
                failed_count += 1
                continue
            
            # Process the file
            race_data, dogs = self.process_csv_file(file_path, race_info)
            
            if race_data and dogs:
                # Save to database
                self.save_to_database(race_data, dogs)
                processed_count += 1
                total_dogs += len(dogs)
            else:
                failed_count += 1
        
        print(f"\n📊 REBUILD SUMMARY:")
        print(f"   ✅ Successfully processed: {processed_count} races")
        print(f"   ❌ Failed: {failed_count} races")
        print(f"   🐕 Total dogs: {total_dogs}")
        
        # Get final stats
        self.show_final_stats()
    
    def show_final_stats(self):
        """Show final database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Count races
            cursor.execute("SELECT COUNT(*) FROM race_metadata")
            total_races = cursor.fetchone()[0]
            
            # Count dogs
            cursor.execute("SELECT COUNT(*) FROM dog_race_data")
            total_dogs = cursor.fetchone()[0]
            
            # Count unique dogs
            cursor.execute("SELECT COUNT(DISTINCT dog_clean_name) FROM dog_race_data WHERE dog_clean_name != ''")
            unique_dogs = cursor.fetchone()[0]
            
            # Count venues
            cursor.execute("SELECT COUNT(DISTINCT venue) FROM race_metadata")
            venues = cursor.fetchone()[0]
            
            # Check for duplicates
            cursor.execute("""
                SELECT race_id, COUNT(*) as count 
                FROM dog_race_data 
                GROUP BY race_id 
                HAVING COUNT(*) > 8
                ORDER BY count DESC
                LIMIT 5
            """)
            potential_duplicates = cursor.fetchall()
            
            print(f"\n📊 FINAL DATABASE STATISTICS:")
            print(f"   Total races: {total_races}")
            print(f"   Total dog entries: {total_dogs}")
            print(f"   Unique dogs: {unique_dogs}")
            print(f"   Venues: {venues}")
            
            if potential_duplicates:
                print(f"\n⚠️  POTENTIAL DUPLICATES FOUND:")
                for race_id, count in potential_duplicates:
                    print(f"   {race_id}: {count} dogs")
            else:
                print(f"\n✅ No duplicate issues found")
                
        except Exception as e:
            print(f"❌ Error getting final stats: {e}")
        finally:
            conn.close()
    
    def run_comprehensive_fix(self):
        """Run the complete database fix process"""
        print("🔧 COMPREHENSIVE DATABASE FIX")
        print("=" * 60)
        
        # Step 1: Backup database
        if not self.backup_database():
            print("❌ Cannot proceed without backup")
            return
        
        # Step 2: Clear database
        self.clear_database()
        
        # Step 3: Rebuild from processed files
        self.rebuild_database()
        
        print(f"\n✅ Comprehensive database fix completed!")
        print(f"💾 Backup saved to: {self.backup_db_path}")

def main():
    """Main function"""
    fixer = ComprehensiveDatabaseFix()
    fixer.run_comprehensive_fix()

if __name__ == "__main__":
    main()
