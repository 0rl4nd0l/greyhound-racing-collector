#!/usr/bin/env python3
"""
Comprehensive Data Cleanup Script
=================================

This script fixes data quality issues in the greyhound racing database:
1. Removes duplicate dog entries in the same race
2. Fixes races with multiple dogs in the same box
3. Removes invalid or inconsistent data
4. Ensures data integrity

Author: AI Assistant
Date: July 11, 2025
"""

import sqlite3
import pandas as pd
from datetime import datetime, date
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCleanup:
    def __init__(self, db_path='./databases/comprehensive_greyhound_data.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cleanup_stats = {
            'future_races_removed': 0,
            'duplicate_dogs_removed': 0,
            'invalid_entries_removed': 0,
            'races_fixed': 0
        }
    
    def backup_database(self):
        """Create a backup of the database before cleanup"""
        import shutil
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"./databases/backup_before_cleanup_{timestamp}.db"
        shutil.copy2(self.db_path, backup_path)
        logger.info(f"Database backed up to: {backup_path}")
        return backup_path
    
    def remove_future_races(self):
        """Remove races that haven't been run yet"""
        cursor = self.conn.cursor()
        today = date.today().strftime('%Y-%m-%d')
        
        # Count future races
        cursor.execute('SELECT COUNT(*) FROM race_metadata WHERE race_date > ?', (today,))
        future_races = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM dog_race_data WHERE race_id IN (SELECT race_id FROM race_metadata WHERE race_date > ?)', (today,))
        future_dogs = cursor.fetchone()[0]
        
        if future_races > 0:
            logger.info(f"Removing {future_races} future races with {future_dogs} dog entries")
            
            # Remove dog entries for future races
            cursor.execute('''
                DELETE FROM dog_race_data
                WHERE race_id IN (SELECT race_id FROM race_metadata WHERE race_date > ?)
            ''', (today,))
            
            # Remove future race metadata
            cursor.execute('DELETE FROM race_metadata WHERE race_date > ?', (today,))
            
            self.conn.commit()
            self.cleanup_stats['future_races_removed'] = future_races
            logger.info(f"✅ Removed {future_races} future races")
        else:
            logger.info("✅ No future races to remove")
    
    def fix_duplicate_dogs_in_same_race(self):
        """Fix cases where the same dog appears multiple times in the same race"""
        cursor = self.conn.cursor()
        
        # Find races with duplicate dogs
        cursor.execute('''
            SELECT race_id, dog_name, COUNT(*) as count
            FROM dog_race_data
            WHERE dog_name != '' AND dog_name IS NOT NULL
            GROUP BY race_id, dog_name
            HAVING COUNT(*) > 1
        ''')
        
        duplicates = cursor.fetchall()
        
        if duplicates:
            logger.info(f"Found {len(duplicates)} duplicate dog entries to fix")
            
            for race_id, dog_name, count in duplicates:
                logger.info(f"Fixing {dog_name} in {race_id} ({count} duplicates)")
                
                # Get all entries for this dog in this race
                cursor.execute('''
                    SELECT rowid, box_number, finish_position, weight, starting_price, individual_time
                    FROM dog_race_data
                    WHERE race_id = ? AND dog_name = ?
                    ORDER BY rowid
                ''', (race_id, dog_name))
                
                entries = cursor.fetchall()
                
                if len(entries) > 1:
                    # Keep the first entry, remove the rest
                    keep_entry = entries[0]
                    remove_entries = entries[1:]
                    
                    for entry in remove_entries:
                        cursor.execute('DELETE FROM dog_race_data WHERE rowid = ?', (entry[0],))
                    
                    self.cleanup_stats['duplicate_dogs_removed'] += len(remove_entries)
            
            self.conn.commit()
            logger.info(f"✅ Fixed duplicate dogs in races")
        else:
            logger.info("✅ No duplicate dogs found")
    
    def fix_multiple_dogs_per_box(self):
        """Fix cases where multiple dogs are assigned to the same box"""
        cursor = self.conn.cursor()
        
        # Find races with multiple dogs in same box
        cursor.execute('''
            SELECT race_id, box_number, COUNT(*) as count
            FROM dog_race_data
            WHERE dog_name != '' AND dog_name IS NOT NULL
            AND box_number IS NOT NULL
            GROUP BY race_id, box_number
            HAVING COUNT(*) > 1
        ''')
        
        box_conflicts = cursor.fetchall()
        
        if box_conflicts:
            logger.info(f"Found {len(box_conflicts)} box conflicts to fix")
            
            for race_id, box_number, count in box_conflicts:
                logger.info(f"Fixing box {box_number} in {race_id} ({count} dogs)")
                
                # Get dogs in this box
                cursor.execute('''
                    SELECT rowid, dog_name, finish_position, weight, starting_price
                    FROM dog_race_data
                    WHERE race_id = ? AND box_number = ?
                    ORDER BY rowid
                ''', (race_id, box_number))
                
                dogs_in_box = cursor.fetchall()
                
                if len(dogs_in_box) > 1:
                    # Keep the first dog, reassign others or remove if necessary
                    keep_dog = dogs_in_box[0]
                    
                    # Find available box numbers for this race
                    cursor.execute('''
                        SELECT DISTINCT box_number
                        FROM dog_race_data
                        WHERE race_id = ? AND box_number IS NOT NULL
                    ''', (race_id,))
                    
                    used_boxes = set(row[0] for row in cursor.fetchall())
                    
                    # Reassign other dogs to available boxes
                    next_box = 1
                    for i, dog in enumerate(dogs_in_box[1:], 1):
                        # Find next available box
                        while next_box in used_boxes:
                            next_box += 1
                        
                        if next_box <= 20:  # Reasonable box number limit
                            cursor.execute('''
                                UPDATE dog_race_data
                                SET box_number = ?
                                WHERE rowid = ?
                            ''', (next_box, dog[0]))
                            used_boxes.add(next_box)
                            logger.info(f"  Moved {dog[1]} to box {next_box}")
                        else:
                            # Remove if we can't find a reasonable box
                            cursor.execute('DELETE FROM dog_race_data WHERE rowid = ?', (dog[0],))
                            logger.info(f"  Removed {dog[1]} (no available box)")
                            self.cleanup_stats['invalid_entries_removed'] += 1
                        
                        next_box += 1
                    
                    self.cleanup_stats['races_fixed'] += 1
            
            self.conn.commit()
            logger.info(f"✅ Fixed box conflicts")
        else:
            logger.info("✅ No box conflicts found")
    
    def remove_invalid_entries(self):
        """Remove entries with invalid or missing critical data"""
        cursor = self.conn.cursor()
        
        # Count invalid entries before removal
        cursor.execute('''
            SELECT COUNT(*) FROM dog_race_data
            WHERE dog_name IS NULL 
               OR dog_name = ''
               OR dog_name = 'nan'
               OR dog_name = 'NaN'
        ''')
        invalid_dogs = cursor.fetchone()[0]
        
        if invalid_dogs > 0:
            logger.info(f"Removing {invalid_dogs} entries with invalid dog names")
            
            cursor.execute('''
                DELETE FROM dog_race_data
                WHERE dog_name IS NULL 
                   OR dog_name = ''
                   OR dog_name = 'nan'
                   OR dog_name = 'NaN'
            ''')
            
            self.cleanup_stats['invalid_entries_removed'] += invalid_dogs
            self.conn.commit()
            logger.info(f"✅ Removed invalid entries")
        else:
            logger.info("✅ No invalid entries found")
    
    def validate_race_integrity(self):
        """Validate that races have reasonable data"""
        cursor = self.conn.cursor()
        
        # Check for races with no dogs
        cursor.execute('''
            SELECT rm.race_id, rm.venue, rm.race_date
            FROM race_metadata rm
            LEFT JOIN dog_race_data drd ON rm.race_id = drd.race_id
            WHERE drd.race_id IS NULL
        ''')
        
        empty_races = cursor.fetchall()
        
        if empty_races:
            logger.info(f"Found {len(empty_races)} races with no dogs - removing them")
            
            for race_id, venue, race_date in empty_races:
                logger.info(f"Removing empty race: {race_id} ({venue} - {race_date})")
                cursor.execute('DELETE FROM race_metadata WHERE race_id = ?', (race_id,))
            
            self.conn.commit()
            logger.info("✅ Removed empty races")
        else:
            logger.info("✅ No empty races found")
    
    def get_final_stats(self):
        """Get final database statistics"""
        cursor = self.conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM race_metadata')
        total_races = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM dog_race_data')
        total_dogs = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT dog_name) FROM dog_race_data WHERE dog_name != ""')
        unique_dogs = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT venue) FROM race_metadata')
        venues = cursor.fetchone()[0]
        
        cursor.execute('SELECT MIN(race_date), MAX(race_date) FROM race_metadata')
        date_range = cursor.fetchone()
        
        return {
            'total_races': total_races,
            'total_dogs': total_dogs,
            'unique_dogs': unique_dogs,
            'venues': venues,
            'date_range': f"{date_range[0]} to {date_range[1]}"
        }
    
    def run_comprehensive_cleanup(self):
        """Run the complete cleanup process"""
        logger.info("🚀 Starting comprehensive data cleanup")
        
        # Backup database
        backup_path = self.backup_database()
        
        # Run cleanup steps
        logger.info("Step 1: Removing future races")
        self.remove_future_races()
        
        logger.info("Step 2: Fixing duplicate dogs in same race")
        self.fix_duplicate_dogs_in_same_race()
        
        logger.info("Step 3: Fixing multiple dogs per box")
        self.fix_multiple_dogs_per_box()
        
        logger.info("Step 4: Removing invalid entries")
        self.remove_invalid_entries()
        
        logger.info("Step 5: Validating race integrity")
        self.validate_race_integrity()
        
        # Get final statistics
        final_stats = self.get_final_stats()
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("📊 CLEANUP SUMMARY")
        logger.info("="*60)
        logger.info(f"Future races removed: {self.cleanup_stats['future_races_removed']}")
        logger.info(f"Duplicate dogs removed: {self.cleanup_stats['duplicate_dogs_removed']}")
        logger.info(f"Invalid entries removed: {self.cleanup_stats['invalid_entries_removed']}")
        logger.info(f"Races fixed: {self.cleanup_stats['races_fixed']}")
        
        logger.info("\n📈 FINAL DATABASE STATS:")
        logger.info(f"Total races: {final_stats['total_races']}")
        logger.info(f"Total dog entries: {final_stats['total_dogs']}")
        logger.info(f"Unique dogs: {final_stats['unique_dogs']}")
        logger.info(f"Venues: {final_stats['venues']}")
        logger.info(f"Date range: {final_stats['date_range']}")
        
        logger.info(f"\n💾 Database backup: {backup_path}")
        logger.info("✅ Cleanup completed successfully!")
        
        return final_stats
    
    def __del__(self):
        """Clean up database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()

def main():
    """Main function"""
    cleaner = DataCleanup()
    final_stats = cleaner.run_comprehensive_cleanup()
    
    print("\n🎉 Data cleanup completed!")
    print("The database is now clean and ready for analysis.")

if __name__ == "__main__":
    main()
