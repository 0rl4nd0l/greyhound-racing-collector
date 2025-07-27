#!/usr/bin/env python3
"""
Database Fix Script
==================

This script cleans up the database by:
1. Removing entries with 'nan' or invalid dog names
2. Fixing data quality issues
3. Updating statistics

Author: AI Assistant
Date: July 11, 2025
"""

import sqlite3
import os
from datetime import datetime

DATABASE_PATH = './databases/comprehensive_greyhound_data.db'

def fix_database():
    """Fix database data quality issues"""
    print("🔧 FIXING DATABASE DATA QUALITY")
    print("=" * 50)
    
    if not os.path.exists(DATABASE_PATH):
        print(f"❌ Database not found at {DATABASE_PATH}")
        return
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        # Check current state
        cursor.execute("SELECT COUNT(*) FROM dog_race_data")
        total_before = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM dog_race_data WHERE dog_name = 'nan' OR dog_name IS NULL OR dog_name = ''")
        invalid_count = cursor.fetchone()[0]
        
        print(f"📊 Current database state:")
        print(f"   Total dog entries: {total_before}")
        print(f"   Invalid dog names: {invalid_count}")
        
        # Delete entries with invalid dog names
        print("\n🧹 Cleaning up invalid entries...")
        cursor.execute("""
            DELETE FROM dog_race_data 
            WHERE dog_name = 'nan' 
            OR dog_name IS NULL 
            OR dog_name = ''
            OR TRIM(dog_name) = ''
        """)
        
        deleted_count = cursor.rowcount
        print(f"   Deleted {deleted_count} invalid entries")
        
        # Update field_size in race_metadata to match actual dog count
        print("\n🔄 Updating race field sizes...")
        cursor.execute("""
            UPDATE race_metadata 
            SET field_size = (
                SELECT COUNT(*) 
                FROM dog_race_data 
                WHERE dog_race_data.race_id = race_metadata.race_id
            )
        """)
        
        updated_races = cursor.rowcount
        print(f"   Updated field sizes for {updated_races} races")
        
        # Check final state
        cursor.execute("SELECT COUNT(*) FROM dog_race_data")
        total_after = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM race_metadata")
        total_races = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT dog_name) FROM dog_race_data")
        unique_dogs = cursor.fetchone()[0]
        
        print(f"\n📊 Final database state:")
        print(f"   Total dog entries: {total_after}")
        print(f"   Total races: {total_races}")
        print(f"   Unique dogs: {unique_dogs}")
        print(f"   Entries cleaned: {total_before - total_after}")
        
        # Show some sample data
        print(f"\n🔍 Sample recent races:")
        cursor.execute("""
            SELECT race_id, venue, race_date, race_number, field_size
            FROM race_metadata 
            WHERE race_date LIKE '2025-07-13%'
            ORDER BY race_date DESC, race_number DESC
            LIMIT 5
        """)
        
        for row in cursor.fetchall():
            race_id, venue, race_date, race_number, field_size = row
            print(f"   {race_id} - {venue} Race {race_number} ({field_size} dogs)")
            
            # Show some dogs for this race
            cursor.execute("""
                SELECT dog_name, box_number, finish_position
                FROM dog_race_data 
                WHERE race_id = ? 
                ORDER BY box_number
                LIMIT 3
            """, (race_id,))
            
            dogs = cursor.fetchall()
            for dog in dogs:
                dog_name, box_number, finish_position = dog
                print(f"      Box {box_number}: {dog_name} (finished {finish_position})")
        
        conn.commit()
        print(f"\n✅ Database cleanup completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during database cleanup: {e}")
        conn.rollback()
    finally:
        conn.close()

def main():
    """Main function"""
    fix_database()

if __name__ == "__main__":
    main()
