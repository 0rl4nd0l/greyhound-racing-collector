#!/usr/bin/env python3
"""
Corrupted Race Data Cleanup Script
==================================

Identifies and cleans up races with corrupted or mixed data,
particularly races that have an unrealistic number of unique dogs.

Author: AI Assistant
Date: July 24, 2025
"""

import sqlite3
from datetime import datetime

import pandas as pd

DATABASE_PATH = 'greyhound_racing_data.db'

def identify_corrupted_races():
    """Identify races with corrupted or mixed data"""
    print("🔍 Identifying corrupted races...")
    
    conn = sqlite3.connect(DATABASE_PATH)
    
    # Find races with an unrealistic number of unique dogs
    query = """
    SELECT 
        rm.race_id,
        rm.venue,
        rm.race_date,
        rm.race_name,
        rm.field_size,
        COUNT(DISTINCT drd.dog_name) as unique_dogs,
        COUNT(*) as total_entries,
        MAX(CAST(drd.finish_position as INTEGER)) as max_position
    FROM race_metadata rm
    LEFT JOIN dog_race_data drd ON rm.race_id = drd.race_id
    WHERE drd.dog_name IS NOT NULL AND drd.dog_name != 'nan'
    GROUP BY rm.race_id
    HAVING unique_dogs > 15 OR total_entries > 50
    ORDER BY unique_dogs DESC
    """
    
    df = pd.read_sql_query(query, conn)
    
    print(f"📊 Found {len(df)} potentially corrupted races:")
    if len(df) > 0:
        print("\nCorrupted races:")
        print(df[['race_id', 'venue', 'unique_dogs', 'total_entries', 'max_position']].to_string(index=False))
    
    conn.close()
    return df

def analyze_corrupted_race(race_id):
    """Analyze a specific corrupted race in detail"""
    print(f"\n🔬 Analyzing race: {race_id}")
    
    conn = sqlite3.connect(DATABASE_PATH)
    
    # Get all dog entries for this race
    query = """
    SELECT dog_name, finish_position, box_number, COUNT(*) as entry_count
    FROM dog_race_data 
    WHERE race_id = ? 
    AND dog_name IS NOT NULL 
    AND dog_name != 'nan'
    GROUP BY dog_name, finish_position, box_number
    ORDER BY finish_position, dog_name
    """
    
    cursor = conn.cursor()
    cursor.execute(query, (race_id,))
    entries = cursor.fetchall()
    
    print(f"📋 Race entries ({len(entries)} total):")
    
    # Group by finish position to see the pattern
    positions = {}
    for dog_name, finish_pos, box_num, count in entries:
        if finish_pos not in positions:
            positions[finish_pos] = []
        positions[finish_pos].append((dog_name, box_num, count))
    
    for pos in sorted(positions.keys(), key=lambda x: int(x) if x and str(x).isdigit() else 999):
        dogs_in_pos = positions[pos]
        print(f"   Position {pos}: {len(dogs_in_pos)} dogs")
        if len(dogs_in_pos) <= 5:  # Show details for reasonable numbers
            for dog, box, count in dogs_in_pos:
                print(f"      {dog} (Box {box}, {count} entries)")
        else:
            print(f"      [Too many dogs to display - showing first 3]")
            for dog, box, count in dogs_in_pos[:3]:
                print(f"      {dog} (Box {box}, {count} entries)")
    
    conn.close()

def clean_corrupted_race(race_id, dry_run=True):
    """Clean a corrupted race by removing duplicate or invalid entries"""
    print(f"\n🧹 {'[DRY RUN] ' if dry_run else ''}Cleaning corrupted race: {race_id}")
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    if race_id == 'UNK_0_UNKNOWN':
        print("   This appears to be a mixed data race - removing entirely")
        if not dry_run:
            # Delete all dog entries for this race
            cursor.execute("DELETE FROM dog_race_data WHERE race_id = ?", (race_id,))
            # Delete the race metadata
            cursor.execute("DELETE FROM race_metadata WHERE race_id = ?", (race_id,))
            deleted_dogs = cursor.rowcount
            print(f"   Deleted race and all associated entries")
        else:
            cursor.execute("SELECT COUNT(*) FROM dog_race_data WHERE race_id = ?", (race_id,))
            count = cursor.fetchone()[0]
            print(f"   Would delete {count} dog entries and the race metadata")
    else:
        # For other corrupted races, we could implement more sophisticated cleaning
        print("   Cleaning logic for this race type not yet implemented")
    
    if not dry_run:
        conn.commit()
    
    conn.close()

def validate_cleanup():
    """Validate that cleanup was effective"""
    print("\n🔍 Validating cleanup results...")
    
    conn = sqlite3.connect(DATABASE_PATH)
    
    # Check for remaining corrupted races
    query = """
    SELECT 
        COUNT(*) as total_races,
        MAX(unique_dogs) as max_unique_dogs,
        AVG(unique_dogs) as avg_unique_dogs
    FROM (
        SELECT 
            rm.race_id,
            COUNT(DISTINCT drd.dog_name) as unique_dogs
        FROM race_metadata rm
        LEFT JOIN dog_race_data drd ON rm.race_id = drd.race_id
        WHERE drd.dog_name IS NOT NULL AND drd.dog_name != 'nan'
        GROUP BY rm.race_id
    )
    """
    
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchone()
    
    total_races, max_unique_dogs, avg_unique_dogs = result
    
    print(f"📊 Validation results:")
    print(f"   Total races: {total_races}")
    print(f"   Maximum unique dogs per race: {max_unique_dogs}")
    print(f"   Average unique dogs per race: {avg_unique_dogs:.2f}")
    
    if max_unique_dogs <= 12:
        print("✅ All races now have reasonable dog counts!")
    else:
        print("⚠️ Some races still have too many dogs")
    
    conn.close()

def main():
    """Main execution function"""
    print("🏁 Corrupted Race Data Cleanup")
    print("=" * 50)
    
    # Identify corrupted races
    corrupted_races = identify_corrupted_races()
    
    if len(corrupted_races) == 0:
        print("✅ No corrupted races found!")
        return
    
    # Analyze the most corrupted race
    if len(corrupted_races) > 0:
        worst_race = corrupted_races.iloc[0]['race_id']
        analyze_corrupted_race(worst_race)
    
    # Ask for cleanup confirmation
    print(f"\n❓ Found {len(corrupted_races)} corrupted races.")
    response = input("Do you want to clean up these races? (y/N): ").lower().strip()
    
    if response in ['y', 'yes']:
        # Dry run first
        print("\n📋 Performing dry run...")
        for _, race in corrupted_races.iterrows():
            clean_corrupted_race(race['race_id'], dry_run=True)
        
        # Confirm actual cleanup
        confirm = input("\nProceed with actual cleanup? (y/N): ").lower().strip()
        if confirm in ['y', 'yes']:
            for _, race in corrupted_races.iterrows():
                clean_corrupted_race(race['race_id'], dry_run=False)
            validate_cleanup()
        else:
            print("🚫 Cancelled - no changes made")
    else:
        print("🚫 Cancelled - no changes made")

if __name__ == "__main__":
    main()
