#!/usr/bin/env python3
"""
Field Size Data Cleanup Script
==============================

Fixes incorrect field_size values in the greyhound racing database.
The field_size should match the actual number of dogs in each race,
with a maximum reasonable limit of 12 dogs per race.

Author: AI Assistant
Date: July 24, 2025
"""

import sqlite3
from datetime import datetime

import pandas as pd

DATABASE_PATH = 'greyhound_racing_data.db'

def analyze_field_size_issues():
    """Analyze and report field size inconsistencies"""
    print("🔍 Analyzing field size inconsistencies...")
    
    conn = sqlite3.connect(DATABASE_PATH)
    
    # Get races with suspicious field sizes
    query = """
    SELECT 
        rm.race_id,
        rm.venue,
        rm.race_date,
        rm.race_name,
        rm.field_size,
        COUNT(DISTINCT CASE WHEN drd.dog_name IS NOT NULL AND drd.dog_name != 'nan' THEN drd.dog_name END) as actual_dog_count,
        MAX(CASE WHEN drd.finish_position IS NOT NULL AND drd.finish_position != 'nan' THEN CAST(drd.finish_position as INTEGER) END) as max_finish_pos
    FROM race_metadata rm
    LEFT JOIN dog_race_data drd ON rm.race_id = drd.race_id
    GROUP BY rm.race_id
    HAVING rm.field_size > 12 OR rm.field_size != actual_dog_count OR rm.field_size IS NULL
    ORDER BY rm.field_size DESC
    """
    
    df = pd.read_sql_query(query, conn)
    
    print(f"📊 Found {len(df)} races with field size issues:")
    print("\nTop 10 problematic races:")
    print(df.head(10)[['race_id', 'venue', 'field_size', 'actual_dog_count', 'max_finish_pos']].to_string(index=False))
    
    # Statistics
    print(f"\n📈 Field size statistics:")
    print(f"   Races with field_size > 12: {len(df[df['field_size'] > 12])}")
    print(f"   Races with field_size = NULL: {len(df[df['field_size'].isna()])}")
    print(f"   Max recorded field_size: {df['field_size'].max()}")
    print(f"   Max actual dog count: {df['actual_dog_count'].max()}")
    
    conn.close()
    return df

def fix_field_sizes(dry_run=True):
    """Fix field size inconsistencies"""
    print(f"\n🔧 {'[DRY RUN] ' if dry_run else ''}Fixing field size inconsistencies...")
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Get all races and calculate correct field sizes
    query = """
    SELECT 
        rm.race_id,
        rm.field_size as current_field_size,
        COUNT(DISTINCT CASE WHEN drd.dog_name IS NOT NULL AND drd.dog_name != 'nan' THEN drd.dog_name END) as actual_dog_count,
        MAX(CASE WHEN drd.finish_position IS NOT NULL AND drd.finish_position != 'nan' THEN CAST(drd.finish_position as INTEGER) END) as max_finish_pos
    FROM race_metadata rm
    LEFT JOIN dog_race_data drd ON rm.race_id = drd.race_id
    GROUP BY rm.race_id
    """
    
    cursor.execute(query)
    races = cursor.fetchall()
    
    updates_needed = 0
    updates_made = 0
    
    for race_id, current_field_size, actual_dog_count, max_finish_pos in races:
        # Determine the correct field size
        # Use the maximum of actual_dog_count and max_finish_pos, but cap at 12
        correct_field_size = min(max(actual_dog_count or 0, max_finish_pos or 0), 12)
        
        # Only update if there's a significant discrepancy or invalid value
        if (current_field_size is None or 
            current_field_size > 12 or 
            abs(current_field_size - correct_field_size) > 1):
            
            updates_needed += 1
            
            if not dry_run:
                update_query = "UPDATE race_metadata SET field_size = ? WHERE race_id = ?"
                cursor.execute(update_query, (correct_field_size, race_id))
                updates_made += 1
                
            print(f"   {race_id}: {current_field_size} → {correct_field_size} (dogs: {actual_dog_count}, max_pos: {max_finish_pos})")
    
    if not dry_run:
        conn.commit()
        print(f"\n✅ Updated {updates_made} races")
    else:
        print(f"\n📋 Would update {updates_needed} races")
    
    conn.close()
    return updates_needed

def validate_fixes():
    """Validate that the fixes worked correctly"""
    print("\n🔍 Validating field size fixes...")
    
    conn = sqlite3.connect(DATABASE_PATH)
    
    # Check for remaining issues
    query = """
    SELECT 
        COUNT(*) as total_races,
        SUM(CASE WHEN field_size > 12 THEN 1 ELSE 0 END) as oversized_fields,
        SUM(CASE WHEN field_size IS NULL THEN 1 ELSE 0 END) as null_fields,
        AVG(field_size) as avg_field_size,
        MAX(field_size) as max_field_size
    FROM race_metadata
    """
    
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchone()
    
    total_races, oversized_fields, null_fields, avg_field_size, max_field_size = result
    
    print(f"📊 Validation results:")
    print(f"   Total races: {total_races}")
    print(f"   Races with field_size > 12: {oversized_fields}")
    print(f"   Races with NULL field_size: {null_fields}")
    print(f"   Average field size: {avg_field_size:.2f}")
    print(f"   Maximum field size: {max_field_size}")
    
    if oversized_fields == 0 and null_fields == 0:
        print("✅ All field sizes are now within reasonable limits!")
    else:
        print("⚠️ Some issues remain")
    
    conn.close()

def main():
    """Main execution function"""
    print("🏁 Greyhound Racing Field Size Cleanup")
    print("=" * 50)
    
    # Analyze current issues
    problematic_races = analyze_field_size_issues()
    
    if len(problematic_races) == 0:
        print("✅ No field size issues found!")
        return
    
    # Ask for confirmation
    print(f"\n❓ Found {len(problematic_races)} races with field size issues.")
    response = input("Do you want to fix these issues? (y/N): ").lower().strip()
    
    if response in ['y', 'yes']:
        # First do a dry run
        print("\n📋 Performing dry run...")
        fix_field_sizes(dry_run=True)
        
        # Confirm actual changes
        confirm = input("\nProceed with actual fixes? (y/N): ").lower().strip()
        if confirm in ['y', 'yes']:
            fix_field_sizes(dry_run=False)
            validate_fixes()
        else:
            print("🚫 Cancelled - no changes made")
    else:
        print("🚫 Cancelled - no changes made")

if __name__ == "__main__":
    main()
