#!/usr/bin/env python3
"""
Prevent UNKNOWN Venue Files
===========================

This script prevents the creation of race files with UNKNOWN venues.
It should be imported by any script that creates race files.
"""

import os
from pathlib import Path

def validate_race_filename(filename):
    """Validate that a race filename doesn't contain UNKNOWN venues"""
    if 'UNKNOWN' in filename:
        print(f"🚫 BLOCKED: Prevented creation of file with UNKNOWN venue: {filename}")
        print("   → This indicates the venue could not be properly extracted from thedogs.com.au")
        print("   → Check venue mapping in the scraper script")
        return False
    return True

def validate_and_create_file(filepath, content):
    """Safely create a file only if it doesn't have UNKNOWN venue"""
    filename = os.path.basename(filepath)
    
    if not validate_race_filename(filename):
        return False
    
    # Create the file
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Created valid race file: {filename}")
        return True
    except Exception as e:
        print(f"❌ Error creating file {filename}: {e}")
        return False

def cleanup_unknown_files(directory):
    """Remove any existing UNKNOWN files from a directory"""
    if not os.path.exists(directory):
        return 0
    
    removed_count = 0
    files = os.listdir(directory)
    
    for filename in files:
        if 'UNKNOWN' in filename and (filename.endswith('.csv') or filename.endswith('.json')):
            filepath = os.path.join(directory, filename)
            try:
                os.remove(filepath)
                print(f"🗑️ Removed UNKNOWN file: {filename}")
                removed_count += 1
            except Exception as e:
                print(f"❌ Error removing {filename}: {e}")
    
    return removed_count

if __name__ == '__main__':
    print("🛡️ UNKNOWN Venue Prevention System")
    print("="*40)
    
    # Clean up existing UNKNOWN files
    directories = ['./upcoming_races', './predictions', './unified_prediction_cache']
    total_removed = 0
    
    for directory in directories:
        removed = cleanup_unknown_files(directory)
        total_removed += removed
    
    print(f"\n🎯 Cleanup complete: {total_removed} UNKNOWN files removed")
    print("\n💡 To prevent UNKNOWN files in your scripts:")
    print("   from prevent_unknown_venues import validate_race_filename")
    print("   if not validate_race_filename(filename): return")

