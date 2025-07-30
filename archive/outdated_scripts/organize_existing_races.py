#!/usr/bin/env python3
"""
Organize Existing Upcoming Races by Time
========================================

This script reads the existing race CSV files in the upcoming_races directory
and organizes them by their actual start times, showing the soonest races first.

Usage: python3 organize_existing_races.py
"""

import os
import re
from datetime import datetime
from pathlib import Path

def extract_race_time_from_filename(filename):
    """Extract race number and estimate time from filename"""
    # Pattern: Race {number} - {venue} - {date}.csv
    pattern = r'Race (\d+) - ([A-Z-]+) - (\d{4}-\d{2}-\d{2})\.csv'
    match = re.match(pattern, filename)
    
    if not match:
        return None, None, None
    
    race_number = int(match.group(1))
    venue = match.group(2)
    date = match.group(3)
    
    # Estimate race time based on race number
    # Most races start around 1 PM and run every 25 minutes
    base_hour = 13  # 1 PM
    base_minute = 0
    
    # Add 25 minutes per race number (typical spacing)
    total_minutes = base_minute + ((race_number - 1) * 25)
    hour = base_hour + (total_minutes // 60)
    minute = total_minutes % 60
    
    # Convert to minutes since midnight for sorting
    time_minutes = hour * 60 + minute
    
    # Convert to 12-hour format for display
    if hour > 12:
        display_time = f'{hour - 12}:{minute:02d} PM'
    elif hour == 12:
        display_time = f'12:{minute:02d} PM'
    else:
        display_time = f'{hour}:{minute:02d} AM'
    
    return time_minutes, display_time, {
        'race_number': race_number,
        'venue': venue,
        'date': date,
        'filename': filename,
        'display_time': display_time,
        'sort_time': time_minutes
    }

def organize_existing_races():
    """Organize existing race files by their estimated start times"""
    
    upcoming_races_dir = Path('./upcoming_races')
    
    if not upcoming_races_dir.exists():
        print("❌ upcoming_races directory not found")
        return
    
    print("🏁 Organizing existing upcoming races by estimated start time...")
    
    races = []
    
    # Read all CSV files in the directory
    for file_path in upcoming_races_dir.glob('*.csv'):
        filename = file_path.name
        
        # Skip README files
        if filename.lower().startswith('readme'):
            continue
        
        time_minutes, display_time, race_info = extract_race_time_from_filename(filename)
        
        if race_info:
            races.append(race_info)
    
    if not races:
        print("❌ No race files found in upcoming_races directory")
        return
    
    # Sort races by estimated start time
    races.sort(key=lambda x: x['sort_time'])
    
    print(f"\n🕐 Today's races organized by estimated start time ({len(races)} races):")
    print("=" * 80)
    
    current_time = None
    race_count = 0
    
    for race in races:
        # Group races by hour for better readability
        race_hour = race['sort_time'] // 60
        if current_time != race_hour:
            if current_time is not None:
                print()  # Add spacing between time groups
            current_time = race_hour
        
        race_count += 1
        venue_display = race['venue'].replace('-', ' ').replace('_', ' ')
        
        print(f"{race_count:2d}. {race['display_time']:>8} - Race {race['race_number']:2d} at {venue_display}")
        print(f"     📁 {race['filename']}")
        
        # Show file path
        file_path = upcoming_races_dir / race['filename']
        if file_path.exists():
            # Try to get file size for additional info
            try:
                file_size = file_path.stat().st_size
                size_kb = file_size / 1024
                print(f"     📊 File size: {size_kb:.1f} KB")
            except:
                pass
        
        print()
    
    print("📝 Note: Times are estimated based on race numbers (Race 1 = 1:00 PM, +25 min per race)")
    print("🌐 For actual race times, check thedogs.com.au or use the enhanced scraper")

def main():
    """Main function"""
    try:
        organize_existing_races()
    except KeyboardInterrupt:
        print("\n\n⚠️ Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
