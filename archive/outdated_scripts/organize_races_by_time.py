#!/usr/bin/env python3
"""
Organize Upcoming Races by Start Time
====================================

This script scrapes the actual race start times from thedogs.com.au and 
organizes the upcoming races in chronological order, with the soonest races first.

Usage: python3 organize_races_by_time.py
"""

import os
import sys
from datetime import datetime, timedelta
from upcoming_race_browser import UpcomingRaceBrowser

def organize_races_by_time():
    """Fetch and organize upcoming races by their actual start times"""
    
    print("🏁 Organizing upcoming races by start time...")
    
    # Initialize the browser
    browser = UpcomingRaceBrowser()
    
    # Get today's races
    today = datetime.now().date()
    upcoming_races = browser.get_races_for_date(today)
    
    if not upcoming_races:
        print("❌ No races found for today")
        return
    
    # Sort races by actual race time
    def parse_race_time(race):
        race_time_str = race.get('race_time', '')
        if not race_time_str or race_time_str == 'TBA':
            # Use estimated time based on race number if no actual time
            race_number = race.get('race_number', 1)
            estimated_minutes = 13 * 60 + (race_number - 1) * 25  # 1 PM + 25min intervals
            return estimated_minutes
        
        try:
            # Parse time like "7:45 PM" or "13:45"
            if 'AM' in race_time_str.upper() or 'PM' in race_time_str.upper():
                time_obj = datetime.strptime(race_time_str.strip(), '%I:%M %p')
            else:
                # Try 24-hour format
                time_obj = datetime.strptime(race_time_str.strip(), '%H:%M')
            
            # Convert to minutes since midnight for sorting
            return time_obj.hour * 60 + time_obj.minute
            
        except ValueError:
            # Fallback to estimated time
            race_number = race.get('race_number', 1)
            return 13 * 60 + (race_number - 1) * 25
    
    # Sort races by time
    upcoming_races.sort(key=parse_race_time)
    
    print(f"\n🕐 Today's races organized by start time ({len(upcoming_races)} races):")
    print("=" * 70)
    
    for i, race in enumerate(upcoming_races, 1):
        race_time = race.get('race_time', 'TBA')
        venue = race.get('venue_name', race.get('venue', 'Unknown'))
        race_num = race.get('race_number', '?')
        time_source = race.get('time_source', 'unknown')
        
        # Format time source indicator
        source_indicator = {
            'scraped': '🌐',
            'estimated': '📅',
            'unknown': '❓'
        }.get(time_source, '❓')
        
        print(f"{i:2d}. {race_time:>8} {source_indicator} - Race {race_num} at {venue}")
        
        # Show additional details
        distance = race.get('distance')
        grade = race.get('grade')
        if distance or grade:
            details = []
            if distance:
                details.append(f"{distance}m")
            if grade:
                details.append(f"Grade: {grade}")
            print(f"     {' | '.join(details)}")
        
        print(f"     URL: {race.get('url', 'N/A')}")
        print()
    
    print("Legend:")
    print("🌐 = Actual time scraped from website")
    print("📅 = Estimated time based on race number")
    print("❓ = Unknown time source")

def main():
    """Main function"""
    try:
        organize_races_by_time()
    except KeyboardInterrupt:
        print("\n\n⚠️ Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
