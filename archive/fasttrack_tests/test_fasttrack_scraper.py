#!/usr/bin/env python3
"""
Test script for the FastTrack scraper.
This script tests the scraper with known IDs from the sample files.
"""

import logging
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from collectors.fasttrack_scraper import FastTrackScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_scraper():
    """Test the FastTrack scraper with known IDs from documentation."""
    
    # Initialize scraper with caching enabled and short rate limit for testing
    scraper = FastTrackScraper(rate_limit=2.0, use_cache=True)
    
    print("=" * 60)
    print("FastTrack Scraper Test")
    print("=" * 60)
    
    # Test 1: Fetch dog profile
    print("\n1. Testing dog profile fetching...")
    dog_id = 890320106  # From the documentation samples
    try:
        dog_data = scraper.fetch_dog(dog_id)
        if dog_data:
            print(f"✅ Successfully fetched dog {dog_id}")
            print(f"   Data keys: {list(dog_data.keys())}")
            if 'name' in dog_data:
                print(f"   Dog name: {dog_data['name']}")
        else:
            print(f"⚠️  No data returned for dog {dog_id}")
    except Exception as e:
        print(f"❌ Error fetching dog {dog_id}: {e}")
    
    # Test 2: Fetch race meeting
    print("\n2. Testing race meeting fetching...")
    meeting_id = 1163670701  # From the documentation samples
    try:
        meeting_data = scraper.fetch_race_meeting(meeting_id)
        if meeting_data:
            print(f"✅ Successfully fetched meeting {meeting_id}")
            print(f"   Data keys: {list(meeting_data.keys())}")
            if 'races' in meeting_data:
                print(f"   Found {len(meeting_data['races'])} races")
        else:
            print(f"⚠️  No data returned for meeting {meeting_id}")
    except Exception as e:
        print(f"❌ Error fetching meeting {meeting_id}: {e}")
    
    # Test 3: Fetch specific race
    print("\n3. Testing specific race fetching...")
    race_id = 1186391057  # From the documentation samples
    try:
        race_data = scraper.fetch_race(meeting_id, race_id)
        if race_data:
            print(f"✅ Successfully fetched race {race_id}")
            print(f"   Data keys: {list(race_data.keys())}")
            if 'results' in race_data:
                print(f"   Found {len(race_data['results'])} dog entries")
        else:
            print(f"⚠️  No data returned for race {race_id}")
    except Exception as e:
        print(f"❌ Error fetching race {race_id}: {e}")
    
    print("\n" + "=" * 60)
    print("Test completed. Check the .ft_cache directory for cached responses.")
    print("=" * 60)

if __name__ == "__main__":
    test_scraper()
