#!/usr/bin/env python3
"""
Test script for the FastTrack scraper and database adapter.
This tests the end-to-end flow of scraping and loading data.
"""

import logging
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from collectors.fasttrack_scraper import FastTrackScraper
from collectors.adapters.fasttrack_adapter import FastTrackDBAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_scraper_and_adapter():
    """Test the end-to-end flow of scraping and loading FastTrack data."""
    
    scraper = FastTrackScraper(rate_limit=2.0, use_cache=True)
    
    print("=" * 70)
    print("FastTrack Scraper and Adapter Test")
    print("=" * 70)
    
    # Scrape the Watchdog page to get upcoming meetings
    watchdog_data = scraper.fetch_watchdog_form_guides()
    if not watchdog_data or not watchdog_data.get('meetings'):
        print("❌ Could not fetch Watchdog data. Aborting test.")
        return

    # Get the first meeting
    meeting = watchdog_data['meetings'][0]
    meeting_id = meeting['meeting_id']
    
    # Fetch the race meeting details
    meeting_data = scraper.fetch_race_meeting(meeting_id)
    if not meeting_data or not meeting_data.get('races'):
        print(f"❌ Could not fetch race data for meeting {meeting_id}. Aborting.")
        return
        
    # Get the first race
    race = meeting_data['races'][0]
    race_id = race['race_id']
    
    # Fetch the specific race details
    race_data = scraper.fetch_race(meeting_id, race_id)
    if not race_data:
        print(f"❌ Could not fetch specific race details for race {race_id}. Aborting.")
        return

    # Use the adapter to load the data
    print(f"\n💾 Attempting to load data for race: {race_data.get('race_name', race_id)}")
    try:
        with FastTrackDBAdapter() as adapter:
            adapter.adapt_and_load_race(race_data)
        print("✅ Data loaded successfully (check logs for details)")
    except Exception as e:
        print(f"❌ Error loading data: {e}")

    print("\n" + "=" * 70)
    print("Scraper and adapter test completed.")
    print("=" * 70)

if __name__ == "__main__":
    test_scraper_and_adapter()
