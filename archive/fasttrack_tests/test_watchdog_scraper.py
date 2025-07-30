#!/usr/bin/env python3
"""
Test script for FastTrack Watchdog form guides scraping.
This tests the new /watchdog functionality to discover upcoming Victorian race meetings.
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

def test_watchdog_scraper():
    """Test the FastTrack Watchdog form guides scraping."""
    
    scraper = FastTrackScraper(rate_limit=2.0, use_cache=True)
    
    print("=" * 70)
    print("FastTrack Watchdog Form Guides Test")
    print("=" * 70)
    
    # Test the Watchdog form guides page
    print("\n🐕 Testing Watchdog form guides fetching...")
    try:
        watchdog_data = scraper.fetch_watchdog_form_guides()
        
        if watchdog_data and 'meetings' in watchdog_data:
            meetings = watchdog_data['meetings']
            print(f"✅ Successfully fetched Watchdog data")
            print(f"   Found {len(meetings)} upcoming meetings")
            
            # Show the first few meetings
            for i, meeting in enumerate(meetings[:5]):
                print(f"   {i+1}. Meeting {meeting['meeting_id']}: {meeting['meeting_name']}")
                
            if len(meetings) > 5:
                print(f"   ... and {len(meetings) - 5} more meetings")
                
            # Test drilling down into a specific meeting
            if meetings:
                first_meeting = meetings[0]
                meeting_id = first_meeting['meeting_id']
                
                print(f"\n🏁 Testing drill-down for meeting {meeting_id}...")
                try:
                    meeting_data = scraper.fetch_race_meeting(meeting_id)
                    
                    if meeting_data and 'races' in meeting_data:
                        races = meeting_data['races']
                        print(f"✅ Successfully fetched meeting details")
                        print(f"   Found {len(races)} races in this meeting")
                        
                        for i, race in enumerate(races[:3]):
                            print(f"   {i+1}. Race {race['race_id']}: {race['race_name']}")
                            
                        if len(races) > 3:
                            print(f"   ... and {len(races) - 3} more races")
                    else:
                        print("⚠️  No race data found for this meeting")
                        
                except Exception as e:
                    print(f"❌ Error fetching meeting details: {e}")
                    
        else:
            print("⚠️  No meetings found in Watchdog data")
            print(f"   Raw data keys: {list(watchdog_data.keys()) if watchdog_data else 'None'}")
            
    except Exception as e:
        print(f"❌ Error fetching Watchdog data: {e}")
    
    print("\n" + "=" * 70)
    print("Watchdog test completed.")
    print("=" * 70)

if __name__ == "__main__":
    test_watchdog_scraper()
