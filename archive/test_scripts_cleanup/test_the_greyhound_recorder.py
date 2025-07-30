#!/usr/bin/env python3
"""
Test script for The Greyhound Recorder scraper
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from collectors.the_greyhound_recorder_scraper import TheGreyhoundRecorderScraper
from collectors.adapters.the_greyhound_recorder_adapter import TheGreyhoundRecorderDBAdapter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_scraper():
    """Test the basic scraper functionality"""
    scraper = TheGreyhoundRecorderScraper(rate_limit=2.0, use_cache=True)
    
    print("Testing The Greyhound Recorder scraper...")
    
    # Test fetching the race calendar
    calendar_data = scraper.fetch_race_calendar()
    
    if calendar_data:
        print(f"✅ Successfully fetched calendar data")
        print(f"Found {len(calendar_data.get('meetings', []))} meetings")
        
        # Print first few meetings
        meetings = calendar_data.get('meetings', [])
        for i, meeting in enumerate(meetings[:5]):
            print(f"  {i+1}. {meeting['meeting_name']} - {meeting['meeting_url']}")
        
        if len(meetings) > 5:
            print(f"  ... and {len(meetings) - 5} more meetings")
    else:
        print("❌ Failed to fetch calendar data")
        
    # Now test the form guides page
    print("\nTesting form guides page...")
    form_guides_data = scraper.fetch_form_guides()
    
    if form_guides_data:
        print(f"✅ Successfully fetched form guides data")
        print(f"Found {len(form_guides_data.get('meetings', []))} meetings with races")
        
        # Print first few meetings
        meetings = form_guides_data.get('meetings', [])
        for i, meeting in enumerate(meetings[:3]):
            print(f"  {i+1}. {meeting.get('meeting_name', 'Unknown')} - {meeting.get('date', 'No date')}")
            races = meeting.get('races', [])
            for j, race in enumerate(races[:3]):
                print(f"    Race {j+1}: {race.get('race_name', 'Unknown')} ({race.get('race_time', 'No time')})")
                
    else:
        print("❌ Failed to fetch form guides data")
        return
        
    # Test database adapter
    print("\n📊 Testing database adapter...")
    try:
        with TheGreyhoundRecorderDBAdapter() as adapter:
            # Test loading a few meetings into the database
            meetings_to_test = form_guides_data.get('meetings', [])[:3]  # Test first 3 meetings
            
            for meeting in meetings_to_test:
                print(f"  Loading meeting: {meeting.get('meeting_title', 'Unknown')}")
                adapter.adapt_and_load_meeting(meeting)
            
            print(f"✅ Successfully loaded {len(meetings_to_test)} meetings to database")
    except Exception as e:
        print(f"❌ Database adapter test failed: {e}")
    
    # Let's also save the raw HTML for inspection
    cache_dir = Path('.tgr_cache')
    if cache_dir.exists():
        cache_files = list(cache_dir.glob('*.html'))
        if cache_files:
            print(f"\n📁 Cached HTML files available in {cache_dir}:")
            for cache_file in cache_files:
                print(f"  - {cache_file.name}")
                
                # Read and show a snippet of the HTML
                with open(cache_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(f"    Content length: {len(content)} chars")
                    
                    # Look for key elements
                    if 'form-guide' in content.lower():
                        print("    ✅ Contains 'form-guide' text")
                    if 'meeting' in content.lower():
                        print("    ✅ Contains 'meeting' text")
                    if 'race' in content.lower():
                        print("    ✅ Contains 'race' text")

if __name__ == "__main__":
    test_scraper()
