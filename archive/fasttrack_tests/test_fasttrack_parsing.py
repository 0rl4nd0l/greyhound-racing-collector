#!/usr/bin/env python3
"""
Test FastTrack parsing logic using the sample HTML files.
This verifies that our parsing functions work correctly.
"""

import logging
import sys
import os
from bs4 import BeautifulSoup

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from collectors.fasttrack_scraper import FastTrackScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_parsing():
    """Test the FastTrack parsing logic using sample files."""
    
    scraper = FastTrackScraper()
    
    print("=" * 60)
    print("FastTrack Parsing Test")
    print("=" * 60)
    
    # Test parsing of race result sample
    print("\n1. Testing race result parsing...")
    race_sample_path = './samples/fasttrack_raw/race_result_1186391057.html'
    
    if os.path.exists(race_sample_path):
        try:
            with open(race_sample_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            race_data = scraper._parse_race(soup)
            
            print(f"✅ Successfully parsed race result sample")
            print(f"   Data keys found: {list(race_data.keys())}")
            
            for key, value in race_data.items():
                if isinstance(value, list):
                    print(f"   {key}: {len(value)} items")
                else:
                    print(f"   {key}: {value}")
                    
        except Exception as e:
            print(f"❌ Error parsing race result: {e}")
    else:
        print(f"⚠️  Race sample file not found: {race_sample_path}")
    
    # Test parsing of race meeting sample
    print("\n2. Testing race meeting parsing...")
    meeting_sample_path = './samples/fasttrack_raw/race_meeting_1163670701.html'
    
    if os.path.exists(meeting_sample_path):
        try:
            with open(meeting_sample_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            meeting_data = scraper._parse_race_meeting(soup)
            
            print(f"✅ Successfully parsed race meeting sample")
            print(f"   Data keys found: {list(meeting_data.keys())}")
            
            if 'races' in meeting_data:
                print(f"   Found {len(meeting_data['races'])} races:")
                for race in meeting_data['races'][:3]:  # Show first 3
                    print(f"     - Race {race['race_id']}: {race['race_name']}")
                    
        except Exception as e:
            print(f"❌ Error parsing race meeting: {e}")
    else:
        print(f"⚠️  Meeting sample file not found: {meeting_sample_path}")
    
    # Test parsing of dog profile sample
    print("\n3. Testing dog profile parsing...")
    dog_sample_path = './samples/fasttrack_raw/dog_890320106.html'
    
    if os.path.exists(dog_sample_path):
        try:
            with open(dog_sample_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            dog_data = scraper._parse_dog(soup)
            
            print(f"✅ Successfully parsed dog profile sample")
            print(f"   Data keys found: {list(dog_data.keys())}")
            
            for key, value in dog_data.items():
                print(f"   {key}: {value}")
                    
        except Exception as e:
            print(f"❌ Error parsing dog profile: {e}")
    else:
        print(f"⚠️  Dog sample file not found: {dog_sample_path}")
    
    print("\n" + "=" * 60)
    print("Parsing test completed.")
    print("=" * 60)

if __name__ == "__main__":
    test_parsing()
