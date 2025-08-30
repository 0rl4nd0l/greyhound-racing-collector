#!/usr/bin/env python3
"""
Test script for the fixed TGR implementation.
This tests the scraper against cached HTML content.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from collectors.the_greyhound_recorder_scraper import TheGreyhoundRecorderScraper
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    def test_tgr_scraper():
        """Test the TGR scraper implementation."""
        
        logger.info("🧪 Testing TGR scraper implementation...")
        
        # Create scraper with caching enabled
        scraper = TheGreyhoundRecorderScraper(rate_limit=1.0, use_cache=True)
        
        # Test URL from our cached content
        test_url = '/form-guides/murray-bridge/long-form/244836/1/'
        
        logger.info(f"📥 Testing with URL: {test_url}")
        
        # Test _fetch_race_details method
        race_data = scraper._fetch_race_details(test_url)
        
        if not race_data:
            logger.error("❌ No race data returned!")
            return False
        
        logger.info("✅ Race data extracted successfully!")
        logger.info(f"📊 Results:")
        logger.info(f"   • Dogs found: {len(race_data.get('dogs', []))}")
        logger.info(f"   • Venue: {race_data.get('venue')}")
        logger.info(f"   • Date: {race_data.get('date')}")
        logger.info(f"   • Race number: {race_data.get('race_number')}")
        logger.info(f"   • Field size: {race_data.get('field_size')}")
        
        if race_data.get('dogs'):
            sample_dog = race_data['dogs'][0]
            logger.info(f"   • Sample dog: {sample_dog['dog_name']}")
            logger.info(f"   • Racing history entries: {sample_dog.get('total_races', 0)}")
            
            if sample_dog.get('racing_history'):
                sample_race = sample_dog['racing_history'][0]
                logger.info(f"   • Sample race entry:")
                for key, value in list(sample_race.items())[:8]:
                    logger.info(f"     - {key}: {value}")
                
                # Test the _extract_dog_entries method
                logger.info(f"\\n🐕 Testing enhanced data extraction for: {sample_dog['dog_name']}")
                
                enhanced_entries = scraper._extract_dog_entries(race_data, sample_dog['dog_name'])
                logger.info(f"   • Enhanced entries found: {len(enhanced_entries)}")
                
                if enhanced_entries:
                    sample_enhanced = enhanced_entries[0]
                    logger.info(f"   • Sample enhanced entry keys: {list(sample_enhanced.keys())}")
                
            return True
        else:
            logger.warning("⚠️ No dogs found in race data")
            return False
    
    if __name__ == "__main__":
        success = test_tgr_scraper()
        if success:
            print("\\n✅ TGR scraper test completed successfully!")
        else:
            print("\\n❌ TGR scraper test failed!")
            sys.exit(1)

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\\nThis likely means required dependencies are not installed.")
    print("The scraper requires: requests, beautifulsoup4")
    print("\\nTo install: pip install requests beautifulsoup4")
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
