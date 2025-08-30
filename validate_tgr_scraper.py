#!/usr/bin/env python3
"""
Comprehensive validation script for TGR scraper implementation.
This script validates all aspects of the fixed scraper.
"""

import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

try:
    from collectors.the_greyhound_recorder_scraper import TheGreyhoundRecorderScraper
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    def validate_race_data_structure(race_data: dict) -> bool:
        """Validate that race data has the expected structure."""
        required_fields = ['url', 'date', 'venue', 'race_number', 'field_size', 'dogs']
        
        for field in required_fields:
            if field not in race_data:
                logger.error(f"❌ Missing required field: {field}")
                return False
        
        # Validate dogs structure
        if not isinstance(race_data['dogs'], list):
            logger.error("❌ Dogs field is not a list")
            return False
        
        for i, dog in enumerate(race_data['dogs']):
            if not isinstance(dog, dict):
                logger.error(f"❌ Dog {i} is not a dictionary")
                return False
            
            dog_required_fields = ['dog_name', 'racing_history', 'total_races']
            for field in dog_required_fields:
                if field not in dog:
                    logger.error(f"❌ Dog {i} missing required field: {field}")
                    return False
            
            # Validate racing history
            if not isinstance(dog['racing_history'], list):
                logger.error(f"❌ Dog {i} racing_history is not a list")
                return False
            
            for j, race in enumerate(dog['racing_history']):
                race_required_fields = ['dog_name', 'race_date', 'finish_position', 'track']
                for field in race_required_fields:
                    if field not in race:
                        logger.error(f"❌ Dog {i} race {j} missing field: {field}")
                        return False
        
        logger.info("✅ Race data structure validation passed")
        return True
    
    def validate_dog_entries_structure(dog_entries: list) -> bool:
        """Validate that dog entries have the expected structure."""
        if not isinstance(dog_entries, list):
            logger.error("❌ Dog entries is not a list")
            return False
        
        for i, entry in enumerate(dog_entries):
            if not isinstance(entry, dict):
                logger.error(f"❌ Dog entry {i} is not a dictionary")
                return False
            
            required_fields = ['dog_name', 'racing_history', 'race_date', 'venue', 'race_url']
            for field in required_fields:
                if field not in entry:
                    logger.error(f"❌ Dog entry {i} missing field: {field}")
                    return False
        
        logger.info("✅ Dog entries structure validation passed")
        return True
    
    def validate_racing_history_data(racing_history: list, dog_name: str) -> bool:
        """Validate the quality and completeness of racing history data."""
        if not racing_history:
            logger.warning(f"⚠️ No racing history for {dog_name}")
            return True
        
        valid_races = 0
        total_races = len(racing_history)
        
        for i, race in enumerate(racing_history):
            # Check essential fields
            if race.get('race_date') and race.get('finish_position') and race.get('track'):
                valid_races += 1
            
            # Validate finish position is reasonable
            if race.get('finish_position') and not (1 <= race['finish_position'] <= 8):
                logger.warning(f"⚠️ {dog_name} race {i}: unusual finish position {race['finish_position']}")
            
            # Validate distance is reasonable
            if race.get('distance') and isinstance(race['distance'], str):
                if not any(char.isdigit() for char in race['distance']):
                    logger.warning(f"⚠️ {dog_name} race {i}: distance has no numbers: {race['distance']}")
        
        completeness = (valid_races / total_races) * 100 if total_races > 0 else 0
        logger.info(f"✅ {dog_name}: {valid_races}/{total_races} races valid ({completeness:.1f}%)")
        
        return completeness >= 80  # At least 80% of races should have essential data
    
    def test_cached_files():
        """Test against all available cached files."""
        cache_dir = Path('.tgr_cache')
        
        if not cache_dir.exists():
            logger.error("❌ No cache directory found")
            return False
        
        cached_files = list(cache_dir.glob('*.html'))
        
        if not cached_files:
            logger.error("❌ No cached HTML files found")
            return False
        
        logger.info(f"📁 Found {len(cached_files)} cached files")
        
        scraper = TheGreyhoundRecorderScraper(rate_limit=0.1, use_cache=True)
        
        test_urls = [
            '/form-guides/murray-bridge/long-form/244836/1/',
            '/form-guides/ballarat/long-form/244839/1/',  # If it exists
            '/form-guides/sandown-park/long-form/244840/1/',  # If it exists
        ]
        
        successful_tests = 0
        total_dogs_tested = 0
        
        for test_url in test_urls:
            logger.info(f"\\n🧪 Testing URL: {test_url}")
            
            try:
                race_data = scraper._fetch_race_details(test_url)
                
                if not race_data:
                    logger.warning(f"⚠️ No data returned for {test_url}")
                    continue
                
                if not validate_race_data_structure(race_data):
                    logger.error(f"❌ Structure validation failed for {test_url}")
                    continue
                
                logger.info(f"✅ Race data: {len(race_data.get('dogs', []))} dogs, venue: {race_data.get('venue')}")
                
                # Test each dog
                for dog in race_data.get('dogs', []):
                    dog_name = dog['dog_name']
                    
                    # Test dog entries extraction
                    dog_entries = scraper._extract_dog_entries(race_data, dog_name)
                    
                    if not validate_dog_entries_structure(dog_entries):
                        logger.error(f"❌ Dog entries validation failed for {dog_name}")
                        continue
                    
                    # Validate racing history quality
                    if dog_entries:
                        racing_history = dog_entries[0].get('racing_history', [])
                        if validate_racing_history_data(racing_history, dog_name):
                            total_dogs_tested += 1
                
                successful_tests += 1
                
            except Exception as e:
                logger.error(f"❌ Error testing {test_url}: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info(f"\\n📊 Test Summary:")
        logger.info(f"   • URLs tested successfully: {successful_tests}/{len(test_urls)}")
        logger.info(f"   • Dogs validated: {total_dogs_tested}")
        
        return successful_tests > 0 and total_dogs_tested > 0
    
    def test_performance_metrics():
        """Test performance metrics calculation."""
        logger.info("\\n🏁 Testing performance metrics calculation...")
        
        scraper = TheGreyhoundRecorderScraper(rate_limit=0.1, use_cache=True)
        
        # Get sample data
        test_url = '/form-guides/murray-bridge/long-form/244836/1/'
        race_data = scraper._fetch_race_details(test_url)
        
        if not race_data or not race_data.get('dogs'):
            logger.error("❌ No test data available for performance metrics")
            return False
        
        sample_dog = race_data['dogs'][0]
        dog_entries = scraper._extract_dog_entries(race_data, sample_dog['dog_name'])
        
        if not dog_entries:
            logger.error("❌ No dog entries for performance metrics test")
            return False
        
        # Test performance metrics
        metrics = scraper._calculate_performance_metrics(dog_entries)
        
        expected_metric_fields = [
            'total_starts', 'wins', 'places', 'win_percentage', 
            'place_percentage', 'average_position', 'best_position'
        ]
        
        for field in expected_metric_fields:
            if field not in metrics:
                logger.error(f"❌ Missing performance metric: {field}")
                return False
        
        logger.info(f"✅ Performance metrics calculated:")
        logger.info(f"   • Total starts: {metrics['total_starts']}")
        logger.info(f"   • Win percentage: {metrics['win_percentage']:.1f}%")
        logger.info(f"   • Place percentage: {metrics['place_percentage']:.1f}%")
        logger.info(f"   • Average position: {metrics['average_position']:.2f}")
        
        return True
    
    def test_venue_and_distance_analysis():
        """Test venue and distance analysis."""
        logger.info("\\n📍 Testing venue and distance analysis...")
        
        scraper = TheGreyhoundRecorderScraper(rate_limit=0.1, use_cache=True)
        
        # Get sample data
        test_url = '/form-guides/murray-bridge/long-form/244836/1/'
        race_data = scraper._fetch_race_details(test_url)
        
        if not race_data or not race_data.get('dogs'):
            logger.error("❌ No test data available for analysis")
            return False
        
        sample_dog = race_data['dogs'][0]
        dog_entries = scraper._extract_dog_entries(race_data, sample_dog['dog_name'])
        
        if not dog_entries:
            logger.error("❌ No dog entries for analysis test")
            return False
        
        # Test venue analysis
        venue_analysis = scraper._analyze_venue_performance(dog_entries)
        distance_analysis = scraper._analyze_distance_performance(dog_entries)
        
        logger.info(f"✅ Venue analysis: {len(venue_analysis)} venues")
        logger.info(f"✅ Distance analysis: {len(distance_analysis)} distances")
        
        for venue, stats in list(venue_analysis.items())[:3]:
            logger.info(f"   • {venue}: {stats['starts']} starts")
        
        return True
    
    def run_comprehensive_validation():
        """Run all validation tests."""
        logger.info("🚀 Starting comprehensive TGR scraper validation...")
        
        tests = [
            ("Cached Files Test", test_cached_files),
            ("Performance Metrics Test", test_performance_metrics),
            ("Venue & Distance Analysis Test", test_venue_and_distance_analysis),
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\\n{'='*60}")
            logger.info(f"Running: {test_name}")
            logger.info(f"{'='*60}")
            
            try:
                if test_func():
                    logger.info(f"✅ {test_name} PASSED")
                    passed_tests += 1
                else:
                    logger.error(f"❌ {test_name} FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name} FAILED with exception: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info(f"\\n{'='*60}")
        logger.info(f"VALIDATION SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Tests passed: {passed_tests}/{total_tests}")
        logger.info(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED! TGR scraper is working correctly.")
            return True
        else:
            logger.error("❌ Some tests failed. Please review the errors above.")
            return False
    
    if __name__ == "__main__":
        success = run_comprehensive_validation()
        sys.exit(0 if success else 1)

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\\nRequired dependencies: requests, beautifulsoup4")
    print("Install with: pip install requests beautifulsoup4")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
