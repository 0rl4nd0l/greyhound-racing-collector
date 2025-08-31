#!/usr/bin/env python3
"""
Full Enhanced Data Collection for TGR Scraper
============================================

This script runs the complete enhanced data collection process using all
available cached race files to demonstrate the full capabilities of the
TGR scraper implementation.
"""

import sys
import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup

# Add src to path
sys.path.insert(0, 'src')

try:
    from collectors.the_greyhound_recorder_scraper import TheGreyhoundRecorderScraper
    import logging
    
    # Set up comprehensive logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler('enhanced_collection.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    def discover_race_files():
        """Discover all cached race files and extract their metadata."""
        cache_dir = Path('.tgr_cache')
        cached_files = list(cache_dir.glob('*.html'))
        
        logger.info(f"🔍 Scanning {len(cached_files)} cached files for race data...")
        
        race_files = []
        for cache_file in cached_files:
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Check if this is a race file
                has_form_guide = soup.find(class_='form-guide-meeting__heading') is not None
                dog_sections = soup.find_all(class_='form-guide-long-form-selection')
                
                if has_form_guide and len(dog_sections) > 0:
                    # Extract race metadata
                    heading = soup.find(class_='form-guide-meeting__heading')
                    heading_text = heading.get_text(strip=True) if heading else "Unknown"
                    
                    # Count non-vacant dogs
                    active_dogs = 0
                    for section in dog_sections:
                        if 'form-guide-long-form-selection--vacant' not in section.get('class', []):
                            active_dogs += 1
                    
                    race_info = {
                        'file': cache_file,
                        'heading': heading_text,
                        'dog_count': active_dogs,
                        'total_sections': len(dog_sections)
                    }
                    
                    race_files.append(race_info)
                    
            except Exception as e:
                logger.warning(f"⚠️ Error reading {cache_file.name}: {e}")
        
        logger.info(f"✅ Found {len(race_files)} race files")
        return race_files
    
    def extract_comprehensive_race_data(scraper, race_files):
        """Extract comprehensive data from all race files."""
        logger.info("🚀 Starting comprehensive race data extraction...")
        
        all_race_data = []
        all_dogs_data = {}
        total_dogs = 0
        total_races = 0
        
        for i, race_info in enumerate(race_files):
            logger.info(f"\n📋 Processing race {i+1}/{len(race_files)}: {race_info['heading']}")
            
            try:
                # Create a dummy URL for cache lookup
                dummy_url = f"/race-{i}/"
                
                # Manually load the cached file
                with open(race_info['file'], 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Extract race details using our scraper logic
                race_details = scraper._parse_race_from_soup(soup, dummy_url)
                
                if race_details and race_details.get('dogs'):
                    all_race_data.append(race_details)
                    dogs = race_details['dogs']
                    
                    logger.info(f"   ✅ Extracted {len(dogs)} dogs from {race_details.get('venue')} ({race_details.get('date')})")
                    total_dogs += len(dogs)
                    
                    # Process each dog's enhanced data
                    for dog in dogs:
                        dog_name = dog['dog_name']
                        racing_history = dog.get('racing_history', [])
                        total_races += len(racing_history)
                        
                        if dog_name not in all_dogs_data:
                            all_dogs_data[dog_name] = {
                                'dog_name': dog_name,
                                'appearances': [],
                                'total_history_records': 0,
                                'venues': set(),
                                'grades': set(),
                                'distances': set()
                            }
                        
                        # Add this appearance
                        appearance = {
                            'race_venue': race_details.get('venue'),
                            'race_date': race_details.get('date'), 
                            'race_number': race_details.get('race_number'),
                            'racing_history': racing_history,
                            'history_count': len(racing_history)
                        }
                        
                        all_dogs_data[dog_name]['appearances'].append(appearance)
                        all_dogs_data[dog_name]['total_history_records'] += len(racing_history)
                        
                        # Collect venue, grade, distance data
                        for race in racing_history:
                            if race.get('track'):
                                all_dogs_data[dog_name]['venues'].add(race['track'])
                            if race.get('grade'):
                                all_dogs_data[dog_name]['grades'].add(race['grade'])
                            if race.get('distance'):
                                all_dogs_data[dog_name]['distances'].add(str(race['distance']))
                
                else:
                    logger.warning(f"   ⚠️ No dogs extracted from {race_info['heading']}")
                    
            except Exception as e:
                logger.error(f"   ❌ Error processing {race_info['heading']}: {e}")
                import traceback
                traceback.print_exc()
        
        # Convert sets to lists for JSON serialization
        for dog_data in all_dogs_data.values():
            dog_data['venues'] = list(dog_data['venues'])
            dog_data['grades'] = list(dog_data['grades'])
            dog_data['distances'] = list(dog_data['distances'])
        
        return all_race_data, all_dogs_data, total_dogs, total_races
    
    def calculate_enhanced_analytics(all_dogs_data):
        """Calculate comprehensive analytics across all collected data."""
        logger.info("📊 Calculating enhanced analytics...")
        
        analytics = {
            'summary': {
                'total_unique_dogs': len(all_dogs_data),
                'dogs_with_histories': 0,
                'total_history_records': 0,
                'average_history_per_dog': 0,
                'unique_venues': set(),
                'unique_grades': set(),
                'unique_distances': set()
            },
            'top_dogs': {
                'most_appearances': [],
                'most_history_records': [],
                'most_venues': [],
                'most_versatile': []
            },
            'venue_coverage': {},
            'grade_distribution': {},
            'distance_analysis': {}
        }
        
        # Aggregate data
        for dog_name, dog_data in all_dogs_data.items():
            if dog_data['total_history_records'] > 0:
                analytics['summary']['dogs_with_histories'] += 1
                analytics['summary']['total_history_records'] += dog_data['total_history_records']
            
            # Collect unique values
            analytics['summary']['unique_venues'].update(dog_data['venues'])
            analytics['summary']['unique_grades'].update(dog_data['grades'])
            analytics['summary']['unique_distances'].update(dog_data['distances'])
        
        # Calculate averages
        if analytics['summary']['dogs_with_histories'] > 0:
            analytics['summary']['average_history_per_dog'] = (
                analytics['summary']['total_history_records'] / 
                analytics['summary']['dogs_with_histories']
            )
        
        # Convert sets to lists and counts
        analytics['summary']['unique_venues'] = list(analytics['summary']['unique_venues'])
        analytics['summary']['unique_grades'] = list(analytics['summary']['unique_grades'])
        analytics['summary']['unique_distances'] = list(analytics['summary']['unique_distances'])
        analytics['summary']['venue_count'] = len(analytics['summary']['unique_venues'])
        analytics['summary']['grade_count'] = len(analytics['summary']['unique_grades'])
        analytics['summary']['distance_count'] = len(analytics['summary']['unique_distances'])
        
        # Top dogs analysis
        dogs_by_appearances = sorted(all_dogs_data.items(), 
                                   key=lambda x: len(x[1]['appearances']), reverse=True)
        dogs_by_history = sorted(all_dogs_data.items(), 
                               key=lambda x: x[1]['total_history_records'], reverse=True)
        dogs_by_venues = sorted(all_dogs_data.items(), 
                              key=lambda x: len(x[1]['venues']), reverse=True)
        
        analytics['top_dogs']['most_appearances'] = [
            {'dog_name': name, 'appearances': len(data['appearances'])}
            for name, data in dogs_by_appearances[:10]
        ]
        
        analytics['top_dogs']['most_history_records'] = [
            {'dog_name': name, 'history_records': data['total_history_records']}
            for name, data in dogs_by_history[:10]
        ]
        
        analytics['top_dogs']['most_venues'] = [
            {'dog_name': name, 'venue_count': len(data['venues']), 'venues': data['venues']}
            for name, data in dogs_by_venues[:10] if len(data['venues']) > 0
        ]
        
        return analytics
    
    def save_enhanced_data(all_race_data, all_dogs_data, analytics):
        """Save all enhanced data to files."""
        logger.info("💾 Saving enhanced data to files...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save race data
        race_data_file = f"enhanced_race_data_{timestamp}.json"
        with open(race_data_file, 'w') as f:
            json.dump(all_race_data, f, indent=2, default=str)
        logger.info(f"✅ Race data saved to {race_data_file}")
        
        # Save dog data
        dog_data_file = f"enhanced_dog_data_{timestamp}.json"
        with open(dog_data_file, 'w') as f:
            json.dump(all_dogs_data, f, indent=2, default=str)
        logger.info(f"✅ Dog data saved to {dog_data_file}")
        
        # Save analytics
        analytics_file = f"enhanced_analytics_{timestamp}.json"
        with open(analytics_file, 'w') as f:
            json.dump(analytics, f, indent=2, default=str)
        logger.info(f"✅ Analytics saved to {analytics_file}")
        
        return race_data_file, dog_data_file, analytics_file
    
    def run_full_enhanced_collection():
        """Run the complete enhanced data collection process."""
        logger.info("🚀 Starting FULL Enhanced Data Collection for TGR Scraper")
        logger.info("=" * 70)
        
        # Initialize scraper
        scraper = TheGreyhoundRecorderScraper(rate_limit=0.1, use_cache=True)
        
        # Add custom parsing method to handle soup directly
        def _parse_race_from_soup(self, soup, url):
            """Parse race data directly from BeautifulSoup object."""
            race_details = {
                'url': url,
                'date': None,
                'venue': None,
                'race_number': None,
                'grade': None,
                'distance': None,
                'field_size': 0,
                'dogs': [],
                'race_result': {},
                'expert_comments': []
            }
            
            try:
                # Extract race title/heading
                heading = soup.find(class_='form-guide-meeting__heading')
                if heading:
                    heading_text = heading.get_text(strip=True)
                    
                    if 'Form Guide' in heading_text:
                        parts = heading_text.split('Form Guide')
                        if len(parts) >= 2:
                            race_details['venue'] = parts[0].strip()
                            
                            race_part = parts[1]
                            if 'Race' in race_part and ')' in race_part:
                                import re
                                race_match = re.search(r'Race (\d+)', race_part)
                                if race_match:
                                    race_details['race_number'] = int(race_match.group(1))
                                
                                if '-' in race_part:
                                    date_part = race_part.split('-')[-1].strip()
                                    race_details['date'] = date_part
                
                # Extract individual dog racing histories
                dog_sections = soup.find_all(class_='form-guide-long-form-selection')
                
                for section in dog_sections:
                    if 'form-guide-long-form-selection--vacant' in section.get('class', []):
                        continue
                    
                    header = section.find(class_='form-guide-long-form-selection__header')
                    if not header:
                        continue
                    
                    dog_name_elem = header.find(class_='form-guide-long-form-selection__header-name')
                    if not dog_name_elem:
                        continue
                    
                    dog_name = dog_name_elem.get_text(strip=True)
                    
                    history_table = section.find('table', class_='form-guide-selection-results')
                    if history_table:
                        dog_history = self._extract_dog_racing_history(history_table, dog_name)
                        if dog_history:
                            dog_entry = {
                                'dog_name': dog_name,
                                'racing_history': dog_history,
                                'total_races': len(dog_history)
                            }
                            race_details['dogs'].append(dog_entry)
                            race_details['field_size'] += 1
                            
            except Exception as e:
                logger.error(f"Error parsing race from soup: {e}")
            
            return race_details
        
        # Bind the method to the scraper instance
        import types
        scraper._parse_race_from_soup = types.MethodType(_parse_race_from_soup, scraper)
        
        try:
            # Step 1: Discover all race files
            race_files = discover_race_files()
            
            # Step 2: Extract comprehensive data
            all_race_data, all_dogs_data, total_dogs, total_races = extract_comprehensive_race_data(
                scraper, race_files
            )
            
            # Step 3: Calculate analytics
            analytics = calculate_enhanced_analytics(all_dogs_data)
            
            # Step 4: Save data
            race_file, dog_file, analytics_file = save_enhanced_data(
                all_race_data, all_dogs_data, analytics
            )
            
            # Step 5: Display summary
            logger.info("\n" + "=" * 70)
            logger.info("🎉 FULL ENHANCED DATA COLLECTION COMPLETE!")
            logger.info("=" * 70)
            logger.info(f"📊 FINAL RESULTS:")
            logger.info(f"   • Race files processed: {len(race_files)}")
            logger.info(f"   • Races extracted: {len(all_race_data)}")
            logger.info(f"   • Unique dogs found: {analytics['summary']['total_unique_dogs']}")
            logger.info(f"   • Dogs with racing histories: {analytics['summary']['dogs_with_histories']}")
            logger.info(f"   • Total race history records: {analytics['summary']['total_history_records']}")
            logger.info(f"   • Average history per dog: {analytics['summary']['average_history_per_dog']:.1f}")
            logger.info(f"   • Unique venues covered: {analytics['summary']['venue_count']}")
            logger.info(f"   • Unique grades found: {analytics['summary']['grade_count']}")
            logger.info(f"   • Unique distances found: {analytics['summary']['distance_count']}")
            
            logger.info(f"\n📁 FILES GENERATED:")
            logger.info(f"   • Race data: {race_file}")
            logger.info(f"   • Dog data: {dog_file}")
            logger.info(f"   • Analytics: {analytics_file}")
            
            logger.info(f"\n🏆 TOP PERFORMERS:")
            if analytics['top_dogs']['most_history_records']:
                top_dog = analytics['top_dogs']['most_history_records'][0]
                logger.info(f"   • Most race records: {top_dog['dog_name']} ({top_dog['history_records']} records)")
            
            if analytics['top_dogs']['most_venues']:
                versatile_dog = analytics['top_dogs']['most_venues'][0]
                logger.info(f"   • Most venues: {versatile_dog['dog_name']} ({versatile_dog['venue_count']} venues)")
            
            logger.info(f"\n🗺️ VENUE COVERAGE:")
            for venue in analytics['summary']['unique_venues'][:10]:
                logger.info(f"   • {venue}")
            if len(analytics['summary']['unique_venues']) > 10:
                logger.info(f"   • ... and {len(analytics['summary']['unique_venues']) - 10} more venues")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Enhanced data collection failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    if __name__ == "__main__":
        success = run_full_enhanced_collection()
        sys.exit(0 if success else 1)

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nRequired dependencies: requests, beautifulsoup4")
    print("Install with: pip install requests beautifulsoup4")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
