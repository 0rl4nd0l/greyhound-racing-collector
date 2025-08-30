#!/usr/bin/env python3
"""
Simplified Enhanced Data Processor
==================================

This module provides TGR integration for enhanced data processing
without requiring numpy/pandas dependencies.
"""

import sys
import os
import csv
import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.insert(0, 'src')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplifiedEnhancedProcessor:
    """
    Simplified enhanced data processor that integrates TGR data
    without heavy dependencies like numpy/pandas.
    """
    
    def __init__(self, use_tgr=True, rate_limit=3.0):
        """Initialize the simplified enhanced processor."""
        
        self.use_tgr = use_tgr
        self.rate_limit = rate_limit
        
        # Initialize TGR components if available and requested
        self.tgr_scraper = None
        self.tgr_integrator = None
        
        if self.use_tgr:
            self._initialize_tgr_components()
    
    def _initialize_tgr_components(self):
        """Initialize TGR components."""
        
        try:
            from collectors.the_greyhound_recorder_scraper import TheGreyhoundRecorderScraper
            self.tgr_scraper = TheGreyhoundRecorderScraper(
                rate_limit=self.rate_limit,
                use_cache=True
            )
            logger.info("✅ TGR scraper initialized")
            
            from tgr_prediction_integration import TGRPredictionIntegrator
            self.tgr_integrator = TGRPredictionIntegrator()
            logger.info("✅ TGR prediction integrator initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ TGR components not available: {e}")
            self.use_tgr = False
    
    def process_csv_file(self, csv_file_path: str) -> Dict[str, Any]:
        """
        Process a single CSV file with optional TGR enhancement.
        
        Args:
            csv_file_path: Path to the CSV file to process
            
        Returns:
            Dictionary with processing results
        """
        
        logger.info(f"📋 Processing CSV file: {Path(csv_file_path).name}")
        
        try:
            # Read CSV file
            dogs = self._read_csv_file(csv_file_path)
            
            if not dogs:
                return {
                    'status': 'error',
                    'message': 'No dogs found in CSV file',
                    'dogs': []
                }
            
            logger.info(f"📊 Found {len(dogs)} dogs in CSV")
            
            # Enhance with TGR data if available
            if self.use_tgr and self.tgr_scraper:
                dogs = self._enhance_dogs_with_tgr(dogs, csv_file_path)
            
            # Save to database
            self._save_to_database(dogs, csv_file_path)
            
            # Calculate summary stats
            tgr_enhanced_count = sum(1 for dog in dogs if dog.get('has_tgr_data', False))
            
            result = {
                'status': 'success',
                'message': f'Processed {len(dogs)} dogs',
                'dogs': dogs,
                'stats': {
                    'total_dogs': len(dogs),
                    'tgr_enhanced': tgr_enhanced_count,
                    'enhancement_ratio': (tgr_enhanced_count / len(dogs)) * 100 if dogs else 0
                }
            }
            
            logger.info(f"✅ Processing complete - {tgr_enhanced_count}/{len(dogs)} dogs enhanced with TGR data")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing CSV file: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'status': 'error',
                'message': f'Error processing {Path(csv_file_path).name}: {e}',
                'dogs': []
            }
    
    def _read_csv_file(self, csv_file_path: str) -> List[Dict[str, Any]]:
        """Read and parse a CSV file."""
        
        import re
        import io
        
        dogs = []
        seen_dogs = set()  # Track dog names we've already processed
        
        try:
            raw_text = Path(csv_file_path).read_text(encoding='utf-8', errors='ignore')
            
            # Some exported files have row numbers like "1|" at the start of each line.
            # Strip a leading "<digits>|" prefix from every line.
            cleaned_lines = []
            for line in raw_text.splitlines():
                cleaned_lines.append(re.sub(r"^\s*\d+\|", "", line))
            cleaned_text = "\n".join(cleaned_lines)
            
            # Try to detect delimiter from the header
            header = cleaned_lines[0] if cleaned_lines else ''
            delimiter = ','
            if '\t' in header and header.count('\t') > header.count(','):
                delimiter = '\t'
            
            # Normalize header keys slightly by trimming BOM if present
            if cleaned_text.startswith('\ufeff'):
                cleaned_text = cleaned_text.lstrip('\ufeff')
            
            reader = csv.DictReader(io.StringIO(cleaned_text), delimiter=delimiter)
            
            current_group = []  # Store rows for the current dog
            current_dog_name = None
            
            for row in reader:
                dog_name = row.get('Dog Name', '').strip()
                
                # Skip sex-only rows that don't add information
                if dog_name == '' and (row.get('Sex') in ['D', 'B']):
                    continue
                
                # Start of a new dog entry if we have a name
                if dog_name and not dog_name == '""':
                    # Process previous group if we have one
                    if current_group and current_dog_name:
                        dog = self._extract_dog_from_group(current_group, current_dog_name)
                        if dog and current_dog_name not in seen_dogs:
                            dogs.append(dog)
                            seen_dogs.add(current_dog_name)
                    
                    # Start new group
                    current_group = [row]
                    current_dog_name = self._clean_dog_name(dog_name)
                else:
                    # Continue current group
                    current_group.append(row)
            
            # Process final group
            if current_group and current_dog_name:
                dog = self._extract_dog_from_group(current_group, current_dog_name)
                if dog and current_dog_name not in seen_dogs:
                    dogs.append(dog)
                    seen_dogs.add(current_dog_name)
        
        except Exception as e:
            logger.error(f"Error reading CSV file {csv_file_path}: {e}")
            raise
        
        return dogs
    
    def _extract_dog_from_group(self, rows: List[Dict[str, Any]], dog_name: str) -> Optional[Dict[str, Any]]:
        """Extract dog information from a group of form guide rows."""
        
        # Skip empty rows and sex-only entries
        if not rows or not dog_name:
            return None
        
        # Get the main row (first row)
        main_row = rows[0]
        
        # Find trap/box number
        trap = None
        box_val = main_row.get('BOX')
        if box_val:
            try:
                trap = int(str(box_val).strip())
            except (ValueError, TypeError):
                pass
        
        # Extract weight
        weight = None
        wgt_val = main_row.get('WGT')
        if wgt_val:
            try:
                weight = float(str(wgt_val).strip())
            except (ValueError, TypeError):
                pass
        
        # Extract important history info
        race_history = []
        for row in rows:
            date = row.get('DATE', '').strip()
            dist = row.get('DIST', '').strip()
            track = row.get('TRACK', '').strip()
            plc = row.get('PLC', '').strip()
            grade = row.get('G', '').strip()
            time = row.get('TIME', '').strip()
            sp = row.get('SP', '').strip()
            
            if any([date, dist, track, plc]):
                race_history.append({
                    'date': date,
                    'distance': dist,
                    'track': track,
                    'place': plc,
                    'grade': grade,
                    'time': time,
                    'starting_price': sp
                })
        
        # Create dog dictionary with comprehensive information
        dog = {
            'dog_name': dog_name,
            'clean_name': dog_name,  # Already cleaned in _read_csv_file
            'trap': trap,
            'weight': weight,
            'sex': main_row.get('Sex', ''),
            'track': main_row.get('TRACK', ''),
            'distance': main_row.get('DIST', ''),
            'race_history': race_history,
            'source_csv_data': main_row,
            'has_tgr_data': False,
            'processed_at': datetime.now().isoformat()
        }
        
        return dog
    
    def _clean_dog_name(self, name: str) -> str:
        """Clean and standardize dog name."""
        
        import re
        
        if not name:
            return ""
        
        # Remove leading numbering like "1." or "1)" or any prefix
        cleaned = re.sub(r"^\s*\d+\s*[\.)]\s*", "", name)
        
        # Remove stray quotes and normalize whitespace
        cleaned = cleaned.strip().replace("'", "").replace('"', '')
        cleaned = ' '.join(cleaned.split())
        
        # Upper case for consistency
        cleaned = cleaned.upper()
        
        # Skip empty strings and standalone sex indicators
        if not cleaned or cleaned in {'D', 'B', 'DOG', 'BITCH'}:
            return ""
        
        return cleaned
    
    def _enhance_dogs_with_tgr(self, dogs: List[Dict[str, Any]], csv_file_path: str) -> List[Dict[str, Any]]:
        """Enhance dogs with TGR data."""
        
        if not self.tgr_scraper:
            logger.warning("TGR scraper not available")
            return dogs
        
        logger.info(f"🔍 Enhancing {len(dogs)} dogs with TGR data...")
        
        enhanced_dogs = []
        
        for i, dog in enumerate(dogs):
            try:
                dog_name = dog.get('clean_name', dog.get('dog_name', ''))
                
                if not dog_name:
                    enhanced_dogs.append(dog)
                    continue
                
                # Fetch TGR data using the correct method
                tgr_data = self.tgr_scraper.fetch_enhanced_dog_data(dog_name)
                
                if tgr_data and tgr_data.get('form_entries'):
                    # Add TGR enhancement
                    dog['has_tgr_data'] = True
                    dog['tgr_data'] = tgr_data
                    dog['tgr_fetch_timestamp'] = datetime.now().isoformat()
                    
                    # Extract key TGR statistics
                    self._extract_tgr_stats(dog, tgr_data)
                    
                    logger.info(f"✅ Enhanced {dog_name} with TGR data")
                else:
                    logger.debug(f"⚠️ No TGR data found for {dog_name}")
                
                enhanced_dogs.append(dog)
                
                # Progress update
                if (i + 1) % 10 == 0:
                    tgr_count = sum(1 for d in enhanced_dogs if d.get('has_tgr_data'))
                    logger.info(f"📊 Progress: {i+1}/{len(dogs)} dogs processed, {tgr_count} enhanced")
            
            except Exception as e:
                logger.error(f"Error enhancing dog {dog.get('dog_name', 'Unknown')}: {e}")
                enhanced_dogs.append(dog)
                continue
        
        tgr_enhanced_count = sum(1 for dog in enhanced_dogs if dog.get('has_tgr_data'))
        logger.info(f"🎯 TGR Enhancement complete: {tgr_enhanced_count}/{len(enhanced_dogs)} dogs enhanced")
        
        return enhanced_dogs
    
    def _extract_tgr_stats(self, dog: Dict[str, Any], tgr_data: Dict[str, Any]):
        """Extract key statistics from TGR data."""
        
        try:
            # fetch_enhanced_dog_data returns form_entries instead of races
            form_entries = tgr_data.get('form_entries', [])
            
            if form_entries:
                dog['tgr_stats'] = {
                    'total_races': len(form_entries),
                    'recent_races': len(form_entries[:5]),  # Last 5 races
                    'wins': 0,
                    'places': 0,
                }
                
                # Count wins and places from form entries
                recent_positions = []
                for entry in form_entries:
                    # Check for finish position in various fields
                    position = None
                    if 'finish_position' in entry:
                        position = entry['finish_position']
                    elif 'position' in entry:
                        position = entry['position']
                    
                    if position:
                        if position == 1:
                            dog['tgr_stats']['wins'] += 1
                        if position <= 3:
                            dog['tgr_stats']['places'] += 1
                        
                        # Collect recent form (first 5 races)
                        if len(recent_positions) < 5:
                            recent_positions.append(position)
                
                # Calculate win/place rates
                if len(form_entries) > 0:
                    dog['tgr_stats']['win_rate'] = (dog['tgr_stats']['wins'] / len(form_entries)) * 100
                    dog['tgr_stats']['place_rate'] = (dog['tgr_stats']['places'] / len(form_entries)) * 100
                
                # Store recent form
                dog['tgr_stats']['recent_form'] = recent_positions
                
                # Copy performance summary if available
                if 'performance_summary' in tgr_data:
                    dog['tgr_performance_summary'] = tgr_data['performance_summary']
                
                # Copy venue analysis if available  
                if 'venue_analysis' in tgr_data:
                    dog['tgr_venue_analysis'] = tgr_data['venue_analysis']
                    
                # Copy distance analysis if available
                if 'distance_analysis' in tgr_data:
                    dog['tgr_distance_analysis'] = tgr_data['distance_analysis']
        
        except Exception as e:
            logger.error(f"Error extracting TGR stats: {e}")
    
    def _save_to_database(self, dogs: List[Dict[str, Any]], csv_file_path: str):
        """Save processed dogs to database."""
        
        try:
            # Use the main database
            db_path = "race_data.db"
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Create table if it doesn't exist
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS processed_dogs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        dog_name TEXT NOT NULL,
                        clean_name TEXT NOT NULL,
                        trap INTEGER,
                        has_tgr_data BOOLEAN DEFAULT FALSE,
                        tgr_stats TEXT,
                        source_csv TEXT,
                        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        csv_data TEXT
                    )
                ''')
                
                # Insert dogs
                for dog in dogs:
                    cursor.execute('''
                        INSERT INTO processed_dogs 
                        (dog_name, clean_name, trap, has_tgr_data, tgr_stats, source_csv, csv_data)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        dog.get('dog_name', ''),
                        dog.get('clean_name', ''),
                        dog.get('trap'),
                        dog.get('has_tgr_data', False),
                        json.dumps(dog.get('tgr_stats', {})),
                        Path(csv_file_path).name,
                        json.dumps(dog.get('source_csv', {}))
                    ))
                
                conn.commit()
                logger.info(f"💾 Saved {len(dogs)} dogs to database")
        
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
    
    def process_multiple_files(self, csv_directory: str, max_files: int = None) -> Dict[str, Any]:
        """Process multiple CSV files from a directory."""
        
        csv_dir = Path(csv_directory)
        if not csv_dir.exists():
            return {
                'status': 'error',
                'message': f'Directory not found: {csv_directory}'
            }
        
        csv_files = list(csv_dir.glob('*.csv'))
        
        if not csv_files:
            return {
                'status': 'error',
                'message': f'No CSV files found in: {csv_directory}'
            }
        
        if max_files:
            csv_files = csv_files[:max_files]
        
        logger.info(f"🚀 Processing {len(csv_files)} CSV files from {csv_directory}")
        
        results = {
            'status': 'success',
            'total_files': len(csv_files),
            'processed_files': 0,
            'total_dogs': 0,
            'total_tgr_enhanced': 0,
            'files': []
        }
        
        for csv_file in csv_files:
            try:
                result = self.process_csv_file(str(csv_file))
                
                if result['status'] == 'success':
                    results['processed_files'] += 1
                    results['total_dogs'] += result['stats']['total_dogs']
                    results['total_tgr_enhanced'] += result['stats']['tgr_enhanced']
                
                results['files'].append({
                    'file': csv_file.name,
                    'status': result['status'],
                    'dogs': result['stats']['total_dogs'] if result['status'] == 'success' else 0,
                    'tgr_enhanced': result['stats']['tgr_enhanced'] if result['status'] == 'success' else 0
                })
                
            except Exception as e:
                logger.error(f"Error processing {csv_file.name}: {e}")
                results['files'].append({
                    'file': csv_file.name,
                    'status': 'error',
                    'error': str(e)
                })
        
        # Calculate final enhancement ratio
        if results['total_dogs'] > 0:
            results['enhancement_ratio'] = (results['total_tgr_enhanced'] / results['total_dogs']) * 100
        else:
            results['enhancement_ratio'] = 0
        
        logger.info(f"🎯 Batch processing complete: {results['processed_files']}/{results['total_files']} files, "
                   f"{results['total_tgr_enhanced']}/{results['total_dogs']} dogs enhanced "
                   f"({results['enhancement_ratio']:.1f}%)")
        
        return results

def main():
    """Test the simplified enhanced processor."""
    
    print("🚀 Testing Simplified Enhanced Processor")
    print("=" * 50)
    
    # Initialize processor
    processor = SimplifiedEnhancedProcessor(use_tgr=True, rate_limit=2.0)
    
    # Test with unprocessed directory
    unprocessed_dir = Path('./unprocessed')
    if unprocessed_dir.exists():
        csv_files = list(unprocessed_dir.glob('*.csv'))
        if csv_files:
            print(f"📁 Found {len(csv_files)} files in unprocessed directory")
            
            # Process just one file as a test
            test_file = csv_files[0]
            print(f"🧪 Testing with: {test_file.name}")
            
            result = processor.process_csv_file(str(test_file))
            
            print(f"📊 Result: {result['status']}")
            if result['status'] == 'success':
                stats = result['stats']
                print(f"   • Dogs processed: {stats['total_dogs']}")
                print(f"   • TGR enhanced: {stats['tgr_enhanced']}")
                print(f"   • Enhancement ratio: {stats['enhancement_ratio']:.1f}%")
            
            return result['status'] == 'success'
        else:
            print("⚠️ No CSV files found in unprocessed directory")
    else:
        print("⚠️ Unprocessed directory not found")
    
    return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\\n✅ Simplified Enhanced Processor is working!")
    else:
        print("\\n❌ Simplified Enhanced Processor needs debugging")
