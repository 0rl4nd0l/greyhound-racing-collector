#!/usr/bin/env python3
"""
Enhanced Race Result Processor - Fixed Version
==============================================

This module provides comprehensive race result processing with:
1. Dead heat handling
2. Multiple data format support
3. Data validation and quality checks
4. Automatic position assignment
5. Error recovery and fallback strategies
"""

import json
import logging
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


class EnhancedRaceProcessor:
    def __init__(self, db_path='greyhound_racing_data.db'):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # Data format patterns
        self.format_patterns = {
            'navigator_results': {
                'columns': ['first', 'second', 'third', 'fourth'],
                'position_method': 'ordered_columns',
                'identifier': lambda df: all(col in df.columns for col in ['first', 'second', 'third', 'fourth'])
            },
            'form_guide': {
                'columns': ['PLC'],
                'position_method': 'plc_column',
                'identifier': lambda df: 'PLC' in df.columns and 'Dog Name' in df.columns
            },
            'standard_results': {
                'columns': ['position', 'finish_position'],
                'position_method': 'position_column',
                'identifier': lambda df: any(col in df.columns for col in ['position', 'finish_position'])
            }
        }
    
    def detect_data_format(self, df: pd.DataFrame) -> str:
        """Detect the format of the race data"""
        for format_name, format_info in self.format_patterns.items():
            if format_info['identifier'](df):
                return format_name
        return 'unknown'
    
    def process_race_results(self, file_path: Path, race_info: Dict = None, move_processed: bool = False) -> Dict[str, Any]:
        """
        Process race results from various file formats
        
        Args:
            file_path: Path to the race results file
            race_info: Optional race information dictionary
            move_processed: Whether to move successfully processed files to avoid reprocessing
            
        Returns:
            Dictionary containing processing results and statistics
        """
        try:
            # Read the CSV file
            df = pd.read_csv(file_path, sep='|' if '|' in open(file_path).read()[:500] else ',')
            
            # Extract race info if not provided
            if not race_info:
                from enhanced_race_parser import EnhancedRaceParser
                parser = EnhancedRaceParser()
                race_info = parser.extract_race_info(file_path.name)
            
            # Detect data format
            data_format = self.detect_data_format(df)
            print(f"Detected data format: {data_format} for file {file_path.name}")
            
            # Process based on format
            if data_format == 'navigator_results':
                results = self._process_navigator_results(df, race_info)
            elif data_format == 'form_guide':
                results = self._process_form_guide(df, race_info)
            elif data_format == 'standard_results':
                results = self._process_standard_results(df, race_info)
            else:
                results = self._process_unknown_format(df, race_info)
            
            # Validate results
            validation_results = self._validate_race_results(results)
            
            # Store in database if validation passes
            if validation_results['is_valid']:
                self._store_race_results(results, race_info)
                status = 'success'
            else:
                self._store_with_quality_issues(results, race_info, validation_results)
                status = 'success_with_issues'
            
            # Move file to processed folder if requested and processing was successful
            moved_to = None
            if move_processed and status in ['success', 'success_with_issues']:
                moved_to = self._move_processed_file(file_path, data_format)
            
            # Count races processed (group by race_id if available)
            race_ids = set()
            for result in results:
                if result.get('race_id'):
                    race_ids.add(result['race_id'])
                else:
                    race_ids.add(race_info.get('race_id', 'unknown'))
            
            return {
                'success': status == 'success',
                'status': status,
                'race_id': race_info.get('race_id'),
                'data_format': data_format,
                'dogs_processed': len(results),
                'races_processed': len(race_ids),
                'data_quality': validation_results,
                'summary': f"{len(race_ids)} races, {len(results)} dogs processed",
                'moved_to': moved_to,
                'processing_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error processing race results from {file_path}: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'processing_timestamp': datetime.now().isoformat()
            }
    
    def _process_navigator_results(self, df: pd.DataFrame, race_info: Dict) -> List[Dict]:
        """Process navigator-style results with first/second/third/fourth columns"""
        results = []
        
        # Process each row (each represents a different race)
        for _, row in df.iterrows():
            # Extract placings
            placings = {
                1: str(row.get('first', '')).strip(),
                2: str(row.get('second', '')).strip(),
                3: str(row.get('third', '')).strip(),
                4: str(row.get('fourth', '')).strip()
            }
            
            # Handle dead heats and create position mapping
            position_map = self._create_position_map_from_placings(placings)
            
            # Extract race metadata
            race_time = str(row.get('time', ''))
            race_id = str(row.get('race_id', ''))
            venue = str(row.get('venue', ''))
            race_number = row.get('race_number', 0)
            
            # Extract rich metadata
            metadata = {
                'race_date': str(row.get('race_date', '')),
                'race_grade': str(row.get('race_grade', '')),
                'prize_money': row.get('prize_money'),
                'track_condition': str(row.get('track_condition', '')),
                'weather': str(row.get('weather', '')),
                'source_url': str(row.get('source_url', '')),
                'filename': str(row.get('filename', '')),
                'field_size': row.get('field_size'),
                'margins': str(row.get('margins', '')),
                'race_time': race_time
            }
            
            # Create results for each dog in this race
            for dog_name, position in position_map.items():
                if dog_name and dog_name.lower() not in ['', 'nan', 'none']:
                    results.append({
                        'dog_name': dog_name,
                        'clean_name': self._clean_dog_name(dog_name),
                        'finish_position': position,
                        'race_time': race_time,
                        'trainer': self._get_trainer_for_position(row, position),
                        'data_source': 'navigator_results',
                        'race_id': race_id,
                        'venue': venue,
                        'race_number': race_number,
                        'metadata': metadata
                    })
        
        return results
    
    def _process_form_guide(self, df: pd.DataFrame, race_info: Dict) -> List[Dict]:
        """Process form guide data - this contains historical data, not race results"""
        print("⚠️  Form guide detected - this contains historical data, not race results")
        
        # Extract metadata from form guide (distance, venue info)
        metadata = {}
        if len(df) > 0:
            # Get distance from first row (most recent race data)
            first_row = df.iloc[0]
            distance = first_row.get('DIST')
            if distance:
                metadata['distance'] = distance
            
            # Extract venue and other race info from historical data
            track_info = first_row.get('TRACK', '')
            if track_info:
                metadata['track_info'] = str(track_info)
            
            grade_info = first_row.get('G', '')
            if grade_info:
                metadata['grade_info'] = str(grade_info)
        
        results = []
        for _, row in df.iterrows():
            dog_name = str(row.get('Dog Name', '')).strip()
            if dog_name and dog_name != '""' and dog_name.lower() not in ['nan', 'none']:
                results.append({
                    'dog_name': dog_name,
                    'clean_name': self._clean_dog_name(dog_name),
                    'finish_position': None,  # No actual race result available
                    'box_number': row.get('BOX'),
                    'weight': row.get('WGT'),
                    'data_source': 'form_guide',
                    'data_quality_note': 'Form guide data - no race results available',
                    'metadata': metadata
                })
        
        return results
    
    def _process_standard_results(self, df: pd.DataFrame, race_info: Dict) -> List[Dict]:
        """Process standard results format with position column"""
        results = []
        
        for _, row in df.iterrows():
            dog_name = str(row.get('Dog Name', row.get('dog_name', ''))).strip()
            if dog_name and dog_name.lower() not in ['', 'nan', 'none']:
                position_col = 'position' if 'position' in df.columns else 'finish_position'
                position = row.get(position_col)
                
                results.append({
                    'dog_name': dog_name,
                    'clean_name': self._clean_dog_name(dog_name),
                    'finish_position': str(position) if position is not None else None,
                    'box_number': row.get('box_number', row.get('BOX')),
                    'weight': row.get('weight', row.get('WGT')),
                    'race_time': row.get('time', row.get('TIME')),
                    'margin': row.get('margin', row.get('MGN')),
                    'trainer': row.get('trainer', row.get('TRAINER')),
                    'data_source': 'standard_results'
                })
        
        return results
    
    def _process_unknown_format(self, df: pd.DataFrame, race_info: Dict) -> List[Dict]:
        """Process unknown format with best-effort extraction"""
        results = []
        print(f"⚠️  Unknown data format detected, attempting best-effort extraction")
        
        # Try to find dog names and any position information
        name_columns = [col for col in df.columns if 'name' in col.lower() or 'dog' in col.lower()]
        position_columns = [col for col in df.columns if any(term in col.lower() for term in ['pos', 'plc', 'place', 'finish'])]
        
        name_col = name_columns[0] if name_columns else df.columns[0]
        position_col = position_columns[0] if position_columns else None
        
        for _, row in df.iterrows():
            dog_name = str(row.get(name_col, '')).strip()
            if dog_name and dog_name != '""' and dog_name.lower() not in ['nan', 'none']:
                position = row.get(position_col) if position_col else None
                
                results.append({
                    'dog_name': dog_name,
                    'clean_name': self._clean_dog_name(dog_name),
                    'finish_position': str(position) if position is not None else None,
                    'data_source': 'unknown_format',
                    'data_quality_note': f'Extracted from unknown format using column: {name_col}'
                })
        
        return results
    
    def _create_position_map_from_placings(self, placings: Dict[int, str]) -> Dict[str, str]:
        """Create position mapping with dead heat handling"""
        position_map = {}
        
        for position, dog_name in placings.items():
            if dog_name and dog_name.lower() not in ['', 'nan', 'none']:
                # Check if this dog is already assigned a position (dead heat)
                if dog_name in position_map:
                    # Mark as dead heat
                    existing_pos = position_map[dog_name].rstrip('=')
                    position_map[dog_name] = f"{existing_pos}="
                    print(f"🏁 Dead heat detected: {dog_name} tied at positions {existing_pos} and {position}")
                else:
                    position_map[dog_name] = str(position)
        
        return position_map
    
    def _get_trainer_for_position(self, row: pd.Series, position: str) -> str:
        """Get trainer name for a specific position"""
        trainer_cols = {
            '1': 'first_trainer',
            '2': 'second_trainer', 
            '3': 'third_trainer',
            '4': 'fourth_trainer'
        }
        
        position_num = position.rstrip('=')
        trainer_col = trainer_cols.get(position_num)
        return str(row.get(trainer_col, '')) if trainer_col else ''
    
    def _clean_dog_name(self, name: str) -> str:
        """Clean dog name for consistent storage"""
        if not name or str(name).lower() in ['nan', 'none']:
            return ''
        
        # Remove box numbers, quotes, and extra spaces
        cleaned = re.sub(r'^["\d\.\s]+', '', str(name))
        cleaned = re.sub(r'["\s]+$', '', cleaned)
        cleaned = cleaned.replace('NBT', '').strip()
        return cleaned.upper()
    
    def _validate_race_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Validate race results for data quality"""
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        if not results:
            validation['errors'].append('No race results found')
            validation['is_valid'] = False
            return validation
        
        # Check if this is a form guide (no positions expected)
        is_form_guide = any(r.get('data_source') == 'form_guide' for r in results)
        
        if not is_form_guide:
            # Check for missing positions (only for actual race results)
            positions = [r['finish_position'] for r in results if r['finish_position']]
            if not positions:
                validation['warnings'].append('No finish positions found - may be form guide data')
            else:
                # Check for position 1 (winner)
                position_nums = [int(p.rstrip('=')) for p in positions if p and p.rstrip('=').isdigit()]
                if 1 not in position_nums:
                    validation['warnings'].append('No winner (position 1) found')
                
                # Check for gaps in positions (less strict - only warn, don't fail)
                if position_nums and len(position_nums) > 1:
                    expected_positions = set(range(1, max(position_nums) + 1))
                    actual_positions = set(position_nums)
                    missing_positions = expected_positions - actual_positions
                    if missing_positions:
                        validation['warnings'].append(f'Missing positions: {sorted(missing_positions)}')
        
        # Check for duplicate dog names (less strict - warn only)
        dog_names = [r['clean_name'] for r in results if r['clean_name']]
        duplicates = [name for name in set(dog_names) if dog_names.count(name) > 1]
        if duplicates:
            validation['warnings'].append(f'Duplicate dog names: {duplicates}')
        
        # Only fail validation for critical errors (empty results)
        # Warnings don't prevent processing
        
        return validation
    
    def _store_race_results(self, results: List[Dict], race_info: Dict):
        """Store validated race results in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Group results by race_id if available
            race_groups = {}
            for result in results:
                race_id = result.get('race_id', race_info.get('race_id', 'unknown'))
                if race_id not in race_groups:
                    race_groups[race_id] = []
                race_groups[race_id].append(result)
            
            # Process each race group
            for race_id, race_results in race_groups.items():
                # Extract metadata from first result or race_info
                first_result = race_results[0]
                metadata = first_result.get('metadata', {})
                
                # Extract race name from URL if available
                race_name = None
                source_url = metadata.get('source_url', '')
                if source_url:
                    # Extract race name from URL path
                    import re
                    name_match = re.search(r'/([^/]+)\?', source_url)
                    if name_match:
                        race_name = name_match.group(1).replace('-', ' ').title()
                
                # Insert/update race metadata with full information
                cursor.execute("""
                    INSERT OR REPLACE INTO race_metadata (
                        race_id, venue, race_number, race_date, race_name, grade, distance,
                        track_condition, weather, prize_money_total, race_time, field_size,
                        url, extraction_timestamp, data_source, race_status, winner_name
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    race_id,
                    first_result.get('venue', race_info.get('venue', 'Unknown')),
                    first_result.get('race_number', race_info.get('race_number', 0)),
                    metadata.get('race_date', race_info.get('date_str', datetime.now().strftime('%Y-%m-%d'))),
                    race_name,
                    metadata.get('race_grade') or metadata.get('grade_info'),
                    self._parse_distance(metadata.get('distance')),
                    metadata.get('track_condition'),
                    metadata.get('weather'),
                    self._parse_float(metadata.get('prize_money')),
                    metadata.get('race_time'),
                    len(race_results),
                    metadata.get('source_url'),
                    datetime.now().isoformat(),
                    'enhanced_race_processor',
                    'completed',
                    self._get_winner_name(race_results)
                ))
                
                # Insert dog race data
                for result in race_results:
                    cursor.execute("""
                        INSERT OR REPLACE INTO dog_race_data (
                            race_id, dog_name, dog_clean_name, box_number, finish_position,
                            trainer_name, weight, individual_time, margin,
                            extraction_timestamp, data_source, data_quality_note
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        race_id,
                        result['dog_name'],
                        result['clean_name'],
                        result.get('box_number'),
                        result.get('finish_position'),
                        result.get('trainer'),
                        result.get('weight'),
                        result.get('race_time'),
                        result.get('margin'),
                        datetime.now().isoformat(),
                        result.get('data_source', 'enhanced_race_processor'),
                        result.get('data_quality_note')
                    ))
            
            conn.commit()
            print(f"✅ Stored race results for {len(race_groups)} races: {sum(len(group) for group in race_groups.values())} dogs total")
            
        except Exception as e:
            conn.rollback()
            print(f"❌ Error storing race results: {e}")
            raise
        finally:
            conn.close()
    
    def _store_with_quality_issues(self, results: List[Dict], race_info: Dict, validation: Dict):
        """Store race results with quality issue markers"""
        # Mark race as needing review
        race_info['race_status'] = 'needs_review'
        race_info['data_quality_note'] = '; '.join(validation['warnings'] + validation['errors'])
        
        # Add quality notes to each result
        for result in results:
            if not result.get('data_quality_note'):
                result['data_quality_note'] = 'Stored with data quality issues'
        
        self._store_race_results(results, race_info)
        print(f"⚠️  Stored race {race_info.get('race_id')} with quality issues: {race_info['data_quality_note']}")
    
    def _parse_distance(self, distance_str):
        """Parse distance string to extract meters"""
        if not distance_str or str(distance_str).lower() in ['nan', 'none', '']:
            return None
        
        # Extract numeric distance
        import re
        distance_match = re.search(r'(\d+)', str(distance_str))
        if distance_match:
            return f"{distance_match.group(1)}m"
        return str(distance_str)
    
    def _parse_float(self, value):
        """Parse float value safely"""
        if not value or str(value).lower() in ['nan', 'none', '']:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _get_winner_name(self, race_results):
        """Get the name of the winner (position 1)"""
        for result in race_results:
            if result.get('finish_position') == '1':
                return result.get('clean_name')
        return None
    
    def _move_processed_file(self, file_path: Path, data_format: str) -> str:
        """Move processed file to appropriate folder to avoid reprocessing"""
        try:
            # Determine destination folder based on data format
            if data_format == 'form_guide':
                dest_folder = Path('./processed/form_guides')
            elif data_format == 'navigator_results':
                dest_folder = Path('./processed/race_results')
            else:
                dest_folder = Path('./processed/other')
            
            # Create destination folder if it doesn't exist
            dest_folder.mkdir(parents=True, exist_ok=True)
            
            # Move file
            dest_path = dest_folder / file_path.name
            
            # Handle duplicate names by adding timestamp
            if dest_path.exists():
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                stem = dest_path.stem
                suffix = dest_path.suffix
                dest_path = dest_folder / f"{stem}_{timestamp}{suffix}"
            
            file_path.rename(dest_path)
            print(f"   📁 Moved processed file to: {dest_path}")
            return str(dest_path)
            
        except Exception as e:
            print(f"   ⚠️ Could not move file {file_path}: {e}")
            return None

def process_race_file(file_path: str, db_path: str = 'greyhound_racing_data.db') -> Dict:
    """Convenience function to process a single race file"""
    processor = EnhancedRaceProcessor(db_path)
    return processor.process_race_results(Path(file_path))

if __name__ == "__main__":
    # Test the processor
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        result = process_race_file(file_path)
        print("\n📊 PROCESSING RESULT:")
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python enhanced_race_processor_fixed.py <race_file.csv>")
        
        # Test with navigator results file
        navigator_file = Path('form_guides/navigator_race_results.csv')
        if navigator_file.exists():
            print("\n🧪 Testing with navigator results file...")
            result = process_race_file(str(navigator_file))
            print("\n📊 PROCESSING RESULT:")
            print(json.dumps(result, indent=2))
