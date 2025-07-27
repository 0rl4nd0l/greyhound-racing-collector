#!/usr/bin/env python3
"""
Enhanced Race Result Processor
=============================

This module provides comprehensive race result processing with:
1. Dead heat handling
2. Multiple data format support
3. Data validation and quality checks
4. Automatic position assignment
5. Error recovery and fallback strategies

Usage:
    from enhanced_race_processor import EnhancedRaceProcessor
    processor = EnhancedRaceProcessor()
    processor.process_race_results('race_file.csv')
"""

import pandas as pd
import sqlite3
import re
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json

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
    
    def process_race_results(self, file_path: Path, race_info: Dict = None) -> Dict[str, Any]:
        """
        Process race results from various file formats
        
        Args:
            file_path: Path to the race results file
            race_info: Optional race information dictionary
            
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
            self.logger.info(f"Detected data format: {data_format} for file {file_path.name}")
            
            # Process based on format
            if data_format == 'navigator_results':
                results = self._process_navigator_results(df, race_info)
            elif data_format == 'form_guide':
                results = self._process_form_guide(df, race_info)
            elif data_format == 'standard_results':
                results = self._process_standard_results(df, race_info)
            else:\n                results = self._process_unknown_format(df, race_info)\n            \n            # Validate results\n            validation_results = self._validate_race_results(results)\n            \n            # Store in database if validation passes\n            if validation_results['is_valid']:\n                self._store_race_results(results, race_info)\n                status = 'success'\n            else:\n                self._store_with_quality_issues(results, race_info, validation_results)\n                status = 'success_with_issues'\n            \n            return {\n                'status': status,\n                'race_id': race_info.get('race_id'),\n                'data_format': data_format,\n                'dogs_processed': len(results),\n                'validation': validation_results,\n                'processing_timestamp': datetime.now().isoformat()\n            }\n            \n        except Exception as e:\n            self.logger.error(f\"Error processing race results from {file_path}: {e}\")\n            return {\n                'status': 'error',\n                'error': str(e),\n                'processing_timestamp': datetime.now().isoformat()\n            }\n    \n    def _process_navigator_results(self, df: pd.DataFrame, race_info: Dict) -> List[Dict]:\n        \"\"\"Process navigator-style results with first/second/third/fourth columns\"\"\"\n        results = []\n        \n        # Get race result information\n        if len(df) > 0:\n            row = df.iloc[0]\n            \n            # Extract placings\n            placings = {\n                1: row.get('first', '').strip(),\n                2: row.get('second', '').strip(),\n                3: row.get('third', '').strip(),\n                4: row.get('fourth', '').strip()\n            }\n            \n            # Handle dead heats and create position mapping\n            position_map = self._create_position_map_from_placings(placings)\n            \n            # Extract other race data\n            margins = self._parse_margins(row.get('margins', ''))\n            race_time = row.get('time', '')\n            field_size = int(row.get('field_size', 0)) if row.get('field_size') else 0\n            \n            # Create results for each dog\n            for dog_name, position in position_map.items():\n                if dog_name:  # Skip empty names\n                    results.append({\n                        'dog_name': dog_name,\n                        'clean_name': self._clean_dog_name(dog_name),\n                        'finish_position': position,\n                        'race_time': race_time,\n                        'margin': margins.get(dog_name),\n                        'trainer': self._get_trainer_for_position(row, position),\n                        'data_source': 'navigator_results'\n                    })\n        \n        return results\n    \n    def _process_form_guide(self, df: pd.DataFrame, race_info: Dict) -> List[Dict]:\n        \"\"\"Process form guide data - this contains historical data, not race results\"\"\"\n        # Form guides don't contain race results for the current race\n        # They contain historical performance data\n        self.logger.warning(\"Form guide detected - this contains historical data, not race results\")\n        \n        results = []\n        for _, row in df.iterrows():\n            dog_name = str(row.get('Dog Name', '')).strip()\n            if dog_name and dog_name != '\"\"':\n                results.append({\n                    'dog_name': dog_name,\n                    'clean_name': self._clean_dog_name(dog_name),\n                    'finish_position': None,  # No actual race result available\n                    'box_number': row.get('BOX'),\n                    'weight': row.get('WGT'),\n                    'data_source': 'form_guide',\n                    'data_quality_note': 'Form guide data - no race results available'\n                })\n        \n        return results\n    \n    def _process_standard_results(self, df: pd.DataFrame, race_info: Dict) -> List[Dict]:\n        \"\"\"Process standard results format with position column\"\"\"\n        results = []\n        \n        for _, row in df.iterrows():\n            dog_name = str(row.get('Dog Name', row.get('dog_name', ''))).strip()\n            if dog_name:\n                position_col = 'position' if 'position' in df.columns else 'finish_position'\n                position = row.get(position_col)\n                \n                results.append({\n                    'dog_name': dog_name,\n                    'clean_name': self._clean_dog_name(dog_name),\n                    'finish_position': str(position) if position is not None else None,\n                    'box_number': row.get('box_number', row.get('BOX')),\n                    'weight': row.get('weight', row.get('WGT')),\n                    'race_time': row.get('time', row.get('TIME')),\n                    'margin': row.get('margin', row.get('MGN')),\n                    'trainer': row.get('trainer', row.get('TRAINER')),\n                    'data_source': 'standard_results'\n                })\n        \n        return results\n    \n    def _process_unknown_format(self, df: pd.DataFrame, race_info: Dict) -> List[Dict]:\n        \"\"\"Process unknown format with best-effort extraction\"\"\"\n        results = []\n        self.logger.warning(f\"Unknown data format detected, attempting best-effort extraction\")\n        \n        # Try to find dog names and any position information\n        name_columns = [col for col in df.columns if 'name' in col.lower() or 'dog' in col.lower()]\n        position_columns = [col for col in df.columns if any(term in col.lower() for term in ['pos', 'plc', 'place', 'finish'])]\n        \n        name_col = name_columns[0] if name_columns else df.columns[0]\n        position_col = position_columns[0] if position_columns else None\n        \n        for _, row in df.iterrows():\n            dog_name = str(row.get(name_col, '')).strip()\n            if dog_name and dog_name != '\"\"':\n                position = row.get(position_col) if position_col else None\n                \n                results.append({\n                    'dog_name': dog_name,\n                    'clean_name': self._clean_dog_name(dog_name),\n                    'finish_position': str(position) if position is not None else None,\n                    'data_source': 'unknown_format',\n                    'data_quality_note': f'Extracted from unknown format using column: {name_col}'\n                })\n        \n        return results\n    \n    def _create_position_map_from_placings(self, placings: Dict[int, str]) -> Dict[str, str]:\n        \"\"\"Create position mapping with dead heat handling\"\"\"\n        position_map = {}\n        \n        for position, dog_name in placings.items():\n            if dog_name:\n                # Check if this dog is already assigned a position (dead heat)\n                if dog_name in position_map:\n                    # Mark as dead heat\n                    existing_pos = position_map[dog_name].rstrip('=')\n                    position_map[dog_name] = f\"{existing_pos}=\"\n                else:\n                    position_map[dog_name] = str(position)\n        \n        return position_map\n    \n    def _parse_margins(self, margins_str: str) -> Dict[str, str]:\n        \"\"\"Parse margins string into dictionary\"\"\"\n        margins = {}\n        if margins_str:\n            # This would need to be implemented based on the specific format\n            # For now, return empty dict\n            pass\n        return margins\n    \n    def _get_trainer_for_position(self, row: pd.Series, position: str) -> str:\n        \"\"\"Get trainer name for a specific position\"\"\"\n        trainer_cols = {\n            '1': 'first_trainer',\n            '2': 'second_trainer', \n            '3': 'third_trainer',\n            '4': 'fourth_trainer'\n        }\n        \n        position_num = position.rstrip('=')\n        trainer_col = trainer_cols.get(position_num)\n        return row.get(trainer_col, '') if trainer_col else ''\n    \n    def _clean_dog_name(self, name: str) -> str:\n        \"\"\"Clean dog name for consistent storage\"\"\"\n        if not name:\n            return ''\n        \n        # Remove box numbers, quotes, and extra spaces\n        cleaned = re.sub(r'^[\"\\d\\.\\s]+', '', str(name))\n        cleaned = re.sub(r'[\"\\s]+$', '', cleaned)\n        cleaned = cleaned.replace('NBT', '').strip()\n        return cleaned.upper()\n    \n    def _validate_race_results(self, results: List[Dict]) -> Dict[str, Any]:\n        \"\"\"Validate race results for data quality\"\"\"\n        validation = {\n            'is_valid': True,\n            'warnings': [],\n            'errors': []\n        }\n        \n        if not results:\n            validation['errors'].append('No race results found')\n            validation['is_valid'] = False\n            return validation\n        \n        # Check for missing positions\n        positions = [r['finish_position'] for r in results if r['finish_position']]\n        if not positions:\n            validation['warnings'].append('No finish positions found')\n        else:\n            # Check for position 1 (winner)\n            position_nums = [int(p.rstrip('=')) for p in positions if p and p.rstrip('=').isdigit()]\n            if 1 not in position_nums:\n                validation['warnings'].append('No winner (position 1) found')\n            \n            # Check for gaps in positions\n            if position_nums:\n                expected_positions = set(range(1, max(position_nums) + 1))\n                actual_positions = set(position_nums)\n                missing_positions = expected_positions - actual_positions\n                if missing_positions:\n                    validation['warnings'].append(f'Missing positions: {sorted(missing_positions)}')\n        \n        # Check for duplicate dog names\n        dog_names = [r['clean_name'] for r in results if r['clean_name']]\n        duplicates = [name for name in set(dog_names) if dog_names.count(name) > 1]\n        if duplicates:\n            validation['warnings'].append(f'Duplicate dog names: {duplicates}')\n        \n        return validation\n    \n    def _store_race_results(self, results: List[Dict], race_info: Dict):\n        \"\"\"Store validated race results in database\"\"\"\n        conn = sqlite3.connect(self.db_path)\n        cursor = conn.cursor()\n        \n        try:\n            race_id = race_info.get('race_id', 'unknown')\n            \n            # Insert/update race metadata\n            cursor.execute(\"\"\"\n                INSERT OR REPLACE INTO race_metadata (\n                    race_id, venue, race_number, race_date, field_size,\n                    extraction_timestamp, data_source, race_status\n                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)\n            \"\"\", (\n                race_id,\n                race_info.get('venue', 'Unknown'),\n                race_info.get('race_number', 0),\n                race_info.get('date_str', datetime.now().strftime('%Y-%m-%d')),\n                len(results),\n                datetime.now().isoformat(),\n                'enhanced_race_processor',\n                'completed'\n            ))\n            \n            # Insert dog race data\n            for result in results:\n                cursor.execute(\"\"\"\n                    INSERT OR REPLACE INTO dog_race_data (\n                        race_id, dog_name, dog_clean_name, box_number, finish_position,\n                        trainer_name, weight, individual_time, margin,\n                        extraction_timestamp, data_source, data_quality_note\n                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\n                \"\"\", (\n                    race_id,\n                    result['dog_name'],\n                    result['clean_name'],\n                    result.get('box_number'),\n                    result.get('finish_position'),\n                    result.get('trainer'),\n                    result.get('weight'),\n                    result.get('race_time'),\n                    result.get('margin'),\n                    datetime.now().isoformat(),\n                    result.get('data_source', 'enhanced_race_processor'),\n                    result.get('data_quality_note')\n                ))\n            \n            conn.commit()\n            self.logger.info(f\"Stored race results for {race_id}: {len(results)} dogs\")\n            \n        except Exception as e:\n            conn.rollback()\n            self.logger.error(f\"Error storing race results: {e}\")\n            raise\n        finally:\n            conn.close()\n    \n    def _store_with_quality_issues(self, results: List[Dict], race_info: Dict, validation: Dict):\n        \"\"\"Store race results with quality issue markers\"\"\"\n        # Mark race as needing review\n        race_info['race_status'] = 'needs_review'\n        race_info['data_quality_note'] = '; '.join(validation['warnings'] + validation['errors'])\n        \n        # Add quality notes to each result\n        for result in results:\n            if not result.get('data_quality_note'):\n                result['data_quality_note'] = 'Stored with data quality issues'\n        \n        self._store_race_results(results, race_info)\n        self.logger.warning(f\"Stored race {race_info.get('race_id')} with quality issues: {race_info['data_quality_note']}\")\n\ndef process_race_file(file_path: str, db_path: str = 'greyhound_racing_data.db') -> Dict:\n    \"\"\"Convenience function to process a single race file\"\"\"\n    processor = EnhancedRaceProcessor(db_path)\n    return processor.process_race_results(Path(file_path))\n\nif __name__ == \"__main__\":\n    # Test the processor\n    import sys\n    \n    if len(sys.argv) > 1:\n        file_path = sys.argv[1]\n        result = process_race_file(file_path)\n        print(json.dumps(result, indent=2))\n    else:\n        print(\"Usage: python enhanced_race_processor.py <race_file.csv>\")\n        \n        # Test with navigator results file\n        navigator_file = Path('form_guides/navigator_race_results.csv')\n        if navigator_file.exists():\n            print(\"\\n🧪 Testing with navigator results file...\")\n            result = process_race_file(str(navigator_file))\n            print(json.dumps(result, indent=2))
