#!/usr/bin/env python3
"""
Database Schema Validator
==========================

This module validates that the test database schema matches the production schema
to prevent SQL errors during testing and development.
"""

import sqlite3
import sys
from pathlib import Path


class DatabaseSchemaValidator:
    """Validates database schema compatibility between test and production."""
    
    def __init__(self):
        self.required_tables = {
            'race_metadata': {
                'required_columns': [
                    'race_id', 'venue', 'race_number', 'race_date', 'race_name', 
                    'grade', 'distance', 'track_condition', 'field_size',
                    'temperature', 'humidity', 'wind_speed', 'wind_direction',
                    'track_record', 'prize_money_total', 'prize_money_breakdown',
                    'race_time', 'extraction_timestamp', 'data_source',
                    'winner_name', 'winner_odds', 'winner_margin', 'race_status',
                    'data_quality_note', 'actual_field_size', 'scratched_count',
                    'scratch_rate', 'box_analysis', 'weather_condition',
                    'precipitation', 'pressure', 'visibility', 'weather_location',
                    'weather_timestamp', 'weather_adjustment_factor',
                    'sportsbet_url', 'venue_slug', 'start_datetime'
                ],
                'primary_key': 'race_id'
            },
            'dog_race_data': {
                'required_columns': [
                    'id', 'race_id', 'dog_name', 'dog_clean_name', 'box_number',
                    'finish_position', 'individual_time', 'weight', 'trainer_name',
                    'odds_decimal', 'starting_price', 'performance_rating',
                    'speed_rating', 'class_rating', 'margin', 'sectional_1st',
                    'sectional_2nd'
                ],
                'foreign_keys': [('race_id', 'race_metadata', 'race_id')]
            }
        }
        
        # Critical queries that must work
        self.critical_queries = [
            """
            SELECT 
                drd.race_id, drd.dog_clean_name, drd.finish_position,
                drd.box_number, drd.weight, drd.starting_price,
                drd.performance_rating, drd.speed_rating, drd.class_rating,
                drd.individual_time, drd.margin,
                rm.field_size, rm.distance, rm.venue, rm.track_condition,
                rm.grade, rm.race_date, rm.race_time,
                rm.weather_condition, rm.temperature, rm.humidity, 
                rm.wind_speed, rm.wind_direction, rm.pressure,
                rm.weather_adjustment_factor
            FROM dog_race_data drd
            JOIN race_metadata rm ON drd.race_id = rm.race_id
            WHERE drd.finish_position IS NOT NULL 
            AND drd.finish_position != ''
            AND drd.finish_position != 'N/A'
            ORDER BY rm.race_date ASC
            """
        ]
    
    def validate_schema(self, db_path):
        """Validate database schema against requirements."""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            validation_results = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'table_info': {}
            }
            
            print(f"🔍 Validating database schema: {db_path}")
            
            # Check each required table
            for table_name, requirements in self.required_tables.items():
                print(f"  📊 Checking table: {table_name}")
                
                # Check if table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name=?
                """, (table_name,))
                
                if not cursor.fetchone():
                    validation_results['valid'] = False
                    validation_results['errors'].append(f"Table '{table_name}' does not exist")
                    continue
                
                # Get table structure
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns_info = cursor.fetchall()
                actual_columns = [col[1] for col in columns_info]
                
                validation_results['table_info'][table_name] = {
                    'columns': actual_columns,
                    'column_details': columns_info
                }
                
                # Check required columns
                missing_columns = []
                for required_col in requirements['required_columns']:
                    if required_col not in actual_columns:
                        missing_columns.append(required_col)
                
                if missing_columns:
                    validation_results['valid'] = False
                    validation_results['errors'].append(
                        f"Table '{table_name}' missing columns: {missing_columns}"
                    )
                
                print(f"    ✅ Table '{table_name}': {len(actual_columns)} columns")
            
            # Test critical queries
            print(f"  🧪 Testing critical queries...")
            for i, query in enumerate(self.critical_queries, 1):
                try:
                    cursor.execute(query)
                    results = cursor.fetchall()
                    print(f"    ✅ Query {i}: {len(results)} records returned")
                except Exception as e:
                    validation_results['valid'] = False
                    validation_results['errors'].append(f"Critical query {i} failed: {e}")
                    print(f"    ❌ Query {i} failed: {e}")
            
            conn.close()
            
            # Summary
            if validation_results['valid']:
                print("✅ Database schema validation PASSED")
            else:
                print("❌ Database schema validation FAILED")
                for error in validation_results['errors']:
                    print(f"   ERROR: {error}")
            
            return validation_results
            
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Schema validation failed: {e}"],
                'warnings': [],
                'table_info': {}
            }
    
    def validate_test_schema(self):
        """Validate test database schema setup."""
        print("🧪 TESTING DATABASE SCHEMA VALIDATION")
        print("=" * 50)
        
        # Test with conftest setup
        try:
            import tempfile
            import os
            from tests.conftest import setup_test_data
            
            # Create temporary test database
            db_fd, db_path = tempfile.mkstemp(suffix='.db')
            try:
                setup_test_data(db_path)
                
                # Validate schema
                validation_results = self.validate_schema(db_path)
                
                return validation_results
                
            finally:
                os.close(db_fd)
                if os.path.exists(db_path):
                    os.unlink(db_path)
                    
        except Exception as e:
            print(f"❌ Test schema validation failed: {e}")
            return {'valid': False, 'errors': [str(e)], 'warnings': [], 'table_info': {}}


def main():
    """Main validation function."""
    validator = DatabaseSchemaValidator()
    
    if len(sys.argv) > 1:
        # Validate specific database file
        db_path = sys.argv[1]
        if not Path(db_path).exists():
            print(f"❌ Database file not found: {db_path}")
            sys.exit(1)
        
        results = validator.validate_schema(db_path)
    else:
        # Validate test schema setup
        results = validator.validate_test_schema()
    
    # Exit with error code if validation failed
    if not results['valid']:
        sys.exit(1)


if __name__ == "__main__":
    main()
