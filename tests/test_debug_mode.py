#!/usr/bin/env python3
"""
Test Debug Mode Implementation
=============================

Simple test to demonstrate debug logging capabilities
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Set debug mode via environment variable
os.environ['DEBUG'] = '1'

# Import our enhanced logger and CSV ingestion
from logger import logger
from csv_ingestion import create_ingestor, FormGuideCsvIngestionError

def create_test_csv():
    """Create a test CSV file to demonstrate parsing with debug logs"""
    test_data = {
        'Dog Name': ['1. Speedy Sam', '', '2. Fast Fido', '', '3. Quick Quinn'],
        'PLC': [1, 2, 1, 3, 2],
        'BOX': [1, 1, 2, 2, 3],
        'DIST': [500, 500, 500, 500, 500],
        'DATE': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04', '2025-01-05'],
        'TRACK': ['TrackA', 'TrackA', 'TrackB', 'TrackB', 'TrackC']
    }
    
    df = pd.DataFrame(test_data)
    test_file = Path('./test_race_data.csv')
    df.to_csv(test_file, index=False)
    return test_file

def test_debug_logging():
    """Test debug logging during CSV parsing"""
    print("🧪 Testing Debug Mode Implementation")
    print("=" * 50)
    
    # Show debug mode status
    debug_status = "🐛 ENABLED" if logger.debug_mode else "DISABLED"
    print(f"🔍 Debug mode: {debug_status}")
    
    if not logger.debug_mode:
        print("❌ Debug mode not enabled! Try running with --debug flag or DEBUG=1")
        return
    
    # Create test CSV
    test_file = create_test_csv()
    print(f"📁 Created test CSV: {test_file}")
    
    # Test CSV ingestion with debug logging
    try:
        print("\n🎯 Starting CSV ingestion with debug logging...")
        
        # Create ingestor
        ingestor = create_ingestor("moderate")
        
        # Add debug logging to the ingestor
        if hasattr(logger, 'debug_mode') and logger.debug_mode:
            print("🔍 Debug mode detected in logger")
            
            # Log race ID and parse details
            race_id = "TEST_RACE_001"
            logger.debug_logger.debug(f"🏁 Starting to parse race: {race_id}")
            
            # Read and count dogs
            df = pd.read_csv(test_file)
            dog_count = len(df[df['Dog Name'].str.contains(r'^\d+\.', na=False)])
            logger.debug_logger.debug(f"🐕 Race {race_id}: {dog_count} dogs detected")
            
            # Check if dog count deviates from expected (8 is typical)
            expected_dogs = 8
            if dog_count != expected_dogs:
                logger.debug_logger.debug(f"⚠️ Race {race_id}: Dog count deviation detected! Expected {expected_dogs}, found {dog_count}")
                
            # Simulate forward-fill detection
            empty_dog_names = df['Dog Name'].isna().sum() + (df['Dog Name'] == '').sum()
            if empty_dog_names > 0:
                logger.debug_logger.debug(f"🔄 Race {race_id}: Forward-fill used for {empty_dog_names} rows (greyhound form guide format)")
        
        # Actually ingest the CSV
        processed_data, validation_result = ingestor.ingest_csv(test_file)
        
        print(f"✅ Successfully processed {len(processed_data)} records")
        print("\n📋 Sample processed records:")
        for i, record in enumerate(processed_data[:3]):
            print(f"  Record {i+1}: {record}")
            
        # Show validation results
        if validation_result.warnings:
            print("\n⚠️ Validation warnings:")
            for warning in validation_result.warnings:
                print(f"  - {warning}")
                
    except FormGuideCsvIngestionError as e:
        logger.log_error("CSV ingestion failed", error=e)
        print(f"❌ CSV ingestion failed: {e}")
        
    except Exception as e:
        logger.log_error("Unexpected error during testing", error=e)
        print(f"❌ Unexpected error: {e}")
        
    finally:
        # Clean up test file
        if test_file.exists():
            test_file.unlink()
            print(f"🧹 Cleaned up test file: {test_file}")
    
    # Show debug log location
    print(f"\n📋 Debug logs written to: {logger.debug_log_file}")
    if logger.debug_log_file.exists():
        print("📖 Recent debug log entries:")
        with open(logger.debug_log_file, 'r') as f:
            lines = f.readlines()
            for line in lines[-5:]:  # Show last 5 lines
                print(f"  {line.strip()}")

if __name__ == "__main__":
    test_debug_logging()
