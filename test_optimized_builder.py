#!/usr/bin/env python3
"""
Simple Test: Optimized Temporal Feature Builder
===============================================

A quick test to verify the optimized temporal feature builder works correctly.
"""

import pandas as pd
import sqlite3
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from temporal_feature_builder_optimized import OptimizedTemporalFeatureBuilder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_optimized_builder():
    """Test the optimized temporal feature builder with real data."""
    logger.info("🧪 Testing Optimized Temporal Feature Builder")
    
    # Initialize the optimized builder
    builder = OptimizedTemporalFeatureBuilder()
    
    # Try to get some test data from the database
    try:
        conn = sqlite3.connect("greyhound_racing_data.db")
        
        # Get a recent race with multiple dogs
        query = """
        SELECT d.*, r.venue, r.grade, r.distance, r.track_condition, r.weather,
               r.temperature, r.humidity, r.wind_speed, r.field_size,
               r.race_date, r.race_time
        FROM dog_race_data d
        LEFT JOIN race_metadata r ON d.race_id = r.race_id
        WHERE r.race_date IS NOT NULL
        ORDER BY r.race_date DESC
        LIMIT 8
        """
        
        race_data = pd.read_sql_query(query, conn)
        conn.close()
        
        if race_data.empty:
            logger.error("No test data found in database")
            return False
        
        logger.info(f"Found test race data with {len(race_data)} dogs")
        logger.info(f"Dogs: {race_data['dog_clean_name'].tolist()}")
        
        # Build features for the race
        race_id = "test_race_optimized"
        features_df = builder.build_features_for_race(race_data, race_id)
        
        logger.info(f"✅ Successfully built features for {len(features_df)} dogs")
        logger.info(f"Feature columns ({len(features_df.columns)}): {list(features_df.columns)[:10]}...")
        
        # Show performance stats
        perf_stats = builder.get_performance_stats()
        logger.info(f"Performance Stats: {perf_stats}")
        
        # Sample some feature values
        sample_dog = features_df.iloc[0]
        logger.info(f"Sample features for '{sample_dog['dog_clean_name']}':")
        for key, value in sample_dog.items():
            if key.startswith('historical_'):
                logger.info(f"  {key}: {value}")
        
        # Validate temporal integrity
        builder.validate_temporal_integrity(features_df, race_data)
        
        logger.info("✅ All tests passed!")
        return True
        
    except FileNotFoundError:
        logger.error("Database file 'greyhound_racing_data.db' not found")
        logger.info("Creating a simple mock test instead...")
        return test_with_mock_data(builder)
    
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


def test_with_mock_data(builder):
    """Test with mock data if real database isn't available."""
    logger.info("🔄 Running test with mock data")
    
    # Create mock race data
    mock_race_data = pd.DataFrame({
        'dog_clean_name': ['TEST DOG A', 'TEST DOG B', 'TEST DOG C'],
        'box_number': [1, 2, 3],
        'weight': [30.0, 31.5, 29.8],
        'venue': ['TEST_VENUE', 'TEST_VENUE', 'TEST_VENUE'],
        'grade': ['5', '5', '5'],
        'distance': [500, 500, 500],
        'race_date': ['2025-01-15', '2025-01-15', '2025-01-15'],
        'race_time': ['14:30', '14:30', '14:30'],
        'trainer_name': ['Trainer A', 'Trainer B', 'Trainer C'],
        'track_condition': ['Good', 'Good', 'Good'],
        'weather': ['Fine', 'Fine', 'Fine'],
        'temperature': [20.0, 20.0, 20.0],
        'humidity': [60.0, 60.0, 60.0],
        'wind_speed': [10.0, 10.0, 10.0],
        'field_size': [3, 3, 3]
    })
    
    try:
        # Build features for the mock race
        race_id = "mock_test_race"
        features_df = builder.build_features_for_race(mock_race_data, race_id)
        
        logger.info(f"✅ Mock test successful - built features for {len(features_df)} dogs")
        logger.info(f"Features generated: {len(features_df.columns)} columns")
        
        # Show performance stats
        perf_stats = builder.get_performance_stats()
        logger.info(f"Performance Stats: {perf_stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"Mock test failed: {e}")
        return False


def main():
    """Run the test."""
    success = test_optimized_builder()
    
    if success:
        print("\n🎉 Optimized Temporal Feature Builder Test: PASSED")
    else:
        print("\n❌ Optimized Temporal Feature Builder Test: FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
