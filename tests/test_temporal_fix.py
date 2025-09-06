#!/usr/bin/env python3
"""
Test script to verify the temporal feature builder fixes
"""

import logging
import sqlite3

import pandas as pd

from temporal_feature_builder import TemporalFeatureBuilder

# Set up logging to see warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_temporal_feature_builder():
    """Test the temporal feature builder with a single race"""

    # Initialize the builder
    builder = TemporalFeatureBuilder("greyhound_racing_data.db")

    # Get a test race
    conn = sqlite3.connect("greyhound_racing_data.db")

    query = """
    SELECT d.*, r.venue, r.grade, r.distance, r.race_date, r.race_time
    FROM dog_race_data d
    LEFT JOIN race_metadata r ON d.race_id = r.race_id
    WHERE d.race_id = ?
    """

    test_race_id = "DAPT_2_22_August_2025"
    race_data = pd.read_sql_query(query, conn, params=[test_race_id])
    conn.close()

    if race_data.empty:
        logger.error("Test race not found!")
        return False

    logger.info(f"Testing with race {test_race_id} containing {len(race_data)} dogs")

    try:
        # Build features for the race
        features_df = builder.build_features_for_race(race_data, test_race_id)

        logger.info(f"✅ Successfully built features for {len(features_df)} dogs")
        logger.info(f"Feature columns: {list(features_df.columns)}")

        # Show a sample of the features
        sample_dog = features_df.iloc[0]
        logger.info(f"Sample features for {sample_dog['dog_clean_name']}:")
        for key, value in sample_dog.items():
            if key.startswith("historical_"):
                logger.info(f"  {key}: {value}")

        return True

    except Exception as e:
        logger.error(f"❌ Error building features: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing temporal feature builder fixes...")
    success = test_temporal_feature_builder()

    if success:
        print("✅ All tests passed! No numpy warnings should have appeared.")
    else:
        print("❌ Tests failed.")
