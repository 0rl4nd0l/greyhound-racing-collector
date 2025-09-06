#!/usr/bin/env python3
"""
Quick test script for the ML system with minimal training data
"""

import logging
import sqlite3

import pandas as pd

from ml_system_v4 import MLSystemV4

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_quick_training():
    """Test ML system with minimal training data"""

    system = MLSystemV4("greyhound_racing_data.db")

    # Override the load_training_data method to use minimal data
    original_load = system.load_training_data

    def quick_load_training_data():
        """Load just a few races for quick testing"""
        logger.info("Loading minimal training data for testing...")

        try:
            conn = sqlite3.connect(system.db_path)

            # Get just 3 races for quick testing
            query = """
            SELECT DISTINCT r.race_id
            FROM race_metadata r 
            INNER JOIN dog_race_data d ON r.race_id = d.race_id
            WHERE r.race_date IS NOT NULL 
                AND d.finish_position IS NOT NULL
                AND r.race_date < date('now', '-30 days')
            ORDER BY r.race_date DESC
            LIMIT 3
            """

            race_ids_df = pd.read_sql_query(query, conn)
            conn.close()

            if race_ids_df.empty:
                raise ValueError("No valid training races found")

            logger.info(f"Found {len(race_ids_df)} training races for quick test")
            return race_ids_df

        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise

    # Temporarily replace the method
    system.load_training_data = quick_load_training_data

    try:
        logger.info("🚀 Starting quick ML system test...")

        # Train the model
        logger.info("Training model with minimal data...")
        success = system.train_model()

        if not success:
            logger.error("Training failed!")
            return False

        logger.info("✅ Training completed successfully!")

        # Test prediction on a race
        logger.info("Testing prediction...")

        # Get a test race
        conn = sqlite3.connect("greyhound_racing_data.db")
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT race_id FROM dog_race_data LIMIT 1")
        test_race_id = cursor.fetchone()[0]
        conn.close()

        predictions = system.predict_race(test_race_id)

        if predictions is not None and len(predictions) > 0:
            logger.info(f"✅ Predictions generated for {len(predictions)} dogs!")
            logger.info("Top 3 predictions:")
            for i, (_, row) in enumerate(predictions.head(3).iterrows()):
                logger.info(
                    f"  {i+1}. {row['dog_clean_name']}: {row['win_probability']:.3f}"
                )
            return True
        else:
            logger.error("No predictions generated!")
            return False

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False
    finally:
        # Restore original method
        system.load_training_data = original_load


if __name__ == "__main__":
    print("🧪 Testing ML system with quick training...")
    success = test_quick_training()

    if success:
        print("✅ Quick ML system test passed!")
    else:
        print("❌ Quick ML system test failed.")
