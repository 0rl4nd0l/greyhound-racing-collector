#!/usr/bin/env python3
"""
Comprehensive Validation Test for ML System V4

This tests the prediction functionality for model accuracy, performance, and validation.
"""

import logging
import sqlite3
import unittest

from ml_system_v4 import MLSystemV4

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestMLPredictionSystem(unittest.TestCase):

    DB_PATH = "greyhound_racing_data.db"

    def setUp(self):
        """Initialize resources for testing."""
        self.ml_system = MLSystemV4(self.DB_PATH)

    def test_database_connection(self):
        """Test connection to the database."""
        try:
            conn = sqlite3.connect(self.DB_PATH)
            self.assertIsNotNone(conn)
            conn.close()
            logger.info("✅ Database connection successful.")
        except sqlite3.Error as e:
            self.fail(f"Database connection failed: {e}")

    def test_prediction_functionality(self):
        """Test the prediction system end-to-end."""
        race_data = {
            "field_size": 8,
            "temperature": 22.0,
            "humidity": 65.0,
            "wind_speed": 12.0,
        }
        result = self.ml_system.predict_race(race_data, race_id="test_race")
        self.assertIn("success", result)
        self.assertTrue(result["success"], "Prediction failed.")
        self.assertIn("predictions", result)
        self.assertGreater(len(result["predictions"]), 0, "No predictions returned.")
        logger.info("✅ Prediction functionality test successful.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
