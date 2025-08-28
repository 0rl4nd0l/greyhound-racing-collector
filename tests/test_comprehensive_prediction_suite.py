#!/usr/bin/env python3
"""
Comprehensive Test Suite for Greyhound Racing Prediction System

This test suite includes:
- Database and data integrity checks
- Model training and validation
- Prediction pipeline verification
- Simulation of predictions with mock data
- Evaluation of model accuracy and drift
"""

import unittest
import logging
import sqlite3
import json
from sklearn.metrics import roc_auc_score
from ml_system_v4 import MLSystemV4

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestGreyhoundPredictionSystem(unittest.TestCase):

    DB_PATH = 'greyhound_racing_data.db'
    
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

    def test_data_integrity(self):
        """Test data integrity and completeness."""
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        # Check race metadata
        cursor.execute("SELECT COUNT(*) FROM race_metadata")
        race_count = cursor.fetchone()[0]
        self.assertGreater(race_count, 0, "No race metadata found.")

        # Check dog race data
        cursor.execute("SELECT COUNT(*) FROM dog_race_data")
        dog_race_count = cursor.fetchone()[0]
        self.assertGreater(dog_race_count, 0, "No dog race data found.")

        conn.close()
        logger.info("✅ Data integrity check passed.")

    def test_ml_model_training(self):
        """Test ML model training process."""
        train_data, test_data = self.ml_system.prepare_time_ordered_data()
        self.assertFalse(train_data.empty, "Training data is empty.")
        self.assertFalse(test_data.empty, "Testing data is empty.")

        features = self.ml_system.build_leakage_safe_features(train_data)
        self.assertFalse(features.empty, "Feature engineering failed.")

        pipeline = self.ml_system.create_sklearn_pipeline(features)
        self.assertIsNotNone(pipeline)
        logger.info("✅ ML model training process completed.")

    def test_prediction_pipeline(self):
        """Test prediction pipeline with mock data."""
        mock_dog = {
            'name': 'Test Dog',
            'box_number': 1,
            'weight': 32.5,
            'starting_price': 2.5,
            'individual_time': 25.5,
            'field_size': 8,
            'temperature': 22.0,
            'humidity': 65.0,
            'wind_speed': 12.0
        }

        prediction_result = self.ml_system.predict(mock_dog)
        self.assertIn('win_probability', prediction_result)
        self.assertIn('confidence', prediction_result)
        logger.info("✅ Prediction pipeline test successful.")

    def test_model_accuracy_and_drift(self):
        """Evaluate model accuracy and check for drift."""
        # Load baseline metrics
        try:
            with open('baseline_metrics.json') as f:
                baseline_metrics = json.load(f)

            test_data, test_labels = self.ml_system.prepare_time_ordered_data(split='test')
            predictions = self.ml_system.evaluate_model(test_data)
            # Calculate current metrics
            current_auc = roc_auc_score(test_labels, predictions)

            # Check drift
            baseline_auc = baseline_metrics['roc_auc']
            drift = abs(baseline_auc - current_auc)
            logger.info(f"Drift: {drift:.3f}")
            self.assertLess(drift, 0.05, "Model drift exceeds acceptable threshold.")
            logger.info("✅ Model accuracy within acceptable drift threshold.")
        except FileNotFoundError:
            self.fail("Baseline metrics file not found.")

if __name__ == '__main__':
    unittest.main(verbosity=2)
