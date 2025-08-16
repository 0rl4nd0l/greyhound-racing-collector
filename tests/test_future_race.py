
import unittest
from datetime import datetime, timedelta
import pandas as pd
from ml_system_v4 import MLSystemV4
import logging

# Set up logging to see debug messages
logging.basicConfig(level=logging.DEBUG)

class TestTemporalLeakage(unittest.TestCase):

    def setUp(self):
        self.ml_system = MLSystemV4(db_path="greyhound_racing_data.db")

    def test_future_race_prediction(self):
        """Test that predict_race raises AssertionError for future race dates."""
        # Create a dummy race DataFrame for a future date
        # Use the format that matches the date format expected by temporal assertion
        future_date = (datetime.now() + timedelta(days=1)).strftime('%d %B %Y')
        
        race_data = pd.DataFrame({
            'dog_clean_name': ['Test Dog'],
            'race_date': [future_date],
            'venue': ['Test Venue'],
            'distance': [500],
            'grade': [5],
            'box_number': [1],
            'weight': [30.0],
            'race_time': ['12:00'],
            'track_condition': ['Good'],
            'weather': ['Fine'],
            'trainer_name': ['Test Trainer']
        })
        race_id = f'test_race_{future_date.replace(" ", "_")}'
        
        print(f"Testing with future race date: {future_date}")
        print(f"Current date: {datetime.now().date()}")

        # Expect prediction to fail due to temporal leakage (future race)
        result = self.ml_system.predict_race(race_data, race_id)
        
        # Verify that prediction failed due to temporal leakage
        self.assertFalse(result.get('success', True), "Prediction should have failed for future race")
        
        # Verify the error message contains temporal leakage detection
        error_message = result.get('error', '')
        self.assertIn("TEMPORAL LEAKAGE DETECTED", error_message)
        self.assertIn("is in the future", error_message)
        
        print(f"\u2705 Test passed! Temporal leakage detected correctly:")
        print(f"   Success: {result.get('success')}")
        print(f"   Error: {result.get('error')}")

if __name__ == '__main__':
    unittest.main()

