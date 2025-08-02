import unittest
from prediction_pipeline_v3 import PredictionPipelineV3
from prediction_v3_diagnostic import create_test_race_file

class TestPredictionPipelineV3(unittest.TestCase):
    def setUp(self):
        # Create a test race file
        self.test_race_file = create_test_race_file()
        # Initialize the prediction pipeline
        self.pipeline = PredictionPipelineV3()

    def test_prediction_response_time_and_shape(self):
        """Test the prediction response time and validate output shape"""
        # Run the prediction
        result = self.pipeline.predict_race_file(self.test_race_file)

        # Check if the result is successful
        self.assertTrue(result['success'], "Prediction failed")

        # Validate response shape
        self.assertIn('predictions', result, "Missing predictions in result")
        self.assertIsInstance(result['predictions'], list, "Predictions should be a list")

        # Validate individual prediction shape
        for pred in result['predictions']:
            self.assertIn('dog_name', pred, "Missing dog_name in prediction")
            self.assertIn('win_probability', pred, "Missing win_probability in prediction")

        # Test for finite response time
        import time
        start_time = time.time()
        self.pipeline.predict_race_file(self.test_race_file)
        end_time = time.time()

        response_time = end_time - start_time
        self.assertLessEqual(response_time, 5, "Prediction should complete within 5 seconds")

if __name__ == '__main__':
    unittest.main()
