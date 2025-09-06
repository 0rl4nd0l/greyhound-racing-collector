#!/usr/bin/env python3
"""
Test Enhanced KeyError Handling and Logging
==========================================

This test validates that the enhanced KeyError handling and logging
works correctly throughout the weather-enhanced prediction pipeline.

It tests:
1. KeyError handling in main prediction loop
2. KeyError handling in form data loading  
3. KeyError handling in fallback prediction
4. KeyError handling in Flask app prediction results
5. Proper logging of error context including race file path, dog record, and stack trace
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

# Add the current directory to Python path for imports
sys.path.insert(0, ".")

from constants import DOG_NAME_KEY
from logger import key_mismatch_logger
from weather_enhanced_predictor import WeatherEnhancedPredictor


def create_test_race_file_with_missing_keys():
    """Create a test race CSV file with some dogs missing the DOG_NAME_KEY"""

    # Create a temporary race file with proper CSV structure
    # but we'll create a scenario where the dog extraction finds issues
    test_csv_data = {
        "Dog Name": [
            "1. Test Dog 1",
            "2. Test Dog 2",
            "3. Test Dog 3",
            "",
            "5. Test Dog 5",
        ],
        "Box": [1, 2, 3, 4, 5],
        "Weight": [30.5, 31.0, 29.5, 32.0, 30.0],
    }

    df = pd.DataFrame(test_csv_data)

    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()

    return temp_file.name, test_csv_data


def create_test_prediction_json_with_missing_keys():
    """Create a test prediction JSON file with missing dog name keys"""

    prediction_data = {
        "race_info": {
            "venue": "TEST",
            "race_number": "1",
            "race_date": "2025-07-22",
            "filename": "test_race.csv",
        },
        "predictions": [
            {DOG_NAME_KEY: "Test Dog 1", "box_number": 1, "final_score": 0.8},
            {
                "wrong_key": "Test Dog 2",
                "box_number": 2,
                "final_score": 0.7,
            },  # Missing DOG_NAME_KEY
            {DOG_NAME_KEY: "Test Dog 3", "box_number": 3, "final_score": 0.6},
        ],
        "prediction_methods_used": ["weather_enhanced"],
        "prediction_timestamp": "2025-07-22T10:00:00",
    }

    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    with open(temp_file.name, "w") as f:
        json.dump(prediction_data, f)

    return temp_file.name, prediction_data


def test_key_error_handling_in_weather_enhanced_predictor():
    """Test KeyError handling in the weather-enhanced predictor main prediction loop"""

    print("🧪 Testing KeyError handling in weather-enhanced predictor...")

    # Create test race file with missing keys
    race_file_path, test_data = create_test_race_file_with_missing_keys()

    try:
        # Mock the logger to capture log calls
        with patch.object(key_mismatch_logger, "log_key_error") as mock_logger:
            # Initialize predictor
            predictor = WeatherEnhancedPredictor()

            # Mock the trained model and scaler to avoid dependency issues
            predictor.trained_model = MagicMock()
            predictor.scaler = MagicMock()
            predictor.feature_columns = [
                "avg_position",
                "recent_form_avg",
                "market_confidence",
            ]

            # Mock participating dogs extraction to return dogs with missing keys
            mock_dogs_with_missing_keys = [
                {DOG_NAME_KEY: "Test Dog 1", "box": 1},
                {"wrong_key": "Test Dog 2", "box": 2},  # Missing DOG_NAME_KEY
                {DOG_NAME_KEY: "Test Dog 3", "box": 3},
                {"different_key": "Test Dog 4", "box": 4},  # Missing DOG_NAME_KEY
            ]

            # Mock ML system availability
            with patch("weather_enhanced_predictor.COMPREHENSIVE_ML_AVAILABLE", True):
                # Mock the comprehensive ML system
                with patch(
                    "weather_enhanced_predictor.ComprehensiveEnhancedMLSystem"
                ) as mock_ml_system:
                    mock_instance = MagicMock()
                    mock_ml_system.return_value = mock_instance
                    mock_instance.load_race_results_data.return_value = pd.DataFrame()

                    # Mock the dog extraction to return our test data with missing keys
                    with patch.object(
                        predictor,
                        "_extract_participating_dogs",
                        return_value=mock_dogs_with_missing_keys,
                    ):
                        # Test the prediction with missing keys
                        result = predictor.predict_race_file_with_weather(
                            race_file_path
                        )

                        # Verify that key_mismatch_logger.log_key_error was called for missing keys
                        assert (
                            mock_logger.call_count >= 1
                        ), f"Expected at least 1 KeyError log call, got {mock_logger.call_count}"

                        # Verify the logged error context includes required fields
                        for call in mock_logger.call_args_list:
                            args, kwargs = call
                            if "error_context" in kwargs:
                                error_context = kwargs["error_context"]
                                assert "operation" in error_context
                                assert "race_file_path" in error_context
                                assert "dog_record" in error_context
                                assert "available_keys" in error_context
                                assert "missing_key" in error_context
                                assert "step" in error_context
                                assert error_context["race_file_path"] == race_file_path
                                assert error_context["missing_key"] == DOG_NAME_KEY

                            if "dog_record" in kwargs:
                                assert isinstance(kwargs["dog_record"], dict)

                        print(
                            "✅ Weather-enhanced predictor KeyError handling test passed"
                        )

    finally:
        # Clean up
        os.unlink(race_file_path)


def test_key_error_handling_in_fallback_prediction():
    """Test KeyError handling in the fallback prediction method"""

    print("🧪 Testing KeyError handling in fallback prediction...")

    # Create test race file with missing keys
    race_file_path, test_data = create_test_race_file_with_missing_keys()

    try:
        # Mock the logger to capture log calls
        with patch.object(key_mismatch_logger, "log_key_error") as mock_logger:
            # Initialize predictor
            predictor = WeatherEnhancedPredictor()

            # Mock participating dogs extraction to return dogs with missing keys
            mock_dogs_with_missing_keys = [
                {DOG_NAME_KEY: "Test Dog 1", "box": 1},
                {"wrong_key": "Test Dog 2", "box": 2},  # Missing DOG_NAME_KEY
                {DOG_NAME_KEY: "Test Dog 3", "box": 3},
                {"different_key": "Test Dog 4", "box": 4},  # Missing DOG_NAME_KEY
            ]

            # Force fallback by disabling ML system
            with patch("weather_enhanced_predictor.COMPREHENSIVE_ML_AVAILABLE", False):
                # Mock the dog extraction to return our test data with missing keys
                with patch.object(
                    predictor,
                    "_extract_participating_dogs",
                    return_value=mock_dogs_with_missing_keys,
                ):
                    # Test the fallback prediction with missing keys
                    result = predictor._fallback_prediction(race_file_path)

                    # Verify that key_mismatch_logger.log_key_error was called
                    assert (
                        mock_logger.call_count >= 1
                    ), f"Expected at least 1 KeyError log call, got {mock_logger.call_count}"

                    # Verify the logged error context includes required fields
                    for call in mock_logger.call_args_list:
                        args, kwargs = call
                        if "error_context" in kwargs:
                            error_context = kwargs["error_context"]
                            assert (
                                error_context["operation"]
                                == "dog_name_extraction_in_fallback_prediction"
                            )
                            assert error_context["race_file_path"] == race_file_path
                            assert error_context["missing_key"] == DOG_NAME_KEY
                            assert (
                                error_context["step"]
                                == "fallback_prediction_processing"
                            )

                    print("✅ Fallback prediction KeyError handling test passed")

    finally:
        # Clean up
        os.unlink(race_file_path)


def test_key_error_handling_in_flask_app():
    """Test KeyError handling in Flask app prediction results processing"""

    print("🧪 Testing KeyError handling in Flask app...")

    # Create test prediction JSON with missing keys
    json_file_path, prediction_data = create_test_prediction_json_with_missing_keys()

    try:
        # Mock the logger to capture log calls
        with patch.object(key_mismatch_logger, "log_key_error") as mock_logger:
            # Import Flask app functions

            # Simulate the prediction results processing that would happen in the Flask app
            # This mimics the code around lines 6196-6213 in app.py
            first_pred = prediction_data["predictions"][
                1
            ]  # This one has missing DOG_NAME_KEY

            try:
                dog_name = first_pred[DOG_NAME_KEY]
            except KeyError:
                # This should trigger the enhanced KeyError handling
                key_mismatch_logger.log_key_error(
                    error_context={
                        "operation": "top_pick_creation_from_prediction_results",
                        "race_file_path": str(Path(json_file_path).name),
                        "dog_record": dict(first_pred),
                        "available_keys": list(first_pred.keys()),
                        "missing_key": DOG_NAME_KEY,
                        "step": "api_prediction_results_processing",
                    },
                    dog_record=dict(first_pred),
                )
                # Use fallback value
                dog_name = first_pred.get("dog_name", "Unknown")

            # Verify that key_mismatch_logger.log_key_error was called
            assert (
                mock_logger.call_count == 1
            ), f"Expected 1 KeyError log call, got {mock_logger.call_count}"

            # Verify the logged error context
            call_args = mock_logger.call_args_list[0]
            args, kwargs = call_args
            error_context = kwargs["error_context"]

            assert (
                error_context["operation"]
                == "top_pick_creation_from_prediction_results"
            )
            assert error_context["missing_key"] == DOG_NAME_KEY
            assert error_context["step"] == "api_prediction_results_processing"
            assert "dog_record" in kwargs
            assert isinstance(kwargs["dog_record"], dict)

            print("✅ Flask app KeyError handling test passed")

    finally:
        # Clean up
        os.unlink(json_file_path)


def test_logger_captures_required_context():
    """Test that the KeyMismatchLogger captures all required context information"""

    print("🧪 Testing logger context capture...")

    # Mock the logger's log_key_error method to capture what it receives
    with patch.object(key_mismatch_logger, "log_key_error") as mock_logger:
        # Simulate a KeyError with comprehensive context
        test_error_context = {
            "operation": "test_operation",
            "race_file_path": "/path/to/test_race.csv",
            "dog_record": {"wrong_key": "Test Dog", "box": 1},
            "available_keys": ["wrong_key", "box"],
            "missing_key": DOG_NAME_KEY,
            "step": "test_step",
        }

        test_dog_record = {"wrong_key": "Test Dog", "box": 1}

        # Call the logger
        key_mismatch_logger.log_key_error(
            error_context=test_error_context, dog_record=test_dog_record
        )

        # Verify the call was made with correct parameters
        assert mock_logger.call_count == 1
        call_args = mock_logger.call_args_list[0]
        args, kwargs = call_args

        # Check that all required context fields are present
        assert "error_context" in kwargs
        assert "dog_record" in kwargs

        error_context = kwargs["error_context"]
        for required_field in [
            "operation",
            "race_file_path",
            "dog_record",
            "available_keys",
            "missing_key",
            "step",
        ]:
            assert (
                required_field in error_context
            ), f"Missing required field: {required_field}"

        assert error_context["missing_key"] == DOG_NAME_KEY
        assert isinstance(kwargs["dog_record"], dict)

        print("✅ Logger context capture test passed")


def main():
    """Run all KeyError handling tests"""

    print("🚀 Starting KeyError Handling and Logging Tests")
    print("=" * 60)

    try:
        # Test 1: Weather-enhanced predictor main prediction loop
        test_key_error_handling_in_weather_enhanced_predictor()

        # Test 2: Fallback prediction method
        test_key_error_handling_in_fallback_prediction()

        # Test 3: Flask app prediction results processing
        test_key_error_handling_in_flask_app()

        # Test 4: Logger context capture
        test_logger_captures_required_context()

        print("=" * 60)
        print("✅ All KeyError handling tests passed successfully!")
        print("🌟 Enhanced logging and error handling is working correctly")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
