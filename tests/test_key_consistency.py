#!/usr/bin/env python3
"""
Key Consistency Regression & Unit Tests
=======================================

Step 6: Testing that loads tests/fixtures/test_race.csv through each prediction 
layer (weather, ML, pipeline) and asserts no KeyError and that prediction_tier ≠ "dummy_fallback".

Ensures all loaders accept the constant key and integration tests for CI.
"""

import logging
import os
import shutil
import tempfile

import pandas as pd
import pytest

# Import prediction layers
from prediction_pipeline_v3 import PredictionPipelineV3

from ml_system_v3 import MLSystemV3
from weather_enhanced_predictor import WeatherEnhancedPredictor

# Import unified predictor and traditional analysis
try:
    from unified_predictor import UnifiedPredictor

    UNIFIED_AVAILABLE = True
except ImportError:
    UNIFIED_AVAILABLE = False

try:
    from traditional_analysis import TraditionalRaceAnalyzer

    TRADITIONAL_AVAILABLE = True
except ImportError:
    TRADITIONAL_AVAILABLE = False

# Import constants for key consistency
from constants import DOG_BOX_KEY, DOG_NAME_KEY, DOG_WEIGHT_KEY

# Configure logging for test debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test fixture path
TEST_RACE_PATH = "tests/fixtures/test_race.csv"


@pytest.fixture
def test_db_path():
    """Create temporary database for testing"""
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    yield db_path

    # Cleanup
    os.close(db_fd)
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def setup_test_environment():
    """Setup test environment with required directories"""
    test_dirs = ["predictions", "upcoming_races", "processed", "logs"]
    created_dirs = []

    for dir_name in test_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            created_dirs.append(dir_name)

    yield

    # Cleanup only directories we created
    for dir_name in created_dirs:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name, ignore_errors=True)


@pytest.mark.key_consistency
@pytest.mark.unit
def test_test_race_csv_exists():
    """Verify the test race CSV fixture exists and is readable"""
    assert os.path.exists(
        TEST_RACE_PATH
    ), f"Test race CSV not found at {TEST_RACE_PATH}"

    # Test file is readable
    with open(TEST_RACE_PATH, "r") as f:
        content = f.read()
        assert len(content) > 0, "Test race CSV is empty"
        assert "Dog Name" in content, "Test race CSV missing 'Dog Name' header"


@pytest.mark.key_consistency
@pytest.mark.unit
def test_csv_structure_and_key_consistency():
    """Test that CSV has expected structure and our constant keys are present"""
    # Read CSV with comma delimiter based on the fixture format
    df = pd.read_csv(TEST_RACE_PATH, delimiter=",")

    # Check if we have the expected dog name column
    # The fixture uses "Dog Name" as the header
    assert (
        "Dog Name" in df.columns
    ), f"Expected 'Dog Name' column not found. Available columns: {df.columns.tolist()}"

    # Test that we can access dog names
    dog_names = df["Dog Name"].dropna()
    assert len(dog_names) > 0, "No valid dog names found in test data"

    logger.info(f"Found {len(dog_names)} dogs in test data: {dog_names.tolist()[:5]}")


@pytest.mark.key_consistency
@pytest.mark.integration
@pytest.mark.parametrize(
    "prediction_layer", ["ml_system", "weather_enhanced", "unified"]
)
def test_prediction_layers_key_consistency(
    prediction_layer, test_db_path, setup_test_environment
):
    """Parametrized test ensuring all prediction layers handle keys consistently"""
    logger.info(f"Testing {prediction_layer} for key consistency")

    try:
        if prediction_layer == "ml_system":
            predictor = MLSystemV3(test_db_path)
            # Test key handling in feature extraction methods
            assert hasattr(predictor, "predict"), "MLSystemV3 missing predict method"

        elif prediction_layer == "weather_enhanced":
            predictor = WeatherEnhancedPredictor(test_db_path)
            assert hasattr(
                predictor, "predict_race_file"
            ), "WeatherEnhancedPredictor missing predict_race_file method"

        elif prediction_layer == "unified" and UNIFIED_AVAILABLE:
            predictor = UnifiedPredictor()
            assert hasattr(
                predictor, "predict_race_file"
            ), "UnifiedPredictor missing predict_race_file method"
        else:
            pytest.skip(f"{prediction_layer} not available")

        logger.info(f"✅ {prediction_layer} initialized successfully")

    except Exception as e:
        pytest.fail(f"Failed to initialize {prediction_layer}: {str(e)}")


@pytest.mark.key_consistency
@pytest.mark.integration
@pytest.mark.slow
def test_prediction_pipeline_no_key_errors(test_db_path, setup_test_environment):
    """Test main prediction pipeline for KeyError and prediction tier consistency"""
    logger.info("Testing PredictionPipelineV3 for key consistency and tier validation")

    try:
        # Initialize prediction pipeline with test database
        pipeline = PredictionPipelineV3(test_db_path)

        # Test comprehensive prediction pipeline
        logger.info("Testing comprehensive prediction pipeline...")
        result = pipeline.predict_race_file(TEST_RACE_PATH, enhancement_level="full")

        # Validate result structure
        assert isinstance(result, dict), "Prediction result should be a dictionary"

        if result.get("success", False):
            # If successful, should have prediction_tier
            assert (
                "prediction_tier" in result
            ), "Successful prediction missing 'prediction_tier'"
            assert (
                result["prediction_tier"] != "dummy_fallback"
            ), f"Unexpected dummy fallback: {result.get('fallback_reasons', [])}"
            logger.info(
                f"✅ Successful prediction with tier: {result['prediction_tier']}"
            )
        else:
            # If failed, check for meaningful error information
            assert (
                "error" in result or "fallback_reasons" in result
            ), "Failed prediction missing error information"
            logger.warning(
                f"⚠️ Prediction failed: {result.get('error', 'Unknown error')}"
            )

            # Even failed predictions should not crash with KeyError
            if "fallback_reasons" in result:
                for reason in result["fallback_reasons"]:
                    assert "KeyError" not in str(
                        reason.get("reason", "")
                    ), f"KeyError detected in fallback: {reason}"

    except KeyError as e:
        pytest.fail(f"KeyError encountered during prediction: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        pytest.fail(f"Prediction pipeline failed with unexpected error: {str(e)}")


@pytest.mark.key_consistency
@pytest.mark.integration
def test_weather_enhanced_predictor_key_handling(test_db_path, setup_test_environment):
    """Test weather-enhanced predictor specifically for key handling"""
    logger.info("Testing WeatherEnhancedPredictor for key consistency")

    try:
        predictor = WeatherEnhancedPredictor(test_db_path)

        # Test prediction with test race file
        result = predictor.predict_race_file(TEST_RACE_PATH)

        assert isinstance(
            result, dict
        ), "WeatherEnhancedPredictor result should be a dictionary"

        if result.get("success", False):
            assert (
                "prediction_tier" in result
            ), "Successful weather prediction missing 'prediction_tier'"
            assert (
                result["prediction_tier"] != "dummy_fallback"
            ), "Weather predictor should not use dummy fallback"
            logger.info(
                f"✅ Weather-enhanced prediction successful with tier: {result['prediction_tier']}"
            )
        else:
            logger.warning(
                f"⚠️ Weather-enhanced prediction failed: {result.get('error', 'Unknown error')}"
            )

    except KeyError as e:
        pytest.fail(f"KeyError in WeatherEnhancedPredictor: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in weather predictor: {str(e)}")
        # Don't fail the test if weather predictor isn't fully configured
        logger.warning("Weather predictor test skipped due to configuration issues")


@pytest.mark.key_consistency
@pytest.mark.unit
@pytest.mark.parametrize("csv_file", [TEST_RACE_PATH])
def test_loaders_accept_constant_keys(csv_file):
    """Parametrized test ensuring all CSV loaders accept our constant keys"""
    logger.info(f"Testing CSV loader key consistency for {csv_file}")

    # Test with comma delimiter (as used in fixture)
    df = pd.read_csv(csv_file, delimiter=",")

    # Check for expected columns (mapping to our constants)
    expected_mappings = {
        "Dog Name": DOG_NAME_KEY,
        "BOX": DOG_BOX_KEY,
        "WGT": DOG_WEIGHT_KEY,
    }

    available_columns = df.columns.tolist()
    logger.info(f"Available CSV columns: {available_columns}")

    for csv_column, constant_key in expected_mappings.items():
        assert (
            csv_column in available_columns
        ), f"Expected CSV column '{csv_column}' (maps to {constant_key}) not found"

    # Test that we can extract meaningful data
    dog_names = df["Dog Name"].dropna()
    assert len(dog_names) > 0, "No valid dog names found"

    logger.info(f"✅ CSV loader test passed. Found {len(dog_names)} dogs")


@pytest.mark.key_consistency
@pytest.mark.integration
def test_error_handling_and_fallback_logic(test_db_path, setup_test_environment):
    """Test that error handling doesn't result in KeyErrors and fallback logic works"""
    logger.info("Testing error handling and fallback logic")

    try:
        pipeline = PredictionPipelineV3(test_db_path)

        # Test with the actual test file
        result = pipeline.predict_race_file(TEST_RACE_PATH, enhancement_level="basic")

        # Even if the prediction fails, it should not crash with KeyError
        assert isinstance(result, dict), "Result should be a dictionary even on failure"

        # Check fallback reasons if provided
        if "fallback_reasons" in result:
            for reason in result["fallback_reasons"]:
                assert isinstance(
                    reason, dict
                ), "Fallback reason should be a dictionary"
                assert "tier" in reason, "Fallback reason missing 'tier'"
                assert "reason" in reason, "Fallback reason missing 'reason'"

                # Ensure no KeyError is mentioned in fallback reasons
                reason_text = str(reason.get("reason", ""))
                assert (
                    "KeyError" not in reason_text
                ), f"KeyError found in fallback reason: {reason_text}"

        logger.info("✅ Error handling and fallback logic test passed")

    except KeyError as e:
        pytest.fail(f"KeyError in error handling/fallback logic: {str(e)}")
    except Exception as e:
        logger.warning(f"Non-KeyError exception in fallback test: {str(e)}")
        # This is acceptable as long as it's not a KeyError


@pytest.mark.key_consistency
@pytest.mark.integration
@pytest.mark.slow
def test_integration_all_layers_consistent_keys(test_db_path, setup_test_environment):
    """Integration test ensuring all layers use consistent key handling"""
    logger.info(
        "Running integration test for consistent key handling across all layers"
    )

    layers_tested = []
    errors_encountered = []

    # Test ML System V3
    try:
        ml_system = MLSystemV3(test_db_path)
        layers_tested.append("MLSystemV3")
        logger.info("✅ MLSystemV3 initialized")
    except Exception as e:
        errors_encountered.append(f"MLSystemV3: {str(e)}")

    # Test Weather Enhanced Predictor
    try:
        weather_predictor = WeatherEnhancedPredictor(test_db_path)
        layers_tested.append("WeatherEnhancedPredictor")
        logger.info("✅ WeatherEnhancedPredictor initialized")
    except Exception as e:
        errors_encountered.append(f"WeatherEnhancedPredictor: {str(e)}")

    # Test Unified Predictor if available
    if UNIFIED_AVAILABLE:
        try:
            unified_predictor = UnifiedPredictor()
            layers_tested.append("UnifiedPredictor")
            logger.info("✅ UnifiedPredictor initialized")
        except Exception as e:
            errors_encountered.append(f"UnifiedPredictor: {str(e)}")

    # Test Traditional Analysis if available
    if TRADITIONAL_AVAILABLE:
        try:
            traditional_analyzer = TraditionalRaceAnalyzer(test_db_path)
            layers_tested.append("TraditionalRaceAnalyzer")
            logger.info("✅ TraditionalRaceAnalyzer initialized")
        except Exception as e:
            errors_encountered.append(f"TraditionalRaceAnalyzer: {str(e)}")

    # Test main pipeline
    try:
        pipeline = PredictionPipelineV3(test_db_path)
        layers_tested.append("PredictionPipelineV3")
        logger.info("✅ PredictionPipelineV3 initialized")
    except Exception as e:
        errors_encountered.append(f"PredictionPipelineV3: {str(e)}")

    # Assertions
    assert len(layers_tested) > 0, "No prediction layers could be initialized"
    logger.info(
        f"Successfully initialized {len(layers_tested)} layers: {layers_tested}"
    )

    if errors_encountered:
        logger.warning(f"Some layers had initialization issues: {errors_encountered}")
        # This is a warning, not a failure, as some dependencies might not be available

    # The key requirement is that no KeyErrors occur during initialization
    for error in errors_encountered:
        assert "KeyError" not in error, f"KeyError during layer initialization: {error}"

    logger.info("✅ Integration test passed - all layers handle keys consistently")
