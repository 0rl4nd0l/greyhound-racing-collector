#!/usr/bin/env python3
"""
Focused Temporal Leakage Test
=============================

This test specifically focuses on identifying and debugging the temporal leakage
protection issues in the ML prediction system.
"""

import logging
import sys
import traceback

import pandas as pd

# Enable debug logging to see where the error occurs
logging.basicConfig(
    level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s"
)


def test_temporal_leakage_protection():
    """Test focused on temporal leakage protection"""
    print("🧪 FOCUSED TEMPORAL LEAKAGE PROTECTION TEST")
    print("=" * 60)

    try:
        # Create a simple test race CSV
        test_data = pd.DataFrame(
            {
                "dog_clean_name": ["Test Dog", "Test Dog 2"],
                "box_number": [1, 2],
                "weight": [30.5, 31.0],
                "trainer": ["Test Trainer", "Test Trainer 2"],
                "venue": ["UNKNOWN", "UNKNOWN"],
                "grade": ["5", "5"],
                "distance": [500, 500],
                "race_date": ["2025-08-04", "2025-08-04"],
                "race_time": ["14:00", "14:00"],
            }
        )

        test_file = "./test_temporal_file.csv"
        test_data.to_csv(test_file, index=False)

        print(f"✅ Created test file: {test_file}")

        # Test with ML System V4 directly
        print("\n1️⃣ Testing ML System V4 Temporal Protection")
        print("-" * 50)

        from ml_system_v4 import MLSystemV4

        ml_v4 = MLSystemV4()

        # Test prediction with temporal protection
        race_id = "test_temporal_race"

        try:
            # Read the test data and prepare for prediction
            prediction_data = pd.read_csv(test_file)
            prediction_data["race_id"] = race_id

            print(f"🔧 Attempting prediction for race: {race_id}")
            print(f"📊 Data shape: {prediction_data.shape}")
            print(f"📊 Columns: {list(prediction_data.columns)}")

            # Use predict_race method which includes temporal protection
            result = ml_v4.predict_race(prediction_data, race_id)

            if result.get("success"):
                print("✅ Temporal protection test PASSED")
                print(f"📊 Predictions: {len(result.get('predictions', []))}")
            else:
                print(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
                print(f"🔍 Full result: {result}")

        except AssertionError as e:
            if "TEMPORAL LEAKAGE DETECTED" in str(e):
                print(f"🛡️ Temporal leakage protection WORKING - caught: {e}")
                return True
            else:
                print(f"❌ Unexpected assertion error: {e}")
                return False

        except Exception as e:
            print(f"❌ Unexpected error during prediction: {e}")
            print("🔍 Full traceback:")
            traceback.print_exc()
            return False

        # Test 2: Try to trigger leakage protection by adding post-race features
        print("\n2️⃣ Testing Leakage Detection with Post-Race Features")
        print("-" * 50)

        try:
            # Test the temporal assertion hook directly
            from temporal_feature_builder import create_temporal_assertion_hook

            test_hook = create_temporal_assertion_hook()

            # Test 1: Should pass with safe features
            safe_features = {
                "box_number": 1,
                "weight": 30.5,
                "distance": 500,
                "historical_avg_position": 3.2,
            }

            try:
                test_hook(safe_features, "test_race", "test_dog")
                print("✅ Safe features passed assertion hook")
            except AssertionError as e:
                print(f"❌ Safe features incorrectly rejected: {e}")
                return False

            # Test 2: Should fail with post-race features
            leaky_features = {
                "box_number": 1,
                "weight": 30.5,
                "finish_position": 1,  # This should trigger protection
                "individual_time": 29.5,  # This should trigger protection
            }

            try:
                test_hook(leaky_features, "test_race_leaky", "test_dog")
                print("⚠️ Leakage protection FAILED - post-race features not detected")
                return False
            except AssertionError as e:
                if "TEMPORAL LEAKAGE DETECTED" in str(e):
                    print(f"✅ Leakage protection WORKING - correctly detected: {e}")
                else:
                    print(f"❌ Unexpected assertion: {e}")
                    return False

            # Test 3: Should fail with disabled features (odds)
            odds_features = {
                "box_number": 1,
                "weight": 30.5,
                "odds": 3.5,  # This should trigger protection
                "starting_price": 4.0,  # This should trigger protection
            }

            try:
                test_hook(odds_features, "test_race_odds", "test_dog")
                print("⚠️ Odds protection FAILED - odds features not detected")
                return False
            except AssertionError as e:
                if "TEMPORAL LEAKAGE DETECTED" in str(e):
                    print(f"✅ Odds protection WORKING - correctly detected: {e}")
                else:
                    print(f"❌ Unexpected assertion: {e}")
                    return False

        except Exception as e:
            print(f"❌ Unexpected error during leakage test: {e}")
            traceback.print_exc()
            return False

        # Test 3: Feature Builder validation
        print("\n3️⃣ Testing Temporal Feature Builder Validation")
        print("-" * 50)

        try:
            from temporal_feature_builder import TemporalFeatureBuilder

            builder = TemporalFeatureBuilder()

            # Build features and validate
            features = builder.build_features_for_race(
                prediction_data, race_id + "_features"
            )

            print(f"📊 Built features shape: {features.shape}")
            print(f"📊 Feature columns: {list(features.columns)}")

            # Validate temporal integrity
            builder.validate_temporal_integrity(features, prediction_data)

            print("✅ Feature builder temporal validation PASSED")

        except Exception as e:
            print(f"❌ Feature builder test failed: {e}")
            traceback.print_exc()
            return False

        print("\n🎉 ALL TEMPORAL LEAKAGE TESTS PASSED")
        return True

    except Exception as e:
        print(f"❌ Temporal leakage test failed with unexpected error: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_temporal_leakage_protection()
    sys.exit(0 if success else 1)
