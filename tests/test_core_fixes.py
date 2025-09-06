#!/usr/bin/env python3
"""
Quick Core Fixes Validation Test
================================

Tests the main fixes we implemented:
1. Pipeline V4 data type handling and sklearn compatibility
2. Temporal leakage protection
3. ML System V4 prediction functionality
"""

import traceback

import numpy as np
import pandas as pd


def test_pipeline_v4_prediction():
    """Test Pipeline V4 prediction functionality"""
    print("🧪 Testing Pipeline V4 Prediction Functionality")
    print("-" * 50)

    try:
        from prediction_pipeline_v4 import PredictionPipelineV4

        # Create test data in the expected CSV format
        test_data = pd.DataFrame(
            {
                "Dog Name": ["1. Test Dog 1", "2. Test Dog 2", "3. Test Dog 3"],
                "BOX": [1, 2, 3],
                "WGT": [30.5, 31.0, 29.8],
                "Trainer": ["Trainer A", "Trainer B", "Trainer C"],
                "TRACK": ["DAPT", "RICH", "BAL"],
                "G": ["5", "4", "5"],
                "DIST": [500, 500, 500],
                "DATE": ["04/08/2025", "04/08/2025", "04/08/2025"],
                "Time": ["14:00", "14:00", "14:00"],
            }
        )

        test_file = "./test_core_fixes.csv"
        test_data.to_csv(test_file, index=False)

        pipeline = PredictionPipelineV4()
        result = pipeline.predict_race_file(test_file)

        if result.get("success"):
            predictions = result.get("predictions", [])
            print(
                f"✅ Pipeline V4 prediction successful: {len(predictions)} predictions"
            )
            if predictions:
                top_pick = predictions[0]
                print(f"   🏆 Top pick: {top_pick.get('dog_clean_name', 'Unknown')}")
                print(f"   📊 Win probability: {top_pick.get('win_prob_norm', 0):.3f}")
            return True
        else:
            print(
                f"❌ Pipeline V4 prediction failed: {result.get('error', 'Unknown error')}"
            )
            return False

    except Exception as e:
        print(f"❌ Pipeline V4 test failed: {e}")
        traceback.print_exc()
        return False


def test_temporal_leakage_protection():
    """Test temporal leakage protection"""
    print("\n🛡️ Testing Temporal Leakage Protection")
    print("-" * 50)

    try:
        from temporal_feature_builder import create_temporal_assertion_hook

        test_hook = create_temporal_assertion_hook()

        # Test safe features (should pass)
        safe_features = {
            "box_number": 1,
            "weight": 30.5,
            "distance": 500,
            "historical_avg_position": 3.2,
        }

        try:
            test_hook(safe_features, "test_race", "test_dog")
            print("✅ Safe features correctly passed")
        except AssertionError:
            print("❌ Safe features incorrectly rejected")
            return False

        # Test leaky features (should fail)
        leaky_features = {
            "box_number": 1,
            "weight": 30.5,
            "finish_position": 1,  # Post-race feature
            "individual_time": 29.5,  # Post-race feature
        }

        try:
            test_hook(leaky_features, "test_race", "test_dog")
            print("❌ Temporal protection FAILED - leaky features not detected")
            return False
        except AssertionError as e:
            if "TEMPORAL LEAKAGE DETECTED" in str(e):
                print("✅ Temporal leakage correctly detected")
                return True
            else:
                print(f"❌ Unexpected assertion: {e}")
                return False

    except Exception as e:
        print(f"❌ Temporal leakage test failed: {e}")
        return False


def test_ml_system_v4_normalization():
    """Test ML System V4 probability normalization"""
    print("\n📊 Testing ML System V4 Normalization")
    print("-" * 50)

    try:
        from ml_system_v4 import MLSystemV4

        ml_v4 = MLSystemV4()

        # Test normalization function
        test_probs = np.array([0.1, 0.3, 0.2, 0.4])
        normalized = ml_v4._group_normalize_probabilities(test_probs)

        prob_sum = np.sum(normalized)
        if abs(prob_sum - 1.0) < 0.001:
            print(f"✅ Normalization working correctly (sum: {prob_sum:.6f})")
            print(f"   Input: {test_probs}")
            print(f"   Output: {normalized}")
            return True
        else:
            print(f"❌ Normalization failed - sum: {prob_sum}")
            return False

    except Exception as e:
        print(f"❌ ML System V4 normalization test failed: {e}")
        return False


def test_data_type_handling():
    """Test robust data type handling"""
    print("\n🔧 Testing Data Type Handling")
    print("-" * 50)

    try:
        from ml_system_v4 import MLSystemV4

        # Create test data with mixed types
        test_data = pd.DataFrame(
            {
                "dog_clean_name": ["Test Dog"],
                "box_number": ["1"],  # String that should be numeric
                "weight": [30.5],
                "trainer": ["Test Trainer"],
                "venue": ["UNKNOWN"],
                "grade": ["5"],
                "distance": ["500"],  # String that should be numeric
                "race_date": ["2025-08-04"],
                "race_time": ["14:00"],
            }
        )

        test_data["race_id"] = "test_data_types"

        ml_v4 = MLSystemV4()
        result = ml_v4.predict_race(test_data, "test_data_types")

        if result.get("success"):
            print("✅ Data type handling working correctly")
            print(f"   Predictions generated: {len(result.get('predictions', []))}")
            return True
        else:
            print(f"❌ Data type handling failed: {result.get('error', 'Unknown')}")
            return False

    except Exception as e:
        print(f"❌ Data type handling test failed: {e}")
        return False


def main():
    """Run all core fix validation tests"""
    print("🧪 CORE FIXES VALIDATION TEST SUITE")
    print("=" * 60)

    tests = [
        ("Pipeline V4 Prediction", test_pipeline_v4_prediction),
        ("Temporal Leakage Protection", test_temporal_leakage_protection),
        ("ML System V4 Normalization", test_ml_system_v4_normalization),
        ("Data Type Handling", test_data_type_handling),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")

    print("\n" + "=" * 60)
    print("📋 VALIDATION RESULTS")
    print("=" * 60)
    print(f"✅ Tests passed: {passed}/{total}")
    print(f"📈 Success rate: {(passed/total)*100:.1f}%")

    if passed == total:
        print("🎉 ALL CORE FIXES VALIDATED SUCCESSFULLY!")
        return True
    else:
        print("⚠️ Some fixes need additional work")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
