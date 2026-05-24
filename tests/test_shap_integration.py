#!/usr/bin/env python3
"""
Test SHAP Explainability Integration
===================================

Test script to verify that SHAP explainability has been properly integrated
into the prediction pipeline and ML system.
"""

import sys
from pathlib import Path

import pandas as pd


def test_shap_explainer():
    """Test the SHAP explainer module directly."""
    print("🧪 Testing SHAP Explainer Module...")

    try:
        from shap_explainer import SHAPExplainer, get_shap_values

        print("✅ SHAP explainer imported successfully")

        # Create dummy features for testing
        dummy_features = pd.DataFrame(
            {
                "box_number": [1.0, 2.0, 3.0],
                "weight": [30.5, 31.2, 29.8],
                "starting_price": [5.0, 8.0, 3.5],
                "individual_time": [29.8, 30.1, 29.5],
                "field_size": [8.0, 8.0, 8.0],
                "temperature": [20.0, 20.0, 20.0],
                "humidity": [60.0, 60.0, 60.0],
            }
        )

        print(f"✅ Created test features with shape: {dummy_features.shape}")

        # Test the convenience function
        try:
            shap_result = get_shap_values(dummy_features.iloc[[0]], top_n=5)
            print("✅ SHAP values calculated successfully")
            print(f"   Result keys: {list(shap_result.keys())}")

            if "top_features" in shap_result:
                print(f"   Top features: {list(shap_result['top_features'].keys())}")

            return True

        except Exception as e:
            print(f"⚠️  SHAP calculation failed (expected during initial setup): {e}")
            return False

    except ImportError as e:
        print(f"❌ Failed to import SHAP explainer: {e}")
        return False


def test_ml_system_integration():
    """Test SHAP integration in ML System V3."""
    print("\n🧪 Testing ML System V3 Integration...")

    try:
        from ml_system_v3 import MLSystemV3

        ml_system = MLSystemV3()
        print("✅ ML System V3 imported and initialized")

        # Create test dog data
        test_dog = {
            "name": "Test Dog",
            "box_number": 1,
            "weight": 30.5,
            "starting_price": 5.0,
            "individual_time": 29.8,
            "field_size": 8,
            "temperature": 20.0,
            "humidity": 60.0,
            "wind_speed": 10.0,
        }

        # Test prediction with explainability
        prediction_result = ml_system.predict(test_dog)
        print("✅ Prediction made successfully")
        print(f"   Result keys: {list(prediction_result.keys())}")

        # Check if explainability is present
        if "explainability" in prediction_result:
            explainability = prediction_result["explainability"]
            print("✅ Explainability data found in prediction")
            print(f"   Explainability keys: {list(explainability.keys())}")

            if "error" not in explainability or "top_features" in explainability:
                print("✅ Explainability appears to be working correctly")
                return True
            else:
                print(
                    f"⚠️  Explainability has error: {explainability.get('error', 'Unknown')}"
                )
                return False
        else:
            print("❌ No explainability data found in prediction")
            return False

    except Exception as e:
        print(f"❌ ML System integration test failed: {e}")
        return False


def test_prediction_pipeline_integration():
    """Test SHAP integration in Prediction Pipeline V3."""
    print("\n🧪 Testing Prediction Pipeline V3 Integration...")

    try:
        from prediction_pipeline_v3 import PredictionPipelineV3

        pipeline = PredictionPipelineV3()
        print("✅ Prediction Pipeline V3 imported and initialized")

        # Create a test race file (minimal CSV format)
        test_race_file = Path("test_race.csv")
        test_csv_content = """Dog Name,WGT,SP,TIME
1. Test Dog One,30.5,5.0,29.8
2. Test Dog Two,31.2,8.0,30.1
3. Test Dog Three,29.8,3.5,29.5
"""

        with open(test_race_file, "w") as f:
            f.write(test_csv_content)

        print("✅ Test race file created")

        # Test prediction
        try:
            result = pipeline.predict_race_file(str(test_race_file))
            print("✅ Race prediction completed")
            print(f"   Success: {result.get('success', False)}")

            if result.get("success") and "predictions" in result:
                predictions = result["predictions"]
                print(f"   Number of predictions: {len(predictions)}")

                # Check first prediction for explainability
                if predictions and "explainability" in predictions[0]:
                    explainability = predictions[0]["explainability"]
                    print("✅ Explainability found in prediction results")
                    print(f"   Explainability type: {type(explainability)}")

                    if isinstance(explainability, dict):
                        print(f"   Explainability keys: {list(explainability.keys())}")

                    return True
                else:
                    print("⚠️  No explainability found in prediction results")
                    return False
            else:
                print("⚠️  Prediction failed or no predictions returned")
                return False

        finally:
            # Clean up test file
            if test_race_file.exists():
                test_race_file.unlink()
                print("✅ Test race file cleaned up")

    except Exception as e:
        print(f"❌ Prediction Pipeline integration test failed: {e}")
        return False


def main():
    """Run all SHAP integration tests."""
    print("🚀 SHAP Explainability Integration Test Suite")
    print("=" * 50)

    results = []

    # Test 1: SHAP Explainer Module
    results.append(test_shap_explainer())

    # Test 2: ML System Integration
    results.append(test_ml_system_integration())

    # Test 3: Prediction Pipeline Integration
    results.append(test_prediction_pipeline_integration())

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"   Tests run: {len(results)}")
    print(f"   Passed: {sum(results)}")
    print(f"   Failed: {len(results) - sum(results)}")

    if all(results):
        print("🎉 All tests passed! SHAP explainability integration is working.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
