#!/usr/bin/env python3
"""
Test Calibration Integration
============================

Test script to verify probability calibration is properly integrated
into the prediction pipelines.
"""

import logging
from probability_calibrator import ProbabilityCalibrator, apply_probability_calibration
from ml_system_v3 import MLSystemV3

logging.basicConfig(level=logging.INFO)

def test_calibrator_standalone():
    """Test the probability calibrator standalone functionality."""
    print("🧪 Testing Probability Calibrator Standalone...")
    
    calibrator = ProbabilityCalibrator()
    
    # Test calibration with sample probabilities
    test_cases = [
        (0.05, 0.15),  # Low probabilities
        (0.15, 0.35),  # Medium probabilities  
        (0.35, 0.65),  # Higher probabilities
        (0.75, 0.90),  # Very high probabilities
    ]
    
    for raw_win, raw_place in test_cases:
        result = calibrator.calibrate_probs(raw_win, raw_place)
        print(f"  Raw: win={raw_win:.3f}, place={raw_place:.3f}")
        print(f"  Calibrated: win={result['calibrated_win_prob']:.3f}, place={result['calibrated_place_prob']:.3f}")
        print(f"  Applied: win={result['win_calibration_applied']}, place={result['place_calibration_applied']}")
        print()
    
    print("✅ Standalone calibrator test complete\n")


def test_ml_integration():
    """Test ML System V3 integration with calibration."""
    print("🧪 Testing ML System V3 Integration...")
    
    ml_system = MLSystemV3()
    
    # Test prediction with sample dog data
    sample_dog = {
        "name": "Test Dog",
        "box_number": 3,
        "weight": 30.5,
        "starting_price": 4.50,
        "individual_time": 29.8,
        "field_size": 8,
        "temperature": 22.0,
        "humidity": 65.0,
        "wind_speed": 12.0
    }
    
    try:
        prediction = ml_system.predict(sample_dog)
        
        print("  Prediction Results:")
        win_prob = prediction.get('win_probability', 'N/A')
        place_prob = prediction.get('place_probability', 'N/A')
        raw_win_prob = prediction.get('raw_win_probability', 'N/A')
        raw_place_prob = prediction.get('raw_place_probability', 'N/A')
        confidence = prediction.get('confidence', 'N/A')
        
        print(f"    Win Probability: {win_prob:.4f if isinstance(win_prob, (int, float)) else win_prob}")
        print(f"    Place Probability: {place_prob:.4f if isinstance(place_prob, (int, float)) else place_prob}")
        print(f"    Raw Win Probability: {raw_win_prob:.4f if isinstance(raw_win_prob, (int, float)) else raw_win_prob}")
        print(f"    Raw Place Probability: {raw_place_prob:.4f if isinstance(raw_place_prob, (int, float)) else raw_place_prob}")
        print(f"    Calibration Applied: {prediction.get('calibration_applied', False)}")
        print(f"    Win Calibration Applied: {prediction.get('win_calibration_applied', False)}")
        print(f"    Place Calibration Applied: {prediction.get('place_calibration_applied', False)}")
        print(f"    Confidence: {confidence:.4f if isinstance(confidence, (int, float)) else confidence}")
        print(f"    Model Info: {prediction.get('model_info', 'N/A')}")
        
        if prediction.get('calibration_applied'):
            print("  ✅ Calibration is working in ML System!")
        else:
            print("  ⚠️ Calibration not applied (may be expected if no calibrator trained)")
            
    except Exception as e:
        print(f"  ❌ ML System prediction failed: {e}")
    
    print("✅ ML System integration test complete\n")


def test_helper_function():
    """Test the helper function for easy calibration."""
    print("🧪 Testing Helper Function...")
    
    try:
        result = apply_probability_calibration(0.25, 0.45)
        
        print("  Helper Function Results:")
        print(f"    Calibrated Win: {result['calibrated_win_prob']:.4f}")
        print(f"    Calibrated Place: {result['calibrated_place_prob']:.4f}")
        print(f"    Win Calibration Applied: {result['win_calibration_applied']}")
        print(f"    Place Calibration Applied: {result['place_calibration_applied']}")
        
        print("  ✅ Helper function working!")
        
    except Exception as e:
        print(f"  ❌ Helper function failed: {e}")
    
    print("✅ Helper function test complete\n")


def main():
    print("🎯 Probability Calibration Integration Test")
    print("=" * 50)
    
    # Test calibrator standalone
    test_calibrator_standalone()
    
    # Test ML system integration
    test_ml_integration()
    
    # Test helper function
    test_helper_function()
    
    print("🏁 All tests completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
