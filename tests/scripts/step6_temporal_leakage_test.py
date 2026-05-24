#!/usr/bin/env python3
"""
Step 6: Test temporal leakage guard on future race

This test verifies that the temporal leakage protection system correctly
detects and prevents predictions on races scheduled for future dates.

Requirements:
- Pick a race_date ≥ today (or simply the next scheduled race in DB)
- Call predict_race and expect a ValueError/AssertionError from create_temporal_assertion_hook
- Flag pass if error is raised; fail otherwise
"""

from datetime import datetime, timedelta

import pandas as pd

from ml_system_v4 import MLSystemV4


def test_temporal_leakage_guard():
    """Test that temporal leakage guard correctly prevents future race predictions."""

    print("🛡️  Step 6: Testing temporal leakage guard on future race")
    print("=" * 60)

    # Initialize ML System
    ml_system = MLSystemV4(db_path="greyhound_racing_data.db")

    # Create a test race for tomorrow (future date)
    future_date = (datetime.now() + timedelta(days=1)).strftime("%d %B %Y")

    print(f"📅 Testing with future race date: {future_date}")
    print(f"📅 Current date: {datetime.now().date()}")

    # Create test race data
    race_data = pd.DataFrame(
        {
            "dog_clean_name": ["Test Dog"],
            "race_date": [future_date],
            "venue": ["Test Venue"],
            "distance": [500],
            "grade": [5],
            "box_number": [1],
            "weight": [30.0],
            "race_time": ["12:00"],
            "track_condition": ["Good"],
            "weather": ["Fine"],
            "trainer_name": ["Test Trainer"],
        }
    )

    race_id = f'test_race_{future_date.replace(" ", "_")}'

    print(f"🎯 Attempting to predict race: {race_id}")

    # Call predict_race - should detect temporal leakage
    result = ml_system.predict_race(race_data, race_id)

    # Check if temporal leakage was detected
    if not result.get("success", True):
        error_message = result.get("error", "")
        if (
            "TEMPORAL LEAKAGE DETECTED" in error_message
            and "is in the future" in error_message
        ):
            print("✅ PASS: Temporal leakage guard correctly detected future race")
            print(f"   Error message: {error_message}")
            return True
        else:
            print("❌ FAIL: Prediction failed but not due to temporal leakage")
            print(f"   Error message: {error_message}")
            return False
    else:
        print(
            "❌ FAIL: Prediction succeeded for future race (temporal leakage not detected)"
        )
        print(f"   Result: {result}")
        return False


def main():
    """Run the temporal leakage test."""
    try:
        success = test_temporal_leakage_guard()

        print("\n" + "=" * 60)
        if success:
            print("🎉 Step 6 PASSED: Temporal leakage guard is working correctly!")
            print("   The system properly prevents predictions on future races.")
        else:
            print("💥 Step 6 FAILED: Temporal leakage guard is not working properly!")
            print("   The system did not detect the future race date.")

        return success

    except Exception as e:
        print(f"💥 Step 6 ERROR: Unexpected error during test: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
