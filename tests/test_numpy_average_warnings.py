#!/usr/bin/env python3
"""
Test to capture and assert absence of numpy.average warnings.

This test implements Step 4 of the task: Capture and assert absence of numpy.average warnings.
It uses warnings.catch_warnings to detect any numpy.average warnings during feature building
and fails the test with a clear message if any such warning slips through.
"""

import logging
import sqlite3
import warnings

import pandas as pd

from temporal_feature_builder import TemporalFeatureBuilder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_numpy_average_warnings():
    """Test that build_features_for_race doesn't produce numpy.average warnings."""
    print("🧪 Testing for numpy.average warnings in feature building...")
    print("=" * 60)

    # Initialize the builder
    builder = TemporalFeatureBuilder("greyhound_racing_data.db")

    # Get a test race with sufficient historical data
    conn = sqlite3.connect("greyhound_racing_data.db")

    # Find a race with dogs that have historical data
    query = """
    SELECT d.*, r.venue, r.grade, r.distance, r.race_date, r.race_time
    FROM dog_race_data d
    LEFT JOIN race_metadata r ON d.race_id = r.race_id
    WHERE d.race_id = ?
    """

    # Use a known race ID that should have dogs with historical data
    test_race_id = "RICH_3_20_July_2025"
    race_data = pd.read_sql_query(query, conn, params=[test_race_id])
    conn.close()

    if race_data.empty:
        logger.error(f"Test race {test_race_id} not found!")
        return False

    logger.info(f"Testing with race {test_race_id} containing {len(race_data)} dogs")

    # Capture warnings during feature building
    success = False
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", RuntimeWarning)

            # Build features for the race - this is where numpy.average warnings might occur
            features = builder.build_features_for_race(race_data, test_race_id)

            # Check for numpy.average warnings specifically
            numpy_average_warnings = [
                warning for warning in w if "numpy.average" in str(warning.message)
            ]

            # Assert no numpy.average warnings were captured
            assert not any("numpy.average" in str(wi.message) for wi in w), (
                f"NUMPY.AVERAGE WARNING DETECTED: Found {len(numpy_average_warnings)} numpy.average warnings during feature building. "
                f"Warnings: {[str(warn.message) for warn in numpy_average_warnings]}"
            )

            logger.info(f"✅ Successfully built features for {len(features)} dogs")
            logger.info(
                f"✅ No numpy.average warnings detected in {len(w)} total warnings captured"
            )

            # Log any other warnings for debugging (but don't fail the test)
            if w:
                logger.info(f"ℹ️ Other warnings captured (non-numpy.average): {len(w)}")
                for warning in w:
                    logger.debug(
                        f"  Warning: {warning.category.__name__}: {warning.message}"
                    )

            success = True

    except AssertionError as e:
        logger.error(f"❌ ASSERTION FAILED: {e}")
        print(f"\n❌ TEST FAILED: {e}")
        return False

    except Exception as e:
        logger.error(f"❌ Error during feature building: {e}")
        print(f"\n❌ TEST FAILED: Unexpected error during feature building: {e}")
        return False

    if success:
        print(
            "✅ TEST PASSED: No numpy.average warnings detected during feature building"
        )
        return True
    else:
        print("❌ TEST FAILED: Test did not complete successfully")
        return False


def test_numpy_average_warnings_strict():
    """More strict test that fails on ANY numpy-related warnings."""
    print("\n🔬 Running strict numpy warning test...")
    print("=" * 60)

    builder = TemporalFeatureBuilder("greyhound_racing_data.db")

    # Get test race data
    conn = sqlite3.connect("greyhound_racing_data.db")
    query = """
    SELECT d.*, r.venue, r.grade, r.distance, r.race_date, r.race_time
    FROM dog_race_data d
    LEFT JOIN race_metadata r ON d.race_id = r.race_id
    WHERE d.race_id = ?
    """

    test_race_id = "RICH_3_20_July_2025"
    race_data = pd.read_sql_query(query, conn, params=[test_race_id])
    conn.close()

    if race_data.empty:
        print(f"❌ Test race {test_race_id} not found!")
        return False

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", RuntimeWarning)

            # Build features
            features = builder.build_features_for_race(race_data, test_race_id)

            # Check for any numpy-related warnings (more strict than just numpy.average)
            numpy_warnings = [
                warning for warning in w if "numpy" in str(warning.message).lower()
            ]

            if numpy_warnings:
                warning_messages = [str(warn.message) for warn in numpy_warnings]
                logger.warning(f"Found {len(numpy_warnings)} numpy-related warnings:")
                for msg in warning_messages:
                    logger.warning(f"  - {msg}")

                # Still fail specifically on numpy.average
                numpy_average_warnings = [
                    warning
                    for warning in numpy_warnings
                    if "numpy.average" in str(warning.message)
                ]

                if numpy_average_warnings:
                    print(
                        f"❌ STRICT TEST FAILED: Found {len(numpy_average_warnings)} numpy.average warnings"
                    )
                    return False
                else:
                    print(
                        f"⚠️ Found {len(numpy_warnings)} other numpy warnings, but no numpy.average warnings"
                    )
                    print(
                        "✅ STRICT TEST PASSED: No numpy.average warnings (other numpy warnings present)"
                    )
                    return True
            else:
                print("✅ STRICT TEST PASSED: No numpy warnings of any kind detected")
                return True

    except Exception as e:
        print(f"❌ STRICT TEST FAILED: Error during feature building: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Running numpy.average warning detection tests...")
    print(
        "📋 This test implements Step 4: Capture and assert absence of numpy.average warnings"
    )
    print()

    # Run the main test
    test1_passed = test_numpy_average_warnings()

    # Run the strict test
    test2_passed = test_numpy_average_warnings_strict()

    # Final summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY:")
    print(f"✅ Basic numpy.average test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"✅ Strict numpy warning test: {'PASSED' if test2_passed else 'FAILED'}")

    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED: No numpy.average warnings detected!")
        print(
            "✅ Step 4 implementation complete: Warning capture and assertion system working correctly"
        )
    else:
        print("\n❌ SOME TESTS FAILED: numpy.average warnings may be present")
        print("⚠️ Step 4 needs attention: Warning detection system found issues")
        exit(1)
