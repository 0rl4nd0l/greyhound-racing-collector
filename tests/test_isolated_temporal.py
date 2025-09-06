#!/usr/bin/env python3
"""
Isolated Test for Temporal Leakage Protection
==============================================

Simple tests that don't rely on conftest.py setup to isolate the temporal leakage
protection functionality.
"""

import os
import sqlite3
import sys
import tempfile

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from temporal_feature_builder import (
    TemporalFeatureBuilder,
    create_temporal_assertion_hook,
)


def test_temporal_assertion_hook():
    """Test that temporal assertion hook detects leakage."""
    print("🧪 Testing temporal assertion hook...")

    assert_no_leakage = create_temporal_assertion_hook()

    # Test with safe features
    safe_features = {
        "box_number": 1,
        "weight": 30.5,
        "starting_price": 2.50,
        "venue": "TRACK_A",
    }

    try:
        assert_no_leakage(safe_features, "test_race", "test_dog")
        print("✅ Safe features passed assertion")
    except Exception as e:
        print(f"❌ Safe features failed: {e}")
        return False

    # Test with leakage features
    leakage_features = {
        "box_number": 1,
        "finish_position": 1,  # This is leakage!
        "weight": 30.5,
    }

    try:
        assert_no_leakage(leakage_features, "test_race", "test_dog")
        print("❌ Leakage features should have failed but passed")
        return False
    except AssertionError as e:
        if "TEMPORAL LEAKAGE DETECTED" in str(e):
            print("✅ Leakage features correctly detected")
        else:
            print(f"❌ Wrong assertion error: {e}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

    return True


def test_temporal_feature_builder_basic():
    """Test basic TemporalFeatureBuilder functionality."""
    print("🧪 Testing TemporalFeatureBuilder basic functionality...")

    # Create temporary database
    temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    conn = sqlite3.connect(temp_db.name)

    # Create minimal test tables
    conn.execute(
        """
    CREATE TABLE race_metadata (
        race_id TEXT PRIMARY KEY,
        venue TEXT,
        grade TEXT,
        distance TEXT,
        track_condition TEXT,
        weather TEXT,
        temperature REAL,
        humidity REAL,
        wind_speed REAL,
        field_size INTEGER,
        race_date TEXT,
        race_time TEXT
    )
    """
    )

    conn.execute(
        """
    CREATE TABLE dog_race_data (
        id INTEGER PRIMARY KEY,
        race_id TEXT,
        dog_clean_name TEXT,
        box_number INTEGER,
        weight REAL,
        starting_price REAL,
        trainer_name TEXT,
        finish_position INTEGER,
        individual_time REAL,
        sectional_1st REAL,
        margin REAL,
        FOREIGN KEY (race_id) REFERENCES race_metadata (race_id)
    )
    """
    )

    conn.commit()
    conn.close()

    try:
        # Test initialization
        builder = TemporalFeatureBuilder(temp_db.name)
        print("✅ TemporalFeatureBuilder initialized successfully")

        # Test feature categorization
        assert len(builder.pre_race_features) > 0, "Should have pre-race features"
        assert len(builder.post_race_features) > 0, "Should have post-race features"
        assert (
            "finish_position" in builder.post_race_features
        ), "finish_position should be post-race"
        assert (
            "box_number" in builder.pre_race_features
        ), "box_number should be pre-race"
        print("✅ Feature categorization working correctly")

        return True

    except Exception as e:
        print(f"❌ TemporalFeatureBuilder test failed: {e}")
        return False
    finally:
        # Cleanup
        os.unlink(temp_db.name)


def test_feature_classification():
    """Test that features are correctly classified as pre/post race."""
    print("🧪 Testing feature classification...")

    # Create dummy database for initialization
    temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    conn = sqlite3.connect(temp_db.name)
    conn.execute("CREATE TABLE race_metadata (race_id TEXT)")
    conn.execute("CREATE TABLE dog_race_data (id INTEGER)")
    conn.commit()
    conn.close()

    try:
        builder = TemporalFeatureBuilder(temp_db.name)

        # Check specific feature classifications
        expected_pre_race = [
            "box_number",
            "weight",
            "starting_price",
            "trainer_name",
            "venue",
            "grade",
            "distance",
            "track_condition",
            "weather",
            "temperature",
            "humidity",
            "wind_speed",
            "field_size",
            "race_date",
            "race_time",
        ]

        expected_post_race = [
            "finish_position",
            "individual_time",
            "sectional_1st",
            "sectional_2nd",
            "margin",
        ]

        for feature in expected_pre_race:
            if feature in builder.post_race_features:
                print(f"❌ {feature} incorrectly classified as post-race")
                return False

        for feature in expected_post_race:
            if feature not in builder.post_race_features:
                print(f"❌ {feature} not classified as post-race")
                return False

        print("✅ All features classified correctly")
        return True

    except Exception as e:
        print(f"❌ Feature classification test failed: {e}")
        return False
    finally:
        os.unlink(temp_db.name)


def main():
    """Run all isolated tests."""
    print("🚀 Running Isolated Temporal Leakage Protection Tests")
    print("=" * 60)

    tests = [
        test_temporal_assertion_hook,
        test_temporal_feature_builder_basic,
        test_feature_classification,
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        print(f"\n📋 Running {test_func.__name__}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_func.__name__} PASSED")
            else:
                print(f"❌ {test_func.__name__} FAILED")
        except Exception as e:
            print(f"❌ {test_func.__name__} ERROR: {e}")

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("💥 Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
