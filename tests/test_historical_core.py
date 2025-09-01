#!/usr/bin/env python3
"""
Simple Test for Historical Filtering Core Functionality
======================================================

Focused test script for the core historical date filtering implementation.
Tests only the date parsing and filtering logic without initializing heavy ML systems.

Author: AI Assistant
Date: January 2025
"""

import os
import sys
from datetime import date, datetime, timedelta

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.date_parsing import is_historical, parse_date_flexible


def test_is_historical_core():
    """Test the core is_historical function"""
    print("🧪 Testing is_historical function...")

    # Test cases with expected results
    test_cases = [
        # Past dates should be historical
        ("2024-01-01", True, "Past date string"),
        ("2023-12-25", True, "Past Christmas date"),
        (date.today() - timedelta(days=1), True, "Yesterday"),
        (datetime.now() - timedelta(days=2), True, "Two days ago"),
        # Present/future dates should not be historical
        (date.today(), False, "Today"),
        (date.today() + timedelta(days=1), False, "Tomorrow"),
        (datetime.now() + timedelta(hours=1), False, "One hour from now"),
        # Invalid inputs should return False
        ("invalid_date", False, "Invalid date string"),
        ("", False, "Empty string"),
        (None, False, "None value"),
        (123, False, "Integer value"),
    ]

    passed = 0
    failed = 0

    for input_val, expected, description in test_cases:
        try:
            result = is_historical(input_val)
            if result == expected:
                print(f"  ✅ {description}: {input_val} -> {result}")
                passed += 1
            else:
                print(
                    f"  ❌ {description}: {input_val} -> {result} (expected {expected})"
                )
                failed += 1
        except Exception as e:
            print(f"  ⚠️  {description}: {input_val} -> Exception: {e}")
            failed += 1

    print(f"📊 is_historical test results: {passed} passed, {failed} failed")
    return failed == 0


def test_parse_date_flexible_core():
    """Test the parse_date_flexible function"""
    print("\n🧪 Testing parse_date_flexible function...")

    test_cases = [
        ("25 July 2025", "2025-07-25", "Standard format"),
        ("2025-07-25", "2025-07-25", "ISO format"),
        ("01 January 2024", "2024-01-01", "New Year format"),
        ("2024-12-31", "2024-12-31", "Year end format"),
    ]

    passed = 0
    failed = 0

    for input_val, expected, description in test_cases:
        try:
            result = parse_date_flexible(input_val)
            if result == expected:
                print(f"  ✅ {description}: '{input_val}' -> '{result}'")
                passed += 1
            else:
                print(
                    f"  ❌ {description}: '{input_val}' -> '{result}' (expected '{expected}')"
                )
                failed += 1
        except Exception as e:
            print(f"  ⚠️  {description}: '{input_val}' -> Exception: {e}")
            failed += 1

    print(f"📊 parse_date_flexible test results: {passed} passed, {failed} failed")
    return failed == 0


def test_date_scenarios():
    """Test real-world date scenarios"""
    print("\n🧪 Testing real-world date scenarios...")

    # Create test dates relative to today
    today = date.today()
    test_dates = [
        (today - timedelta(days=1), True, "Yesterday"),
        (today, False, "Today"),
        (today + timedelta(days=1), False, "Tomorrow"),
        (today - timedelta(days=7), True, "Last week"),
        (today + timedelta(days=7), False, "Next week"),
        (today - timedelta(days=30), True, "Last month"),
        (today + timedelta(days=30), False, "Next month"),
    ]

    passed = 0
    failed = 0

    for test_date, expected, description in test_dates:
        try:
            result = is_historical(test_date)
            if result == expected:
                print(f"  ✅ {description} ({test_date}): {result}")
                passed += 1
            else:
                print(
                    f"  ❌ {description} ({test_date}): {result} (expected {expected})"
                )
                failed += 1
        except Exception as e:
            print(f"  ⚠️  {description} ({test_date}): Exception: {e}")
            failed += 1

    print(f"📊 Date scenarios test results: {passed} passed, {failed} failed")
    return failed == 0


def test_filename_pattern_simulation():
    """Simulate filename pattern recognition without file I/O"""
    print("\n🧪 Testing filename pattern recognition logic...")

    import re

    def simulate_filename_check(filename):
        """Simulate the filename checking logic from _is_file_historical"""
        # Pattern 1: YYYY-MM-DD format in filename
        date_pattern_1 = r"(\d{4}-\d{2}-\d{2})"
        match = re.search(date_pattern_1, filename)
        if match:
            date_str = match.group(1)
            return is_historical(date_str)

        # Pattern 2: DDMMYYYY format in filename
        date_pattern_2 = r"(\d{2})(\d{2})(\d{4})"
        match = re.search(date_pattern_2, filename)
        if match:
            day, month, year = match.groups()
            date_str = f"{year}-{month}-{day}"
            return is_historical(date_str)

        # No date pattern found
        return None  # Would fall back to content checking

    # Test filename patterns
    today_str = date.today().strftime("%Y-%m-%d")
    yesterday_str = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    tomorrow_str = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    filename_tests = [
        (f"race_{yesterday_str}.csv", True, "Historical date in filename"),
        (f"race_{today_str}.csv", False, "Today's date in filename"),
        (f"race_{tomorrow_str}.csv", False, "Future date in filename"),
        ("race_01012024.csv", True, "Historical DDMMYYYY format"),
        ("race_31122025.csv", False, "Future DDMMYYYY format"),
        ("race_01012023.csv", True, "Historical DDMMYYYY format 2023"),
        ("general_race.csv", None, "No date in filename"),
    ]

    passed = 0
    failed = 0

    for filename, expected, description in filename_tests:
        try:
            result = simulate_filename_check(filename)
            if result == expected:
                print(f"  ✅ {description}: '{filename}' -> {result}")
                passed += 1
            else:
                print(
                    f"  ❌ {description}: '{filename}' -> {result} (expected {expected})"
                )
                failed += 1
        except Exception as e:
            print(f"  ⚠️  {description}: '{filename}' -> Exception: {e}")
            failed += 1

    print(f"📊 Filename pattern test results: {passed} passed, {failed} failed")
    return failed == 0


def main():
    """Run core functionality tests"""
    print("=" * 60)
    print("🧪 HISTORICAL FILTERING CORE FUNCTIONALITY TESTS")
    print("=" * 60)
    print(f"📅 Test run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📅 Today's date: {date.today().strftime('%Y-%m-%d')}")
    print()

    all_tests_passed = True

    # Run core tests
    test_functions = [
        test_is_historical_core,
        test_parse_date_flexible_core,
        test_date_scenarios,
        test_filename_pattern_simulation,
    ]

    for test_func in test_functions:
        try:
            if not test_func():
                all_tests_passed = False
        except Exception as e:
            print(f"❌ Test function {test_func.__name__} failed: {e}")
            all_tests_passed = False

    print("\n" + "=" * 60)
    if all_tests_passed:
        print("✅ ALL CORE TESTS PASSED!")
        print("🎯 Historical filtering core functionality is working correctly")
        print()
        print("✨ Implementation Summary:")
        print("• is_historical() function correctly identifies dates < today")
        print("• parse_date_flexible() handles multiple date formats")
        print("• Filename pattern recognition extracts dates correctly")
        print("• Edge cases (invalid inputs) are handled gracefully")
    else:
        print("❌ SOME CORE TESTS FAILED!")
        print("⚠️  Please check the implementation")
    print("=" * 60)

    return 0 if all_tests_passed else 1


if __name__ == "__main__":
    sys.exit(main())
