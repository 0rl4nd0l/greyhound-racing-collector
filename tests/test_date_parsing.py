#!/usr/bin/env python3
"""
Test script for flexible date parsing functionality.

This script verifies that the new parse_date_flexible function
works correctly with both '%d %B %Y' and '%Y-%m-%d' formats.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.date_parsing import parse_date_flexible


def test_date_parsing():
    """Test the flexible date parsing function with various inputs."""

    print("🧪 Testing flexible date parsing...")
    print("=" * 50)

    # Test cases: (input, expected_output, description)
    test_cases = [
        ("25 July 2025", "2025-07-25", "Standard '%d %B %Y' format"),
        ("2025-07-25", "2025-07-25", "ISO '%Y-%m-%d' format"),
        ("1 January 2024", "2024-01-01", "Start of year format"),
        ("2024-01-01", "2024-01-01", "ISO format - start of year"),
        ("31 December 2025", "2025-12-31", "End of year format"),
        ("2025-12-31", "2025-12-31", "ISO format - end of year"),
        ("15 March 2024", "2024-03-15", "Mid-year format"),
        ("2024-03-15", "2024-03-15", "ISO format - mid-year"),
    ]

    passed = 0
    failed = 0

    for input_date, expected, description in test_cases:
        try:
            result = parse_date_flexible(input_date)
            if result == expected:
                print(f"✅ PASS: {description}")
                print(f"   Input: '{input_date}' → Output: '{result}'")
                passed += 1
            else:
                print(f"❌ FAIL: {description}")
                print(
                    f"   Input: '{input_date}' → Expected: '{expected}' → Got: '{result}'"
                )
                failed += 1
        except Exception as e:
            print(f"❌ ERROR: {description}")
            print(f"   Input: '{input_date}' → Exception: {str(e)}")
            failed += 1
        print()

    # Test error cases
    print("Testing error cases...")
    print("-" * 30)

    error_cases = [
        ("", "Empty string"),
        ("invalid date", "Invalid format"),
        ("32 January 2025", "Invalid day"),
        ("25 InvalidMonth 2025", "Invalid month"),
        ("2025-13-01", "Invalid month in ISO format"),
        ("2025-01-32", "Invalid day in ISO format"),
    ]

    for input_date, description in error_cases:
        try:
            result = parse_date_flexible(input_date)
            print(f"❌ UNEXPECTED SUCCESS: {description}")
            print(f"   Input: '{input_date}' → Should have failed but got: '{result}'")
            failed += 1
        except ValueError as e:
            print(f"✅ EXPECTED ERROR: {description}")
            print(f"   Input: '{input_date}' → Correctly raised ValueError: {str(e)}")
            passed += 1
        except Exception as e:
            print(f"❌ UNEXPECTED ERROR: {description}")
            print(f"   Input: '{input_date}' → Unexpected exception: {str(e)}")
            failed += 1
        print()

    # Summary
    print("=" * 50)
    print(f"📊 TEST SUMMARY:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📈 Success Rate: {passed/(passed+failed)*100:.1f}%")

    if failed == 0:
        print(f"\n🎉 All tests passed! Flexible date parsing is working correctly.")
        return True
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = test_date_parsing()
    sys.exit(0 if success else 1)
