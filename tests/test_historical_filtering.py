#!/usr/bin/env python3
"""
Test Historical Filtering Functionality
======================================

Test script to validate the historical race date filtering implementation.
Tests the is_historical function and the batch pipeline filtering logic.

Author: AI Assistant
Date: January 2025
"""

import sys
import os
from datetime import datetime, date, timedelta

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.date_parsing import is_historical, parse_date_flexible

def test_is_historical_function():
    """Test the is_historical function with various date inputs"""
    print("🧪 Testing is_historical function...")
    
    # Test cases
    test_cases = [
        # (input, expected_result, description)
        ("2024-01-01", True, "Past date string"),
        ("2023-12-25", True, "Past Christmas date"),
        (date.today() - timedelta(days=1), True, "Yesterday's date object"),
        (date.today(), False, "Today's date object"),
        (date.today() + timedelta(days=1), False, "Tomorrow's date object"),
        (datetime.now() - timedelta(days=2), True, "Past datetime object"),
        (datetime.now() + timedelta(hours=1), False, "Future datetime object"),
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
                print(f"  ❌ {description}: {input_val} -> {result} (expected {expected})")
                failed += 1
        except Exception as e:
            print(f"  ⚠️  {description}: {input_val} -> Exception: {e}")
            failed += 1
    
    print(f"\n📊 Results: {passed} passed, {failed} failed")
    return failed == 0

def test_parse_date_flexible():
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
                print(f"  ❌ {description}: '{input_val}' -> '{result}' (expected '{expected}')")
                failed += 1
        except Exception as e:
            print(f"  ⚠️  {description}: '{input_val}' -> Exception: {e}")
            failed += 1
    
    print(f"\n📊 Results: {passed} passed, {failed} failed")
    return failed == 0

def test_batch_pipeline_historical_filtering():
    """Test the batch pipeline historical filtering logic"""
    print("\n🧪 Testing batch pipeline historical filtering...")
    
    try:
        from batch_prediction_pipeline import BatchPredictionPipeline
        
        # Create a test pipeline
        pipeline = BatchPredictionPipeline()
        
        # Test filename-based historical detection
        test_files = [
            ("race_2024-01-01.csv", True, "Historical filename with date"),
            ("race_2025-12-31.csv", False, "Future filename with date"),
            (f"race_{date.today().strftime('%Y-%m-%d')}.csv", False, "Today's filename"),
            ("race_01012024.csv", True, "Historical DDMMYYYY format"),
            ("regular_race.csv", False, "No date in filename (fallback to file mtime)"),
        ]
        
        passed = 0
        failed = 0
        
        for filename, expected, description in test_files:
            try:
                # Create a temporary test file to check
                test_file_path = f"/tmp/{filename}"
                with open(test_file_path, 'w') as f:
                    f.write("dog_name,race_date\nTest Dog,2024-01-01\n")
                
                result = pipeline._is_file_historical(test_file_path)
                
                # Clean up
                if os.path.exists(test_file_path):
                    os.remove(test_file_path)
                
                if result == expected:
                    print(f"  ✅ {description}: {filename} -> {result}")
                    passed += 1
                else:
                    print(f"  ❌ {description}: {filename} -> {result} (expected {expected})")
                    failed += 1
                    
            except Exception as e:
                print(f"  ⚠️  {description}: {filename} -> Exception: {e}")
                failed += 1
        
        print(f"\n📊 Results: {passed} passed, {failed} failed")
        return failed == 0
        
    except ImportError:
        print("  ⚠️  BatchPredictionPipeline not available, skipping this test")
        return True

def test_integration():
    """Test integration of historical filtering with actual date scenarios"""
    print("\n🧪 Testing integration scenarios...")
    
    # Create some test dates
    today = date.today()
    yesterday = today - timedelta(days=1)
    tomorrow = today + timedelta(days=1)
    last_week = today - timedelta(days=7)
    next_week = today + timedelta(days=7)
    
    scenarios = [
        (yesterday.strftime("%Y-%m-%d"), True, "Yesterday should be historical"),
        (today.strftime("%Y-%m-%d"), False, "Today should not be historical"),
        (tomorrow.strftime("%Y-%m-%d"), False, "Tomorrow should not be historical"),
        (last_week.strftime("%Y-%m-%d"), True, "Last week should be historical"),
        (next_week.strftime("%Y-%m-%d"), False, "Next week should not be historical"),
    ]
    
    passed = 0
    failed = 0
    
    for date_str, expected, description in scenarios:
        try:
            result = is_historical(date_str)
            if result == expected:
                print(f"  ✅ {description}: {date_str} -> {result}")
                passed += 1
            else:
                print(f"  ❌ {description}: {date_str} -> {result} (expected {expected})")
                failed += 1
        except Exception as e:
            print(f"  ⚠️  {description}: {date_str} -> Exception: {e}")
            failed += 1
    
    print(f"\n📊 Results: {passed} passed, {failed} failed")
    return failed == 0

def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 HISTORICAL FILTERING FUNCTIONALITY TESTS")
    print("=" * 60)
    print(f"📅 Test run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📅 Today's date: {date.today().strftime('%Y-%m-%d')}")
    print()
    
    all_tests_passed = True
    
    # Run all test functions
    test_functions = [
        test_is_historical_function,
        test_parse_date_flexible, 
        test_batch_pipeline_historical_filtering,
        test_integration
    ]
    
    for test_func in test_functions:
        try:
            if not test_func():
                all_tests_passed = False
        except Exception as e:
            print(f"❌ Test function {test_func.__name__} failed with exception: {e}")
            all_tests_passed = False
    
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("✅ ALL TESTS PASSED!")
        print("🎯 Historical filtering functionality is working correctly")
    else:
        print("❌ SOME TESTS FAILED!")
        print("⚠️  Please check the implementation")
    print("=" * 60)
    
    return 0 if all_tests_passed else 1

if __name__ == "__main__":
    sys.exit(main())
