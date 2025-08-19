#!/usr/bin/env python3
"""
Manual test runner for CSV parser tests
"""

import sys
import traceback
from csv_ingestion import FormGuideCsvIngestor
from test_csv_parser import test_perfect_file, test_missing_header, test_continuation_rows, test_extra_unexpected_columns, test_validate_csv_schema, test_process_form_guide_format_with_real_csv, test_validate_real_race_csv

def run_test(test_func, test_name):
    """Run a single test function and report results."""
    try:
        # Set up fixture if needed
        if test_func.__code__.co_argcount > 0:
            ingestor_instance = FormGuideCsvIngestor()
            test_func(ingestor_instance)
        else:
            test_func()
        print(f"✅ {test_name} - PASSED")
        return True
    except Exception as e:
        print(f"❌ {test_name} - FAILED: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Running CSV Parser Tests...")
    print("=" * 50)
    
    tests = [
        (test_perfect_file, "test_perfect_file"),
        (test_missing_header, "test_missing_header"), 
        (test_validate_csv_schema, "test_validate_csv_schema"),
        (test_validate_real_race_csv, "test_validate_real_race_csv"),
        (test_continuation_rows, "test_continuation_rows"),
        (test_process_form_guide_format_with_real_csv, "test_process_form_guide_format_with_real_csv"),
        (test_extra_unexpected_columns, "test_extra_unexpected_columns")
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func, test_name in tests:
        if run_test(test_func, test_name):
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"💥 {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
