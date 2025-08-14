# Regression Test Suite for Race Data Processing

## Overview
This test suite provides comprehensive regression testing for race data processing functionality. It covers the core requirements specified in Step 7 of the broader development plan.

## Test Coverage

### 1. CSV with 10 rows returns 1 race
- **Test**: `test_csv_10_rows_returns_1_race`
- **Purpose**: Validates that processing a CSV file with 10 rows of race data produces exactly 1 race object
- **Validates**: CSV parsing logic, race aggregation from multiple dog entries

### 2. Corrupt CSV skipped
- **Test**: `test_corrupt_csv_skipped`
- **Purpose**: Ensures corrupted CSV files are properly handled and skipped without crashing
- **Validates**: Error handling for HTML content in CSV files, empty files, malformed data

### 3. JSON file returns unchanged
- **Test**: `test_json_file_returns_unchanged`
- **Purpose**: Verifies that JSON race data is processed and returned without modification
- **Validates**: JSON parsing, data integrity preservation

### 4. Combined list length matches (#json + #csv)
- **Test**: `test_combined_list_length_matches`
- **Purpose**: Confirms that when combining CSV and JSON sources, the total count equals the sum of individual sources
- **Validates**: Data combination logic, no data loss during merging

### 5. Duplicate venue/date/race_number not repeated
- **Test**: `test_duplicate_venue_date_race_not_repeated`
- **Purpose**: Ensures duplicate races (same venue, date, and race number) are deduplicated correctly
- **Validates**: Deduplication algorithm, race identity matching

## Additional Tests

### 6. Comprehensive workflow test
- **Test**: `test_comprehensive_workflow`
- **Purpose**: Tests the entire workflow with mixed valid/invalid data
- **Validates**: End-to-end processing, error handling in production scenarios

### 7. CSV ingestion integration test
- **Test**: `test_csv_ingestion_integration`
- **Purpose**: Integration test with the actual CSV ingestion module if available
- **Validates**: Real-world CSV processing, dog_name field mapping

## Test Architecture

### RaceDataProcessor Class
The test suite uses a mock `RaceDataProcessor` class that simulates the core functionality:

- `process_csv_file()` - Processes CSV files and extracts race data
- `process_json_file()` - Processes JSON files and preserves data structure  
- `combine_data_sources()` - Combines data from multiple sources with deduplication

### Test Fixtures
- `temp_dir` - Creates temporary directory for test files
- `processor` - Provides RaceDataProcessor instance
- `sample_csv_10_rows` - Creates valid 10-row CSV file
- `corrupt_csv_file` - Creates HTML-corrupted CSV file
- `empty_csv_file` - Creates empty CSV file
- `sample_json_file` - Creates valid JSON race data
- `duplicate_race_data` - Creates CSV and JSON with same race information

## Running the Tests

### Direct execution:
```bash
python test_regression_race_data_processing.py
```

### With pytest:
```bash
pytest test_regression_race_data_processing.py -v
```

### Expected Output
```
🧪 RACE DATA PROCESSING REGRESSION TEST SUITE
======================================================================
Testing race data processing with CSV, JSON, and deduplication
======================================================================

🎉 ALL REGRESSION TESTS PASSED!
✅ CSV with 10 rows returns 1 race
✅ Corrupt CSV files are skipped
✅ JSON files return unchanged
✅ Combined list length matches sources
✅ Duplicates are properly deduplicated
```

## Key Features

1. **Comprehensive Error Handling**: Tests verify proper handling of corrupt, empty, and malformed files
2. **Data Integrity**: Ensures JSON data remains unchanged while CSV data is properly processed
3. **Deduplication Logic**: Validates that duplicate races are identified by venue/date/race_number combination
4. **Integration Ready**: Includes optional integration tests with actual CSV ingestion modules
5. **Clear Output**: Provides detailed test progress and results with emoji indicators

## Dependencies

- `pytest` - Test framework
- `pandas` - CSV processing
- `json` - JSON handling
- `tempfile` - Temporary file creation
- Optional: `csv_ingestion` module for integration tests

## File Structure

```
test_regression_race_data_processing.py    # Main test file
test_regression_race_data_processing_README.md    # This documentation
```

## Integration Notes

The test suite is designed to work with or without the actual CSV ingestion modules. If the modules are not available, integration tests are automatically skipped while core functionality tests continue to run.

The tests use a mock processor that simulates the expected behavior of the real race data processing system, making them suitable for both development and regression testing scenarios.
