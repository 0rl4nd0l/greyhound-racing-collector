# Moved Test Files

The following files were moved from the root directory to `tests/` for better organization:

## Test Scripts (moved 2025-07-30)
- `integrity_test.py` - Database integrity testing
- `ml_backtesting_trainer.py` - ML model backtesting system
- `automated_backtesting_system.py` - Automated prediction backtesting

## Test Databases and Results
- `greyhound_racing_data_test.db` - Test database for development
- `test.db` - General test database
- `flask_api_test_results.json` - API test results
- `test_unified_prediction.json` - Unified prediction test results

## Test Result Directories (already existed)
- `automated_backtesting_results/` - Backtesting output files
- `ml_backtesting_results/` - ML backtesting output files

These files were moved to establish proper directory hygiene and separate test code from production code.
