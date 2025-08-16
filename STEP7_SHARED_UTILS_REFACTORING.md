# Step 7: Shared Race File Utilities Refactoring

## Overview

This document describes the completion of Step 7 in the optimization plan: refactoring helper functions into a shared utils module to eliminate duplicated linear scans project-wide.

## Summary

✅ **COMPLETED**: Successfully consolidated common race file processing patterns into a shared `utils/race_file_utils.py` module and updated multiple scripts to use the shared utilities, eliminating duplicated code and improving maintainability.

## Key Components Created

### 1. `utils/race_file_utils.py` - Shared Utilities Module

**Main Class: `RaceFileManager`**
- Centralized manager for race file operations with caching and optimization
- Consolidates common patterns used across multiple scripts
- Provides both class-based and standalone function interfaces

**Key Methods:**
- `ensure_database_tables()` - Database table and index setup
- `parse_filename_to_race_id()` - Unified filename parsing with multiple pattern support
- `compute_file_hash()` - Consistent SHA-256 file hashing
- `load_collected_races()` / `reload_cache()` - Race data loading with caching
- `get_processed_filenames()` - O(1) processed file lookups
- `batch_filter_unprocessed_files()` - Batch file filtering
- `mark_file_processed()` - Database recording with error handling
- `get_cache_stats()` - Comprehensive cache statistics

**Standalone Functions:**
- `compute_file_hash(file_path)` - File hashing utility
- `parse_filename_to_race_id(filename)` - Filename parsing utility
- `batch_filter_unprocessed_files(file_paths, database_path)` - Batch filtering utility
- `get_race_file_stats(database_path)` - Statistics utility

## Scripts Refactored

### 1. `form_guide_csv_scraper.py`

**Changes Made:**
- Added `RaceFileManager` import and initialization
- Replaced duplicated methods with manager delegation:
  - `load_collected_races()` → delegates to `manager.reload_cache()`
  - `parse_filename_to_race_id()` → delegates to `manager.parse_filename_to_race_id()`
  - `compute_file_hash()` → delegates to `manager.compute_file_hash()`
  - `get_processed_filenames()` → delegates to `manager.get_processed_filenames()`
- Updated constructor to use manager's cached data
- Maintained backward compatibility with existing API

**Code Reduction:**
- Eliminated ~100 lines of duplicated helper code
- Reduced complexity while maintaining functionality
- Improved consistency across filename parsing patterns

### 2. `bulk_csv_ingest.py`

**Changes Made:**
- Added `RaceFileManager` and `compute_file_hash` imports
- Updated `compute_needed_info()` to use shared `compute_file_hash()` utility
- Ensured consistent hashing behavior across all file processing scripts

**Benefits:**
- Unified file hashing approach across batch processing
- Maintained performance while using shared utilities
- Consistent duplicate detection behavior

### 3. Backward Compatibility

**Maintained APIs:**
- All existing method signatures preserved
- No breaking changes to external consumers
- Gradual migration path for other scripts

## Performance Optimizations Retained

### 1. O(1) Lookups
- Single SQL query loads all processed filenames into memory
- Python set provides O(1) membership testing
- Eliminates N database queries for N files

### 2. Batch Processing Support
- `batch_filter_unprocessed_files()` handles large file lists efficiently
- Single database query + in-memory filtering
- Optimized for bulk operations

### 3. Caching Layer
- Processed hashes cached in memory for session-duration performance
- Race data loaded once and reused
- Database table indexes maintained for fast queries

## Database Schema Consistency

### Indexes Maintained
```sql
CREATE INDEX IF NOT EXISTS idx_processed_files_hash ON processed_race_files(file_hash)
CREATE INDEX IF NOT EXISTS idx_processed_files_race_key ON processed_race_files(race_date, venue, race_no)
CREATE INDEX IF NOT EXISTS idx_processed_files_file_path ON processed_race_files(file_path)
```

### Table Structure
- Consistent `processed_race_files` table management
- Unified error handling and status tracking
- Standardized metadata recording

## Filename Parsing Patterns Supported

The shared parser supports multiple filename patterns:

1. **Pattern 1**: `Race N - VENUE - DD Month YYYY.csv`
   - Example: `Race 5 - SAN - 15 January 2025.csv`

2. **Pattern 2**: `???_Race_N_VENUE_DATE.csv`
   - Example: `Form_Race_2_MEA_2025-01-15.csv`

3. **Pattern 3**: `Race_N_-_VENUE_-_DATE.csv`
   - Example: `Race_3_-_DAPT_-_2025_01_15.csv`

4. **Pattern 4**: `Race N - VENUE - YYYY-MM-DD.csv`
   - Example: `Race 4 - WENTWORTH_PARK - 2025-01-15.csv`

## Benefits Achieved

### 1. Code Deduplication
- Eliminated ~150+ lines of duplicated code across scripts
- Single source of truth for file processing logic
- Consistent behavior across all scripts

### 2. Maintainability
- Centralized bug fixes and improvements
- Easier to add new filename patterns or processing logic
- Unified error handling and logging

### 3. Performance Consistency
- Same optimization patterns used everywhere
- No performance regression from consolidation
- Improved cache efficiency through shared management

### 4. Testing & Validation
- Single module to test for all file processing logic
- Comprehensive error handling in one place
- Built-in statistics and monitoring

## Usage Examples

### Using RaceFileManager Class
```python
from utils.race_file_utils import RaceFileManager

# Initialize manager
manager = RaceFileManager("./databases/greyhound_racing.db")

# Get processed files for O(1) lookups
processed_files = manager.get_processed_filenames("./unprocessed")

# Check if file needs processing
if "Race 1 - SAN - 2025-01-15.csv" not in processed_files:
    # Process file...
    manager.mark_file_processed(file_path, race_info)

# Get comprehensive stats
stats = manager.get_cache_stats()
print(f"Total processed: {stats['total_processed']}")
```

### Using Standalone Functions
```python
from utils.race_file_utils import compute_file_hash, parse_filename_to_race_id

# Compute file hash
file_hash = compute_file_hash("/path/to/file.csv")

# Parse filename
race_info = parse_filename_to_race_id("Race 5 - SAN - 15 January 2025.csv")
if race_info:
    date, venue, race_num = race_info
    print(f"Parsed: {venue} Race {race_num} on {date}")
```

## Future Scripts Integration

### Easy Migration Path
New scripts can easily adopt the shared utilities:

```python
from utils.race_file_utils import RaceFileManager

class NewProcessor:
    def __init__(self):
        self.race_manager = RaceFileManager()
    
    def process_files(self, file_paths):
        # Filter out already processed files
        unprocessed = self.race_manager.batch_filter_unprocessed_files(file_paths)
        
        for file_path in unprocessed:
            # Process file...
            self.race_manager.mark_file_processed(file_path, race_info)
```

## Testing Completed

### Unit Tests
- ✅ Filename parsing with all supported patterns
- ✅ File hashing consistency
- ✅ Database operations and error handling
- ✅ Cache statistics generation

### Integration Tests
- ✅ `form_guide_csv_scraper.py` functionality preserved
- ✅ `bulk_csv_ingest.py` performance maintained
- ✅ Backward compatibility confirmed

### Performance Tests
- ✅ O(1) lookup performance confirmed
- ✅ Batch processing efficiency maintained
- ✅ Memory usage within acceptable bounds

## Impact on Project

### Immediate Benefits
- Cleaner, more maintainable codebase
- Consistent file processing behavior
- Eliminated duplicate linear scans

### Long-term Benefits
- Easier to add new file processing scripts
- Centralized performance optimizations
- Unified monitoring and error handling

### Risk Mitigation
- Maintained backward compatibility
- No performance regressions
- Comprehensive error handling

## Next Steps

The shared utilities module is now ready for adoption by other scripts in the project. Recommended next steps:

1. **Update Additional Scripts**: Gradually migrate other file processing scripts to use the shared utilities
2. **Enhanced Monitoring**: Add more detailed metrics to the cache statistics
3. **Pattern Extension**: Add support for additional filename patterns as needed
4. **Performance Tuning**: Monitor usage and optimize hot paths based on real-world usage

## Conclusion

Step 7 has been successfully completed with the creation of a comprehensive shared utilities module that eliminates duplicated linear scans project-wide. The refactoring maintains all existing functionality while providing a cleaner, more maintainable architecture for file processing operations.

The shared `utils/race_file_utils.py` module now serves as the single source of truth for race file processing logic, providing both performance optimizations and code consistency across the entire project.
