# Robust Caching & De-duplication System

## Overview

This document describes the robust caching and de-duplication system implemented in FormGuideCsvScraper to ensure we **never parse/process the same race twice**. The system uses SHA-256 file hashing and database tracking to achieve efficient, persistent caching.

## Key Features

### 🎯 Core Functionality
- **SHA-256 File Hashing**: Every CSV file is hashed to create a unique fingerprint
- **Database Persistence**: Cache information survives application restarts
- **Intelligent De-duplication**: Identical content is detected regardless of filename
- **Force Override**: `--force` flag allows bypassing cache when needed
- **Cache Statistics**: Detailed reporting on cache performance

### 📊 Database Schema

The system adds a new table `processed_race_files`:

```sql
CREATE TABLE processed_race_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_hash TEXT UNIQUE NOT NULL,           -- SHA-256 hash
    race_date DATE NOT NULL,                  -- Extracted from filename
    venue TEXT NOT NULL,                      -- Racing venue code
    race_no INTEGER NOT NULL,                 -- Race number
    file_path TEXT NOT NULL,                  -- Original file path
    file_size INTEGER,                        -- File size in bytes
    processed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'processed',          -- processed, failed, skipped
    error_message TEXT                        -- Error details if failed
);
```

### 🚀 Performance Optimizations

**Indexes Created:**
- `idx_processed_files_hash` - Fast hash lookups
- `idx_processed_files_race_key` - Race-based queries
- `idx_processed_files_processed_at` - Temporal queries

## Usage Examples

### Basic Usage

```python
from form_guide_csv_scraper import FormGuideCsvScraper

scraper = FormGuideCsvScraper()

# Process a file (cache miss on first run)
result = scraper.parse_csv_with_ingestion("race_file.csv")
print(result)  # "miss"

# Process same file again (cache hit)
result = scraper.parse_csv_with_ingestion("race_file.csv")
print(result)  # "hit"

# Force reprocessing
result = scraper.parse_csv_with_ingestion("race_file.csv", force=True)
print(result)  # "miss"
```

### Command-Line Interface

```bash
# Show cache statistics
python3 form_guide_csv_scraper.py --stats

# Test caching with a specific file
python3 form_guide_csv_scraper.py --test-file path/to/race.csv

# Force reprocessing (ignore cache)
python3 form_guide_csv_scraper.py --test-file path/to/race.csv --force

# Show help
python3 form_guide_csv_scraper.py --help
```

## Cache Flow Diagram

```
┌─────────────────┐
│   CSV File      │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Compute SHA-256 │
│     Hash        │
└─────────────────┘
         │
         ▼
┌─────────────────┐    Yes   ┌─────────────────┐
│ Hash in Cache?  │─────────▶│  Return "hit"   │
└─────────────────┘          └─────────────────┘
         │ No
         ▼
┌─────────────────┐
│ Process CSV     │
│ using CsvIng.   │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Store hash in   │
│ database cache  │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Return "miss"   │
└─────────────────┘
```

## Cache Status Returns

The `parse_csv_with_ingestion()` method returns:

- **`"hit"`** - File was previously processed (cache hit)
- **`"miss"`** - File processed for first time (cache miss)  
- **`"error"`** - Processing failed (logged to database)

## Implementation Details

### File Hash Computation

```python
def compute_file_hash(self, file_path):
    """Compute SHA-256 hash of a file"""
    import hashlib
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read in 4KB chunks for memory efficiency
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()
```

### Cache Check Logic

```python
def parse_csv_with_ingestion(self, file_path, force=False):
    file_hash = self.compute_file_hash(file_path)
    
    # Check cache (unless force=True)
    if file_hash in self.processed_hashes and not force:
        return "hit"
    
    # Process file and cache result
    # ... processing logic ...
    
    self.processed_hashes.add(file_hash)
    return "miss"
```

## Database Migration

The system includes automatic table creation and a migration script:

```bash
# Run migration on all databases
python3 migrations/add_processed_race_files_table.py
```

### Migration Features:
- ✅ Detects existing tables (safe to run multiple times)
- 🎯 Migrates all found databases automatically  
- 📊 Provides detailed progress reporting
- 🔒 Graceful error handling

## Performance Benefits

### Before Caching
- ❌ Reprocessed same files repeatedly
- ❌ Wasted CPU cycles on duplicate content
- ❌ No persistence across sessions
- ❌ No duplicate detection

### After Caching
- ✅ **Instant cache hits** for processed files
- ✅ **Persistent cache** survives application restarts
- ✅ **Content-based deduplication** (same content = cache hit)
- ✅ **Database tracking** with metadata
- ✅ **Configurable force override**

## Cache Statistics Example

```bash
$ python3 form_guide_csv_scraper.py --stats

📈 Cache Statistics:
   Total processed files: 1,247
   Unique venues: 47
   Processed: 1,198
   Failed: 49
   Date range: 2025-01-01 12:34:56 to 2025-08-03 21:34:12
```

## Testing

Comprehensive test suite validates:

- ✅ Hash computation accuracy
- ✅ Cache hit/miss logic
- ✅ Database persistence
- ✅ Force override functionality  
- ✅ Duplicate content detection
- ✅ Error handling

Run tests:
```bash
python3 test_caching_system.py
```

## Integration Points

### FormGuideCsvScraper.load_collected_races()

Updated to:
- Load processed hashes from database
- Create tables if missing
- Initialize in-memory cache
- Report cache statistics

### Database Integration

- Seamlessly integrates with existing SQLite databases
- Uses `INSERT OR REPLACE` for safe updates
- Maintains referential integrity
- Optimized with strategic indexes

## Force Flag Behavior

The `--force` flag provides complete cache bypass:

```python
# Normal processing (respects cache)
result = scraper.parse_csv_with_ingestion(file_path, force=False)

# Force processing (ignores cache) 
result = scraper.parse_csv_with_ingestion(file_path, force=True)
```

**Use Cases for Force:**
- 🔄 Reprocessing after algorithm updates
- 🧪 Testing and debugging
- 🚀 Processing format changes
- 📊 Performance benchmarking

## Error Handling

The system gracefully handles:

- **Missing CsvIngestion classes**: Falls back to basic CSV parsing
- **Database connection issues**: Continues with in-memory cache only
- **File permission errors**: Logs errors to database
- **Corrupt files**: Records failure status
- **Network interruptions**: Preserves partial cache state

## Security Considerations

- **SHA-256 hashing** provides cryptographic integrity
- **File content verification** prevents cache poisoning
- **Database sanitization** prevents SQL injection
- **Path validation** prevents directory traversal

## Future Enhancements

Potential improvements:
- 🌐 **Distributed caching** across multiple machines
- 🗜️ **Compression** for large cache databases
- 📈 **Cache analytics** and optimization suggestions
- 🔄 **Cache expiration** policies
- 📦 **Export/import** cache data

## Troubleshooting

### Common Issues

**Cache not working:**
```bash
# Check database permissions
ls -la databases/
# Verify table exists
sqlite3 databases/greyhound_racing.db ".schema processed_race_files"
```

**Performance issues:**
```bash
# Check cache statistics
python3 form_guide_csv_scraper.py --stats
# Rebuild indexes if needed
```

**Hash mismatches:**
- Ensure file encoding consistency (UTF-8)
- Check for BOM (Byte Order Mark) differences
- Verify line ending normalization

## Summary

The robust caching and de-duplication system delivers:

- 🎯 **Zero duplicate processing** - guaranteed via SHA-256 hashing
- ⚡ **Instant cache hits** - sub-millisecond lookups
- 💾 **Persistent storage** - survives application restarts  
- 🔧 **Force override** - complete cache bypass when needed
- 📊 **Detailed reporting** - comprehensive cache statistics
- 🚀 **Production ready** - comprehensive error handling and testing

**Result: Never parse/predict the same race twice!** ✅
