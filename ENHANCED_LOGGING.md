# Enhanced Logging Features

This document describes the enhanced logging features implemented for greyhound racing data collection operations, specifically focusing on per-race logging with HTTP status codes and conditional verbosity.

## Overview

The enhanced logging system provides three key improvements:

1. **Per-race log lines** in a structured format: `[SKIP|CACHE|FETCH] 2025-07-25 AP_K R4 – reason`
2. **HTTP status codes** logged on every `fetches_attempted` operation
3. **Conditional logging** based on `verbose_fetch` flag, with warnings/errors always emitted

## Log Format

### Structured Log Line Format

```
[OPERATION] YYYY-MM-DD VENUE RN – reason (HTTP status)
```

Where:
- **OPERATION**: One of `SKIP`, `CACHE`, or `FETCH`
- **YYYY-MM-DD**: Race date in ISO format
- **VENUE**: Venue code (e.g., AP_K, SAN, MEA)
- **RN**: Race number (e.g., R4, R1, R12)
- **reason**: Descriptive reason for the operation
- **HTTP status**: Optional HTTP status code (when applicable)

### Examples

```
[CACHE] 2025-07-25 AP_K R4 – Race already collected
[FETCH] 2025-07-25 MEA R8 – CSV downloaded successfully (HTTP 200)
[FETCH] 2025-07-25 GEE R3 – CSV download failed - no CSV link found (HTTP 404)
[SKIP] invalid-date BEN R2 – Date parsing error: time data 'invalid-date' does not match expected format
```

## Usage

### In Python Code

```python
from logger import logger

# Cache hit (only logged if verbose_fetch=True)
logger.log_race_operation(
    race_date="2025-07-25",
    venue="AP_K",
    race_number="4",
    operation="CACHE",
    reason="Race already collected",
    verbose_fetch=True  # Conditional logging
)

# Successful fetch with HTTP status
logger.log_race_operation(
    race_date="2025-07-25",
    venue="MEA",
    race_number="8", 
    operation="FETCH",
    reason="CSV downloaded successfully",
    http_status=200,
    verbose_fetch=True
)

# Error (always logged regardless of verbose_fetch)
logger.log_race_operation(
    race_date="2025-07-25",
    venue="DAPT",
    race_number="6",
    operation="FETCH",
    reason="Exception during download: Connection timeout",
    http_status=500,
    verbose_fetch=False,  # Still logged because it's an ERROR
    level="ERROR"
)
```

### In Scrapers

The enhanced logging is integrated into the `FormGuideCsvScraper` class:

```python
# Run scraper with verbose fetch logging
scraper = FormGuideCsvScraper(verbose_fetch=True)

# Run scraper with minimal logging (warnings/errors only)
scraper = FormGuideCsvScraper(verbose_fetch=False)
```

### Command Line Usage

```bash
# Enable verbose fetch logging
python form_guide_csv_scraper.py --verbose-fetch

# Run with minimal logging (default)
python form_guide_csv_scraper.py

# Test the enhanced logging features
python example_enhanced_logging.py --verbose-fetch
```

## Conditional Logging Rules

The `verbose_fetch` parameter controls logging behavior:

| Level | verbose_fetch=True | verbose_fetch=False |
|-------|-------------------|-------------------|
| INFO | ✅ Logged | ❌ Not logged |
| WARNING | ✅ Logged | ✅ Always logged |
| ERROR | ✅ Logged | ✅ Always logged |

This ensures that important issues are never missed while allowing detailed operation tracking when needed.

## Integration with Statistics

The enhanced logging works seamlessly with existing fetch statistics:

```python
# Before HTTP call - track fetch attempt
self.stats.increment('fetches_attempted')

# Make HTTP request and capture status
success, http_status = self.download_csv_from_race_page(race_info)

# Log with HTTP status included
logger.log_race_operation(
    race_date=formatted_date,
    venue=race_info['venue'],
    race_number=str(race_info['race_number']),
    operation="FETCH",
    reason="CSV downloaded successfully" if success else "CSV download failed",
    http_status=http_status,
    verbose_fetch=self.verbose_fetch,
    level="INFO" if success else "WARNING"
)
```

## Log Destinations

Enhanced race operation logs are written to multiple destinations:

1. **Process Log File**: `logs/process.log` - Human-readable format
2. **Workflow JSONL**: `logs/main_workflow.jsonl` - Structured JSON format
3. **Web Accessible JSON**: `logs/web_access.json` - For web interfaces

## HTTP Status Code Tracking

HTTP status codes are captured from:
- Initial race page requests
- CSV download requests  
- Request exceptions (when available)

The most recent HTTP status code is always logged, providing visibility into web scraping issues.

## Example Scenarios

### Scenario 1: Cache Hit
```
Operation: CACHE
Statistics: races_requested++, cache_hits++
Log: [CACHE] 2025-07-25 AP_K R4 – Race already collected
HTTP Status: N/A
Logged when: verbose_fetch=True
```

### Scenario 2: Successful Fetch
```
Operation: FETCH  
Statistics: races_requested++, fetches_attempted++, successful_saves++
Log: [FETCH] 2025-07-25 MEA R8 – CSV downloaded successfully (HTTP 200)
HTTP Status: 200
Logged when: verbose_fetch=True OR level=WARNING/ERROR
```

### Scenario 3: Failed Fetch
```
Operation: FETCH
Statistics: races_requested++, fetches_attempted++, fetches_failed++
Log: [FETCH] 2025-07-25 GEE R3 – CSV download failed - no CSV link found (HTTP 404)
HTTP Status: 404
Logged when: Always (WARNING level)
```

### Scenario 4: Skip Due to Error
```
Operation: SKIP
Statistics: races_requested++, fetches_failed++
Log: [SKIP] invalid-date BEN R2 – Date parsing error: time data does not match format
HTTP Status: N/A  
Logged when: Always (ERROR level)
```

## Testing

Use the example script to test the enhanced logging features:

```bash
# Test with verbose logging
python example_enhanced_logging.py --verbose-fetch

# Test with minimal logging
python example_enhanced_logging.py

# Test statistics integration
python example_enhanced_logging.py --stats-integration
```

## Implementation Notes

- HTTP status codes are captured from both successful and failed requests
- The logging is thread-safe using locks
- Log entries are automatically rotated to prevent huge files
- Structured data is preserved for programmatic analysis
- The system is backward compatible with existing logging calls
