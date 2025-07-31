# Greyhound Racing Datetime Parsing Fix

## Overview
This document outlines the critical datetime parsing fix implemented to handle ISO format timestamps with microseconds and 'T' separators in the greyhound racing application.

## Problem Statement
The application was experiencing JSON serialization errors when processing `extraction_timestamp` values in the following format:
```
2025-07-23T19:13:28.830973
```

This format contains:
- ISO date format with 'T' separator
- Microseconds (6 decimal places)
- Non-standard format that wasn't handled by the original parsing logic

## Solution Overview
Implemented robust datetime parsing logic that:
1. Removes microseconds from timestamps
2. Handles both ISO format (`YYYY-MM-DDTHH:MM:SS`) and standard format (`YYYY-MM-DD HH:MM:SS`)
3. Provides consistent output format for display (`YYYY-MM-DD HH:MM`)
4. Includes proper error handling and logging

## Files Modified

### 1. app.py - `/api/races/paginated` endpoint (lines 753-765)
```python
# Format extraction timestamp
extraction_time = race[12]
if extraction_time:
    try:
        # Handle various datetime formats, remove microseconds if present
        time_str = str(extraction_time).split('.')[0]
        
        # Try different datetime formats
        if 'T' in time_str:
            # ISO format: 2025-07-23T19:13:28
            dt = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')
        else:
            # Standard format: 2025-07-23 19:13:28
            dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        
        formatted_time = dt.strftime('%Y-%m-%d %H:%M')
    except Exception as e:
        logger.warning(f"Failed to parse extraction time '{extraction_time}': {e}")
        formatted_time = str(extraction_time) if extraction_time else 'Unknown'
else:
    formatted_time = 'Unknown'
```

### 2. app.py - DatabaseManager.get_paginated_races method (lines 1202-1214)
```python
# Format extraction timestamp for better display
extraction_time = race[12]
if extraction_time:
    try:
        # Handle various datetime formats, remove microseconds if present
        time_str = str(extraction_time).split('.')[0]
        
        # Try different datetime formats
        if 'T' in time_str:
            # ISO format: 2025-07-23T19:13:28
            dt = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')
        else:
            # Standard format: 2025-07-23 19:13:28
            dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        
        formatted_time = dt.strftime('%Y-%m-%d %H:%M')
    except Exception as e:
        logger.warning(f"Failed to parse extraction time '{extraction_time}': {e}")
        formatted_time = str(extraction_time) if extraction_time else 'Unknown'
else:
    formatted_time = 'Unknown'
```

## Implementation Details

### Step 1: Microsecond Removal
```python
time_str = str(extraction_time).split('.')[0]
```
This removes microseconds by splitting on the decimal point and taking only the first part.

### Step 2: Format Detection
```python
if 'T' in time_str:
    # ISO format
else:
    # Standard format
```
Detects whether the timestamp uses ISO format (with 'T') or standard format (with space).

### Step 3: Parsing
- ISO format: `%Y-%m-%dT%H:%M:%S`
- Standard format: `%Y-%m-%d %H:%M:%S`

### Step 4: Output Formatting
```python
formatted_time = dt.strftime('%Y-%m-%d %H:%M')
```
Standardizes all timestamps to `YYYY-MM-DD HH:MM` format for consistent display.

## Error Handling
- Comprehensive try-catch blocks around parsing logic
- Warning logs for debugging failed parsing attempts
- Graceful fallback to original string value if parsing fails
- Null/None value handling

## Frontend Impact
The frontend JavaScript code automatically handles the standardized datetime format. No changes required to existing frontend components as they expect the `YYYY-MM-DD HH:MM` format.

## Testing Scenarios
The fix handles these timestamp formats:
- `2025-07-23T19:13:28.830973` (ISO with microseconds)
- `2025-07-23T19:13:28` (ISO without microseconds)
- `2025-07-23 19:13:28` (Standard format)
- `null` or empty values
- Invalid/malformed timestamps

## Future Development Notes

### ⚠️ CRITICAL FOR DEVELOPERS ⚠️
When working with extraction timestamps in ANY part of the application:

1. **Always use the parsing logic from this documentation**
2. **Never assume timestamp format - always handle both ISO and standard formats**
3. **Always remove microseconds before parsing**
4. **Include proper error handling with logging**
5. **Test with various timestamp formats**

### Code Template for Datetime Parsing
```python
def parse_extraction_timestamp(timestamp_value):
    """
    Parse extraction timestamp handling various formats.
    
    Args:
        timestamp_value: Raw timestamp value from database
        
    Returns:
        str: Formatted timestamp as 'YYYY-MM-DD HH:MM' or 'Unknown'
    """
    if not timestamp_value:
        return 'Unknown'
    
    try:
        # Handle various datetime formats, remove microseconds if present
        time_str = str(timestamp_value).split('.')[0]
        
        # Try different datetime formats
        if 'T' in time_str:
            # ISO format: 2025-07-23T19:13:28
            dt = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')
        else:
            # Standard format: 2025-07-23 19:13:28
            dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        
        return dt.strftime('%Y-%m-%d %H:%M')
    except Exception as e:
        logger.warning(f"Failed to parse extraction time '{timestamp_value}': {e}")
        return str(timestamp_value) if timestamp_value else 'Unknown'
```

## Migration and Rollback
- **No database migration required** - this is a display formatting fix
- **Backward compatible** - handles existing standard format timestamps
- **Forward compatible** - handles new ISO format timestamps
- **Rollback safe** - removing the fix will not break existing data

## Monitoring
Watch for these log messages:
```
WARNING: Failed to parse extraction time '...'
```
These indicate new timestamp formats that may need additional handling.

## Related Files
- `app.py` - Main application with API endpoints
- `templates/races.html` - Frontend race display template
- `static/js/interactive-races.js` - Frontend JavaScript for race interactions

## Version History
- **v1.0** (2025-01-31): Initial implementation of robust datetime parsing for extraction timestamps

---

**⚠️ IMPORTANT: Always reference this documentation when working with datetime parsing in the greyhound racing application to maintain consistency and prevent regression.**
