"""
CSV metadata extraction utilities for greyhound race files.
Provides lightweight metadata extraction from race CSV files with fallback to filename parsing.
"""

import os
import re
import csv
from datetime import datetime
from typing import Dict, Any, Optional, Union
# Optional heavy dependency - make pandas optional in constrained test envs
try:
    import pandas as pd  # noqa: F401
except Exception:  # pragma: no cover
    pd = None


def parse_race_csv_meta(file_path: str) -> Dict[str, Any]:
    """
    Extract lightweight metadata from race CSV files with intelligent fallback.
    
    Handles common column aliases and fallback to regex parsing of filename
    patterns like "Race 11 - TAREE - 2025-08-02.csv". Gracefully skips 
    malformed files and includes "status":"error" entry in response if needed.
    
    Args:
        file_path: Path to the CSV file to analyze
        
    Returns:
        Dict containing extracted metadata with the following structure:
        {
            "race_number": int,           # Race number (from filename or data)
            "venue": str,                 # Racing venue/track
            "race_date": str,            # Race date (YYYY-MM-DD format)
            "distance": str,             # Race distance (from CSV data)
            "grade": str,                # Race grade/class (from CSV data) 
            "field_size": int,           # Number of runners
            "source": str,               # "csv_data", "filename", or "mixed"
            "status": str,               # "success" or "error"
            "error_message": str,        # Error details (if status="error")
            "filename": str,             # Original filename
            "file_exists": bool,         # Whether file exists
            "file_size": int            # File size in bytes (if exists)
        }
        
    Examples:
        >>> parse_race_csv_meta("Race 11 - TAREE - 2025-08-02.csv")
        {
            "race_number": 11,
            "venue": "TAREE", 
            "race_date": "2025-08-02",
            "distance": "300",
            "grade": "5",
            "field_size": 8,
            "source": "mixed",
            "status": "success",
            "error_message": "",
            "filename": "Race 11 - TAREE - 2025-08-02.csv",
            "file_exists": True,
            "file_size": 4521
        }
    """
    
    # Initialize response with defaults
    response = {
        "race_number": 0,
        "venue": "Unknown",
        "race_date": "Unknown", 
        "distance": "Unknown",
        "grade": "Unknown",
        "field_size": 0,
        "source": "unknown",
        "status": "success",
        "error_message": "",
        "filename": os.path.basename(file_path),
        "file_exists": False,
        "file_size": 0
    }
    
    try:
        # Check if file exists and get basic file info
        if os.path.exists(file_path):
            response["file_exists"] = True
            response["file_size"] = os.path.getsize(file_path)
        else:
            response["status"] = "error"
            response["error_message"] = f"File not found: {file_path}"
            return response
            
        # Track what data sources we use
        data_sources = []
        
        # Step 1: Extract what we can from filename using regex
        filename_meta = _extract_from_filename(os.path.basename(file_path))
        if filename_meta:
            response.update(filename_meta)
            data_sources.append("filename")
            
        # Step 2: Try to extract from CSV data with error handling
        try:
            csv_meta = _extract_from_csv_data(file_path)
            if csv_meta:
                # Merge CSV data, preferring CSV over filename for data fields
                # but keeping filename data for race identification
                for key, value in csv_meta.items():
                    if value and value != "Unknown":  # Only override with valid CSV data
                        response[key] = value
                data_sources.append("csv_data")
        except Exception as csv_error:
            # CSV parsing failed, but we might still have filename data
            response.update({
                "status": "error" if not data_sources else "success",
                "error_message": f"CSV parsing failed: {str(csv_error)}"
            })
            
        # Step 3: Set source indicator
        if len(data_sources) > 1:
            response["source"] = "mixed"
        elif data_sources:
            response["source"] = data_sources[0]
        else:
            response["source"] = "none"
            response["status"] = "error"
            response["error_message"] = "No metadata could be extracted"
            
        return response
        
    except Exception as e:
        # Catch-all for any unexpected errors
        response.update({
            "status": "error",
            "error_message": f"Unexpected error: {str(e)}",
            "source": "error"
        })
        return response


def _extract_from_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    Extract metadata from filename using regex patterns.
    
    Supports patterns like:
    - "Race 11 - TAREE - 2025-08-02.csv"
    - "Race 5 - GEE - 08 July 2025.csv" 
    - "20250730162231_Race 3 - TAR - 28 June 2025.csv"
    
    Args:
        filename: The filename to parse
        
    Returns:
        Dict with extracted metadata or None if no pattern matches
    """
    
    # Remove any timestamp prefix and .csv extension
    clean_name = re.sub(r'^\d+_', '', filename)  # Remove timestamp prefix
    clean_name = re.sub(r'\.csv$', '', clean_name, flags=re.IGNORECASE)  # Remove extension
    
    # Pattern 1: "Race 11 - TAREE - 2025-08-02" (ISO date format)
    pattern1 = r'Race\s+(\d+)\s*-\s*([A-Z_]+)\s*-\s*(\d{4}-\d{2}-\d{2})'
    match1 = re.search(pattern1, clean_name, re.IGNORECASE)
    
    if match1:
        race_num, venue, date_str = match1.groups()
        return {
            "race_number": int(race_num),
            "venue": venue.upper(),
            "race_date": date_str  # Already in YYYY-MM-DD format
        }
    
    # Pattern 2: "Race 5 - GEE - 08 July 2025" (human readable date)
    pattern2 = r'Race\s+(\d+)\s*-\s*([A-Z_]+)\s*-\s*(\d{1,2})\s+(\w+)\s+(\d{4})'
    match2 = re.search(pattern2, clean_name, re.IGNORECASE)
    
    if match2:
        race_num, venue, day, month_name, year = match2.groups()
        
        # Convert month name to number
        date_str = _parse_human_date(day, month_name, year)
        if date_str:
            return {
                "race_number": int(race_num),
                "venue": venue.upper(),
                "race_date": date_str
            }
    
    # Pattern 3: Try to extract just race number and venue if date parsing fails
    pattern3 = r'Race\s+(\d+)\s*-\s*([A-Z_]+)'
    match3 = re.search(pattern3, clean_name, re.IGNORECASE)
    
    if match3:
        race_num, venue = match3.groups()
        return {
            "race_number": int(race_num),
            "venue": venue.upper(),
            "race_date": "Unknown"
        }
    
    return None


def _parse_human_date(day: str, month_name: str, year: str) -> Optional[str]:
    """
    Convert human readable date to YYYY-MM-DD format.
    
    Args:
        day: Day of month (e.g., "08", "28")
        month_name: Month name (e.g., "July", "June")
        year: Year (e.g., "2025")
        
    Returns:
        Date string in YYYY-MM-DD format or None if parsing fails
    """
    
    month_mapping = {
        'january': '01', 'jan': '01',
        'february': '02', 'feb': '02', 
        'march': '03', 'mar': '03',
        'april': '04', 'apr': '04',
        'may': '05',
        'june': '06', 'jun': '06',
        'july': '07', 'jul': '07',
        'august': '08', 'aug': '08',
        'september': '09', 'sep': '09', 'sept': '09',
        'october': '10', 'oct': '10',
        'november': '11', 'nov': '11',
        'december': '12', 'dec': '12'
    }
    
    month_num = month_mapping.get(month_name.lower())
    if month_num:
        day_padded = day.zfill(2)  # Ensure 2-digit day
        return f"{year}-{month_num}-{day_padded}"
    
    return None


def _extract_from_csv_data(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Extract metadata from CSV file contents.
    
    Looks for common columns like TRACK, DIST, G (grade), and analyzes
    the data to determine race characteristics.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Dict with extracted metadata or None if extraction fails
    """
    
    try:
        # Try pandas first for robust CSV handling
        if pd is None:
            raise ImportError("pandas not available")
        df = pd.read_csv(file_path, nrows=50)  # Only read first 50 rows for efficiency
        
        # Clean up the dataframe - remove rows where all values are empty quotes
        df = df.replace('""', '')  # Replace empty quotes with empty strings
        df = df.replace('', pd.NA)  # Convert empty strings to NaN
        
        result = {}
        
        # Extract venue from TRACK column (most reliable source)
        if 'TRACK' in df.columns:
            venues = df['TRACK'].dropna().unique()
            # Get the most common venue (in case of mixed data)
            if len(venues) > 0:
                venue_counts = df['TRACK'].value_counts()
                raw_venue = str(venue_counts.index[0]).upper()
                result['venue'] = standardize_venue_name(raw_venue)
        
        # Extract distance from DIST column
        if 'DIST' in df.columns:
            distances = df['DIST'].dropna().unique()
            if len(distances) > 0:
                # Get most common distance
                distance_counts = df['DIST'].value_counts()
                result['distance'] = str(distance_counts.index[0])
        
        # Extract grade from G column
        if 'G' in df.columns:
            grades = df['G'].dropna().unique()
            if len(grades) > 0:
                # Get most common grade
                grade_counts = df['G'].value_counts()
                result['grade'] = str(grade_counts.index[0])
        
        # Calculate field size (number of unique dogs/boxes)
        if 'BOX' in df.columns:
            boxes = df['BOX'].dropna().unique()
            result['field_size'] = len(boxes)
        elif 'Dog Name' in df.columns:
            # Count unique dogs (excluding empty quotes and NaN)
            dogs = df['Dog Name'].dropna()
            dogs = dogs[dogs != '""']  # Remove empty quotes
            # Only count dogs that don't start with a number followed by period (these are the primary entries)
            primary_dogs = dogs[dogs.str.match(r'^\d+\.\s')]
            result['field_size'] = len(primary_dogs)
        
        # Try to extract race date from DATE column
        if 'DATE' in df.columns:
            dates = df['DATE'].dropna().unique()
            if len(dates) > 0:
                # Get most common date and try to parse it
                date_counts = df['DATE'].value_counts()
                most_common_date = str(date_counts.index[0])
                
                # Try to parse the date into standard format
                parsed_date = _standardize_date(most_common_date)
                if parsed_date:
                    result['race_date'] = parsed_date
        
        return result if result else None
        
    except Exception as e:
        # If pandas fails, try basic CSV reader as fallback
        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                # Read first few rows to analyze
                rows = []
                for i, row in enumerate(reader):
                    if i >= 20:  # Only read first 20 rows
                        break
                    rows.append(row)
                
                if not rows:
                    return None
                
                result = {}
                
                # Simple extraction from first few rows
                for row in rows:
                    if 'TRACK' in row and row['TRACK'] and row['TRACK'] != '""':
                        result['venue'] = row['TRACK'].upper()
                        break
                
                return result if result else None
                
        except Exception as fallback_error:
            # Both pandas and basic CSV failed
            return None


def _standardize_date(date_str: str) -> Optional[str]:
    """
    Convert various date formats to YYYY-MM-DD standard format.
    
    Args:
        date_str: Date string in various formats
        
    Returns:
        Standardized date string or None if parsing fails
    """
    
    # List of common date formats to try
    date_formats = [
        '%Y-%m-%d',        # 2025-08-02
        '%d/%m/%Y',        # 02/08/2025  
        '%m/%d/%Y',        # 08/02/2025
        '%d-%m-%Y',        # 02-08-2025
        '%Y%m%d',          # 20250802
        '%d %B %Y',        # 02 August 2025
        '%d %b %Y',        # 02 Aug 2025
    ]
    
    for fmt in date_formats:
        try:
            parsed_date = datetime.strptime(date_str, fmt)
            return parsed_date.strftime('%Y-%m-%d')
        except ValueError:
            continue
    
    return None


# Venue name standardization mapping
VENUE_ALIASES = {
    'TARE': 'TAREE',
    'TAR': 'TAREE', 
    'BEN': 'BENDIGO',
    'GEE': 'GEELONG',
    'BAL': 'BALLARAT',
    'WAR': 'WARRNAMBOOL',
    'SHE': 'SHEPPARTON',
    'MEA': 'THE_MEADOWS',
    'SAN': 'SANDOWN',
    'HOR': 'HORSHAM',
    'RICH': 'RICHMOND',
    'GRDN': 'THE_GARDENS',
    'GOSF': 'GOSFORD',
    'MAIT': 'MAITLAND',
    'RICS': 'RICHMOND',
    'MUSW': 'MUSWELLBROOK',
    'NOWR': 'NOWRA',
    'PPK': 'PENRITH',
    'GUNN': 'GUNNEDAH',
    'AP_K': 'ANGLE_PARK',
    'CANN': 'CANNINGTON',
    'MAND': 'MANDURAH',
    'MURR': 'MURRAY_BRIDGE',
    'SAL': 'SALE',
    'MOUNT': 'MOUNT_GAMBIER',
    'HEA': 'HEALESVILLE',
    'W_PK': 'WENTWORTH_PARK',
    'DAPT': 'DAPTO'
}


def standardize_venue_name(venue: str) -> str:
    """
    Standardize venue names using common aliases.
    
    Args:
        venue: Raw venue name from filename or CSV
        
    Returns:
        Standardized venue name
    """
    
    venue_upper = venue.upper().strip()
    return VENUE_ALIASES.get(venue_upper, venue_upper)
