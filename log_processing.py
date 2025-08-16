import pandas as pd
import json
import glob
import os
import re
from datetime import datetime

# Function to extract race ID from filename or log content
def extract_race_id(filename, content=None):
    """Extract race ID from filename or content"""
    # Try to extract from filename first
    patterns = [
        r'Race[_\s]+([0-9]+)[_\s]*-[_\s]*([A-Z]+)[_\s]*-[_\s]*([0-9]{1,2}[_\s]+[A-Za-z]+[_\s]+[0-9]{4})',  # Race 1 - AP_K - 01 July 2025
        r'Race_([0-9]+)_UNKNOWN_([0-9]{4}-[0-9]{2}-[0-9]{2})',  # Race_01_UNKNOWN_2025-07-26
        r'([A-Za-z_0-9]+)\.csv'  # Generic CSV filename
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            if len(match.groups()) >= 3:
                return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
            elif len(match.groups()) >= 2:
                return f"{match.group(1)}-{match.group(2)}"
            else:
                return match.group(1)
    
    # Fallback to filename without extension
    return os.path.splitext(os.path.basename(filename))[0]

# Function to load JSONL log files (line-by-line JSON)
def load_jsonl_to_dataframe(file_path):
    """Load JSONL files where each line is a JSON object"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = []
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if line:
                    try:
                        parsed = json.loads(line)
                        parsed['source_file'] = os.path.basename(file_path)
                        parsed['line_number'] = line_num
                        data.append(parsed)
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error in {file_path} line {line_num}: {e}")
                        continue
        
        if data:
            df = pd.json_normalize(data)
            # Normalize timestamps
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            # Extract race ID
            df['race_id'] = extract_race_id(file_path)
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

# Function to load JSON log files (single JSON object)
def load_json_to_dataframe(file_path):
    """Load JSON files with single JSON object containing arrays"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Handle different JSON structures
        records = []
        
        if isinstance(data, dict):
            # Check for common log structure patterns
            for key in ['process', 'system', 'errors']:
                if key in data and isinstance(data[key], list):
                    for record in data[key]:
                        record['log_type'] = key
                        record['source_file'] = os.path.basename(file_path)
                        records.append(record)
        elif isinstance(data, list):
            for record in data:
                record['source_file'] = os.path.basename(file_path)
                records.append(record)
        
        if records:
            df = pd.json_normalize(records)
            # Normalize timestamps
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            # Extract race ID
            df['race_id'] = extract_race_id(file_path)
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

# Function to load all log files from directories
def load_all_race_logs():
    """Load all race-related log files from various directories"""
    log_directories = ['./logs', './debug_logs', './tests/logs', './repair_logs', './diagnostic_logs']
    
    all_dataframes = []
    
    for log_dir in log_directories:
        if os.path.exists(log_dir):
            print(f"Processing directory: {log_dir}")
            
            # Look for JSONL files
            jsonl_files = glob.glob(os.path.join(log_dir, '*.jsonl'))
            for file_path in jsonl_files:
                print(f"Loading JSONL: {file_path}")
                df = load_jsonl_to_dataframe(file_path)
                if not df.empty:
                    all_dataframes.append(df)
            
            # Look for JSON files
            json_files = glob.glob(os.path.join(log_dir, '*.json'))
            for file_path in json_files:
                print(f"Loading JSON: {file_path}")
                df = load_json_to_dataframe(file_path)
                if not df.empty:
                    all_dataframes.append(df)
    
    # Combine all dataframes
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
        
        # Final timestamp normalization
        if 'timestamp' in combined_df.columns:
            combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], errors='coerce')
        
        # Sort by timestamp
        combined_df = combined_df.sort_values('timestamp', na_position='last').reset_index(drop=True)
        
        return combined_df
    else:
        return pd.DataFrame()

# Load all race logs
print("Loading race logs from all directories...")
race_logs_df = load_all_race_logs()

print(f"\nLoaded {len(race_logs_df)} log entries from race logs")
print(f"Columns: {list(race_logs_df.columns)}")

if not race_logs_df.empty:
    print("\nFirst 5 entries:")
    print(race_logs_df.head())
    
    print("\nUnique race IDs found:")
    print(race_logs_df['race_id'].value_counts().head(10))
    
    print("\nTimestamp range:")
    if 'timestamp' in race_logs_df.columns:
        valid_timestamps = race_logs_df['timestamp'].dropna()
        if not valid_timestamps.empty:
            print(f"From: {valid_timestamps.min()}")
            print(f"To: {valid_timestamps.max()}")
    
    # Save to CSV for cross-reference with single-dog data
    race_logs_df.to_csv('race_logs_normalized.csv', index=False)
    print("\nRace logs saved to 'race_logs_normalized.csv'")
else:
    print("No race log data found.")
