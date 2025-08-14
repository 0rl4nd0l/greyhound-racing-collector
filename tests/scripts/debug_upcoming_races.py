#!/usr/bin/env python3

import os
import sys
import pandas as pd
import hashlib
import re
from datetime import datetime

def _extract_csv_metadata(file_path):
    """Extract metadata from CSV filename using regex."""
    filename = os.path.basename(file_path)
    
    # Regex pattern to match: "Race {number} – {venue} – {date}.csv"
    pattern = r'Race\s+(\d+)\s*[–-]\s*([A-Z_]+)\s*[–-]\s*(\d{4}-\d{2}-\d{2})\.csv'
    
    match = re.match(pattern, filename, re.IGNORECASE)
    if match:
        return {
            'race_number': int(match.group(1)),
            'venue': match.group(2),
            'date': match.group(3)
        }
    
    # Fallback: try to extract individual components
    race_number = None
    venue = None
    date = None
    
    # Extract race number
    race_match = re.search(r'Race[_\s]+(\d+)', filename, re.IGNORECASE)
    if race_match:
        race_number = int(race_match.group(1))
    
    # Extract venue (look for uppercase 3-4 letter codes)
    venue_match = re.search(r'([A-Z_]{2,4})', filename)
    if venue_match:
        venue = venue_match.group(1)
    
    # Extract date (YYYY-MM-DD format)
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if date_match:
        date = date_match.group(1)
    
    return {
        'race_number': race_number,
        'venue': venue,
        'date': date
    }

def debug_upcoming_races():
    """Debug the upcoming races loading process"""
    upcoming_races_dir = "./upcoming_races"
    
    print(f"Checking directory: {upcoming_races_dir}")
    print(f"Directory exists: {os.path.exists(upcoming_races_dir)}")
    
    if not os.path.exists(upcoming_races_dir):
        print("Directory doesn't exist!")
        return
    
    try:
        files = os.listdir(upcoming_races_dir)
        print(f"Files in directory: {files}")
        
        csv_files = [f for f in files if f.endswith('.csv') and not f.startswith('.')]
        print(f"CSV files: {csv_files}")
        
        for filename in csv_files:
            print(f"\nProcessing: {filename}")
            file_path = os.path.join(upcoming_races_dir, filename)
            
            # Extract metadata
            try:
                metadata = _extract_csv_metadata(file_path)
                print(f"  Metadata: {metadata}")
            except Exception as e:
                print(f"  Error extracting metadata: {e}")
                continue
            
            # Try to read CSV header
            try:
                print(f"  Reading CSV header...")
                df_header = pd.read_csv(file_path, nrows=1)
                print(f"  Columns: {list(df_header.columns)}")
                
                header_grade = df_header.get("Grade", pd.Series([None])).iloc[0] if "Grade" in df_header.columns else None
                header_distance = df_header.get("Distance", pd.Series([None])).iloc[0] if "Distance" in df_header.columns else None
                
                print(f"  Grade: {header_grade}")
                print(f"  Distance: {header_distance}")
                
            except Exception as e:
                print(f"  Error reading CSV: {e}")
                import traceback
                traceback.print_exc()
                
            # Check if we can read it differently
            try:
                print(f"  Trying to read with different encoding...")
                df_header = pd.read_csv(file_path, nrows=1, encoding='utf-8')
                print(f"  Success with utf-8")
            except Exception:
                try:
                    df_header = pd.read_csv(file_path, nrows=1, encoding='latin-1')
                    print(f"  Success with latin-1")
                except Exception as e:
                    print(f"  Failed with both encodings: {e}")
    
    except Exception as e:
        print(f"Error listing directory: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_upcoming_races()
