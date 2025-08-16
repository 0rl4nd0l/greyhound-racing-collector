#!/usr/bin/env python3
"""
Standardize filenames in upcoming_races directory to Race_X_-_VENUE_-_DD_Month_YYYY.csv format
"""

import os
import re
from datetime import datetime

def standardize_filenames():
    directory = "/Users/orlandolee/greyhound_racing_collector/upcoming_races/"
    print(f"Standardizing filenames in: {directory}")

    renamed_count = 0
    skipped_count = 0
    deleted_count = 0

    for filename in os.listdir(directory):
        if not filename.endswith('.csv'):
            continue

        original_path = os.path.join(directory, filename)
        race_number, venue, date_obj = None, None, None

        # Pattern 1: timestamp_Race_01_VENUE_2025-07-26.csv
        match = re.match(r'\w+_Race_(\d+)_([A-Z_]+)_(\d{4}-\d{2}-\d{2})\.csv', filename)
        if match:
            race_number, venue, date_str = match.groups()
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')

        # Pattern 2: timestamp_Race 1 - VENUE - 2025-07-24.csv
        if not date_obj:
            match = re.match(r'\w+_Race (\d+) - ([A-Z_ ]+) - (\d{4}-\d{2}-\d{2})\.csv', filename)
            if match:
                race_number, venue, date_str = match.groups()
                venue = venue.strip().replace(' ', '_')
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                
        # Pattern 3: timestamp_Race_1_-_TAREE_-_2025-07-26.csv
        if not date_obj:
            match = re.match(r'\w*_?Race_(\d+)_-_([A-Z_]+)_-_(\d{4}-\d{2}-\d{2})\.csv', filename)
            if match:
                 race_number, venue, date_str = match.groups()
                 date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Pattern 4: Race_11_-_CASO_-_22_July_2025.csv (already in correct format)
        if not date_obj:
            match = re.match(r'Race_(\d+)_-_([A-Z_]+)_-_(\d{1,2}_[A-Za-z]+_\d{4})\.csv', filename)
            if match:
                print(f"INFO: '{filename}' is already in the correct format.")
                skipped_count += 1
                continue

        if race_number and venue and date_obj:
            new_date_str = date_obj.strftime('%d_%B_%Y')
            new_filename = f"Race_{int(race_number)}_-_{venue.upper()}_-_{new_date_str}.csv"
            new_path = os.path.join(directory, new_filename)

            if original_path == new_path:
                skipped_count += 1
                continue

            if not os.path.exists(new_path):
                print(f"RENAME: '{filename}' -> '{new_filename}'")
                os.rename(original_path, new_path)
                renamed_count += 1
            else:
                print(f"SKIP: Target '{new_filename}' already exists. Deleting original '{filename}'.")
                os.remove(original_path)
                deleted_count += 1
        else:
            print(f"DELETE: '{filename}' - Could not parse filename pattern")
            os.remove(original_path)
            deleted_count += 1

    print(f"\nResults:")
    print(f"Renamed: {renamed_count} files")
    print(f"Skipped (already correct): {skipped_count} files")
    print(f"Deleted (unparseable or duplicates): {deleted_count} files")

if __name__ == "__main__":
    standardize_filenames()
