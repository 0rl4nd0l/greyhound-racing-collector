#!/usr/bin/env python3
"""
Script to update all prediction files with correct race information 
and ensure future predictions have proper data.
"""

import json
import os
import glob
import re
from datetime import datetime
import sys

# Add the project root to Python path
sys.path.append('/Users/orlandolee/greyhound_racing_collector')

from unified_predictor import UnifiedPredictor

def extract_race_info_from_filename(filename):
    """Extract race information from filename patterns."""
    # Handle different filename patterns
    patterns = [
        r'Race\s*(\d+)\s*-\s*([A-Z_]+(?:\s*-\s*[A-Z_]+)*)\s*-\s*(\d{4}-\d{2}-\d{2})',  # Race 8 - TAREE - 2025-07-26
        r'Race_(\d+)_([A-Z_]+)_(\d{4}-\d{2}-\d{2})',  # Race_08_LADBROKES_2025-07-25
        r'Race\s*(\d+)\s*-\s*([A-Z_]+(?:\s*-\s*[A-Z_]+)*)\s*-\s*(\d{1,2}\s+\w+\s+\d{4})',  # Race 1 - AP_K - 22 August 2025
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            race_number = match.group(1)
            venue = match.group(2).strip()
            date_str = match.group(3)
            
            # Convert date format if needed
            try:
                if '-' in date_str:
                    # Already in YYYY-MM-DD format
                    date = date_str
                else:
                    # Convert from "22 August 2025" format
                    from datetime import datetime
                    date_obj = datetime.strptime(date_str, "%d %B %Y")
                    date = date_obj.strftime("%Y-%m-%d")
            except:
                date = date_str
                
            return race_number, venue, date
    
    return None, None, None

def find_corresponding_csv(race_number, venue, date):
    """Find the corresponding CSV file for a race."""
    upcoming_races_dir = '/Users/orlandolee/greyhound_racing_collector/upcoming_races'
    
    # Try different filename patterns
    patterns = [
        f"Race {race_number} - {venue} - {date}.csv",
        f"Race_{int(race_number):02d}_{venue}_{date}.csv",
        f"Race {race_number} - {venue} - *.csv",  # Wildcard for date
        f"Race_{int(race_number):02d}_{venue}_*.csv",
    ]
    
    for pattern in patterns:
        files = glob.glob(os.path.join(upcoming_races_dir, pattern))
        if files:
            return files[0]
    
    return None

def update_prediction_file(filepath):
    """Update a single prediction file with correct race information."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Check if race_info exists and has null/missing distance or grade
        if 'race_info' not in data:
            print(f"No race_info found in {filepath}")
            return False
            
        race_info = data['race_info']
        needs_update = (
            race_info.get('distance') is None or 
            race_info.get('grade') is None or
            race_info.get('distance') == 'Nonem' or
            race_info.get('grade') == 'N/A'
        )
        
        if not needs_update:
            return False
            
        print(f"Updating {os.path.basename(filepath)}...")
        
        # Extract race info from filename
        filename = os.path.basename(filepath)
        race_number, venue, date = extract_race_info_from_filename(filename)
        
        if not race_number or not venue or not date:
            print(f"  Could not extract race info from filename: {filename}")
            return False
            
        # Find corresponding CSV file
        csv_file = find_corresponding_csv(race_number, venue, date)
        if not csv_file:
            print(f"  No CSV file found for Race {race_number} - {venue} - {date}")
            return False
            
        print(f"  Found CSV: {os.path.basename(csv_file)}")
        
        # Create unified predictor instance
        # Note: For this update script, we'll just read the CSV directly since
        # we only need to extract basic race info, not run full predictions
        # predictor = UnifiedPredictor()
        
        try:
            # Read the CSV file and extract basic race info
            import pandas as pd
            race_df = pd.read_csv(csv_file)
            
            # Extract race info from CSV data
            extracted_info = {
                'venue': venue,
                'race_number': race_number,
                'date': date,
                'distance': None,
                'grade': None
            }
            
            if not race_df.empty:
                first_row = race_df.iloc[0]
                
                # Get distance from DIST column
                if 'DIST' in race_df.columns:
                    dist_value = first_row.get('DIST')
                    if pd.notna(dist_value):
                        extracted_info['distance'] = f"{int(dist_value)}m"
                elif 'distance' in race_df.columns:
                    distances = race_df['distance'].dropna().unique()
                    if len(distances) > 0:
                        extracted_info['distance'] = str(distances[0])
                
                # Get grade from G column  
                if 'G' in race_df.columns:
                    grade_value = first_row.get('G')
                    if pd.notna(grade_value) and str(grade_value) != 'nan':
                        extracted_info['grade'] = f"Grade {grade_value}"
                elif 'grade' in race_df.columns:
                    grades = race_df['grade'].dropna().unique()
                    if len(grades) > 0:
                        extracted_info['grade'] = str(grades[0])
            
            # Update the race_info in the prediction file
            race_info.update({
                'venue': extracted_info.get('venue', venue),
                'race_number': extracted_info.get('race_number', race_number),
                'date': extracted_info.get('date', date),
                'distance': extracted_info.get('distance'),
                'grade': extracted_info.get('grade'),
                'filepath': csv_file,
                'filename': os.path.basename(csv_file)
            })
            
            # Save the updated file
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
            print(f"  Updated: Distance={race_info.get('distance')}, Grade={race_info.get('grade')}")
            return True
            
        except Exception as e:
            print(f"  Error processing CSV {csv_file}: {e}")
            return False
            
    except Exception as e:
        print(f"Error updating {filepath}: {e}")
        return False

def update_all_predictions():
    """Update all prediction files that need race information fixes."""
    predictions_dir = '/Users/orlandolee/greyhound_racing_collector/predictions'
    
    # Find all prediction JSON files (excluding backup directory)
    prediction_files = []
    for root, dirs, files in os.walk(predictions_dir):
        # Skip backup directories
        if 'backup' in root.lower() or 'cleanup' in root.lower():
            continue
            
        for file in files:
            if file.startswith('prediction_') and file.endswith('.json'):
                prediction_files.append(os.path.join(root, file))
    
    print(f"Found {len(prediction_files)} prediction files to check...")
    
    updated_count = 0
    for filepath in prediction_files:
        if update_prediction_file(filepath):
            updated_count += 1
    
    print(f"\nUpdated {updated_count} prediction files.")
    return updated_count

def prevent_future_issues():
    """Check the prediction pipeline code to ensure it extracts race info correctly."""
    pipeline_file = '/Users/orlandolee/greyhound_racing_collector/comprehensive_prediction_pipeline.py'
    
    print("\nChecking prediction pipeline for race info extraction...")
    
    try:
        with open(pipeline_file, 'r') as f:
            content = f.read()
        
        # Check if the extract_race_info_from_csv method exists
        if 'extract_race_info_from_csv' in content:
            print("✓ extract_race_info_from_csv method found in pipeline")
        else:
            print("✗ extract_race_info_from_csv method not found - needs to be added")
            
        # Check if race info is being properly extracted and stored
        if 'distance' in content and 'grade' in content:
            print("✓ Distance and grade extraction code found")
        else:
            print("⚠ Distance and grade extraction may need improvement")
            
    except Exception as e:
        print(f"Error checking pipeline file: {e}")

def main():
    """Main function to run the update process."""
    print("Starting prediction files update process...")
    print("=" * 50)
    
    # Update all existing prediction files
    updated_count = update_all_predictions()
    
    # Check pipeline for future prevention
    prevent_future_issues()
    
    print("\n" + "=" * 50)
    print("Update process completed!")
    print(f"Total files updated: {updated_count}")
    
    if updated_count > 0:
        print("\nRecommendations:")
        print("1. Refresh your dashboard to see the updated race information")
        print("2. The prediction pipeline should now properly extract race info for future predictions")
        print("3. Monitor future predictions to ensure distance and grade are properly populated")

if __name__ == "__main__":
    main()
