#!/usr/bin/env python3
"""
Test script to verify that CSV loading captures all historical races for each dog
"""

import pandas as pd
from pathlib import Path

def test_csv_loading():
    """Test CSV loading to ensure all historical races are captured"""
    
    # Load one sample CSV file
    csv_file = Path("/Users/orlandolee/greyhound_racing_collector/unprocessed/Race 1 - AP_K - 01 July 2025.csv")
    
    print(f"Testing CSV file: {csv_file.name}")
    
    # Read the CSV
    df = pd.read_csv(csv_file, on_bad_lines='skip', encoding='utf-8')
    
    print(f"Total rows in CSV: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 10 rows:")
    print(df.head(10))
    
    # Test the parsing logic from the comprehensive system
    form_data = {}
    current_dog_name = None
    
    for idx, row in df.iterrows():
        dog_name_raw = str(row['Dog Name']).strip()
        
        # Check if this is a new dog or continuation of previous
        if dog_name_raw != '""' and dog_name_raw != '' and dog_name_raw != 'nan':
            # New dog - clean the name
            current_dog_name = dog_name_raw
            # Remove box number prefix (e.g., "1. Mel Monelli" -> "Mel Monelli")
            if '. ' in current_dog_name:
                current_dog_name = current_dog_name.split('. ', 1)[1]
            
            # Initialize dog record if not exists
            if current_dog_name not in form_data:
                form_data[current_dog_name] = []
                print(f"\n🐕 Found new dog: {current_dog_name}")
        
        # Skip if we don't have a current dog
        if current_dog_name is None:
            continue
        
        # Parse this row as historical race data for current dog
        historical_race = {
            'sex': str(row.get('Sex', '')).strip(),
            'place': str(row.get('PLC', '')).strip(),
            'box': str(row.get('BOX', '')).strip(),
            'weight': str(row.get('WGT', '')).strip(),
            'distance': str(row.get('DIST', '')).strip(),
            'date': str(row.get('DATE', '')).strip(),
            'track': str(row.get('TRACK', '')).strip(),
            'grade': str(row.get('G', '')).strip(),
            'time': str(row.get('TIME', '')).strip(),
            'win_time': str(row.get('WIN', '')).strip(),
            'bonus': str(row.get('BON', '')).strip(),
            'first_sectional': str(row.get('1 SEC', '')).strip(),
            'margin': str(row.get('MGN', '')).strip(),
            'runner_up': str(row.get('W/2G', '')).strip(),
            'pir': str(row.get('PIR', '')).strip(),
            'starting_price': str(row.get('SP', '')).strip(),
        }
        
        # Only add if we have meaningful data (at least place and date)
        if historical_race['place'] and historical_race['date']:
            form_data[current_dog_name].append(historical_race)
            print(f"  📊 Added race: {historical_race['place']} place on {historical_race['date']}")
    
    print(f"\n📈 Summary:")
    print(f"Dogs found: {len(form_data)}")
    for dog_name, races in form_data.items():
        print(f"  {dog_name}: {len(races)} historical races")
    
    # Show detailed data for first dog
    if form_data:
        first_dog = list(form_data.keys())[0]
        print(f"\n🔍 Detailed history for '{first_dog}':")
        for i, race in enumerate(form_data[first_dog]):
            print(f"  Race {i+1}: Place {race['place']}, Date {race['date']}, Track {race['track']}")

if __name__ == "__main__":
    test_csv_loading()
