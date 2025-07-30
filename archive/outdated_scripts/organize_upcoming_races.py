#!/usr/bin/env python3
# Script to organize upcoming races by the soonest time to run.

import os
import re
from datetime import datetime
from pathlib import Path

# Directory containing the upcoming race CSV files
directory = Path('./upcoming_races')

# Pattern to extract start time and race number from filenames
filename_pattern = re.compile(r'Race (\d+) - .+ - \d{4}-\d{2}-\d{2}\.csv')

# Collect race details
races = []

for file in directory.glob('*.csv'):
    match = filename_pattern.match(file.name)
    if match:
        race_number = int(match.group(1))
        # Calculate race start time based on race number
        # Assuming races start every 25 minutes from 1:00 PM
        time_offset = (race_number - 1) * 25
        start_time = (datetime(2025, 7, 27, 13, 0) + timedelta(minutes=time_offset)).time()
        
        races.append((start_time, file))

# Sort races by start time
races.sort(key=lambda x: x[0])

# Print races in order
for start_time, file in races:
    print(f'{start_time} - {file.name}')
