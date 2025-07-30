#!/usr/bin/env python3
# Script to scrape and organize upcoming races by start time from thedogs.com

import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path

# Define the base URL and directories
BASE_URL = "https://www.thedogs.com.au/races"
UPCOMING_RACES_DIR = Path('./upcoming_races')

# Ensure the directory exists
UPCOMING_RACES_DIR.mkdir(parents=True, exist_ok=True)

# Fetch upcoming races page
response = requests.get(BASE_URL)
soup = BeautifulSoup(response.text, 'html.parser')

# Placeholder for race information
races = []

# Parse the race information
for race in soup.find_all('div', class_='race-info'):
    race_name = race.find('h2').get_text(strip=True) if race.find('h2') else 'Unknown Race'
    race_time = race.find('span', class_='race-time').get_text(strip=True) if race.find('span', class_='race-time') else '00:00'
    race_time_obj = datetime.strptime(race_time, "%I:%M %p").time()
    races.append((race_time_obj, race_name))

# Sort races by race time
races.sort(key=lambda x: x[0])

# Output sorted races
for race_time, race_name in races:
    print(f"{race_time} - {race_name}")
