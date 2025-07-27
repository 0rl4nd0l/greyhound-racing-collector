#!/usr/bin/env python3
"""
Race Result Analyzer
Analyzes pre-race form guide data against actual race results to identify predictive patterns.
"""

import csv
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import re
from datetime import datetime

class RaceResultAnalyzer:
    def __init__(self):
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--window-size=1920,1080")
        
    def parse_form_guide(self, csv_file_path):
        """Parse the pre-race form guide CSV file"""
        dogs_data = []
        current_dog = None
        
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter='|')
            for row in reader:
                if len(row) < 2:
                    continue
                    
                # Skip header row
                if row[0] == '1':
                    continue
                
                # Parse dog data
                dog_info = row[1].split(',')
                if len(dog_info) < 17:
                    continue
                
                # Check if this is a new dog (has a name) or continuation of previous dog
                dog_name = dog_info[0].strip()
                if dog_name and not dog_name.startswith('"'):
                    # New dog
                    current_dog = {
                        'name': dog_name,
                        'sex': dog_info[1],
                        'recent_form': []
                    }
                    dogs_data.append(current_dog)
                
                # Add race history (including current row)
                if current_dog:
                    try:
                        race_record = {
                            'place': dog_info[2],
                            'box': dog_info[3],
                            'weight': dog_info[4],
                            'distance': dog_info[5],
                            'date': dog_info[6],
                            'track': dog_info[7],
                            'grade': dog_info[8],
                            'time': dog_info[9],
                            'win_time': dog_info[10],
                            'bonus': dog_info[11],
                            'first_section': dog_info[12],
                            'margin': dog_info[13],
                            'w2g': dog_info[14],
                            'pir': dog_info[15],
                            'sp': dog_info[16] if len(dog_info) > 16 else ''
                        }
                        current_dog['recent_form'].append(race_record)
                    except (IndexError, ValueError):
                        continue
        
        return dogs_data
    
    def analyze_form_patterns(self, dogs_data):
        """Analyze form patterns from the dogs' recent performances"""
        analysis = {}
        
        for dog in dogs_data:
            dog_name = dog['name']
            form = dog['recent_form']
            
            if not form:
                continue
                
            # Get most recent race (first in list)
            recent_race = form[0]
            
            analysis[dog_name] = {
                'recent_place': recent_race['place'],
                'recent_time': recent_race['time'],
                'recent_sp': recent_race['sp'],
                'recent_box': recent_race['box'],
                'recent_grade': recent_race['grade'],
                'wins_in_last_5': sum(1 for race in form[:5] if race['place'] == '1'),
                'avg_place_last_5': sum(int(race['place']) for race in form[:5] if race['place'].isdigit()) / len([r for r in form[:5] if r['place'].isdigit()]) if any(r['place'].isdigit() for r in form[:5]) else 0,
                'form_summary': [race['place'] for race in form[:5]],
                'consistency_score': self.calculate_consistency_score(form[:5])
            }
        
        return analysis
    
    def calculate_consistency_score(self, recent_form):
        """Calculate a consistency score based on recent form"""
        if not recent_form:
            return 0
            
        places = [int(race['place']) for race in recent_form if race['place'].isdigit()]
        if not places:
            return 0
            
        # Lower variance = higher consistency
        variance = sum((p - sum(places)/len(places))**2 for p in places) / len(places)
        return max(0, 100 - variance * 10)  # Convert to 0-100 scale
    
    def scrape_race_result(self, track, date, race_number):
        """Scrape the actual race result from thedogs.com.au"""
        try:
            driver = webdriver.Chrome(options=self.chrome_options)
            
            # Convert date format for URL
            date_obj = datetime.strptime(date, "%d %B %Y")
            url_date = date_obj.strftime("%Y-%m-%d")
            
            # Construct URL - this is an educated guess based on typical racing sites
            url = f"https://www.thedogs.com.au/racing/{track.lower()}/{url_date}/race-{race_number}"
            
            print(f"Attempting to scrape: {url}")
            driver.get(url)
            
            # Wait for page to load
            time.sleep(3)
            
            # Try to find result table
            try:
                # Look for common result table patterns
                result_elements = driver.find_elements(By.CLASS_NAME, "result-table")
                if not result_elements:
                    result_elements = driver.find_elements(By.CLASS_NAME, "race-result")
                if not result_elements:
                    result_elements = driver.find_elements(By.TAG_NAME, "table")
                
                if result_elements:
                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    
                    # Extract race results
                    results = self.extract_race_results(soup)
                    driver.quit()
                    return results
                    
            except Exception as e:
                print(f"Error finding results: {e}")
                
            driver.quit()
            return None
            
        except Exception as e:
            print(f"Error scraping race result: {e}")
            return None
    
    def extract_race_results(self, soup):
        """Extract race results from the scraped HTML"""
        results = {}
        
        # Look for tables containing race results
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 4:  # Minimum columns expected
                    row_text = [cell.get_text().strip() for cell in cells]
                    
                    # Look for position, dog name, time patterns
                    for i, cell in enumerate(row_text):
                        if cell.isdigit() and int(cell) <= 8:  # Position 1-8
                            try:
                                position = int(cell)
                                if i + 1 < len(row_text):
                                    dog_name = row_text[i + 1]
                                    time_cell = ""
                                    
                                    # Look for time in subsequent cells
                                    for j in range(i + 2, min(i + 6, len(row_text))):
                                        if re.match(r'\d+\.\d+', row_text[j]):
                                            time_cell = row_text[j]
                                            break
                                    
                                    if dog_name and not dog_name.lower() in ['pos', 'position', 'place']:
                                        results[dog_name.upper()] = {
                                            'position': position,
                                            'time': time_cell,
                                            'winner': position == 1
                                        }
                            except (ValueError, IndexError):
                                continue
        
        return results
    
    def compare_prediction_vs_result(self, form_analysis, race_results):
        """Compare form guide predictions against actual results"""
        if not race_results:
            print("No race results found to compare against")
            return
        
        print("=== RACE RESULT COMPARISON ===")
        print(f"Found {len(race_results)} dogs in race results")
        
        # Find the winner
        winner = None
        for dog_name, result in race_results.items():
            if result['winner']:
                winner = dog_name
                break
        
        if winner:
            print(f"\n🏆 WINNER: {winner}")
            
            # Analyze if the winner was predictable from form
            winner_form = None
            for dog_name, form in form_analysis.items():
                if dog_name.upper() in winner or winner in dog_name.upper():
                    winner_form = form
                    break
            
            if winner_form:
                print(f"Winner's Form Analysis:")
                print(f"  Recent Place: {winner_form['recent_place']}")
                print(f"  Recent Time: {winner_form['recent_time']}")
                print(f"  Recent SP: {winner_form['recent_sp']}")
                print(f"  Wins in last 5: {winner_form['wins_in_last_5']}")
                print(f"  Avg place last 5: {winner_form['avg_place_last_5']:.1f}")
                print(f"  Form: {winner_form['form_summary']}")
                print(f"  Consistency: {winner_form['consistency_score']:.1f}")
                
                # Prediction analysis
                predictable_factors = []
                if winner_form['recent_place'] == '1':
                    predictable_factors.append("Last start winner")
                if winner_form['wins_in_last_5'] >= 2:
                    predictable_factors.append("Multiple recent wins")
                if winner_form['avg_place_last_5'] <= 2.5:
                    predictable_factors.append("Consistent placings")
                if winner_form['consistency_score'] > 70:
                    predictable_factors.append("High consistency")
                
                print(f"\n📊 Predictable factors: {', '.join(predictable_factors) if predictable_factors else 'None obvious'}")
        
        # Compare all dogs
        print(f"\n=== ALL DOGS COMPARISON ===")
        for dog_name, form in form_analysis.items():
            result = None
            for race_dog, race_result in race_results.items():
                if dog_name.upper() in race_dog or race_dog in dog_name.upper():
                    result = race_result
                    break
            
            if result:
                print(f"{dog_name}: Predicted factors vs Actual {result['position']} place")
                print(f"  Form: {form['form_summary']} | Recent SP: {form['recent_sp']}")
            else:
                print(f"{dog_name}: No matching result found")

def main():
    analyzer = RaceResultAnalyzer()
    
    # Parse the form guide
    csv_file = "./processed/Race 6 - WAR - 10 July 2025.csv"
    print("Parsing form guide...")
    dogs_data = analyzer.parse_form_guide(csv_file)
    
    print(f"Found {len(dogs_data)} dogs in form guide:")
    for dog in dogs_data:
        print(f"  - {dog['name']}")
    
    # Analyze form patterns
    print("\nAnalyzing form patterns...")
    form_analysis = analyzer.analyze_form_patterns(dogs_data)
    
    # Scrape actual race result
    print("\nScraping race result...")
    race_results = analyzer.scrape_race_result("WAR", "10 July 2025", "6")
    
    # Compare prediction vs result
    analyzer.compare_prediction_vs_result(form_analysis, race_results)

if __name__ == "__main__":
    main()
