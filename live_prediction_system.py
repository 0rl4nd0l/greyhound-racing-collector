
#!/usr/bin/env python3
"""
Live Greyhound Racing Prediction System
=======================================

This system provides an end-to-end solution for fetching upcoming races,
collecting live odds, and generating real-time predictions.

Author: AI Assistant
Date: 2025-01-28
"""

import os
import sys
import time
import json
import sqlite3
import logging
import requests
import random
from datetime import datetime, timedelta
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Import the prediction systems
try:
    from advanced_ensemble_ml_system import AdvancedEnsembleMLSystem
    from prediction_orchestrator import PredictionOrchestrator
    ML_SYSTEMS_AVAILABLE = True
except ImportError:
    ML_SYSTEMS_AVAILABLE = False

class LivePredictionSystem:
    """Orchestrates the live prediction pipeline."""

    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.setup_logging()
        self.session = self.setup_session()
        
        # Initialize prediction systems
        if ML_SYSTEMS_AVAILABLE:
            try:
                self.predictor = PredictionOrchestrator(db_path)
                self.logger.info("✅ ML prediction systems initialized")
            except Exception as e:
                self.logger.error(f"Error initializing ML systems: {e}")
                self.predictor = None
        else:
            self.logger.warning("⚠️ ML prediction systems not available")
            self.predictor = None

    def setup_logging(self):
        """Sets up logging."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("logs/live_prediction_system.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def setup_session(self):
        """Sets up a requests session."""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        return session

    def get_upcoming_races(self, days_ahead=1):
        """Fetches upcoming races from thedogs.com.au."""
        self.logger.info(f"--- Fetching upcoming races for the next {days_ahead} day(s) ---")
        all_races = []
        base_url = "https://www.thedogs.com.au"

        for i in range(days_ahead + 1):
            check_date = datetime.now().date() + timedelta(days=i)
            date_str = check_date.strftime('%Y-%m-%d')
            url = f"{base_url}/racing-fields/{date_str}"
            
            try:
                response = self.session.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')

                race_meetings = soup.select("div.meeting-card")
                for meeting in race_meetings:
                    venue = meeting.select_one("h3.meeting-card__title").get_text(strip=True)
                    races = meeting.select("a.meeting-card__race-link")
                    for race in races:
                        race_info = {
                            'venue': venue,
                            'race_number': race.select_one(".race-number").get_text(strip=True),
                            'race_url': urljoin(base_url, race['href']),
                            'race_date': date_str
                        }
                        all_races.append(race_info)

                self.logger.info(f"Found {len(races)} races for {venue} on {date_str}")
                time.sleep(random.uniform(0.5, 1.5)) # Respectful scraping

            except requests.RequestException as e:
                self.logger.error(f"Error fetching races for {date_str}: {e}")

        self.logger.info(f"--- Fetched a total of {len(all_races)} upcoming races ---")
        return all_races

    def store_upcoming_races(self, races):
        """Stores upcoming race information in the database."""
        self.logger.info(f"--- Storing {len(races)} upcoming races to the database ---")
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Basic table for upcoming races
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS upcoming_races (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                venue TEXT,
                race_number INTEGER,
                race_date TEXT,
                race_url TEXT UNIQUE,
                status TEXT DEFAULT 'pending',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)

            for race in races:
                cursor.execute("""
                INSERT OR IGNORE INTO upcoming_races (venue, race_number, race_date, race_url)
                VALUES (?, ?, ?, ?)
                """, (race['venue'], race['race_number'], race['race_date'], race['race_url']))
            conn.commit()
            self.logger.info(f"Stored/updated {len(races)} races.")

        finally:
            conn.close()
            
    def get_race_details(self, race_url):
        """Gets detailed information about a specific race."""
        try:
            response = self.session.get(race_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract dog information from the race page
            dogs = []
            dog_rows = soup.select("tr.runner-row, .dog-row, .runner")
            
            for row in dog_rows:
                dog_info = {}
                
                # Try to extract dog name
                name_elem = row.select_one(".dog-name, .runner-name, .name")
                if name_elem:
                    dog_info['name'] = name_elem.get_text(strip=True)
                
                # Try to extract box number
                box_elem = row.select_one(".box-number, .box, .barrier")
                if box_elem:
                    try:
                        dog_info['box_number'] = int(box_elem.get_text(strip=True))
                    except ValueError:
                        pass
                
                # Try to extract odds if available
                odds_elem = row.select_one(".odds, .price")
                if odds_elem:
                    try:
                        odds_text = odds_elem.get_text(strip=True)
                        # Convert fractional odds to decimal if needed
                        if '/' in odds_text:
                            parts = odds_text.split('/')
                            dog_info['odds'] = (float(parts[0]) / float(parts[1])) + 1
                        else:
                            dog_info['odds'] = float(odds_text.replace('$', ''))
                    except (ValueError, ZeroDivisionError):
                        pass
                
                if dog_info.get('name'):
                    dogs.append(dog_info)
            
            return dogs
            
        except Exception as e:
            self.logger.error(f"Error getting race details from {race_url}: {e}")
            return []
    
    def generate_predictions(self, race_info, dogs):
        """Generates predictions for a race."""
        if not self.predictor:
            self.logger.warning("No prediction system available")
            return []
        
        predictions = []
        
        for dog in dogs:
            try:
                # Create test data for the dog (using defaults where data not available)
                dog_data = {
                    'name': dog['name'],
                    'box_number': dog.get('box_number', 1),
                    'weight': 30.5,  # Default weight
                    'starting_price': dog.get('odds', 5.0),
                    'individual_time': 29.20,  # Default time
                    'field_size': len(dogs),
                    'temperature': 20.0,
                    'humidity': 60.0,
                    'wind_speed': 8.0
                }
                
                # Generate prediction
                result = self.predictor.predict_race(dog_data, dog.get('odds'))
                
                if result['success']:
                    prediction = {
                        'dog_name': dog['name'],
                        'race_venue': race_info['venue'],
                        'race_number': race_info['race_number'],
                        'race_date': race_info['race_date'],
                        'box_number': dog.get('box_number'),
                        'market_odds': dog.get('odds'),
                        'predicted_probability': result['prediction']['win_probability'],
                        'confidence': result['prediction']['confidence'],
                        'betting_recommendation': result['prediction'].get('betting_recommendation', {}),
                        'timestamp': datetime.now().isoformat()
                    }
                    predictions.append(prediction)
                
            except Exception as e:
                self.logger.error(f"Error predicting for dog {dog['name']}: {e}")
        
        return predictions
    
    def store_predictions(self, predictions):
        """Stores predictions in the database."""
        if not predictions:
            return
        
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            
            # Create live predictions table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS live_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dog_name TEXT,
                race_venue TEXT,
                race_number INTEGER,
                race_date TEXT,
                box_number INTEGER,
                market_odds REAL,
                predicted_probability REAL,
                confidence REAL,
                betting_recommendation TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            for pred in predictions:
                cursor.execute("""
                INSERT INTO live_predictions 
                (dog_name, race_venue, race_number, race_date, box_number, 
                 market_odds, predicted_probability, confidence, betting_recommendation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pred['dog_name'], pred['race_venue'], pred['race_number'],
                    pred['race_date'], pred['box_number'], pred['market_odds'],
                    pred['predicted_probability'], pred['confidence'],
                    json.dumps(pred['betting_recommendation'])
                ))
            
            conn.commit()
            self.logger.info(f"Stored {len(predictions)} predictions")
            
        finally:
            conn.close()
    
    def run(self, max_races=5):
        """Executes the live prediction pipeline."""
        self.logger.info("===== Live Prediction System Started =====")

        # Step 1: Fetch Upcoming Races
        upcoming_races = self.get_upcoming_races()
        if not upcoming_races:
            self.logger.warning("No upcoming races found")
            return
        
        self.store_upcoming_races(upcoming_races)
        
        # Step 2: Generate predictions for a subset of races
        races_processed = 0
        all_predictions = []
        
        for race in upcoming_races[:max_races]:  # Limit to prevent overload
            self.logger.info(f"Processing race: {race['venue']} Race {race['race_number']}")
            
            # Get race details (dogs, odds, etc.)
            dogs = self.get_race_details(race['race_url'])
            if not dogs:
                self.logger.warning(f"No dog information found for race: {race['race_url']}")
                continue
            
            self.logger.info(f"Found {len(dogs)} dogs in the race")
            
            # Generate predictions
            predictions = self.generate_predictions(race, dogs)
            if predictions:
                all_predictions.extend(predictions)
                races_processed += 1
                
                # Display top predictions
                sorted_preds = sorted(predictions, key=lambda x: x['predicted_probability'], reverse=True)
                self.logger.info(f"Top prediction: {sorted_preds[0]['dog_name']} - {sorted_preds[0]['predicted_probability']:.2%} chance")
            
            # Rate limiting
            time.sleep(2)
        
        # Step 3: Store all predictions
        if all_predictions:
            self.store_predictions(all_predictions)
            
            # Summary
            self.logger.info(f"\n=== PREDICTION SUMMARY ===")
            self.logger.info(f"Races processed: {races_processed}")
            self.logger.info(f"Total predictions: {len(all_predictions)}")
            
            # Show value bets
            value_bets = [p for p in all_predictions if p['betting_recommendation'].get('has_value', False)]
            if value_bets:
                self.logger.info(f"Value betting opportunities: {len(value_bets)}")
                for bet in value_bets[:3]:  # Show top 3
                    self.logger.info(f"  - {bet['dog_name']} @ {bet['market_odds']:.2f} (Edge: {bet['betting_recommendation'].get('edge', 0):.2%})")
            else:
                self.logger.info("No value betting opportunities identified")

        self.logger.info("===== Live Prediction System Finished =====")


if __name__ == "__main__":
    live_system = LivePredictionSystem()
    live_system.run()

