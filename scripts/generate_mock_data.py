#!/usr/bin/env python3
"""
Generate Mock Training Data

Creates synthetic greyhound racing data for testing the ML training pipeline.
Populates the required tables: dog_race_data, race_metadata, enhanced_expert_data.

Usage:
    python scripts/generate_mock_data.py --races 100 --dogs-per-race 6
    python scripts/generate_mock_data.py --clean-first
"""

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import random
import string

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np


def generate_dog_names(count: int) -> List[str]:
    """Generate realistic greyhound names."""
    prefixes = ['Lightning', 'Thunder', 'Storm', 'Flash', 'Rocket', 'Swift', 'Rapid', 'Quick',
               'Bold', 'Brave', 'Lucky', 'Happy', 'Mighty', 'Super', 'Golden', 'Silver',
               'Black', 'Red', 'Blue', 'Green', 'Star', 'Moon', 'Sun', 'Fire', 'Ice']
    
    suffixes = ['Runner', 'Chaser', 'Striker', 'Winner', 'Hero', 'Champion', 'Master',
               'King', 'Queen', 'Prince', 'Princess', 'Ace', 'Spirit', 'Storm', 'Blaze',
               'Arrow', 'Bullet', 'Comet', 'Dash', 'Eagle', 'Falcon', 'Hawk', 'Tiger']
    
    names = []
    used_names = set()
    
    for _ in range(count):
        while True:
            name = f"{random.choice(prefixes)} {random.choice(suffixes)}"
            if name not in used_names:
                names.append(name)
                used_names.add(name)
                break
                
    return names


def generate_mock_race_data(num_races: int = 100, dogs_per_race: int = 6, 
                           db_path: str = "greyhound_racing_data.db") -> None:
    """Generate mock racing data and insert into database tables."""
    
    print(f"🏁 Generating {num_races} races with ~{dogs_per_race} dogs each...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Venues and grades
    venues = ['Trackside', 'Greyhound Park', 'Racing Stadium', 'Speedway', 'Track Central',
              'Victory Track', 'Champion Circuit', 'Winners Way', 'Racing Point']
    
    grades = ['G1', 'G2', 'G3', 'G4', 'G5', 'M', 'M1', 'M2', 'M3']
    distances = [300, 350, 400, 450, 500, 550, 600, 650, 700, 750]
    track_conditions = ['Fast', 'Good', 'Slow', 'Heavy']
    weather_conditions = ['Fine', 'Cloudy', 'Light Rain', 'Heavy Rain', 'Windy']
    
    # Generate dog pool
    total_dogs_needed = num_races * dogs_per_race
    dog_names = generate_dog_names(total_dogs_needed + 50)  # Extra buffer
    
    # Track race IDs
    race_id_counter = 1
    start_date = datetime.now() - timedelta(days=365)  # Start a year ago
    
    race_metadata_records = []
    dog_race_data_records = []
    enhanced_expert_records = []
    
    print("📊 Generating race data...")
    
    for race_idx in range(num_races):
        race_id = f"R{race_id_counter:04d}"
        race_id_counter += 1
        
        # Race timing (spread over past year)
        race_date = start_date + timedelta(
            days=random.randint(0, 364),
            hours=random.randint(12, 22),
            minutes=random.choice([0, 15, 30, 45])
        )
        
        venue = random.choice(venues)
        grade = random.choice(grades)
        distance = random.choice(distances)
        track_condition = random.choice(track_conditions)
        weather = random.choice(weather_conditions)
        actual_field_size = random.randint(max(3, dogs_per_race - 2), dogs_per_race + 2)
        
        # Generate race results
        race_dogs = random.sample(dog_names, actual_field_size)
        
        # Determine winner and finishing order
        finishing_positions = list(range(1, actual_field_size + 1))
        random.shuffle(finishing_positions)
        
        winner_idx = finishing_positions.index(1)
        winner_name = race_dogs[winner_idx]
        winner_odds = round(random.uniform(1.5, 25.0), 2)
        winner_margin = round(random.uniform(0.1, 5.0), 2)
        
        # Calculate race time based on distance and conditions
        base_time = distance * 0.08  # Base time per meter
        condition_modifier = {
            'Fast': 1.0, 'Good': 1.02, 'Slow': 1.05, 'Heavy': 1.08
        }[track_condition]
        
        race_time = round(base_time * condition_modifier + random.uniform(-2, 2), 2)
        
        # Race metadata record
        race_metadata_records.append({
            'race_id': race_id,
            'venue': venue,
            'grade': grade,
            'distance': distance,
            'track_condition': track_condition,
            'weather': weather,
            'field_size': actual_field_size,
            'race_date': race_date.strftime('%Y-%m-%d'),
            'race_time': race_date.strftime('%H:%M:%S'),
            'winner_name': winner_name,
            'winner_odds': winner_odds,
            'winner_margin': winner_margin
        })
        
        # Generate dog-level records
        for dog_idx, dog_name in enumerate(race_dogs):
            box_number = dog_idx + 1
            finish_position = finishing_positions[dog_idx]
            
            # Generate realistic dog attributes
            weight = round(random.uniform(28.0, 35.0), 1)
            age_days = random.randint(300, 2000)  # 10 months to 5.5 years
            
            # Performance attributes (winner tends to have better stats)
            is_winner = finish_position == 1
            performance_boost = 0.3 if is_winner else 0.0
            
            recent_wins = random.randint(0, 8) + (2 if is_winner else 0)
            recent_races = random.randint(max(recent_wins, 3), 15)
            win_rate = (recent_wins / recent_races) + (performance_boost * random.uniform(0, 0.2))
            win_rate = min(1.0, max(0.0, win_rate))
            
            avg_finish_position = random.uniform(2.5, 6.0) - (performance_boost * 2)
            avg_finish_position = max(1.0, min(8.0, avg_finish_position))
            
            best_time = round(distance * 0.075 - (performance_boost * 2) + random.uniform(-1, 1), 2)
            avg_time = best_time + random.uniform(0.5, 2.5)
            
            starting_price = round(random.uniform(1.8, 50.0) - (performance_boost * 10), 2)
            starting_price = max(1.2, starting_price)
            
            # Dog race data record
            dog_race_data_records.append({
                'race_id': race_id,
                'dog_clean_name': dog_name,
                'box_number': box_number,
                'finish_position': finish_position,
                'weight': weight,
                'age_days': age_days,
                'starting_price': starting_price,
                'recent_wins': recent_wins,
                'recent_races': recent_races,
                'win_rate': round(win_rate, 3),
                'avg_finish_position': round(avg_finish_position, 2),
                'best_time': best_time,
                'avg_time': round(avg_time, 2)
            })
            
            # Enhanced expert data record
            pir_rating = round(random.uniform(60, 100) + (performance_boost * 15), 1)
            first_sectional = round(distance * 0.02 - (performance_boost * 0.5) + random.uniform(-0.3, 0.3), 2)
            win_time = race_time + random.uniform(-1.5, 3.0) + (0 if is_winner else random.uniform(0, 2))
            bonus_time = round(random.uniform(-2.0, 2.0), 2)
            
            enhanced_expert_records.append({
                'race_id': race_id,
                'dog_clean_name': dog_name,
                'pir_rating': pir_rating,
                'first_sectional': first_sectional,
                'win_time': round(win_time, 2),
                'bonus_time': bonus_time
            })
    
    print("💾 Inserting race metadata...")
    race_metadata_df = pd.DataFrame(race_metadata_records)
    race_metadata_df.to_sql('race_metadata', conn, if_exists='append', index=False)
    
    print("🐕 Inserting dog race data...")
    dog_race_df = pd.DataFrame(dog_race_data_records)
    dog_race_df.to_sql('dog_race_data', conn, if_exists='append', index=False)
    
    print("📈 Inserting enhanced expert data...")
    enhanced_df = pd.DataFrame(enhanced_expert_records)
    enhanced_df.to_sql('enhanced_expert_data', conn, if_exists='append', index=False)
    
    conn.commit()
    conn.close()
    
    print("✅ Mock data generation completed!")
    print(f"   📊 Generated {len(race_metadata_records)} races")
    print(f"   🐕 Generated {len(dog_race_data_records)} dog entries")
    print(f"   📈 Generated {len(enhanced_expert_records)} expert records")


def clean_tables(db_path: str) -> None:
    """Clean existing data from training tables."""
    print("🧹 Cleaning existing mock data...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    tables_to_clean = ['dog_race_data', 'race_metadata', 'enhanced_expert_data']
    
    for table in tables_to_clean:
        try:
            cursor.execute(f"DELETE FROM {table}")
            print(f"   Cleaned {table}")
        except sqlite3.OperationalError as e:
            print(f"   Note: {table} doesn't exist or is empty ({e})")
    
    conn.commit()
    conn.close()
    
    print("✅ Cleanup completed!")


def main():
    parser = argparse.ArgumentParser(description="Generate Mock Training Data")
    parser.add_argument("--races", type=int, default=100,
                       help="Number of races to generate")
    parser.add_argument("--dogs-per-race", type=int, default=6,
                       help="Average number of dogs per race")
    parser.add_argument("--clean-first", action="store_true",
                       help="Clean existing data before generating new data")
    parser.add_argument("--db-path", default="greyhound_racing_data.db",
                       help="Database path")
    
    args = parser.parse_args()
    
    # Get database path from environment or argument
    db_path = os.getenv("GREYHOUND_DB_PATH") or args.db_path
    
    if args.clean_first:
        clean_tables(db_path)
    
    # Generate mock data
    generate_mock_race_data(
        num_races=args.races,
        dogs_per_race=args.dogs_per_race,
        db_path=db_path
    )
    
    print("\n🧪 Mock data ready for ML training pipeline testing!")
    print(f"   Database: {db_path}")
    print("   You can now run: python scripts/train_optimized_v4.py")


if __name__ == "__main__":
    main()
