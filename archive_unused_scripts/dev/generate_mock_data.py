#!/usr/bin/env python3
"""
Generate Mock Training Data (archived)

This script generates synthetic greyhound racing data for development/testing.
It is archived to avoid accidental production use. Do not use for live predictions.
"""

# The original script contents were preserved. Any imports should remain valid if
# executed directly from the archive. See repository history for provenance.

import argparse
import os
import random
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ---- Original functionality preserved below ----

def generate_dog_names(count: int) -> List[str]:
    prefixes = [
        "Lightning","Thunder","Storm","Flash","Rocket","Swift","Rapid","Quick","Bold","Brave",
        "Lucky","Happy","Mighty","Super","Golden","Silver","Black","Red","Blue","Green",
        "Star","Moon","Sun","Fire","Ice",
    ]
    suffixes = [
        "Runner","Chaser","Striker","Winner","Hero","Champion","Master","King","Queen","Prince",
        "Princess","Ace","Spirit","Storm","Blaze","Arrow","Bullet","Comet","Dash","Eagle","Falcon",
        "Hawk","Tiger",
    ]
    names, used = [], set()
    for _ in range(count):
        while True:
            name = f"{random.choice(prefixes)} {random.choice(suffixes)}"
            if name not in used:
                names.append(name); used.add(name); break
    return names


def generate_mock_race_data(num_races: int = 100, dogs_per_race: int = 6, db_path: str = "greyhound_racing_data.db") -> None:
    print(f"🏁 Generating {num_races} races with ~{dogs_per_race} dogs each...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    venues = ["Trackside","Greyhound Park","Racing Stadium","Speedway","Track Central","Victory Track","Champion Circuit","Winners Way","Racing Point"]
    grades = ["G1","G2","G3","G4","G5","M","M1","M2","M3"]
    distances = [300,350,400,450,500,550,600,650,700,750]
    track_conditions = ["Fast","Good","Slow","Heavy"]
    weather_conditions = ["Fine","Cloudy","Light Rain","Heavy Rain","Windy"]
    dog_names = generate_dog_names(num_races * dogs_per_race + 50)
    race_id_counter = 1
    start_date = datetime.now() - timedelta(days=365)
    race_metadata_records, dog_race_data_records, enhanced_expert_records = [], [], []
    print("📊 Generating race data...")
    for _ in range(num_races):
        race_id = f"R{race_id_counter:04d}"; race_id_counter += 1
        race_date = start_date + timedelta(days=random.randint(0,364), hours=random.randint(12,22), minutes=random.choice([0,15,30,45]))
        venue = random.choice(venues); grade = random.choice(grades); distance = random.choice(distances)
        track_condition = random.choice(track_conditions); weather = random.choice(weather_conditions)
        actual_field_size = random.randint(max(3, dogs_per_race - 2), dogs_per_race + 2)
        race_dogs = random.sample(dog_names, actual_field_size)
        finishing_positions = list(range(1, actual_field_size + 1)); random.shuffle(finishing_positions)
        winner_idx = finishing_positions.index(1); winner_name = race_dogs[winner_idx]
        winner_odds = round(random.uniform(1.5, 25.0), 2); winner_margin = round(random.uniform(0.1, 5.0), 2)
        base_time = distance * 0.08; condition_modifier = {"Fast":1.0,"Good":1.02,"Slow":1.05,"Heavy":1.08}[track_condition]
        race_time = round(base_time * condition_modifier + random.uniform(-2, 2), 2)
        race_metadata_records.append({
            "race_id": race_id, "venue": venue, "grade": grade, "distance": distance, "track_condition": track_condition,
            "weather": weather, "field_size": actual_field_size, "race_date": race_date.strftime("%Y-%m-%d"),
            "race_time": race_date.strftime("%H:%M:%S"), "winner_name": winner_name, "winner_odds": winner_odds, "winner_margin": winner_margin,
        })
        for dog_idx, dog_name in enumerate(race_dogs):
            box_number = dog_idx + 1; finish_position = finishing_positions[dog_idx]
            weight = round(random.uniform(28.0, 35.0), 1); age_days = random.randint(300, 2000)
            is_winner = finish_position == 1; performance_boost = 0.3 if is_winner else 0.0
            recent_wins = random.randint(0, 8) + (2 if is_winner else 0); recent_races = random.randint(max(recent_wins, 3), 15)
            win_rate = (recent_wins / recent_races) + (performance_boost * random.uniform(0, 0.2)); win_rate = min(1.0, max(0.0, win_rate))
            avg_finish_position = max(1.0, min(8.0, random.uniform(2.5, 6.0) - (performance_boost * 2)))
            best_time = round(distance * 0.075 - (performance_boost * 2) + random.uniform(-1, 1), 2)
            avg_time = best_time + random.uniform(0.5, 2.5)
            starting_price = max(1.2, round(random.uniform(1.8, 50.0) - (performance_boost * 10), 2))
            dog_race_data_records.append({
                "race_id": race_id, "dog_clean_name": dog_name, "box_number": box_number, "finish_position": finish_position,
                "weight": weight, "age_days": age_days, "starting_price": starting_price, "recent_wins": recent_wins,
                "recent_races": recent_races, "win_rate": round(win_rate, 3), "avg_finish_position": round(avg_finish_position, 2),
                "best_time": best_time, "avg_time": round(avg_time, 2),
            })
            pir_rating = round(random.uniform(60, 100) + (performance_boost * 15), 1)
            first_sectional = round(distance * 0.02 - (performance_boost * 0.5) + random.uniform(-0.3, 0.3), 2)
            win_time = race_time + random.uniform(-1.5, 3.0) + (0 if is_winner else random.uniform(0, 2))
            bonus_time = round(random.uniform(-2.0, 2.0), 2)
            enhanced_expert_records.append({
                "race_id": race_id, "dog_clean_name": dog_name, "pir_rating": pir_rating, "first_sectional": first_sectional,
                "win_time": round(win_time, 2), "bonus_time": bonus_time,
            })
    pd.DataFrame(race_metadata_records).to_sql("race_metadata", conn, if_exists="append", index=False)
    pd.DataFrame(dog_race_data_records).to_sql("dog_race_data", conn, if_exists="append", index=False)
    pd.DataFrame(enhanced_expert_records).to_sql("enhanced_expert_data", conn, if_exists="append", index=False)
    conn.commit(); conn.close()
    print("✅ Mock data generation completed (archived script)")


def clean_tables(db_path: str) -> None:
    conn = sqlite3.connect(db_path); cur = conn.cursor()
    for table in ["dog_race_data","race_metadata","enhanced_expert_data"]:
        try:
            cur.execute(f'DELETE FROM "{table}"')  # nosec B608
        except Exception:
            continue
    conn.commit(); conn.close()


def main():
    ap = argparse.ArgumentParser(description="Generate Mock Training Data (archived)")
    ap.add_argument("--races", type=int, default=100)
    ap.add_argument("--dogs-per-race", type=int, default=6)
    ap.add_argument("--clean-first", action="store_true")
    ap.add_argument("--db-path", default="greyhound_racing_data.db")
    args = ap.parse_args()
    db_path = os.getenv("GREYHOUND_DB_PATH") or args.db_path
    if args.clean_first:
        clean_tables(db_path)
    generate_mock_race_data(args.races, args.dogs_per_race, db_path)

if __name__ == "__main__":
    main()

