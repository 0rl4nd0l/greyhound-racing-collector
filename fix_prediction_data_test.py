#!/usr/bin/env python3
"""
Test script to demonstrate the fix for prediction data enrichment issue.

The problem is that CSV parsing works correctly but historical statistics
are not being properly calculated and passed to the prediction engine.
"""

from pathlib import Path

import pandas as pd


def parse_race_csv_correct(csv_path):
    """Parse race CSV and correctly calculate dog statistics"""
    df = pd.read_csv(csv_path)

    participants = {}
    current_dog = None

    print(f"Processing: {Path(csv_path).name}")
    print(f"Total CSV rows: {len(df)}")
    print()

    for idx, row in df.iterrows():
        dog_name_raw = str(row.get("Dog Name", "")).strip()

        # Check if this is a new participant (has number prefix)
        if dog_name_raw and not dog_name_raw.startswith('"') and dog_name_raw != "nan":
            # Extract box number and clean name
            if ". " in dog_name_raw:
                box_str, clean_name = dog_name_raw.split(". ", 1)
                box_num = int(box_str)
            else:
                box_num = None
                clean_name = dog_name_raw

            current_dog = clean_name
            participants[current_dog] = {
                "box": box_num,
                "clean_name": clean_name.upper(),
                "races": [],
                "wins": 0,
                "places": 0,
                "total_races": 0,
                "raw_positions": [],
                "raw_times": [],
            }

        # Add historical race data for current dog
        if current_dog and str(row.get("PLC", "")).strip():
            place = str(row.get("PLC", "")).strip()
            time_str = str(row.get("TIME", "")).strip()

            if place.isdigit():
                place_num = int(place)
                participants[current_dog]["races"].append(
                    {
                        "place": place_num,
                        "date": str(row.get("DATE", "")),
                        "track": str(row.get("TRACK", "")),
                        "time": time_str,
                        "sp": str(row.get("SP", "")),
                    }
                )
                participants[current_dog]["total_races"] += 1
                participants[current_dog]["raw_positions"].append(place_num)

                if place_num == 1:
                    participants[current_dog]["wins"] += 1
                if place_num <= 3:
                    participants[current_dog]["places"] += 1

                # Store valid times
                if time_str and time_str.replace(".", "").replace("-", "").isdigit():
                    try:
                        participants[current_dog]["raw_times"].append(float(time_str))
                    except ValueError:
                        pass

    # Calculate enriched statistics for each dog
    enriched_predictions = []

    for dog_name, data in participants.items():
        total = data["total_races"]
        wins = data["wins"]
        places = data["places"]
        positions = data["raw_positions"]
        times = data["raw_times"]

        # Calculate proper statistics
        win_rate = (wins / total) if total > 0 else 0.0
        place_rate = (places / total) if total > 0 else 0.0
        avg_pos = sum(positions) / len(positions) if positions else 10.0
        best_time = min(times) if times else None
        avg_time = sum(times) / len(times) if times else 0.0
        consistency = (
            1.0 / (1.0 + (max(positions) - min(positions)))
            if len(positions) > 1
            else 0.0
        )

        # Create properly enriched prediction object (like the system should)
        enriched_dog = {
            "box_number": data["box"],
            "clean_name": data["clean_name"],
            "dog_clean_name": data["clean_name"],
            "dog_name": dog_name,
            "dog_history_summary": {
                "average_position": avg_pos,
                "best_time": best_time,
                "career_places": places,
                "career_starts": total,
                "career_wins": wins,
                "consistency": consistency,
                "last_5": positions[:5],
                "last_5_string": "".join(map(str, positions[:5])),
                "place_rate": place_rate,
                "win_rate": win_rate,
            },
            "historical_stats": {
                "avg_time": avg_time,
                "total_races": total,
                "recent_form": positions[:5],
            },
            # Derive confidence and predictions from stats
            "confidence": min(
                0.9, 0.3 + (total * 0.05) + (win_rate * 0.3) + (place_rate * 0.2)
            ),
            "win_prob": max(0.05, min(0.4, win_rate * 1.2 + (1.0 / avg_pos) * 0.2)),
            "place_prob": max(0.15, min(0.7, place_rate * 1.1 + (1.0 / avg_pos) * 0.3)),
        }

        enriched_predictions.append(enriched_dog)

    return enriched_predictions


def main():
    """Test the fix on the problematic CAPALABA race"""
    csv_path = "/Users/test/Desktop/greyhound_racing_collector/processed/excluded/Race 1 - CAPALABA - 2025-08-24.csv"

    print("DEMONSTRATING THE FIX:")
    print("=" * 60)

    # Process with correct enrichment
    enriched_dogs = parse_race_csv_correct(csv_path)

    print("PROPERLY ENRICHED PREDICTIONS:")
    print("=" * 40)

    for dog in sorted(enriched_dogs, key=lambda x: x["box_number"]):
        hist = dog["dog_history_summary"]
        print(f"Box {dog['box_number']}: {dog['dog_name']}")
        print(
            f"  Career: {hist['career_starts']} starts, {hist['career_wins']} wins, {hist['career_places']} places"
        )
        print(
            f"  Win Rate: {hist['win_rate']:.1%}, Place Rate: {hist['place_rate']:.1%}"
        )
        print(f"  Avg Position: {hist['average_position']:.1f}")
        print(f"  Recent Form: {hist['last_5_string']}")
        print(f"  Win Prob: {dog['win_prob']:.1%}, Place Prob: {dog['place_prob']:.1%}")
        print(f"  Confidence: {dog['confidence']:.1%}")
        print()

    print("\nCOMPARE TO CURRENT SYSTEM OUTPUT:")
    print("Current system shows all dogs with 0 starts, 0 wins, 0 places")
    print("This demonstrates the enrichment pipeline is broken!")


if __name__ == "__main__":
    main()
