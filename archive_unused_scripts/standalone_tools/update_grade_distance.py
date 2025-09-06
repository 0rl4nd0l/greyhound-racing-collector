#!/usr/bin/env python3
"""
Update Grade and Distance from Processed CSV Files
=================================================

This script reads the processed CSV files and updates the race_metadata table
with the grade and distance information that was missing from the original processing.
"""

import os
import re
import sqlite3
from datetime import datetime

import pandas as pd


def extract_race_info_from_csv(filepath, df):
    """Extract race information from CSV filename and data"""
    filename = os.path.basename(filepath)

    # Pattern: "Race X - VENUE - DATE.csv"
    pattern = r"Race (\d+) - ([A-Z_]+) - (\d{1,2} \w+ \d{4})\.csv"
    match = re.match(pattern, filename)

    if match:
        race_number = int(match.group(1))
        venue = match.group(2)
        date_str = match.group(3)

        # Parse date
        try:
            race_date = datetime.strptime(date_str, "%d %B %Y").date()
        except ValueError:
            race_date = datetime.now().date()

        # Generate race ID
        race_id = f"{venue.lower()}_{race_date}_{race_number}"

        # Extract grade and distance from CSV data
        grade = ""
        distance = ""

        if not df.empty and len(df) > 0:
            # Get the first row to extract race details
            first_row = df.iloc[0]

            # Extract grade from 'G' column
            if "G" in df.columns:
                grade_value = first_row["G"]
                if pd.notna(grade_value):
                    grade = str(grade_value)

            # Extract distance from 'DIST' column
            if "DIST" in df.columns:
                dist_value = first_row["DIST"]
                if pd.notna(dist_value):
                    distance = (
                        str(int(dist_value)) + "m"
                    )  # Convert to string with 'm' suffix

        return {
            "race_id": race_id,
            "race_number": race_number,
            "venue": venue,
            "race_date": race_date,
            "filename": filename,
            "grade": grade,
            "distance": distance,
        }

    return None


def update_database():
    """Update the database with grade and distance from processed CSV files"""
    processed_dir = "./processed"
    database_path = "greyhound_racing_data.db"

    if not os.path.exists(processed_dir):
        print("❌ Processed directory not found")
        return

    csv_files = [f for f in os.listdir(processed_dir) if f.endswith(".csv")]

    if not csv_files:
        print("❌ No CSV files found in processed directory")
        return

    print(f"📊 Found {len(csv_files)} CSV files to process...")

    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    updated_count = 0
    skipped_count = 0

    for filename in csv_files:
        file_path = os.path.join(processed_dir, filename)

        try:
            # Read CSV file
            df = pd.read_csv(file_path)

            # Extract race info with grade and distance
            race_info = extract_race_info_from_csv(file_path, df)

            if not race_info:
                print(f"⚠️ Could not extract race info from {filename}")
                skipped_count += 1
                continue

            # Check if race exists in database
            cursor.execute(
                "SELECT COUNT(*) FROM race_metadata WHERE race_id = ?",
                (race_info["race_id"],),
            )

            if cursor.fetchone()[0] == 0:
                print(f"⚠️ Race {race_info['race_id']} not found in database - skipping")
                skipped_count += 1
                continue

            # Get existing track_condition to preserve it
            cursor.execute(
                "SELECT track_condition FROM race_metadata WHERE race_id = ?",
                (race_info["race_id"],),
            )
            existing_data = cursor.fetchone()
            existing_track_condition = existing_data[0] if existing_data else ""

            # Update the race_metadata with grade and distance (preserving track_condition)
            cursor.execute(
                """
                UPDATE race_metadata 
                SET grade = ?, distance = ?, track_condition = ?
                WHERE race_id = ?
            """,
                (
                    race_info["grade"],
                    race_info["distance"],
                    existing_track_condition or "",  # Preserve existing track condition
                    race_info["race_id"],
                ),
            )

            if cursor.rowcount > 0:
                print(
                    f"✅ Updated {race_info['race_id']}: Grade={race_info['grade']}, Distance={race_info['distance']}"
                )
                updated_count += 1
            else:
                print(f"⚠️ No update needed for {race_info['race_id']}")
                skipped_count += 1

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            skipped_count += 1

    # Commit all changes
    conn.commit()
    conn.close()

    print(f"\n📈 Update Summary:")
    print(f"   ✅ Updated: {updated_count}")
    print(f"   ⏭️ Skipped: {skipped_count}")
    print(f"   📊 Total: {len(csv_files)}")


if __name__ == "__main__":
    update_database()
