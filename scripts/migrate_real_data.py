#!/usr/bin/env python3
"""
Data Migration Script for ML Training Pipeline

Migrates real data from staging tables (csv_dog_history_staging, csv_race_metadata_staging)
to the training tables (dog_race_data, race_metadata, enhanced_expert_data) that the ML system expects.

This script transforms and enriches the staging data to create a proper training dataset.

Usage:
    python scripts/migrate_real_data.py
    python scripts/migrate_real_data.py --clean-first
    python scripts/migrate_real_data.py --limit 1000
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Route DB writes via staging by default
try:
    from scripts.db_utils import open_sqlite_writable
except Exception:
    def open_sqlite_writable(db_path: str | None = None):
        import os as _os, sqlite3 as _sqlite3
        path = db_path or _os.getenv("STAGING_DB_PATH") or "greyhound_racing_data_stage.db"
        return _sqlite3.connect(_os.path.abspath(path))

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


def parse_raw_row_json(raw_json_str: str) -> Dict[str, Any]:
    """Parse the raw_row_json field to extract additional race information."""
    try:
        if raw_json_str:
            data = json.loads(raw_json_str)
            return {
                "grade": data.get("G", ""),
                "distance": data.get("DIST", ""),
                "win_time": data.get("WIN", ""),
                "bonus_time": data.get("BON", ""),
                "first_sectional": data.get("1 SEC", ""),
                "pir_rating": data.get("PIR", ""),
                "margin": data.get("MGN", ""),
                "winner_name": data.get("W/2G", ""),
                "track": data.get("TRACK", ""),
            }
    except (json.JSONDecodeError, TypeError):
        pass

    return {}


def clean_numeric_field(value: Any, default: float = 0.0) -> float:
    """Clean and convert numeric fields, handling various string formats."""
    if pd.isna(value) or value == "" or value is None:
        return default

    try:
        # Convert to string first, then clean
        str_val = str(value).strip()
        if str_val == "" or str_val.lower() in ["nan", "null", "none"]:
            return default

        # Remove common non-numeric characters
        cleaned = str_val.replace(",", "").replace("$", "").replace("%", "")
        return float(cleaned)
    except (ValueError, TypeError):
        return default


def migrate_race_data(
    db_path: str, limit: Optional[int] = None, clean_first: bool = False
) -> Dict[str, int]:
    """Migrate real data from staging tables to training tables."""

    print("🔄 Starting real data migration for ML training...")
    print(f"📂 Database: {db_path}")

    conn = open_sqlite_writable(db_path)
    cursor = conn.cursor()

    # Clean existing data if requested
    if clean_first:
        print("🧹 Cleaning existing training data...")
        for table in ["dog_race_data", "race_metadata", "enhanced_expert_data"]:
            try:
                cursor.execute(f"DELETE FROM {table}")
                print(f"   Cleaned {table}")
            except sqlite3.OperationalError as e:
                print(f"   Note: {table} - {e}")

    print("📊 Loading staging data...")

    # Load dog history data with limit if specified
    limit_clause = f"LIMIT {limit}" if limit else ""
    dog_history_query = f"""
    SELECT * FROM csv_dog_history_staging 
    WHERE race_id IS NOT NULL 
    AND dog_clean_name IS NOT NULL 
    AND finish_position IS NOT NULL 
    ORDER BY race_date ASC, race_id, box_number 
    {limit_clause}
    """

    dog_history_df = pd.read_sql_query(dog_history_query, conn)
    print(f"   📋 Loaded {len(dog_history_df)} dog history records")

    # Load race metadata
    race_metadata_df = pd.read_sql_query(
        "SELECT * FROM csv_race_metadata_staging ORDER BY race_date ASC", conn
    )
    print(f"   🏁 Loaded {len(race_metadata_df)} race metadata records")

    if dog_history_df.empty:
        print("❌ No dog history data found in staging tables")
        return {"migrated_races": 0, "migrated_dogs": 0, "migrated_expert": 0}

    # Process and migrate data
    migrated_counts = {"migrated_races": 0, "migrated_dogs": 0, "migrated_expert": 0}

    print("🔄 Processing and migrating race data...")

    # Group by race_id to process races
    race_groups = dog_history_df.groupby("race_id")

    for race_id, race_group in race_groups:
        if len(race_group) < 3:  # Skip races with too few dogs
            continue

        # Create race metadata record
        first_dog = race_group.iloc[0]

        # Parse additional data from raw JSON
        parsed_data = parse_raw_row_json(first_dog.get("raw_row_json", ""))

        # Find corresponding race metadata if exists
        race_meta = race_metadata_df[race_metadata_df["race_id"] == race_id]

        race_metadata_record = {
            "race_id": race_id,
            "venue": first_dog["venue"],
            "race_number": first_dog.get("race_number"),
            "race_date": first_dog["race_date"],
            "race_name": "",  # Not available in staging
            "grade": parsed_data.get("grade", ""),
            "distance": parsed_data.get("distance", ""),
            "field_size": len(race_group),
            "race_time": "",  # Will be derived from individual times
            "winner_name": "",  # Will be determined from finish positions
            "winner_odds": None,
            "winner_margin": None,
            "track_condition": "",  # Not available in staging
            "weather": "",  # Not available in staging
        }

        # Determine winner and race statistics
        winners = race_group[race_group["finish_position"] == 1]
        if len(winners) == 1:
            winner = winners.iloc[0]
            race_metadata_record["winner_name"] = winner["dog_clean_name"]
            race_metadata_record["winner_odds"] = clean_numeric_field(
                winner.get("starting_price")
            )

            # Calculate winner margin if available
            if "margin" in winner and pd.notna(winner["margin"]):
                race_metadata_record["winner_margin"] = clean_numeric_field(
                    winner["margin"]
                )

        # Insert race metadata
        try:
            race_meta_df = pd.DataFrame([race_metadata_record])
            race_meta_df.to_sql("race_metadata", conn, if_exists="append", index=False)
            migrated_counts["migrated_races"] += 1
        except sqlite3.IntegrityError:
            # Race already exists, skip
            pass

        # Process individual dog records for this race
        for _, dog_row in race_group.iterrows():
            # Create dog_race_data record
            dog_race_record = {
                "race_id": race_id,
                "dog_name": dog_row["dog_name"],
                "dog_clean_name": dog_row["dog_clean_name"],
                "box_number": dog_row.get("box_number"),
                "finish_position": dog_row["finish_position"],
            }

            # Insert dog race data
            try:
                dog_df = pd.DataFrame([dog_race_record])
                dog_df.to_sql("dog_race_data", conn, if_exists="append", index=False)
                migrated_counts["migrated_dogs"] += 1
            except sqlite3.IntegrityError:
                # Dog record already exists, skip
                pass

            # Create enhanced expert data record
            parsed_dog_data = parse_raw_row_json(dog_row.get("raw_row_json", ""))

            expert_record = {
                "race_id": race_id,
                "dog_clean_name": dog_row["dog_clean_name"],
                "pir_rating": clean_numeric_field(parsed_dog_data.get("pir_rating")),
                "first_sectional": parsed_dog_data.get("first_sectional", ""),
                "win_time": parsed_dog_data.get("win_time", ""),
                "bonus_time": parsed_dog_data.get("bonus_time", ""),
            }

            # Insert enhanced expert data
            try:
                expert_df = pd.DataFrame([expert_record])
                expert_df.to_sql(
                    "enhanced_expert_data", conn, if_exists="append", index=False
                )
                migrated_counts["migrated_expert"] += 1
            except sqlite3.IntegrityError:
                # Expert record already exists, skip
                pass

        # Progress reporting
        if (
            migrated_counts["migrated_races"] % 100 == 0
            and migrated_counts["migrated_races"] > 0
        ):
            print(f"   Processed {migrated_counts['migrated_races']} races...")

    conn.commit()
    conn.close()

    print("✅ Data migration completed!")
    print(f"   📊 Migrated {migrated_counts['migrated_races']} races")
    print(f"   🐕 Migrated {migrated_counts['migrated_dogs']} dog entries")
    print(f"   📈 Migrated {migrated_counts['migrated_expert']} expert records")

    return migrated_counts


def validate_migration(db_path: str) -> None:
    """Validate the migrated data quality."""
    print("🔍 Validating migrated data quality...")

    conn = open_sqlite_writable(db_path)

    # Check record counts
    tables_to_check = ["dog_race_data", "race_metadata", "enhanced_expert_data"]
    for table in tables_to_check:
        try:
            count = pd.read_sql_query(
                f"SELECT COUNT(*) as count FROM {table}", conn
            ).iloc[0]["count"]
            print(f"   {table}: {count} records")
        except Exception as e:
            print(f"   {table}: Error - {e}")

    # Check data quality
    try:
        # Check for races with winners
        winners_query = """
        SELECT COUNT(DISTINCT r.race_id) as races_with_winners 
        FROM race_metadata r 
        WHERE r.winner_name IS NOT NULL AND r.winner_name != ''
        """
        winners_count = pd.read_sql_query(winners_query, conn).iloc[0][
            "races_with_winners"
        ]
        print(f"   Races with identified winners: {winners_count}")

        # Check finish position distribution
        positions_query = """
        SELECT finish_position, COUNT(*) as count 
        FROM dog_race_data 
        WHERE finish_position IS NOT NULL 
        GROUP BY finish_position 
        ORDER BY finish_position 
        LIMIT 10
        """
        position_dist = pd.read_sql_query(positions_query, conn)
        print(f"   Finish position distribution (top 10):")
        for _, row in position_dist.iterrows():
            print(f"     Position {row['finish_position']}: {row['count']} dogs")

    except Exception as e:
        print(f"   Validation error: {e}")

    conn.close()
    print("✅ Validation completed!")


def main():
    parser = argparse.ArgumentParser(description="Migrate Real Data for ML Training")
    parser.add_argument(
        "--clean-first",
        action="store_true",
        help="Clean existing training data before migration",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of dog history records to process (for testing)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing migrated data, don't migrate",
    )
    parser.add_argument(
        "--db-path", default="greyhound_racing_data.db", help="Database path"
    )

    args = parser.parse_args()

    # Get database path from environment or argument
    db_path = os.getenv("GREYHOUND_DB_PATH") or args.db_path

    if args.validate_only:
        validate_migration(db_path)
        return

    # Run migration
    migrated_counts = migrate_race_data(
        db_path=db_path, limit=args.limit, clean_first=args.clean_first
    )

    # Validate results
    if migrated_counts["migrated_races"] > 0:
        validate_migration(db_path)

        print("\n🚀 Real data is now ready for ML training!")
        print("   You can now run: python scripts/train_optimized_v4.py")
    else:
        print("\n❌ No data was migrated. Check staging tables for data availability.")


if __name__ == "__main__":
    main()
