#!/usr/bin/env python3
"""
Race File Manager
================

This script manages race files based on their state:
- historical_races/: Past races with results (ready for processing)
- upcoming_races/: Future races without results (for prediction only)
- unprocessed/: Mixed files that need classification
- processed/: Fully processed historical races

Author: AI Assistant
Date: July 11, 2025
"""

import os
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from utils.race_lifecycle import (
    JUMPED_PENDING_RESULTS,
    RESULTED,
    STALE_FORM_GUIDE,
    UPCOMING_NOT_JUMPED,
    classify_race_file,
)


class RaceFileManager:
    def __init__(self):
        self.base_dir = Path(".")
        self.unprocessed_dir = self.base_dir / "unprocessed"
        self.historical_dir = self.base_dir / "historical_races"
        self.upcoming_dir = self.base_dir / "upcoming_races"
        self.processed_dir = self.base_dir / "form_guides" / "processed"

        # Create directories
        for dir_path in [self.unprocessed_dir, self.historical_dir, self.upcoming_dir, self.processed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _current_db_path(self):
        """Return a local DB path for read-only official-result lifecycle checks."""
        for candidate in (
            os.environ.get("GREYHOUND_DB_PATH"),
            os.environ.get("DATABASE_PATH"),
            "greyhound_racing_data_writable.db",
            "greyhound_racing_data.db",
        ):
            if candidate and Path(candidate).exists():
                return str(candidate)
        return None

    def extract_race_date(self, filename):
        """Extract race date from filename"""
        try:
            # Look for patterns like "13 July 2025" or "15 July 2025"
            date_pattern = r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})"
            match = re.search(date_pattern, filename, re.IGNORECASE)

            if match:
                day, month, year = match.groups()
                month_map = {
                    "january": 1,
                    "february": 2,
                    "march": 3,
                    "april": 4,
                    "may": 5,
                    "june": 6,
                    "july": 7,
                    "august": 8,
                    "september": 9,
                    "october": 10,
                    "november": 11,
                    "december": 12,
                }

                month_num = month_map.get(month.lower())
                if month_num:
                    return datetime(int(year), month_num, int(day))

        except Exception as e:
            print(f"Error parsing date from {filename}: {e}")

        return None

    def has_race_results(self, file_path):
        """Check if a CSV file contains race results"""
        try:
            df = pd.read_csv(file_path)

            # Check for result columns
            result_indicators = [
                "finish_position",
                "position",
                "pos",
                "result",
                "win",
                "place",
            ]

            # Check if any result columns exist and have valid data
            for col in df.columns:
                col_lower = col.lower()
                if any(indicator in col_lower for indicator in result_indicators):
                    # Check if the column has meaningful data (not all NaN or empty)
                    if not df[col].isna().all() and not (df[col] == "").all():
                        return True

            # Additional check - look for specific patterns in data
            # If we see actual finish positions (1, 2, 3, etc.) it's likely a finished race
            for col in df.columns:
                if df[col].dtype == "object":
                    # Look for numeric finish positions
                    positions = df[col].dropna().astype(str)
                    if any(
                        pos.strip().isdigit() and 1 <= int(pos.strip()) <= 8
                        for pos in positions
                    ):
                        return True

            return False

        except Exception as e:
            print(f"Error checking results in {file_path}: {e}")
            return False

    def classify_and_move_files(self):
        """Classify files in unprocessed directory and move them appropriately"""
        print("🗂️  CLASSIFYING RACE FILES")
        print("=" * 50)

        if not self.unprocessed_dir.exists():
            print(f"⚠️  Unprocessed directory not found: {self.unprocessed_dir}")
            return

        csv_files = list(self.unprocessed_dir.glob("*.csv"))

        if not csv_files:
            print("ℹ️  No CSV files found in unprocessed directory")
            return

        print(f"📁 Found {len(csv_files)} files to classify")

        current_date = datetime.now()
        historical_count = 0
        upcoming_count = 0
        error_count = 0

        for file_path in csv_files:
            try:
                filename = file_path.name
                print(f"\n🔍 Analyzing: {filename}")

                lifecycle = classify_race_file(
                    file_path, db_path=self._current_db_path()
                )
                print(
                    f"   📅 Lifecycle: {lifecycle.status} ({lifecycle.status_reason})"
                )

                if lifecycle.status == RESULTED:
                    dest_path = self.historical_dir / filename
                    shutil.move(str(file_path), str(dest_path))
                    print("   ✅ Moved to historical_races (official result)")
                    historical_count += 1
                elif lifecycle.status == UPCOMING_NOT_JUMPED:
                    dest_path = self.upcoming_dir / filename
                    shutil.move(str(file_path), str(dest_path))
                    print("   📅 Moved to upcoming_races (pre-jump target)")
                    upcoming_count += 1
                elif lifecycle.status == JUMPED_PENDING_RESULTS:
                    print("   ⏸️  Left in unprocessed (jumped, awaiting official result)")
                elif lifecycle.status == STALE_FORM_GUIDE:
                    print("   🧪 Left in unprocessed (stale form-guide mechanics only)")
                else:
                    print("   ❌ Unknown lifecycle status; left in unprocessed")
                    error_count += 1

            except Exception as e:
                print(f"   ❌ Error processing {filename}: {e}")
                error_count += 1

        print(f"\n📊 CLASSIFICATION SUMMARY")
        print(f"   📚 Historical races: {historical_count}")
        print(f"   📅 Upcoming races: {upcoming_count}")
        print(f"   ❌ Errors: {error_count}")
        print(f"   ✅ Classification complete!")

    def get_directory_stats(self):
        """Get statistics for all directories"""
        stats = {}

        directories = {
            "unprocessed": self.unprocessed_dir,
            "historical_races": self.historical_dir,
            "upcoming_races": self.upcoming_dir,
            "processed": self.processed_dir,
        }

        for name, path in directories.items():
            if path.exists():
                csv_files = list(path.glob("*.csv"))
                stats[name] = len(csv_files)
            else:
                stats[name] = 0

        return stats

    def move_historical_to_unprocessed(self):
        """Move historical races to unprocessed for processing"""
        print("\n🔄 MOVING HISTORICAL RACES FOR PROCESSING")
        print("=" * 50)

        if not self.historical_dir.exists():
            print("⚠️  Historical races directory not found")
            return

        historical_files = list(self.historical_dir.glob("*.csv"))

        if not historical_files:
            print("ℹ️  No historical race files found")
            return

        moved_count = 0
        for file_path in historical_files:
            try:
                dest_path = self.unprocessed_dir / file_path.name
                shutil.move(str(file_path), str(dest_path))
                print(f"   ✅ Moved {file_path.name} to unprocessed")
                moved_count += 1
            except Exception as e:
                print(f"   ❌ Error moving {file_path.name}: {e}")

        print(f"\n📊 Moved {moved_count} historical race files for processing")


def main():
    """Main function"""
    manager = RaceFileManager()

    print("🗂️  RACE FILE MANAGEMENT SYSTEM")
    print("=" * 60)

    # Show current stats
    stats = manager.get_directory_stats()
    print(f"\n📊 CURRENT FILE DISTRIBUTION:")
    for directory, count in stats.items():
        print(f"   {directory}: {count} files")

    # Classify files
    manager.classify_and_move_files()

    # Show final stats
    final_stats = manager.get_directory_stats()
    print(f"\n📊 FINAL FILE DISTRIBUTION:")
    for directory, count in final_stats.items():
        print(f"   {directory}: {count} files")


if __name__ == "__main__":
    main()
