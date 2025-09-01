#!/usr/bin/env python3
"""
Simple Bulk Processor for Unprocessed Files
Processes remaining CSV files without heavy dependencies like numpy/pandas
"""

import csv
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path


class SimpleBulkProcessor:
    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.unprocessed_dir = "unprocessed"

    def connect_db(self):
        """Connect to SQLite database"""
        return sqlite3.connect(self.db_path)

    def parse_filename(self, filename):
        """Parse filename to extract race info - handles multiple formats"""
        # Remove .csv extension
        name = filename.replace(".csv", "")

        # Pattern 1: Race N - VENUE - YYYY-MM-DD
        pattern1 = r"^Race\s+(\d{1,2})\s+-\s+([A-Za-z0-9\s\'\&\.\-]+?)\s+-\s+(\d{4}-\d{2}-\d{2})$"
        match = re.match(pattern1, name)
        if match:
            race_number, venue, date = match.groups()
            return venue.strip(), int(race_number), date

        # Pattern 2: Race N - VENUE - DD Month YYYY
        pattern2 = r"^Race\s+(\d{1,2})\s+-\s+([A-Za-z0-9\s\'\&\.\-]+?)\s+-\s+(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})$"
        match = re.match(pattern2, name)
        if match:
            race_number, venue, day, month, year = match.groups()

            # Convert month name to number
            month_map = {
                "January": "01",
                "February": "02",
                "March": "03",
                "April": "04",
                "May": "05",
                "June": "06",
                "July": "07",
                "August": "08",
                "September": "09",
                "October": "10",
                "November": "11",
                "December": "12",
            }
            month_num = month_map.get(month, "01")
            date = f"{year}-{month_num}-{day.zfill(2)}"
            return venue.strip(), int(race_number), date

        return None, None, None

    def normalize_venue(self, venue):
        """Normalize venue name for race_id generation"""
        if not venue:
            return ""
        # Replace spaces and special chars with underscores, remove duplicates
        normalized = re.sub(r"[^A-Za-z0-9]", "_", venue.upper())
        normalized = re.sub(r"_+", "_", normalized)
        return normalized.strip("_")

    def generate_race_id(self, venue, date, race_number):
        """Generate consistent race_id"""
        normalized_venue = self.normalize_venue(venue)
        return f"{normalized_venue}_{date}_{race_number:02d}"

    def race_exists(self, cursor, race_id):
        """Check if race already exists in database"""
        cursor.execute(
            "SELECT COUNT(*) FROM race_metadata WHERE race_id = ?", (race_id,)
        )
        return cursor.fetchone()[0] > 0

    def process_csv_content(self, filepath):
        """Extract basic race info from CSV content"""
        dogs = []
        winner_name = None

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)

            # Look for dog names and winner in the data
            for row in rows:
                if len(row) > 0:
                    # Skip empty or header-like rows
                    if not row[0] or row[0].lower().startswith(
                        ("dog", "name", "pos", "#")
                    ):
                        continue

                    dog_name = row[0].strip()
                    if dog_name and len(dog_name) > 1:
                        dogs.append(dog_name)

                        # Check if this row indicates a winner (position 1)
                        if len(row) > 1:
                            pos_cell = str(row[1]).strip()
                            if pos_cell == "1" or pos_cell.lower() == "first":
                                winner_name = dog_name

            # Cap at 10 dogs max
            dogs = dogs[:10]

        except Exception as e:
            print(f"   ⚠️ Error reading CSV: {e}")

        return dogs, winner_name

    def save_race_to_db(
        self,
        cursor,
        race_id,
        venue,
        race_date,
        race_number,
        winner_name,
        dogs,
        filename,
    ):
        """Save race metadata and dog data to database"""

        # Determine status based on whether we have a winner
        if winner_name:
            results_status = "complete"
            winner_source = "inferred"
        else:
            results_status = "pending"
            winner_source = None

        # Insert race metadata
        cursor.execute(
            """
            INSERT INTO race_metadata (
                race_id, venue, race_date, race_number, winner_name,
                results_status, winner_source, extraction_timestamp,
                data_quality_note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                race_id,
                venue,
                race_date,
                race_number,
                winner_name,
                results_status,
                winner_source,
                datetime.now(),
                f"Processed from {filename}",
            ),
        )

        # Insert dog race data
        for i, dog_name in enumerate(dogs, 1):
            cursor.execute(
                """
                INSERT INTO dog_race_data (
                    race_id, dog_clean_name, box_number
                ) VALUES (?, ?, ?)
            """,
                (race_id, dog_name, i),
            )

    def process_single_file(self, filepath):
        """Process a single CSV file"""
        filename = os.path.basename(filepath)

        # Parse filename
        venue, race_number, race_date = self.parse_filename(filename)

        if not all([venue, race_number, race_date]):
            return "failed", f"Could not parse filename: {filename}"

        # Generate race_id
        race_id = self.generate_race_id(venue, race_date, race_number)

        # Check if already exists
        with self.connect_db() as conn:
            cursor = conn.cursor()

            if self.race_exists(cursor, race_id):
                return "skipped", f"Race {race_id} already exists"

            # Process CSV content
            dogs, winner_name = self.process_csv_content(filepath)

            if not dogs:
                return "failed", f"No dog data found in {filename}"

            # Save to database
            try:
                self.save_race_to_db(
                    cursor,
                    race_id,
                    venue,
                    race_date,
                    race_number,
                    winner_name,
                    dogs,
                    filename,
                )
                conn.commit()

                status = "complete" if winner_name else "pending"
                return (
                    "success",
                    f"Saved {race_id} ({len(dogs)} dogs, status: {status})",
                )

            except Exception as e:
                return "failed", f"Database error: {e}"

    def process_all_unprocessed(self, limit=None):
        """Process all files in unprocessed directory"""
        unprocessed_path = Path(self.unprocessed_dir)

        if not unprocessed_path.exists():
            print(f"❌ Unprocessed directory not found: {self.unprocessed_dir}")
            return

        # Get all CSV files
        csv_files = list(unprocessed_path.glob("*.csv"))

        if limit:
            csv_files = csv_files[:limit]

        print(f"🚀 Processing {len(csv_files)} files from {self.unprocessed_dir}/")
        print("=" * 60)

        results = {"success": 0, "failed": 0, "skipped": 0, "details": []}

        for i, filepath in enumerate(csv_files, 1):
            filename = filepath.name
            print(f"\n🔄 [{i:3d}/{len(csv_files)}] {filename}")

            status, message = self.process_single_file(filepath)

            if status == "success":
                print(f"   ✅ {message}")
                results["success"] += 1
            elif status == "skipped":
                print(f"   ⏭️ {message}")
                results["skipped"] += 1
            else:
                print(f"   ❌ {message}")
                results["failed"] += 1

            results["details"].append(
                {"filename": filename, "status": status, "message": message}
            )

            # Progress update every 50 files
            if i % 50 == 0:
                success_rate = (results["success"] / i) * 100
                print(
                    f"\n📊 Progress: {i}/{len(csv_files)} ({success_rate:.1f}% success rate)"
                )

        print("\n" + "=" * 60)
        print(f"📈 PROCESSING SUMMARY:")
        print(f"   ✅ Successful: {results['success']}")
        print(f"   ⏭️ Skipped: {results['skipped']}")
        print(f"   ❌ Failed: {results['failed']}")
        print(f"   📊 Success Rate: {(results['success']/(len(csv_files)) * 100):.1f}%")

        return results


def main():
    print("🏁 SIMPLE BULK PROCESSOR FOR UNPROCESSED FILES")
    print("=" * 60)

    processor = SimpleBulkProcessor()

    # Process all unprocessed files
    results = processor.process_all_unprocessed()

    print(f"\n🎯 Processing complete!")
    print(f"Check status with: python3 check_status_standalone.py")


if __name__ == "__main__":
    main()
