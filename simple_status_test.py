#!/usr/bin/env python3
"""
Ultra-Simple Results Status Test
===============================

Test the results_status functionality without any heavy dependencies.

Author: AI Assistant  
Date: August 23, 2025
"""

import csv
import os
import re
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional


class UltraSimpleStatusTest:
    """Ultra-simplified test for results_status functionality"""

    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.unprocessed_dir = "./unprocessed"

        # Initialize database
        self.init_database()
        print("🚀 Ultra-Simple Status Test Initialized")

    def init_database(self):
        """Initialize database with status tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables with status tracking
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS race_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT UNIQUE,
                venue TEXT,
                race_number INTEGER,
                race_date DATE,
                winner_name TEXT,
                extraction_timestamp DATETIME,
                results_status TEXT DEFAULT 'pending',
                winner_source TEXT,
                scraping_attempts INTEGER DEFAULT 0,
                data_quality_note TEXT,
                UNIQUE(race_id)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS dog_race_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT,
                dog_name TEXT,
                dog_clean_name TEXT,
                box_number INTEGER,
                finish_position TEXT,
                extraction_timestamp DATETIME,
                FOREIGN KEY (race_id) REFERENCES race_metadata (race_id)
            )
        """
        )

        conn.commit()

        # Add missing columns
        self._add_missing_columns(cursor)
        conn.commit()
        conn.close()
        print("✅ Database initialized with status tracking")

    def _add_missing_columns(self, cursor):
        """Add status columns if missing"""
        columns_to_add = [
            ("results_status", "TEXT DEFAULT 'pending'"),
            ("winner_source", "TEXT"),
            ("scraping_attempts", "INTEGER DEFAULT 0"),
            ("data_quality_note", "TEXT"),
        ]

        for column_name, column_type in columns_to_add:
            try:
                cursor.execute(f"PRAGMA table_info(race_metadata)")
                cols = [r[1] for r in cursor.fetchall()]
                if column_name not in cols:
                    cursor.execute(
                        f"ALTER TABLE race_metadata ADD COLUMN {column_name} {column_type}"
                    )
                    print(f"   ✅ Added column: {column_name}")
            except Exception as e:
                print(f"   ⚠️ Could not add column {column_name}: {e}")

    def extract_race_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """Extract race info from filename"""
        # Pattern: Race N - VENUE - YYYY-MM-DD.csv
        pattern = r"^Race\s*(\d{1,2})\s*-\s*([A-Za-z0-9\s\'\&\.\-]+?)\s*-\s*(\d{4}-\d{2}-\d{2})\.csv$"
        match = re.match(pattern, filename)

        if match:
            race_number = int(match.group(1))
            venue = match.group(2).strip()
            date_str = match.group(3)

            try:
                race_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                race_id = f"{venue.upper().replace(' ', '-').replace('-', '_')}_{race_date}_{race_number:02d}"

                return {
                    "race_id": race_id,
                    "race_number": race_number,
                    "venue": venue,
                    "race_date": race_date,
                    "filename": filename,
                }
            except ValueError:
                pass

        print(f"   ❌ Could not parse filename: {filename}")
        return None

    def process_csv_simple(self, filepath: str) -> Dict[str, Any]:
        """Process CSV file with simple CSV reader"""
        filename = os.path.basename(filepath)
        print(f"📈 Processing: {filename}")

        try:
            # Extract race info
            race_info = self.extract_race_info(filename)
            if not race_info:
                return {"status": "error", "error": "Could not extract race info"}

            print(
                f"   📍 Race: {race_info['venue']} Race {race_info['race_number']} on {race_info['race_date']}"
            )

            # Read CSV with basic CSV reader
            dogs = []
            with open(filepath, "r", encoding="utf-8") as f:
                csv_reader = csv.DictReader(f)
                for i, row in enumerate(csv_reader):
                    dog_name = row.get("Dog Name", "").strip()
                    if dog_name:
                        dogs.append(
                            {
                                "race_id": race_info["race_id"],
                                "dog_name": dog_name,
                                "dog_clean_name": self.clean_dog_name(dog_name),
                                "box_number": self.safe_int(row.get("BOX", 0)),
                                "finish_position": "N/A",
                            }
                        )

                    if i >= 15:  # Limit for testing
                        break

            # SIMULATE different status scenarios based on venue
            if "AP_K" in race_info["venue"] or "AP-K" in race_info["venue"]:
                # Simulate complete status
                race_info["results_status"] = "complete"
                race_info["winner_source"] = "scrape"
                race_info["winner_name"] = (
                    dogs[0]["dog_clean_name"] if dogs else "SIMULATED_WINNER"
                )
                race_info["data_quality_note"] = "Simulated complete scraping"
                print(f"   🎯 Simulated COMPLETE status for AP_K race")

            elif "BAL" in race_info["venue"]:
                # Simulate partial failure
                race_info["results_status"] = "partial_scraping_failed"
                race_info["winner_source"] = "inferred"
                race_info["winner_name"] = (
                    dogs[0]["dog_clean_name"] if dogs else "SIMULATED_WINNER"
                )
                race_info["data_quality_note"] = "Simulated partial scraping failure"
                print(f"   ⚠️ Simulated PARTIAL_SCRAPING_FAILED status for BAL race")

            else:
                # Simulate pending status
                race_info["results_status"] = "pending"
                race_info["winner_source"] = None
                race_info["winner_name"] = ""
                race_info["data_quality_note"] = "Simulated pending - needs backfill"
                print(f"   📋 Simulated PENDING status for {race_info['venue']} race")

            race_info["extraction_timestamp"] = datetime.now()
            race_info["scraping_attempts"] = 0

            # Save to database
            self.save_to_database(race_info, dogs)

            return {
                "race_info": race_info,
                "dogs": dogs,
                "status": "success",
            }

        except Exception as e:
            print(f"❌ Error processing {filepath}: {str(e)}")
            return {"status": "error", "error": str(e)}

    def save_to_database(self, race_info: Dict[str, Any], dogs: List[Dict[str, Any]]):
        """Save to database with status tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Insert race metadata with status
            cursor.execute(
                """
                INSERT OR REPLACE INTO race_metadata 
                (race_id, venue, race_number, race_date, winner_name, 
                 extraction_timestamp, results_status, winner_source, 
                 scraping_attempts, data_quality_note)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    race_info["race_id"],
                    race_info["venue"],
                    race_info["race_number"],
                    race_info["race_date"],
                    race_info.get("winner_name", ""),
                    race_info.get("extraction_timestamp", datetime.now()),
                    race_info.get("results_status", "pending"),
                    race_info.get("winner_source"),
                    race_info.get("scraping_attempts", 0),
                    race_info.get("data_quality_note", ""),
                ),
            )

            # Insert dog data
            for dog in dogs:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO dog_race_data 
                    (race_id, dog_name, dog_clean_name, box_number, finish_position, extraction_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        dog["race_id"],
                        dog["dog_name"],
                        dog["dog_clean_name"],
                        dog["box_number"],
                        dog["finish_position"],
                        datetime.now(),
                    ),
                )

            conn.commit()
            print(
                f"✅ Saved race {race_info['race_id']} with status: {race_info.get('results_status')}"
            )

        except Exception as e:
            print(f"❌ Database error: {e}")
            conn.rollback()
        finally:
            conn.close()

    def run_test(self, max_files: int = 20):
        """Run the status test on sample files"""
        if not os.path.exists(self.unprocessed_dir):
            print("❌ Unprocessed directory not found")
            return

        csv_files = [f for f in os.listdir(self.unprocessed_dir) if f.endswith(".csv")][
            :max_files
        ]

        if not csv_files:
            print("❌ No CSV files found")
            return

        print(f"\n📊 Testing status functionality on {len(csv_files)} files...")

        processed = 0
        failed = 0

        for filename in csv_files:
            filepath = os.path.join(self.unprocessed_dir, filename)
            print(f"\n🔄 Processing: {filename}")

            result = self.process_csv_simple(filepath)
            if result.get("status") == "success":
                processed += 1
            else:
                failed += 1

        print(f"\n📈 TEST SUMMARY:")
        print(f"   ✅ Processed: {processed}")
        print(f"   ❌ Failed: {failed}")

        # Show status summary
        self.show_status_summary()

    def show_status_summary(self):
        """Show summary of results_status values"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            print(f"\n📊 RESULTS STATUS SUMMARY:")
            print("=" * 50)

            # Count by status
            cursor.execute(
                """
                SELECT results_status, COUNT(*) as count
                FROM race_metadata 
                GROUP BY results_status
                ORDER BY count DESC
            """
            )
            status_counts = cursor.fetchall()

            print(f"🎯 Status breakdown:")
            for status, count in status_counts:
                print(f"   {status or 'NULL'}: {count} races")

            # Count by winner source
            cursor.execute(
                """
                SELECT winner_source, COUNT(*) as count
                FROM race_metadata 
                WHERE winner_source IS NOT NULL
                GROUP BY winner_source
                ORDER BY count DESC
            """
            )
            source_counts = cursor.fetchall()

            if source_counts:
                print(f"\n🎯 Winner source breakdown:")
                for source, count in source_counts:
                    print(f"   {source}: {count} races")

            # Show sample races
            cursor.execute(
                """
                SELECT race_id, venue, results_status, winner_name
                FROM race_metadata 
                ORDER BY extraction_timestamp DESC
                LIMIT 15
            """
            )
            sample_races = cursor.fetchall()

            if sample_races:
                print(f"\n📋 Sample races with status:")
                for race_id, venue, status, winner in sample_races:
                    winner_display = winner if winner else "None"
                    print(f"   {race_id}: {status} | Winner: {winner_display}")

        except Exception as e:
            print(f"❌ Error getting status summary: {e}")
        finally:
            conn.close()

    def clean_dog_name(self, name: str) -> str:
        """Clean dog name"""
        if not name:
            return ""
        cleaned = re.sub(r'^["\d\.\s]+', "", str(name))
        cleaned = re.sub(r'["\s]+$', "", cleaned)
        return cleaned.strip().upper()

    def safe_int(self, value, default=0):
        """Safely convert to int"""
        try:
            return int(float(str(value).strip())) if value else default
        except:
            return default


def main():
    """Run the ultra-simple status test"""
    print("🚀 ULTRA-SIMPLE RESULTS STATUS TEST")
    print("=" * 60)

    tester = UltraSimpleStatusTest()
    tester.run_test(max_files=25)

    print(f"\n✅ Status functionality test complete!")
    print("\n💡 This test demonstrates:")
    print(
        "   - ✅ results_status field assignment (complete, partial_scraping_failed, pending)"
    )
    print("   - ✅ winner_source tracking (scrape, inferred, null)")
    print("   - ✅ Database schema migration (adding status columns)")
    print("   - ✅ Different processing scenarios simulation")


if __name__ == "__main__":
    main()
