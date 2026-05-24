#!/usr/bin/env python3
"""
Simple Test Processor for Results Status Testing
===============================================

A simplified version of the processor to test the new results_status functionality
without heavy dependencies like numpy, selenium, etc.

Author: AI Assistant
Date: August 23, 2025
"""

import json
import os
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class SimpleStatusTestProcessor:
    """
    Simplified processor to test results_status functionality
    """

    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.unprocessed_dir = "./unprocessed"
        self.processed_dir = "./processed"

        # Create directories
        os.makedirs(self.processed_dir, exist_ok=True)

        # Initialize database
        self.init_database()

        print("🚀 Simple Status Test Processor Initialized")

    def init_database(self):
        """Initialize database with status tracking columns"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Enhanced race metadata table with status tracking
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS race_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT UNIQUE,
                venue TEXT,
                race_number INTEGER,
                race_date DATE,
                race_name TEXT,
                grade TEXT,
                distance TEXT,
                field_size INTEGER,
                winner_name TEXT,
                extraction_timestamp DATETIME,
                data_source TEXT,
                results_status TEXT DEFAULT 'pending',
                winner_source TEXT,
                scraping_attempts INTEGER DEFAULT 0,
                last_scraped_at DATETIME,
                parse_confidence REAL DEFAULT 1.0,
                data_quality_note TEXT,
                UNIQUE(race_id)
            )
        """
        )

        # Dog race data table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS dog_race_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT,
                dog_name TEXT,
                dog_clean_name TEXT,
                box_number INTEGER,
                finish_position TEXT,
                trainer_name TEXT,
                extraction_timestamp DATETIME,
                data_source TEXT,
                FOREIGN KEY (race_id) REFERENCES race_metadata (race_id)
            )
        """
        )

        conn.commit()

        # Add missing columns if they don't exist
        self._migrate_database_schema(cursor)
        conn.commit()
        conn.close()
        print("✅ Database initialized with status tracking")

    def _migrate_database_schema(self, cursor):
        """Add status tracking columns if missing"""
        status_columns = [
            ("results_status", "TEXT DEFAULT 'pending'"),
            ("winner_source", "TEXT"),
            ("scraping_attempts", "INTEGER DEFAULT 0"),
            ("last_scraped_at", "DATETIME"),
            ("parse_confidence", "REAL DEFAULT 1.0"),
            ("data_quality_note", "TEXT"),
        ]

        for column_name, column_type in status_columns:
            self._add_column_if_missing(
                cursor, "race_metadata", column_name, column_type
            )

    def _add_column_if_missing(self, cursor, table: str, column: str, col_type: str):
        """Add a column to a table if it doesn't already exist"""
        try:
            cursor.execute(f"PRAGMA table_info({table})")
            cols = [r[1] for r in cursor.fetchall()]
            if column not in cols:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                print(f"   ✅ Added column: {table}.{column}")
        except Exception as e:
            print(f"   ⚠️ Could not add column {table}.{column}: {e}")

    def extract_race_info_from_filename_and_csv(
        self, filepath: str, df: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Extract race information from CSV filename"""
        filename = os.path.basename(filepath)
        base_filename = filename.replace(".csv", "")

        print(f"   🔍 Parsing filename: {filename}")

        # Enhanced patterns supporting hyphenated venues
        patterns = [
            (
                r"^Race\s*(\d{1,2})\s*[-_]\s*([A-Za-z0-9\s\'&\.\-]+?)\s*[-_]\s*(\d{4}[-/]\d{2}[-/]\d{2})$",
                "%Y-%m-%d",
            ),
        ]

        race_number = None
        venue = None
        race_date = None

        # Try each pattern
        for pattern, date_format in patterns:
            match = re.search(pattern, base_filename)
            if match:
                groups = match.groups()
                race_number = int(groups[0])
                venue = groups[1].strip()
                date_str = groups[2].replace("/", "-")

                try:
                    race_date = datetime.strptime(date_str, date_format).date()
                    print(
                        f"   ✅ Filename parsed: Venue='{venue}', Race={race_number}, Date={race_date}"
                    )
                    break
                except ValueError as e:
                    print(f"   ⚠️ Date parsing failed: {e}")
                    continue

        if race_number is not None and venue is not None and race_date is not None:
            # Generate race ID
            canonical_venue = re.sub(r"[-\s]+", "-", venue.upper())
            race_id = f"{canonical_venue}_{race_date}_{race_number:02d}"

            return {
                "race_id": race_id,
                "race_number": race_number,
                "venue": venue,
                "race_date": race_date,
                "filename": filename,
                "grade": "",
                "distance": "",
            }

        print(f"   ❌ Could not extract race info from filename")
        return None

    def process_csv_file(self, csv_file_path: str) -> Dict[str, Any]:
        """Process a single CSV file and test results_status assignment"""
        print(f"📈 Processing: {os.path.basename(csv_file_path)}")

        try:
            # Read CSV file
            df = pd.read_csv(csv_file_path)

            # Extract race information
            race_info = self.extract_race_info_from_filename_and_csv(csv_file_path, df)
            if not race_info:
                return {"status": "error", "error": "Could not extract race info"}

            print(
                f"   📍 Race: {race_info['venue']} Race {race_info['race_number']} on {race_info['race_date']}"
            )

            # Process dogs (simplified)
            processed_dogs = []
            for index, row in df.iterrows():
                dog_name = str(row.get("Dog Name", "")).strip()
                if dog_name and dog_name != "":
                    dog_data = {
                        "race_id": race_info["race_id"],
                        "dog_name": dog_name,
                        "dog_clean_name": self.clean_dog_name(dog_name),
                        "box_number": self.safe_int(row.get("BOX", 0)),
                        "finish_position": "N/A",  # No scraping in test
                        "trainer_name": "",
                        "extraction_timestamp": datetime.now(),
                        "data_source": "simple_test_processor",
                    }
                    processed_dogs.append(dog_data)

            # SIMULATE different processing scenarios for testing results_status
            has_scraped_data = False  # Simulating no web scraping
            has_winner = False  # Simulating no winner found
            has_url = False  # Simulating no URL
            has_positioned_dogs = False  # Simulating no finish positions

            # Test scenario based on filename to create variety
            if "AP_K" in race_info["venue"]:
                # Simulate successful scraping for AP_K races
                has_scraped_data = True
                has_winner = True
                has_url = True
                race_info["winner_name"] = (
                    processed_dogs[0]["dog_clean_name"]
                    if processed_dogs
                    else "Test Winner"
                )
                race_info["results_status"] = "complete"
                race_info["winner_source"] = "scrape"
                print(f"   🎯 Simulated successful scraping for AP_K race")

            elif "BAL" in race_info["venue"]:
                # Simulate partial scraping failure
                has_scraped_data = True
                has_winner = True
                has_url = True
                race_info["winner_name"] = (
                    processed_dogs[0]["dog_clean_name"]
                    if processed_dogs
                    else "Test Winner"
                )
                race_info["results_status"] = "partial_scraping_failed"
                race_info["winner_source"] = "inferred"
                race_info["data_quality_note"] = (
                    "Scraping had issues but winner determined"
                )
                print(f"   ⚠️ Simulated partial scraping failure for BAL race")

            else:
                # Simulate pending status (no winner, needs backfill)
                race_info["winner_name"] = ""
                race_info["results_status"] = "pending"
                race_info["winner_source"] = None
                race_info["data_quality_note"] = (
                    "Winner pending - requires scraping backfill"
                )
                print(f"   📋 Simulated pending status for {race_info['venue']} race")

            # Add metadata
            race_info["extraction_timestamp"] = datetime.now()
            race_info["data_source"] = "simple_test_processor"
            race_info["field_size"] = len(processed_dogs)
            race_info["scraping_attempts"] = 0
            race_info["parse_confidence"] = 1.0

            # Save to database
            self.save_to_database(race_info, processed_dogs)

            # Move file to processed directory
            self.move_to_processed(csv_file_path, status="success")

            return {
                "race_info": race_info,
                "dogs": processed_dogs,
                "status": "success",
            }

        except Exception as e:
            print(f"❌ Error processing {csv_file_path}: {str(e)}")
            return {"status": "error", "error": str(e)}

    def save_to_database(self, race_info: Dict[str, Any], dogs: List[Dict[str, Any]]):
        """Save race data to database with results_status tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Save race metadata with status tracking
            cursor.execute(
                """
                INSERT OR REPLACE INTO race_metadata 
                (race_id, venue, race_number, race_date, race_name, grade, distance, 
                 field_size, winner_name, extraction_timestamp, data_source, 
                 results_status, winner_source, scraping_attempts, parse_confidence, data_quality_note)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    race_info["race_id"],
                    race_info["venue"],
                    race_info["race_number"],
                    race_info["race_date"],
                    race_info.get("race_name", ""),
                    race_info.get("grade", ""),
                    race_info.get("distance", ""),
                    race_info.get("field_size"),
                    race_info.get("winner_name", ""),
                    race_info.get("extraction_timestamp", datetime.now()),
                    race_info.get("data_source", "simple_test_processor"),
                    race_info.get("results_status", "pending"),
                    race_info.get("winner_source"),
                    race_info.get("scraping_attempts", 0),
                    race_info.get("parse_confidence", 1.0),
                    race_info.get("data_quality_note", ""),
                ),
            )

            # Save dog data
            for dog in dogs:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO dog_race_data 
                    (race_id, dog_name, dog_clean_name, box_number, finish_position, 
                     trainer_name, extraction_timestamp, data_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        dog["race_id"],
                        dog["dog_name"],
                        dog["dog_clean_name"],
                        dog["box_number"],
                        dog["finish_position"],
                        dog["trainer_name"],
                        dog["extraction_timestamp"],
                        dog["data_source"],
                    ),
                )

            conn.commit()
            print(
                f"✅ Saved race {race_info['race_id']} with status: {race_info.get('results_status', 'unknown')}"
            )

        except Exception as e:
            print(f"❌ Database error: {e}")
            conn.rollback()
        finally:
            conn.close()

    def move_to_processed(self, csv_file_path: str, status: str = "success"):
        """Move processed file to processed directory"""
        try:
            filename = os.path.basename(csv_file_path)
            processed_path = os.path.join(self.processed_dir, filename)

            if not os.path.exists(processed_path):
                import shutil

                shutil.move(csv_file_path, processed_path)
                print(f"📁 Moved {filename} to processed directory")
        except Exception as e:
            print(f"⚠️ Error moving file: {e}")

    def process_sample_files(self, max_files: int = 10) -> Dict[str, Any]:
        """Process a sample of files to test results_status functionality"""
        if not os.path.exists(self.unprocessed_dir):
            return {"status": "error", "message": "Unprocessed directory not found"}

        csv_files = [f for f in os.listdir(self.unprocessed_dir) if f.endswith(".csv")][
            :max_files
        ]

        if not csv_files:
            return {
                "status": "success",
                "message": "No files to process",
                "processed_count": 0,
            }

        results = {
            "status": "success",
            "processed_count": 0,
            "failed_count": 0,
            "results": [],
        }

        print(f"\n📊 Processing {len(csv_files)} sample files...")

        for filename in csv_files:
            file_path = os.path.join(self.unprocessed_dir, filename)
            print(f"\n🔄 Processing: {filename}")

            result = self.process_csv_file(file_path)
            if result and result.get("status") == "success":
                results["processed_count"] += 1
            else:
                results["failed_count"] += 1

            results["results"].append({"filename": filename, "result": result})

        print(f"\n📈 Processing Summary:")
        print(f"   ✅ Processed: {results['processed_count']}")
        print(f"   ❌ Failed: {results['failed_count']}")

        return results

    def get_results_status_summary(self) -> Dict[str, Any]:
        """Get summary of results_status values in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Count by results_status
            cursor.execute(
                """
                SELECT results_status, COUNT(*) as count
                FROM race_metadata 
                GROUP BY results_status
                ORDER BY count DESC
            """
            )
            status_counts = dict(cursor.fetchall())

            # Count by winner_source
            cursor.execute(
                """
                SELECT winner_source, COUNT(*) as count
                FROM race_metadata 
                WHERE winner_source IS NOT NULL
                GROUP BY winner_source
                ORDER BY count DESC
            """
            )
            source_counts = dict(cursor.fetchall())

            # Total races
            cursor.execute("SELECT COUNT(*) FROM race_metadata")
            total_races = cursor.fetchone()[0]

            # Sample races by status
            cursor.execute(
                """
                SELECT race_id, venue, race_number, results_status, winner_source, winner_name
                FROM race_metadata 
                ORDER BY extraction_timestamp DESC
                LIMIT 20
            """
            )
            sample_races = cursor.fetchall()

            summary = {
                "total_races": total_races,
                "status_counts": status_counts,
                "source_counts": source_counts,
                "sample_races": [
                    {
                        "race_id": row[0],
                        "venue": row[1],
                        "race_number": row[2],
                        "results_status": row[3],
                        "winner_source": row[4],
                        "winner_name": row[5] or "None",
                    }
                    for row in sample_races
                ],
            }

            return summary

        except Exception as e:
            return {"error": str(e)}
        finally:
            conn.close()

    # Helper methods
    def safe_int(self, value, default=0):
        try:
            return int(float(str(value).strip())) if value else default
        except:
            return default

    def clean_dog_name(self, name: str) -> str:
        if not name:
            return ""
        cleaned = re.sub(r'^["\d\.\s]+', "", str(name))
        cleaned = re.sub(r'["\s]+$', "", cleaned)
        return cleaned.strip().upper()


def main():
    """Test the results_status functionality"""
    print("🚀 RESULTS STATUS TEST PROCESSOR")
    print("=" * 50)

    processor = SimpleStatusTestProcessor()

    # Process sample files
    results = processor.process_sample_files(max_files=15)

    print(f"\n📊 PROCESSING COMPLETE")
    print("=" * 50)
    print(f"✅ Successfully processed: {results.get('processed_count', 0)} files")
    print(f"❌ Failed to process: {results.get('failed_count', 0)} files")

    # Show results_status summary
    print(f"\n📈 RESULTS STATUS SUMMARY")
    print("=" * 50)
    summary = processor.get_results_status_summary()

    if "error" not in summary:
        print(f"📊 Total races in database: {summary['total_races']}")
        print(f"\n🎯 Status breakdown:")
        for status, count in summary["status_counts"].items():
            print(f"   {status or 'NULL'}: {count}")

        print(f"\n🎯 Winner source breakdown:")
        for source, count in summary["source_counts"].items():
            print(f"   {source}: {count}")

        print(f"\n📋 Sample races:")
        for race in summary["sample_races"][:10]:
            print(
                f"   {race['race_id']}: {race['results_status']} | Winner: {race['winner_name']}"
            )
    else:
        print(f"❌ Error getting summary: {summary['error']}")


if __name__ == "__main__":
    main()
