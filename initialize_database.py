#!/usr/bin/env python3
"""
Database Schema Initialization and CSV Data Migration
====================================================
This script creates the database schema and migrates existing CSV data into the database.
"""

import hashlib
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatabaseInitializer:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.db_path = self.base_path / "greyhound_racing_data.db"
        self.migration_stats = {
            "tables_created": 0,
            "records_imported": 0,
            "files_processed": 0,
            "errors": [],
        }

    def create_schema(self):
        """Create comprehensive database schema for greyhound racing data"""
        logger.info("Creating database schema...")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create venues table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS venues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                venue_code TEXT UNIQUE NOT NULL,
                venue_name TEXT,
                location TEXT,
                track_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create races table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS races (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_name TEXT NOT NULL,
                race_number INTEGER,
                race_date DATE NOT NULL,
                race_time TIME,
                venue_id INTEGER,
                distance INTEGER,
                grade TEXT,
                prize_money DECIMAL(10,2),
                track_condition TEXT,
                weather TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (venue_id) REFERENCES venues (id)
            )
        """
        )

        # Create dogs table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS dogs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dog_name TEXT NOT NULL,
                trainer TEXT,
                owner TEXT,
                color TEXT,
                sex TEXT,
                age INTEGER,
                weight DECIMAL(5,2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create race_entries table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS race_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id INTEGER NOT NULL,
                dog_id INTEGER NOT NULL,
                box_number INTEGER,
                starting_price DECIMAL(8,2),
                final_odds DECIMAL(8,2),
                finish_position INTEGER,
                finish_time DECIMAL(6,3),
                margin DECIMAL(6,2),
                section_times TEXT,
                comments TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (race_id) REFERENCES races (id),
                FOREIGN KEY (dog_id) REFERENCES dogs (id)
            )
        """
        )

        # Create form_guides table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS form_guides (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id INTEGER NOT NULL,
                dog_id INTEGER NOT NULL,
                recent_form TEXT,
                best_time DECIMAL(6,3),
                average_time DECIMAL(6,3),
                wins INTEGER DEFAULT 0,
                places INTEGER DEFAULT 0,
                starts INTEGER DEFAULT 0,
                earnings DECIMAL(10,2),
                rating INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (race_id) REFERENCES races (id),
                FOREIGN KEY (dog_id) REFERENCES dogs (id)
            )
        """
        )

        # Create enhanced_analysis table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS enhanced_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id INTEGER NOT NULL,
                analysis_type TEXT,
                prediction_data TEXT,
                confidence_score DECIMAL(5,2),
                model_version TEXT,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (race_id) REFERENCES races (id)
            )
        """
        )

        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_races_date ON races(race_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_races_venue ON races(venue_id)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_entries_race ON race_entries(race_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_entries_dog ON race_entries(dog_id)"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dogs_name ON dogs(dog_name)")

        conn.commit()
        conn.close()

        self.migration_stats["tables_created"] = 6
        logger.info("Database schema created successfully")

    def extract_venue_from_filename(self, filename):
        """Extract venue code from filename"""
        # Common venue patterns in filenames
        venue_patterns = [
            "AP_K",
            "APWE",
            "BAL",
            "BEN",
            "CANN",
            "CASO",
            "DAPT",
            "GEE",
            "GOSF",
            "GRDN",
            "HEA",
            "HOR",
            "LADBROKES",
            "MAND",
            "MOUNT",
            "MURR",
            "NOR",
            "QOT",
            "RICH",
            "SAL",
            "SAN",
            "TRA",
            "WAR",
            "W_PK",
        ]

        filename_upper = filename.upper()
        for venue in venue_patterns:
            if venue in filename_upper:
                return venue
        return "UNKNOWN"

    def extract_race_info_from_filename(self, filename):
        """Extract race information from filename"""
        import re

        # Extract race number
        race_match = re.search(r"Race\s*(\d+)", filename, re.IGNORECASE)
        race_number = int(race_match.group(1)) if race_match else 1

        # Extract date
        date_patterns = [
            r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})",
            r"(\d{4})-(\d{2})-(\d{2})",
            r"(\d{2})/(\d{2})/(\d{4})",
        ]

        race_date = None
        for pattern in date_patterns:
            date_match = re.search(pattern, filename, re.IGNORECASE)
            if date_match:
                try:
                    if "January" in pattern:  # Month name format
                        day, month_name, year = date_match.groups()
                        month_names = {
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
                        month = month_names.get(month_name.lower(), 1)
                        race_date = f"{year}-{month:02d}-{int(day):02d}"
                    elif "-" in pattern:  # YYYY-MM-DD format
                        year, month, day = date_match.groups()
                        race_date = f"{year}-{month}-{day}"
                    else:  # MM/DD/YYYY format
                        month, day, year = date_match.groups()
                        race_date = f"{year}-{month}-{day}"
                    break
                except:
                    continue

        if not race_date:
            race_date = "2025-01-01"  # Default date

        return race_number, race_date

    def insert_venue(self, conn, venue_code):
        """Insert venue and return venue_id"""
        cursor = conn.cursor()

        # Check if venue exists
        cursor.execute("SELECT id FROM venues WHERE venue_code = ?", (venue_code,))
        result = cursor.fetchone()

        if result:
            return result[0]

        # Insert new venue
        cursor.execute(
            """
            INSERT INTO venues (venue_code, venue_name)
            VALUES (?, ?)
        """,
            (venue_code, venue_code),
        )

        return cursor.lastrowid

    def insert_dog(self, conn, dog_name, trainer=None):
        """Insert dog and return dog_id"""
        cursor = conn.cursor()

        # Check if dog exists
        cursor.execute("SELECT id FROM dogs WHERE dog_name = ?", (dog_name,))
        result = cursor.fetchone()

        if result:
            return result[0]

        # Insert new dog
        cursor.execute(
            """
            INSERT INTO dogs (dog_name, trainer)
            VALUES (?, ?)
        """,
            (dog_name, trainer),
        )

        return cursor.lastrowid

    def migrate_csv_data(self):
        """Migrate data from CSV files to database"""
        logger.info("Starting CSV data migration...")

        conn = sqlite3.connect(self.db_path)

        # Find all race data CSV files
        csv_files = []
        for pattern in ["**/*race*.csv", "**/Race*.csv", "**/race_data/*.csv"]:
            csv_files.extend(self.base_path.glob(pattern))

        logger.info(f"Found {len(csv_files)} CSV files to process")

        for csv_file in csv_files:
            try:
                self.process_csv_file(conn, csv_file)
                self.migration_stats["files_processed"] += 1
            except Exception as e:
                error_msg = f"Error processing {csv_file}: {e}"
                logger.error(error_msg)
                self.migration_stats["errors"].append(error_msg)

        conn.commit()
        conn.close()

        logger.info(
            f"Migration completed. Processed {self.migration_stats['files_processed']} files"
        )

    def process_csv_file(self, conn, csv_file):
        """Process individual CSV file"""
        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                return

            # Extract race information from filename
            venue_code = self.extract_venue_from_filename(csv_file.name)
            race_number, race_date = self.extract_race_info_from_filename(csv_file.name)

            # Insert venue
            venue_id = self.insert_venue(conn, venue_code)

            # Insert race
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO races (race_name, race_number, race_date, venue_id)
                VALUES (?, ?, ?, ?)
            """,
                (csv_file.stem, race_number, race_date, venue_id),
            )

            race_id = cursor.lastrowid
            if race_id == 0:  # Race already exists
                cursor.execute(
                    """
                    SELECT id FROM races WHERE race_name = ? AND race_date = ? AND venue_id = ?
                """,
                    (csv_file.stem, race_date, venue_id),
                )
                result = cursor.fetchone()
                if result:
                    race_id = result[0]

            # Process each row in the CSV
            for _, row in df.iterrows():
                try:
                    # Insert dog (look for common dog name columns)
                    dog_name = None
                    for col in ["dog_name", "Dog", "Name", "Greyhound"]:
                        if col in df.columns:
                            dog_name = str(row[col]).strip()
                            break

                    if not dog_name or dog_name == "nan":
                        continue

                    trainer = None
                    for col in ["trainer", "Trainer", "Trainer_Name"]:
                        if col in df.columns:
                            trainer = str(row[col]).strip()
                            if trainer != "nan":
                                break

                    dog_id = self.insert_dog(conn, dog_name, trainer)

                    # Insert race entry
                    box_number = None
                    finish_position = None
                    starting_price = None

                    # Try to extract common fields
                    for col in ["box", "Box", "Box_Number", "Trap"]:
                        if col in df.columns:
                            try:
                                box_number = int(float(str(row[col])))
                                break
                            except:
                                pass

                    for col in ["position", "Position", "Finish", "Place"]:
                        if col in df.columns:
                            try:
                                finish_position = int(float(str(row[col])))
                                break
                            except:
                                pass

                    for col in ["odds", "Odds", "SP", "Starting_Price"]:
                        if col in df.columns:
                            try:
                                starting_price = float(str(row[col]))
                                break
                            except:
                                pass

                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO race_entries 
                        (race_id, dog_id, box_number, starting_price, finish_position)
                        VALUES (?, ?, ?, ?, ?)
                    """,
                        (race_id, dog_id, box_number, starting_price, finish_position),
                    )

                    self.migration_stats["records_imported"] += 1

                except Exception as e:
                    logger.debug(f"Error processing row in {csv_file}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error reading CSV {csv_file}: {e}")
            raise

    def generate_migration_report(self):
        """Generate migration report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get record counts
        counts = {}
        tables = [
            "venues",
            "races",
            "dogs",
            "race_entries",
            "form_guides",
            "enhanced_analysis",
        ]

        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = cursor.fetchone()[0]

        conn.close()

        report = {
            "migration_timestamp": datetime.now().isoformat(),
            "migration_stats": self.migration_stats,
            "database_counts": counts,
            "schema_version": "1.0",
        }

        report_path = self.base_path / "database_migration_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Migration report saved to {report_path}")
        return report

    def run_initialization(self):
        """Run complete database initialization"""
        logger.info("Starting database initialization...")

        self.create_schema()
        self.migrate_csv_data()
        report = self.generate_migration_report()

        logger.info("Database initialization completed successfully!")
        return report


def print_migration_summary(report):
    """Print migration summary"""
    print("\n" + "=" * 60)
    print("DATABASE MIGRATION SUMMARY")
    print("=" * 60)

    stats = report["migration_stats"]
    counts = report["database_counts"]

    print(f"Tables created: {stats['tables_created']}")
    print(f"Files processed: {stats['files_processed']}")
    print(f"Records imported: {stats['records_imported']:,}")
    print(f"Errors encountered: {len(stats['errors'])}")

    print(f"\nDatabase Contents:")
    for table, count in counts.items():
        print(f"  - {table}: {count:,} records")

    if stats["errors"]:
        print(f"\nErrors (first 3):")
        for error in stats["errors"][:3]:
            print(f"  - {error}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    base_path = "/Users/orlandolee/greyhound_racing_collector"
    initializer = DatabaseInitializer(base_path)

    try:
        report = initializer.run_initialization()
        print_migration_summary(report)

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        import traceback

        traceback.print_exc()
