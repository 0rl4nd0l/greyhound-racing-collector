#!/usr/bin/env python3
"""
Database Repair and Maintenance System
======================================

Comprehensive system to identify and fix database issues in the greyhound racing system.
Addresses data quality, integrity, and consistency problems.

Author: AI Assistant
Date: July 28, 2025
"""

import json
import logging
import os
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


class DatabaseRepairSystem:
    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.backup_dir = Path("./database_backups")
        self.backup_dir.mkdir(exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("logs/database_repair.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

        print("🔧 Database Repair System Initialized")

    def create_backup(self):
        """Create a backup of the current database"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = (
                self.backup_dir / f"greyhound_racing_data_backup_{timestamp}.db"
            )

            # Create backup using sqlite3 backup API
            source = sqlite3.connect(self.db_path)
            backup = sqlite3.connect(str(backup_path))
            source.backup(backup)
            source.close()
            backup.close()

            self.logger.info(f"✅ Database backup created: {backup_path}")
            return str(backup_path)

        except Exception as e:
            self.logger.error(f"❌ Failed to create backup: {e}")
            return None

    def analyze_database_issues(self):
        """Comprehensive analysis of database issues"""
        try:
            conn = sqlite3.connect(self.db_path)
            issues = {}

            print("\n🔍 ANALYZING DATABASE ISSUES")
            print("=" * 50)

            # 1. Race metadata issues
            print("\n📊 Race Metadata Analysis:")

            # Missing winners
            missing_winners = pd.read_sql_query(
                """
                SELECT race_id, venue, race_date, race_name 
                FROM race_metadata 
                WHERE winner_name IS NULL OR winner_name = '' OR winner_name = 'nan'
                ORDER BY race_date DESC
            """,
                conn,
            )

            issues["missing_winners"] = len(missing_winners)
            print(f"   • Missing winners: {len(missing_winners)}")

            # Missing critical race info
            missing_venue = pd.read_sql_query(
                """
                SELECT COUNT(*) as count FROM race_metadata 
                WHERE venue IS NULL OR venue = ''
            """,
                conn,
            )
            issues["missing_venue"] = missing_venue.iloc[0]["count"]
            print(f"   • Missing venue: {issues['missing_venue']}")

            missing_date = pd.read_sql_query(
                """
                SELECT COUNT(*) as count FROM race_metadata 
                WHERE race_date IS NULL OR race_date = ''
            """,
                conn,
            )
            issues["missing_date"] = missing_date.iloc[0]["count"]
            print(f"   • Missing race date: {issues['missing_date']}")

            # 2. Dog race data issues
            print("\n🐕 Dog Race Data Analysis:")

            # Missing box numbers
            missing_box = pd.read_sql_query(
                """
                SELECT COUNT(*) as count FROM dog_race_data 
                WHERE box_number IS NULL
            """,
                conn,
            )
            issues["missing_box_numbers"] = missing_box.iloc[0]["count"]
            print(f"   • Missing box numbers: {issues['missing_box_numbers']}")

            # Missing dog names
            missing_dogs = pd.read_sql_query(
                """
                SELECT COUNT(*) as count FROM dog_race_data 
                WHERE dog_name IS NULL OR dog_name = '' OR dog_name = 'nan'
            """,
                conn,
            )
            issues["missing_dog_names"] = missing_dogs.iloc[0]["count"]
            print(f"   • Missing dog names: {issues['missing_dog_names']}")

            # Orphaned dog records
            orphaned = pd.read_sql_query(
                """
                SELECT COUNT(*) as count FROM dog_race_data d 
                WHERE NOT EXISTS (
                    SELECT 1 FROM race_metadata r WHERE r.race_id = d.race_id
                )
            """,
                conn,
            )
            issues["orphaned_records"] = orphaned.iloc[0]["count"]
            print(f"   • Orphaned dog records: {issues['orphaned_records']}")

            # 3. Data quality issues
            print("\n🎯 Data Quality Analysis:")

            # Invalid odds
            invalid_odds = pd.read_sql_query(
                """
                SELECT COUNT(*) as count FROM dog_race_data 
                WHERE (odds_decimal IS NOT NULL AND odds_decimal <= 0) 
                   OR (starting_price IS NOT NULL AND starting_price <= 0)
            """,
                conn,
            )
            issues["invalid_odds"] = invalid_odds.iloc[0]["count"]
            print(f"   • Invalid odds: {issues['invalid_odds']}")

            # Inconsistent finish positions
            invalid_positions = pd.read_sql_query(
                """
                SELECT race_id, COUNT(*) as same_position_count
                FROM dog_race_data 
                WHERE finish_position IS NOT NULL 
                  AND finish_position != 'N/A' 
                  AND finish_position != ''
                GROUP BY race_id, finish_position
                HAVING COUNT(*) > 1
            """,
                conn,
            )
            issues["duplicate_positions"] = len(invalid_positions)
            print(
                f"   • Races with duplicate finish positions: {issues['duplicate_positions']}"
            )

            conn.close()

            # Summary
            total_issues = sum(issues.values())
            print(f"\n📋 SUMMARY: {total_issues} issues found")

            return issues

        except Exception as e:
            self.logger.error(f"❌ Error analyzing database: {e}")
            return {}

    def fix_missing_box_numbers(self):
        """Fix missing box numbers by inferring from dog order or position"""
        try:
            conn = sqlite3.connect(self.db_path)

            # Get records with missing box numbers
            missing_box_query = """
                SELECT id, race_id, dog_name, finish_position
                FROM dog_race_data 
                WHERE box_number IS NULL
                ORDER BY race_id, id
            """

            missing_records = pd.read_sql_query(missing_box_query, conn)

            if len(missing_records) == 0:
                print("✅ No missing box numbers to fix")
                return

            fixed_count = 0

            for race_id in missing_records["race_id"].unique():
                race_dogs = missing_records[missing_records["race_id"] == race_id]

                # Strategy 1: Use finish position if available
                for idx, row in race_dogs.iterrows():
                    box_number = None

                    # Try to use finish position as box number
                    if (
                        pd.notna(row["finish_position"])
                        and str(row["finish_position"]).isdigit()
                    ):
                        box_number = int(row["finish_position"])
                    else:
                        # Strategy 2: Assign sequential box numbers
                        existing_boxes = pd.read_sql_query(
                            """
                            SELECT DISTINCT box_number FROM dog_race_data 
                            WHERE race_id = ? AND box_number IS NOT NULL
                        """,
                            conn,
                            params=[race_id],
                        )

                        used_boxes = (
                            set(existing_boxes["box_number"].tolist())
                            if len(existing_boxes) > 0
                            else set()
                        )

                        # Find next available box number
                        for i in range(1, 9):  # Standard 8 boxes
                            if i not in used_boxes:
                                box_number = i
                                break

                    if box_number:
                        conn.execute(
                            """
                            UPDATE dog_race_data 
                            SET box_number = ? 
                            WHERE id = ?
                        """,
                            (box_number, row["id"]),
                        )
                        fixed_count += 1

            conn.commit()
            conn.close()

            print(f"✅ Fixed {fixed_count} missing box numbers")

        except Exception as e:
            self.logger.error(f"❌ Error fixing box numbers: {e}")

    def fix_missing_race_metadata(self):
        """Fix missing race metadata by inferring from file names and data"""
        try:
            conn = sqlite3.connect(self.db_path)

            # Get races with missing metadata
            missing_metadata = pd.read_sql_query(
                """
                SELECT race_id, venue, race_date, race_name, url
                FROM race_metadata 
                WHERE venue IS NULL OR venue = '' 
                   OR race_date IS NULL OR race_date = ''
            """,
                conn,
            )

            if len(missing_metadata) == 0:
                print("✅ No missing race metadata to fix")
                conn.close()
                return

            fixed_count = 0

            for _, row in missing_metadata.iterrows():
                race_id = row["race_id"]
                updates = {}

                # Try to extract info from race_id or URL
                if race_id:
                    # Pattern: Race_XX_VENUE_DATE or similar
                    match = re.search(
                        r"Race[_\s]*(\d+)[_\s]*([A-Z_]+)[_\s]*([\d-]+)",
                        race_id,
                        re.IGNORECASE,
                    )
                    if match:
                        race_num, venue, date = match.groups()

                        if not row["venue"] or row["venue"] == "":
                            updates["venue"] = venue.replace("_", "")

                        if not row["race_date"] or row["race_date"] == "":
                            # Try to parse date
                            try:
                                if len(date) == 8:  # YYYYMMDD
                                    parsed_date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
                                else:
                                    parsed_date = date
                                updates["race_date"] = parsed_date
                            except:
                                pass

                # Apply updates
                if updates:
                    update_query = "UPDATE race_metadata SET "
                    update_values = []

                    for key, value in updates.items():
                        update_query += f"{key} = ?, "
                        update_values.append(value)

                    update_query = update_query.rstrip(", ") + " WHERE race_id = ?"
                    update_values.append(race_id)

                    conn.execute(update_query, update_values)
                    fixed_count += 1

            conn.commit()
            conn.close()

            print(f"✅ Fixed metadata for {fixed_count} races")

        except Exception as e:
            self.logger.error(f"❌ Error fixing race metadata: {e}")

    def fix_duplicate_finish_positions(self):
        """Fix races where multiple dogs have the same finish position"""
        try:
            conn = sqlite3.connect(self.db_path)

            # Find races with duplicate positions
            duplicate_positions = pd.read_sql_query(
                """
                SELECT race_id, finish_position, COUNT(*) as count
                FROM dog_race_data 
                WHERE finish_position IS NOT NULL 
                  AND finish_position != 'N/A' 
                  AND finish_position != ''
                  AND finish_position != 'DNS'
                  AND finish_position != 'DNF'
                GROUP BY race_id, finish_position
                HAVING COUNT(*) > 1
                ORDER BY race_id, finish_position
            """,
                conn,
            )

            if len(duplicate_positions) == 0:
                print("✅ No duplicate finish positions to fix")
                conn.close()
                return

            fixed_races = 0

            for _, row in duplicate_positions.iterrows():
                race_id = row["race_id"]
                position = row["finish_position"]

                # Get all dogs with this position in this race
                dogs_with_position = pd.read_sql_query(
                    """
                    SELECT id, dog_name, individual_time
                    FROM dog_race_data 
                    WHERE race_id = ? AND finish_position = ?
                    ORDER BY individual_time ASC NULLS LAST, id
                """,
                    conn,
                    params=[race_id, position],
                )

                # Reassign positions based on time or order
                for idx, dog in dogs_with_position.iterrows():
                    new_position = int(position) + idx

                    conn.execute(
                        """
                        UPDATE dog_race_data 
                        SET finish_position = ? 
                        WHERE id = ?
                    """,
                        (str(new_position), dog["id"]),
                    )

                fixed_races += 1

            conn.commit()
            conn.close()

            print(f"✅ Fixed duplicate positions in {fixed_races} races")

        except Exception as e:
            self.logger.error(f"❌ Error fixing duplicate positions: {e}")

    def optimize_database(self):
        """Optimize database performance"""
        try:
            conn = sqlite3.connect(self.db_path)

            print("\n🔧 Optimizing database...")

            # Analyze tables
            conn.execute("ANALYZE")

            # Vacuum database
            conn.execute("VACUUM")

            # Update statistics
            conn.execute("PRAGMA optimize")

            conn.close()

            print("✅ Database optimization complete")

        except Exception as e:
            self.logger.error(f"❌ Error optimizing database: {e}")

    def run_comprehensive_repair(self):
        """Run all repair operations"""
        print("🚀 Starting Comprehensive Database Repair")
        print("=" * 50)

        # Create backup first
        backup_path = self.create_backup()
        if not backup_path:
            print("❌ Cannot proceed without backup")
            return False

        try:
            # Analyze issues
            issues = self.analyze_database_issues()

            if sum(issues.values()) == 0:
                print("\n✅ No issues found - database is healthy!")
                return True

            print(f"\n🔨 Starting repairs...")

            # Fix issues in order of priority
            self.fix_missing_box_numbers()
            self.fix_missing_race_metadata()
            self.fix_duplicate_finish_positions()

            # Optimize database
            self.optimize_database()

            # Final analysis
            print("\n🎯 POST-REPAIR ANALYSIS")
            print("=" * 30)
            final_issues = self.analyze_database_issues()

            initial_count = sum(issues.values())
            final_count = sum(final_issues.values())
            fixed_count = initial_count - final_count

            print(f"\n📊 REPAIR SUMMARY:")
            print(f"   • Initial issues: {initial_count}")
            print(f"   • Remaining issues: {final_count}")
            print(f"   • Issues fixed: {fixed_count}")
            print(f"   • Success rate: {(fixed_count/initial_count)*100:.1f}%")

            return True

        except Exception as e:
            self.logger.error(f"❌ Error during repair: {e}")
            return False


if __name__ == "__main__":
    repair_system = DatabaseRepairSystem()
    repair_system.run_comprehensive_repair()
