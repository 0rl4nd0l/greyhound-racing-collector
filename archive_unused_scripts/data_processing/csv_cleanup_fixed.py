#!/usr/bin/env python3
"""
CSV Cleanup and UI Integration Tool
==================================

This tool cleans up the CSV mess and ensures proper UI integration
by fixing database issues and removing unnecessary files.

Author: AI Assistant  
Date: July 26, 2025
"""

import hashlib
import json
import os
import re
import shutil
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd


class CSVCleanupIntegrator:
    def __init__(self):
        self.database_path = "greyhound_racing_data.db"
        self.enhanced_csv_dir = "./enhanced_expert_data/csv"
        self.enhanced_json_dir = "./enhanced_expert_data/json"

        # Statistics
        self.stats = {
            "files_cleaned": 0,
            "files_integrated": 0,
            "duplicates_removed": 0,
            "database_records_added": 0,
            "space_saved_mb": 0,
        }

    def run_comprehensive_cleanup(self):
        """Run comprehensive cleanup and integration"""
        print("🧹 Starting comprehensive CSV cleanup and UI integration...")

        # Step 1: Fix database schema issues
        self.fix_database_schema()

        # Step 2: Remove duplicate and malformed files
        self.cleanup_duplicate_files()

        # Step 3: Integrate enhanced data into UI database
        self.integrate_enhanced_data_to_ui()

        # Step 4: Organize remaining useful files
        self.organize_useful_files()

        # Step 5: Generate final report
        self.generate_final_report()

        return self.stats

    def fix_database_schema(self):
        """Fix database schema issues for proper UI integration"""
        print("🔧 Fixing database schema...")

        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            # Check current race_metadata structure
            cursor.execute("PRAGMA table_info(race_metadata)")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]

            # Add missing columns if needed
            if "filename" not in column_names:
                print("  + Adding filename column to race_metadata")
                cursor.execute("ALTER TABLE race_metadata ADD COLUMN filename TEXT")

            if "file_path" not in column_names:
                print("  + Adding file_path column to race_metadata")
                cursor.execute("ALTER TABLE race_metadata ADD COLUMN file_path TEXT")

            if "processing_status" not in column_names:
                print("  + Adding processing_status column to race_metadata")
                cursor.execute(
                    "ALTER TABLE race_metadata ADD COLUMN processing_status TEXT DEFAULT 'processed'"
                )

            conn.commit()
            conn.close()

            print("✅ Database schema fixed")

        except Exception as e:
            print(f"⚠️ Error fixing database schema: {e}")

    def cleanup_duplicate_files(self):
        """Remove duplicate and malformed CSV files"""
        print("🗑️ Cleaning up duplicate and malformed files...")

        # Read the audit report for guidance
        audit_files = []
        try:
            # Find the most recent audit report file
            audit_reports = [
                f
                for f in os.listdir(".")
                if f.startswith("csv_audit_report_") and f.endswith(".json")
            ]
            if audit_reports:
                latest_report = sorted(audit_reports)[-1]
                with open(latest_report, "r") as f:
                    audit_data = json.load(f)
                    audit_files = (
                        audit_data["details"]["duplicate_files"]
                        + audit_data["details"]["malformed_files"]
                    )
                print(f"  Using audit report: {latest_report}")
            else:
                print("  No audit report found, scanning for duplicates manually...")
        except Exception as e:
            print(f"  Warning: Could not read audit report: {e}")

        files_removed = 0
        space_saved = 0

        # Create cleanup directory
        os.makedirs("./cleanup_archive", exist_ok=True)

        # Process audit-identified files
        for file_info in audit_files:
            file_path = file_info["path"]
            if os.path.exists(file_path):
                try:
                    # Get file size before deletion
                    file_size = os.path.getsize(file_path)

                    # Move to cleanup archive
                    filename = os.path.basename(file_path)
                    archive_path = f"./cleanup_archive/{filename}"

                    # Avoid conflicts in archive
                    counter = 1
                    base_name, ext = os.path.splitext(filename)
                    while os.path.exists(archive_path):
                        archive_path = f"./cleanup_archive/{base_name}_{counter}{ext}"
                        counter += 1

                    shutil.move(file_path, archive_path)
                    files_removed += 1
                    space_saved += file_size

                except Exception as e:
                    print(f"    ⚠️ Error moving {file_path}: {e}")

        # Additional cleanup - remove empty directories
        self.remove_empty_directories(
            ["./unprocessed", "./processed", "./historical_races"]
        )

        self.stats["files_cleaned"] = files_removed
        self.stats["space_saved_mb"] = round(space_saved / (1024 * 1024), 2)

        print(
            f"✅ Cleaned up {files_removed} files, saved {self.stats['space_saved_mb']} MB"
        )

    def remove_empty_directories(self, directories):
        """Remove empty directories and subdirectories"""
        for directory in directories:
            if os.path.exists(directory):
                for root, dirs, files in os.walk(directory, topdown=False):
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        try:
                            if not os.listdir(dir_path):  # Empty directory
                                os.rmdir(dir_path)
                        except OSError:
                            pass  # Directory not empty or other issue

    def integrate_enhanced_data_to_ui(self):
        """Integrate enhanced data into the UI database"""
        print("🔗 Integrating enhanced data into UI database...")

        if not os.path.exists(self.enhanced_json_dir):
            print("  No enhanced JSON data found")
            return

        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            json_files = [
                f for f in os.listdir(self.enhanced_json_dir) if f.endswith(".json")
            ]
            integrated_count = 0

            for json_file in json_files[:100]:  # Process first 100 files for UI
                try:
                    json_path = os.path.join(self.enhanced_json_dir, json_file)

                    with open(json_path, "r") as f:
                        data = json.load(f)

                    # Extract race metadata for UI
                    race_info = self.extract_race_metadata_from_enhanced(
                        data, json_file
                    )

                    if race_info:
                        # Insert or update race_metadata
                        cursor.execute(
                            """
                            INSERT OR REPLACE INTO race_metadata 
                            (race_id, venue, race_number, race_date, race_name, grade, distance, 
                             field_size, track_condition, filename, file_path, processing_status, extraction_timestamp)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            race_info,
                        )

                        integrated_count += 1

                except Exception as e:
                    print(f"    ⚠️ Error processing {json_file}: {e}")
                    continue

            conn.commit()
            conn.close()

            self.stats["database_records_added"] = integrated_count
            print(f"✅ Integrated {integrated_count} enhanced records into UI database")

        except Exception as e:
            print(f"⚠️ Error integrating enhanced data: {e}")

    def extract_race_metadata_from_enhanced(self, data, json_file):
        """Extract race metadata from enhanced JSON data"""
        try:
            # Try to extract relevant information from enhanced data structure
            race_id = json_file.replace(".json", "")

            # Look for race information in various possible structures
            race_date = None
            venue = None
            race_number = None
            race_name = race_id
            grade = None
            distance = None
            field_size = 0
            track_condition = "Good"

            # Parse filename for basic info
            if "Race" in race_id and " - " in race_id:
                parts = race_id.split(" - ")
                if len(parts) >= 3:
                    race_number = parts[0].replace("Race ", "")
                    venue = parts[1]
                    race_date = parts[2]

            # Try to get more detailed info from JSON structure
            if isinstance(data, dict):
                # Check various possible keys
                for key in ["race_info", "metadata", "race_data", "basic_info"]:
                    if key in data and isinstance(data[key], dict):
                        info = data[key]
                        race_date = info.get("race_date", race_date)
                        venue = info.get("venue", venue)
                        race_number = info.get("race_number", race_number)
                        grade = info.get("grade", grade)
                        distance = info.get("distance", distance)
                        track_condition = info.get("track_condition", track_condition)
                        break

                # Count field size from dogs
                for key in ["dogs", "entries", "participants"]:
                    if key in data and isinstance(data[key], list):
                        field_size = len(data[key])
                        break

            filename = f"{race_id}.csv"
            file_path = f"enhanced_expert_data/json/{json_file}"

            return (
                race_id,
                venue,
                race_number,
                race_date,
                race_name,
                grade,
                distance,
                field_size,
                track_condition,
                filename,
                file_path,
                "enhanced",
                datetime.now().isoformat(),
            )

        except Exception as e:
            print(f"    ⚠️ Error extracting metadata from {json_file}: {e}")
            return None

    def organize_useful_files(self):
        """Organize remaining useful CSV files into proper structure"""
        print("📁 Organizing remaining useful files...")

        # Create organized structure
        directories = {
            "active_races": "./data/active_races",
            "historical_races": "./data/historical_races",
            "form_guides": "./data/form_guides",
            "enhanced_data": "./data/enhanced_data",
        }

        for dir_path in directories.values():
            os.makedirs(dir_path, exist_ok=True)

        organized_count = 0

        # Move enhanced CSV files to organized location
        if os.path.exists(self.enhanced_csv_dir):
            enhanced_files = os.listdir(self.enhanced_csv_dir)
            for file_name in enhanced_files[:50]:  # Organize first 50 enhanced files
                try:
                    source = os.path.join(self.enhanced_csv_dir, file_name)
                    dest = os.path.join(directories["enhanced_data"], file_name)

                    if not os.path.exists(dest):
                        shutil.copy2(source, dest)
                        organized_count += 1

                except Exception as e:
                    print(f"    ⚠️ Error organizing {file_name}: {e}")

        # Create symbolic links for easy access
        try:
            if os.path.exists("./data") and not os.path.exists("./ui_data"):
                os.symlink("./data", "./ui_data")
        except:
            pass  # Symlink may already exist

        self.stats["files_integrated"] = organized_count
        print(f"✅ Organized {organized_count} useful files")

    def generate_final_report(self):
        """Generate final cleanup and integration report"""
        print("📊 Generating final report...")

        # Check final database state
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM race_metadata")
            total_races = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM enhanced_expert_data")
            enhanced_records = cursor.fetchone()[0]

            conn.close()

        except Exception as e:
            print(f"    ⚠️ Error checking database: {e}")
            total_races = 0
            enhanced_records = 0

        # Count remaining files
        remaining_csvs = 0
        for root, dirs, files in os.walk("."):
            remaining_csvs += len([f for f in files if f.endswith(".csv")])

        # Generate comprehensive report
        report = {
            "cleanup_timestamp": datetime.now().isoformat(),
            "cleanup_summary": {
                "files_cleaned": self.stats["files_cleaned"],
                "files_integrated": self.stats["files_integrated"],
                "database_records_added": self.stats["database_records_added"],
                "space_saved_mb": self.stats["space_saved_mb"],
                "remaining_csv_files": remaining_csvs,
            },
            "database_status": {
                "race_metadata_records": total_races,
                "enhanced_expert_records": enhanced_records,
                "ui_integration_status": (
                    "active" if total_races > 0 else "needs_attention"
                ),
            },
            "organized_structure": {
                "data_directory": "./data",
                "enhanced_data": "./data/enhanced_data",
                "ui_link": "./ui_data",
                "cleanup_archive": "./cleanup_archive",
            },
            "next_steps": [
                "Verify Flask app displays race data correctly",
                "Run enhanced data processor on remaining unprocessed files",
                "Set up automated cleanup schedule",
                "Monitor disk usage and clean up archives periodically",
            ],
        }

        report_path = f"cleanup_integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"✅ Final report saved to: {report_path}")

        # Print summary
        print(f"\n🎯 CLEANUP & INTEGRATION SUMMARY:")
        print(f"  Files Cleaned: {self.stats['files_cleaned']:,}")
        print(f"  Files Integrated: {self.stats['files_integrated']:,}")
        print(f"  Database Records Added: {self.stats['database_records_added']:,}")
        print(f"  Space Saved: {self.stats['space_saved_mb']} MB")
        print(f"  Remaining CSV Files: {remaining_csvs:,}")
        print(
            f"  UI Integration: {'✅ Active' if total_races > 0 else '⚠️ Needs Attention'}"
        )

        return report_path


def main():
    """Main execution function"""
    print("🎯 CSV Cleanup and UI Integration Tool")
    print("=" * 50)

    cleaner = CSVCleanupIntegrator()

    # Run comprehensive cleanup
    stats = cleaner.run_comprehensive_cleanup()

    print("✅ CSV cleanup and UI integration complete!")
    print("\n🚀 Next steps:")
    print("  1. Start Flask app to verify UI integration")
    print("  2. Check that race data displays properly")
    print("  3. Run enhanced processor on any remaining files")


if __name__ == "__main__":
    main()
