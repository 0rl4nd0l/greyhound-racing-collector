#!/usr/bin/env python3
"""
CSV Audit and Integration Tool
=============================

This tool audits all CSV files in the system and ensures they serve a purpose
by either being in the database, enhanced data, or properly categorized.

Author: AI Assistant
Date: July 26, 2025
"""

import json
import os
import re
import shutil
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd


class CSVAuditIntegrator:
    def __init__(self):
        self.database_path = "greyhound_racing_data.db"
        self.enhanced_data_dir = "enhanced_expert_data"
        self.directories = {
            "unprocessed": "./unprocessed",
            "processed": "./processed",
            "historical": "./historical_races",
            "upcoming": "./upcoming_races",
            "enhanced_csv": "./enhanced_expert_data/csv",
            "enhanced_json": "./enhanced_expert_data/json",
        }

        # Initialize audit results
        self.audit_results = {
            "total_csvs": 0,
            "database_integrated": 0,
            "enhanced_processed": 0,
            "properly_categorized": 0,
            "orphaned_files": [],
            "duplicate_files": [],
            "malformed_files": [],
            "integration_candidates": [],
            "cleanup_candidates": [],
        }

    def run_comprehensive_audit(self):
        """Run comprehensive audit of all CSV files"""
        print("🔍 Starting comprehensive CSV audit...")

        # Step 1: Discover all CSV files
        all_csvs = self.discover_all_csvs()
        self.audit_results["total_csvs"] = len(all_csvs)
        print(f"📊 Found {len(all_csvs)} total CSV files")

        # Step 2: Check database integration
        db_integrated = self.check_database_integration(all_csvs)
        self.audit_results["database_integrated"] = len(db_integrated)
        print(f"💾 {len(db_integrated)} files integrated in database")

        # Step 3: Check enhanced data processing
        enhanced_processed = self.check_enhanced_processing(all_csvs)
        self.audit_results["enhanced_processed"] = len(enhanced_processed)
        print(f"⚡ {len(enhanced_processed)} files in enhanced data")

        # Step 4: Identify files serving no purpose
        self.identify_purposeless_files(all_csvs, db_integrated, enhanced_processed)

        # Step 5: Generate integration plan
        self.generate_integration_plan()

        # Step 6: Execute cleanup and integration
        self.execute_integration()

        return self.audit_results

    def discover_all_csvs(self):
        """Discover all CSV files in the system"""
        all_csvs = {}

        # Search in all directories
        for name, path in self.directories.items():
            if os.path.exists(path):
                csvs = self.find_csvs_in_directory(path, name)
                all_csvs.update(csvs)

        # Also search current directory and any other potential locations
        current_dir_csvs = self.find_csvs_in_directory(".", "root")
        all_csvs.update(current_dir_csvs)

        return all_csvs

    def find_csvs_in_directory(self, directory, category):
        """Find all CSV files in a directory"""
        csvs = {}

        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".csv") and file != "README.csv":
                    full_path = os.path.join(root, file)
                    try:
                        stat = os.stat(full_path)
                        csvs[full_path] = {
                            "filename": file,
                            "category": category,
                            "size": stat.st_size,
                            "modified": datetime.fromtimestamp(stat.st_mtime),
                            "directory": root,
                            "relative_path": os.path.relpath(full_path),
                        }
                    except (OSError, IOError):
                        continue

        return csvs

    def check_database_integration(self, all_csvs):
        """Check which files are integrated in the database"""
        integrated_files = set()

        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            # Get all race files from race_metadata
            cursor.execute(
                "SELECT DISTINCT filename FROM race_metadata WHERE filename IS NOT NULL"
            )
            db_files = cursor.fetchall()

            for (filename,) in db_files:
                # Find matching CSV files
                for path, info in all_csvs.items():
                    if info["filename"] == filename:
                        integrated_files.add(path)

            # Also check enhanced_expert_data table
            cursor.execute(
                "SELECT DISTINCT race_file FROM enhanced_expert_data WHERE race_file IS NOT NULL"
            )
            enhanced_files = cursor.fetchall()

            for (filename,) in enhanced_files:
                for path, info in all_csvs.items():
                    if info["filename"] == filename or filename in path:
                        integrated_files.add(path)

            conn.close()

        except Exception as e:
            print(f"⚠️ Error checking database integration: {e}")

        return integrated_files

    def check_enhanced_processing(self, all_csvs):
        """Check which files have been processed for enhanced data"""
        enhanced_files = set()

        # Check enhanced CSV directory
        enhanced_csv_dir = self.directories.get("enhanced_csv")
        if enhanced_csv_dir and os.path.exists(enhanced_csv_dir):
            enhanced_filenames = set(os.listdir(enhanced_csv_dir))

            for path, info in all_csvs.items():
                # Check if there's a corresponding enhanced file
                enhanced_name = f"enhanced_{info['filename']}"
                if (
                    info["filename"] in enhanced_filenames
                    or enhanced_name in enhanced_filenames
                ):
                    enhanced_files.add(path)

        # Check enhanced JSON directory
        enhanced_json_dir = self.directories.get("enhanced_json")
        if enhanced_json_dir and os.path.exists(enhanced_json_dir):
            json_files = [
                f.replace(".json", ".csv")
                for f in os.listdir(enhanced_json_dir)
                if f.endswith(".json")
            ]

            for path, info in all_csvs.items():
                if info["filename"] in json_files:
                    enhanced_files.add(path)

        return enhanced_files

    def identify_purposeless_files(self, all_csvs, db_integrated, enhanced_processed):
        """Identify files that serve no clear purpose"""

        serving_purpose = db_integrated.union(enhanced_processed)

        for path, info in all_csvs.items():
            if path not in serving_purpose:
                # Check if it's a malformed or duplicate file
                if self.is_malformed_file(path):
                    self.audit_results["malformed_files"].append(
                        {"path": path, "reason": "Malformed or corrupted file", **info}
                    )
                elif self.is_duplicate_file(path, all_csvs):
                    self.audit_results["duplicate_files"].append(
                        {"path": path, "reason": "Duplicate file detected", **info}
                    )
                else:
                    # This file could potentially be integrated
                    self.audit_results["integration_candidates"].append(
                        {
                            "path": path,
                            "reason": "Not integrated but potentially useful",
                            **info,
                        }
                    )

    def is_malformed_file(self, file_path):
        """Check if a CSV file is malformed"""
        try:
            # Try to read the file
            df = pd.read_csv(file_path, nrows=5)

            # Check basic structure
            if df.empty or len(df.columns) < 3:
                return True

            # Check for race-like structure
            expected_patterns = ["race", "dog", "time", "position", "margin", "odds"]
            column_str = " ".join(df.columns).lower()

            if not any(pattern in column_str for pattern in expected_patterns):
                return True

            return False

        except Exception:
            return True

    def is_duplicate_file(self, file_path, all_csvs):
        """Check if a file is a duplicate based on content similarity"""
        try:
            current_info = all_csvs[file_path]

            # Simple duplicate detection based on filename patterns
            filename = current_info["filename"]

            duplicates = 0
            for other_path, other_info in all_csvs.items():
                if other_path != file_path and other_info["filename"] == filename:
                    duplicates += 1

            return duplicates > 0

        except Exception:
            return False

    def generate_integration_plan(self):
        """Generate a plan for integrating orphaned files"""

        print(f"\n📋 Integration Plan:")
        print(
            f"  • Integration candidates: {len(self.audit_results['integration_candidates'])}"
        )
        print(
            f"  • Cleanup candidates: {len(self.audit_results['malformed_files']) + len(self.audit_results['duplicate_files'])}"
        )

        # Categorize integration candidates
        race_files = []
        form_guides = []
        other_files = []

        for candidate in self.audit_results["integration_candidates"]:
            filename = candidate["filename"].lower()

            if "race" in filename and any(
                month in filename
                for month in [
                    "january",
                    "february",
                    "march",
                    "april",
                    "may",
                    "june",
                    "july",
                    "august",
                    "september",
                    "october",
                    "november",
                    "december",
                ]
            ):
                race_files.append(candidate)
            elif "form" in filename or "guide" in filename:
                form_guides.append(candidate)
            else:
                other_files.append(candidate)

        print(f"    - Race files: {len(race_files)}")
        print(f"    - Form guides: {len(form_guides)}")
        print(f"    - Other files: {len(other_files)}")

        # Store categorized files
        self.audit_results["categorized_integration"] = {
            "race_files": race_files,
            "form_guides": form_guides,
            "other_files": other_files,
        }

    def execute_integration(self):
        """Execute the integration plan"""

        print(f"\n🚀 Executing integration...")

        # Create necessary directories
        os.makedirs("./organized_csvs/race_data", exist_ok=True)
        os.makedirs("./organized_csvs/form_guides", exist_ok=True)
        os.makedirs("./organized_csvs/archive", exist_ok=True)
        os.makedirs("./organized_csvs/cleanup_needed", exist_ok=True)

        integrated_count = 0
        cleaned_count = 0

        # Process integration candidates
        categorized = self.audit_results.get("categorized_integration", {})

        for race_file in categorized.get("race_files", []):
            try:
                # Move to organized race_data directory
                dest_path = f"./organized_csvs/race_data/{race_file['filename']}"
                if not os.path.exists(dest_path):
                    shutil.copy2(race_file["path"], dest_path)
                    integrated_count += 1
            except Exception as e:
                print(f"⚠️ Error processing {race_file['filename']}: {e}")

        for form_file in categorized.get("form_guides", []):
            try:
                # Move to organized form_guides directory
                dest_path = f"./organized_csvs/form_guides/{form_file['filename']}"
                if not os.path.exists(dest_path):
                    shutil.copy2(form_file["path"], dest_path)
                    integrated_count += 1
            except Exception as e:
                print(f"⚠️ Error processing {form_file['filename']}: {e}")

        # Process cleanup candidates
        for malformed_file in self.audit_results["malformed_files"]:
            try:
                dest_path = (
                    f"./organized_csvs/cleanup_needed/{malformed_file['filename']}"
                )
                if not os.path.exists(dest_path):
                    shutil.copy2(malformed_file["path"], dest_path)
                    cleaned_count += 1
            except Exception as e:
                print(f"⚠️ Error moving malformed file: {e}")

        print(f"✅ Integration complete:")
        print(f"    - Files organized: {integrated_count}")
        print(f"    - Files flagged for cleanup: {cleaned_count}")

        self.audit_results["integration_executed"] = {
            "organized_files": integrated_count,
            "cleanup_files": cleaned_count,
            "timestamp": datetime.now().isoformat(),
        }

    def generate_report(self):
        """Generate comprehensive audit report"""

        report_path = (
            f"csv_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        # Calculate utilization rates
        total_csvs = self.audit_results["total_csvs"]
        serving_purpose = (
            self.audit_results["database_integrated"]
            + self.audit_results["enhanced_processed"]
        )
        utilization_rate = (serving_purpose / total_csvs * 100) if total_csvs > 0 else 0

        # Enhanced report
        report = {
            "audit_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_csv_files": total_csvs,
                "files_serving_purpose": serving_purpose,
                "utilization_rate_percent": round(utilization_rate, 2),
                "database_integrated": self.audit_results["database_integrated"],
                "enhanced_processed": self.audit_results["enhanced_processed"],
                "integration_candidates": len(
                    self.audit_results["integration_candidates"]
                ),
                "cleanup_needed": len(self.audit_results["malformed_files"])
                + len(self.audit_results["duplicate_files"]),
            },
            "details": self.audit_results,
            "recommendations": self.generate_recommendations(),
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n📊 Audit Report Generated: {report_path}")
        return report_path

    def generate_recommendations(self):
        """Generate recommendations based on audit results"""

        recommendations = []

        total_csvs = self.audit_results["total_csvs"]
        serving_purpose = (
            self.audit_results["database_integrated"]
            + self.audit_results["enhanced_processed"]
        )
        utilization_rate = (serving_purpose / total_csvs * 100) if total_csvs > 0 else 0

        if utilization_rate < 50:
            recommendations.append(
                {
                    "priority": "HIGH",
                    "action": "Integrate orphaned CSV files",
                    "description": f"Only {utilization_rate:.1f}% of CSV files are serving a purpose. Consider processing integration candidates.",
                    "file_count": len(self.audit_results["integration_candidates"]),
                }
            )

        if len(self.audit_results["malformed_files"]) > 0:
            recommendations.append(
                {
                    "priority": "MEDIUM",
                    "action": "Clean up malformed files",
                    "description": "Remove or fix malformed CSV files that cannot be processed.",
                    "file_count": len(self.audit_results["malformed_files"]),
                }
            )

        if len(self.audit_results["duplicate_files"]) > 0:
            recommendations.append(
                {
                    "priority": "LOW",
                    "action": "Remove duplicate files",
                    "description": "Clean up duplicate CSV files to save storage space.",
                    "file_count": len(self.audit_results["duplicate_files"]),
                }
            )

        return recommendations


def main():
    """Main execution function"""
    print("🎯 CSV Audit and Integration Tool")
    print("=" * 50)

    auditor = CSVAuditIntegrator()

    # Run comprehensive audit
    results = auditor.run_comprehensive_audit()

    # Generate detailed report
    report_path = auditor.generate_report()

    # Print summary
    print(f"\n📈 AUDIT SUMMARY:")
    print(f"  Total CSV Files: {results['total_csvs']:,}")
    print(f"  Database Integrated: {results['database_integrated']:,}")
    print(f"  Enhanced Processed: {results['enhanced_processed']:,}")
    print(f"  Integration Candidates: {len(results['integration_candidates']):,}")
    print(
        f"  Cleanup Needed: {len(results['malformed_files']) + len(results['duplicate_files']):,}"
    )

    utilization = (
        (
            (results["database_integrated"] + results["enhanced_processed"])
            / results["total_csvs"]
            * 100
        )
        if results["total_csvs"] > 0
        else 0
    )
    print(f"  Utilization Rate: {utilization:.1f}%")

    print(f"\n📋 Report saved to: {report_path}")
    print("✅ CSV audit and integration complete!")


if __name__ == "__main__":
    main()
