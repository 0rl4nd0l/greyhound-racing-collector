#!/usr/bin/env python3
"""
Comprehensive CSV Enhanced Data Processor
=========================================

This script processes existing CSV files to extract enhanced expert form data
without requiring web scraping. It analyzes the CSV files directly and extracts
sectional times, PIR ratings, margins, and other enhanced metrics.

Author: AI Assistant  
Date: July 26, 2025
"""

import json
import os
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class ComprehensiveCSVEnhancedProcessor:
    """Process existing CSV files to extract enhanced data"""

    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.enhanced_data_dir = Path("./enhanced_expert_data")
        self.enhanced_csv_dir = self.enhanced_data_dir / "csv"
        self.enhanced_json_dir = self.enhanced_data_dir / "json"

        # Create directories
        self.enhanced_data_dir.mkdir(exist_ok=True)
        self.enhanced_csv_dir.mkdir(exist_ok=True)
        self.enhanced_json_dir.mkdir(exist_ok=True)

        # CSV directories to process
        self.csv_directories = [
            "./form_guides/downloaded",
            "./unprocessed",
            "./processed",
            "./historical_races",
        ]

        print("🚀 Comprehensive CSV Enhanced Processor Initialized")

    def find_all_csv_files(self):
        """Find all CSV files across directories"""
        all_csv_files = []

        for directory in self.csv_directories:
            if os.path.exists(directory):
                csv_files = list(Path(directory).glob("*.csv"))
                for csv_file in csv_files:
                    all_csv_files.append(
                        {
                            "file_path": str(csv_file),
                            "directory": directory,
                            "filename": csv_file.name,
                        }
                    )

        print(f"📁 Found {len(all_csv_files)} CSV files total")
        return all_csv_files

    def already_processed(self, csv_filename):
        """Check if a CSV file has already been processed"""
        # Check if enhanced version exists
        base_name = csv_filename.replace(".csv", "")

        # Look for existing enhanced files
        existing_enhanced = list(self.enhanced_csv_dir.glob(f"*{base_name}*.csv"))
        existing_json = list(self.enhanced_json_dir.glob(f"*{base_name}*.json"))

        return len(existing_enhanced) > 0 or len(existing_json) > 0

    def extract_race_info_from_filename(self, filename):
        """Extract race information from CSV filename"""
        try:
            # Example: "Race 1 - AP_K - 24 July 2025.csv"
            base_name = filename.replace(".csv", "")

            if " - " in base_name:
                parts = base_name.split(" - ")
                if len(parts) >= 3:
                    race_part = parts[0].strip()
                    venue = parts[1].strip()
                    date_part = parts[2].strip()

                    # Extract race number
                    race_number = "1"
                    if race_part.lower().startswith("race "):
                        race_number = race_part[5:].strip()

                    return {
                        "race_number": race_number,
                        "venue": venue,
                        "date_str": date_part,
                        "race_id": f"{venue}_{race_number}_{date_part.replace(' ', '_')}",
                    }

            # Fallback parsing
            return {
                "race_number": "1",
                "venue": "Unknown",
                "date_str": "Unknown Date",
                "race_id": base_name.replace(" ", "_"),
            }

        except Exception as e:
            print(f"⚠️ Error parsing filename {filename}: {e}")
            return {
                "race_number": "1",
                "venue": "Unknown",
                "date_str": "Unknown Date",
                "race_id": filename.replace(".csv", "").replace(" ", "_"),
            }

    def process_csv_file(self, csv_file_info):
        """Process a single CSV file to extract enhanced data"""
        try:
            file_path = csv_file_info["file_path"]
            filename = csv_file_info["filename"]

            print(f"🔍 Processing: {filename}")

            # Read the CSV file
            df = pd.read_csv(file_path)

            if len(df) == 0:
                print(f"   ⚠️ Empty file: {filename}")
                return False

            # Check if this looks like a form guide CSV
            required_columns = ["Dog Name"]
            if not all(col in df.columns for col in required_columns):
                print(f"   ⚠️ Not a form guide CSV: {filename}")
                return False

            # Extract race information
            race_info = self.extract_race_info_from_filename(filename)

            # Process the CSV data to extract enhanced metrics
            enhanced_data = self.extract_enhanced_metrics_from_csv(df, race_info)

            if enhanced_data:
                # Save enhanced CSV
                enhanced_csv_filename = f"{race_info['venue']}_Race{race_info['race_number']}_{race_info['date_str'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_enhanced.csv"
                enhanced_csv_path = self.enhanced_csv_dir / enhanced_csv_filename

                df.to_csv(enhanced_csv_path, index=False)

                # Save comprehensive JSON
                comprehensive_data = {
                    "race_info": race_info,
                    "extraction_timestamp": datetime.now().isoformat(),
                    "csv_data": {
                        "headers": list(df.columns),
                        "row_count": len(df),
                        "column_count": len(df.columns),
                        "sample_data": df.head(3).values.tolist(),
                    },
                    "enhanced_metrics": enhanced_data,
                    "source_file": file_path,
                }

                json_filename = f"{race_info['venue']}_Race{race_info['race_number']}_{race_info['date_str'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_comprehensive.json"
                json_path = self.enhanced_json_dir / json_filename

                with open(json_path, "w") as f:
                    json.dump(comprehensive_data, f, indent=2, default=str)

                print(f"   ✅ Processed: {filename}")
                return True
            else:
                print(f"   ⚠️ No enhanced data extracted: {filename}")
                return False

        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")
            return False

    def extract_enhanced_metrics_from_csv(self, df, race_info):
        """Extract enhanced metrics from CSV data"""
        try:
            enhanced_data = {"dogs": [], "race_metrics": {}, "extraction_success": True}

            # Process each dog in the CSV
            current_dog = None

            for idx, row in df.iterrows():
                dog_name_raw = str(row.get("Dog Name", "")).strip()

                # Check if this is a new dog or continuation of previous
                if dog_name_raw and dog_name_raw not in ['""', "", "nan", "NaN"]:
                    # New dog - clean the name
                    if ". " in dog_name_raw:
                        current_dog = dog_name_raw.split(". ", 1)[1]
                        box_number = dog_name_raw.split(".")[0]
                    else:
                        current_dog = dog_name_raw
                        box_number = None

                    # Initialize dog data
                    dog_data = {
                        "dog_name": current_dog,
                        "box_number": box_number,
                        "historical_races": [],
                        "enhanced_metrics": {},
                    }
                    enhanced_data["dogs"].append(dog_data)

                # Process this row as historical race data for current dog
                if current_dog:
                    race_data = self.extract_race_data_from_row(row)
                    if race_data:
                        # Add to current dog's historical races
                        if enhanced_data["dogs"]:
                            enhanced_data["dogs"][-1]["historical_races"].append(
                                race_data
                            )

            # Calculate enhanced metrics for each dog
            for dog_data in enhanced_data["dogs"]:
                dog_data["enhanced_metrics"] = self.calculate_enhanced_metrics_for_dog(
                    dog_data["historical_races"]
                )

            return enhanced_data

        except Exception as e:
            print(f"   ❌ Error extracting enhanced metrics: {e}")
            return None

    def extract_race_data_from_row(self, row):
        """Extract race data from a CSV row"""
        try:
            race_data = {}

            # Common CSV columns and their mappings
            column_mappings = {
                "PLC": "position",
                "BOX": "box",
                "WGT": "weight",
                "DIST": "distance",
                "DATE": "date",
                "TRACK": "track",
                "G": "grade",
                "TIME": "time",
                "WIN": "win_time",
                "BON": "bonus_time",
                "1 SEC": "first_sectional",
                "MGN": "margin",
                "PIR": "pir_rating",
                "SP": "starting_price",
                "Sex": "sex",
            }

            # Extract available data
            for csv_col, data_key in column_mappings.items():
                if csv_col in row.index:
                    value = row[csv_col]
                    if pd.notna(value) and str(value).strip() not in ["", "nan", "NaN"]:
                        race_data[data_key] = self._safe_convert(value, data_key)

            # Only return data if we have meaningful information
            if len(race_data) >= 3:  # At least 3 data points
                return race_data
            else:
                return None

        except Exception as e:
            return None

    def _safe_convert(self, value, data_type):
        """Safely convert values to appropriate types"""
        try:
            if pd.isna(value) or str(value).strip() in ["", "nan", "NaN"]:
                return None

            # Numeric conversions
            numeric_fields = [
                "position",
                "box",
                "weight",
                "distance",
                "time",
                "win_time",
                "bonus_time",
                "first_sectional",
                "margin",
                "pir_rating",
                "starting_price",
            ]

            if data_type in numeric_fields:
                return float(str(value).replace(",", ""))
            else:
                return str(value).strip()

        except (ValueError, TypeError):
            return str(value).strip() if value else None

    def calculate_enhanced_metrics_for_dog(self, historical_races):
        """Calculate enhanced metrics from historical race data"""
        if not historical_races:
            return {}

        try:
            metrics = {}

            # Extract numeric data
            positions = [
                r["position"]
                for r in historical_races
                if r.get("position") and r["position"] > 0
            ]
            sectionals = [
                r["first_sectional"]
                for r in historical_races
                if r.get("first_sectional") and r["first_sectional"] > 0
            ]
            weights = [
                r["weight"]
                for r in historical_races
                if r.get("weight") and r["weight"] > 0
            ]
            margins = [
                r["margin"] for r in historical_races if r.get("margin") is not None
            ]
            pir_ratings = [
                r["pir_rating"]
                for r in historical_races
                if r.get("pir_rating") and r["pir_rating"] > 0
            ]
            times = [
                r["time"] for r in historical_races if r.get("time") and r["time"] > 0
            ]

            # Calculate metrics
            if positions:
                metrics["avg_position"] = np.mean(positions)
                metrics["best_position"] = min(positions)
                metrics["position_consistency"] = 1 / (np.std(positions) + 0.1)
                metrics["win_rate"] = sum(1 for p in positions if p == 1) / len(
                    positions
                )
                metrics["place_rate"] = sum(1 for p in positions if p <= 3) / len(
                    positions
                )

            if sectionals:
                metrics["avg_first_sectional"] = np.mean(sectionals)
                metrics["best_sectional"] = min(sectionals)
                metrics["sectional_consistency"] = 1 / (np.std(sectionals) + 0.1)

            if weights:
                metrics["avg_weight"] = np.mean(weights)
                metrics["weight_consistency"] = 1 / (np.std(weights) + 0.1)
                if len(weights) > 1:
                    metrics["weight_trend"] = np.polyfit(
                        range(len(weights)), weights, 1
                    )[0]

            if margins:
                winning_margins = [m for m in margins if m > 0]
                losing_margins = [abs(m) for m in margins if m < 0]
                if winning_margins:
                    metrics["avg_winning_margin"] = np.mean(winning_margins)
                if losing_margins:
                    metrics["avg_losing_margin"] = np.mean(losing_margins)

            if pir_ratings:
                metrics["avg_pir_rating"] = np.mean(pir_ratings)
                metrics["best_pir_rating"] = max(pir_ratings)
                if len(pir_ratings) > 1:
                    metrics["pir_trend"] = np.polyfit(
                        range(len(pir_ratings)), pir_ratings, 1
                    )[0]

            if times:
                metrics["avg_time"] = np.mean(times)
                metrics["best_time"] = min(times)
                metrics["time_consistency"] = 1 / (np.std(times) + 0.1)

            # Overall metrics
            metrics["total_races"] = len(historical_races)
            metrics["data_quality"] = len([r for r in historical_races if len(r) >= 5])
            metrics["enhanced_data_available"] = (
                len(sectionals) > 0 or len(pir_ratings) > 0
            )

            return metrics

        except Exception as e:
            print(f"   ⚠️ Error calculating metrics: {e}")
            return {}

    def sync_to_database(self):
        """Sync enhanced data to the main database"""
        try:
            from enhanced_data_integration import EnhancedDataIntegrator

            integrator = EnhancedDataIntegrator(self.db_path)
            return integrator.sync_enhanced_data_to_database()
        except Exception as e:
            print(f"❌ Error syncing to database: {e}")
            return False

    def process_batch(self, max_files=100, skip_existing=True):
        """Process a batch of CSV files"""
        print(f"🚀 PROCESSING BATCH OF CSV FILES")
        print(f"🎯 Target: {max_files} files")
        print(f"⏭️ Skip existing: {skip_existing}")
        print("=" * 60)

        # Find all CSV files
        all_csv_files = self.find_all_csv_files()

        # Filter out already processed files if requested
        files_to_process = []
        skipped_count = 0

        for csv_file in all_csv_files:
            if skip_existing and self.already_processed(csv_file["filename"]):
                skipped_count += 1
                continue
            files_to_process.append(csv_file)

            if len(files_to_process) >= max_files:
                break

        print(f"📊 Files to process: {len(files_to_process)}")
        print(f"⏭️ Files skipped (already processed): {skipped_count}")

        # Process files
        successful = 0
        failed = 0

        for i, csv_file in enumerate(files_to_process, 1):
            print(f"\n--- PROCESSING {i}/{len(files_to_process)} ---")

            if self.process_csv_file(csv_file):
                successful += 1
            else:
                failed += 1

        # Sync to database
        print(f"\n🔄 Syncing enhanced data to database...")
        db_sync_success = self.sync_to_database()

        # Summary
        print(f"\n📊 BATCH PROCESSING COMPLETE")
        print("=" * 50)
        print(f"✅ Successfully processed: {successful}")
        print(f"❌ Failed to process: {failed}")
        print(f"📈 Success rate: {(successful/(successful+failed)*100):.1f}%")
        print(f"💾 Database sync: {'✅ Success' if db_sync_success else '❌ Failed'}")

        return {
            "successful": successful,
            "failed": failed,
            "total_processed": successful + failed,
            "success_rate": (
                successful / (successful + failed) * 100
                if (successful + failed) > 0
                else 0
            ),
            "database_synced": db_sync_success,
        }


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive CSV Enhanced Data Processor"
    )
    parser.add_argument(
        "--max-files", type=int, default=100, help="Maximum files to process"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip already processed files",
    )
    parser.add_argument(
        "--force-reprocess", action="store_true", help="Reprocess all files"
    )

    args = parser.parse_args()

    processor = ComprehensiveCSVEnhancedProcessor()

    skip_existing = args.skip_existing and not args.force_reprocess

    result = processor.process_batch(
        max_files=args.max_files, skip_existing=skip_existing
    )

    print(f"\n🎉 Processing completed with {result['success_rate']:.1f}% success rate")


if __name__ == "__main__":
    main()
