#!/usr/bin/env python3
"""
Enhanced Data Integration for Prediction Pipeline
===============================================

This module ensures that all enhanced expert form data from the enriched dataset
is properly integrated into the prediction and analysis pipeline.

Features:
- Integration of enhanced CSV data with sectional times, PIR ratings, margins
- JSON comprehensive data utilization
- Database synchronization with enhanced data
- Feature engineering from enhanced metrics
- Real-time enhanced data availability checking

Author: AI Assistant
Date: July 26, 2025
"""

import json
import os
import sqlite3
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class EnhancedDataIntegrator:
    """Integrates enhanced expert form data into the prediction pipeline"""

    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.enhanced_data_dir = Path("./enhanced_expert_data")
        self.enhanced_csv_dir = self.enhanced_data_dir / "csv"
        self.enhanced_json_dir = self.enhanced_data_dir / "json"

        # Validate enhanced data availability
        self.enhanced_data_available = self._check_enhanced_data_availability()

        print(f"🔗 Enhanced Data Integrator Initialized")
        print(f"📊 Enhanced data available: {self.enhanced_data_available}")
        if self.enhanced_data_available:
            print(
                f"📁 Enhanced CSV files: {len(list(self.enhanced_csv_dir.glob('*.csv')))}"
            )
            print(
                f"📄 Enhanced JSON files: {len(list(self.enhanced_json_dir.glob('*.json')))}"
            )

    def _check_enhanced_data_availability(self):
        """Check if enhanced expert form data is available"""
        try:
            return (
                self.enhanced_data_dir.exists()
                and self.enhanced_csv_dir.exists()
                and self.enhanced_json_dir.exists()
                and len(list(self.enhanced_csv_dir.glob("*.csv"))) > 0
            )
        except Exception:
            return False

    def get_enhanced_dog_data(self, dog_name, venue=None, max_races=10):
        """Get enhanced data for a specific dog from expert form sources"""
        if not self.enhanced_data_available:
            return {}

        enhanced_data = {
            "sectional_times": [],
            "pir_ratings": [],
            "weight_history": [],
            "margins": [],
            "performance_indicators": {},
            "enhanced_features": {},
        }

        try:
            # Search through enhanced CSV files for the dog
            csv_files = list(self.enhanced_csv_dir.glob("*.csv"))
            races_found = 0

            for csv_file in csv_files:
                if races_found >= max_races:
                    break

                try:
                    # Filter by venue if specified
                    if venue and venue not in str(csv_file):
                        continue

                    df = pd.read_csv(csv_file)

                    # Find rows for this dog (handle multi-row format)
                    current_dog = None
                    for idx, row in df.iterrows():
                        dog_name_col = str(row.get("Dog Name", "")).strip()

                        # Check if this is a new dog row or continuation
                        if dog_name_col and dog_name_col != '""' and dog_name_col != "":
                            # Clean dog name (remove box number prefix)
                            if ". " in dog_name_col:
                                current_dog = dog_name_col.split(". ", 1)[1]
                            else:
                                current_dog = dog_name_col

                        # Check if this row matches our target dog
                        if current_dog and self._normalize_dog_name(
                            current_dog
                        ) == self._normalize_dog_name(dog_name):
                            race_data = self._extract_enhanced_race_data(row)
                            if race_data:
                                # Add sectional times
                                if race_data.get("sectional_1st"):
                                    enhanced_data["sectional_times"].append(
                                        {
                                            "first_section": race_data["sectional_1st"],
                                            "race_time": race_data.get("time"),
                                            "date": race_data.get("date"),
                                            "distance": race_data.get("distance"),
                                        }
                                    )

                                # Add PIR ratings
                                if race_data.get("pir"):
                                    enhanced_data["pir_ratings"].append(
                                        {
                                            "pir": race_data["pir"],
                                            "date": race_data.get("date"),
                                            "position": race_data.get("position"),
                                        }
                                    )

                                # Add weight history
                                if race_data.get("weight"):
                                    enhanced_data["weight_history"].append(
                                        {
                                            "weight": race_data["weight"],
                                            "date": race_data.get("date"),
                                            "performance": race_data.get("position"),
                                        }
                                    )

                                # Add margins
                                if race_data.get("margin"):
                                    enhanced_data["margins"].append(
                                        {
                                            "margin": race_data["margin"],
                                            "date": race_data.get("date"),
                                            "position": race_data.get("position"),
                                        }
                                    )

                                races_found += 1
                                if races_found >= max_races:
                                    break

                except Exception as e:
                    print(f"⚠️ Error processing enhanced CSV {csv_file}: {e}")
                    continue

            # Calculate enhanced features from collected data
            enhanced_data["enhanced_features"] = self._calculate_enhanced_features(
                enhanced_data
            )

            return enhanced_data

        except Exception as e:
            print(f"❌ Error getting enhanced dog data: {e}")
            return {}

    def _normalize_dog_name(self, name):
        """Normalize dog name for comparison"""
        if not name:
            return ""
        return str(name).strip().upper().replace("'", "").replace(".", "")

    def _extract_enhanced_race_data(self, row):
        """Extract enhanced race data from a CSV row"""
        try:
            return {
                "position": self._safe_convert(row.get("PLC"), int),
                "box": self._safe_convert(row.get("BOX"), int),
                "weight": self._safe_convert(row.get("WGT"), float),
                "distance": self._safe_convert(row.get("DIST"), int),
                "date": str(row.get("DATE", "")),
                "track": str(row.get("TRACK", "")),
                "grade": str(row.get("G", "")),
                "time": self._safe_convert(row.get("TIME"), float),
                "win_time": self._safe_convert(row.get("WIN"), float),
                "bonus_time": self._safe_convert(row.get("BON"), float),
                "sectional_1st": self._safe_convert(row.get("1 SEC"), float),
                "margin": self._safe_convert(row.get("MGN"), float),
                "winner_info": str(row.get("W/2G", "")),
                "pir": self._safe_convert(row.get("PIR"), int),
                "starting_price": self._safe_convert(row.get("SP"), float),
            }
        except Exception:
            return {}

    def _safe_convert(self, value, convert_type):
        """Safely convert a value to the specified type"""
        try:
            if pd.isna(value) or str(value).strip() == "" or str(value) == "nan":
                return None
            return convert_type(value)
        except (ValueError, TypeError):
            return None

    def _calculate_enhanced_features(self, enhanced_data):
        """Calculate advanced features from enhanced data"""
        features = {}

        try:
            # Sectional time analysis
            sectional_times = enhanced_data.get("sectional_times", [])
            if sectional_times:
                first_sections = [
                    st["first_section"] for st in sectional_times if st["first_section"]
                ]
                if first_sections:
                    features["avg_first_section"] = np.mean(first_sections)
                    features["sectional_consistency"] = 1 / (np.std(first_sections) + 1)
                    features["best_sectional"] = min(first_sections)

            # PIR rating analysis
            pir_ratings = enhanced_data.get("pir_ratings", [])
            if pir_ratings:
                pir_values = [pr["pir"] for pr in pir_ratings if pr["pir"]]
                if pir_values:
                    features["avg_pir_rating"] = np.mean(pir_values)
                    features["best_pir_rating"] = max(pir_values)
                    features["pir_trend"] = self._calculate_trend(pir_values)

            # Weight analysis
            weight_history = enhanced_data.get("weight_history", [])
            if weight_history:
                weights = [wh["weight"] for wh in weight_history if wh["weight"]]
                if len(weights) > 1:
                    features["weight_consistency"] = 1 / (np.std(weights) + 0.1)
                    features["weight_trend"] = self._calculate_trend(weights)
                    features["current_weight_vs_avg"] = (
                        (weights[0] - np.mean(weights)) / np.mean(weights)
                        if weights
                        else 0
                    )

            # Margin analysis
            margins = enhanced_data.get("margins", [])
            if margins:
                margin_values = [
                    m["margin"] for m in margins if m["margin"] is not None
                ]
                if margin_values:
                    features["avg_winning_margin"] = np.mean(
                        [m for m in margin_values if m > 0]
                    )
                    features["avg_losing_margin"] = np.mean(
                        [abs(m) for m in margin_values if m < 0]
                    )
                    features["margin_consistency"] = 1 / (np.std(margin_values) + 1)

            # Performance indicators
            features["enhanced_data_quality"] = (
                len(sectional_times) + len(pir_ratings) + len(weight_history)
            )
            features["enhanced_data_available"] = (
                1 if features["enhanced_data_quality"] > 0 else 0
            )

        except Exception as e:
            print(f"⚠️ Error calculating enhanced features: {e}")

        return features

    def _calculate_trend(self, values):
        """Calculate trend in a series of values"""
        if len(values) < 2:
            return 0
        try:
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            return slope
        except:
            return 0

    def get_enhanced_race_context(self, venue, race_date, race_number):
        """Get enhanced context data for a specific race"""
        if not self.enhanced_data_available:
            return {}

        try:
            # Find the corresponding enhanced data file
            target_filename_pattern = (
                f"{venue}_Race{race_number}_{race_date.replace('-', '_')}"
            )

            json_files = list(self.enhanced_json_dir.glob("*.json"))
            for json_file in json_files:
                if target_filename_pattern in str(json_file):
                    with open(json_file, "r") as f:
                        data = json.load(f)
                        return {
                            "enhanced_race_data": data,
                            "expert_form_url": data.get("expert_form_url"),
                            "extraction_timestamp": data.get("extraction_timestamp"),
                            "data_quality": data.get("csv_data", {}).get(
                                "row_count", 0
                            ),
                        }

            return {}

        except Exception as e:
            print(f"⚠️ Error getting enhanced race context: {e}")
            return {}

    def integrate_enhanced_features_for_prediction(self, dog_stats, race_context):
        """Integrate enhanced features into existing dog statistics for prediction"""
        if not self.enhanced_data_available:
            return dog_stats

        try:
            # Get enhanced data for the dog
            dog_name = dog_stats.get("dog_name") or dog_stats.get("clean_name", "")
            venue = race_context.get("venue") if race_context else None

            enhanced_data = self.get_enhanced_dog_data(dog_name, venue)

            if enhanced_data and enhanced_data.get("enhanced_features"):
                # Merge enhanced features into dog stats
                enhanced_features = enhanced_data["enhanced_features"]

                # Update existing stats with enhanced data
                if "avg_first_section" in enhanced_features:
                    dog_stats["sectional_speed"] = enhanced_features[
                        "avg_first_section"
                    ]

                if "avg_pir_rating" in enhanced_features:
                    dog_stats["pir_rating"] = enhanced_features["avg_pir_rating"]

                if "sectional_consistency" in enhanced_features:
                    dog_stats["speed_consistency"] = enhanced_features[
                        "sectional_consistency"
                    ]

                if "weight_consistency" in enhanced_features:
                    dog_stats["weight_reliability"] = enhanced_features[
                        "weight_consistency"
                    ]

                # Add new enhanced-specific features
                dog_stats.update(
                    {
                        "enhanced_features": enhanced_features,
                        "enhanced_data_quality": enhanced_features.get(
                            "enhanced_data_quality", 0
                        ),
                        "has_enhanced_data": enhanced_features.get(
                            "enhanced_data_available", 0
                        )
                        == 1,
                    }
                )

                print(
                    f"✅ Enhanced data integrated for {dog_name}: {enhanced_features.get('enhanced_data_quality', 0)} data points"
                )

        except Exception as e:
            print(f"⚠️ Error integrating enhanced features: {e}")

        return dog_stats

    def create_enhanced_database_table(self):
        """Create database table to store enhanced expert form data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create enhanced_expert_data table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS enhanced_expert_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    race_id TEXT,
                    dog_name TEXT,
                    dog_clean_name TEXT,
                    venue TEXT,
                    race_date TEXT,
                    race_number INTEGER,
                    position INTEGER,
                    box_number INTEGER,
                    weight REAL,
                    distance INTEGER,
                    grade TEXT,
                    race_time REAL,
                    win_time REAL,
                    bonus_time REAL,
                    first_sectional REAL,
                    margin REAL,
                    pir_rating INTEGER,
                    starting_price REAL,
                    extraction_timestamp TEXT,
                    data_source TEXT DEFAULT 'enhanced_expert_form',
                    UNIQUE(race_id, dog_clean_name)
                )
            """
            )

            # Create indexes for efficient querying
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_enhanced_dog_name ON enhanced_expert_data(dog_clean_name)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_enhanced_race_date ON enhanced_expert_data(race_date)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_enhanced_venue ON enhanced_expert_data(venue)"
            )

            conn.commit()
            conn.close()

            print("✅ Enhanced expert data database table created")
            return True

        except Exception as e:
            print(f"❌ Error creating enhanced database table: {e}")
            return False

    def sync_enhanced_data_to_database(self):
        """Synchronize enhanced expert form data to database"""
        if not self.enhanced_data_available:
            print("⚠️ No enhanced data available to sync")
            return False

        try:
            # Create table if it doesn't exist
            self.create_enhanced_database_table()

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            records_added = 0
            csv_files = list(self.enhanced_csv_dir.glob("*.csv"))

            for csv_file in csv_files:
                try:
                    # Parse race info from filename
                    filename = csv_file.stem
                    parts = filename.replace("_enhanced", "").split("_")

                    if len(parts) >= 4:
                        venue = parts[0]
                        race_number = parts[1].replace("Race", "")
                        race_date = f"{parts[3]}-{self._month_to_number(parts[2])}-{parts[1] if parts[1].isdigit() else parts[4]}"

                        # Read CSV data
                        df = pd.read_csv(csv_file)

                        current_dog = None
                        for idx, row in df.iterrows():
                            dog_name_col = str(row.get("Dog Name", "")).strip()

                            # Check if this is a new dog row or continuation
                            if (
                                dog_name_col
                                and dog_name_col != '""'
                                and dog_name_col != ""
                            ):
                                if ". " in dog_name_col:
                                    current_dog = dog_name_col.split(". ", 1)[1]
                                else:
                                    current_dog = dog_name_col

                            if current_dog:
                                race_data = self._extract_enhanced_race_data(row)
                                if race_data:
                                    # Insert into database (using INSERT OR REPLACE to handle duplicates)
                                    race_id = f"{venue}_{race_number}_{race_date.replace('-', '_')}"

                                    cursor.execute(
                                        """
                                        INSERT OR REPLACE INTO enhanced_expert_data (
                                            race_id, dog_name, dog_clean_name, venue, race_date, race_number,
                                            position, box_number, weight, distance, grade, race_time,
                                            win_time, bonus_time, first_sectional, margin, pir_rating,
                                            starting_price, extraction_timestamp, data_source
                                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                        (
                                            race_id,
                                            current_dog,
                                            self._normalize_dog_name(current_dog),
                                            venue,
                                            race_date,
                                            race_number,
                                            race_data.get("position"),
                                            race_data.get("box"),
                                            race_data.get("weight"),
                                            race_data.get("distance"),
                                            race_data.get("grade"),
                                            race_data.get("time"),
                                            race_data.get("win_time"),
                                            race_data.get("bonus_time"),
                                            race_data.get("sectional_1st"),
                                            race_data.get("margin"),
                                            race_data.get("pir"),
                                            race_data.get("starting_price"),
                                            datetime.now().isoformat(),
                                            "enhanced_expert_form",
                                        ),
                                    )

                                    records_added += 1

                except Exception as e:
                    print(f"⚠️ Error processing {csv_file}: {e}")
                    continue

            conn.commit()
            conn.close()

            print(
                f"✅ Enhanced data sync completed: {records_added} records added/updated"
            )
            return True

        except Exception as e:
            print(f"❌ Error syncing enhanced data to database: {e}")
            return False

    def _month_to_number(self, month_name):
        """Convert month name to number"""
        months = {
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
        return months.get(month_name, "01")

    def get_enhanced_statistics(self):
        """Get statistics about enhanced data coverage"""
        try:
            conn = sqlite3.connect(self.db_path)

            # Check if enhanced table exists
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='enhanced_expert_data'"
            )
            if not cursor.fetchone():
                conn.close()
                return {"enhanced_data_available": False}

            # Get enhanced data statistics
            stats = {}

            # Total enhanced records
            cursor.execute("SELECT COUNT(*) FROM enhanced_expert_data")
            stats["total_enhanced_records"] = cursor.fetchone()[0]

            # Unique dogs with enhanced data
            cursor.execute(
                "SELECT COUNT(DISTINCT dog_clean_name) FROM enhanced_expert_data"
            )
            stats["dogs_with_enhanced_data"] = cursor.fetchone()[0]

            # Unique races with enhanced data
            cursor.execute("SELECT COUNT(DISTINCT race_id) FROM enhanced_expert_data")
            stats["races_with_enhanced_data"] = cursor.fetchone()[0]

            # Latest enhanced data date
            cursor.execute("SELECT MAX(race_date) FROM enhanced_expert_data")
            stats["latest_enhanced_date"] = cursor.fetchone()[0]

            conn.close()

            stats["enhanced_data_available"] = True
            return stats

        except Exception as e:
            print(f"❌ Error getting enhanced statistics: {e}")
            return {"enhanced_data_available": False}


def integrate_enhanced_data_pipeline():
    """Main function to integrate enhanced data into the prediction pipeline"""
    print("🚀 Starting Enhanced Data Integration...")

    integrator = EnhancedDataIntegrator()

    if not integrator.enhanced_data_available:
        print("❌ No enhanced data available for integration")
        return False

    # Sync enhanced data to database
    print("📊 Syncing enhanced data to database...")
    sync_success = integrator.sync_enhanced_data_to_database()

    # Get statistics
    stats = integrator.get_enhanced_statistics()
    print(f"📈 Enhanced data statistics: {stats}")

    print("✅ Enhanced data integration completed successfully!")
    return sync_success


if __name__ == "__main__":
    integrate_enhanced_data_pipeline()
