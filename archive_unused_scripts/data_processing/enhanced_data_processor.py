#!/usr/bin/env python3
"""
Enhanced Data Processor for Expert Form Data
============================================

This script processes the comprehensive data extracted by the Enhanced Expert Form Scraper
and integrates it into the existing database structure with new enriched fields.

Features:
- Process comprehensive JSON data files
- Extract and normalize enhanced CSV data
- Store sectional times and performance metrics
- Update existing database schema with new fields
- Create enriched feature sets for ML models
- Handle data validation and quality checks

Author: AI Assistant
Date: July 25, 2025
"""

import csv
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


class EnhancedDataProcessor:
    def __init__(self):
        self.database_path = "./databases/comprehensive_greyhound_data.db"
        self.enhanced_data_dir = "./enhanced_expert_data"
        self.csv_data_dir = "./enhanced_expert_data/csv"
        self.json_data_dir = "./enhanced_expert_data/json"

        # Ensure directories exist
        os.makedirs(self.enhanced_data_dir, exist_ok=True)
        os.makedirs(self.csv_data_dir, exist_ok=True)
        os.makedirs(self.json_data_dir, exist_ok=True)

        print("🔧 Enhanced Data Processor initialized")
        print(f"💾 Database: {self.database_path}")
        print(f"📁 Enhanced data directory: {self.enhanced_data_dir}")

        # Initialize database with enhanced schema
        self.init_enhanced_database()

    def init_enhanced_database(self):
        """Initialize database with enhanced tables for expert form data"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            # Create enhanced dog performance table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS enhanced_dog_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    race_id TEXT,
                    dog_name TEXT,
                    dog_clean_name TEXT,
                    box_number INTEGER,
                    
                    -- Enhanced CSV fields
                    sex TEXT,
                    trainer_name TEXT,
                    weight REAL,
                    distance TEXT,
                    race_date DATE,
                    track TEXT,
                    grade TEXT,
                    win_time TEXT,
                    first_section TEXT,
                    winning_time TEXT,
                    bonus TEXT,
                    margin TEXT,
                    pir_rating REAL,
                    starting_price REAL,
                    
                    -- Sectional times
                    sectional_400m TEXT,
                    sectional_500m TEXT,
                    sectional_600m TEXT,
                    sectional_700m TEXT,
                    sectional_800m TEXT,
                    
                    -- Performance ratings
                    speed_rating REAL,
                    class_rating REAL,
                    track_rating REAL,
                    form_rating REAL,
                    
                    -- Advanced metrics
                    acceleration_rating REAL,
                    finishing_rating REAL,
                    consistency_rating REAL,
                    track_bias_adjustment REAL,
                    
                    -- Odds and market data
                    opening_odds REAL,
                    closing_odds REAL,
                    market_movers TEXT,
                    betting_confidence REAL,
                    
                    -- External data
                    weather_impact REAL,
                    track_condition_impact REAL,
                    
                    -- Metadata
                    data_source TEXT DEFAULT 'expert_form',
                    extraction_timestamp DATETIME,
                    data_quality_score REAL,
                    
                    FOREIGN KEY (race_id) REFERENCES race_metadata (race_id),
                    UNIQUE(race_id, dog_clean_name, box_number, data_source)
                )
            """
            )

            # Create sectional analysis table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sectional_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    race_id TEXT,
                    dog_name TEXT,
                    dog_clean_name TEXT,
                    
                    -- Sectional breakdowns
                    first_section_time REAL,
                    first_section_speed REAL,
                    first_section_rank INTEGER,
                    
                    middle_section_time REAL,
                    middle_section_speed REAL,
                    middle_section_rank INTEGER,
                    
                    final_section_time REAL,
                    final_section_speed REAL,
                    final_section_rank INTEGER,
                    
                    -- Performance patterns
                    early_speed_type TEXT,
                    running_pattern TEXT,
                    finishing_ability TEXT,
                    
                    -- Comparative analysis
                    relative_to_winner REAL,
                    relative_to_field_avg REAL,
                    
                    -- Metadata
                    extraction_timestamp DATETIME,
                    analysis_version TEXT,
                    
                    FOREIGN KEY (race_id) REFERENCES race_metadata (race_id),
                    UNIQUE(race_id, dog_clean_name, analysis_version)
                )
            """
            )

            # Create track performance metrics table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS track_performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    race_id TEXT,
                    venue TEXT,
                    race_date DATE,
                    
                    -- Track records and benchmarks
                    track_record_time REAL,
                    track_record_holder TEXT,
                    average_winning_time REAL,
                    field_strength_rating REAL,
                    
                    -- Conditions
                    track_condition TEXT,
                    rail_position INTEGER,
                    weather_conditions TEXT,
                    temperature REAL,
                    humidity REAL,
                    wind_speed REAL,
                    wind_direction TEXT,
                    
                    -- Performance patterns
                    inside_draw_advantage REAL,
                    outside_draw_advantage REAL,
                    early_speed_advantage REAL,
                    
                    -- Race quality metrics
                    competitive_balance REAL,
                    pace_scenario TEXT,
                    winning_margin REAL,
                    
                    -- Metadata
                    extraction_timestamp DATETIME,
                    data_completeness REAL,
                    
                    FOREIGN KEY (race_id) REFERENCES race_metadata (race_id),
                    UNIQUE(race_id, venue, race_date)
                )
            """
            )

            # Create enhanced race analytics table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS enhanced_race_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    race_id TEXT,
                    
                    -- Performance distribution
                    winning_time_percentile REAL,
                    field_depth_score REAL,
                    competitive_rating REAL,
                    
                    -- Pace analysis
                    early_pace_pressure REAL,
                    middle_pace_rating REAL,
                    finishing_sprint_rating REAL,
                    
                    -- Market analysis
                    market_accuracy REAL,
                    value_opportunities TEXT,
                    betting_patterns TEXT,
                    
                    -- Predictive features
                    form_reversal_probability REAL,
                    upset_likelihood REAL,
                    confidence_interval REAL,
                    
                    -- Model inputs
                    feature_vector TEXT,  -- JSON string of all features
                    prediction_inputs TEXT,  -- JSON string of model inputs
                    
                    -- Quality metrics
                    data_reliability REAL,
                    prediction_confidence REAL,
                    
                    -- Metadata
                    analysis_timestamp DATETIME,
                    model_version TEXT,
                    
                    FOREIGN KEY (race_id) REFERENCES race_metadata (race_id),
                    UNIQUE(race_id, model_version)
                )
            """
            )

            conn.commit()
            conn.close()

            print("✅ Enhanced database schema initialized")

        except Exception as e:
            print(f"❌ Error initializing enhanced database: {e}")

    def process_comprehensive_json_files(self) -> Dict[str, Any]:
        """Process all comprehensive JSON files in the data directory"""
        print(f"\n🔄 PROCESSING COMPREHENSIVE JSON FILES")
        print("=" * 60)

        json_files = []
        if os.path.exists(self.json_data_dir):
            json_files = [
                f
                for f in os.listdir(self.json_data_dir)
                if f.endswith("_comprehensive.json")
            ]

        if not json_files:
            print("⚠️ No comprehensive JSON files found")
            return {"processed": 0, "failed": 0, "results": []}

        print(f"📁 Found {len(json_files)} comprehensive JSON files")

        results = {
            "processed": 0,
            "failed": 0,
            "results": [],
            "processing_start": datetime.now().isoformat(),
        }

        for json_file in json_files:
            print(f"\n🔍 Processing: {json_file}")

            try:
                file_path = os.path.join(self.json_data_dir, json_file)
                with open(file_path, "r", encoding="utf-8") as f:
                    comprehensive_data = json.load(f)

                # Process the comprehensive data
                processing_result = self.process_single_comprehensive_data(
                    comprehensive_data, json_file
                )

                if processing_result["success"]:
                    results["processed"] += 1
                    print(f"✅ Successfully processed: {json_file}")
                else:
                    results["failed"] += 1
                    print(f"❌ Failed to process: {json_file}")

                results["results"].append(
                    {
                        "file": json_file,
                        "success": processing_result["success"],
                        "details": processing_result,
                    }
                )

            except Exception as e:
                print(f"❌ Error processing {json_file}: {e}")
                results["failed"] += 1
                results["results"].append(
                    {"file": json_file, "success": False, "error": str(e)}
                )

        results["processing_end"] = datetime.now().isoformat()
        results["success_rate"] = (
            results["processed"] / len(json_files) * 100 if json_files else 0
        )

        print(f"\n🎯 PROCESSING SUMMARY")
        print("=" * 60)
        print(f"📊 Total files: {len(json_files)}")
        print(f"✅ Processed: {results['processed']}")
        print(f"❌ Failed: {results['failed']}")
        print(f"📈 Success rate: {results['success_rate']:.1f}%")

        return results

    def process_single_comprehensive_data(
        self, comprehensive_data: Dict[str, Any], filename: str
    ) -> Dict[str, Any]:
        """Process a single comprehensive data structure"""
        result = {"success": False, "race_id": None, "records_created": 0, "errors": []}

        try:
            # Extract race information
            race_info = comprehensive_data.get("race_info", {})
            if not race_info:
                result["errors"].append("No race info found")
                return result

            race_id = race_info.get("race_id")
            if not race_id:
                result["errors"].append("No race ID found")
                return result

            result["race_id"] = race_id

            # Process CSV data
            csv_data = comprehensive_data.get("csv_data", {})
            if csv_data:
                csv_result = self.process_enhanced_csv_data(race_id, csv_data)
                result["records_created"] += csv_result.get("records_created", 0)
                if csv_result.get("errors"):
                    result["errors"].extend(csv_result["errors"])

            # Process sectional data
            sectional_data = comprehensive_data.get("sectional_data", {})
            if sectional_data:
                sectional_result = self.process_sectional_data(race_id, sectional_data)
                result["records_created"] += sectional_result.get("records_created", 0)
                if sectional_result.get("errors"):
                    result["errors"].extend(sectional_result["errors"])

            # Process performance tables
            performance_tables = comprehensive_data.get("performance_tables", [])
            if performance_tables:
                perf_result = self.process_performance_tables(
                    race_id, performance_tables
                )
                result["records_created"] += perf_result.get("records_created", 0)
                if perf_result.get("errors"):
                    result["errors"].extend(perf_result["errors"])

            # Process additional metrics
            additional_metrics = comprehensive_data.get("additional_metrics", {})
            if additional_metrics:
                metrics_result = self.process_additional_metrics(
                    race_id, additional_metrics, race_info
                )
                result["records_created"] += metrics_result.get("records_created", 0)
                if metrics_result.get("errors"):
                    result["errors"].extend(metrics_result["errors"])

            # Process embedded JSON data
            embedded_json = comprehensive_data.get("embedded_json", {})
            if embedded_json:
                json_result = self.process_embedded_json_data(race_id, embedded_json)
                result["records_created"] += json_result.get("records_created", 0)
                if json_result.get("errors"):
                    result["errors"].extend(json_result["errors"])

            result["success"] = True

        except Exception as e:
            result["errors"].append(f"Processing error: {str(e)}")

        return result

    def process_enhanced_csv_data(
        self, race_id: str, csv_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process enhanced CSV data and store in database"""
        result = {"records_created": 0, "errors": []}

        try:
            headers = csv_data.get("headers", [])
            sample_data = csv_data.get("sample_data", [])

            if not headers or not sample_data:
                result["errors"].append("No CSV headers or data found")
                return result

            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            # Process each row of CSV data
            for row_data in sample_data:
                if len(row_data) != len(headers):
                    continue

                # Create a dictionary from headers and data
                row_dict = dict(zip(headers, row_data))

                # Extract and clean relevant fields
                enhanced_record = self.extract_enhanced_fields(race_id, row_dict)

                if enhanced_record:
                    # Insert into enhanced_dog_performance table
                    insert_sql = """
                        INSERT OR REPLACE INTO enhanced_dog_performance 
                        (race_id, dog_name, dog_clean_name, box_number, sex, trainer_name, 
                         weight, distance, race_date, track, grade, win_time, first_section, 
                         winning_time, bonus, margin, pir_rating, starting_price, 
                         data_source, extraction_timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """

                    cursor.execute(
                        insert_sql,
                        (
                            enhanced_record["race_id"],
                            enhanced_record["dog_name"],
                            enhanced_record["dog_clean_name"],
                            enhanced_record["box_number"],
                            enhanced_record["sex"],
                            enhanced_record["trainer_name"],
                            enhanced_record["weight"],
                            enhanced_record["distance"],
                            enhanced_record["race_date"],
                            enhanced_record["track"],
                            enhanced_record["grade"],
                            enhanced_record["win_time"],
                            enhanced_record["first_section"],
                            enhanced_record["winning_time"],
                            enhanced_record["bonus"],
                            enhanced_record["margin"],
                            enhanced_record["pir_rating"],
                            enhanced_record["starting_price"],
                            "expert_form_csv",
                            datetime.now().isoformat(),
                        ),
                    )

                    result["records_created"] += 1

            conn.commit()
            conn.close()

        except Exception as e:
            result["errors"].append(f"CSV processing error: {str(e)}")

        return result

    def extract_enhanced_fields(
        self, race_id: str, row_dict: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """Extract and normalize enhanced fields from CSV row"""
        try:
            # Map common field variations to standardized names
            field_mappings = {
                "dog_name": ["Dog Name", "Dog", "Runner", "Name"],
                "sex": ["Sex", "Gender"],
                "box_number": ["Box", "Box Number", "Draw"],
                "trainer_name": ["Trainer", "Trainer Name"],
                "weight": ["Weight", "Wt"],
                "distance": ["Distance", "Dist"],
                "race_date": ["Date", "Race Date"],
                "track": ["Track", "Venue", "Course"],
                "grade": ["Grade", "Class"],
                "win_time": ["Time", "Win Time", "Final Time"],
                "first_section": ["First Split", "1st Split", "First Section"],
                "winning_time": ["Winning Time", "Winner Time"],
                "bonus": ["Bonus", "Bnx"],
                "margin": ["Margin", "Mgn"],
                "pir_rating": ["PIR", "PIR Rating", "Rating"],
                "starting_price": ["SP", "Starting Price", "Price"],
            }

            enhanced_record = {"race_id": race_id}

            # Extract fields using mappings
            for standard_field, possible_names in field_mappings.items():
                value = None
                for name in possible_names:
                    if name in row_dict:
                        value = row_dict[name]
                        break

                # Clean and convert the value
                if standard_field == "dog_name":
                    enhanced_record["dog_name"] = value or ""
                    enhanced_record["dog_clean_name"] = self.clean_dog_name(value or "")
                elif standard_field in ["box_number"]:
                    enhanced_record[standard_field] = self.safe_int(value)
                elif standard_field in ["weight", "pir_rating", "starting_price"]:
                    enhanced_record[standard_field] = self.safe_float(value)
                else:
                    enhanced_record[standard_field] = value or ""

            return enhanced_record

        except Exception as e:
            print(f"❌ Error extracting enhanced fields: {e}")
            return None

    def process_sectional_data(
        self, race_id: str, sectional_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process sectional timing data"""
        result = {"records_created": 0, "errors": []}

        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            # Process each sectional table
            for table_key, table_data in sectional_data.items():
                if not isinstance(table_data, dict) or "dogs" not in table_data:
                    continue

                headers = table_data.get("headers", [])
                dogs_data = table_data.get("dogs", [])

                for dog_data in dogs_data:
                    # Extract sectional analysis
                    sectional_record = self.extract_sectional_analysis(
                        race_id, dog_data, headers
                    )

                    if sectional_record:
                        insert_sql = """
                            INSERT OR REPLACE INTO sectional_analysis 
                            (race_id, dog_name, dog_clean_name, first_section_time, 
                             first_section_speed, middle_section_time, middle_section_speed,
                             final_section_time, final_section_speed, running_pattern,
                             extraction_timestamp, analysis_version)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """

                        cursor.execute(
                            insert_sql,
                            (
                                sectional_record["race_id"],
                                sectional_record["dog_name"],
                                sectional_record["dog_clean_name"],
                                sectional_record["first_section_time"],
                                sectional_record["first_section_speed"],
                                sectional_record["middle_section_time"],
                                sectional_record["middle_section_speed"],
                                sectional_record["final_section_time"],
                                sectional_record["final_section_speed"],
                                sectional_record["running_pattern"],
                                datetime.now().isoformat(),
                                "expert_form_v1",
                            ),
                        )

                        result["records_created"] += 1

            conn.commit()
            conn.close()

        except Exception as e:
            result["errors"].append(f"Sectional data processing error: {str(e)}")

        return result

    def extract_sectional_analysis(
        self, race_id: str, dog_data: Dict[str, str], headers: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Extract sectional analysis from dog data"""
        try:
            record = {
                "race_id": race_id,
                "dog_name": "",
                "dog_clean_name": "",
                "first_section_time": None,
                "first_section_speed": None,
                "middle_section_time": None,
                "middle_section_speed": None,
                "final_section_time": None,
                "final_section_speed": None,
                "running_pattern": "",
            }

            # Extract dog name
            dog_name_fields = ["dog name", "dog", "runner", "name"]
            for field in dog_name_fields:
                if field in dog_data:
                    record["dog_name"] = dog_data[field]
                    record["dog_clean_name"] = self.clean_dog_name(dog_data[field])
                    break

            # Extract sectional times
            sectional_fields = {
                "first_section_time": [
                    "first section",
                    "1st section",
                    "400m",
                    "first split",
                ],
                "middle_section_time": [
                    "middle section",
                    "mid section",
                    "500m",
                    "600m",
                ],
                "final_section_time": ["final section", "last section", "700m", "800m"],
            }

            for record_field, possible_fields in sectional_fields.items():
                for field in possible_fields:
                    if field in dog_data:
                        time_value = dog_data[field]
                        record[record_field] = self.parse_time_to_seconds(time_value)
                        break

            # Determine running pattern
            record["running_pattern"] = self.determine_running_pattern(record)

            return record if record["dog_name"] else None

        except Exception as e:
            print(f"❌ Error extracting sectional analysis: {e}")
            return None

    def process_performance_tables(
        self, race_id: str, performance_tables: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process performance tables data"""
        result = {"records_created": 0, "errors": []}

        try:
            # Process each performance table
            for table in performance_tables:
                headers = table.get("headers", [])
                data = table.get("data", [])

                # Look for ratings, speeds, and other performance metrics
                for row in data:
                    if self.contains_performance_data(row, headers):
                        # Extract performance metrics and update enhanced_dog_performance
                        self.update_performance_metrics(race_id, row, headers)
                        result["records_created"] += 1

        except Exception as e:
            result["errors"].append(f"Performance tables processing error: {str(e)}")

        return result

    def process_additional_metrics(
        self,
        race_id: str,
        additional_metrics: Dict[str, Any],
        race_info: Dict[str, str],
    ) -> Dict[str, Any]:
        """Process additional metrics and track conditions"""
        result = {"records_created": 0, "errors": []}

        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            # Extract track performance metrics
            track_metrics = {
                "race_id": race_id,
                "venue": race_info.get("venue", ""),
                "race_date": race_info.get("date", ""),
                "track_condition": self.extract_track_condition(additional_metrics),
                "weather_conditions": self.extract_weather_conditions(
                    additional_metrics
                ),
                "temperature": self.extract_temperature(additional_metrics),
                "extraction_timestamp": datetime.now().isoformat(),
                "data_completeness": self.calculate_data_completeness(
                    additional_metrics
                ),
            }

            # Insert track performance metrics
            insert_sql = """
                INSERT OR REPLACE INTO track_performance_metrics 
                (race_id, venue, race_date, track_condition, weather_conditions, 
                 temperature, extraction_timestamp, data_completeness)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """

            cursor.execute(
                insert_sql,
                (
                    track_metrics["race_id"],
                    track_metrics["venue"],
                    track_metrics["race_date"],
                    track_metrics["track_condition"],
                    track_metrics["weather_conditions"],
                    track_metrics["temperature"],
                    track_metrics["extraction_timestamp"],
                    track_metrics["data_completeness"],
                ),
            )

            result["records_created"] += 1

            conn.commit()
            conn.close()

        except Exception as e:
            result["errors"].append(f"Additional metrics processing error: {str(e)}")

        return result

    def process_embedded_json_data(
        self, race_id: str, embedded_json: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process embedded JSON data for advanced analytics"""
        result = {"records_created": 0, "errors": []}

        try:
            # Extract feature vectors and prediction inputs
            feature_vector = self.extract_feature_vector(embedded_json)
            prediction_inputs = self.extract_prediction_inputs(embedded_json)

            if feature_vector or prediction_inputs:
                conn = sqlite3.connect(self.database_path)
                cursor = conn.cursor()

                # Store in enhanced_race_analytics
                insert_sql = """
                    INSERT OR REPLACE INTO enhanced_race_analytics 
                    (race_id, feature_vector, prediction_inputs, data_reliability,
                     analysis_timestamp, model_version)
                    VALUES (?, ?, ?, ?, ?, ?)
                """

                cursor.execute(
                    insert_sql,
                    (
                        race_id,
                        json.dumps(feature_vector) if feature_vector else None,
                        json.dumps(prediction_inputs) if prediction_inputs else None,
                        self.calculate_data_reliability(embedded_json),
                        datetime.now().isoformat(),
                        "expert_form_v1",
                    ),
                )

                result["records_created"] += 1

                conn.commit()
                conn.close()

        except Exception as e:
            result["errors"].append(f"Embedded JSON processing error: {str(e)}")

        return result

    def clean_dog_name(self, name: str) -> str:
        """Clean and standardize dog name"""
        if not name:
            return ""

        # Remove extra spaces and convert to title case
        cleaned = " ".join(name.strip().split())
        return cleaned.title()

    def safe_int(self, value: str) -> Optional[int]:
        """Safely convert string to integer"""
        if not value or not str(value).strip():
            return None
        try:
            return int(float(str(value).strip()))
        except (ValueError, TypeError):
            return None

    def safe_float(self, value: str) -> Optional[float]:
        """Safely convert string to float"""
        if not value or not str(value).strip():
            return None
        try:
            # Handle currency symbols and commas
            cleaned_value = str(value).strip().replace("$", "").replace(",", "")
            return float(cleaned_value)
        except (ValueError, TypeError):
            return None

    def parse_time_to_seconds(self, time_str: str) -> Optional[float]:
        """Parse time string to seconds"""
        if not time_str or not str(time_str).strip():
            return None

        try:
            time_str = str(time_str).strip()

            # Handle formats like "12.34" (seconds)
            if "." in time_str and ":" not in time_str:
                return float(time_str)

            # Handle formats like "1:23.45" (minutes:seconds)
            if ":" in time_str:
                parts = time_str.split(":")
                if len(parts) == 2:
                    minutes = float(parts[0])
                    seconds = float(parts[1])
                    return minutes * 60 + seconds

            return float(time_str)

        except (ValueError, TypeError):
            return None

    def determine_running_pattern(self, sectional_record: Dict[str, Any]) -> str:
        """Determine running pattern from sectional times"""
        first_time = sectional_record.get("first_section_time")
        middle_time = sectional_record.get("middle_section_time")
        final_time = sectional_record.get("final_section_time")

        if not any([first_time, middle_time, final_time]):
            return "unknown"

        # Simple pattern analysis
        if first_time and middle_time:
            if first_time < middle_time:
                return "front_runner"
            elif first_time > middle_time:
                return "come_from_behind"

        return "on_pace"

    def contains_performance_data(
        self, row: Dict[str, str], headers: List[str]
    ) -> bool:
        """Check if row contains performance data"""
        performance_keywords = ["rating", "speed", "time", "points", "score"]
        return any(
            keyword in " ".join(headers).lower() for keyword in performance_keywords
        )

    def update_performance_metrics(
        self, race_id: str, row: Dict[str, str], headers: List[str]
    ):
        """Update performance metrics for a dog"""
        # This would extract and update additional performance metrics
        # Implementation depends on the specific structure of performance tables
        pass

    def extract_track_condition(self, additional_metrics: Dict[str, Any]) -> str:
        """Extract track condition from additional metrics"""
        conditions = additional_metrics.get("track_conditions", [])
        if conditions and isinstance(conditions, list):
            return conditions[0] if conditions else ""
        return ""

    def extract_weather_conditions(self, additional_metrics: Dict[str, Any]) -> str:
        """Extract weather conditions from additional metrics"""
        weather = additional_metrics.get("weather", [])
        if weather and isinstance(weather, list):
            return " ".join(weather[:3])  # First 3 weather descriptors
        return ""

    def extract_temperature(
        self, additional_metrics: Dict[str, Any]
    ) -> Optional[float]:
        """Extract temperature from additional metrics"""
        temps = additional_metrics.get("temperature", [])
        if temps and isinstance(temps, list):
            try:
                return float(temps[0])
            except (ValueError, TypeError, IndexError):
                pass
        return None

    def calculate_data_completeness(self, additional_metrics: Dict[str, Any]) -> float:
        """Calculate data completeness score"""
        total_fields = 10  # Expected number of fields
        available_fields = len([v for v in additional_metrics.values() if v])
        return min(available_fields / total_fields, 1.0)

    def extract_feature_vector(
        self, embedded_json: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract feature vector from embedded JSON"""
        features = {}

        # Extract numerical arrays
        for key, value in embedded_json.items():
            if isinstance(value, list) and value:
                if "ratings" in key.lower():
                    features["ratings"] = value
                elif "speeds" in key.lower():
                    features["speeds"] = value
                elif "times" in key.lower():
                    features["times"] = value

        return features if features else None

    def extract_prediction_inputs(
        self, embedded_json: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract prediction inputs from embedded JSON"""
        inputs = {}

        # Look for structured data that could be used for predictions
        for key, value in embedded_json.items():
            if isinstance(value, dict) and value:
                if "race_data" in key.lower():
                    inputs["race_data"] = value
                elif "dog_data" in key.lower():
                    inputs["dog_data"] = value
                elif "performance" in key.lower():
                    inputs["performance_data"] = value

        return inputs if inputs else None

    def calculate_data_reliability(self, embedded_json: Dict[str, Any]) -> float:
        """Calculate data reliability score"""
        # Simple scoring based on data availability and consistency
        total_possible = 10
        available_data = len([v for v in embedded_json.values() if v])
        return min(available_data / total_possible, 1.0)

    def generate_processing_report(self) -> Dict[str, Any]:
        """Generate a comprehensive processing report"""
        print(f"\n📊 GENERATING PROCESSING REPORT")
        print("=" * 60)

        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            # Count records in each enhanced table
            tables_info = {}

            enhanced_tables = [
                "enhanced_dog_performance",
                "sectional_analysis",
                "track_performance_metrics",
                "enhanced_race_analytics",
            ]

            for table in enhanced_tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                tables_info[table] = count
                print(f"📋 {table}: {count} records")

            # Get recent processing activity
            cursor.execute(
                """
                SELECT COUNT(*) FROM enhanced_dog_performance 
                WHERE extraction_timestamp > datetime('now', '-1 day')
            """
            )
            recent_records = cursor.fetchone()[0]

            conn.close()

            report = {
                "timestamp": datetime.now().isoformat(),
                "table_counts": tables_info,
                "recent_activity": recent_records,
                "total_enhanced_records": sum(tables_info.values()),
            }

            print(f"🕐 Recent activity (24h): {recent_records} records")
            print(f"📈 Total enhanced records: {report['total_enhanced_records']}")

            # Save report
            report_path = os.path.join(
                self.enhanced_data_dir,
                f"processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            )
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)

            print(f"💾 Report saved: {report_path}")

            return report

        except Exception as e:
            print(f"❌ Error generating processing report: {e}")
            return {}


def main():
    """Main function for testing"""
    processor = EnhancedDataProcessor()

    # Process all comprehensive JSON files
    processing_results = processor.process_comprehensive_json_files()

    print(f"\n📊 Processing Results:")
    print(f"✅ Successful: {processing_results['processed']}")
    print(f"❌ Failed: {processing_results['failed']}")
    print(f"📈 Success Rate: {processing_results.get('success_rate', 0):.1f}%")

    # Generate processing report
    report = processor.generate_processing_report()

    return processing_results, report


if __name__ == "__main__":
    main()
