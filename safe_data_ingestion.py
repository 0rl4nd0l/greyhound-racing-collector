#!/usr/bin/env python3
"""
Safe Data Ingestion Wrapper
============================

This script wraps all data ingestion operations with validation and duplicate prevention.
It provides a safe interface for inserting new data while maintaining data integrity.

Usage:
    python safe_data_ingestion.py --data-file races.csv --table race_metadata
    python safe_data_ingestion.py --json-data '{"race_id": "test_123", ...}' --table race_metadata

Author: AI Assistant
Date: 2025-01-27
"""

import argparse
import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd

from data_integrity_system import DataIntegrityManager


class SafeDataIngestion:
    """Safe data ingestion with comprehensive validation"""

    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path
        self.integrity_manager = DataIntegrityManager(db_path)
        self.setup_logging()
        self.ingestion_stats = {
            "records_processed": 0,
            "records_inserted": 0,
            "records_rejected": 0,
            "validation_errors": [],
            "successful_inserts": [],
        }

    def setup_logging(self):
        """Setup logging for ingestion operations"""
        os.makedirs("logs", exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - INGESTION - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("logs/safe_data_ingestion.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def validate_and_clean_record(
        self, record: Dict, table_name: str
    ) -> Tuple[Dict, List[str]]:
        """Validate and clean a single record before insertion"""
        cleaned_record = record.copy()
        warnings = []

        # Clean common data issues
        for key, value in cleaned_record.items():
            if isinstance(value, str):
                # Clean string values
                cleaned_value = value.strip()
                if cleaned_value.lower() in ["", "nan", "null", "none"]:
                    cleaned_record[key] = None
                    warnings.append(f"Converted empty string to NULL for field {key}")
                else:
                    cleaned_record[key] = cleaned_value
            elif pd.isna(value):
                cleaned_record[key] = None
                warnings.append(f"Converted NaN to NULL for field {key}")

        # Table-specific cleaning
        if table_name == "race_metadata":
            # Ensure race_date is properly formatted
            if "race_date" in cleaned_record and cleaned_record["race_date"]:
                try:
                    # Try to parse and reformat date
                    date_str = str(cleaned_record["race_date"])
                    if "July" in date_str:
                        # Handle format like "July-01-2025"
                        cleaned_record["race_date"] = date_str
                    else:
                        # Try to parse as standard date
                        parsed_date = pd.to_datetime(date_str)
                        cleaned_record["race_date"] = parsed_date.strftime("%Y-%m-%d")
                except:
                    warnings.append(
                        f"Could not parse race_date: {cleaned_record['race_date']}"
                    )

        elif table_name == "dog_race_data":
            # Ensure box_number is valid
            if (
                "box_number" in cleaned_record
                and cleaned_record["box_number"] is not None
            ):
                try:
                    box_num = int(cleaned_record["box_number"])
                    if box_num < 1 or box_num > 8:
                        cleaned_record["box_number"] = None
                        warnings.append(f"Invalid box number {box_num} set to NULL")
                    else:
                        cleaned_record["box_number"] = box_num
                except (ValueError, TypeError):
                    cleaned_record["box_number"] = None
                    warnings.append(f"Non-numeric box number set to NULL")

        elif table_name == "enhanced_expert_data":
            # Ensure position is valid
            if "position" in cleaned_record and cleaned_record["position"] is not None:
                try:
                    pos = int(cleaned_record["position"])
                    if pos < 1 or pos > 8:
                        cleaned_record["position"] = None
                        warnings.append(f"Invalid position {pos} set to NULL")
                    else:
                        cleaned_record["position"] = pos
                except (ValueError, TypeError):
                    cleaned_record["position"] = None
                    warnings.append(f"Non-numeric position set to NULL")

        return cleaned_record, warnings

    def insert_single_record(
        self, record: Dict, table_name: str
    ) -> Tuple[bool, List[str]]:
        """Safely insert a single record with validation"""
        self.ingestion_stats["records_processed"] += 1

        # Clean and validate the record
        cleaned_record, warnings = self.validate_and_clean_record(record, table_name)

        if warnings:
            self.logger.warning(f"Data cleaning warnings for record: {warnings}")

        # Use integrity manager for safe insertion
        with self.integrity_manager:
            success, errors = self.integrity_manager.safe_insert_record(
                table_name, cleaned_record
            )

        if success:
            self.ingestion_stats["records_inserted"] += 1
            self.ingestion_stats["successful_inserts"].append(
                {
                    "table": table_name,
                    "record_identifier": cleaned_record.get("race_id")
                    or cleaned_record.get("id")
                    or "unknown",
                    "timestamp": datetime.now().isoformat(),
                }
            )
            self.logger.info(f"Successfully inserted record into {table_name}")
            return True, warnings
        else:
            self.ingestion_stats["records_rejected"] += 1
            self.ingestion_stats["validation_errors"].extend(errors)
            self.logger.error(f"Failed to insert record into {table_name}: {errors}")
            return False, errors + warnings

    def insert_batch_records(self, records: List[Dict], table_name: str) -> Dict:
        """Safely insert a batch of records"""
        self.logger.info(
            f"Starting batch insertion of {len(records)} records into {table_name}"
        )

        batch_results = {
            "total_records": len(records),
            "successful_inserts": 0,
            "failed_inserts": 0,
            "errors": [],
            "warnings": [],
        }

        for i, record in enumerate(records):
            try:
                success, messages = self.insert_single_record(record, table_name)

                if success:
                    batch_results["successful_inserts"] += 1
                else:
                    batch_results["failed_inserts"] += 1
                    batch_results["errors"].extend(messages)

                # Add any warnings
                batch_results["warnings"].extend(
                    [msg for msg in messages if "warning" in msg.lower()]
                )

                # Log progress every 100 records
                if (i + 1) % 100 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(records)} records")

            except Exception as e:
                batch_results["failed_inserts"] += 1
                batch_results["errors"].append(
                    f"Unexpected error processing record {i}: {str(e)}"
                )
                self.logger.error(f"Unexpected error processing record {i}: {e}")

        self.logger.info(
            f"Batch insertion complete: {batch_results['successful_inserts']} success, {batch_results['failed_inserts']} failed"
        )
        return batch_results

    def load_from_csv(self, csv_path: str, table_name: str, **pandas_kwargs) -> Dict:
        """Load and safely insert data from CSV file"""
        self.logger.info(f"Loading data from CSV: {csv_path}")

        try:
            # Load CSV with pandas
            df = pd.read_csv(csv_path, **pandas_kwargs)

            # Convert to list of dictionaries
            records = df.to_dict("records")

            self.logger.info(f"Loaded {len(records)} records from CSV")

            # Insert batch
            return self.insert_batch_records(records, table_name)

        except Exception as e:
            error_msg = f"Failed to load CSV {csv_path}: {str(e)}"
            self.logger.error(error_msg)
            return {
                "total_records": 0,
                "successful_inserts": 0,
                "failed_inserts": 0,
                "errors": [error_msg],
                "warnings": [],
            }

    def load_from_json(self, json_data: str, table_name: str) -> Dict:
        """Load and safely insert data from JSON"""
        self.logger.info(f"Loading data from JSON string")

        try:
            # Parse JSON
            if isinstance(json_data, str):
                data = json.loads(json_data)
            else:
                data = json_data

            # Handle single record or list of records
            if isinstance(data, dict):
                records = [data]
            elif isinstance(data, list):
                records = data
            else:
                raise ValueError(
                    "JSON data must be a dictionary or list of dictionaries"
                )

            self.logger.info(f"Parsed {len(records)} records from JSON")

            # Insert batch
            return self.insert_batch_records(records, table_name)

        except Exception as e:
            error_msg = f"Failed to parse JSON data: {str(e)}"
            self.logger.error(error_msg)
            return {
                "total_records": 0,
                "successful_inserts": 0,
                "failed_inserts": 0,
                "errors": [error_msg],
                "warnings": [],
            }

    def generate_ingestion_report(self) -> str:
        """Generate detailed ingestion report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"reports/ingestion_report_{timestamp}.json"

        os.makedirs("reports", exist_ok=True)

        report = {
            "timestamp": datetime.now().isoformat(),
            "database_path": self.db_path,
            "ingestion_statistics": self.ingestion_stats,
            "success_rate": (
                self.ingestion_stats["records_inserted"]
                / max(self.ingestion_stats["records_processed"], 1)
            )
            * 100,
            "recommendations": [],
        }

        # Add recommendations based on results
        if self.ingestion_stats["records_rejected"] > 0:
            report["recommendations"].append(
                "Review rejected records and fix data quality issues"
            )

        if len(self.ingestion_stats["validation_errors"]) > 10:
            report["recommendations"].append(
                "Consider improving data validation at the source"
            )

        if report["success_rate"] < 95:
            report["recommendations"].append(
                "Data quality issues detected - review ingestion process"
            )
        else:
            report["recommendations"].append(
                "Good data quality - maintain current practices"
            )

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Ingestion report generated: {report_path}")
        return report_path


def main():
    """Main CLI interface for safe data ingestion"""
    parser = argparse.ArgumentParser(description="Safe Data Ingestion with Validation")
    parser.add_argument(
        "--db-path", default="greyhound_racing_data.db", help="Database path"
    )
    parser.add_argument("--table", required=True, help="Target table name")
    parser.add_argument("--data-file", help="CSV file to ingest")
    parser.add_argument("--json-data", help="JSON data string to ingest")
    parser.add_argument(
        "--generate-report", action="store_true", help="Generate ingestion report"
    )

    args = parser.parse_args()

    # Initialize safe ingestion
    ingester = SafeDataIngestion(args.db_path)

    print("=== Safe Data Ingestion System ===\\n")

    results = None

    try:
        if args.data_file:
            print(f"Ingesting data from CSV file: {args.data_file}")
            results = ingester.load_from_csv(args.data_file, args.table)

        elif args.json_data:
            print(f"Ingesting data from JSON")
            results = ingester.load_from_json(args.json_data, args.table)

        else:
            print("❌ Error: Must specify either --data-file or --json-data")
            return 1

        # Display results
        if results:
            print(f"\\n=== INGESTION RESULTS ===")
            print(f"Total records: {results['total_records']}")
            print(f"Successful inserts: {results['successful_inserts']}")
            print(f"Failed inserts: {results['failed_inserts']}")

            if results["errors"]:
                print(f"\\n⚠️  ERRORS ({len(results['errors'])}):")
                for error in results["errors"][:5]:  # Show first 5 errors
                    print(f"  - {error}")
                if len(results["errors"]) > 5:
                    print(f"  ... and {len(results['errors']) - 5} more errors")

            if results["warnings"]:
                print(f"\\n💡 WARNINGS ({len(results['warnings'])}):")
                for warning in results["warnings"][:3]:  # Show first 3 warnings
                    print(f"  - {warning}")
                if len(results["warnings"]) > 3:
                    print(f"  ... and {len(results['warnings']) - 3} more warnings")

        # Generate report if requested
        if args.generate_report:
            report_path = ingester.generate_ingestion_report()
            print(f"\\n📊 Report generated: {report_path}")

        print("\\n✅ Ingestion completed!")
        return 0

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
