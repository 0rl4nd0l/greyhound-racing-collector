#!/usr/bin/env python3
"""
Data Integrity Validation Script
=================================

This script performs comprehensive data integrity checks on the greyhound racing database:
1. Duplicate detection
2. Missing data validation 
3. Referential integrity checks
4. Data quality validation
5. Schema constraint verification

Based on existing data_integrity_system.py and extended_integrity_analysis.py
"""

import json
import logging
import sqlite3
import sys
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)


class DataIntegrityValidator:
    """Comprehensive data integrity validation system"""

    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "database_path": db_path,
            "checks": {},
            "summary": {},
            "errors": [],
            "warnings": [],
        }

    def connect_db(self):
        """Connect to database with error handling"""
        try:
            conn = sqlite3.connect(self.db_path)
            logger.info(f"Connected to database: {self.db_path}")
            return conn
        except Exception as e:
            error_msg = f"Failed to connect to database {self.db_path}: {e}"
            logger.error(error_msg)
            self.results["errors"].append(error_msg)
            raise

    def check_duplicate_records(self, conn):
        """Check for duplicate records in key tables"""
        logger.info("Checking for duplicate records...")
        duplicate_checks = {}

        # Define tables and their unique key combinations
        tables_to_check = {
            "race_metadata": ["race_id"],
            "dog_race_data": ["race_id", "dog_clean_name", "box_number"],
            "enhanced_expert_data": ["race_id", "dog_clean_name"],
        }

        for table_name, key_columns in tables_to_check.items():
            try:
                # Check if table exists
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,),
                )
                if not cursor.fetchone():
                    logger.warning(f"Table {table_name} does not exist")
                    continue

                # Build duplicate detection query
                key_columns_str = ", ".join(key_columns)
                query = f"""
                SELECT {key_columns_str}, COUNT(*) as duplicate_count
                FROM {table_name}
                WHERE {' AND '.join([f'{col} IS NOT NULL' for col in key_columns])}
                GROUP BY {key_columns_str}
                HAVING COUNT(*) > 1
                ORDER BY duplicate_count DESC
                """

                duplicates_df = pd.read_sql_query(query, conn)

                duplicate_checks[table_name] = {
                    "total_duplicates": len(duplicates_df),
                    "max_duplicate_count": (
                        int(duplicates_df["duplicate_count"].max())
                        if len(duplicates_df) > 0
                        else 0
                    ),
                    "sample_duplicates": (
                        duplicates_df.head(10).to_dict("records")
                        if len(duplicates_df) > 0
                        else []
                    ),
                }

                if len(duplicates_df) > 0:
                    logger.warning(
                        f"Found {len(duplicates_df)} duplicate key combinations in {table_name}"
                    )
                else:
                    logger.info(f"No duplicates found in {table_name}")

            except Exception as e:
                error_msg = f"Error checking duplicates in {table_name}: {e}"
                logger.error(error_msg)
                self.results["errors"].append(error_msg)
                duplicate_checks[table_name] = {"error": str(e)}

        self.results["checks"]["duplicates"] = duplicate_checks
        return duplicate_checks

    def check_missing_data(self, conn):
        """Check for missing critical data"""
        logger.info("Checking for missing critical data...")
        missing_data_checks = {}

        # Define critical fields for each table
        critical_fields = {
            "race_metadata": ["race_id", "venue", "race_date", "race_number"],
            "dog_race_data": ["race_id", "dog_clean_name", "box_number"],
            "enhanced_expert_data": ["race_id", "dog_clean_name"],
        }

        for table_name, fields in critical_fields.items():
            try:
                # Check if table exists
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,),
                )
                if not cursor.fetchone():
                    continue

                # Get total record count
                total_count = pd.read_sql_query(
                    f"SELECT COUNT(*) as count FROM {table_name}", conn
                ).iloc[0]["count"]

                field_stats = {}
                for field in fields:
                    # Check for NULL, empty string, and zero values
                    missing_query = f"""
                    SELECT COUNT(*) as missing_count 
                    FROM {table_name} 
                    WHERE {field} IS NULL OR {field} = '' OR {field} = '0'
                    """
                    missing_count = pd.read_sql_query(missing_query, conn).iloc[0][
                        "missing_count"
                    ]

                    field_stats[field] = {
                        "missing_count": int(missing_count),
                        "missing_percentage": (
                            round((missing_count / total_count) * 100, 2)
                            if total_count > 0
                            else 0
                        ),
                    }

                    if missing_count > 0:
                        logger.warning(
                            f"{table_name}.{field}: {missing_count}/{total_count} ({field_stats[field]['missing_percentage']}%) missing"
                        )

                missing_data_checks[table_name] = {
                    "total_records": int(total_count),
                    "field_stats": field_stats,
                }

            except Exception as e:
                error_msg = f"Error checking missing data in {table_name}: {e}"
                logger.error(error_msg)
                self.results["errors"].append(error_msg)
                missing_data_checks[table_name] = {"error": str(e)}

        self.results["checks"]["missing_data"] = missing_data_checks
        return missing_data_checks

    def check_referential_integrity(self, conn):
        """Check referential integrity between tables"""
        logger.info("Checking referential integrity...")
        referential_checks = {}

        try:
            # Check race_id consistency between tables
            race_metadata_ids = set(
                pd.read_sql_query(
                    "SELECT DISTINCT race_id FROM race_metadata WHERE race_id IS NOT NULL",
                    conn,
                )["race_id"]
            )
            dog_race_ids = set(
                pd.read_sql_query(
                    "SELECT DISTINCT race_id FROM dog_race_data WHERE race_id IS NOT NULL",
                    conn,
                )["race_id"]
            )
            expert_ids = set(
                pd.read_sql_query(
                    "SELECT DISTINCT race_id FROM enhanced_expert_data WHERE race_id IS NOT NULL",
                    conn,
                )["race_id"]
            )

            # Find orphaned records
            orphaned_dog_races = dog_race_ids - race_metadata_ids
            orphaned_expert_data = expert_ids - race_metadata_ids

            referential_checks["race_id_integrity"] = {
                "race_metadata_count": len(race_metadata_ids),
                "dog_race_data_count": len(dog_race_ids),
                "enhanced_expert_data_count": len(expert_ids),
                "orphaned_dog_races": len(orphaned_dog_races),
                "orphaned_expert_data": len(orphaned_expert_data),
                "sample_orphaned_dog_races": list(orphaned_dog_races)[:10],
                "sample_orphaned_expert_data": list(orphaned_expert_data)[:10],
            }

            if orphaned_dog_races:
                logger.warning(
                    f"Found {len(orphaned_dog_races)} orphaned dog race records"
                )
            if orphaned_expert_data:
                logger.warning(
                    f"Found {len(orphaned_expert_data)} orphaned expert data records"
                )

        except Exception as e:
            error_msg = f"Error checking referential integrity: {e}"
            logger.error(error_msg)
            self.results["errors"].append(error_msg)
            referential_checks["error"] = str(e)

        self.results["checks"]["referential_integrity"] = referential_checks
        return referential_checks

    def check_data_quality(self, conn):
        """Check data quality and business rule violations"""
        logger.info("Checking data quality...")
        quality_checks = {}

        try:
            # Check box number ranges (should be 1-8)
            invalid_box_numbers = pd.read_sql_query(
                """
                SELECT race_id, dog_clean_name, box_number, COUNT(*) as count
                FROM dog_race_data 
                WHERE box_number NOT BETWEEN 1 AND 8 
                   OR box_number IS NULL
                GROUP BY race_id, dog_clean_name, box_number
                ORDER BY count DESC
                LIMIT 20
            """,
                conn,
            )

            # Check finish position ranges (should be 1-8, but allow higher for larger fields)
            invalid_finish_positions = pd.read_sql_query(
                """
                SELECT race_id, dog_clean_name, finish_position, COUNT(*) as count
                FROM dog_race_data 
                WHERE (CAST(finish_position AS INTEGER) < 1 OR CAST(finish_position AS INTEGER) > 12)
                   AND finish_position IS NOT NULL 
                   AND finish_position != ''
                GROUP BY race_id, dog_clean_name, finish_position
                ORDER BY count DESC
                LIMIT 20
            """,
                conn,
            )

            # Check for future race dates
            future_races = pd.read_sql_query(
                """
                SELECT race_id, race_date, venue, COUNT(*) as count
                FROM race_metadata 
                WHERE race_date > date('now')
                GROUP BY race_id, race_date, venue
                ORDER BY race_date DESC
                LIMIT 10
            """,
                conn,
            )

            quality_checks = {
                "invalid_box_numbers": {
                    "count": len(invalid_box_numbers),
                    "samples": invalid_box_numbers.to_dict("records"),
                },
                "invalid_finish_positions": {
                    "count": len(invalid_finish_positions),
                    "samples": invalid_finish_positions.to_dict("records"),
                },
                "future_races": {
                    "count": len(future_races),
                    "samples": future_races.to_dict("records"),
                },
            }

            if len(invalid_box_numbers) > 0:
                logger.warning(
                    f"Found {len(invalid_box_numbers)} records with invalid box numbers"
                )
            if len(invalid_finish_positions) > 0:
                logger.warning(
                    f"Found {len(invalid_finish_positions)} records with invalid finish positions"
                )
            if len(future_races) > 0:
                logger.warning(f"Found {len(future_races)} races with future dates")

        except Exception as e:
            error_msg = f"Error checking data quality: {e}"
            logger.error(error_msg)
            self.results["errors"].append(error_msg)
            quality_checks["error"] = str(e)

        self.results["checks"]["data_quality"] = quality_checks
        return quality_checks

    def generate_summary(self):
        """Generate overall integrity summary"""
        logger.info("Generating integrity summary...")

        summary = {
            "total_errors": len(self.results["errors"]),
            "total_warnings": len(self.results["warnings"]),
            "overall_status": "PASS",
            "recommendations": [],
        }

        # Check duplicate issues
        if "duplicates" in self.results["checks"]:
            total_duplicates = sum(
                [
                    check.get("total_duplicates", 0)
                    for check in self.results["checks"]["duplicates"].values()
                    if isinstance(check, dict) and "total_duplicates" in check
                ]
            )
            if total_duplicates > 0:
                summary["overall_status"] = "WARNING"
                summary["recommendations"].append(
                    f"Resolve {total_duplicates} duplicate records"
                )

        # Check referential integrity issues
        if "referential_integrity" in self.results["checks"]:
            ref_check = self.results["checks"]["referential_integrity"]
            orphaned_total = ref_check.get("orphaned_dog_races", 0) + ref_check.get(
                "orphaned_expert_data", 0
            )
            if orphaned_total > 0:
                summary["overall_status"] = "WARNING"
                summary["recommendations"].append(
                    f"Fix {orphaned_total} orphaned records"
                )

        # Check data quality issues
        if "data_quality" in self.results["checks"]:
            quality_check = self.results["checks"]["data_quality"]
            quality_issues = (
                quality_check.get("invalid_box_numbers", {}).get("count", 0)
                + quality_check.get("invalid_finish_positions", {}).get("count", 0)
                + quality_check.get("future_races", {}).get("count", 0)
            )
            if quality_issues > 0:
                summary["overall_status"] = "WARNING"
                summary["recommendations"].append(
                    f"Fix {quality_issues} data quality issues"
                )

        if summary["total_errors"] > 0:
            summary["overall_status"] = "FAIL"

        self.results["summary"] = summary
        return summary

    def run_all_checks(self):
        """Run all integrity checks"""
        logger.info("Starting comprehensive data integrity validation...")

        try:
            conn = self.connect_db()

            # Run all checks
            self.check_duplicate_records(conn)
            self.check_missing_data(conn)
            self.check_referential_integrity(conn)
            self.check_data_quality(conn)

            conn.close()

            # Generate summary
            self.generate_summary()

            logger.info(
                f"Data integrity validation completed. Status: {self.results['summary']['overall_status']}"
            )

        except Exception as e:
            error_msg = f"Critical error during integrity validation: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.results["errors"].append(error_msg)
            self.results["summary"] = {
                "overall_status": "FAIL",
                "critical_error": str(e),
            }

        return self.results


def main():
    """Main execution function"""
    print("=" * 60)
    print("DATA INTEGRITY VALIDATION")
    print("=" * 60)

    validator = DataIntegrityValidator()
    results = validator.run_all_checks()

    # Print summary to stdout
    print(f"\nValidation Status: {results['summary']['overall_status']}")
    print(f"Total Errors: {results['summary'].get('total_errors', 0)}")
    print(f"Total Warnings: {results['summary'].get('total_warnings', 0)}")

    if results["summary"].get("recommendations"):
        print("\nRecommendations:")
        for rec in results["summary"]["recommendations"]:
            print(f"  - {rec}")

    # Output detailed results as JSON
    print("\n" + "=" * 60)
    print("DETAILED RESULTS (JSON)")
    print("=" * 60)
    print(json.dumps(results, indent=2, default=str))

    return results


if __name__ == "__main__":
    main()
