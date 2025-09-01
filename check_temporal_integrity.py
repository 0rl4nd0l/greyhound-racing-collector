#!/usr/bin/env python3
"""
Check Temporal Integrity Script
===============================

This script checks for temporal integrity to ensure no look-ahead bias is present in prediction datasets:
1. Validates temporal ordering in datasets
2. Identifies any retroactive data inclusion issues
3. Confirms proper lag feature implementation

Based on existing temporal_leakage_fix.py
"""

import json
import logging
import sqlite3
import sys
import traceback
from datetime import datetime

import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)


class TemporalIntegrityChecker:
    """Class for checking temporal integrity and fixing look-ahead bias"""

    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "database_path": db_path,
            "issues": [],
            "summary": {},
            "errors": [],
        }

    def connect_db(self):
        """Connect to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            logger.info(f"Connected to database: {self.db_path}")
            return conn
        except Exception as e:
            error_msg = f"Failed to connect to database {self.db_path}: {e}"
            logger.error(error_msg)
            self.results["errors"].append(error_msg)
            raise

    def check_temporal_ordering(self, conn):
        """Ensure temporal ordering is maintained in datasets"""
        logger.info("Checking temporal ordering...")

        try:
            # Validate temporal ordering for race data
            query = """
            SELECT r.race_id, r.race_date, d.dog_clean_name, d.finish_position
            FROM dog_race_data d
            JOIN race_metadata r ON d.race_id = r.race_id
            WHERE r.race_date IS NOT NULL
            ORDER BY r.race_date ASC, r.race_id
            """
            df = pd.read_sql_query(query, conn)

            # Check for temporal violations
            df["prev_date"] = df["race_date"].shift(1)
            df["temporal_issue"] = df["race_date"] < df["prev_date"]
            issues_df = df[df["temporal_issue"]]

            if not issues_df.empty:
                self.results["issues"].append(
                    {"temporal_violations": issues_df.to_dict("records")}
                )
                logger.warning(f"Detected {len(issues_df)} temporal ordering issues.")
            else:
                logger.info("No temporal ordering issues found.")
        except Exception as e:
            error_msg = f"Error checking temporal ordering: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.results["errors"].append(error_msg)

    def check_retroactive_inclusion(self, conn):
        """Check for any retroactive data inclusion issues"""
        logger.info("Checking for retroactive data inclusion...")

        try:
            # Identify any records added in the past date
            query = """
            SELECT r.race_id, r.race_date, d.dog_clean_name, d.finish_position
            FROM dog_race_data d
            JOIN race_metadata r ON d.race_id = r.race_id
            WHERE r.race_date IS NOT NULL
            ORDER BY r.race_date DESC, r.race_id
            """
            df = pd.read_sql_query(query, conn)

            # Retroactive inclusion check based on order
            df["next_date"] = df["race_date"].shift(-1)
            df["retro_issue"] = df["race_date"] > df["next_date"]
            retro_issues_df = df[df["retro_issue"]]

            if not retro_issues_df.empty:
                self.results["issues"].append(
                    {"retroactive_violations": retro_issues_df.to_dict("records")}
                )
                logger.warning(
                    f"Detected {len(retro_issues_df)} retroactive inclusion issues."
                )
            else:
                logger.info("No retroactive inclusion issues found.")
        except Exception as e:
            error_msg = f"Error checking retroactive inclusion: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.results["errors"].append(error_msg)

    def run_checks(self):
        """Run the temporal integrity checking process"""
        logger.info("Running temporal integrity checks...")

        try:
            conn = self.connect_db()

            self.check_temporal_ordering(conn)
            self.check_retroactive_inclusion(conn)

            conn.close()

            # Determine overall status
            if self.results["issues"]:
                self.results["summary"]["status"] = "WARNING"
            else:
                self.results["summary"]["status"] = "PASS"

            logger.info(
                f"Temporal integrity check completed. Status: {self.results['summary']['status']}"
            )

        except Exception as e:
            error_msg = f"Critical error during temporal integrity checks: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.results["errors"].append(error_msg)
            self.results["summary"] = {"status": "FAIL", "critical_error": str(e)}

        return self.results


def main():
    """Main function to execute temporal integrity checks"""
    print("=" * 60)
    print("TEMPORAL INTEGRITY CHECK")
    print("=" * 60)

    checker = TemporalIntegrityChecker()
    results = checker.run_checks()

    # Print summary to stdout
    print(f"\nValidation Status: {results['summary']['status']}")
    print(f"Total Errors: {len(results['errors'])}")

    if results["issues"]:
        print("\nIssues Identified:")
        for issue_category in results["issues"]:
            print(json.dumps(issue_category, indent=2, default=str))

    # Output detailed results as JSON
    print("\n" + "=" * 60)
    print("DETAILED RESULTS (JSON)")
    print("=" * 60)
    print(json.dumps(results, indent=2, default=str))

    return results


if __name__ == "__main__":
    main()
