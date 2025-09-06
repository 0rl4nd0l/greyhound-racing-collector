#!/usr/bin/env python3
"""
Data Validation and Remediation Script

Based on the investigation findings:
- 10,668 races with field_size=1 in database (83% of all races!)
- Only 1 actual single-runner race found in dog_race_data
- Significant discrepancy between CSV data and database records
"""

import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidationRemediation:
    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.validation_results = []

    def connect_db(self):
        """Create database connection"""
        return sqlite3.connect(self.db_path)

    def analyze_field_size_discrepancies(self):
        """Analyze field size calculation errors"""
        logger.info("=== ANALYZING FIELD SIZE DISCREPANCIES ===")

        with self.connect_db() as conn:
            query = """
            SELECT 
                rm.id,
                rm.race_id,
                rm.venue,
                rm.race_date,
                rm.distance,
                rm.grade,
                rm.field_size as recorded_field_size,
                rm.actual_field_size as recorded_actual_field_size,
                COUNT(drd.id) as actual_runner_count
            FROM race_metadata rm
            LEFT JOIN dog_race_data drd ON rm.race_id = drd.race_id
            GROUP BY rm.id, rm.race_id, rm.venue, rm.race_date, rm.distance, rm.grade, rm.field_size, rm.actual_field_size
            ORDER BY rm.race_date DESC
            """

            df = pd.read_sql_query(query, conn)

            # Identify discrepancies
            discrepancies = df[
                (df["recorded_field_size"] != df["actual_runner_count"])
                | (df["recorded_actual_field_size"] != df["actual_runner_count"])
            ]

            logger.info(f"Total races analyzed: {len(df)}")
            logger.info(f"Races with field size discrepancies: {len(discrepancies)}")

            # Show sample discrepancies
            logger.info("\nSample discrepancies:")
            for _, row in discrepancies.head(10).iterrows():
                logger.info(
                    f"  {row['venue']} {row['race_date']}: recorded={row['recorded_field_size']}, actual={row['actual_runner_count']}"
                )

            return discrepancies

    def validate_csv_structure(self):
        """Validate CSV file structure and content"""
        logger.info("=== VALIDATING CSV STRUCTURE ===")

        processed_dir = Path("processed")
        validation_issues = []

        # Check various processing steps
        for step_dir in processed_dir.glob("step*"):
            csv_files = list(step_dir.glob("*.csv"))
            logger.info(f"\nChecking {step_dir.name}: {len(csv_files)} files")

            for csv_file in csv_files[:5]:  # Sample first 5 files
                try:
                    df = pd.read_csv(csv_file)

                    # Basic validation
                    issue = {"file": str(csv_file), "rows": len(df), "issues": []}

                    # Check for single-row files
                    if len(df) == 1:
                        issue["issues"].append("single_row_file")

                    # Check for missing essential columns
                    essential_columns = ["Dog Name", "DATE", "TRACK"]
                    missing_columns = [
                        col for col in essential_columns if col not in df.columns
                    ]
                    if missing_columns:
                        issue["issues"].append(f"missing_columns: {missing_columns}")

                    if issue["issues"]:
                        validation_issues.append(issue)

                except Exception as e:
                    validation_issues.append(
                        {"file": str(csv_file), "issues": [f"read_error: {str(e)}"]}
                    )

        logger.info(
            f"\nValidation complete. Found {len(validation_issues)} files with issues."
        )
        return validation_issues

    def create_validation_rules(self):
        """Create automated validation rules"""
        logger.info("=== CREATING VALIDATION RULES ===")

        validation_rules = {
            "field_size_consistency": {
                "description": "field_size should match actual runner count",
                "sql": """
                    SELECT race_id, venue, race_date, field_size, 
                           (SELECT COUNT(*) FROM dog_race_data WHERE race_id = rm.race_id) as actual_count
                    FROM race_metadata rm 
                    WHERE field_size != (SELECT COUNT(*) FROM dog_race_data WHERE race_id = rm.race_id)
                    LIMIT 10
                """,
                "severity": "HIGH",
            },
            "missing_runners": {
                "description": "races should have at least 2 runners typically",
                "sql": """
                    SELECT race_id, venue, race_date, field_size
                    FROM race_metadata 
                    WHERE field_size < 2 AND grade NOT LIKE '%walkover%'
                    LIMIT 10
                """,
                "severity": "MEDIUM",
            },
        }

        # Test each validation rule
        with self.connect_db() as conn:
            for rule_name, rule in validation_rules.items():
                try:
                    result_df = pd.read_sql_query(rule["sql"], conn)
                    logger.info(
                        f"{rule_name} ({rule['severity']}): {len(result_df)} violations"
                    )

                except Exception as e:
                    logger.error(f"Failed to execute rule {rule_name}: {e}")

        return validation_rules

    def generate_remediation_report(self):
        """Generate comprehensive remediation report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"reports/data_remediation_{timestamp}.md"

        # Analyze current state
        discrepancies = self.analyze_field_size_discrepancies()
        validation_issues = self.validate_csv_structure()
        validation_rules = self.create_validation_rules()

        os.makedirs("reports", exist_ok=True)

        with open(report_path, "w") as f:
            f.write("# Data Validation and Remediation Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Executive Summary\n")
            f.write(
                f"- **Critical Issue**: {len(discrepancies)} races have incorrect field_size values\n"
            )
            f.write(
                f"- **CSV Issues**: {len(validation_issues)} files with structural problems\n"
            )
            f.write(
                f"- **Validation Rules**: {len(validation_rules)} automated checks implemented\n\n"
            )

            f.write("## Key Findings\n")
            f.write(
                "1. **Database Field Size Error**: Most races show field_size=1 when actual runner count is higher\n"
            )
            f.write(
                "2. **CSV Single-Dog Issue**: Many CSV files contain only one dog's historical data, not race results\n"
            )
            f.write(
                "3. **Data Structure Mismatch**: CSV format appears to be dog performance history, not race results\n\n"
            )

            f.write("## Immediate Actions Required\n")
            f.write("1. **Fix Field Sizes**: Update database field_size calculations\n")
            f.write(
                "2. **Review CSV Format**: Clarify whether CSV files are race results or dog performance history\n"
            )
            f.write(
                "3. **Implement Monitoring**: Set up automated validation rule checking\n\n"
            )

        logger.info(f"Remediation report saved to: {report_path}")
        return report_path


def main():
    """Main remediation function"""
    logger.info("Starting Data Validation and Remediation...")

    remediation = DataValidationRemediation()

    # Generate report
    report_path = remediation.generate_remediation_report()

    logger.info(f"\n=== REMEDIATION COMPLETE ===")
    logger.info(f"Report available at: {report_path}")


if __name__ == "__main__":
    main()
