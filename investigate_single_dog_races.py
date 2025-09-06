#!/usr/bin/env python3
"""
Single Dog Race Investigation Script

This script investigates single dog races to determine if they are:
1. Legitimate single-runner races (rare but possible)
2. Data collection errors/incomplete races
3. Parsing/ingestion issues

Based on the analysis report showing 30.39% single-dog races.
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


def load_single_dog_races():
    """Load the identified single dog races from the report"""
    try:
        single_dog_file = "reports/data_quality/single_dog_races.csv"
        if os.path.exists(single_dog_file):
            return pd.read_csv(single_dog_file)
        else:
            logger.warning(f"Single dog races file not found: {single_dog_file}")
            return None
    except Exception as e:
        logger.error(f"Error loading single dog races: {e}")
        return None


def analyze_single_dog_patterns(df):
    """Analyze patterns in single dog races"""
    if df is None or df.empty:
        logger.error("No single dog race data to analyze")
        return

    logger.info("=== SINGLE DOG RACE PATTERN ANALYSIS ===")

    # Venue analysis
    if "venue" in df.columns:
        venue_counts = df["venue"].value_counts()
        logger.info(f"\nTop 10 venues with single dog races:")
        for venue, count in venue_counts.head(10).items():
            percentage = (count / len(df)) * 100
            logger.info(f"  {venue}: {count} races ({percentage:.1f}%)")

    # Distance analysis
    if "distance" in df.columns:
        distance_counts = df["distance"].value_counts()
        logger.info(f"\nTop 10 distances with single dog races:")
        for distance, count in distance_counts.head(10).items():
            percentage = (count / len(df)) * 100
            logger.info(f"  {distance}m: {count} races ({percentage:.1f}%)")

    # Grade analysis
    if "grade" in df.columns:
        grade_counts = df["grade"].value_counts()
        logger.info(f"\nTop 10 grades with single dog races:")
        for grade, count in grade_counts.head(10).items():
            percentage = (count / len(df)) * 100
            logger.info(f"  {grade}: {count} races ({percentage:.1f}%)")

    # Date pattern analysis
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = df["date"].dt.month
        df["day_of_week"] = df["date"].dt.day_name()

        monthly_counts = df["month"].value_counts().sort_index()
        logger.info(f"\nMonthly distribution of single dog races:")
        for month, count in monthly_counts.items():
            percentage = (count / len(df)) * 100
            logger.info(f"  Month {month}: {count} races ({percentage:.1f}%)")

        dow_counts = df["day_of_week"].value_counts()
        logger.info(f"\nDay of week distribution:")
        for day, count in dow_counts.items():
            percentage = (count / len(df)) * 100
            logger.info(f"  {day}: {count} races ({percentage:.1f}%)")


def check_database_completeness():
    """Check if single dog races in CSV files match database records"""
    try:
        conn = sqlite3.connect("greyhound_racing_data.db")

        # Check total races in database
        db_total = pd.read_sql_query(
            "SELECT COUNT(*) as total FROM race_metadata", conn
        )
        logger.info(f"Total races in database: {db_total['total'].iloc[0]}")

        # Check field sizes from race_metadata
        field_size_query = """
        SELECT 
            field_size, 
            actual_field_size,
            COUNT(*) as race_count,
            venue,
            distance,
            grade
        FROM race_metadata 
        WHERE field_size IS NOT NULL OR actual_field_size IS NOT NULL
        GROUP BY field_size, actual_field_size, venue, distance, grade
        ORDER BY race_count DESC
        """
        field_sizes = pd.read_sql_query(field_size_query, conn)
        logger.info(f"\n=== FIELD SIZE ANALYSIS FROM DATABASE ===")
        single_field_races = field_sizes[
            (field_sizes["field_size"] == 1) | (field_sizes["actual_field_size"] == 1)
        ]
        logger.info(
            f"Races with field_size=1: {single_field_races['race_count'].sum() if not single_field_races.empty else 0}"
        )

        # Check races with only one runner by counting dog entries
        single_runner_query = """
        SELECT 
            rm.race_id,
            rm.venue,
            rm.race_date,
            rm.distance,
            rm.grade,
            rm.field_size,
            rm.actual_field_size,
            COUNT(drd.id) as actual_runners
        FROM race_metadata rm
        LEFT JOIN dog_race_data drd ON rm.race_id = drd.race_id
        GROUP BY rm.race_id, rm.venue, rm.race_date, rm.distance, rm.grade, rm.field_size, rm.actual_field_size
        HAVING COUNT(drd.id) = 1
        ORDER BY rm.race_date DESC
        LIMIT 20
        """
        single_runners_db = pd.read_sql_query(single_runner_query, conn)
        logger.info(
            f"Races with only 1 actual runner in database: {len(single_runners_db)}"
        )

        if not single_runners_db.empty:
            logger.info("Sample single-runner races from database:")
            for _, race in single_runners_db.head(5).iterrows():
                logger.info(
                    f"  {race['venue']} Race {race['race_id']} on {race['race_date']}: {race['distance']}m Grade {race['grade']}"
                )

        # Check races with no runners at all
        no_runners_query = """
        SELECT 
            rm.race_id,
            rm.venue,
            rm.race_date,
            rm.distance,
            rm.grade,
            COUNT(drd.id) as actual_runners
        FROM race_metadata rm
        LEFT JOIN dog_race_data drd ON rm.race_id = drd.race_id
        GROUP BY rm.race_id, rm.venue, rm.race_date, rm.distance, rm.grade
        HAVING COUNT(drd.id) = 0
        LIMIT 10
        """
        no_runners_db = pd.read_sql_query(no_runners_query, conn)
        logger.info(f"Races with 0 runners in database: {len(no_runners_db)}")

        # Analyze field size discrepancies
        discrepancy_query = """
        SELECT 
            rm.venue,
            rm.distance,
            rm.grade,
            rm.field_size,
            rm.actual_field_size,
            COUNT(drd.id) as db_runner_count,
            COUNT(*) as race_count
        FROM race_metadata rm
        LEFT JOIN dog_race_data drd ON rm.race_id = drd.race_id
        WHERE rm.field_size != COUNT(drd.id) OR rm.actual_field_size != COUNT(drd.id)
        GROUP BY rm.race_id, rm.venue, rm.distance, rm.grade, rm.field_size, rm.actual_field_size
        HAVING COUNT(*) > 0
        LIMIT 10
        """
        try:
            discrepancies = pd.read_sql_query(discrepancy_query, conn)
            logger.info(f"Field size discrepancies found: {len(discrepancies)}")
        except Exception as e:
            logger.warning(f"Could not analyze field size discrepancies: {e}")

        conn.close()

        return single_runners_db, no_runners_db

    except Exception as e:
        logger.error(f"Error checking database: {e}")
        return None, None


def investigate_sample_files():
    """Investigate a sample of single dog race CSV files"""
    processed_dir = Path("processed")
    sample_files = []

    # Look for CSV files that might be single dog races
    for step_dir in processed_dir.glob("step*"):
        csv_files = list(step_dir.glob("*WAR*.csv"))  # WAR had single dog races
        if csv_files:
            sample_files.extend(csv_files[:3])  # Take first 3 from each step

    logger.info(f"\n=== INVESTIGATING SAMPLE FILES ===")
    for file_path in sample_files[:5]:  # Limit to first 5 files
        try:
            df = pd.read_csv(file_path)
            logger.info(f"\nFile: {file_path.name}")
            logger.info(f"  Rows: {len(df)}")
            logger.info(f"  Columns: {list(df.columns)}")

            # Check for dog names or runner information
            dog_columns = [
                col
                for col in df.columns
                if "dog" in col.lower() or "name" in col.lower()
            ]
            if dog_columns:
                logger.info(f"  Dog-related columns: {dog_columns}")
                for col in dog_columns[:2]:  # Check first 2 dog columns
                    unique_values = df[col].dropna().unique()
                    logger.info(f"    {col}: {len(unique_values)} unique values")
                    if len(unique_values) <= 5:
                        logger.info(f"      Values: {list(unique_values)}")

        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")


def generate_investigation_report():
    """Generate a comprehensive investigation report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"reports/single_dog_investigation_{timestamp}.md"

    # Load data and perform analysis
    single_dog_races = load_single_dog_races()
    single_runners_db, no_runners_db = check_database_completeness()

    with open(report_path, "w") as f:
        f.write("# Single Dog Race Investigation Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Summary\n")
        f.write(
            "Investigation into the 30.39% single-dog races identified in the data quality analysis.\n\n"
        )

        if single_dog_races is not None:
            f.write(f"- CSV single dog races analyzed: {len(single_dog_races)}\n")

        if single_runners_db is not None:
            f.write(f"- Database races with 1 runner: {len(single_runners_db)}\n")

        if no_runners_db is not None:
            f.write(f"- Database races with 0 runners: {len(no_runners_db)}\n")

        f.write("\n## Recommendations\n")
        f.write(
            "1. **Data Collection Review**: Verify data collection processes at venues with high single-dog rates\n"
        )
        f.write(
            "2. **Source Validation**: Cross-reference with official racing records\n"
        )
        f.write("3. **Parsing Logic**: Review CSV parsing logic for potential issues\n")
        f.write(
            "4. **Automated Alerts**: Implement monitoring for unusual runner counts\n"
        )

        f.write(f"\n## Next Steps\n")
        f.write("- Review sample CSV files manually\n")
        f.write("- Contact data sources to verify single-runner races\n")
        f.write("- Implement data validation rules\n")

    logger.info(f"Investigation report saved to: {report_path}")
    return report_path


def main():
    """Main investigation function"""
    logger.info("Starting Single Dog Race Investigation...")

    # Load and analyze patterns
    single_dog_races = load_single_dog_races()
    if single_dog_races is not None:
        analyze_single_dog_patterns(single_dog_races)

    # Check database
    check_database_completeness()

    # Investigate sample files
    investigate_sample_files()

    # Generate report
    report_path = generate_investigation_report()

    logger.info(f"Investigation complete. Report available at: {report_path}")


if __name__ == "__main__":
    main()
