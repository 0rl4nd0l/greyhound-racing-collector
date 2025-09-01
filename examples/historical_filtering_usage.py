#!/usr/bin/env python3
"""
Historical Race Filtering Usage Examples
=======================================

This file demonstrates how to use the historical race filtering functionality
implemented in Step 5. Shows various usage patterns and CLI examples.

Author: AI Assistant  
Date: January 2025
"""

import os
import sys
from datetime import date, datetime, timedelta

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.date_parsing import is_historical


def example_basic_usage():
    """Basic usage examples of the is_historical function"""
    print("=" * 60)
    print("📚 BASIC USAGE EXAMPLES")
    print("=" * 60)

    print("🔍 Using is_historical() function directly:")
    print()

    # Example dates
    example_dates = [
        "2024-01-01",  # Past date
        "2025-12-31",  # Future date
        date.today(),  # Today
        datetime.now() - timedelta(days=5),  # 5 days ago
    ]

    for test_date in example_dates:
        result = is_historical(test_date)
        print(f"  is_historical({test_date}) = {result}")

    print()


def example_cli_usage():
    """Show CLI usage examples"""
    print("=" * 60)
    print("🖥️  CLI USAGE EXAMPLES")
    print("=" * 60)

    print("📋 Batch Prediction CLI with --historical flag:")
    print()

    cli_examples = [
        {
            "command": "python batch_prediction_cli.py --input ./data --output ./results --historical",
            "description": "Process only historical races from data directory",
        },
        {
            "command": "python batch_prediction_cli.py --input ./races --output ./historical_results --historical --workers 4",
            "description": "Historical mode with 4 parallel workers",
        },
        {
            "command": "python cli_batch_predictor.py --batch ./race_files --historical",
            "description": "CLI batch predictor in historical mode",
        },
        {
            "command": "python cli_batch_predictor.py --file race_2024-01-01.csv --historical",
            "description": "Process single historical race file",
        },
    ]

    for i, example in enumerate(cli_examples, 1):
        print(f"{i}. {example['description']}:")
        print(f"   {example['command']}")
        print()


def example_programmatic_usage():
    """Show programmatic usage examples"""
    print("=" * 60)
    print("🐍 PROGRAMMATIC USAGE EXAMPLES")
    print("=" * 60)

    print("📝 Using BatchPredictionPipeline with historical filtering:")
    print()

    code_example = """
# Example: Create batch job with historical filtering
from batch_prediction_pipeline import BatchPredictionPipeline

pipeline = BatchPredictionPipeline()

# Find CSV files
csv_files = ["race_2024-01-01.csv", "race_2025-01-01.csv", "race_2024-06-15.csv"]

# Create job with historical=True - only processes files with dates < today
job_id = pipeline.create_batch_job(
    name="Historical Race Analysis",
    input_files=csv_files,
    output_dir="./historical_results",
    historical=True  # This enables historical filtering
)

# Files with dates >= today will be automatically filtered out
job = pipeline.run_batch_job(job_id)
print(f"Processed {job.completed_files} historical races")
"""

    print(code_example)


def example_file_filtering_logic():
    """Show how file filtering works"""
    print("=" * 60)
    print("🗂️  FILE FILTERING LOGIC")
    print("=" * 60)

    print("📁 How historical filtering determines which files to process:")
    print()

    # Simulate different filename patterns
    today_str = date.today().strftime("%Y-%m-%d")
    yesterday_str = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    tomorrow_str = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    filename_examples = [
        (f"race_{yesterday_str}.csv", "✅ PROCESSED", "Date in filename is < today"),
        (f"race_{today_str}.csv", "❌ SKIPPED", "Date in filename is today"),
        (f"race_{tomorrow_str}.csv", "❌ SKIPPED", "Date in filename is > today"),
        ("race_01012024.csv", "✅ PROCESSED", "DDMMYYYY format shows historical date"),
        ("race_31122025.csv", "❌ SKIPPED", "DDMMYYYY format shows future date"),
        (
            "general_race.csv",
            "📝 DEPENDS",
            "No date in filename - checks file content/mtime",
        ),
    ]

    print("Filename Pattern Analysis:")
    for filename, result, reason in filename_examples:
        print(f"  {filename:<25} {result:<12} - {reason}")

    print()
    print("🔍 For files without dates in filenames:")
    print("  1. Examines CSV content for date columns (date, race_date, etc.)")
    print("  2. If dates found, checks if any are historical using is_historical()")
    print("  3. Falls back to file modification time if no date columns found")
    print()


def example_integration_scenarios():
    """Show real-world integration scenarios"""
    print("=" * 60)
    print("🌍 REAL-WORLD INTEGRATION SCENARIOS")
    print("=" * 60)

    scenarios = [
        {
            "title": "Daily Batch Processing",
            "description": "Run nightly batch job to process only historical races",
            "use_case": "Backfill analysis, model training on past data",
            "command": "python batch_prediction_cli.py --input ./daily_races --output ./processed --historical --quiet",
        },
        {
            "title": "Data Migration",
            "description": "Process large archive of race files, filtering out future/current races",
            "use_case": "Clean up mixed datasets, separate historical from upcoming",
            "command": "python batch_prediction_cli.py --input ./archive --output ./historical_only --historical --workers 6",
        },
        {
            "title": "Model Training Data",
            "description": "Ensure training data only contains historical races to prevent data leakage",
            "use_case": "ML model training, backtesting validation",
            "command": "python cli_batch_predictor.py --batch ./training_data --historical",
        },
        {
            "title": "Audit and Compliance",
            "description": "Process only completed races for regulatory reporting",
            "use_case": "Audit trails, compliance reporting, post-race analysis",
            "command": "python batch_prediction_cli.py --input ./audit_data --output ./compliance --historical --format json",
        },
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario['title']}")
        print(f"   Description: {scenario['description']}")
        print(f"   Use Case: {scenario['use_case']}")
        print(f"   Command: {scenario['command']}")
        print()


def main():
    """Run all examples"""
    print("🎯 HISTORICAL RACE FILTERING - USAGE EXAMPLES")
    print(f"📅 Current date: {date.today().strftime('%Y-%m-%d')}")
    print()

    # Run all example sections
    example_basic_usage()
    example_cli_usage()
    example_programmatic_usage()
    example_file_filtering_logic()
    example_integration_scenarios()

    print("=" * 60)
    print("✅ SUMMARY")
    print("=" * 60)
    print(
        "The --historical flag enables filtering to process only races with dates < today."
    )
    print(
        "This prevents data leakage and ensures clean separation of historical vs future data."
    )
    print()
    print("Key benefits:")
    print("• 🛡️  Prevents temporal data leakage in ML training")
    print("• 🧹 Cleans mixed datasets automatically")
    print("• ⚡ Improves processing efficiency by skipping irrelevant files")
    print("• 📊 Ensures compliance with audit requirements")
    print("• 🎯 Enables focused historical analysis workflows")
    print()


if __name__ == "__main__":
    main()
