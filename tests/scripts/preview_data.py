#!/usr/bin/env python3
"""
Data Quality Preview Script
Loads single_dog_races.csv and performs initial data inspection
"""

import os

import pandas as pd


def preview_single_dog_races():
    """Load and preview the single_dog_races.csv file"""

    file_path = "reports/data_quality/single_dog_races.csv"

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        # Load the CSV file
        print(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)

        print("\n" + "=" * 60)
        print("DATA OVERVIEW")
        print("=" * 60)

        # Print basic info about the dataset
        print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

        print("\n" + "-" * 40)
        print("DATAFRAME INFO")
        print("-" * 40)

        # Print df.info() for schema and data types
        df.info()

        print("\n" + "-" * 40)
        print("FIRST 10 ROWS")
        print("-" * 40)

        # Display first 10 rows with full width
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", 50)
        print(df.head(10))

        print("\n" + "-" * 40)
        print("COLUMN NAMES CHECK")
        print("-" * 40)

        # Check for potential issues with column names
        columns = df.columns.tolist()
        print(f"Column names: {columns}")

        # Check for weird column names
        issues = []
        for col in columns:
            if col != col.strip():
                issues.append(f"Column '{col}' has leading/trailing whitespace")
            if col.lower() != col:
                issues.append(f"Column '{col}' has mixed case")
            if " " in col:
                issues.append(f"Column '{col}' contains spaces")

        if issues:
            print("\nPotential column name issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("No obvious column name issues detected")

        print("\n" + "-" * 40)
        print("NULL VALUES CHECK")
        print("-" * 40)

        # Check for null values
        null_counts = df.isnull().sum()
        null_percentages = (df.isnull().sum() / len(df)) * 100

        print("Null values per column:")
        for col in df.columns:
            count = null_counts[col]
            percentage = null_percentages[col]
            print(f"  {col}: {count} ({percentage:.1f}%)")

        print("\n" + "-" * 40)
        print("DATA TYPES SUMMARY")
        print("-" * 40)

        # Summary of data types
        dtype_summary = df.dtypes.value_counts()
        print("Data type distribution:")
        for dtype, count in dtype_summary.items():
            print(f"  {dtype}: {count} columns")

    except Exception as e:
        print(f"Error loading or processing the file: {str(e)}")


def analyze_cleanup_options():
    """Analyze which records should be deleted vs fixed"""

    file_path = "reports/data_quality/single_dog_races.csv"

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    df = pd.read_csv(file_path)

    print("\n" + "=" * 60)
    print("CLEANUP ANALYSIS FOR SINGLE DOG RACES")
    print("=" * 60)

    # Analyze by venue
    venue_analysis = (
        df.groupby("venue")
        .agg({"race_id": "count", "race_date": "nunique"})
        .rename(columns={"race_id": "single_dog_count", "race_date": "unique_dates"})
    )
    venue_analysis["percentage"] = (venue_analysis["single_dog_count"] / len(df)) * 100
    venue_analysis = venue_analysis.sort_values("single_dog_count", ascending=False)

    print("\nSINGLE DOG RACES BY VENUE:")
    print("-" * 40)
    for venue, data in venue_analysis.head(10).iterrows():
        print(
            f"{venue:8s}: {int(data['single_dog_count']):4d} races ({data['percentage']:5.1f}%) across {int(data['unique_dates'])} dates"
        )

    # Analyze by date patterns
    df["race_date"] = pd.to_datetime(df["race_date"])
    date_analysis = df.groupby("race_date").size().sort_values(ascending=False)

    print("\nWORST DATES FOR SINGLE DOG RACES:")
    print("-" * 40)
    for date, count in date_analysis.head(10).items():
        print(f"{date.strftime('%Y-%m-%d')}: {count:3d} single-dog races")

    # Look for patterns in race_ids that might indicate data collection issues
    print("\nSAMPLE RACE IDs (to identify patterns):")
    print("-" * 40)
    sample_races = df.sample(min(10, len(df)))[
        ["race_id", "venue", "race_date", "single_dog_name"]
    ]
    for _, race in sample_races.iterrows():
        print(
            f"{race['race_id'][:50]:<50} | {race['venue']} | {race['single_dog_name'][:20]}"
        )

    # Recommendations
    print("\n" + "=" * 60)
    print("CLEANUP RECOMMENDATIONS")
    print("=" * 60)

    total_races = len(df)

    print("\n📊 CURRENT STATUS:")
    print(f"   Total single-dog races: {total_races:,}")
    print(f"   Unique venues affected: {df['venue'].nunique()}")
    print(
        f"   Date range: {df['race_date'].min().strftime('%Y-%m-%d')} to {df['race_date'].max().strftime('%Y-%m-%d')}"
    )

    print("\n🎯 RECOMMENDED ACTIONS:")

    # Option 1: Conservative cleanup
    print("\n1. 🟢 CONSERVATIVE APPROACH (Recommended):")
    print("   - Keep all single-dog races (they contain valid individual dog data)")
    print("   - Fix race_name column (currently 100% null)")
    print("   - Use for individual dog performance analysis")
    print("   - Mark clearly as 'incomplete race data' for race predictions")

    # Option 2: Selective cleanup
    recent_cutoff = df["race_date"].max() - pd.Timedelta(days=30)
    recent_races = df[df["race_date"] >= recent_cutoff]
    old_races = df[df["race_date"] < recent_cutoff]

    print("\n2. 🟡 SELECTIVE CLEANUP:")
    print(f"   - Keep recent races ({len(recent_races):,} races from last 30 days)")
    print(f"   - Archive older incomplete races ({len(old_races):,} races)")
    print("   - Focus data collection improvements on recent data")

    # Option 3: Aggressive cleanup
    print("\n3. 🔴 AGGRESSIVE CLEANUP (Not recommended):")
    print(f"   - Delete all {total_races:,} single-dog races")
    print("   - Risk: Lose valuable individual dog performance data")
    print("   - Risk: May indicate broader data collection issues")

    return {
        "total_single_dog_races": total_races,
        "venue_breakdown": venue_analysis.to_dict("index"),
        "recent_races": len(recent_races),
        "old_races": len(old_races),
        "recommendation": "conservative_keep_and_fix",
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--analyze-cleanup":
        analyze_cleanup_options()
    else:
        preview_single_dog_races()
