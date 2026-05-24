import re
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def load_and_parse_csv(file_path: str) -> pd.DataFrame:
    """
    Load CSV file and handle the special format where empty quotes ("")
    indicate continuation of the previous dog's records.
    """
    # Read CSV with comma separator
    data = pd.read_csv(file_path)

    # Forward fill dog names where we have empty quotes
    current_dog = None
    dog_names = []

    for idx, row in data.iterrows():
        dog_name = row["Dog Name"]
        # Handle NaN values and convert to string
        if pd.isna(dog_name):
            dog_name = '""'
        else:
            dog_name = str(dog_name).strip()

        if dog_name and dog_name != '""':
            # Extract dog name and number (e.g., "1. Sky Chaser" -> "Sky Chaser")
            clean_name = re.sub(r"^\d+\.\s*", "", dog_name)
            current_dog = clean_name
        dog_names.append(current_dog)

    data["Dog Name"] = dog_names
    return data


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to be consistent and lowercase.
    """
    column_mapping = {
        "Dog Name": "dog_name",
        "Sex": "sex",
        "PLC": "place",
        "BOX": "box_number",
        "WGT": "weight_kg",
        "DIST": "distance_m",
        "DATE": "race_date",
        "TRACK": "track_code",
        "G": "grade",
        "TIME": "race_time",
        "WIN": "winning_time",
        "BON": "bonus_points",
        "1 SEC": "first_section_time",
        "MGN": "margin",
        "W/2G": "winner_two_greyhounds",
        "PIR": "performance_index_rating",
        "SP": "starting_price",
        "2 SEC": "second_section_time",
        "3 SEC": "third_section_time",
    }

    df = df.rename(columns=column_mapping)
    return df


def validate_and_convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and convert data types for numeric columns.
    """
    numeric_columns = [
        "place",
        "box_number",
        "weight_kg",
        "distance_m",
        "race_time",
        "winning_time",
        "bonus_points",
        "first_section_time",
        "margin",
        "performance_index_rating",
        "starting_price",
        "second_section_time",
        "third_section_time",
    ]

    for col in numeric_columns:
        if col in df.columns:
            # Handle empty strings and convert to numeric
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert date column
    if "race_date" in df.columns:
        df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values appropriately for different column types.
    """
    # For categorical columns, keep NaN as is
    categorical_cols = ["sex", "grade", "track_code", "winner_two_greyhounds"]

    # For numeric columns, we can use various strategies
    numeric_cols = [
        "weight_kg",
        "distance_m",
        "race_time",
        "winning_time",
        "bonus_points",
        "first_section_time",
        "margin",
        "performance_index_rating",
        "starting_price",
    ]

    # Fill missing weight with median weight for each dog
    if "weight_kg" in df.columns:
        df["weight_kg"] = df.groupby("dog_name")["weight_kg"].transform(
            lambda x: x.fillna(x.median())
        )

    return df


def group_dog_records(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Group records by dog name, ensuring each dog has up to 5 historical runs.
    """
    dog_records = {}

    for dog_name in df["dog_name"].unique():
        if pd.isna(dog_name):
            continue

        dog_data = df[df["dog_name"] == dog_name].copy()

        # Sort by date (most recent first) and take up to 5 records
        dog_data = dog_data.sort_values("race_date", ascending=False).head(5)

        # Reset index for clean per-dog DataFrame
        dog_data = dog_data.reset_index(drop=True)

        dog_records[dog_name] = dog_data

        print(f"Dog: {dog_name} - {len(dog_data)} historical runs")

    return dog_records


def main():
    """
    Main function to process the upcoming race CSV data.
    """
    print("=== Step 1: Loading and parsing CSV data ===")

    # Try the reference file first
    file_path = "reference_csv_file_20250804_104845.csv"

    try:
        # Load and parse the CSV
        df = load_and_parse_csv(file_path)
        print(f"Loaded {len(df)} rows from {file_path}")
        print(f"Columns: {list(df.columns)}")

        print("\n=== Step 2: Normalizing column names ===")
        df = normalize_column_names(df)
        print(f"Normalized columns: {list(df.columns)}")

        print("\n=== Step 3: Validating data types ===")
        df = validate_and_convert_data_types(df)

        # Show data type info
        print("Data types after conversion:")
        print(df.dtypes)

        print("\n=== Step 4: Handling missing values ===")
        df = handle_missing_values(df)

        # Show missing value counts
        print("Missing values per column:")
        print(df.isnull().sum())

        print("\n=== Step 5: Grouping records by dog ===")
        dog_records = group_dog_records(df)

        print(f"\nProcessed {len(dog_records)} dogs with historical data")

        # Save individual dog records
        print("\n=== Step 6: Saving processed data ===")

        # Save the cleaned complete dataset
        df.to_csv("cleaned_race_data.csv", index=False)
        print("Saved complete cleaned dataset to 'cleaned_race_data.csv'")

        # Save individual dog records
        for dog_name, dog_df in dog_records.items():
            safe_filename = re.sub(r"[^a-zA-Z0-9_-]", "_", dog_name)
            filename = f"dog_records/{safe_filename}_historical_runs.csv"
            dog_df.to_csv(filename, index=False)

        print(f"Saved individual dog records to 'dog_records/' directory")

        # Create summary statistics
        summary_stats = []
        for dog_name, dog_df in dog_records.items():
            stats = {
                "dog_name": dog_name,
                "num_races": len(dog_df),
                "avg_weight": (
                    dog_df["weight_kg"].mean() if "weight_kg" in dog_df else None
                ),
                "avg_race_time": (
                    dog_df["race_time"].mean() if "race_time" in dog_df else None
                ),
                "best_place": dog_df["place"].min() if "place" in dog_df else None,
                "avg_starting_price": (
                    dog_df["starting_price"].mean()
                    if "starting_price" in dog_df
                    else None
                ),
            }
            summary_stats.append(stats)

        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv("dog_summary_statistics.csv", index=False)
        print("Saved summary statistics to 'dog_summary_statistics.csv'")

        print("\n=== Processing Complete! ===")
        print(f"Successfully processed {len(dog_records)} dogs")
        print("Files created:")
        print("- cleaned_race_data.csv (complete dataset)")
        print("- dog_records/*.csv (individual dog files)")
        print("- dog_summary_statistics.csv (summary stats)")

    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Create directory for individual dog records
    import os

    os.makedirs("dog_records", exist_ok=True)

    main()
