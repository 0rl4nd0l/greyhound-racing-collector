#!/usr/bin/env python3
"""
Step 7: Produce ranked output file
Creates a table with Dog, Rank, WinProbability% columns
Rounds percentages to one decimal place and rescales to sum 100%
Outputs JSON, CSV, and console printout for downstream use
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd


def load_calibration_data():
    """Load the calibration results data"""
    data_file = "step6_calibration_results_20250804_140736.csv"

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file {data_file} not found")

    # Read the CSV file
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} records from {data_file}")

    return df


def process_win_probabilities(df):
    """Process and rank dogs by win probability"""

    # Extract relevant columns
    dogs_data = df[["dog_clean_name", "predicted_probability"]].copy()

    # Remove duplicates (keep first occurrence)
    dogs_data = dogs_data.drop_duplicates(subset=["dog_clean_name"])

    # Remove any rows with null/nan values
    dogs_data = dogs_data.dropna()

    print(f"Processing {len(dogs_data)} unique dogs")

    # Convert probabilities to percentages
    dogs_data["win_probability_pct"] = dogs_data["predicted_probability"] * 100

    # Sort by probability descending (highest probability = rank 1)
    dogs_data = dogs_data.sort_values("win_probability_pct", ascending=False)

    # Add rank (1-based)
    dogs_data["rank"] = range(1, len(dogs_data) + 1)

    # Round percentages to one decimal place
    dogs_data["win_probability_pct"] = dogs_data["win_probability_pct"].round(1)

    # Rescale to sum to 100%
    total_probability = dogs_data["win_probability_pct"].sum()
    if total_probability > 0:
        dogs_data["win_probability_pct_rescaled"] = (
            dogs_data["win_probability_pct"] / total_probability * 100
        ).round(1)

        # Adjust for rounding errors to ensure exact 100% sum
        current_sum = dogs_data["win_probability_pct_rescaled"].sum()
        if current_sum != 100.0:
            # Add the difference to the highest probability dog
            diff = 100.0 - current_sum
            dogs_data.iloc[
                0, dogs_data.columns.get_loc("win_probability_pct_rescaled")
            ] += diff
            dogs_data["win_probability_pct_rescaled"] = dogs_data[
                "win_probability_pct_rescaled"
            ].round(1)
    else:
        dogs_data["win_probability_pct_rescaled"] = dogs_data["win_probability_pct"]

    # Create final output dataframe
    output_df = dogs_data[
        ["dog_clean_name", "rank", "win_probability_pct_rescaled"]
    ].copy()
    output_df.columns = ["Dog", "Rank", "WinProbability%"]

    return output_df


def output_results(df):
    """Output results in multiple formats"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Console printout
    print("\n" + "=" * 60)
    print("STEP 7: RANKED OUTPUT TABLE")
    print("=" * 60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total dogs: {len(df)}")
    print(f"Total probability: {df['WinProbability%'].sum():.1f}%")
    print("-" * 60)

    # Pretty print the table
    print(f"{'Rank':<6} {'Dog':<25} {'WinProbability%':<15}")
    print("-" * 50)
    for _, row in df.iterrows():
        print(f"{row['Rank']:<6} {row['Dog']:<25} {row['WinProbability%']:<15.1f}")

    print("-" * 60)
    print(f"Verification - Sum of probabilities: {df['WinProbability%'].sum():.1f}%")
    print("=" * 60)

    # 2. CSV output
    csv_filename = f"step7_ranked_output_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nCSV file saved: {csv_filename}")

    # 3. JSON output
    json_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_dogs": len(df),
            "total_probability": float(df["WinProbability%"].sum()),
            "description": "Ranked dogs by win probability",
        },
        "rankings": df.to_dict("records"),
    }

    json_filename = f"step7_ranked_output_{timestamp}.json"
    with open(json_filename, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON file saved: {json_filename}")

    # 4. Summary statistics
    print(f"\nSUMMARY STATISTICS:")
    print(
        f"Top 5 dogs account for {df.head(5)['WinProbability%'].sum():.1f}% of total probability"
    )
    print(
        f"Highest probability: {df.iloc[0]['Dog']} ({df.iloc[0]['WinProbability%']:.1f}%)"
    )
    print(
        f"Lowest probability: {df.iloc[-1]['Dog']} ({df.iloc[-1]['WinProbability%']:.1f}%)"
    )
    print(f"Average probability: {df['WinProbability%'].mean():.1f}%")
    print(f"Median probability: {df['WinProbability%'].median():.1f}%")

    return csv_filename, json_filename


def main():
    """Main execution function"""
    try:
        print("Step 7: Producing ranked output file...")

        # Load data
        df = load_calibration_data()

        # Process win probabilities
        ranked_df = process_win_probabilities(df)

        # Output results
        csv_file, json_file = output_results(ranked_df)

        print(f"\n✅ Step 7 completed successfully!")
        print(f"Generated files:")
        print(f"  - {csv_file}")
        print(f"  - {json_file}")
        print(f"\nFiles are ready for downstream use.")

    except Exception as e:
        print(f"❌ Error in Step 7: {e}")
        raise


if __name__ == "__main__":
    main()
