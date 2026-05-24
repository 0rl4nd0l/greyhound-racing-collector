import glob
from typing import List

import pandas as pd


def process_csv_files(
    logs_csv: str = "race_logs_normalized.csv",
    input_glob: str = "./processed/step6_cleanup/*.csv",
    output_csv: str = "discrepancies_report.csv",
) -> None:
    """Process and compare race CSV files against logs to find discrepancies.

    This function is safe to import (no side effects) and only executes when
    called directly. Column assumptions: 'race_id' and 'DATE' exist in inputs.
    """
    # Load the race logs CSV lazily here to avoid import-time I/O
    logs_df = pd.read_csv(logs_csv)

    # Glob for all CSV files in the provided directory
    csv_files: List[str] = glob.glob(input_glob)

    discrepancies = []

    for file_path in csv_files:
        # Load each CSV file
        csv_df = pd.read_csv(file_path)

        # Extract race ID and timestamp columns from the CSV; assumed to be 'race_id' and 'DATE'
        race_id_col = "race_id"
        timestamp_col = "DATE"

        # Perform left joins to identify missing log entries
        merged_df = pd.merge(
            csv_df,
            logs_df,
            left_on=[race_id_col, timestamp_col],
            right_on=[race_id_col, timestamp_col],
            how="left",
            indicator=True,
        )

        # Identify missing log entries, log-only races, and conflicting fields
        missing_logs_df = merged_df[merged_df["_merge"] == "left_only"]
        log_only_races_df = merged_df[merged_df["_merge"] == "right_only"]
        conflicting_fields = merged_df[merged_df["_merge"] == "both"]

        # Collect discrepancies
        discrepancies.append(missing_logs_df)
        discrepancies.append(log_only_races_df)
        discrepancies.append(conflicting_fields)

    if discrepancies:
        # Concatenate all discrepancies into one DataFrame
        discrepancies_df = pd.concat(discrepancies, ignore_index=True)
        # Save the discrepancies to a CSV file for reporting
        discrepancies_df.to_csv(output_csv, index=False)
    else:
        # Create an empty report for consistency if no files found
        pd.DataFrame().to_csv(output_csv, index=False)


if __name__ == "__main__":
    process_csv_files()
