import pandas as pd
import glob

# Load the race logs CSV
logs_df = pd.read_csv('race_logs_normalized.csv')

# Function to process and compare CSV files
def process_csv_files():
    # Glob for all CSV files in the provided directory
    csv_files = glob.glob('./processed/step6_cleanup/*.csv')

    discrepancies = []

    for file_path in csv_files:
        # Load each CSV file
        csv_df = pd.read_csv(file_path)
        
        # Extract race ID and timestamp columns from the CSV; assumed to be 'race_id' and 'DATE'
        race_id_col = 'race_id'
        timestamp_col = 'DATE'

        # Perform left joins to identify missing log entries
        merged_df = pd.merge(csv_df, logs_df, left_on=[race_id_col, timestamp_col], right_on=[race_id_col, timestamp_col], how='left', indicator=True)

        # Identify missing log entries, log-only races, and conflicting fields
        missing_logs_df = merged_df[merged_df['_merge'] == 'left_only']
        log_only_races_df = merged_df[merged_df['_merge'] == 'right_only']
        conflicting_fields = merged_df[merged_df['_merge'] == 'both']

        # Collect discrepancies
        discrepancies.append(missing_logs_df)
        discrepancies.append(log_only_races_df)
        discrepancies.append(conflicting_fields)

    # Concatenate all discrepancies into one DataFrame
    discrepancies_df = pd.concat(discrepancies, ignore_index=True)
    
    # Save the discrepancies to a CSV file for reporting
    discrepancies_df.to_csv('discrepancies_report.csv', index=False)

# Execute the processing function
process_csv_files()
