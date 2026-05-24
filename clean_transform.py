import os

import pandas as pd

# Directory to process
input_directory = "./processed/step6_cleanup"


# Processing function
def clean_and_transform(file_path):
    # Read the file to check its structure
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Skip line numbers at the beginning and process CSV content
    csv_lines = []
    for line in lines:
        if "|" in line:
            # Extract content after the line number
            csv_lines.append(line.split("|", 1)[1])
        else:
            csv_lines.append(line)

    # Write temporary file without line numbers
    temp_file = file_path + ".temp"
    with open(temp_file, "w") as f:
        f.writelines(csv_lines)

    # Read the cleaned CSV
    df = pd.read_csv(temp_file)

    # Remove temporary file
    os.remove(temp_file)

    # Drop duplicates
    df = df.drop_duplicates()

    # Check if 'DATE' column exists
    if "DATE" not in df.columns:
        print(f"Missing 'DATE' column in {file_path}")
        return

    # Fill or drop missing values
    df = df.ffill()  # Forward fill as an example

    # Convert date to datetime format
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    # Add new derived columns
    df["day_of_week"] = df["DATE"].dt.day_name()
    df["month"] = df["DATE"].dt.month_name()
    # Create track_code only if TRACK column exists
    if "TRACK" in df.columns:
        df["track_code"] = df["TRACK"].str.extract(r"(\b[A-Z]+\b)", expand=False)
    else:
        print(f"Missing 'TRACK' column in {file_path}")

    # Save cleaned file
    cleaned_file_path = os.path.splitext(file_path)[0] + "_cleaned.csv"
    df.to_csv(cleaned_file_path, index=False)


# Process all files in the directory
for filename in os.listdir(input_directory):
    if filename.endswith(".csv"):
        try:
            print(f"Processing {filename}...")
            clean_and_transform(os.path.join(input_directory, filename))
            print(f"Successfully processed {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
