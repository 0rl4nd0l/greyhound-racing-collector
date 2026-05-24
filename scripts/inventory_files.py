import csv
import hashlib
import os
from datetime import datetime

# Directories to scan
dirs_to_scan = [
    "form_guides/downloaded",
    "upcoming_races",
    "processed",
    "archive/corrupt_historical_race_data",
    "historical_races",
]

# Output CSV file
output_file = f'audit/inventory_{datetime.now().strftime("%Y%m%d")}.csv'


# Function to get file details
def get_file_info(file_path):
    file_info = {}
    file_info["path"] = file_path
    file_info["size"] = os.path.getsize(file_path)
    file_info["modified_date"] = datetime.fromtimestamp(
        os.path.getmtime(file_path)
    ).strftime("%Y-%m-%d %H:%M:%S")

    # Calculate SHA-256
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    file_info["sha256"] = sha256_hash.hexdigest()

    return file_info


# Scan directories
all_files = []
for directory in dirs_to_scan:
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(get_file_info(file_path))

# Write results to CSV
with open(output_file, "w", newline="") as csvfile:
    fieldnames = ["path", "size", "modified_date", "sha256"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for file_info in all_files:
        writer.writerow(file_info)

print(f"Inventory written to {output_file}")
