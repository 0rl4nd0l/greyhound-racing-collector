
import os
import hashlib
import pandas as pd
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("logs/file_io_audit_v2.log"),
                        logging.StreamHandler()
                    ])

def get_file_checksum(file_path):
    """Calculates the SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def audit_files(file_paths, file_type):
    """Audits a list of files, logging their checksum, size, and mod-time."""
    logging.info(f"Auditing {len(file_paths)} {file_type} files.")
    for file_path in file_paths:
        try:
            file_size = os.path.getsize(file_path)
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
            checksum = get_file_checksum(file_path)
            logging.info(f"File: {file_path}, Size: {file_size}, Mod-Time: {mod_time}, Checksum: {checksum}")
        except Exception as e:
            logging.error(f"Could not process file {file_path}: {e}")

def validate_and_archive_csvs(directory, required_columns, archive_dir):
    """Validates CSVs in a directory and moves invalid ones to an archive."""
    os.makedirs(archive_dir, exist_ok=True)
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path, nrows=1)  # Read only the header
                    if not required_columns.issubset(df.columns):
                        logging.warning(f"File {file_path} is missing required columns. Moving to archive.")
                        new_path = os.path.join(archive_dir, f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.path.basename(file_path)}")
                        os.rename(file_path, new_path)
                        logging.info(f"Moved {file_path} to {new_path}")
                except Exception as e:
                    logging.error(f"Could not read or validate {file_path}: {e}. Moving to archive.")
                    new_path = os.path.join(archive_dir, f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.path.basename(file_path)}")
                    os.rename(file_path, new_path)
                    logging.info(f"Moved {file_path} to {new_path}")

if __name__ == "__main__":
    # 1. Enumerate and audit model files
    model_files = [os.path.join(root, file) for root, _, files in os.walk(".") if "venv" not in root for file in files if file.endswith((".pkl", ".joblib", ".h5"))]
    audit_files(model_files, "Model")

    # 2. Enumerate and audit prediction files
    prediction_files = [os.path.join(root, file) for root, _, files in os.walk("./predictions") for file in files if file.endswith((".json", ".csv"))]
    audit_files(prediction_files, "Prediction")

    # 3. Validate historical race data (assuming they are in 'processed_races' or similar)
    historical_race_dirs = ["processed_races", "form_guides", "historical_races"]
    historical_required_cols = {'Track', 'RaceNum', 'DogName', 'TrainerName', 'Box', 'Weight', 'StartPrice', 'Handy', 'Margin1', 'Margin2', 'AdjTime', 'Comment', 'RaceDate'}
    for directory in historical_race_dirs:
        if os.path.isdir(directory):
            logging.info(f"Validating historical CSVs in {directory}")
            validate_and_archive_csvs(directory, historical_required_cols, "archive/corrupt_historical_race_data")

    # Upcoming races are raw and don't need schema validation, but we can log them.
    upcoming_races_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk("./upcoming_races") for f in filenames if f.endswith(".csv")]
    audit_files(upcoming_races_files, "Upcoming Race")

    logging.info("File I/O integrity audit finished.")

