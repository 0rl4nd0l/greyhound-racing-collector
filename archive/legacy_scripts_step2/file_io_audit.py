
import os
import hashlib
import json
import pandas as pd
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("logs/file_io_audit.log"),
                        logging.StreamHandler()
                    ])

def get_file_checksum(file_path):
    """Calculates the SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def audit_model_and_prediction_files():
    """Audits model and prediction files."""
    logging.info("Starting audit of model and prediction files.")
    model_files = []
    for root, _, files in os.walk("."):
        if "venv" in root:
            continue
        for file in files:
            if file.endswith((".pkl", ".joblib", ".h5")):
                model_files.append(os.path.join(root, file))

    prediction_files = []
    for root, _, files in os.walk("./predictions"):
        for file in files:
            if file.endswith((".csv", ".json")):
                prediction_files.append(os.path.join(root, file))

    for file_path in model_files + prediction_files:
        try:
            file_size = os.path.getsize(file_path)
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
            checksum = get_file_checksum(file_path)
            logging.info(f"File: {file_path}, Size: {file_size}, Mod-Time: {mod_time}, Checksum: {checksum}")
        except Exception as e:
            logging.error(f"Could not process file {file_path}: {e}")

def validate_race_csv(file_path):
    """Validates the columns of a race CSV."""
    required_columns = {
        'Track', 'RaceNum', 'DogName', 'TrainerName', 'Box', 'Weight', 'StartPrice',
        'Handy', 'Margin1', 'Margin2', 'AdjTime', 'Comment', 'RaceDate'
    }
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        if not required_columns.issubset(df.columns):
            logging.warning(f"File {file_path} is missing required columns. Moving to archive.")
            return False
        return True
    except Exception as e:
        logging.error(f"Could not read or validate {file_path}: {e}. Moving to archive.")
        return False

def audit_race_csv_files():
    """Audits upcoming and historical race CSV files."""
    logging.info("Starting audit of race CSV files.")
    race_files = []
    for root, _, files in os.walk("."):
        if "venv" in root or "archive" in root:
            continue
        for file in files:
            if "race" in file.lower() and file.endswith(".csv"):
                race_files.append(os.path.join(root, file))

    archive_dir = "archive/corrupt_or_legacy_race_files"
    os.makedirs(archive_dir, exist_ok=True)

    for file_path in race_files:
        if not validate_race_csv(file_path):
            try:
                # To avoid overwriting files, we'll add a timestamp to the archived file name
                new_path = os.path.join(archive_dir, f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.path.basename(file_path)}")
                os.rename(file_path, new_path)
                logging.info(f"Moved {file_path} to {new_path}")
            except Exception as e:
                logging.error(f"Could not move file {file_path} to archive: {e}")

if __name__ == "__main__":
    audit_model_and_prediction_files()
    audit_race_csv_files()
    logging.info("File I/O integrity audit finished.")

