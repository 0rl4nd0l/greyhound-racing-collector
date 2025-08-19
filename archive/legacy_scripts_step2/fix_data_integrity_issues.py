
#!/usr/bin/env python3
"""
Data Integrity Correction Script
================================

This script corrects known data integrity issues in the greyhound racing database.

Author: AI Assistant
Date: 2025-01-28
"""

import sqlite3
import logging
from pathlib import Path

class DataIntegrityFixer:
    """Fixes data integrity issues in the database."""

    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.setup_logging()

    def setup_logging(self):
        """Sets up logging."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("logs/data_fixer.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def connect_db(self):
        """Connects to the SQLite database."""
        return sqlite3.connect(self.db_path)

    def fix_duplicate_races(self):
        """Removes race_metadata records where race_id is NULL."""
        self.logger.info("--- Fixing Duplicate Races (race_id is NULL) ---")
        conn = self.connect_db()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM race_metadata WHERE race_id IS NULL")
            rows_deleted = cursor.rowcount
            conn.commit()
            self.logger.info(f"Deleted {rows_deleted} race_metadata records where race_id was NULL.")
        finally:
            conn.close()

    def fix_invalid_box_numbers(self):
        """Corrects invalid box numbers in dog_race_data."""
        self.logger.info("--- Fixing Invalid Box Numbers ---")
        conn = self.connect_db()
        try:
            cursor = conn.cursor()
            # Set invalid box numbers to NULL for later review
            cursor.execute("UPDATE dog_race_data SET box_number = NULL WHERE box_number < 1 OR box_number > 8")
            rows_updated = cursor.rowcount
            conn.commit()
            self.logger.info(f"Updated {rows_updated} dog_race_data records with invalid box numbers.")
        finally:
            conn.close()

    def run_fixes(self):
        """Runs all data integrity fixes."""
        self.logger.info("===== Starting Data Integrity Fixes =====")
        self.fix_duplicate_races()
        self.fix_invalid_box_numbers()
        self.logger.info("===== Data Integrity Fixes Finished =====")


if __name__ == "__main__":
    fixer = DataIntegrityFixer()
    fixer.run_fixes()

