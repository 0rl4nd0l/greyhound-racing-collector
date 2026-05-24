#!/usr/bin/env python3
"""
File Modification Time Heuristic Utility
========================================

Provides optimized file scanning using modification time heuristics to skip
files that have already been processed in previous runs. This significantly
improves performance when scanning large directories.

Features:
- Store and retrieve last processed mtime from database meta table
- Filter files using os.scandir with mtime comparison
- CLI flag support for strict scanning (disable heuristic)
- Automatic mtime tracking and updates

Author: AI Assistant
Date: 2025-01-04
"""

import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from logger import logger


@dataclass
class FileEntry:
    """Represents a file entry with metadata"""

    path: str
    name: str
    mtime: float
    size: int
    is_csv: bool


class MtimeHeuristic:
    """
    Manages file modification time heuristics for optimized scanning.

    This class provides functionality to:
    1. Store and retrieve last processed mtime from database
    2. Filter files based on modification time
    3. Support strict scanning mode that bypasses heuristics
    """

    def __init__(self, db_path: str = "database.sqlite"):
        """
        Initialize the mtime heuristic manager.

        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = db_path
        self.meta_key = "last_processed_mtime"
        self._ensure_meta_table()

    def _ensure_meta_table(self):
        """Ensure the db_meta table exists"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create table if it doesn't exist
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS db_meta (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    meta_key VARCHAR(255) UNIQUE NOT NULL,
                    meta_value VARCHAR(500),
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create index if it doesn't exist
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_db_meta_key ON db_meta(meta_key)
            """
            )

            conn.commit()
            conn.close()
            logger.debug("Database meta table initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize meta table: {e}")
            raise

    def get_last_processed_mtime(self) -> Optional[float]:
        """
        Get the last processed mtime from database.

        Returns:
            Last processed mtime as timestamp, or None if not set
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT meta_value FROM db_meta WHERE meta_key = ?", (self.meta_key,)
            )

            result = cursor.fetchone()
            conn.close()

            if result and result[0]:
                return float(result[0])
            return None

        except Exception as e:
            logger.error(f"Failed to get last processed mtime: {e}")
            return None

    def set_last_processed_mtime(self, mtime: float):
        """
        Set the last processed mtime in database.

        Args:
            mtime: Modification time to store as timestamp
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Use INSERT OR REPLACE to handle both insert and update
            cursor.execute(
                """
                INSERT OR REPLACE INTO db_meta (meta_key, meta_value, last_updated)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
                (self.meta_key, str(mtime)),
            )

            conn.commit()
            conn.close()

            formatted_time = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            logger.debug(f"Updated last processed mtime to {mtime} ({formatted_time})")

        except Exception as e:
            logger.error(f"Failed to set last processed mtime: {e}")
            raise

    def scan_directory_optimized(
        self,
        directory: str,
        strict_scan: bool = False,
        file_extensions: Optional[List[str]] = None,
    ) -> Generator[FileEntry, None, None]:
        """
        Scan directory with mtime optimization.

        Args:
            directory: Directory path to scan
            strict_scan: If True, disable mtime heuristic and scan all files
            file_extensions: List of extensions to filter (e.g., ['.csv'])

        Yields:
            FileEntry objects for files that need processing
        """
        if not os.path.exists(directory):
            logger.warning(f"Directory does not exist: {directory}")
            return

        if file_extensions is None:
            file_extensions = [".csv"]

        # Get last processed mtime for filtering
        last_processed_mtime = None
        if not strict_scan:
            last_processed_mtime = self.get_last_processed_mtime()
            if last_processed_mtime:
                formatted_time = datetime.fromtimestamp(last_processed_mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                logger.info(
                    f"Using mtime heuristic - filtering files older than {formatted_time}"
                )

        scanned_count = 0
        filtered_count = 0
        yielded_count = 0

        try:
            # Use os.scandir for efficient directory scanning
            with os.scandir(directory) as entries:
                for entry in entries:
                    if not entry.is_file():
                        continue

                    scanned_count += 1

                    # Check file extension
                    file_ext = Path(entry.name).suffix.lower()
                    if file_extensions and file_ext not in file_extensions:
                        continue

                    # Get file stats efficiently
                    stat_result = entry.stat()
                    file_mtime = stat_result.st_mtime

                    # Apply mtime heuristic if not in strict mode
                    if not strict_scan and last_processed_mtime is not None:
                        if file_mtime <= last_processed_mtime:
                            filtered_count += 1
                            continue

                    # Create FileEntry for files that need processing
                    file_entry = FileEntry(
                        path=entry.path,
                        name=entry.name,
                        mtime=file_mtime,
                        size=stat_result.st_size,
                        is_csv=(file_ext == ".csv"),
                    )

                    yielded_count += 1
                    yield file_entry

        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")
            raise

        # Log scan statistics
        mode = "strict" if strict_scan else "optimized"
        logger.info(f"Directory scan completed ({mode} mode)")
        logger.info(f"  Total files scanned: {scanned_count}")
        logger.info(f"  Files filtered by mtime: {filtered_count}")
        logger.info(f"  Files requiring processing: {yielded_count}")

    def update_processed_mtime_from_files(self, processed_files: List[str]):
        """
        Update last processed mtime based on the maximum mtime of processed files.

        Args:
            processed_files: List of file paths that were processed
        """
        if not processed_files:
            logger.debug("No files to update mtime from")
            return

        max_mtime = 0.0

        try:
            for file_path in processed_files:
                if os.path.exists(file_path):
                    file_mtime = os.path.getmtime(file_path)
                    max_mtime = max(max_mtime, file_mtime)

            if max_mtime > 0:
                # Only update if we found a newer mtime
                current_mtime = self.get_last_processed_mtime()
                if current_mtime is None or max_mtime > current_mtime:
                    self.set_last_processed_mtime(max_mtime)
                    formatted_time = datetime.fromtimestamp(max_mtime).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    logger.info(f"Updated last processed mtime to {formatted_time}")
                else:
                    logger.debug(
                        "No mtime update needed - processed files are not newer"
                    )
            else:
                logger.warning("No valid mtimes found in processed files")

        except Exception as e:
            logger.error(f"Failed to update processed mtime from files: {e}")

    def get_scan_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the mtime heuristic usage.

        Returns:
            Dictionary with statistics
        """
        try:
            last_mtime = self.get_last_processed_mtime()

            return {
                "last_processed_mtime": last_mtime,
                "last_processed_datetime": (
                    datetime.fromtimestamp(last_mtime).isoformat()
                    if last_mtime
                    else None
                ),
                "heuristic_enabled": last_mtime is not None,
                "database_path": self.db_path,
            }

        except Exception as e:
            logger.error(f"Failed to get scan statistics: {e}")
            return {"error": str(e)}

    def reset_mtime_heuristic(self):
        """
        Reset the mtime heuristic by removing the stored value.
        This forces a full scan on the next processing run.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("DELETE FROM db_meta WHERE meta_key = ?", (self.meta_key,))

            conn.commit()
            conn.close()

            logger.info("Mtime heuristic reset - next scan will be a full scan")

        except Exception as e:
            logger.error(f"Failed to reset mtime heuristic: {e}")
            raise


def create_mtime_heuristic(db_path: str = "database.sqlite") -> MtimeHeuristic:
    """
    Factory function to create a MtimeHeuristic instance.

    Args:
        db_path: Path to SQLite database

    Returns:
        Configured MtimeHeuristic instance
    """
    return MtimeHeuristic(db_path=db_path)


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test mtime heuristic functionality")
    parser.add_argument("directory", help="Directory to scan")
    parser.add_argument(
        "--strict-scan",
        action="store_true",
        help="Disable mtime heuristic for full scan",
    )
    parser.add_argument("--reset", action="store_true", help="Reset mtime heuristic")

    args = parser.parse_args()

    # Create heuristic instance
    heuristic = create_mtime_heuristic()

    if args.reset:
        heuristic.reset_mtime_heuristic()
        print("Mtime heuristic reset successfully")
        exit(0)

    # Scan directory
    print(f"Scanning directory: {args.directory}")
    print(f"Strict scan mode: {args.strict_scan}")

    files_found = []
    for file_entry in heuristic.scan_directory_optimized(
        args.directory, strict_scan=args.strict_scan
    ):
        files_found.append(file_entry.path)
        print(
            f"  {file_entry.name} (mtime: {datetime.fromtimestamp(file_entry.mtime)})"
        )

    print(f"\nFound {len(files_found)} files requiring processing")

    # Show statistics
    stats = heuristic.get_scan_statistics()
    print(f"\nScan statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
