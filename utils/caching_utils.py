#!/usr/bin/env python3
"""
Caching Utilities for Greyhound Racing Collector
===============================================

This module provides utility functions for the caching layer,
including fast lookups for processed files.

Author: AI Assistant
Date: January 2025
Version: 1.0.0 - Initial implementation with get_processed_filenames API
"""

import os
import sqlite3
from typing import Set


def get_processed_filenames(
    directory: str, database_path: str = "./databases/greyhound_racing.db"
) -> Set[str]:
    """Get set of processed filenames from specified directory for O(1) membership tests.

    This function performs a single SQL query to the processed_race_files table
    and returns a Python set of filenames for fast O(1) membership tests.

    Args:
        directory (str): Directory path to filter file_paths by
        database_path (str): Path to the SQLite database

    Returns:
        Set[str]: Set of filenames (without directory path) for O(1) membership tests

    Example:
        >>> processed_files = get_processed_filenames("./unprocessed")
        >>> if "Race 1 - SAN - 2025-01-15.csv" in processed_files:
        ...     print("File already processed")
    """
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()

        # Ensure the file_path index exists for performance
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_processed_files_file_path ON processed_race_files(file_path)"
        )

        # Single SQL query to get file_paths from processed_race_files
        # Filter by directory if provided
        if directory:
            # Normalize the directory path
            directory = os.path.normpath(directory)
            # Use LIKE pattern to match files in the specified directory
            cursor.execute(
                "SELECT file_path FROM processed_race_files WHERE file_path LIKE ?",
                (f"{directory}%",),
            )
        else:
            # Get all processed file paths
            cursor.execute("SELECT file_path FROM processed_race_files")

        results = cursor.fetchall()
        conn.close()

        # Extract just the filename (strip directory) and return as set for O(1) lookups
        filenames = set()
        for (file_path,) in results:
            filename = os.path.basename(file_path)
            filenames.add(filename)

        return filenames

    except Exception as e:
        print(f"❌ Error getting processed filenames: {e}")
        return set()


def ensure_processed_files_table(
    database_path: str = "./databases/greyhound_racing.db",
) -> bool:
    """Ensure the processed_race_files table exists with proper indexes.

    Args:
        database_path (str): Path to the SQLite database

    Returns:
        bool: True if table exists or was created successfully, False otherwise
    """
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()

        # Create table if it doesn't exist
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_race_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash TEXT UNIQUE NOT NULL,
                race_date DATE NOT NULL,
                venue TEXT NOT NULL, 
                race_no INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER,
                processed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'processed',
                error_message TEXT
            )
        """
        )

        # Create indexes for performance
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_processed_files_hash ON processed_race_files(file_hash)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_processed_files_race_key ON processed_race_files(race_date, venue, race_no)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_processed_files_file_path ON processed_race_files(file_path)"
        )

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        print(f"❌ Error ensuring processed_race_files table: {e}")
        return False


def get_cache_stats(database_path: str = "./databases/greyhound_racing.db") -> dict:
    """Get statistics about the processed files cache.

    Args:
        database_path (str): Path to the SQLite database

    Returns:
        dict: Dictionary containing cache statistics
    """
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()

        stats = {}

        # Total processed files
        cursor.execute("SELECT COUNT(*) FROM processed_race_files")
        stats["total_processed"] = cursor.fetchone()[0]

        # Status breakdown
        cursor.execute(
            "SELECT status, COUNT(*) FROM processed_race_files GROUP BY status"
        )
        stats["status_counts"] = dict(cursor.fetchall())

        # Unique venues
        cursor.execute(
            "SELECT COUNT(DISTINCT venue) FROM processed_race_files WHERE venue != 'unknown'"
        )
        stats["unique_venues"] = cursor.fetchone()[0]

        # Date range
        cursor.execute(
            "SELECT MIN(processed_at), MAX(processed_at) FROM processed_race_files"
        )
        date_range = cursor.fetchone()
        stats["date_range"] = {"earliest": date_range[0], "latest": date_range[1]}

        conn.close()
        return stats

    except Exception as e:
        print(f"❌ Error getting cache stats: {e}")
        return {}


if __name__ == "__main__":
    # Example usage and testing
    print("🧪 Testing caching utilities...")

    # Test table creation
    if ensure_processed_files_table():
        print("✅ Database table ensured")
    else:
        print("❌ Failed to ensure database table")
        exit(1)

    # Test getting processed filenames for different directories
    test_directories = ["./unprocessed", "./processed", "./form_guides/downloaded"]

    for directory in test_directories:
        processed_files = get_processed_filenames(directory)
        print(f"📂 {directory}: {len(processed_files)} processed files")

    # Test getting all processed filenames
    all_processed = get_processed_filenames("")
    print(f"📊 Total processed files: {len(all_processed)}")

    # Test cache statistics
    stats = get_cache_stats()
    if stats:
        print(f"📈 Cache Statistics:")
        print(f"   Total: {stats.get('total_processed', 0)}")
        print(f"   Unique venues: {stats.get('unique_venues', 0)}")
        for status, count in stats.get("status_counts", {}).items():
            print(f"   {status.capitalize()}: {count}")

    print("🎉 Testing complete!")
