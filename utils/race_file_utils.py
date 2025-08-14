#!/usr/bin/env python3
"""
Race File Utilities for Greyhound Racing Collector
=================================================

This module provides shared utility functions for race file processing,
consolidating common patterns used across multiple scripts to eliminate
duplicated linear scans and improve performance.

Key Functions:
- File hashing and validation
- Filename parsing and race ID extraction
- Database table management
- Batch file processing utilities
- Pre-filtering and caching logic

Author: AI Assistant
Date: January 2025
Version: 1.0.0 - Initial consolidation of shared utilities
"""

import os
import re
import sqlite3
import hashlib
from pathlib import Path
from typing import Set, Dict, List, Tuple, Optional, Union, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.date_parsing import parse_date_flexible


class RaceFileManager:
    """
    Centralized manager for race file operations with caching and optimization.
    
    This class consolidates common file processing patterns used across
    form_guide_csv_scraper.py, bulk_csv_ingest.py, and other scripts.
    """
    
    def __init__(self, database_path: str = "./databases/greyhound_racing.db"):
        self.database_path = database_path
        self.processed_hashes = set()
        self.collected_races = set()
        self.existing_files = set()
        
        # Ensure database tables exist
        self.ensure_database_tables()
        
        # Load cached data
        self.reload_cache()
    
    def ensure_database_tables(self) -> bool:
        """
        Ensure the processed_race_files table exists with proper indexes.
        
        Returns:
            bool: True if table exists or was created successfully, False otherwise
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Create table if it doesn't exist
            cursor.execute('''
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
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_processed_files_hash ON processed_race_files(file_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_processed_files_race_key ON processed_race_files(race_date, venue, race_no)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_processed_files_file_path ON processed_race_files(file_path)')
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Error ensuring processed_race_files table: {e}")
            return False
    
    def reload_cache(self):
        """Reload all cached data from database and filesystem"""
        self.load_processed_hashes()
        self.load_collected_races()
    
    def load_processed_hashes(self):
        """Load processed file hashes from database for O(1) duplicate detection"""
        self.processed_hashes.clear()
        
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT file_hash FROM processed_race_files")
            self.processed_hashes = set(row[0] for row in cursor.fetchall())
            conn.close()
            
        except Exception as e:
            print(f"❌ Error loading processed hashes: {e}")
            self.processed_hashes = set()
    
    def load_collected_races(self, directories: Optional[List[str]] = None):
        """
        Load all collected races from specified directories to avoid re-processing.
        
        Args:
            directories: List of directories to scan. If None, uses default set.
        """
        if directories is None:
            try:
                from config.paths import DATA_DIR, UPCOMING_RACES_DIR
                data_processed = str(DATA_DIR / "processed")
                default_upcoming = str(UPCOMING_RACES_DIR)
            except Exception:
                # Fallbacks if config is unavailable in certain execution contexts
                data_processed = "./processed"
                default_upcoming = "./upcoming_races"
            directories = [
                "./unprocessed",
                "./form_guides/downloaded",
                "./form_guides/processed",
                "./historical_races",
                data_processed,
                default_upcoming,
            ]
        
        self.collected_races.clear()
        self.existing_files.clear()
        
        total_files = 0
        
        for directory in directories:
            if os.path.exists(directory):
                files = [f for f in os.listdir(directory) if f.endswith('.csv')]
                total_files += len(files)
                
                for filename in files:
                    self.existing_files.add(filename)
                    
                    # Extract race info from filename
                    race_id = self.parse_filename_to_race_id(filename)
                    if race_id:
                        self.collected_races.add(race_id)
        
        print(f"📊 Loaded {len(self.collected_races)} unique races from {total_files} files")
    
    def parse_filename_to_race_id(self, filename: str) -> Optional[Tuple[str, str, str]]:
        """
        Parse filename to extract race_id tuple (date, venue, race_number).
        
        Supports multiple filename patterns:
        1. Race N - VENUE - DD Month YYYY.csv
        2. ???_Race_N_VENUE_DATE.csv  
        3. Race_N_-_VENUE_-_DATE.csv
        4. Race N - VENUE - YYYY-MM-DD.csv
        
        Args:
            filename: The filename to parse
            
        Returns:
            Tuple of (formatted_date, venue, race_number) or None if parsing fails
        """
        patterns = [
            # Pattern 1: Race N - VENUE - DD Month YYYY.csv
            r'Race (\d+) - ([A-Z_]+) - (\d{1,2} \w+ \d{4})\.csv',
            
            # Pattern 2: ???_Race_N_VENUE_DATE.csv  
            r'\w+_Race_(\d+)_([A-Z_]+)_([\d-]+)\.csv',
            
            # Pattern 3: Race_N_-_VENUE_-_DATE.csv
            r'Race_(\d+)_-_([A-Z_]+)_-_([\d_A-Za-z]+)\.csv',
            
            # Pattern 4: Race N - VENUE - YYYY-MM-DD.csv
            r'Race (\d+) - ([A-Z_]+) - (\d{4}-\d{2}-\d{2})\.csv'
        ]
        
        for pattern in patterns:
            match = re.match(pattern, filename)
            if match:
                if len(match.groups()) == 3:
                    race_number, venue, date_str = match.groups()
                else:
                    # Handle different group orders for some patterns
                    groups = match.groups()
                    race_number, venue, date_str = groups[0], groups[1], groups[2]
                
                try:
                    formatted_date = parse_date_flexible(date_str)
                    return (formatted_date, venue, race_number)
                except ValueError:
                    continue
        
        return None
    
    def compute_file_hash(self, file_path: str) -> str:
        """
        Compute SHA-256 hash of a file for duplicate detection.
        
        Args:
            file_path: Path to the file
            
        Returns:
            SHA-256 hash as hex string
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read and update hash in chunks of 4K for memory efficiency
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def is_file_processed(self, file_path: str) -> bool:
        """
        Check if a file has been processed using cached hash lookup.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if file has been processed, False otherwise
        """
        try:
            file_hash = self.compute_file_hash(file_path)
            return file_hash in self.processed_hashes
        except Exception:
            return False
    
    def mark_file_processed(self, file_path: str, race_info: Optional[Tuple[str, str, str]] = None, 
                          status: str = 'processed', error_message: Optional[str] = None):
        """
        Mark a file as processed in the database.
        
        Args:
            file_path: Path to the processed file
            race_info: Optional tuple of (race_date, venue, race_no)
            status: Processing status ('processed', 'failed', etc.)
            error_message: Optional error message if processing failed
        """
        try:
            file_hash = self.compute_file_hash(file_path)
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            if race_info:
                race_date, venue, race_no = race_info
                cursor.execute("""
                    INSERT OR REPLACE INTO processed_race_files 
                    (file_hash, race_date, venue, race_no, file_path, file_size, status, error_message) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (file_hash, race_date, venue, race_no, file_path, file_size, status, error_message))
            else:
                cursor.execute("""
                    INSERT OR REPLACE INTO processed_race_files 
                    (file_hash, race_date, venue, race_no, file_path, file_size, status, error_message) 
                    VALUES (?, 'unknown', 'unknown', 0, ?, ?, ?, ?)
                """, (file_hash, file_path, file_size, status, error_message))
                
            conn.commit()
            conn.close()
            
            # Update cache
            self.processed_hashes.add(file_hash)
            
        except Exception as e:
            print(f"❌ Error marking file as processed: {e}")
    
    def get_processed_filenames(self, directory: str = "") -> Set[str]:
        """
        Get set of processed filenames from specified directory for O(1) membership tests.
        
        Args:
            directory: Directory path to filter file_paths by. Empty string returns all.
            
        Returns:
            Set of filenames (without directory path) for O(1) membership tests
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            if directory:
                # Normalize the directory path
                directory = os.path.normpath(directory)
                cursor.execute(
                    "SELECT file_path FROM processed_race_files WHERE file_path LIKE ?",
                    (f"{directory}%",)
                )
            else:
                cursor.execute("SELECT file_path FROM processed_race_files")
            
            results = cursor.fetchall()
            conn.close()
            
            # Extract just the filename and return as set for O(1) lookups
            filenames = set()
            for (file_path,) in results:
                filename = os.path.basename(file_path)
                filenames.add(filename)
            
            return filenames
            
        except Exception as e:
            print(f"❌ Error getting processed filenames: {e}")
            return set()
    
    def batch_filter_unprocessed_files(self, file_paths: List[str]) -> List[str]:
        """
        Filter out already processed files from a list using batch processing.
        
        This method is optimized for large file lists by:
        1. Single database query to get all processed files
        2. O(1) hash-based filtering
        3. Avoiding per-file database queries
        
        Args:
            file_paths: List of file paths to filter
            
        Returns:
            List of unprocessed file paths
        """
        if not file_paths:
            return []
        
        # Get all processed filenames in one query
        processed_filenames = self.get_processed_filenames("")
        
        # Filter using O(1) set membership
        unprocessed_files = []
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            if filename not in processed_filenames:
                unprocessed_files.append(file_path)
        
        return unprocessed_files
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the processed files cache.
        
        Returns:
            Dictionary containing cache statistics
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            stats = {}
            
            # Total processed files
            cursor.execute("SELECT COUNT(*) FROM processed_race_files")
            stats['total_processed'] = cursor.fetchone()[0]
            
            # Status breakdown
            cursor.execute("SELECT status, COUNT(*) FROM processed_race_files GROUP BY status")
            stats['status_counts'] = dict(cursor.fetchall())
            
            # Unique venues
            cursor.execute("SELECT COUNT(DISTINCT venue) FROM processed_race_files WHERE venue != 'unknown'")
            stats['unique_venues'] = cursor.fetchone()[0]
            
            # Date range
            cursor.execute("SELECT MIN(processed_at), MAX(processed_at) FROM processed_race_files")
            date_range = cursor.fetchone()
            stats['date_range'] = {
                'earliest': date_range[0],
                'latest': date_range[1]
            }
            
            # File size statistics
            cursor.execute("SELECT AVG(file_size), MIN(file_size), MAX(file_size) FROM processed_race_files WHERE file_size > 0")
            size_stats = cursor.fetchone()
            if size_stats[0] is not None:
                stats['file_size_stats'] = {
                    'average_bytes': int(size_stats[0]),
                    'min_bytes': size_stats[1],
                    'max_bytes': size_stats[2]
                }
            
            # Memory cache stats
            stats['memory_cache'] = {
                'processed_hashes': len(self.processed_hashes),
                'collected_races': len(self.collected_races),
                'existing_files': len(self.existing_files)
            }
            
            conn.close()
            return stats
            
        except Exception as e:
            print(f"❌ Error getting cache stats: {e}")
            return {}


# Standalone utility functions for backward compatibility and convenience

def compute_file_hash(file_path: str) -> str:
    """
    Standalone function to compute SHA-256 hash of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        SHA-256 hash as hex string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def parse_filename_to_race_id(filename: str) -> Optional[Tuple[str, str, str]]:
    """
    Standalone function to parse filename and extract race ID.
    
    Args:
        filename: The filename to parse
        
    Returns:
        Tuple of (formatted_date, venue, race_number) or None if parsing fails
    """
    manager = RaceFileManager()
    return manager.parse_filename_to_race_id(filename)


def batch_filter_unprocessed_files(file_paths: List[str], 
                                 database_path: str = "./databases/greyhound_racing.db") -> List[str]:
    """
    Standalone function to filter unprocessed files from a list.
    
    Args:
        file_paths: List of file paths to filter
        database_path: Path to the SQLite database
        
    Returns:
        List of unprocessed file paths
    """
    manager = RaceFileManager(database_path)
    return manager.batch_filter_unprocessed_files(file_paths)


def get_race_file_stats(database_path: str = "./databases/greyhound_racing.db") -> Dict[str, Any]:
    """
    Standalone function to get race file processing statistics.
    
    Args:
        database_path: Path to the SQLite database
        
    Returns:
        Dictionary containing comprehensive statistics
    """
    manager = RaceFileManager(database_path)
    return manager.get_cache_stats()


if __name__ == "__main__":
    # Example usage and testing
    print("🧪 Testing race file utilities...")
    
    # Initialize manager
    manager = RaceFileManager()
    print(f"✅ RaceFileManager initialized")
    
    # Test filename parsing
    test_filenames = [
        "Race 1 - SAN - 15 January 2025.csv",
        "Form_Race_2_MEA_2025-01-15.csv", 
        "Race_3_-_DAPT_-_2025_01_15.csv",
        "Race 4 - WENTWORTH_PARK - 2025-01-15.csv"
    ]
    
    print(f"\n📋 Testing filename parsing:")
    for filename in test_filenames:
        race_id = manager.parse_filename_to_race_id(filename)
        if race_id:
            date, venue, race_num = race_id
            print(f"   ✅ {filename} -> {date}, {venue}, Race {race_num}")
        else:
            print(f"   ❌ {filename} -> Parse failed")
    
    # Test cache statistics
    stats = manager.get_cache_stats()
    if stats:
        print(f"\n📊 Cache Statistics:")
        print(f"   Total processed: {stats.get('total_processed', 0)}")
        print(f"   Unique venues: {stats.get('unique_venues', 0)}")
        memory_stats = stats.get('memory_cache', {})
        print(f"   Memory cache: {memory_stats.get('processed_hashes', 0)} hashes, "
              f"{memory_stats.get('collected_races', 0)} races")
    
    print(f"\n🎉 Testing complete!")
