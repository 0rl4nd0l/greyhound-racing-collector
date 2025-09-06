#!/usr/bin/env python3
"""
Unit Tests for Pre-filter Caching System
========================================

Tests the pre-filter functionality that checks processed files against SQLite
to avoid reprocessing. Creates 1000 mock CSV files, marks 990 as processed,
and verifies only 10 are returned by the pre-filter.

Author: AI Assistant
Date: 2025-01-15
"""

import os
import shutil
import sqlite3
import tempfile
from datetime import datetime, timedelta
from typing import List, Set

import pytest

# Import the modules under test
from utils.caching_utils import ensure_processed_files_table, get_processed_filenames
from utils.early_exit_optimizer import EarlyExitConfig, EarlyExitOptimizer


class TestPrefilterCaching:
    """Test suite for pre-filter caching functionality."""

    @pytest.fixture
    def temp_environment(self):
        """Create a temporary environment for testing."""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="test_prefilter_")

        # Create test subdirectories
        csv_dir = os.path.join(temp_dir, "test_csvs")
        os.makedirs(csv_dir, exist_ok=True)

        # Create temporary database
        db_path = os.path.join(temp_dir, "test_racing.db")

        # Ensure processed_race_files table exists
        assert ensure_processed_files_table(
            db_path
        ), "Failed to create test database table"

        yield {"temp_dir": temp_dir, "csv_dir": csv_dir, "db_path": db_path}

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    def _create_mock_csv_files(self, csv_dir: str, count: int = 1000) -> List[str]:
        """Create mock CSV files for testing.

        Args:
            csv_dir: Directory to create CSV files in
            count: Number of CSV files to create

        Returns:
            List of created file paths
        """
        csv_files = []
        base_date = datetime(2024, 1, 1)

        venues = ["MEA", "SAN", "ALB", "BAL", "HEA", "TAS", "THO", "WAR"]

        for i in range(count):
            # Create varied but realistic filenames
            venue = venues[i % len(venues)]
            race_num = (i % 12) + 1
            date_offset = i % 365
            race_date = base_date + timedelta(days=date_offset)
            date_str = race_date.strftime("%Y-%m-%d")

            filename = f"Race {race_num} - {venue} - {date_str}.csv"
            filepath = os.path.join(csv_dir, filename)

            # Create CSV with realistic content
            content = f"""Dog Name,Sex,Placing,Box,Weight,Distance,Date,Track,Grade,Time,Win Time,Bonus,First Split,Margin,PIR,Starting Price
TestDog{i}_1,D,1,1,30.{i%50 + 10},{500 + (i%3)*20},{date_str},{venue},5,{30.00 + (i%100)/100:.2f},{30.00 + (i%50)/100:.2f},+0.{i%50:02d},{5.0 + (i%30)/100:.2f},0.0,1,${2.50 + (i%100)/100:.2f}
TestDog{i}_2,D,2,2,31.{i%40 + 20},{500 + (i%3)*20},{date_str},{venue},5,{30.20 + (i%80)/100:.2f},{30.00 + (i%50)/100:.2f},+0.{i%60:02d},{5.2 + (i%25)/100:.2f},0.{i%50}/100,2,${3.20 + (i%150)/100:.2f}
TestDog{i}_3,D,3,3,29.{i%60 + 50},{500 + (i%3)*20},{date_str},{venue},5,{30.50 + (i%120)/100:.2f},{30.00 + (i%50)/100:.2f},+0.{i%70:02d},{5.5 + (i%40)/100:.2f},0.{i%100}/100,3,${4.80 + (i%200)/100:.2f}
"""

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

            csv_files.append(filepath)

        return csv_files

    def _mark_files_as_processed(
        self, db_path: str, csv_files: List[str], count_to_mark: int = 990
    ) -> Set[str]:
        """Mark specified number of files as processed in the database.

        Args:
            db_path: Path to SQLite database
            csv_files: List of CSV file paths
            count_to_mark: Number of files to mark as processed

        Returns:
            Set of filenames that were marked as processed
        """
        if count_to_mark > len(csv_files):
            count_to_mark = len(csv_files)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        marked_filenames = set()

        try:
            for i in range(count_to_mark):
                filepath = csv_files[i]
                filename = os.path.basename(filepath)

                # Extract race metadata from filename
                # Format: "Race X - VEN - YYYY-MM-DD.csv"
                parts = filename.replace(".csv", "").split(" - ")
                if len(parts) >= 3:
                    race_no = int(parts[0].replace("Race ", ""))
                    venue = parts[1]
                    race_date = parts[2]
                else:
                    # Fallback for unexpected formats
                    race_no = i % 12 + 1
                    venue = "TEST"
                    race_date = "2024-01-01"

                # Generate a pseudo file hash for testing
                file_hash = f"test_hash_{i:06d}_{filename[:10]}"
                file_size = len(filename) * 100  # Mock file size

                # Insert into processed_race_files table
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO processed_race_files 
                    (file_hash, race_date, venue, race_no, file_path, file_size, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        file_hash,
                        race_date,
                        venue,
                        race_no,
                        filepath,
                        file_size,
                        "processed",
                    ),
                )

                marked_filenames.add(filename)

            conn.commit()
            print(f"✅ Marked {count_to_mark} files as processed in database")

        except Exception as e:
            print(f"❌ Error marking files as processed: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

        return marked_filenames

    def test_create_1000_mock_csv_files(self, temp_environment):
        """Test creating 1000 mock CSV files."""
        csv_dir = temp_environment["csv_dir"]

        # Create 1000 mock CSV files
        csv_files = self._create_mock_csv_files(csv_dir, count=1000)

        # Verify all files were created
        assert len(csv_files) == 1000, f"Expected 1000 files, got {len(csv_files)}"

        # Verify files exist on disk
        existing_files = 0
        for filepath in csv_files:
            if os.path.exists(filepath):
                existing_files += 1

        assert (
            existing_files == 1000
        ), f"Expected 1000 files on disk, found {existing_files}"

        # Verify file content structure
        sample_file = csv_files[0]
        with open(sample_file, "r", encoding="utf-8") as f:
            content = f.read()
            assert "Dog Name,Sex,Placing,Box,Weight" in content, "CSV header not found"
            assert "TestDog" in content, "Test data not found"

        print(f"✅ Successfully created {len(csv_files)} mock CSV files")

    def test_mark_990_files_as_processed(self, temp_environment):
        """Test marking 990 files as processed in SQLite database."""
        csv_dir = temp_environment["csv_dir"]
        db_path = temp_environment["db_path"]

        # Create 1000 mock CSV files
        csv_files = self._create_mock_csv_files(csv_dir, count=1000)

        # Mark 990 files as processed
        marked_filenames = self._mark_files_as_processed(
            db_path, csv_files, count_to_mark=990
        )

        # Verify correct number marked
        assert (
            len(marked_filenames) == 990
        ), f"Expected 990 marked files, got {len(marked_filenames)}"

        # Verify database entries
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT COUNT(*) FROM processed_race_files WHERE status = 'processed'"
        )
        processed_count = cursor.fetchone()[0]
        assert (
            processed_count == 990
        ), f"Expected 990 processed entries, got {processed_count}"

        cursor.execute("SELECT COUNT(DISTINCT file_hash) FROM processed_race_files")
        unique_hashes = cursor.fetchone()[0]
        assert unique_hashes == 990, f"Expected 990 unique hashes, got {unique_hashes}"

        conn.close()
        print(f"✅ Successfully marked {len(marked_filenames)} files as processed")

    def test_prefilter_returns_only_10_unprocessed(self, temp_environment):
        """Test that pre-filter correctly identifies only 10 unprocessed files."""
        csv_dir = temp_environment["csv_dir"]
        db_path = temp_environment["db_path"]

        # Create 1000 mock CSV files
        csv_files = self._create_mock_csv_files(csv_dir, count=1000)

        # Mark 990 files as processed (leaving 10 unprocessed)
        marked_filenames = self._mark_files_as_processed(
            db_path, csv_files, count_to_mark=990
        )

        # Use caching utilities to get processed filenames
        processed_filenames_set = get_processed_filenames(csv_dir, db_path)

        # Verify cached set contains the marked files
        assert (
            len(processed_filenames_set) == 990
        ), f"Expected 990 cached files, got {len(processed_filenames_set)}"

        # Verify all marked files are in the cached set
        missing_from_cache = marked_filenames - processed_filenames_set
        assert (
            len(missing_from_cache) == 0
        ), f"Missing {len(missing_from_cache)} files from cache"

        # Identify unprocessed files by checking which files are NOT in the cache
        all_filenames = {os.path.basename(fp) for fp in csv_files}
        unprocessed_filenames = all_filenames - processed_filenames_set

        # Verify exactly 10 files are unprocessed
        assert (
            len(unprocessed_filenames) == 10
        ), f"Expected 10 unprocessed files, got {len(unprocessed_filenames)}"

        print(
            f"✅ Pre-filter correctly identified {len(unprocessed_filenames)} unprocessed files"
        )
        print(f"   Processed (cached): {len(processed_filenames_set)}")
        print(f"   Unprocessed: {len(unprocessed_filenames)}")

        # Sample of unprocessed files
        sample_unprocessed = list(unprocessed_filenames)[:5]
        print(f"   Sample unprocessed files: {sample_unprocessed}")

    def test_early_exit_optimizer_with_cached_directory(self, temp_environment):
        """Test early exit optimizer correctly identifies mostly cached directory."""
        csv_dir = temp_environment["csv_dir"]
        db_path = temp_environment["db_path"]

        # Create 1000 mock CSV files
        csv_files = self._create_mock_csv_files(csv_dir, count=1000)

        # Mark 990 files as processed (leaving 10 unprocessed)
        self._mark_files_as_processed(db_path, csv_files, count_to_mark=990)

        # Get processed filenames set
        processed_filenames_set = get_processed_filenames(csv_dir, db_path)

        # Initialize early exit optimizer with strict config for testing
        config = EarlyExitConfig(
            cache_ratio_threshold=0.95,  # 95% cached
            unprocessed_threshold=15,  # Max 15 unprocessed files
            enable_early_exit=True,
            verbose_summary=False,  # Quiet for testing
        )
        optimizer = EarlyExitOptimizer(config)

        # Test directory analysis
        scan_result = optimizer.analyze_directory_cache_status(
            csv_dir, processed_filenames_set, file_extensions=[".csv"]
        )

        # Verify scan results
        assert (
            scan_result.total_files == 1000
        ), f"Expected 1000 total files, got {scan_result.total_files}"
        assert (
            scan_result.processed_files == 990
        ), f"Expected 990 processed files, got {scan_result.processed_files}"
        assert (
            scan_result.unprocessed_files == 10
        ), f"Expected 10 unprocessed files, got {scan_result.unprocessed_files}"

        # Verify cache ratio calculation
        expected_ratio = 990 / 1000  # 0.99
        assert (
            abs(scan_result.cache_ratio - expected_ratio) < 0.001
        ), f"Expected cache ratio ~{expected_ratio}, got {scan_result.cache_ratio}"

        # Verify early exit decision
        assert (
            scan_result.should_early_exit
        ), "Early exit should be triggered with 99% cache ratio"

        print("✅ Early exit optimizer correctly analyzed directory:")
        print(f"   Total files: {scan_result.total_files}")
        print(f"   Processed: {scan_result.processed_files}")
        print(f"   Unprocessed: {scan_result.unprocessed_files}")
        print(f"   Cache ratio: {scan_result.cache_ratio:.1%}")
        print(f"   Should early exit: {scan_result.should_early_exit}")

    def test_early_exit_optimizer_get_unprocessed_files(self, temp_environment):
        """Test early exit optimizer can quickly retrieve unprocessed files."""
        csv_dir = temp_environment["csv_dir"]
        db_path = temp_environment["db_path"]

        # Create 1000 mock CSV files
        csv_files = self._create_mock_csv_files(csv_dir, count=1000)

        # Mark 990 files as processed (leaving 10 unprocessed)
        marked_filenames = self._mark_files_as_processed(
            db_path, csv_files, count_to_mark=990
        )

        # Get processed filenames set
        processed_filenames_set = get_processed_filenames(csv_dir, db_path)

        # Initialize early exit optimizer with appropriate threshold
        config = EarlyExitConfig(
            cache_ratio_threshold=0.95,
            unprocessed_threshold=15,  # Allow up to 15 unprocessed files
            enable_early_exit=True,
            verbose_summary=False,
        )
        optimizer = EarlyExitOptimizer(config)

        # Get unprocessed files quickly
        unprocessed_file_paths = optimizer.get_unprocessed_files_fast(
            csv_dir, processed_filenames_set, file_extensions=[".csv"]
        )

        # Verify correct number of unprocessed files
        assert (
            len(unprocessed_file_paths) == 10
        ), f"Expected 10 unprocessed files, got {len(unprocessed_file_paths)}"

        # Verify all returned files exist and are actually unprocessed
        for file_path in unprocessed_file_paths:
            assert os.path.exists(
                file_path
            ), f"Unprocessed file doesn't exist: {file_path}"
            filename = os.path.basename(file_path)
            assert (
                filename not in processed_filenames_set
            ), f"File marked as unprocessed but found in processed set: {filename}"

        # Verify all unprocessed files are in the expected range
        # (should be the last 10 files since we processed the first 990)
        expected_unprocessed_files = [os.path.basename(fp) for fp in csv_files[990:]]
        actual_unprocessed_files = [
            os.path.basename(fp) for fp in unprocessed_file_paths
        ]

        # Sort both lists for comparison
        expected_unprocessed_files.sort()
        actual_unprocessed_files.sort()

        assert set(expected_unprocessed_files) == set(
            actual_unprocessed_files
        ), "Mismatch between expected and actual unprocessed files"

        print(
            f"✅ Early exit optimizer quickly found {len(unprocessed_file_paths)} unprocessed files"
        )
        print("   Sample unprocessed files:")
        for i, fp in enumerate(unprocessed_file_paths[:3]):
            print(f"     {i+1}. {os.path.basename(fp)}")

    def test_performance_o1_lookup(self, temp_environment):
        """Test that processed file lookups are O(1) using set membership."""
        csv_dir = temp_environment["csv_dir"]
        db_path = temp_environment["db_path"]

        # Create larger dataset for performance testing
        csv_files = self._create_mock_csv_files(csv_dir, count=1000)

        # Mark 990 files as processed
        self._mark_files_as_processed(db_path, csv_files, count_to_mark=990)

        # Get processed filenames set
        processed_filenames_set = get_processed_filenames(csv_dir, db_path)

        # Time O(1) membership tests
        import time

        # Test with 1000 membership checks
        test_filenames = [os.path.basename(fp) for fp in csv_files]

        start_time = time.time()
        hit_count = 0
        for filename in test_filenames:
            if filename in processed_filenames_set:  # O(1) set lookup
                hit_count += 1
        lookup_time = time.time() - start_time

        # Verify performance (should be very fast for set lookups)
        assert lookup_time < 0.1, f"O(1) lookups took too long: {lookup_time:.3f}s"
        assert hit_count == 990, f"Expected 990 hits, got {hit_count}"

        print("✅ O(1) performance test passed:")
        print(f"   1000 membership tests in {lookup_time:.4f}s")
        print(f"   Average per lookup: {lookup_time/1000*1000000:.1f} microseconds")
        print(f"   Cache hits: {hit_count}/1000")

    def test_edge_cases(self, temp_environment):
        """Test edge cases for pre-filter caching."""
        csv_dir = temp_environment["csv_dir"]
        db_path = temp_environment["db_path"]

        # Test 1: Empty directory
        empty_dir = os.path.join(temp_environment["temp_dir"], "empty")
        os.makedirs(empty_dir, exist_ok=True)

        processed_set = get_processed_filenames(empty_dir, db_path)
        assert (
            len(processed_set) == 0
        ), "Empty directory should return empty processed set"

        # Test 2: Non-existent directory
        non_existent_dir = os.path.join(temp_environment["temp_dir"], "nonexistent")
        processed_set = get_processed_filenames(non_existent_dir, db_path)
        assert (
            len(processed_set) == 0
        ), "Non-existent directory should return empty processed set"

        # Test 3: Directory with non-CSV files
        mixed_dir = os.path.join(temp_environment["temp_dir"], "mixed")
        os.makedirs(mixed_dir, exist_ok=True)

        # Create some non-CSV files
        for ext in [".txt", ".json", ".xml"]:
            non_csv_file = os.path.join(mixed_dir, f"test{ext}")
            with open(non_csv_file, "w") as f:
                f.write("test content")

        # Create one CSV file
        csv_file = os.path.join(mixed_dir, "test.csv")
        with open(csv_file, "w") as f:
            f.write("CSV content")

        processed_set = get_processed_filenames(mixed_dir, db_path)
        # Should return empty since no files are marked as processed
        assert (
            len(processed_set) == 0
        ), "Mixed directory with no processed files should return empty set"

        print("✅ Edge cases handled correctly")

    @pytest.mark.slow
    def test_large_scale_verification(self, temp_environment):
        """Test pre-filter with larger dataset to verify scalability."""
        csv_dir = temp_environment["csv_dir"]
        db_path = temp_environment["db_path"]

        # Create 5000 files for large-scale testing
        print("Creating 5000 mock CSV files for large-scale test...")
        csv_files = self._create_mock_csv_files(csv_dir, count=5000)

        # Mark 4750 files as processed (95% cached)
        print("Marking 4750 files as processed...")
        marked_filenames = self._mark_files_as_processed(
            db_path, csv_files, count_to_mark=4750
        )

        # Test pre-filter performance and accuracy
        print("Testing pre-filter performance...")
        import time

        start_time = time.time()
        processed_filenames_set = get_processed_filenames(csv_dir, db_path)
        cache_load_time = time.time() - start_time

        # Verify accuracy
        assert (
            len(processed_filenames_set) == 4750
        ), f"Expected 4750 cached files, got {len(processed_filenames_set)}"

        # Test early exit optimizer with appropriate threshold for large scale
        config = EarlyExitConfig(
            cache_ratio_threshold=0.95,
            unprocessed_threshold=300,  # Allow up to 300 unprocessed files for large-scale test
            enable_early_exit=True,
            verbose_summary=False,
        )
        optimizer = EarlyExitOptimizer(config)

        start_time = time.time()
        should_exit, scan_result = optimizer.should_use_early_exit(
            csv_dir, processed_filenames_set, file_extensions=[".csv"]
        )
        analysis_time = time.time() - start_time

        # Verify early exit decision
        assert (
            should_exit
        ), "Early exit should be triggered with 95% cache ratio and 250 unprocessed files"
        assert (
            scan_result.total_files == 5000
        ), f"Expected 5000 total files, got {scan_result.total_files}"
        assert (
            scan_result.processed_files == 4750
        ), f"Expected 4750 processed files, got {scan_result.processed_files}"
        assert (
            scan_result.unprocessed_files == 250
        ), f"Expected 250 unprocessed files, got {scan_result.unprocessed_files}"

        print("✅ Large-scale test passed:")
        print(f"   Cache load time: {cache_load_time:.3f}s for 4750 files")
        print(f"   Directory analysis time: {analysis_time:.3f}s for 5000 files")
        print(f"   Cache ratio: {scan_result.cache_ratio:.1%}")
        print(f"   Early exit triggered: {should_exit}")


if __name__ == "__main__":
    # Run tests manually if executed directly
    import sys

    print("🧪 Running Pre-filter Caching Tests...")
    print("=" * 50)

    # Create a temporary test environment
    temp_dir = tempfile.mkdtemp(prefix="manual_test_prefilter_")
    csv_dir = os.path.join(temp_dir, "test_csvs")
    os.makedirs(csv_dir, exist_ok=True)
    db_path = os.path.join(temp_dir, "test_racing.db")

    try:
        # Ensure database table exists
        if not ensure_processed_files_table(db_path):
            print("❌ Failed to create test database")
            sys.exit(1)

        # Create test instance
        test_env = {"temp_dir": temp_dir, "csv_dir": csv_dir, "db_path": db_path}

        tester = TestPrefilterCaching()

        # Run individual tests
        print("\n1. Testing 1000 mock CSV file creation...")
        tester.test_create_1000_mock_csv_files(test_env)

        print("\n2. Testing marking 990 files as processed...")
        tester.test_mark_990_files_as_processed(test_env)

        print("\n3. Testing pre-filter returns only 10 unprocessed...")
        tester.test_prefilter_returns_only_10_unprocessed(test_env)

        print("\n4. Testing early exit optimizer...")
        tester.test_early_exit_optimizer_with_cached_directory(test_env)

        print("\n5. Testing unprocessed file retrieval...")
        tester.test_early_exit_optimizer_get_unprocessed_files(test_env)

        print("\n6. Testing O(1) lookup performance...")
        tester.test_performance_o1_lookup(test_env)

        print("\n7. Testing edge cases...")
        tester.test_edge_cases(test_env)

        print("\n" + "=" * 50)
        print("✅ All pre-filter caching tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\n🧹 Cleaned up test environment: {temp_dir}")
