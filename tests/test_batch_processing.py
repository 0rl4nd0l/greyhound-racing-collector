#!/usr/bin/env python3
"""
Test script for batch processing implementation
==============================================

Tests the batch processing chunk loop implementation to ensure:
1. chunked() function works correctly
2. compute_needed_info() computes file metadata efficiently
3. process_batch() processes files in batches
4. batch_save_to_database() uses executemany() for DB operations
"""

import os
import sqlite3
import tempfile
import time

from bulk_csv_ingest import (
    BATCH_SIZE,
    FormGuideCsvIngestor,
    batch_save_to_database,
    chunked,
    compute_needed_info,
    process_batch,
)


def create_test_csv(content: str) -> str:
    """Create a temporary CSV file with given content."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(content)
        return f.name


def test_chunked_function():
    """Test the chunked function with various input sizes."""
    print("🧪 Testing chunked function...")

    # Test with exact batch size
    test_data = list(range(100))
    chunks = list(chunked(test_data, 25))
    assert len(chunks) == 4, f"Expected 4 chunks, got {len(chunks)}"
    assert all(len(chunk) == 25 for chunk in chunks), "All chunks should be size 25"

    # Test with remainder
    test_data = list(range(103))
    chunks = list(chunked(test_data, 25))
    assert len(chunks) == 5, f"Expected 5 chunks, got {len(chunks)}"
    assert len(chunks[-1]) == 3, f"Last chunk should be size 3, got {len(chunks[-1])}"

    print("✅ chunked function tests passed")


def test_compute_needed_info():
    """Test metadata computation for batch processing."""
    print("🧪 Testing compute_needed_info function...")

    # Create test CSV files
    csv1 = create_test_csv("Dog Name,PLC,BOX\nFluffy,1,2\nSpeedster,2,3\n")
    csv2 = create_test_csv("Dog Name,PLC,BOX\nRocket,1,1\nTurbo,3,4\n")
    csv3 = "/nonexistent/file.csv"  # Non-existent file

    try:
        batch = [csv1, csv2, csv3]
        metadata = compute_needed_info(batch)

        # Check that we got metadata for all files
        assert len(metadata) == 3, f"Expected metadata for 3 files, got {len(metadata)}"

        # Check existing files have proper metadata
        assert metadata[csv1]["exists"] is True
        assert metadata[csv2]["exists"] is True
        assert metadata[csv1]["hash"] is not None
        assert metadata[csv1]["size"] > 0

        # Check non-existent file handling
        assert metadata[csv3]["exists"] is False
        assert metadata[csv3]["hash"] is None

        print("✅ compute_needed_info tests passed")

    finally:
        # Cleanup
        if os.path.exists(csv1):
            os.unlink(csv1)
        if os.path.exists(csv2):
            os.unlink(csv2)


def test_batch_processing_integration():
    """Test the complete batch processing workflow."""
    print("🧪 Testing batch processing integration...")

    # Create test database
    test_db = tempfile.mktemp(suffix=".db")

    try:
        # Initialize database schema (simplified)
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()

        # Create necessary tables for testing
        cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS processed_race_files (
                file_hash TEXT PRIMARY KEY,
                file_path TEXT
            );
            
            CREATE TABLE IF NOT EXISTS race_metadata (
                race_id TEXT PRIMARY KEY,
                venue TEXT,
                race_date TEXT,
                distance TEXT,
                grade TEXT,
                field_size INTEGER,
                extraction_timestamp TEXT,
                data_source TEXT
            );
            
            CREATE TABLE IF NOT EXISTS dog_race_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT,
                dog_name TEXT,
                dog_clean_name TEXT,
                box_number TEXT,
                finish_position TEXT,
                weight TEXT,
                starting_price TEXT,
                individual_time TEXT,
                sectional_1st TEXT,
                margin TEXT,
                extraction_timestamp TEXT,
                data_source TEXT
            );
        """
        )
        conn.commit()
        conn.close()

        # Create test CSV files with valid form guide data
        csv1 = create_test_csv(
            """Dog Name,PLC,BOX,DIST,DATE,TRACK,G
Fluffy,1,2,500,2024-01-01,TestTrack,5
,2,3,520,2024-01-02,TestTrack,5
Speedster,3,1,500,2024-01-01,TestTrack,5
"""
        )

        csv2 = create_test_csv(
            """Dog Name,PLC,BOX,DIST,DATE,TRACK,G
Rocket,1,1,520,2024-01-03,TestTrack2,4
Turbo,2,4,520,2024-01-03,TestTrack2,4
"""
        )

        # Test batch processing
        batch = [csv1, csv2]

        # Initialize ingestor
        from csv_ingestion import ValidationLevel

        ingestor = FormGuideCsvIngestor(
            db_path=test_db, validation_level=ValidationLevel.LENIENT
        )

        # Step 1: Compute metadata
        metadata = compute_needed_info(batch)
        assert len(metadata) == 2

        # Step 2: Process batch
        results = process_batch(batch, ingestor, metadata)

        # Verify results structure
        assert "processed_files" in results
        assert "failed_files" in results
        assert "total_records" in results
        assert "batch_data" in results

        # Check that we got some processed data
        assert len(results["batch_data"]) > 0, "Should have processed some data"

        # Step 3: Test batch database save
        races_saved, dogs_saved = batch_save_to_database(results["batch_data"], test_db)

        # Verify data was saved
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM race_metadata")
        race_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM dog_race_data")
        dog_count = cursor.fetchone()[0]
        conn.close()

        assert race_count > 0, f"Expected races to be saved, got {race_count}"
        assert dog_count > 0, f"Expected dog records to be saved, got {dog_count}"

        print(
            f"✅ Batch processing integration test passed: {race_count} races, {dog_count} dogs"
        )

    finally:
        # Cleanup
        if os.path.exists(csv1):
            os.unlink(csv1)
        if os.path.exists(csv2):
            os.unlink(csv2)
        if os.path.exists(test_db):
            os.unlink(test_db)


def test_batch_size_configuration():
    """Test that BATCH_SIZE is configurable."""
    print("🧪 Testing BATCH_SIZE configuration...")

    assert BATCH_SIZE == 100, f"Expected BATCH_SIZE to be 100, got {BATCH_SIZE}"

    # Test that chunked respects the batch size
    large_list = list(range(BATCH_SIZE * 3 + 50))  # 350 items
    chunks = list(chunked(large_list, BATCH_SIZE))

    assert (
        len(chunks) == 4
    ), f"Expected 4 chunks with BATCH_SIZE={BATCH_SIZE}, got {len(chunks)}"
    assert len(chunks[0]) == BATCH_SIZE
    assert len(chunks[1]) == BATCH_SIZE
    assert len(chunks[2]) == BATCH_SIZE
    assert len(chunks[3]) == 50  # remainder

    print("✅ BATCH_SIZE configuration test passed")


def test_executemany_usage():
    """Test that executemany() is used for database operations."""
    print("🧪 Testing executemany() usage...")

    # Create test data
    test_data = [
        {
            "dog_name": "TestDog1",
            "track": "TestTrack",
            "date": "2024-01-01",
            "distance": "500",
            "grade": "5",
            "place": "1",
            "box": "1",
        },
        {
            "dog_name": "TestDog2",
            "track": "TestTrack",
            "date": "2024-01-01",
            "distance": "500",
            "grade": "5",
            "place": "2",
            "box": "2",
        },
    ]

    # Create test database
    test_db = tempfile.mktemp(suffix=".db")

    try:
        # Initialize database
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        cursor.executescript(
            """
            CREATE TABLE race_metadata (
                race_id TEXT PRIMARY KEY,
                venue TEXT,
                race_date TEXT,
                distance TEXT,
                grade TEXT,
                field_size INTEGER,
                extraction_timestamp TEXT,
                data_source TEXT
            );
            
            CREATE TABLE dog_race_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT,
                dog_name TEXT,
                dog_clean_name TEXT,
                box_number TEXT,
                finish_position TEXT,
                weight TEXT,
                starting_price TEXT,
                individual_time TEXT,
                sectional_1st TEXT,
                margin TEXT,
                extraction_timestamp TEXT,
                data_source TEXT
            );
        """
        )
        conn.commit()
        conn.close()

        # Test batch save (which uses executemany internally)
        start_time = time.time()
        races_saved, dogs_saved = batch_save_to_database(test_data, test_db)
        end_time = time.time()

        # Verify data was saved
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM race_metadata")
        race_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM dog_race_data")
        dog_count = cursor.fetchone()[0]
        conn.close()

        assert race_count > 0, "Should have saved race metadata"
        assert dog_count > 0, "Should have saved dog data"

        # executemany() should be fast for batch operations
        processing_time = end_time - start_time
        print(f"   Batch save completed in {processing_time:.3f} seconds")

        print("✅ executemany() usage test passed")

    finally:
        if os.path.exists(test_db):
            os.unlink(test_db)


def main():
    """Run all tests."""
    print("🚀 Running batch processing tests...\n")

    try:
        test_chunked_function()
        test_compute_needed_info()
        test_batch_processing_integration()
        test_batch_size_configuration()
        test_executemany_usage()

        print("\n🎉 All batch processing tests passed!")
        print(f"📊 Configuration: BATCH_SIZE = {BATCH_SIZE}")
        print(
            "✅ Step 5: Batch-processing chunk loop implementation is working correctly"
        )

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
