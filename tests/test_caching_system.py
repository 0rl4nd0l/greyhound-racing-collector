#!/usr/bin/env python3
"""
Test script for the robust caching and de-duplication system
============================================================

This script tests the new caching and de-duplication functionality
in FormGuideCsvScraper to ensure we never parse the same race twice.

Author: AI Assistant
Date: August 3, 2025
Version: 1.0.0 - Initial implementation
"""

import os
import shutil
import sqlite3
import sys
import tempfile

from form_guide_csv_scraper import FormGuideCsvScraper


def create_test_csv(content, filename):
    """Create a test CSV file with given content"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)


def test_caching_system():
    """Test the caching and de-duplication functionality"""
    print("🧪 Testing robust caching & de-duplication system...")

    # Create a temporary directory for testing
    test_dir = tempfile.mkdtemp()
    test_db = os.path.join(test_dir, "test_greyhound_racing.db")

    try:
        # Create test CSV content
        test_csv_content = """Dog Name,Sex,Placing,Box,Weight,Distance,Date,Track,Grade,Time,Win Time,Bonus,First Split,Margin,PIR,Starting Price
Test Dog 1,D,1,1,30.5,520,2025-08-01,MEA,5,30.12,30.12,+0.05,5.68,0.0,1,$2.50
Test Dog 2,D,2,2,31.0,520,2025-08-01,MEA,5,30.25,30.12,+0.13,5.75,0.13,2,$4.20
"""

        # Set up test environment
        original_db_path = None

        # Create scraper instance with test database
        scraper = FormGuideCsvScraper()
        original_db_path = scraper.database_path
        scraper.database_path = test_db

        # Ensure database tables exist
        scraper._ensure_database_tables()

        # Test 1: Create a test CSV file
        test_csv_file = os.path.join(test_dir, "Race 1 - MEA - 2025-08-01.csv")
        create_test_csv(test_csv_content, test_csv_file)

        print(f"📝 Created test CSV: {test_csv_file}")

        # Test 2: First processing (should be cache miss)
        print("\n🔄 Test 1: First processing (should be cache MISS)")
        result1 = scraper.parse_csv_with_ingestion(test_csv_file, force=False)
        print(f"Result: {result1}")
        assert result1 == "miss", f"Expected 'miss', got '{result1}'"

        # Test 3: Second processing (should be cache hit)
        print("\n🔄 Test 2: Second processing (should be cache HIT)")
        result2 = scraper.parse_csv_with_ingestion(test_csv_file, force=False)
        print(f"Result: {result2}")
        assert result2 == "hit", f"Expected 'hit', got '{result2}'"

        # Test 4: Force reprocessing (should be cache miss even though file was processed)
        print("\n🔄 Test 3: Force reprocessing (should be cache MISS)")
        result3 = scraper.parse_csv_with_ingestion(test_csv_file, force=True)
        print(f"Result: {result3}")
        assert result3 == "miss", f"Expected 'miss', got '{result3}'"

        # Test 5: Verify database entries
        print("\n🔍 Test 4: Checking database entries")
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM processed_race_files")
        count = cursor.fetchone()[0]
        print(f"Database entries: {count}")

        cursor.execute(
            "SELECT file_hash, race_date, venue, race_no, status FROM processed_race_files"
        )
        rows = cursor.fetchall()
        for row in rows:
            print(
                f"  Hash: {row[0][:8]}..., Date: {row[1]}, Venue: {row[2]}, Race: {row[3]}, Status: {row[4]}"
            )

        conn.close()

        # Test 6: Test with identical content but different filename
        print("\n🔄 Test 5: Same content, different filename (should be cache HIT)")
        test_csv_file2 = os.path.join(test_dir, "Race 1 - MEA - 01 August 2025.csv")
        create_test_csv(test_csv_content, test_csv_file2)

        result4 = scraper.parse_csv_with_ingestion(test_csv_file2, force=False)
        print(f"Result: {result4}")
        assert result4 == "hit", f"Expected 'hit', got '{result4}'"

        # Test 7: Test with different content
        print("\n🔄 Test 6: Different content (should be cache MISS)")
        different_csv_content = test_csv_content.replace(
            "Test Dog 1", "Different Dog 1"
        )
        test_csv_file3 = os.path.join(test_dir, "Race 2 - MEA - 2025-08-01.csv")
        create_test_csv(different_csv_content, test_csv_file3)

        result5 = scraper.parse_csv_with_ingestion(test_csv_file3, force=False)
        print(f"Result: {result5}")
        assert result5 == "miss", f"Expected 'miss', got '{result5}'"

        print("\n✅ All tests passed successfully!")
        print("\n📊 Cache Statistics:")
        print(f"   - Processed hashes in memory: {len(scraper.processed_hashes)}")

        # Final database check
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM processed_race_files")
        final_count = cursor.fetchone()[0]
        print(f"   - Total database entries: {final_count}")

        cursor.execute(
            "SELECT status, COUNT(*) FROM processed_race_files GROUP BY status"
        )
        status_counts = cursor.fetchall()
        for status, count in status_counts:
            print(f"   - {status.capitalize()} files: {count}")

        conn.close()

        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Cleanup
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"🧹 Cleaned up test directory: {test_dir}")


def test_hash_computation():
    """Test hash computation functionality"""
    print("\n🧪 Testing hash computation...")

    # Create temporary files with same and different content
    test_dir = tempfile.mkdtemp()

    try:
        scraper = FormGuideCsvScraper()

        # Create two files with identical content
        content1 = "Same content\nSecond line\n"
        file1 = os.path.join(test_dir, "file1.csv")
        file2 = os.path.join(test_dir, "file2.csv")

        with open(file1, "w") as f:
            f.write(content1)
        with open(file2, "w") as f:
            f.write(content1)

        # Create file with different content
        content2 = "Different content\nSecond line\n"
        file3 = os.path.join(test_dir, "file3.csv")
        with open(file3, "w") as f:
            f.write(content2)

        # Compute hashes
        hash1 = scraper.compute_file_hash(file1)
        hash2 = scraper.compute_file_hash(file2)
        hash3 = scraper.compute_file_hash(file3)

        print(f"Hash of file1: {hash1}")
        print(f"Hash of file2: {hash2}")
        print(f"Hash of file3: {hash3}")

        # Verify identical content produces identical hashes
        assert hash1 == hash2, f"Expected {hash1} == {hash2}"
        assert hash1 != hash3, f"Expected {hash1} != {hash3}"

        print("✅ Hash computation test passed!")
        return True

    except Exception as e:
        print(f"❌ Hash computation test failed: {e}")
        return False

    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


if __name__ == "__main__":
    print("🚀 Starting caching system tests...\n")

    # Test hash computation
    hash_test_passed = test_hash_computation()

    # Test caching system
    cache_test_passed = test_caching_system()

    print("\n🏁 Test Results Summary:")
    print(f"   Hash Computation: {'✅ PASSED' if hash_test_passed else '❌ FAILED'}")
    print(f"   Caching System:   {'✅ PASSED' if cache_test_passed else '❌ FAILED'}")

    if hash_test_passed and cache_test_passed:
        print(
            "\n🎉 All tests passed! The robust caching & de-duplication system is working correctly."
        )
        sys.exit(0)
    else:
        print("\n💥 Some tests failed. Please check the implementation.")
        sys.exit(1)
