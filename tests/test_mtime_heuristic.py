#!/usr/bin/env python3
"""
Test Script for Mtime Heuristic Implementation
==============================================

This script tests the mtime heuristic functionality to ensure it works correctly
for optimizing file scanning performance.

Usage:
    python test_mtime_heuristic.py [test_directory]

Author: AI Assistant
Date: 2025-01-04
"""

import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

try:
    from utils.mtime_heuristic import FileEntry, create_mtime_heuristic

    MTIME_AVAILABLE = True
except ImportError as e:
    print(f"❌ Mtime heuristic not available: {e}")
    MTIME_AVAILABLE = False


def create_test_files(test_dir: Path, num_files: int = 5):
    """Create test CSV files with different modification times"""
    print(f"📁 Creating {num_files} test files in {test_dir}")

    test_dir.mkdir(exist_ok=True)
    created_files = []

    base_time = time.time() - (num_files * 60)  # Start from N minutes ago

    for i in range(num_files):
        file_path = test_dir / f"test_race_{i+1}.csv"

        # Create simple CSV content
        content = f"""Dog Name,PLC,BOX,DIST,DATE,TRACK
Test Dog {i+1},1,{i+1},500,2025-01-04,TEST
Test Dog {i+2},2,{i+2},500,2025-01-04,TEST
"""

        with open(file_path, "w") as f:
            f.write(content)

        # Set modification time to be incremental
        file_mtime = base_time + (i * 60)  # Each file 1 minute newer
        os.utime(file_path, (file_mtime, file_mtime))

        created_files.append((file_path, file_mtime))

        formatted_time = datetime.fromtimestamp(file_mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        print(f"  ✅ Created {file_path.name} (mtime: {formatted_time})")

    return created_files


def test_basic_functionality():
    """Test basic mtime heuristic functionality"""
    print("\n🧪 Testing basic mtime heuristic functionality")
    print("=" * 50)

    if not MTIME_AVAILABLE:
        print("❌ Mtime heuristic not available, skipping tests")
        return False

    # Create temporary test directory
    with tempfile.TemporaryDirectory(prefix="mtime_test_") as temp_dir:
        test_dir = Path(temp_dir)

        # Create test database in temp directory
        db_path = test_dir / "test.sqlite"

        try:
            # Initialize heuristic
            heuristic = create_mtime_heuristic(str(db_path))

            # Test 1: Initial state - no last processed mtime
            print("\n📋 Test 1: Initial state")
            last_mtime = heuristic.get_last_processed_mtime()
            print(f"  Last processed mtime: {last_mtime}")
            assert last_mtime is None, "Initial mtime should be None"
            print("  ✅ Initial state test passed")

            # Test 2: Create test files
            created_files = create_test_files(test_dir)

            # Test 3: Scan directory with no heuristic (should find all files)
            print("\n📋 Test 3: Full scan (no heuristic)")
            files_found = list(
                heuristic.scan_directory_optimized(str(test_dir), strict_scan=True)
            )
            print(f"  Files found: {len(files_found)}")
            assert len(files_found) == len(
                created_files
            ), f"Should find all {len(created_files)} files"
            print("  ✅ Full scan test passed")

            # Test 4: Set mtime to middle point
            middle_file_mtime = created_files[2][1]  # 3rd file's mtime
            heuristic.set_last_processed_mtime(middle_file_mtime)
            print("\n📋 Test 4: Set last processed mtime to middle point")
            print(
                f"  Set mtime: {datetime.fromtimestamp(middle_file_mtime).strftime('%Y-%m-%d %H:%M:%S')}"
            )

            # Test 5: Optimized scan (should only find newer files)
            print("\n📋 Test 5: Optimized scan (should find only newer files)")
            newer_files = list(
                heuristic.scan_directory_optimized(str(test_dir), strict_scan=False)
            )
            print(f"  Files found: {len(newer_files)}")
            expected_newer = sum(
                1 for _, mtime in created_files if mtime > middle_file_mtime
            )
            assert (
                len(newer_files) == expected_newer
            ), f"Should find {expected_newer} newer files, found {len(newer_files)}"
            print("  ✅ Optimized scan test passed")

            # Test 6: Update mtime from processed files
            print("\n📋 Test 6: Update mtime from processed files")
            processed_files = [str(f[0]) for f in created_files[-2:]]  # Last 2 files
            heuristic.update_processed_mtime_from_files(processed_files)

            updated_mtime = heuristic.get_last_processed_mtime()
            expected_max_mtime = max(f[1] for f in created_files[-2:])
            assert (
                updated_mtime == expected_max_mtime
            ), f"Mtime should be updated to {expected_max_mtime}, got {updated_mtime}"
            print("  ✅ Mtime update test passed")

            # Test 7: Scan statistics
            print("\n📋 Test 7: Scan statistics")
            stats = heuristic.get_scan_statistics()
            print(f"  Statistics: {stats}")
            assert stats["heuristic_enabled"] == True, "Heuristic should be enabled"
            assert (
                stats["last_processed_mtime"] is not None
            ), "Should have last processed mtime"
            print("  ✅ Statistics test passed")

            # Test 8: Reset heuristic
            print("\n📋 Test 8: Reset heuristic")
            heuristic.reset_mtime_heuristic()
            reset_mtime = heuristic.get_last_processed_mtime()
            assert reset_mtime is None, "Mtime should be None after reset"
            print("  ✅ Reset test passed")

            print("\n🎉 All tests passed!")
            return True

        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback

            traceback.print_exc()
            return False


def test_with_user_directory(test_dir: str):
    """Test with user-provided directory"""
    print(f"\n🧪 Testing with user directory: {test_dir}")
    print("=" * 50)

    if not MTIME_AVAILABLE:
        print("❌ Mtime heuristic not available, skipping tests")
        return False

    if not os.path.exists(test_dir):
        print(f"❌ Directory does not exist: {test_dir}")
        return False

    try:
        heuristic = create_mtime_heuristic()

        print("📊 Current scan statistics:")
        stats = heuristic.get_scan_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print(f"\n📋 Full scan of {test_dir}:")
        full_scan_files = list(
            heuristic.scan_directory_optimized(test_dir, strict_scan=True)
        )
        print(f"  Total CSV files: {len(full_scan_files)}")

        if full_scan_files:
            print("  Sample files:")
            for i, file_entry in enumerate(full_scan_files[:3]):
                mtime_str = datetime.fromtimestamp(file_entry.mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                print(f"    {file_entry.name} (mtime: {mtime_str})")
            if len(full_scan_files) > 3:
                print(f"    ... and {len(full_scan_files) - 3} more files")

        print(f"\n📋 Optimized scan of {test_dir}:")
        optimized_files = list(
            heuristic.scan_directory_optimized(test_dir, strict_scan=False)
        )
        print(f"  Files requiring processing: {len(optimized_files)}")

        if len(optimized_files) != len(full_scan_files):
            print(
                f"  🚀 Optimization filtered out {len(full_scan_files) - len(optimized_files)} files"
            )
        else:
            print(
                "  ℹ️ No optimization applied (no previous mtime or all files are newer)"
            )

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("🔬 Mtime Heuristic Test Suite")
    print("=" * 50)

    success = True

    # Run basic functionality tests
    if not test_basic_functionality():
        success = False

    # Test with user directory if provided
    if len(sys.argv) > 1:
        test_directory = sys.argv[1]
        if not test_with_user_directory(test_directory):
            success = False

    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests completed successfully!")
        print("\nThe mtime heuristic is working correctly and ready for use.")
        print("\nNext steps:")
        print(
            "  1. Run migration to add db_meta table: python migrations/add_db_meta_table.py"
        )
        print("  2. Use optimized scanning: python run.py analyze")
        print(
            "  3. Force full re-scan when needed: python run.py analyze --strict-scan"
        )
    else:
        print("❌ Some tests failed!")
        print("Please check the implementation and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
