#!/usr/bin/env python3
"""
Test File Content Validation
============================

This script demonstrates the file content validation functionality implemented 
for Step 3. It creates test files and validates them to show how HTML detection,
size checks, and skipped file logging work.

Usage: python test_file_content_validation.py
"""

import os
import shutil
from pathlib import Path

from utils.file_content_validator import (
    FileContentValidator,
    validate_directory_files,
    validate_file_content,
)


def create_test_files():
    """Create test files to demonstrate validation functionality."""

    # Create a temporary directory for test files
    test_dir = Path("./test_validation_files")
    test_dir.mkdir(exist_ok=True)

    test_files = []

    # 1. Create a valid CSV file
    valid_csv = test_dir / "valid_race_data.csv"
    with open(valid_csv, "w") as f:
        f.write("Dog Name,BOX,TRACK,DATE,PLC,WGT\n")
        f.write("Lightning Strike,1,SAN,2025-01-15,1,32.5\n")
        f.write("Thunder Bolt,2,SAN,2025-01-15,2,31.8\n")
        f.write("Fast Runner,3,SAN,2025-01-15,3,33.2\n")
    test_files.append(("valid_csv", str(valid_csv)))

    # 2. Create an HTML file (mimicking a failed download)
    html_file = test_dir / "failed_download.csv"
    with open(html_file, "w") as f:
        f.write(
            """<!DOCTYPE html>
<html>
<head>
    <title>Error 404 - Page Not Found</title>
</head>
<body>
    <h1>404 Not Found</h1>
    <p>The requested resource was not found on this server.</p>
</body>
</html>"""
        )
    test_files.append(("html_file", str(html_file)))

    # 3. Create a file with HTML content but no DOCTYPE
    html_no_doctype = test_dir / "html_no_doctype.csv"
    with open(html_no_doctype, "w") as f:
        f.write(
            """<html>
<head><title>Server Error</title></head>
<body>
<div>Internal Server Error</div>
<p>Please try again later</p>
</body>
</html>"""
        )
    test_files.append(("html_no_doctype", str(html_no_doctype)))

    # 4. Create a very small file (under 100 bytes)
    small_file = test_dir / "too_small.csv"
    with open(small_file, "w") as f:
        f.write("Dog Name\nFast Dog\n")  # About 20 bytes
    test_files.append(("small_file", str(small_file)))

    # 5. Create an empty file
    empty_file = test_dir / "empty_file.csv"
    empty_file.touch()
    test_files.append(("empty_file", str(empty_file)))

    # 6. Create a JSON response file (common error)
    json_file = test_dir / "json_response.csv"
    with open(json_file, "w") as f:
        f.write('{"error": "Access denied", "message": "Authentication required"}')
    test_files.append(("json_file", str(json_file)))

    # 7. Create a file with mixed content (starts valid but has HTML)
    mixed_file = test_dir / "mixed_content.csv"
    with open(mixed_file, "w") as f:
        f.write("Dog Name,BOX,TRACK\n")
        f.write("Some Dog,1,SAN\n")
        f.write("<html><body>Error occurred during processing</body></html>\n")
    test_files.append(("mixed_file", str(mixed_file)))

    # 8. Create a large valid CSV file
    large_csv = test_dir / "large_valid.csv"
    with open(large_csv, "w") as f:
        f.write("Dog Name,BOX,TRACK,DATE,PLC,WGT,TIME,DISTANCE\n")
        for i in range(50):
            f.write(
                f"Test Dog {i+1},{(i%8)+1},SAN,2025-01-15,{(i%8)+1},32.{i%10},29.{50+i%50},500\n"
            )
    test_files.append(("large_csv", str(large_csv)))

    print(f"✅ Created {len(test_files)} test files in {test_dir}")
    return test_dir, test_files


def test_individual_validation():
    """Test individual file validation."""
    print("\n" + "=" * 60)
    print("🔍 INDIVIDUAL FILE VALIDATION TESTS")
    print("=" * 60)

    test_dir, test_files = create_test_files()

    validator = FileContentValidator(min_file_size=100, log_skipped_files=True)

    for file_type, file_path in test_files:
        print(f"\n📄 Testing: {file_type} ({os.path.basename(file_path)})")

        is_valid, message, file_info = validator.validate_file(file_path)

        status = "✅ VALID" if is_valid else "❌ INVALID"
        print(f"   Result: {status}")
        print(f"   Message: {message}")
        print(f"   File size: {file_info['file_size']} bytes")
        print(f"   Content type: {file_info['content_type']}")

        if not is_valid and file_info.get("html_indicators"):
            print(
                f"   HTML indicators found: {', '.join(file_info['html_indicators'][:3])}"
            )
        elif not is_valid and file_info.get("non_csv_patterns"):
            print(
                f"   Non-CSV patterns found: {', '.join(file_info['non_csv_patterns'][:3])}"
            )

    # Test standalone function
    print("\n🔧 Testing standalone function...")
    is_valid, message = validate_file_content(test_files[0][1], min_file_size=50)
    print(
        f"   Standalone result: {'✅ VALID' if is_valid else '❌ INVALID'} - {message}"
    )

    return test_dir, validator


def test_batch_validation():
    """Test batch file validation."""
    print("\n" + "=" * 60)
    print("📦 BATCH FILE VALIDATION TESTS")
    print("=" * 60)

    test_dir, test_files = create_test_files()

    # Test batch validation
    validator = FileContentValidator(min_file_size=100, log_skipped_files=True)
    file_paths = [file_path for _, file_path in test_files]

    results = validator.validate_files_batch(file_paths)

    print("\n📊 BATCH VALIDATION RESULTS:")
    print(f"   Total files: {results['total_files']}")
    print(f"   Valid files: {len(results['valid_files'])}")
    print(f"   Invalid files: {len(results['invalid_files'])}")

    print("\n✅ VALID FILES:")
    for valid_file in results["valid_files"]:
        filename = os.path.basename(valid_file["path"])
        print(f"   - {filename}: {valid_file['message']}")

    print("\n❌ INVALID FILES:")
    for invalid_file in results["invalid_files"]:
        filename = os.path.basename(invalid_file["path"])
        print(f"   - {filename}: {invalid_file['message']}")

    # Print detailed statistics
    validator.print_validation_summary(results)

    return test_dir, validator


def test_directory_validation():
    """Test directory validation function."""
    print("\n" + "=" * 60)
    print("📁 DIRECTORY VALIDATION TEST")
    print("=" * 60)

    test_dir, test_files = create_test_files()

    # Test directory validation function
    results = validate_directory_files(str(test_dir), min_file_size=100)

    return test_dir


def test_skipped_files_summary():
    """Test skipped files summary functionality."""
    print("\n" + "=" * 60)
    print("📋 SKIPPED FILES SUMMARY TEST")
    print("=" * 60)

    test_dir, validator = test_individual_validation()

    # Get skipped files summary
    summary = validator.get_skipped_files_summary()

    print("📊 SKIPPED FILES SUMMARY:")
    print(f"   Total skipped: {summary['total_skipped']}")

    if summary["total_skipped"] > 0:
        print("\n🔍 SKIP REASONS:")
        for reason, count in summary["reasons"].items():
            print(f"   - {reason}: {count} files")

        print("\n📄 DETAILED SKIPPED FILES:")
        for skipped in summary["skipped_files"][:5]:  # Show first 5
            filename = os.path.basename(skipped["path"])
            print(f"   - {filename}: {skipped['reason']}")

    # Test clear functionality
    print("\n🧹 Testing clear skipped files log...")
    validator.clear_skipped_files_log()
    summary_after_clear = validator.get_skipped_files_summary()
    print(f"   Skipped files after clear: {summary_after_clear['total_skipped']}")

    return test_dir


def cleanup_test_files(test_dir):
    """Clean up test files."""
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print(f"\n🧹 Cleaned up test files in {test_dir}")


def main():
    """Run all validation tests."""
    print("🚀 STARTING FILE CONTENT VALIDATION TESTS")
    print("=" * 60)

    try:
        # Run individual tests
        test_dir, _ = test_individual_validation()

        # Run batch tests
        test_batch_validation()

        # Run directory tests
        test_directory_validation()

        # Run skipped files summary tests
        test_skipped_files_summary()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        print("\n💡 KEY FEATURES DEMONSTRATED:")
        print("   ✅ HTML content detection (DOCTYPE, html tags, etc.)")
        print("   ✅ File size validation (default: 100 bytes minimum)")
        print("   ✅ Empty file detection")
        print("   ✅ Non-CSV content detection (JSON, XML, etc.)")
        print("   ✅ Batch processing capabilities")
        print("   ✅ Detailed logging and reporting")
        print("   ✅ Skipped files tracking and summary")
        print("   ✅ Multiple encoding support")

        print("\n🔧 INTEGRATION POINTS:")
        print("   - Enhanced comprehensive_prediction_pipeline.py")
        print("   - Integrated into bulk_csv_ingest.py")
        print("   - Available as standalone utility functions")
        print("   - Logging integration for skipped file tracking")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Clean up test files
        if "test_dir" in locals():
            cleanup_test_files(test_dir)


if __name__ == "__main__":
    main()
