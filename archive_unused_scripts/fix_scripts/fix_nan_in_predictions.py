#!/usr/bin/env python3
"""
Utility script to fix NaN values in prediction JSON files.

This script:
1. Scans all prediction JSON files for literal "NaN" strings
2. Replaces them with null values
3. Creates a backup before modifying files
4. Reports on changes made

Usage: python fix_nan_in_predictions.py
"""

import json
import math
import os
import re
import shutil
from datetime import datetime


def safe_json_value(value):
    """Convert problematic values to JSON-safe equivalents"""
    if value is None:
        return None

    # Handle string representations of NaN/Infinity
    if isinstance(value, str):
        if value.strip().lower() in ["nan", "inf", "-inf", "infinity", "-infinity"]:
            return None
        return value

    # Handle numeric NaN/Infinity
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    return value


def clean_json_data(data):
    """Recursively clean JSON data, replacing NaN values with null"""
    if isinstance(data, dict):
        return {key: clean_json_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [clean_json_data(item) for item in data]
    else:
        return safe_json_value(data)


def fix_nan_in_json_file(file_path):
    """Fix NaN values in a single JSON file"""
    print(f"Processing: {file_path}")

    try:
        # First, try to read as text and check for literal NaN strings
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check if file contains literal NaN strings
        nan_patterns = [
            r":\s*NaN\s*[,}]",  # NaN values
            r":\s*Infinity\s*[,}]",  # Infinity values
            r":\s*-Infinity\s*[,}]",  # -Infinity values
        ]

        has_nan = any(re.search(pattern, content) for pattern in nan_patterns)

        if not has_nan:
            print(f"  ✓ No NaN values found")
            return False

        print(f"  ⚠️  Found NaN values, fixing...")

        # Replace literal NaN/Infinity strings with null
        original_content = content
        content = re.sub(r":\s*NaN\s*([,}])", r": null\1", content)
        content = re.sub(r":\s*Infinity\s*([,}])", r": null\1", content)
        content = re.sub(r":\s*-Infinity\s*([,}])", r": null\1", content)

        # Try to parse as JSON to ensure validity
        try:
            data = json.loads(content)
            # Clean any remaining NaN values in the parsed data
            cleaned_data = clean_json_data(data)

            # Create backup
            backup_path = file_path + ".backup"
            shutil.copy2(file_path, backup_path)
            print(f"  📄 Backup created: {backup_path}")

            # Write cleaned data
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

            print(f"  ✅ Fixed and saved")
            return True

        except json.JSONDecodeError as e:
            print(f"  ❌ JSON parsing error after NaN replacement: {e}")
            # Restore original content
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(original_content)
            return False

    except Exception as e:
        print(f"  ❌ Error processing file: {e}")
        return False


def main():
    """Main function to fix NaN values in all prediction files"""
    predictions_dir = "./predictions"

    if not os.path.exists(predictions_dir):
        print(f"❌ Predictions directory not found: {predictions_dir}")
        return

    print(f"🔍 Scanning prediction files in: {predictions_dir}")
    print("=" * 60)

    # Find all prediction JSON files
    prediction_files = []
    for filename in os.listdir(predictions_dir):
        if filename.startswith("prediction_") and filename.endswith(".json"):
            file_path = os.path.join(predictions_dir, filename)
            prediction_files.append(file_path)

    if not prediction_files:
        print("No prediction files found.")
        return

    print(f"Found {len(prediction_files)} prediction files")
    print("-" * 60)

    # Process each file
    fixed_count = 0
    error_count = 0

    for file_path in sorted(prediction_files):
        try:
            if fix_nan_in_json_file(file_path):
                fixed_count += 1
        except Exception as e:
            print(f"❌ Error with {file_path}: {e}")
            error_count += 1

    # Summary
    print("=" * 60)
    print(f"📊 SUMMARY:")
    print(f"  Total files processed: {len(prediction_files)}")
    print(f"  Files fixed: {fixed_count}")
    print(f"  Files with errors: {error_count}")
    print(f"  Files unchanged: {len(prediction_files) - fixed_count - error_count}")

    if fixed_count > 0:
        print(f"\n✅ Successfully fixed {fixed_count} files!")
        print("💾 Backup files (.backup) have been created for all modified files.")
    else:
        print("\n✅ No files needed fixing.")


if __name__ == "__main__":
    main()
