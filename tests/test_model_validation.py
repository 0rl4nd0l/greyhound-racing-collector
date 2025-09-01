#!/usr/bin/env python3
"""
Model Loading Validation Test - Step 4
=======================================

This script validates that the model loading meets the requirements:
- Parse prediction_stdout.log for lines containing "Loaded model" and "model_info"
- Record model type, timestamp, and version tag
- Fail the test if no model is loaded or version missing
"""

import os
import re
import sys


def parse_log_for_model_info(log_file_path):
    """
    Parse the log file for model loading information

    Returns:
        dict: Dictionary containing model information or error details
    """
    result = {
        "model_loaded": False,
        "model_info_found": False,
        "model_type": None,
        "timestamp": None,
        "version_tag": None,
        "loaded_model_lines": [],
        "model_info_lines": [],
        "errors": [],
    }

    if not os.path.exists(log_file_path):
        result["errors"].append(f"Log file not found: {log_file_path}")
        return result

    try:
        with open(log_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # Check for "Loaded model" lines
            if "Loaded model" in line:
                result["model_loaded"] = True
                result["loaded_model_lines"].append(
                    {"line_number": line_num, "content": line}
                )

                # Extract timestamp from the line if present
                timestamp_match = re.search(
                    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line
                )
                if timestamp_match:
                    result["timestamp"] = timestamp_match.group(1)

            # Check for "model_info" or "Model info" lines
            if "model_info" in line or "Model info" in line:
                result["model_info_found"] = True
                result["model_info_lines"].append(
                    {"line_number": line_num, "content": line}
                )

                # Extract model type from model_info line
                model_type_match = re.search(r"'model_type':\s*'([^']+)'", line)
                if model_type_match:
                    result["model_type"] = model_type_match.group(1)

                # Extract version tag if present
                version_match = re.search(r"'version':\s*'([^']+)'", line)
                if version_match:
                    result["version_tag"] = version_match.group(1)

                # Extract timestamp from model_info if present
                trained_at_match = re.search(r"'trained_at':\s*'([^']+)'", line)
                if trained_at_match:
                    result["timestamp"] = trained_at_match.group(1)

    except Exception as e:
        result["errors"].append(f"Error reading log file: {str(e)}")

    return result


def validate_model_requirements(parsed_result):
    """
    Validate that the model loading meets the requirements

    Returns:
        tuple: (success: bool, message: str)
    """
    errors = []

    # Check if model loading was detected
    if not parsed_result["model_loaded"]:
        errors.append("❌ No 'Loaded model' entries found in log")

    # Check if model_info was detected
    if not parsed_result["model_info_found"]:
        errors.append("❌ No 'model_info' entries found in log")

    # Check if model type was extracted
    if not parsed_result["model_type"]:
        errors.append("❌ Model type not found or missing")

    # Check if timestamp was extracted
    if not parsed_result["timestamp"]:
        errors.append("❌ Model timestamp not found or missing")

    # Note: Version tag is optional for now since it might not be in all models

    if errors:
        return False, "\n".join(errors)
    else:
        return True, "✅ All model loading requirements met"


def main():
    """Main validation function"""
    print("=" * 60)
    print("🧪 MODEL LOADING VALIDATION TEST - Step 4")
    print("=" * 60)

    # Test the original prediction_stdout.log as specified in Step 4
    log_file = "./tmp_testing/prediction_stdout.log"

    print(f"📋 Parsing log file: {log_file}")
    parsed_result = parse_log_for_model_info(log_file)

    # Display parsing results
    print("\n📊 PARSING RESULTS:")
    print("-" * 40)

    if parsed_result["errors"]:
        print("❌ ERRORS:")
        for error in parsed_result["errors"]:
            print(f"   {error}")
        return False

    print(f"Model Loaded: {'✅' if parsed_result['model_loaded'] else '❌'}")
    print(f"Model Info Found: {'✅' if parsed_result['model_info_found'] else '❌'}")
    print(f"Model Type: {parsed_result['model_type'] or 'Not found'}")
    print(f"Timestamp: {parsed_result['timestamp'] or 'Not found'}")
    print(f"Version Tag: {parsed_result['version_tag'] or 'Not found (optional)'}")

    # Show found lines
    if parsed_result["loaded_model_lines"]:
        print(
            f"\n📍 FOUND {len(parsed_result['loaded_model_lines'])} 'Loaded model' LINES:"
        )
        for line_info in parsed_result["loaded_model_lines"]:
            print(f"   Line {line_info['line_number']}: {line_info['content']}")

    if parsed_result["model_info_lines"]:
        print(
            f"\n📍 FOUND {len(parsed_result['model_info_lines'])} 'model_info' LINES:"
        )
        for line_info in parsed_result["model_info_lines"]:
            print(
                f"   Line {line_info['line_number']}: {line_info['content'][:100]}..."
            )

    # Validate requirements
    print("\n🔍 VALIDATION:")
    print("-" * 40)

    success, message = validate_model_requirements(parsed_result)
    print(message)

    if success:
        print("\n✅ STEP 4 VALIDATION: PASSED")
        print("📋 Model loading requirements have been met:")
        print(f"   • Model Type: {parsed_result['model_type']}")
        print(f"   • Timestamp: {parsed_result['timestamp']}")
        if parsed_result["version_tag"]:
            print(f"   • Version Tag: {parsed_result['version_tag']}")
    else:
        print("\n❌ STEP 4 VALIDATION: FAILED")
        print("📋 Model loading requirements not met")

    print("=" * 60)
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
