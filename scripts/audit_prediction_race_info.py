#!/usr/bin/env python3
"""
Prediction Race Info Audit Script
=================================

Recursively scans for prediction JSON files and audits race_info completeness.
Identifies files missing distance/grade fields and generates reports for downstream processing.

Usage:
    python scripts/audit_prediction_race_info.py [--input PATH] [--output PATH] [--dry-run]
    python scripts/audit_prediction_race_info.py --predictions-dir ./predictions --reports-dir ./reports

Environment:
- PREDICTIONS_DIR: directory containing prediction JSONs (default: ./predictions)
- REPORTS_DIR: output directory for reports (default: ./reports)
- DRY_RUN: if "1", perform a dry run

Output files:
- predictions_race_info_audit_{timestamp}.csv
- missing_race_info_manifest_{timestamp}.json
"""

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Structured logging following project patterns
from config.logging_config import get_component_logger  # type: ignore

log = get_component_logger()


def is_prediction_json(file_path: Path) -> bool:
    """Check if a JSON file is a prediction file based on schema heuristic."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Schema heuristic: JSON contains top-level "predictions" or "race_info" field
        return isinstance(data, dict) and ("predictions" in data or "race_info" in data)
    except (json.JSONDecodeError, IOError, KeyError):
        return False


def extract_race_identifiers(
    data: Dict[str, Any], file_path: Path
) -> Dict[str, Optional[str]]:
    """Extract race identifiers from prediction JSON data."""
    race_info = data.get("race_info", {}) or {}
    race_context = data.get("race_context", {}) or {}

    # Merge race_info and race_context, prioritizing race_info
    combined = {**race_context, **race_info}

    return {
        "race_url": combined.get("race_url"),
        "track": combined.get("venue") or combined.get("track"),
        "date": combined.get("date") or combined.get("race_date"),
        "meeting": combined.get("meeting_name") or combined.get("meeting"),
        "race_number": combined.get("race_number"),
        "filename": combined.get("filename") or file_path.name,
    }


def audit_race_info(data: Dict[str, Any]) -> Tuple[bool, bool, Any, Any, str, str]:
    """
    Audit race_info for distance and grade completeness.

    Returns:
        (has_race_info, has_distance, has_grade, distance_value, grade_value, status, reason)
    """
    race_info = data.get("race_info", {})

    if not race_info:
        return False, False, False, None, None, "MISSING", "No race_info field"

    distance = race_info.get("distance")
    grade = race_info.get("grade")

    has_distance = distance is not None and str(distance).strip().lower() not in (
        "",
        "unknown",
        "null",
    )
    has_grade = grade is not None and str(grade).strip().lower() not in (
        "",
        "unknown",
        "null",
    )

    if has_distance and has_grade:
        status = "OK"
        reason = "Complete race_info"
    elif not has_distance and not has_grade:
        status = "MISSING"
        reason = "Missing both distance and grade"
    elif not has_distance:
        status = "MISSING"
        reason = "Missing distance"
    else:  # not has_grade
        status = "MISSING"
        reason = "Missing grade"

    return True, has_distance, has_grade, distance, grade, status, reason


def audit_predictions_directory(
    predictions_dir: Path, dry_run: bool = False
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Recursively audit all prediction JSONs in directory.

    Returns:
        (audit_results, missing_files_manifest)
    """
    audit_results = []
    missing_files = []

    log.info(
        "Starting prediction race_info audit",
        action="audit_start",
        details={"directory": str(predictions_dir), "dry_run": dry_run},
        component="data_quality",
    )

    if not predictions_dir.exists():
        log.error(
            "Predictions directory does not exist",
            action="audit_error",
            details={"directory": str(predictions_dir)},
            component="data_quality",
        )
        return audit_results, missing_files

    # Find all JSON files recursively
    json_files = list(predictions_dir.rglob("*.json"))
    prediction_files = [f for f in json_files if is_prediction_json(f)]

    log.info(
        "Discovery completed",
        action="audit_discovery",
        details={
            "total_json_files": len(json_files),
            "prediction_files": len(prediction_files),
            "directory": str(predictions_dir),
        },
        component="data_quality",
    )

    for i, file_path in enumerate(prediction_files):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract identifiers
            identifiers = extract_race_identifiers(data, file_path)

            # Audit race_info
            (
                has_race_info,
                has_distance,
                has_grade,
                distance_value,
                grade_value,
                status,
                reason,
            ) = audit_race_info(data)

            # Detect anomalies in distance/grade types
            distance_type = (
                type(distance_value).__name__ if distance_value is not None else "None"
            )
            grade_type = (
                type(grade_value).__name__ if grade_value is not None else "None"
            )

            # Check for distance format anomalies (e.g., "520m" vs 520)
            distance_anomaly = ""
            if distance_value and isinstance(distance_value, str):
                if re.match(r"^\d+m?$", str(distance_value)):
                    distance_anomaly = f"String format: {distance_value}"

            audit_result = {
                "file_path": str(file_path.relative_to(predictions_dir.parent)),
                "has_race_info": has_race_info,
                "has_distance": has_distance,
                "has_grade": has_grade,
                "distance_value": distance_value,
                "grade_value": grade_value,
                "distance_type": distance_type,
                "grade_type": grade_type,
                "distance_anomaly": distance_anomaly,
                "race_url": identifiers.get("race_url", ""),
                "track": identifiers.get("track", ""),
                "date": identifiers.get("date", ""),
                "race_number": identifiers.get("race_number", ""),
                "status": status,
                "reason": reason,
            }

            audit_results.append(audit_result)

            # Add to missing files manifest if needed
            if status == "MISSING":
                missing_file = {
                    "file_path": str(file_path.relative_to(predictions_dir.parent)),
                    "absolute_path": str(file_path),
                    "reason": reason,
                    "identifiers": identifiers,
                    "current_race_info": data.get("race_info", {}),
                    "missing_fields": {
                        "distance": not has_distance,
                        "grade": not has_grade,
                    },
                }
                missing_files.append(missing_file)

            # Progress logging every 100 files
            if (i + 1) % 100 == 0:
                log.info(
                    f"Audit progress: {i + 1}/{len(prediction_files)} files processed",
                    action="audit_progress",
                    details={
                        "processed": i + 1,
                        "total": len(prediction_files),
                        "missing_so_far": len(missing_files),
                    },
                    component="data_quality",
                )

        except (json.JSONDecodeError, IOError) as e:
            log.error(
                f"Failed to process file: {e}",
                action="audit_file_error",
                details={"file_path": str(file_path), "error": str(e)},
                component="data_quality",
            )

            # Add error entry to audit results
            audit_result = {
                "file_path": str(file_path.relative_to(predictions_dir.parent)),
                "has_race_info": False,
                "has_distance": False,
                "has_grade": False,
                "distance_value": None,
                "grade_value": None,
                "distance_type": "None",
                "grade_type": "None",
                "distance_anomaly": "",
                "race_url": "",
                "track": "",
                "date": "",
                "race_number": "",
                "status": "BAD_FORMAT",
                "reason": f"Failed to parse JSON: {str(e)}",
            }
            audit_results.append(audit_result)

    log.info(
        "Audit completed",
        action="audit_complete",
        details={
            "total_files": len(prediction_files),
            "complete_files": len([r for r in audit_results if r["status"] == "OK"]),
            "missing_files": len(missing_files),
            "error_files": len(
                [r for r in audit_results if r["status"] == "BAD_FORMAT"]
            ),
        },
        component="data_quality",
    )

    return audit_results, missing_files


def write_audit_csv(audit_results: List[Dict[str, Any]], output_path: Path) -> None:
    """Write audit results to CSV file."""
    if not audit_results:
        return

    fieldnames = [
        "file_path",
        "has_race_info",
        "has_distance",
        "has_grade",
        "distance_value",
        "grade_value",
        "distance_type",
        "grade_type",
        "distance_anomaly",
        "race_url",
        "track",
        "date",
        "race_number",
        "status",
        "reason",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_results)


def write_missing_manifest(
    missing_files: List[Dict[str, Any]], output_path: Path
) -> None:
    """Write missing files manifest to JSON file."""
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "total_missing_files": len(missing_files),
        "summary": {
            "missing_both": len(
                [
                    f
                    for f in missing_files
                    if f["missing_fields"]["distance"] and f["missing_fields"]["grade"]
                ]
            ),
            "missing_distance_only": len(
                [
                    f
                    for f in missing_files
                    if f["missing_fields"]["distance"]
                    and not f["missing_fields"]["grade"]
                ]
            ),
            "missing_grade_only": len(
                [
                    f
                    for f in missing_files
                    if not f["missing_fields"]["distance"]
                    and f["missing_fields"]["grade"]
                ]
            ),
        },
        "files": missing_files,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Audit prediction JSONs for race_info completeness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--predictions-dir",
        "-p",
        help="Directory containing prediction JSONs (default: ./predictions)",
        type=Path,
        default=Path("./predictions"),
    )

    parser.add_argument(
        "--reports-dir",
        "-r",
        help="Output directory for reports (default: ./reports)",
        type=Path,
        default=Path("./reports"),
    )

    parser.add_argument(
        "--dry-run", help="Perform dry run without writing reports", action="store_true"
    )

    args = parser.parse_args(argv)

    # Resolve paths
    predictions_dir = args.predictions_dir.resolve()
    reports_dir = args.reports_dir.resolve()

    # Create reports directory
    if not args.dry_run:
        reports_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Perform audit
    audit_results, missing_files = audit_predictions_directory(
        predictions_dir, args.dry_run
    )

    if not args.dry_run and audit_results:
        # Write audit CSV
        csv_path = reports_dir / f"predictions_race_info_audit_{timestamp}.csv"
        write_audit_csv(audit_results, csv_path)
        log.info(
            f"Audit CSV written to {csv_path}",
            action="output_csv",
            details={"path": str(csv_path), "records": len(audit_results)},
            component="data_quality",
        )

        # Write missing files manifest
        if missing_files:
            manifest_path = reports_dir / f"missing_race_info_manifest_{timestamp}.json"
            write_missing_manifest(missing_files, manifest_path)
            log.info(
                f"Missing files manifest written to {manifest_path}",
                action="output_manifest",
                details={
                    "path": str(manifest_path),
                    "missing_files": len(missing_files),
                },
                component="data_quality",
            )

    # Summary output
    total_files = len(audit_results)
    complete_files = len([r for r in audit_results if r["status"] == "OK"])
    missing_files_count = len(missing_files)
    error_files = len([r for r in audit_results if r["status"] == "BAD_FORMAT"])

    print(f"\n=== Prediction Race Info Audit Summary ===")
    print(f"Total prediction files: {total_files}")
    print(f"Complete race_info: {complete_files}")
    print(f"Missing race_info fields: {missing_files_count}")
    print(f"Parse errors: {error_files}")

    if not args.dry_run and audit_results:
        print(f"\nReports generated:")
        print(f"- Audit CSV: {reports_dir}/predictions_race_info_audit_{timestamp}.csv")
        if missing_files:
            print(
                f"- Missing files manifest: {reports_dir}/missing_race_info_manifest_{timestamp}.json"
            )

    return 0 if error_files == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
