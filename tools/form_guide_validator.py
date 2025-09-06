#!/usr/bin/env python3
"""
Form Guide Validator
====================

A spec-compliant form guide validator that implements all requirements from FORM_GUIDE_SPEC.md.

Features implemented:
- CSV preprocessing (encoding, delimiter detection, unicode cleaning, embedded newlines)
- Forward-fill rule for continuation rows
- Field/type validation producing GuideIssue objects
- Comprehensive error handling and recovery strategies

Author: AI Assistant
Date: 2025
Version: 1.0.0 - FORM_GUIDE_SPEC.md compliant implementation
"""

import argparse
import csv
import io
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class GuideIssue:
    severity: ValidationSeverity
    message: str
    line_number: Optional[int] = None
    column: Optional[str] = None
    suggested_fix: Optional[str] = None


class FormGuideValidator:
    """Main form guide validator implementing FORM_GUIDE_SPEC.md requirements."""

    def __init__(self):
        self.parsing_stats = {
            "files_processed": 0,
            "mixed_delimiters_detected": 0,
            "invisible_chars_cleaned": 0,
            "embedded_newlines_found": 0,
        }

    def preprocess_csv_content(self, content: str) -> Tuple[str, List[GuideIssue]]:
        """Preprocess CSV content according to FORM_GUIDE_SPEC.md requirements."""
        issues = []

        # Remove BOM
        if content.startswith("\ufeff"):
            content = content.replace("\ufeff", "")
            issues.append(
                GuideIssue(ValidationSeverity.INFO, "Removed BOM (Byte Order Mark)")
            )

        # Normalize line endings
        original_content = content
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        if content != original_content:
            issues.append(
                GuideIssue(ValidationSeverity.INFO, "Normalized line endings")
            )

        # Detect mixed delimiters
        mixed_delimiter_pattern = r"[,;\t]{2,}|,[;\t]|[;\t],"
        for line_num, line in enumerate(content.split("\n"), 1):
            if line.strip():  # Only check non-empty lines
                # Check if line contains different types of delimiters
                has_comma = "," in line
                has_semicolon = ";" in line
                has_tab = "\t" in line

                delimiter_count = sum([has_comma, has_semicolon, has_tab])

                if delimiter_count > 1 or re.search(mixed_delimiter_pattern, line):
                    issues.append(
                        GuideIssue(
                            ValidationSeverity.WARNING,
                            "Mixed delimiters detected",
                            line_number=line_num,
                            suggested_fix="Standardize to comma-separated values",
                        )
                    )
                    self.parsing_stats["mixed_delimiters_detected"] += 1

        # Detect and clean invisible characters
        cleaned_content, invisible_char_issues = self._clean_invisible_characters(
            content
        )
        issues.extend(invisible_char_issues)

        return cleaned_content, issues

    def _clean_invisible_characters(self, content: str) -> Tuple[str, List[GuideIssue]]:
        """Detect and clean invisible Unicode characters."""
        issues = []
        cleaned_chars = []
        invisible_chars_found = 0

        for char in content:
            char_category = unicodedata.category(char)
            if char_category in ["Cf", "Cc"] and char not in ["\n", "\r", "\t"]:
                invisible_chars_found += 1
                continue
            cleaned_chars.append(char)

        if invisible_chars_found > 0:
            issues.append(
                GuideIssue(
                    ValidationSeverity.WARNING,
                    f"Removed {invisible_chars_found} invisible Unicode characters",
                    suggested_fix="File contained hidden formatting characters that were cleaned",
                )
            )
            self.parsing_stats["invisible_chars_cleaned"] += invisible_chars_found

        return "".join(cleaned_chars), issues

    def detect_delimiter(self, content: str) -> str:
        """Auto-detect the most common delimiter in CSV content."""
        # Count occurrences of common delimiters
        comma_count = content.count(",")
        semicolon_count = content.count(";")
        tab_count = content.count("\t")

        # Return the most frequent delimiter
        delimiter_counts = {",": comma_count, ";": semicolon_count, "\t": tab_count}

        return max(delimiter_counts, key=delimiter_counts.get)

    def parse_csv_with_encoding_detection(
        self, file_path: Path
    ) -> Tuple[str, List[GuideIssue]]:
        """Parse CSV with automatic encoding detection.

        Important: open the file with newline='' to avoid universal newline translation,
        so that CRLF/CR can be explicitly detected and normalized later.
        """
        issues = []
        encodings_to_try = ["utf-8", "latin-1", "cp1252"]

        for encoding in encodings_to_try:
            try:
                # Use newline='' to preserve original line endings for normalization detection
                with open(file_path, "r", encoding=encoding, newline="") as f:
                    content = f.read()
                if encoding != "utf-8":
                    issues.append(
                        GuideIssue(
                            ValidationSeverity.INFO,
                            f"File decoded using {encoding} encoding",
                        )
                    )
                return content, issues
            except UnicodeDecodeError:
                continue

        # If all encodings fail, raise an error
        issues.append(
            GuideIssue(
                ValidationSeverity.ERROR,
                "Unable to decode file with any supported encoding",
                suggested_fix="Check file encoding or save as UTF-8",
            )
        )
        return "", issues

    def forward_fill_continuation_rows(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[GuideIssue]]:
        """Apply forward-fill rule for continuation rows."""
        issues = []
        df_copy = df.copy()

        current_dog_name = None
        current_box_number = None

        for idx, row in df_copy.iterrows():
            # Forward-fill dog name
            dog_name_col = self._find_dog_name_column(df_copy.columns)
            if dog_name_col and (
                pd.isna(row[dog_name_col]) or str(row[dog_name_col]).strip() == ""
            ):
                if current_dog_name is not None:
                    df_copy.at[idx, dog_name_col] = current_dog_name
                    issues.append(
                        GuideIssue(
                            ValidationSeverity.INFO,
                            f"Forward-filled dog name at row {idx + 1}",
                            line_number=idx + 1,
                        )
                    )
            else:
                if dog_name_col:
                    current_dog_name = str(row[dog_name_col]).strip()

            # Forward-fill box number
            box_col = self._find_box_column(df_copy.columns)
            if box_col and (pd.isna(row[box_col]) or str(row[box_col]).strip() == ""):
                if current_box_number is not None:
                    df_copy.at[idx, box_col] = current_box_number
                    issues.append(
                        GuideIssue(
                            ValidationSeverity.INFO,
                            f"Forward-filled box number at row {idx + 1}",
                            line_number=idx + 1,
                        )
                    )
            else:
                if box_col:
                    box_value = str(row[box_col]).strip()
                    if box_value:
                        current_box_number = box_value

        return df_copy, issues

    def _find_dog_name_column(self, columns: List[str]) -> Optional[str]:
        """Find the dog name column using pattern matching."""
        dog_name_patterns = [r"dog.?name", r"name", r"greyhound"]
        for col in columns:
            for pattern in dog_name_patterns:
                if re.search(pattern, col.lower()):
                    return col
        return None

    def _find_box_column(self, columns: List[str]) -> Optional[str]:
        """Find the box number column using pattern matching."""
        box_patterns = [r"box", r"trap", r"number"]
        for col in columns:
            for pattern in box_patterns:
                if re.search(pattern, col.lower()):
                    return col
        return None

    def detect_embedded_newlines(self, df: pd.DataFrame) -> List[GuideIssue]:
        """Detect embedded newlines in CSV fields."""
        issues = []

        for col_name in df.columns:
            for idx, value in df[col_name].items():
                if isinstance(value, str) and ("\n" in value or "\r" in value):
                    issues.append(
                        GuideIssue(
                            ValidationSeverity.WARNING,
                            f"Embedded newline found in field '{col_name}' at row {idx + 1}",
                            line_number=idx + 1,
                            column=col_name,
                            suggested_fix="Multi-line data should be properly quoted or cleaned",
                        )
                    )
                    self.parsing_stats["embedded_newlines_found"] += 1

        return issues

    def validate_dog_data(
        self, dog_data: Dict[str, Any], row_number: int = None
    ) -> List[GuideIssue]:
        """Validate dog block data according to FORM_GUIDE_SPEC.md requirements."""
        issues = []

        # Dog name validation (required field)
        dog_name = (
            dog_data.get("dog_name") or dog_data.get("Dog Name") or dog_data.get("name")
        )
        if not dog_name or not isinstance(dog_name, str) or not dog_name.strip():
            issues.append(
                GuideIssue(
                    ValidationSeverity.ERROR,
                    "Invalid or missing dog name",
                    line_number=row_number,
                )
            )

        # Box number validation (1-8 range per spec)
        box_number = None
        for key in ["box", "Box", "BOX"]:
            if key in dog_data:
                box_number = dog_data[key]
                break

        if box_number is not None:
            try:
                box_int = int(float(box_number))
                if box_int < 1 or box_int > 8:
                    issues.append(
                        GuideIssue(
                            ValidationSeverity.ERROR,
                            f"Box number {box_int} outside valid range (1-8)",
                            line_number=row_number,
                        )
                    )
            except (ValueError, TypeError):
                issues.append(
                    GuideIssue(
                        ValidationSeverity.ERROR,
                        f"Invalid box number format: {box_number}",
                        line_number=row_number,
                    )
                )

        # Weight validation (20.0-40.0 range per spec)
        weight = None
        for key in ["weight", "Weight", "WGT"]:
            if key in dog_data:
                weight = dog_data[key]
                break

        if weight is not None:
            try:
                weight_float = float(weight)
                if weight_float < 20.0 or weight_float > 40.0:
                    issues.append(
                        GuideIssue(
                            ValidationSeverity.WARNING,
                            f"Weight {weight_float}kg outside typical range (20.0-40.0kg)",
                            line_number=row_number,
                        )
                    )
            except (ValueError, TypeError):
                issues.append(
                    GuideIssue(
                        ValidationSeverity.WARNING,
                        f"Invalid weight format: {weight}",
                        line_number=row_number,
                    )
                )

        return issues

    def validate_csv_file(
        self, file_path: Path
    ) -> Tuple[pd.DataFrame, List[GuideIssue]]:
        """Complete validation pipeline for a CSV file."""
        all_issues = []

        # Step 1: Read file with encoding detection
        content, encoding_issues = self.parse_csv_with_encoding_detection(file_path)
        all_issues.extend(encoding_issues)

        if not content:
            return pd.DataFrame(), all_issues

        # Step 2: Preprocess content
        cleaned_content, preprocessing_issues = self.preprocess_csv_content(content)
        all_issues.extend(preprocessing_issues)

        # Step 3: Detect delimiter
        delimiter = self.detect_delimiter(cleaned_content)

        # Step 4: Parse CSV
        try:
            df = pd.read_csv(io.StringIO(cleaned_content), delimiter=delimiter)
        except Exception as e:
            all_issues.append(
                GuideIssue(
                    ValidationSeverity.ERROR,
                    f"Failed to parse CSV: {str(e)}",
                    suggested_fix="Check CSV format and structure",
                )
            )
            return pd.DataFrame(), all_issues

        # Step 5: Detect embedded newlines
        embedded_newline_issues = self.detect_embedded_newlines(df)
        all_issues.extend(embedded_newline_issues)

        # Step 6: Apply forward-fill rule
        df, forward_fill_issues = self.forward_fill_continuation_rows(df)
        all_issues.extend(forward_fill_issues)

        # Step 7: Validate each row
        for idx, row in df.iterrows():
            dog_validation_issues = self.validate_dog_data(row.to_dict(), idx + 1)
            all_issues.extend(dog_validation_issues)

        self.parsing_stats["files_processed"] += 1

        return df, all_issues


def validate_form_guide_csv(file_path: Union[str, Path]) -> Tuple[bool, Optional[pd.DataFrame], Dict, Optional[str]]:
    """Centralized form guide CSV validation routine.
    
    Returns tuple of (ok, normalized_df_or_none, validation_report_dict, quarantine_reason_or_none)
    
    Args:
        file_path: Path to the CSV file to validate
        
    Returns:
        ok: True if validation passed, False if should be quarantined
        normalized_df: Processed DataFrame if successful, None if failed
        validation_report: Dict containing issues, warnings, and stats
        quarantine_reason: String reason for quarantine if ok=False, None otherwise
    """
    file_path = Path(file_path)
    
    # Initialize validation report
    validation_report = {
        "file": str(file_path),
        "status": "unknown",
        "errors": [],
        "warnings": [],
        "info": [],
        "parsing_stats": {},
        "unique_dogs_found": 0,
        "total_rows": 0
    }
    
    quarantine_reason = None
    
    try:
        # Check if file exists and is readable
        if not file_path.exists():
            validation_report["status"] = "error"
            validation_report["errors"].append("File does not exist")
            quarantine_reason = f"File not found: {file_path}"
            return False, None, validation_report, quarantine_reason
        
        if file_path.stat().st_size == 0:
            validation_report["status"] = "error"
            validation_report["errors"].append("File is empty")
            quarantine_reason = f"Empty file: {file_path}"
            return False, None, validation_report, quarantine_reason
        
        # Validate file extension
        if not file_path.suffix.lower() == '.csv':
            validation_report["status"] = "error"
            validation_report["errors"].append(f"Invalid file extension: {file_path.suffix}")
            quarantine_reason = f"Non-CSV file extension: {file_path.suffix}"
            return False, None, validation_report, quarantine_reason
        
        # Create validator and process file
        validator = FormGuideValidator()
        
        try:
            df, all_issues = validator.validate_csv_file(file_path)
        except Exception as parse_error:
            validation_report["status"] = "error"
            validation_report["errors"].append(f"CSV parsing failed: {str(parse_error)}")
            quarantine_reason = f"Parse error: {str(parse_error)}"
            return False, None, validation_report, quarantine_reason
        
        # Process issues into validation report
        for issue in all_issues:
            if issue.severity == ValidationSeverity.ERROR:
                validation_report["errors"].append(issue.message)
            elif issue.severity == ValidationSeverity.WARNING:
                validation_report["warnings"].append(issue.message)
            else:  # INFO
                validation_report["info"].append(issue.message)
        
        # Add parsing stats
        validation_report["parsing_stats"] = validator.parsing_stats.copy()
        validation_report["total_rows"] = len(df) if df is not None else 0
        
        # Count unique dogs if we have data
        if df is not None and not df.empty:
            # Try to find dog name column
            dog_name_cols = [col for col in df.columns if 'dog' in col.lower() and 'name' in col.lower()]
            if dog_name_cols:
                dog_name_col = dog_name_cols[0]
                unique_dogs = df[dog_name_col].dropna().nunique()
                validation_report["unique_dogs_found"] = unique_dogs
                
                # Check 10 dog max constraint
                if unique_dogs > 10:
                    validation_report["errors"].append(f"Too many unique dogs: {unique_dogs} (max 10 allowed)")
        
        # Determine if validation passed
        has_critical_errors = len(validation_report["errors"]) > 0
        
        if has_critical_errors:
            validation_report["status"] = "error"
            # Construct quarantine reason from errors
            error_summary = "; ".join(validation_report["errors"][:3])  # First 3 errors
            if len(validation_report["errors"]) > 3:
                error_summary += f" (+{len(validation_report['errors']) - 3} more)"
            quarantine_reason = f"Validation failed: {error_summary}"
            return False, None, validation_report, quarantine_reason
        else:
            validation_report["status"] = "success"
            return True, df, validation_report, None
    
    except Exception as unexpected_error:
        validation_report["status"] = "error"
        validation_report["errors"].append(f"Unexpected validation error: {str(unexpected_error)}")
        quarantine_reason = f"Unexpected error: {str(unexpected_error)}"
        return False, None, validation_report, quarantine_reason


def save_to_json(file_path: Path, data: dict):
    with file_path.open("w") as json_file:
        json.dump(data, json_file, indent=4)


def validate_only(file_path: Path):
    """Parse and validate form guide, write JSON report per file."""
    logger.info(f"Starting validation-only for {file_path}")

    # Use form guide validator to parse the file
    validator = FormGuideValidator()
    df, issues = validator.validate_csv_file(file_path)

    # Determine success and quarantine status
    error_count = sum(
        1 for issue in issues if issue.severity == ValidationSeverity.ERROR
    )
    success = error_count == 0
    quarantined = error_count > len(df) * 0.5 if len(df) > 0 else error_count > 0

    # Prepare comprehensive report data
    report = {
        "file_path": str(file_path),
        "timestamp": datetime.now().isoformat(),
        "success": success,
        "quarantined": quarantined,
        "statistics": validator.parsing_stats,
        "data_records_count": len(df),
        "issues": [
            {
                "severity": issue.severity.value,
                "message": issue.message,
                "line_number": issue.line_number,
                "column": issue.column,
                "suggested_fix": issue.suggested_fix,
            }
            for issue in issues
        ],
        "issues_summary": {
            "error_count": sum(
                1 for issue in issues if issue.severity == ValidationSeverity.ERROR
            ),
            "warning_count": sum(
                1 for issue in issues if issue.severity == ValidationSeverity.WARNING
            ),
            "info_count": sum(
                1 for issue in issues if issue.severity == ValidationSeverity.INFO
            ),
        },
        "sample_data": (
            df.head(3).to_dict("records") if not df.empty else []
        ),  # First 3 records as sample
    }

    # Save report as JSON
    json_report_path = file_path.with_suffix(".report.json")
    save_to_json(json_report_path, report)

    # Print summary
    print(f"✅ Validation completed for {file_path}")
    print(f"📊 Records parsed: {len(df)}")
    print(f"⚠️  Issues found: {len(issues)}")
    print(f"📄 Report saved to: {json_report_path}")

    if quarantined:
        print(f"🚨 File would be quarantined due to critical issues")

    return report


def dry_run(file_path: Path):
    """Run full pipeline until right before DB insert/model predict, then exit."""
    logger.info(f"Starting dry-run for {file_path}")

    # Step 1: Parse and validate (same as validate_only)
    print("🔄 Step 1: Parsing and validation...")
    parser = EnhancedFormGuideParser()
    result = parser.parse_form_guide(file_path)

    print(f"   ✅ Parsed {len(result.data)} records")
    print(f"   ⚠️  Found {len(result.issues)} validation issues")

    if result.quarantined:
        print(f"   🚨 File would be quarantined - stopping dry run")
        return

    if not result.data:
        print(f"   ❌ No valid data to process - stopping dry run")
        return

    # Step 2: Feature extraction (if available)
    print("🔄 Step 2: Feature extraction...")
    if FEATURE_STORE_AVAILABLE:
        try:
            # Simulate feature extraction without DB queries
            print("   ✅ Feature extraction would be performed")
            print(
                "   📊 Features would include: box_position, recent_form, trainer_stats, venue_analysis"
            )
        except Exception as e:
            print(f"   ⚠️  Feature extraction would fail: {e}")
    else:
        print("   ⚠️  Feature store not available - basic features only")

    # Step 3: ML Model loading (if available)
    print("🔄 Step 3: ML model preparation...")
    if ML_SYSTEM_AVAILABLE:
        try:
            # Simulate model loading without actual instantiation
            print("   ✅ ML System V3 would be loaded")
            print("   🤖 Model predictions would be generated")
            print(f"   📈 {len(result.data)} dogs would receive predictions")
        except Exception as e:
            print(f"   ⚠️  ML system would fail: {e}")
    else:
        print("   ⚠️  ML System V3 not available - predictions would use fallback")

    # Step 4: Database operations (BLOCKED in dry-run)
    print("🔄 Step 4: Database operations (BLOCKED)...")
    print("   🛑 Would insert race metadata to race_metadata table")
    print(f"   🛑 Would insert {len(result.data)} dog records to dog_race_data table")
    print("   🛑 Would update dog statistics in dogs table")
    print("   🛑 Would save predictions to prediction_history table")

    # Step 5: Summary
    print("\n📋 Dry-run summary:")
    print(f"   📁 File: {file_path}")
    print(f"   📊 Records: {len(result.data)}")
    print(f"   ⚠️  Issues: {len(result.issues)}")
    print(f"   🏁 Status: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"   🗄️  Database writes: BLOCKED (dry-run mode)")

    # Create dry-run report
    dry_run_report = {
        "file_path": str(file_path),
        "timestamp": datetime.now().isoformat(),
        "dry_run": True,
        "steps_completed": [
            "parsing_validation",
            "feature_extraction_simulation",
            "ml_model_preparation",
            "database_operations_blocked",
        ],
        "success": result.success,
        "quarantined": result.quarantined,
        "records_count": len(result.data),
        "issues_count": len(result.issues),
        "database_writes_blocked": True,
        "would_insert_records": len(result.data) if result.data else 0,
    }

    # Save dry-run report
    dry_run_report_path = file_path.with_suffix(".dry-run.json")
    save_to_json(dry_run_report_path, dry_run_report)
    print(f"   📄 Dry-run report saved to: {dry_run_report_path}")

    print("\n✅ Dry-run completed. No changes were made to the database.")
    return dry_run_report


def main():
    parser = argparse.ArgumentParser(description="Form Guide Validator CLI")
    parser.add_argument(
        "input_files",
        metavar="FILE",
        type=Path,
        nargs="+",
        help="CSV files to be validated",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Parse and validate, then write JSON report for each file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full pipeline until just before DB insert/model predict, then exit.",
    )
    args = parser.parse_args()

    for file_path in args.input_files:
        if args.validate_only:
            validate_only(file_path)
        elif args.dry_run:
            dry_run(file_path)


if __name__ == "__main__":
    main()
