#!/usr/bin/env python3
"""
Enhanced Form Guide Parser
=========================

A spec-compliant form guide parser that implements all requirements from FORM_GUIDE_SPEC.md.
This module addresses all gaps found in the static audit.

Features implemented:
- Mixed delimiter detection
- Invisible Unicode character detection and cleaning
- Header drift detection  
- Embedded newlines detection
- Comprehensive data type validation
- Pre-processing pipeline with BOM removal
- Enhanced forward-fill logic
- Quarantine system for problematic files
- Recovery strategies for common issues

Author: AI Assistant
Date: August 3, 2025
Version: 2.0.0 - Spec-compliant implementation
"""

import csv
import logging
import os
import re
import shutil
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues"""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A validation issue found during parsing"""

    severity: ValidationSeverity
    message: str
    line_number: Optional[int] = None
    column: Optional[str] = None
    suggested_fix: Optional[str] = None


@dataclass
class ParsingResult:
    """Result of form guide parsing"""

    success: bool
    data: List[Dict[str, Any]]
    issues: List[ValidationIssue]
    quarantined: bool
    statistics: Dict[str, Any]


class EnhancedFormGuideParser:
    """
    Enhanced form guide parser implementing all FORM_GUIDE_SPEC.md requirements.
    """

    def __init__(self, quarantine_dir: str = "./quarantine"):
        """
        Initialize the enhanced parser.

        Args:
            quarantine_dir: Directory to move problematic files
        """
        self.quarantine_dir = Path(quarantine_dir)
        self.quarantine_dir.mkdir(exist_ok=True)

        # Expected headers for drift detection
        self.expected_headers = [
            "Dog Name",
            "Sex",
            "PLC",
            "BOX",
            "WGT",
            "DIST",
            "DATE",
            "TRACK",
            "G",
            "TIME",
            "WIN",
            "BON",
            "1 SEC",
            "MGN",
            "W/2G",
            "PIR",
            "SP",
        ]

        # Statistics tracking
        self.parsing_stats = {
            "files_processed": 0,
            "files_quarantined": 0,
            "mixed_delimiters_detected": 0,
            "invisible_chars_cleaned": 0,
            "header_drifts_detected": 0,
            "embedded_newlines_found": 0,
        }

    def preprocess_csv_content(self, content: str) -> Tuple[str, List[ValidationIssue]]:
        """
        Pre-process CSV content according to FORM_GUIDE_SPEC.md requirements.

        Args:
            content: Raw CSV content

        Returns:
            Tuple of (cleaned_content, issues)
        """
        issues = []

        # Step 1: Remove BOM
        if content.startswith("\ufeff"):
            content = content.replace("\ufeff", "")
            issues.append(
                ValidationIssue(
                    ValidationSeverity.INFO, "Removed BOM (Byte Order Mark) from file"
                )
            )

        # Step 2: Normalize line endings
        original_content = content
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        if content != original_content:
            issues.append(
                ValidationIssue(
                    ValidationSeverity.INFO, "Normalized line endings to Unix format"
                )
            )

        # Step 3: Check for mixed delimiters
        mixed_delimiter_issues = self._detect_mixed_delimiters(content)
        issues.extend(mixed_delimiter_issues)

        # Step 4: Detect and clean invisible characters
        cleaned_content, invisible_char_issues = self._clean_invisible_characters(
            content
        )
        content = cleaned_content
        issues.extend(invisible_char_issues)

        return content, issues

    def _detect_mixed_delimiters(self, content: str) -> List[ValidationIssue]:
        """
        Detect mixed delimiters as specified in FORM_GUIDE_SPEC.md.

        Args:
            content: CSV content to check

        Returns:
            List of validation issues
        """
        issues = []
        mixed_delimiter_pattern = r"[,;\t]{2,}|,[;\t]|[;\t],"

        lines = content.split("\n")
        for line_num, line in enumerate(lines, 1):
            if re.search(mixed_delimiter_pattern, line):
                issues.append(
                    ValidationIssue(
                        ValidationSeverity.WARNING,
                        f"Mixed delimiters detected: inconsistent use of commas, semicolons, or tabs",
                        line_number=line_num,
                        suggested_fix="Standardize to comma-separated values",
                    )
                )
                self.parsing_stats["mixed_delimiters_detected"] += 1

        return issues

    def _clean_invisible_characters(
        self, content: str
    ) -> Tuple[str, List[ValidationIssue]]:
        """
        Detect and clean invisible Unicode characters.

        Args:
            content: Content to clean

        Returns:
            Tuple of (cleaned_content, issues)
        """
        issues = []
        cleaned_chars = []
        invisible_chars_found = 0

        for char in content:
            char_category = unicodedata.category(char)
            if char_category in ["Cf", "Cc"] and char not in ["\n", "\r", "\t"]:
                # Skip invisible control/format characters (except allowed ones)
                invisible_chars_found += 1
                continue
            cleaned_chars.append(char)

        if invisible_chars_found > 0:
            issues.append(
                ValidationIssue(
                    ValidationSeverity.WARNING,
                    f"Removed {invisible_chars_found} invisible Unicode characters",
                    suggested_fix="File contained hidden formatting characters that were cleaned",
                )
            )
            self.parsing_stats["invisible_chars_cleaned"] += invisible_chars_found

        return "".join(cleaned_chars), issues

    def detect_header_drift(self, headers: List[str]) -> List[ValidationIssue]:
        """
        Detect header drift according to FORM_GUIDE_SPEC.md (30% threshold).

        Args:
            headers: Headers from CSV file

        Returns:
            List of validation issues
        """
        issues = []

        if not headers:
            return issues

        drift_score = 0
        expected_count = min(len(headers), len(self.expected_headers))

        for i in range(expected_count):
            actual_header = headers[i].strip().upper()
            expected_header = self.expected_headers[i].upper()

            if actual_header != expected_header:
                drift_score += 1

        drift_percentage = drift_score / expected_count if expected_count > 0 else 0

        if drift_percentage > 0.3:  # 30% threshold from spec
            issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    f"Header drift detected: {drift_percentage:.1%} of headers don't match expected positions",
                    suggested_fix="Check if columns have been added/removed or reordered",
                )
            )
            self.parsing_stats["header_drifts_detected"] += 1

        return issues

    def detect_embedded_newlines(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """
        Detect embedded newlines in CSV fields.

        Args:
            df: DataFrame to check

        Returns:
            List of validation issues
        """
        issues = []

        for col_name in df.columns:
            for idx, value in df[col_name].items():
                if isinstance(value, str) and ("\n" in value or "\r" in value):
                    issues.append(
                        ValidationIssue(
                            ValidationSeverity.WARNING,
                            f"Embedded newline found in field '{col_name}' at row {idx + 1}",
                            line_number=idx + 1,
                            column=col_name,
                            suggested_fix="Multi-line data should be properly quoted or cleaned",
                        )
                    )
                    self.parsing_stats["embedded_newlines_found"] += 1

        return issues

    def validate_dog_block(self, dog_data: Dict[str, Any]) -> List[ValidationIssue]:
        """
        Validate dog block data according to FORM_GUIDE_SPEC.md requirements.

        Args:
            dog_data: Dog data dictionary

        Returns:
            List of validation issues
        """
        issues = []

        # Dog name validation
        dog_name = dog_data.get("dog_name")
        if not dog_name or not isinstance(dog_name, str) or not dog_name.strip():
            issues.append(
                ValidationIssue(ValidationSeverity.ERROR, "Invalid or missing dog name")
            )

        # Box number validation (1-8 range per spec)
        box_number = dog_data.get("box")
        if box_number is not None:
            try:
                box_int = int(box_number)
                if box_int < 1 or box_int > 8:
                    issues.append(
                        ValidationIssue(
                            ValidationSeverity.ERROR,
                            f"Box number {box_int} outside valid range (1-8)",
                        )
                    )
            except (ValueError, TypeError):
                issues.append(
                    ValidationIssue(
                        ValidationSeverity.ERROR,
                        f"Invalid box number format: {box_number}",
                    )
                )

        # Weight validation (20.0-40.0 range per spec)
        weight = dog_data.get("weight")
        if weight is not None and weight != "":
            try:
                weight_float = float(weight)
                if weight_float < 20.0 or weight_float > 40.0:
                    issues.append(
                        ValidationIssue(
                            ValidationSeverity.WARNING,
                            f"Weight {weight_float}kg outside typical range (20.0-40.0kg)",
                        )
                    )
            except (ValueError, TypeError):
                issues.append(
                    ValidationIssue(
                        ValidationSeverity.WARNING, f"Invalid weight format: {weight}"
                    )
                )

        return issues

    def enhanced_forward_fill(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[ValidationIssue]]:
        """
        Enhanced forward-fill logic for dog names and box numbers.

        Args:
            df: DataFrame to process

        Returns:
            Tuple of (processed_df, issues)
        """
        issues = []
        df_copy = df.copy()

        current_dog_name = None
        current_box_number = None

        for idx, row in df_copy.iterrows():
            # Forward-fill dog name
            if (
                pd.isna(row.get("dog_name"))
                or str(row.get("dog_name", "")).strip() == ""
            ):
                if current_dog_name is not None:
                    df_copy.at[idx, "dog_name"] = current_dog_name
                    issues.append(
                        ValidationIssue(
                            ValidationSeverity.INFO,
                            f"Forward-filled dog name at row {idx + 1}",
                            line_number=idx + 1,
                        )
                    )
            else:
                current_dog_name = str(row["dog_name"]).strip()
                # Remove number prefix if present (e.g., "1. Dog Name" -> "Dog Name")
                if ". " in current_dog_name:
                    current_dog_name = current_dog_name.split(". ", 1)[1]

            # Forward-fill box number (new feature per spec)
            if pd.isna(row.get("box")) or str(row.get("box", "")).strip() == "":
                if current_box_number is not None:
                    df_copy.at[idx, "box"] = current_box_number
                    issues.append(
                        ValidationIssue(
                            ValidationSeverity.INFO,
                            f"Forward-filled box number at row {idx + 1}",
                            line_number=idx + 1,
                        )
                    )
            else:
                box_value = str(row["box"]).strip()
                if box_value:
                    current_box_number = box_value

        return df_copy, issues

    def should_quarantine(
        self, issues: List[ValidationIssue], data: List[Dict[str, Any]]
    ) -> bool:
        """
        Determine if file should be quarantined based on FORM_GUIDE_SPEC.md criteria.

        Args:
            issues: List of validation issues
            data: Processed data

        Returns:
            True if file should be quarantined
        """
        # Count error severity issues
        error_count = sum(
            1 for issue in issues if issue.severity == ValidationSeverity.ERROR
        )
        total_rows = len(data)

        # Quarantine conditions from spec
        if total_rows == 0:
            return True  # No valid dog blocks detected

        if error_count > total_rows * 0.5:
            return True  # More than 50% of rows fail validation

        # Check for critical fields missing
        critical_missing = any(
            "dog name" in issue.message.lower() or "box number" in issue.message.lower()
            for issue in issues
            if issue.severity == ValidationSeverity.ERROR
        )

        if critical_missing and error_count > 0:
            return True

        return False

    def quarantine_file(self, file_path: Path, issues: List[ValidationIssue]) -> str:
        """
        Move problematic file to quarantine directory.

        Args:
            file_path: Path to file to quarantine
            issues: List of issues that caused quarantine

        Returns:
            Path to quarantined file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quarantine_filename = f"{timestamp}_{file_path.name}"
        quarantine_path = self.quarantine_dir / quarantine_filename

        # Copy file to quarantine
        shutil.copy2(file_path, quarantine_path)

        # Create issue report
        report_path = quarantine_path.with_suffix(".issues.txt")
        with open(report_path, "w") as f:
            f.write(f"File quarantined: {file_path}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Issues found:\n\n")

            for issue in issues:
                f.write(f"[{issue.severity.value.upper()}] {issue.message}\n")
                if issue.line_number:
                    f.write(f"  Line: {issue.line_number}\n")
                if issue.column:
                    f.write(f"  Column: {issue.column}\n")
                if issue.suggested_fix:
                    f.write(f"  Suggested fix: {issue.suggested_fix}\n")
                f.write("\n")

        self.parsing_stats["files_quarantined"] += 1
        logger.warning(f"File quarantined: {file_path} -> {quarantine_path}")

        return str(quarantine_path)

    def parse_form_guide(self, file_path: Union[str, Path]) -> ParsingResult:
        """
        Main parsing method that implements the complete FORM_GUIDE_SPEC.md algorithm.

        Args:
            file_path: Path to CSV file to parse

        Returns:
            ParsingResult with data and issues
        """
        file_path = Path(file_path)
        all_issues = []
        data = []
        quarantined = False

        self.parsing_stats["files_processed"] += 1

        try:
            # Step 1: Read and preprocess content
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                raw_content = f.read()

            content, preprocessing_issues = self.preprocess_csv_content(raw_content)
            all_issues.extend(preprocessing_issues)

            # Step 2: Parse CSV
            from io import StringIO

            df = pd.read_csv(StringIO(content), on_bad_lines="skip")

            if df.empty:
                all_issues.append(
                    ValidationIssue(
                        ValidationSeverity.ERROR,
                        "CSV file is empty or contains no valid data",
                    )
                )
                return ParsingResult(
                    False, [], all_issues, True, self.parsing_stats.copy()
                )

            # Step 3: Header detection and drift analysis
            headers = list(df.columns)
            header_issues = self.detect_header_drift(headers)
            all_issues.extend(header_issues)

            # Step 4: Detect embedded newlines
            newline_issues = self.detect_embedded_newlines(df)
            all_issues.extend(newline_issues)

            # Step 5: Enhanced forward-fill processing
            df_processed, forward_fill_issues = self.enhanced_forward_fill(df)
            all_issues.extend(forward_fill_issues)

            # Step 6: Convert to list of dictionaries and validate each dog block
            for idx, row in df_processed.iterrows():
                record = row.to_dict()

                # Validate this dog block
                validation_issues = self.validate_dog_block(record)
                all_issues.extend(validation_issues)

                # Only add if it has required fields
                if record.get("dog_name") and str(record.get("dog_name", "")).strip():
                    data.append(record)

            # Step 7: Determine if quarantine is needed
            quarantined = self.should_quarantine(all_issues, data)

            if quarantined:
                self.quarantine_file(file_path, all_issues)

            success = len(data) > 0 and not quarantined

            return ParsingResult(
                success=success,
                data=data,
                issues=all_issues,
                quarantined=quarantined,
                statistics=self.parsing_stats.copy(),
            )

        except Exception as e:
            all_issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR, f"Unexpected parsing error: {str(e)}"
                )
            )

            return ParsingResult(
                success=False,
                data=[],
                issues=all_issues,
                quarantined=True,
                statistics=self.parsing_stats.copy(),
            )

    def get_parsing_statistics(self) -> Dict[str, Any]:
        """Get current parsing statistics."""
        return self.parsing_stats.copy()

    def reset_statistics(self):
        """Reset parsing statistics."""
        for key in self.parsing_stats:
            self.parsing_stats[key] = 0


# Example usage and testing
if __name__ == "__main__":
    parser = EnhancedFormGuideParser()

    # Example usage
    result = parser.parse_form_guide("example_form_guide.csv")

    print(f"Success: {result.success}")
    print(f"Records parsed: {len(result.data)}")
    print(f"Issues found: {len(result.issues)}")
    print(f"Quarantined: {result.quarantined}")

    for issue in result.issues:
        print(f"[{issue.severity.value.upper()}] {issue.message}")
