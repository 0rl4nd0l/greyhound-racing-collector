#!/usr/bin/env python3
"""
File Integrity Guardian
======================

Prevents corrupted files and test file pollution from affecting the system.
Provides validation, quarantine, and cleanup mechanisms.

Author: AI Assistant  
Date: August 4, 2025
"""

import csv
import hashlib
import json
import os
import pickle
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class FileValidationResult:
    """Result of file validation"""

    is_valid: bool
    file_path: str
    file_size: int
    issues: List[str]
    file_type: str
    should_quarantine: bool = False
    file_hash: Optional[str] = None


@dataclass
class FileHashState:
    """State for incremental file hashing"""

    file_path: str
    file_size: int
    last_modified: float
    hash_value: str
    chunk_hashes: List[str]  # Hashes of individual chunks


class FileIntegrityGuardian:
    """Guardian to prevent corrupted and test files from polluting the system"""

    def __init__(self):
        self.quarantine_dir = "./quarantine"
        # Detect testing mode (pytest) to avoid quarantining test fixtures
        self.testing_mode = (
            bool(os.environ.get("PYTEST_CURRENT_TEST"))
            or os.environ.get("TESTING") == "1"
        )
        self.max_csv_size = 50 * 1024 * 1024  # 50MB max for CSV files
        self.min_csv_size = 10  # 10 bytes minimum
        self.chunk_size = 64 * 1024  # 64KB chunks for incremental hashing
        self.hash_cache_file = "./.guardian_hash_cache.pkl"
        self.hash_cache = {}

        self.test_patterns = [
            r".*test.*\.csv$",
            r".*tmp.*test.*\.csv$",
            r".*debug.*\.csv$",
            r".*sample.*\.csv$",
            r".*mock.*\.csv$",
            r".*dummy.*\.csv$",
            r".*temp.*test.*\.csv$",
        ]

        # Testing allowance: do not quarantine test-named files when running pytest or explicitly allowed
        self.allow_test_files = (
            bool(os.getenv("PYTEST_CURRENT_TEST"))
            or os.getenv("GUARDIAN_ALLOW_TEST_FILES", "0") == "1"
        )

        # Ensure quarantine directory exists
        os.makedirs(self.quarantine_dir, exist_ok=True)

        # Load hash cache
        self._load_hash_cache()

        print("🛡️ File Integrity Guardian initialized")
        print(f"📁 Quarantine directory: {self.quarantine_dir}")
        print(
            f"📏 CSV size limits: {self.min_csv_size} bytes to {self.max_csv_size/1024/1024:.1f}MB"
        )
        print(f"🔗 Incremental hashing enabled with {self.chunk_size//1024}KB chunks")
        print(f"💾 Hash cache loaded: {len(self.hash_cache)} files")

    def is_test_file(self, file_path: str) -> bool:
        """Check if a file appears to be a test file based on naming patterns"""
        # Never treat files as test files when in testing mode or when explicitly allowed
        if self.testing_mode or self.allow_test_files:
            return False
        filename = os.path.basename(file_path).lower()

        for pattern in self.test_patterns:
            if re.match(pattern, filename, re.IGNORECASE):
                return True

        return False

    def _load_hash_cache(self):
        """Load hash cache from disk"""
        try:
            if os.path.exists(self.hash_cache_file):
                with open(self.hash_cache_file, "rb") as f:
                    self.hash_cache = pickle.load(f)
                    # Clean up stale cache entries
                    self._cleanup_hash_cache()
        except Exception as e:
            print(f"⚠️ Failed to load hash cache: {e}")
            self.hash_cache = {}

    def _save_hash_cache(self):
        """Save hash cache to disk"""
        try:
            with open(self.hash_cache_file, "wb") as f:
                pickle.dump(self.hash_cache, f)
        except Exception as e:
            print(f"⚠️ Failed to save hash cache: {e}")

    def _cleanup_hash_cache(self):
        """Remove cache entries for files that no longer exist or have changed"""
        stale_keys = []
        for file_path, hash_state in self.hash_cache.items():
            if not os.path.exists(file_path):
                stale_keys.append(file_path)
                continue

            try:
                current_mtime = os.path.getmtime(file_path)
                current_size = os.path.getsize(file_path)

                if (
                    current_mtime != hash_state.last_modified
                    or current_size != hash_state.file_size
                ):
                    stale_keys.append(file_path)
            except OSError:
                stale_keys.append(file_path)

        for key in stale_keys:
            del self.hash_cache[key]

    def _compute_incremental_hash(self, file_path: str) -> Optional[FileHashState]:
        """Compute file hash incrementally using chunks to avoid reading entire file"""
        try:
            file_stat = os.stat(file_path)
            file_size = file_stat.st_size
            last_modified = file_stat.st_mtime

            # Check if we have a valid cache entry
            if file_path in self.hash_cache:
                cached_state = self.hash_cache[file_path]
                if (
                    cached_state.last_modified == last_modified
                    and cached_state.file_size == file_size
                ):
                    return cached_state

            # Compute hash incrementally
            chunk_hashes = []
            full_hasher = hashlib.sha256()

            with open(file_path, "rb") as f:
                while True:
                    chunk = f.read(self.chunk_size)
                    if not chunk:
                        break

                    # Hash the chunk
                    chunk_hasher = hashlib.sha256()
                    chunk_hasher.update(chunk)
                    chunk_hash = chunk_hasher.hexdigest()
                    chunk_hashes.append(chunk_hash)

                    # Update full file hash
                    full_hasher.update(chunk)

                    # CPU yield to prevent high CPU usage
                    time.sleep(0.0001)

            hash_state = FileHashState(
                file_path=file_path,
                file_size=file_size,
                last_modified=last_modified,
                hash_value=full_hasher.hexdigest(),
                chunk_hashes=chunk_hashes,
            )

            # Cache the result
            self.hash_cache[file_path] = hash_state

            return hash_state

        except Exception as e:
            print(f"⚠️ Failed to compute hash for {file_path}: {e}")
            return None

    def _detect_hash_anomalies(self, hash_state: FileHashState) -> List[str]:
        """Detect anomalies in file hash patterns that might indicate corruption"""
        issues = []

        try:
            # Check for identical consecutive chunks (might indicate corruption)
            consecutive_identical = 0
            max_consecutive = 0

            for i in range(len(hash_state.chunk_hashes) - 1):
                if hash_state.chunk_hashes[i] == hash_state.chunk_hashes[i + 1]:
                    consecutive_identical += 1
                    max_consecutive = max(max_consecutive, consecutive_identical)
                else:
                    consecutive_identical = 0

            # If more than 5 consecutive chunks are identical, flag it
            if max_consecutive > 5:
                issues.append(
                    f"Suspicious pattern: {max_consecutive + 1} consecutive identical chunks detected"
                )

            # Check for predominantly empty chunks (null bytes)
            null_chunk_hash = hashlib.sha256(b"\x00" * self.chunk_size).hexdigest()
            null_chunks = sum(
                1 for h in hash_state.chunk_hashes if h == null_chunk_hash
            )

            if null_chunks > len(hash_state.chunk_hashes) * 0.5:
                issues.append(
                    f"File appears to be mostly null bytes: {null_chunks}/{len(hash_state.chunk_hashes)} chunks"
                )

            # Check for unusual hash distribution (too many repeated hashes)
            unique_hashes = len(set(hash_state.chunk_hashes))
            if (
                unique_hashes < len(hash_state.chunk_hashes) * 0.3
                and len(hash_state.chunk_hashes) > 10
            ):
                issues.append(
                    f"Low hash diversity: only {unique_hashes} unique hashes in {len(hash_state.chunk_hashes)} chunks"
                )

        except Exception as e:
            issues.append(f"Hash analysis error: {str(e)}")

        return issues

    def check_file_corruption(self, file_path: str) -> Tuple[List[str], Optional[str]]:
        """Check for common signs of file corruption using incremental hashing"""
        issues = []
        file_hash = None

        try:
            file_size = os.path.getsize(file_path)

            # Check for suspiciously round file sizes (often indicates corruption)
            if (
                file_size > 0
                and file_size % (1024 * 1024) == 0
                and file_size >= 10 * 1024 * 1024
            ):
                issues.append(f"Suspiciously round file size: {file_size} bytes")

            # Check for files that are too large or too small
            if file_size > self.max_csv_size:
                issues.append(
                    f"File too large: {file_size} bytes (max: {self.max_csv_size})"
                )

            if file_size < self.min_csv_size:
                issues.append(
                    f"File too small: {file_size} bytes (min: {self.min_csv_size})"
                )

            # Use incremental hashing for corruption detection
            hash_state = self._compute_incremental_hash(file_path)
            if hash_state:
                file_hash = hash_state.hash_value
                hash_issues = self._detect_hash_anomalies(hash_state)
                issues.extend(hash_issues)

            # Check file content for corruption signs (only read small portions)
            with open(file_path, "rb") as f:
                # Read first 1KB to check for null bytes and HTML content
                first_chunk = f.read(1024)

                # Check if file starts with mostly null bytes
                null_count = first_chunk.count(b"\x00")
                if null_count > len(first_chunk) * 0.8:  # More than 80% null bytes
                    issues.append(
                        "File appears to start with mostly null bytes (corrupted)"
                    )

                # Check for HTML content in CSV files
                if file_path.endswith(".csv"):
                    first_text = first_chunk.decode("utf-8", errors="ignore").lower()
                    if any(
                        tag in first_text
                        for tag in [
                            "<html>",
                            "<body>",
                            "<!doctype",
                            "<div>",
                            "<script>",
                        ]
                    ):
                        issues.append("CSV file contains HTML content")

        except Exception as e:
            issues.append(f"Error reading file: {str(e)}")

        return issues, file_hash

    def validate_csv_structure(self, file_path: str) -> List[str]:
        """Validate CSV file structure"""
        issues = []

        if not file_path.endswith(".csv"):
            return issues

        try:
            # Temporarily increase CSV field size limit for validation
            original_limit = csv.field_size_limit()
            csv.field_size_limit(self.max_csv_size)

            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)

                # Check if we can read the first row (headers)
                try:
                    headers = next(reader)
                    if not headers or len(headers) == 0:
                        issues.append("CSV has no headers")
                    elif len(headers) == 1 and headers[0].strip() == "":
                        issues.append("CSV has empty header row")
                except StopIteration:
                    issues.append("CSV file is empty")
                except Exception as e:
                    issues.append(f"CSV parsing error: {str(e)}")

                # Try to read a few more rows
                row_count = 0
                try:
                    for _ in range(5):  # Check first 5 rows
                        next(reader)
                        row_count += 1
                except StopIteration:
                    pass  # Normal end of file
                except Exception as e:
                    issues.append(f"CSV row parsing error: {str(e)}")

            # Restore original CSV field size limit
            csv.field_size_limit(original_limit)

        except Exception as e:
            issues.append(f"CSV validation error: {str(e)}")

        return issues

    def validate_file(self, file_path: str) -> FileValidationResult:
        """Comprehensive file validation"""
        issues = []
        should_quarantine = False

        if not os.path.exists(file_path):
            return FileValidationResult(
                is_valid=False,
                file_path=file_path,
                file_size=0,
                issues=["File does not exist"],
                file_type="unknown",
                should_quarantine=False,
            )

        file_size = os.path.getsize(file_path)
        file_ext = Path(file_path).suffix.lower()

        # Check if it's a test file
        if self.is_test_file(file_path):
            issues.append("File appears to be a test file")
            should_quarantine = True

        # Check for corruption using incremental hashing
        corruption_issues, file_hash = self.check_file_corruption(file_path)
        issues.extend(corruption_issues)

        # If there are corruption issues, quarantine the file
        if corruption_issues:
            should_quarantine = True

        # Validate CSV structure if applicable
        if file_ext == ".csv":
            csv_issues = self.validate_csv_structure(file_path)
            issues.extend(csv_issues)

            # Critical CSV issues should trigger quarantine
            critical_csv_issues = [
                "CSV parsing error",
                "CSV file is empty",
                "CSV has no headers",
                "File appears to be mostly null bytes",
                "CSV file contains HTML content",
            ]

            for issue in issues:
                if any(critical in issue for critical in critical_csv_issues):
                    should_quarantine = True
                    break

        is_valid = len(issues) == 0 or not should_quarantine

        return FileValidationResult(
            is_valid=is_valid,
            file_path=file_path,
            file_size=file_size,
            issues=issues,
            file_type=file_ext,
            should_quarantine=should_quarantine,
            file_hash=file_hash,
        )

    def quarantine_file(self, file_path: str, reason: str) -> bool:
        """Move a problematic file to quarantine"""
        try:
            if not os.path.exists(file_path):
                print(f"⚠️ Cannot quarantine non-existent file: {file_path}")
                return False

            filename = os.path.basename(file_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            quarantine_filename = f"{timestamp}_{filename}"
            quarantine_path = os.path.join(self.quarantine_dir, quarantine_filename)

            # Move file to quarantine
            os.rename(file_path, quarantine_path)

            # Create quarantine report
            report = {
                "original_path": file_path,
                "quarantine_path": quarantine_path,
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
                "file_size": os.path.getsize(quarantine_path),
            }

            report_path = quarantine_path + ".report.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

            print(f"🚨 QUARANTINED: {file_path} -> {quarantine_path}")
            print(f"📝 Reason: {reason}")

            return True

        except Exception as e:
            print(f"❌ Failed to quarantine {file_path}: {e}")
            return False

    def scan_directory(
        self,
        directory: str,
        extensions: List[str] = [".csv"],
        max_age_hours: int = None,
    ) -> List[FileValidationResult]:
        """Scan directory for problematic files"""
        results = []

        if not os.path.exists(directory):
            print(f"⚠️ Directory does not exist: {directory}")
            return results

        print(f"🔍 Scanning directory: {directory}")

        # Calculate cutoff time for file age filtering
        cutoff_time = None
        if max_age_hours is not None:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        for root, dirs, files in os.walk(directory):
            for file in files:
                # CPU yield to prevent high CPU usage
                time.sleep(0.001)

                if any(file.lower().endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)

                    # Apply file age filtering if specified
                    if cutoff_time is not None:
                        try:
                            file_mtime = datetime.fromtimestamp(
                                os.path.getmtime(file_path)
                            )
                            if file_mtime < cutoff_time:
                                continue  # Skip old files
                        except OSError:
                            continue  # Skip files we can't access

                    result = self.validate_file(file_path)
                    results.append(result)

                    if result.should_quarantine:
                        issues_summary = "; ".join(result.issues)
                        self.quarantine_file(
                            file_path, f"Validation failed: {issues_summary}"
                        )

        # Save hash cache after scan
        self._save_hash_cache()

        return results

    def cleanup_test_files(
        self, directories: List[str], max_age_hours: int = None
    ) -> int:
        """Remove test files from specified directories"""
        # In testing mode or when explicitly allowed, never remove test files
        if self.testing_mode or self.allow_test_files:
            return 0
        removed_count = 0

        # Calculate cutoff time for file age filtering
        cutoff_time = None
        if max_age_hours is not None:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        for directory in directories:
            if not os.path.exists(directory):
                continue

            for root, dirs, files in os.walk(directory):
                for file in files:
                    # CPU yield to prevent high CPU usage
                    time.sleep(0.001)

                    file_path = os.path.join(root, file)

                    # Apply file age filtering if specified
                    if cutoff_time is not None:
                        try:
                            file_mtime = datetime.fromtimestamp(
                                os.path.getmtime(file_path)
                            )
                            if file_mtime < cutoff_time:
                                continue  # Skip old files
                        except OSError:
                            continue  # Skip files we can't access

                    if self.is_test_file(file_path):
                        try:
                            print(f"🗑️ Removing test file: {file_path}")
                            os.remove(file_path)
                            removed_count += 1
                        except Exception as e:
                            print(f"❌ Failed to remove {file_path}: {e}")

        return removed_count

    def generate_integrity_report(self, results: List[FileValidationResult]) -> Dict:
        """Generate comprehensive integrity report"""
        total_files = len(results)
        valid_files = len([r for r in results if r.is_valid])
        quarantined_files = len([r for r in results if r.should_quarantine])

        issues_by_type = {}
        for result in results:
            for issue in result.issues:
                issues_by_type[issue] = issues_by_type.get(issue, 0) + 1

        report = {
            "scan_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_files": total_files,
                "valid_files": valid_files,
                "invalid_files": total_files - valid_files,
                "quarantined_files": quarantined_files,
            },
            "issues_by_type": issues_by_type,
            "problematic_files": [
                {
                    "path": r.file_path,
                    "size": r.file_size,
                    "issues": r.issues,
                    "quarantined": r.should_quarantine,
                }
                for r in results
                if not r.is_valid
            ],
        }

        return report


def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="File Integrity Guardian")
    parser.add_argument(
        "--scan", type=str, help="Directory to scan for problematic files"
    )
    parser.add_argument(
        "--cleanup-tests",
        action="store_true",
        help="Remove test files from main directories",
    )
    parser.add_argument("--validate-file", type=str, help="Validate a specific file")
    parser.add_argument(
        "--report", type=str, help="Generate integrity report and save to file"
    )

    args = parser.parse_args()

    guardian = FileIntegrityGuardian()

    if args.validate_file:
        result = guardian.validate_file(args.validate_file)
        print(f"\n📊 Validation Result for {args.validate_file}:")
        print(f"   Valid: {result.is_valid}")
        print(f"   Size: {result.file_size} bytes")
        print(f"   Issues: {result.issues}")
        print(f"   Should Quarantine: {result.should_quarantine}")

        if result.should_quarantine:
            guardian.quarantine_file(args.validate_file, "; ".join(result.issues))

    elif args.scan:
        results = guardian.scan_directory(args.scan)
        report = guardian.generate_integrity_report(results)

        print(f"\n📊 Integrity Scan Results:")
        print(f"   Total files: {report['summary']['total_files']}")
        print(f"   Valid files: {report['summary']['valid_files']}")
        print(f"   Invalid files: {report['summary']['invalid_files']}")
        print(f"   Quarantined files: {report['summary']['quarantined_files']}")

        if args.report:
            with open(args.report, "w") as f:
                json.dump(report, f, indent=2)
            print(f"📝 Report saved to: {args.report}")

    elif args.cleanup_tests:
        directories = ["./upcoming_races", "./processed", "./unprocessed"]
        removed = guardian.cleanup_test_files(directories)
        print(f"🧹 Removed {removed} test files")

    else:
        print("🛡️ File Integrity Guardian")
        print("Use --help for available options")


if __name__ == "__main__":
    main()
