#!/usr/bin/env python3
"""
In-place Race Info Patcher
==========================

Patches missing distance and grade fields in prediction JSON files using multiple data sources.
This script serves as a fallback enrichment strategy when API re-prediction is not viable.

Usage:
    python scripts/patch_race_info_inplace.py [--manifest PATH] [--dry-run] [--backup-dir PATH]
    python scripts/patch_race_info_inplace.py --manifest ./reports/missing_race_info_manifest_*.json

Data Sources (in order of preference):
1. race_url: Fetch and parse race webpage (if URL exists and accessible)
2. Co-located race_data: Check for JSON/HTML files previously scraped for the same race
3. Filename patterns: Extract from file paths (e.g., "..._515m_Grade5_predictions.json")
4. JSON content fields: Extract from race_name, event_name, meeting_name
5. Form guide headers: Only if they contain upcoming race metadata (not historical dog data)

Environment Variables:
- BACKUP_DIR: backup directory (default: ./archive/predictions_fix/{timestamp})
- DRY_RUN: if "1", perform dry run without modifications
- MANIFEST_PATH: path to missing files manifest JSON
"""

import argparse
import json
import os
import re
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin, urlparse

# Optional HTTP dependencies
try:
    import requests
    from bs4 import BeautifulSoup

    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.logging_config import get_component_logger  # type: ignore
from config.venue_mapping import VENUE_MAPPING  # type: ignore
from config.venue_mapping import normalize_venue

log = get_component_logger()

# Grade normalization patterns
GRADE_PATTERNS = {
    r"\bGrade\s*(\d+)\b": r"\1",
    r"\bG(\d+)\b": r"\1",
    r"\b(Maiden)\b": r"Maiden",
    r"^M$": r"Maiden",  # Single M for Maiden
    r"\b(Open)\b": r"Open",
    r"\b(Mixed\s*\d+/\d+)\b": r"\1",
    r"\b(Group\s*\d+)\b": r"\1",
    r"\b(FFA)\b": r"FFA",
}

# Distance normalization - convert to meters integer if numeric, otherwise preserve
DISTANCE_PATTERNS = [
    r"(\d{3,4})\s*m\b",  # "520m" -> "520"
    r"\b(\d{3,4})\b(?=.*(?:meter|metre))",  # "520 meters" -> "520"
    r"(\d{3,4})\s*(?=\s|$)",  # standalone numbers like "520"
]


class RaceInfoPatcher:
    """Main patcher class for enriching race_info fields."""

    def __init__(self, backup_dir: Path, dry_run: bool = False):
        self.backup_dir = backup_dir
        self.dry_run = dry_run
        self.patched_count = 0
        self.skipped_count = 0
        self.error_count = 0
        self.patched_files = []

        # Create backup directory
        if not dry_run:
            backup_dir.mkdir(parents=True, exist_ok=True)

    def normalize_distance(self, distance_text: str) -> Optional[str]:
        """Extract and normalize distance from text."""
        if not distance_text:
            return None

        distance_text = str(distance_text).strip()

        # Try distance patterns
        for pattern in DISTANCE_PATTERNS:
            match = re.search(pattern, distance_text, re.IGNORECASE)
            if match:
                distance_num = match.group(1)
                # Convert to integer meters if possible
                try:
                    return f"{int(distance_num)}"
                except ValueError:
                    return f"{distance_num}m"

        return None

    def normalize_grade(self, grade_text: str) -> Optional[str]:
        """Extract and normalize grade from text."""
        if not grade_text:
            return None

        grade_text = str(grade_text).strip()

        # Apply grade normalization patterns
        for pattern, replacement in GRADE_PATTERNS.items():
            match = re.search(pattern, grade_text, re.IGNORECASE)
            if match:
                if "\\1" in replacement:
                    return match.group(1)
                else:
                    return replacement

        return None

    def extract_from_race_url(
        self, race_url: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract distance and grade from race webpage."""
        if not race_url or not HTTP_AVAILABLE:
            if not HTTP_AVAILABLE:
                log.debug("HTTP dependencies not available, skipping URL extraction")
            return None, None

        try:
            # Simple HTTP request with timeout
            response = requests.get(
                race_url,
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0 (compatible; RaceInfoPatcher/1.0)"},
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            page_text = soup.get_text()

            # Extract distance and grade from page text
            distance = self.normalize_distance(page_text)
            grade = self.normalize_grade(page_text)

            return distance, grade

        except Exception as e:
            log.warning(
                f"Failed to extract from race URL: {e}",
                action="url_extraction_failed",
                details={"race_url": race_url, "error": str(e)},
                component="race_info_patcher",
            )
            return None, None

    def find_colocated_race_data(
        self, identifiers: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Look for co-located race data files in common directories."""
        track = identifiers.get("track", "").upper()
        date = identifiers.get("date", "")
        race_number = str(identifiers.get("race_number", "")).strip()

        if not all([track, date, race_number]):
            return None, None

        # Check common data directories (avoid large processed directory)
        search_dirs = [Path("./data/races"), Path("./data/scrapes"), Path("./archive")]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            # Look for files matching race pattern
            patterns = [
                f"*{track}*{date}*{race_number}*.json",
                f"*{track}*{date}*.json",
                f"*{date}*{track}*{race_number}*.json",
            ]

            for pattern in patterns:
                matching_files = list(search_dir.rglob(pattern))
                for file_path in matching_files:
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        # Extract race info if present
                        race_info = data.get("race_info", {})
                        distance = race_info.get("distance")
                        grade = race_info.get("grade")

                        if distance or grade:
                            return (
                                self.normalize_distance(distance) if distance else None,
                                self.normalize_grade(grade) if grade else None,
                            )
                    except Exception:
                        continue

        return None, None

    def extract_from_filename(
        self, file_path: Path
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract distance and grade from filename patterns."""
        filename = file_path.name
        parent_dir = file_path.parent.name if file_path.parent else ""
        full_path = str(file_path)

        # Try to extract from various path components
        text_to_search = f"{filename} {parent_dir} {full_path}"

        distance = self.normalize_distance(text_to_search)
        grade = self.normalize_grade(text_to_search)

        return distance, grade

    def extract_venue_from_filename(self, file_path: Path) -> Optional[str]:
        """Extract venue from filename patterns for UNKNOWN venues."""
        filename = file_path.name
        parent_dir = file_path.parent.name if file_path.parent else ""
        full_path = str(file_path)

        # Common filename patterns for venues
        venue_patterns = [
            r"\b(WAR|WARRAGUL|WARRNAMBOOL)\b",
            r"\b(AP|ALBION[_\s]?PARK)\b",
            r"\b(BAL|BALLARAT)\b",
            r"\b(BEN|BENDIGO)\b",
            r"\b(DAPT|DAPTO)\b",
            r"\b(GOUL|GOULBURN)\b",
            r"\b(HEAL|HEA|HEALESVILLE)\b",
            r"\b(IPS|IPSWICH)\b",
            r"\b(MAITLAND)\b",
            r"\b(MEA|MEADOWS)\b",
            r"\b(MURRAY|MURRAY[_\s]?BRIDGE)\b",
            r"\b(NOWRA)\b",
            r"\b(RICH|RICHMOND)\b",
            r"\b(SHEP|SHE|SHEPPARTON)\b",
            r"\b(TAR|TAREE)\b",
            r"\b(GARD|GRDN|THE[_\s]?GARDENS)\b",
            r"\b(WP|W_PK|WENTWORTH[_\s]?PARK)\b",
            r"\b(QOT|LADBROKES[_-]?Q[_-]?STRAIGHT)\b",
            r"\b(SAN|SANDOWN)\b",
            r"\b(ROCK|ROCKHAMPTON)\b",
            r"\b(MT_G|MOUNT[_\s]?GAMBIER)\b",
            r"\b(CANN|CANNINGTON)\b",
            r"\b(SAL|SALE)\b",
            r"\b(GEE|GEELONG)\b",
            r"\b(DUBBO)\b",
        ]

        # Search in filename, parent directory, and full path
        text_to_search = f"{filename} {parent_dir} {full_path}"

        for pattern in venue_patterns:
            match = re.search(pattern, text_to_search, re.IGNORECASE)
            if match:
                venue_raw = match.group(1).upper()
                # Normalize the venue using our mapping
                venue_normalized = VENUE_MAPPING.get(venue_raw, venue_raw)
                log.info(
                    f"Extracted venue from filename: {venue_raw} -> {venue_normalized}",
                    action="venue_extraction_success",
                    details={
                        "file": str(file_path),
                        "venue_raw": venue_raw,
                        "venue_normalized": venue_normalized,
                    },
                    component="race_info_patcher",
                )
                return venue_normalized

        return None

    def extract_from_json_content(
        self, data: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract distance and grade from JSON content fields."""
        # Fields to search in
        search_fields = [
            "race_name",
            "event_name",
            "meeting_name",
            "title",
            "description",
            "race_title",
        ]

        text_to_search = ""
        for field in search_fields:
            value = data.get(field, "")
            if value:
                text_to_search += f" {value}"

        # Also check race_info and race_context for text fields
        race_info = data.get("race_info", {})
        race_context = data.get("race_context", {})

        for info_dict in [race_info, race_context]:
            for key, value in info_dict.items():
                if isinstance(value, str):
                    text_to_search += f" {value}"

        distance = self.normalize_distance(text_to_search)
        grade = self.normalize_grade(text_to_search)

        return distance, grade

    def find_csv_file_fuzzy(
        self, filename: str, search_dirs: List[Path]
    ) -> Optional[Path]:
        """Find CSV file using fuzzy matching to handle naming variations."""
        if not filename:
            return None

        # Normalize the target filename for comparison
        def normalize_name(name: str) -> str:
            """Normalize filename for fuzzy matching."""
            # Remove .csv extension, convert to lowercase, normalize separators
            name = name.lower().replace(".csv", "")
            # Replace various separators with spaces
            for sep in ["_", "-", "."]:
                name = name.replace(sep, " ")
            # Remove extra spaces
            return " ".join(name.split())

        target_normalized = normalize_name(filename)

        import sys

        print(f"[DEBUG] Looking for CSV file: {filename}", file=sys.stderr)
        print(f"[DEBUG] Target normalized: {target_normalized}", file=sys.stderr)

        for search_dir in search_dirs:
            if not search_dir.exists():
                print(f"[DEBUG] Directory doesn't exist: {search_dir}")
                continue

            try:
                print(f"[DEBUG] Searching in: {search_dir}")
                # Search recursively for CSV files
                csv_files = list(search_dir.rglob("*.csv"))
                print(f"[DEBUG] Found {len(csv_files)} CSV files")

                for csv_file in csv_files:
                    if csv_file.is_file():
                        file_normalized = normalize_name(csv_file.name)
                        print(f"[DEBUG] Checking: {csv_file.name} -> {file_normalized}")

                        # Check for exact match after normalization
                        if target_normalized == file_normalized:
                            log.info(
                                f"Found CSV file via fuzzy matching: {filename} -> {csv_file.name}",
                                action="csv_fuzzy_match_success",
                                details={"target": filename, "found": str(csv_file)},
                                component="race_info_patcher",
                            )
                            print(f"[DEBUG] EXACT MATCH FOUND: {csv_file}")
                            return csv_file

                        # Check for partial matches (contains all key parts)
                        target_parts = set(target_normalized.split())
                        file_parts = set(file_normalized.split())

                        # If target parts are mostly contained in file parts
                        if (
                            len(target_parts - file_parts) <= 1
                            and len(target_parts) > 2
                        ):
                            log.info(
                                f"Found CSV file via partial matching: {filename} -> {csv_file.name}",
                                action="csv_partial_match_success",
                                details={
                                    "target": filename,
                                    "found": str(csv_file),
                                    "target_parts": list(target_parts),
                                    "file_parts": list(file_parts),
                                },
                                component="race_info_patcher",
                            )
                            print(f"[DEBUG] PARTIAL MATCH FOUND: {csv_file}")
                            return csv_file
            except Exception as e:
                log.debug(f"Error searching in {search_dir}: {e}")
                print(f"[DEBUG] Error searching in {search_dir}: {e}")
                continue

        print(f"[DEBUG] No CSV file found for: {filename}")
        return None

    def extract_from_csv_headers(
        self, identifiers: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract from CSV headers if they contain race-level metadata."""
        filename = identifiers.get("filename", "")
        if not filename:
            return None, None

        # Try to find the CSV file - search recursively in processed directories
        csv_search_dirs = [
            Path("./upcoming_races"),
            Path("./data/upcoming_races"),
            Path("./processed"),
            Path("./data/processed"),
        ]

        # Try direct matching first
        for search_dir in csv_search_dirs:
            if not search_dir.exists():
                continue

            # First try direct path
            csv_path = search_dir / filename
            if csv_path.exists():
                try:
                    distance, grade = self._extract_from_csv_file(csv_path)
                    if distance or grade:
                        return distance, grade

                except Exception:
                    continue

        # If direct matching failed, try fuzzy matching
        found_csv = self.find_csv_file_fuzzy(filename, csv_search_dirs)
        if found_csv:
            try:
                distance, grade = self._extract_from_csv_file(found_csv)
                if distance or grade:
                    return distance, grade
            except Exception as e:
                log.debug(f"Error extracting from fuzzy-matched CSV {found_csv}: {e}")

        return None, None

    def _extract_from_csv_file(
        self, csv_path: Path
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract distance and grade from a CSV file."""
        # Read first few lines to get headers and any metadata
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = [
                f.readline().strip() for _ in range(10)
            ]  # Read more lines to get data rows

        # Join all lines for analysis
        full_text = " ".join(lines)

        # Check if this looks like dog form data (avoid extracting from historical data)
        if re.search(r"\b(dog|greyhound|runner)\b", full_text, re.IGNORECASE):
            # This looks like a form guide with historical dog data
            # Extract distance and grade from the first data row instead of headers

            # Look for distance in column headers or data
            distance = None
            grade = None

            # Parse as CSV-like structure
            for line in lines:
                if not line or line.startswith(
                    '""'
                ):  # Skip empty lines or continuation lines
                    continue

                parts = [part.strip('"').strip() for part in line.split(",")]

                if len(parts) > 5:  # Ensure we have enough columns
                    # Look for distance column (often labeled DIST or similar)
                    for i, part in enumerate(parts):
                        # Check for distance patterns
                        if re.match(
                            r"^\d{3,4}$", part
                        ):  # 3-4 digit numbers (likely distance)
                            potential_distance = self.normalize_distance(part)
                            if potential_distance:
                                distance = potential_distance
                                break

                    # Look for grade column (often labeled G or Grade)
                    for part in parts:
                        potential_grade = self.normalize_grade(part)
                        if potential_grade:
                            grade = potential_grade
                            break

                # If we found both, we can return
                if distance and grade:
                    break
        else:
            # Extract from headers/metadata (race-level data)
            distance = self.normalize_distance(full_text)
            grade = self.normalize_grade(full_text)

        return distance, grade

    def query_database_fallback(
        self, identifiers: Dict[str, Any], file_path: Path
    ) -> Tuple[Optional[str], Optional[str]]:
        """Query race_metadata database as final fallback."""
        venue = identifiers.get("track")
        date = identifiers.get("date")
        race_number = identifiers.get("race_number")

        # If venue is UNKNOWN, try to extract from filename
        if venue == "UNKNOWN" or not venue:
            extracted_venue = self.extract_venue_from_filename(file_path)
            if extracted_venue:
                venue = extracted_venue
                log.info(
                    f"Using extracted venue for database lookup: {extracted_venue}",
                    action="venue_extraction_for_db",
                    details={
                        "file": str(file_path),
                        "extracted_venue": extracted_venue,
                    },
                    component="race_info_patcher",
                )

        if not all([venue, date, race_number]):
            return None, None

        # Normalize venue
        venue_normalized = (
            VENUE_MAPPING.get(venue.upper(), venue.upper()) if venue else None
        )

        # Database paths to try
        db_paths = [
            Path("./greyhound_racing_data.db"),
            Path("./database.sqlite"),
            Path("./data/greyhound_racing_data.db"),
        ]

        for db_path in db_paths:
            if not db_path.exists():
                continue

            try:
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()

                    # Query race_metadata table
                    query = """
                        SELECT distance, grade 
                        FROM race_metadata 
                        WHERE venue = ? AND race_date = ? AND race_number = ?
                        LIMIT 1
                    """

                    cursor.execute(query, (venue_normalized, date, str(race_number)))
                    result = cursor.fetchone()

                    if result:
                        distance_raw, grade_raw = result
                        distance = (
                            self.normalize_distance(str(distance_raw))
                            if distance_raw
                            else None
                        )
                        grade = (
                            self.normalize_grade(str(grade_raw)) if grade_raw else None
                        )

                        if distance or grade:
                            return distance, grade

            except Exception as e:
                log.debug(f"Database query failed for {db_path}: {e}")
                continue

        return None, None

    def derive_race_info(
        self, file_info: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Derive distance and grade using multiple data sources."""
        file_path = Path(file_info["absolute_path"])
        identifiers = file_info["identifiers"]
        current_race_info = file_info["current_race_info"]
        missing_fields = file_info["missing_fields"]

        print(
            f"[DEBUG] Starting derive_race_info for: {file_path.name}", file=sys.stderr
        )
        print(f"[DEBUG] Missing fields: {missing_fields}", file=sys.stderr)
        print(f"[DEBUG] Identifiers: {identifiers}", file=sys.stderr)
        print(f"[DEBUG] Current race_info: {current_race_info}", file=sys.stderr)

        # Track what we've found so far
        final_distance = None
        final_grade = None

        # Check what we need to find
        needs_distance = missing_fields.get("distance", False)
        needs_grade = missing_fields.get("grade", False)

        # Strategy 1: race_url (if present and accessible)
        race_url = identifiers.get("race_url") or current_race_info.get("race_url")
        print(f"[DEBUG] Strategy 1 - Race URL: {race_url}", file=sys.stderr)
        if race_url and (not final_distance or not final_grade):
            distance, grade = self.extract_from_race_url(race_url)
            print(
                f"[DEBUG] Strategy 1 result: distance={distance}, grade={grade}",
                file=sys.stderr,
            )
            if distance and needs_distance and not final_distance:
                final_distance = distance
                log.info(
                    f"Extracted distance from race URL: {distance}",
                    action="extraction_success",
                    details={
                        "source": "race_url",
                        "field": "distance",
                        "file": str(file_path),
                    },
                    component="race_info_patcher",
                )
            if grade and needs_grade and not final_grade:
                final_grade = grade
                log.info(
                    f"Extracted grade from race URL: {grade}",
                    action="extraction_success",
                    details={
                        "source": "race_url",
                        "field": "grade",
                        "file": str(file_path),
                    },
                    component="race_info_patcher",
                )

        # Strategy 2: Co-located race data (disabled for performance)
        print(f"[DEBUG] Strategy 2 - Co-located race data: DISABLED", file=sys.stderr)

        # Strategy 3: Filename patterns
        print(f"[DEBUG] Strategy 3 - Filename patterns", file=sys.stderr)
        if (needs_distance and not final_distance) or (needs_grade and not final_grade):
            distance, grade = self.extract_from_filename(file_path)
            print(
                f"[DEBUG] Strategy 3 result: distance={distance}, grade={grade}",
                file=sys.stderr,
            )
            if distance and needs_distance and not final_distance:
                final_distance = distance
                log.info(
                    f"Extracted distance from filename: {distance}",
                    action="extraction_success",
                    details={
                        "source": "filename",
                        "field": "distance",
                        "file": str(file_path),
                    },
                    component="race_info_patcher",
                )
            if grade and needs_grade and not final_grade:
                final_grade = grade
                log.info(
                    f"Extracted grade from filename: {grade}",
                    action="extraction_success",
                    details={
                        "source": "filename",
                        "field": "grade",
                        "file": str(file_path),
                    },
                    component="race_info_patcher",
                )

        # Strategy 4: JSON content fields
        print(f"[DEBUG] Strategy 4 - JSON content fields", file=sys.stderr)
        if (needs_distance and not final_distance) or (needs_grade and not final_grade):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                distance, grade = self.extract_from_json_content(data)
                print(
                    f"[DEBUG] Strategy 4 result: distance={distance}, grade={grade}",
                    file=sys.stderr,
                )
                if distance and needs_distance and not final_distance:
                    final_distance = distance
                    log.info(
                        f"Extracted distance from JSON content: {distance}",
                        action="extraction_success",
                        details={
                            "source": "json_content",
                            "field": "distance",
                            "file": str(file_path),
                        },
                        component="race_info_patcher",
                    )
                if grade and needs_grade and not final_grade:
                    final_grade = grade
                    log.info(
                        f"Extracted grade from JSON content: {grade}",
                        action="extraction_success",
                        details={
                            "source": "json_content",
                            "field": "grade",
                            "file": str(file_path),
                        },
                        component="race_info_patcher",
                    )
            except Exception:
                pass

        # Strategy 5: CSV headers
        print(f"[DEBUG] Strategy 5 - CSV headers", file=sys.stderr)
        if (needs_distance and not final_distance) or (needs_grade and not final_grade):
            distance, grade = self.extract_from_csv_headers(identifiers)
            print(
                f"[DEBUG] Strategy 5 result: distance={distance}, grade={grade}",
                file=sys.stderr,
            )
            if distance and needs_distance and not final_distance:
                final_distance = distance
                log.info(
                    f"Extracted distance from CSV headers: {distance}",
                    action="extraction_success",
                    details={
                        "source": "csv_headers",
                        "field": "distance",
                        "file": str(file_path),
                    },
                    component="race_info_patcher",
                )
            if grade and needs_grade and not final_grade:
                final_grade = grade
                log.info(
                    f"Extracted grade from CSV headers: {grade}",
                    action="extraction_success",
                    details={
                        "source": "csv_headers",
                        "field": "grade",
                        "file": str(file_path),
                    },
                    component="race_info_patcher",
                )

        # Strategy 6: Database fallback
        print(f"[DEBUG] Strategy 6 - Database fallback", file=sys.stderr)
        if (needs_distance and not final_distance) or (needs_grade and not final_grade):
            distance, grade = self.query_database_fallback(identifiers, file_path)
            print(
                f"[DEBUG] Strategy 6 result: distance={distance}, grade={grade}",
                file=sys.stderr,
            )
            if distance and needs_distance and not final_distance:
                final_distance = distance
                log.info(
                    f"Extracted distance from database: {distance}",
                    action="extraction_success",
                    details={
                        "source": "database",
                        "field": "distance",
                        "file": str(file_path),
                    },
                    component="race_info_patcher",
                )
            if grade and needs_grade and not final_grade:
                final_grade = grade
                log.info(
                    f"Extracted grade from database: {grade}",
                    action="extraction_success",
                    details={
                        "source": "database",
                        "field": "grade",
                        "file": str(file_path),
                    },
                    component="race_info_patcher",
                )

        print(
            f"[DEBUG] Final results: distance={final_distance}, grade={final_grade}",
            file=sys.stderr,
        )
        return final_distance, final_grade

    def is_test_or_synthetic_file(self, file_path: Path) -> bool:
        """Check if a file is a test or synthetic file that should be ignored."""
        file_path_str = str(file_path).lower()
        filename = file_path.name.lower()

        # Common test/synthetic file patterns (be more specific to avoid false positives)
        test_patterns = [
            r"\btest_race\b",  # test_race_1.json but not race_4.json
            r"_test_\w+",  # prediction_test_race but not race_test
            r"test_\w+_prediction",  # test_something_prediction
            r"\bsample\b",
            r"\bdummy\b",
            r"\bsynthetic\b",
            r"\bmock\b",
            r"\bfake\b",
            r"\bexample_id\b",  # example_id but not example_race
            r"\bexample\b(?!.*race)",  # example but not if followed by race
            r"\bdemo\b",
            r"predictions/test/",  # Only match test in predictions path not user path
            r"/tests/",
            r"batch_chunk",  # batch processing files
        ]

        for pattern in test_patterns:
            if re.search(pattern, file_path_str):
                log.debug(
                    f"File matches test pattern '{pattern}': {filename}",
                    action="test_file_identified",
                    details={"file": str(file_path), "pattern": pattern},
                    component="race_info_patcher",
                )
                return True
        return False

    def patch_file(self, file_info: Dict[str, Any]) -> bool:
        """Patch a single prediction JSON file."""
        file_path = Path(file_info["absolute_path"])
        missing_fields = file_info["missing_fields"]

        # Skip test/synthetic files
        if self.is_test_or_synthetic_file(file_path):
            log.info(
                f"Skipping test/synthetic file: {file_path.name}",
                action="patch_skipped",
                details={"file": str(file_path), "reason": "Test/synthetic file"},
                component="race_info_patcher",
            )
            return False

        if not file_path.exists():
            log.error(
                f"File not found: {file_path}",
                action="patch_error",
                details={"file": str(file_path), "error": "File not found"},
                component="race_info_patcher",
            )
            return False

        try:
            # Derive distance and grade
            distance, grade = self.derive_race_info(file_info)

            # Check if we have anything to patch
            needs_distance = missing_fields.get("distance", False)
            needs_grade = missing_fields.get("grade", False)

            # Check if we found what we need
            has_needed_distance = not needs_distance or distance
            has_needed_grade = not needs_grade or grade

            if not (has_needed_distance and has_needed_grade):
                log.warning(
                    f"Could not derive missing fields for {file_path.name}",
                    action="patch_skipped",
                    details={
                        "file": str(file_path),
                        "reason": "No derivable data found",
                        "needed": {"distance": needs_distance, "grade": needs_grade},
                        "found": {"distance": distance, "grade": grade},
                    },
                    component="race_info_patcher",
                )
                return False

            # Load and patch the JSON
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Ensure race_info exists
            if "race_info" not in data:
                data["race_info"] = {}

            # Patch missing fields
            patched_fields = {}
            if needs_distance and distance:
                data["race_info"]["distance"] = distance
                patched_fields["distance"] = distance

            if needs_grade and grade:
                data["race_info"]["grade"] = grade
                patched_fields["grade"] = grade

            if not patched_fields:
                return False

            # Backup original file
            if not self.dry_run:
                backup_path = self.backup_dir / file_path.relative_to(Path.cwd())
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, backup_path)

            # Write patched file
            if not self.dry_run:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=str)

            log.info(
                f"Patched file successfully: {patched_fields}",
                action="patch_success",
                details={
                    "file": str(file_path),
                    "patched_fields": patched_fields,
                    "dry_run": self.dry_run,
                },
                component="race_info_patcher",
            )

            self.patched_files.append(
                {
                    "file_path": str(file_path),
                    "patched_fields": patched_fields,
                    "backup_path": (
                        str(self.backup_dir / file_path.relative_to(Path.cwd()))
                        if not self.dry_run
                        else None
                    ),
                }
            )

            return True

        except Exception as e:
            log.error(
                f"Failed to patch file {file_path}: {e}",
                action="patch_error",
                details={"file": str(file_path), "error": str(e)},
                component="race_info_patcher",
            )
            return False

    def patch_from_manifest(self, manifest_path: Path) -> Dict[str, Any]:
        """Patch all files listed in missing files manifest."""
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        files_to_patch = manifest.get("files", [])

        log.info(
            f"Starting in-place patching from manifest",
            action="patch_start",
            details={
                "manifest": str(manifest_path),
                "total_files": len(files_to_patch),
                "dry_run": self.dry_run,
            },
            component="race_info_patcher",
        )

        for i, file_info in enumerate(files_to_patch):
            success = self.patch_file(file_info)
            if success:
                self.patched_count += 1
            else:
                self.skipped_count += 1

            # Progress logging
            if (i + 1) % 50 == 0 or i + 1 == len(files_to_patch):
                log.info(
                    f"Patch progress: {i + 1}/{len(files_to_patch)} files processed",
                    action="patch_progress",
                    details={
                        "processed": i + 1,
                        "total": len(files_to_patch),
                        "patched": self.patched_count,
                        "skipped": self.skipped_count,
                    },
                    component="race_info_patcher",
                )

        # Generate summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "manifest_file": str(manifest_path),
            "total_files": len(files_to_patch),
            "patched_count": self.patched_count,
            "skipped_count": self.skipped_count,
            "error_count": self.error_count,
            "success_rate": (
                f"{(self.patched_count / len(files_to_patch) * 100):.1f}%"
                if files_to_patch
                else "0.0%"
            ),
            "patched_files": self.patched_files,
            "dry_run": self.dry_run,
        }

        log.info(
            f"In-place patching completed",
            action="patch_complete",
            details=summary,
            component="race_info_patcher",
        )

        return summary


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Patch missing race_info distance/grade fields in prediction JSONs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--manifest",
        "-m",
        help="Path to missing files manifest JSON",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--backup-dir",
        "-b",
        help="Backup directory for original files",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--dry-run", help="Perform dry run without making changes", action="store_true"
    )

    parser.add_argument(
        "--reports-dir",
        "-r",
        help="Output directory for reports (default: ./reports)",
        type=Path,
        default=Path("./reports"),
    )

    args = parser.parse_args(argv)

    # Set up backup directory
    if args.backup_dir:
        backup_dir = args.backup_dir.resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path(f"./archive/predictions_fix/{timestamp}").resolve()

    # Set up reports directory
    reports_dir = args.reports_dir.resolve()
    if not args.dry_run:
        reports_dir.mkdir(parents=True, exist_ok=True)

    # Create patcher and run
    patcher = RaceInfoPatcher(backup_dir, args.dry_run)

    try:
        summary = patcher.patch_from_manifest(args.manifest.resolve())

        # Write summary report
        if not args.dry_run:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_path = reports_dir / f"patched_inplace_{timestamp}.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=str)

            log_path = reports_dir.parent / "logs" / f"patch_inplace_{timestamp}.log"
            print(f"\nSummary written to: {summary_path}")
            print(f"Detailed logs: {log_path}")

        # Print summary
        print(f"\n=== In-Place Patching Summary ===")
        print(f"Total files processed: {summary['total_files']}")
        print(f"Successfully patched: {summary['patched_count']}")
        print(f"Skipped (no data): {summary['skipped_count']}")
        print(f"Errors: {summary['error_count']}")
        print(f"Success rate: {summary['success_rate']}")

        if args.dry_run:
            print(f"\n[DRY RUN] No files were actually modified")
        else:
            print(f"\nBackups created in: {backup_dir}")

        return 0 if summary["error_count"] == 0 else 1

    except Exception as e:
        log.error(
            f"Patching failed: {e}",
            action="patch_failed",
            details={"error": str(e)},
            component="race_info_patcher",
        )
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
