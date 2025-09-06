#!/usr/bin/env python3
"""
Enhanced Race Parser for Greyhound Racing Data
=============================================

This module provides improved race information extraction to fix the issues
that led to "UNK_0_UNKNOWN" races and other parsing problems.

Key improvements:
1. Multiple parsing patterns for different filename formats
2. Fallback parsing strategies
3. Better error handling and logging
4. Validation of extracted information
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd


class EnhancedRaceParser:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Define parsing patterns in order of specificity
        self.filename_patterns = [
            # Standard format: "Race 5 - GEE - 22 July 2025.csv"
            {
                "pattern": r"Race\s+(\d+)\s*-\s*([A-Z_]+)\s*-\s*(\d{1,2}\s+\w+\s+\d{4})",
                "groups": ["race_number", "venue", "date_str"],
                "format": "standard",
            },
            # Alternative format: "Race_5_GEE_22_July_2025.csv"
            {
                "pattern": r"Race_(\d+)_([A-Z_]+)_(.+)",
                "groups": ["race_number", "venue", "date_str"],
                "format": "underscore",
            },
            # Venue-first format: "GEE_5_22_July_2025.csv"
            {
                "pattern": r"([A-Z_]+)_(\d+)_(.+)",
                "groups": ["venue", "race_number", "date_str"],
                "format": "venue_first",
            },
            # Lowercase venue format: "gee_2025-07-22_5.csv"
            {
                "pattern": r"([a-z]+)_(\d{4}-\d{2}-\d{2})_(\d+)",
                "groups": ["venue", "date_str", "race_number"],
                "format": "lowercase_date",
            },
            # Simple format: "race5_gee_july22.csv"
            {
                "pattern": r"race(\d+)[_-]([a-zA-Z]+)[_-](.+)",
                "groups": ["race_number", "venue", "date_str"],
                "format": "simple",
            },
        ]

        # Date parsing patterns
        self.date_patterns = [
            r"(\d{1,2})\s+(\w+)\s+(\d{4})",  # "22 July 2025"
            r"(\d{4})-(\d{2})-(\d{2})",  # "2025-07-22"
            r"(\w+)(\d{1,2})",  # "july22"
            r"(\d{2})(\d{2})(\d{4})",  # "22072025"
        ]

        # Month name mappings
        self.month_mapping = {
            "jan": "01",
            "january": "01",
            "feb": "02",
            "february": "02",
            "mar": "03",
            "march": "03",
            "apr": "04",
            "april": "04",
            "may": "05",
            "jun": "06",
            "june": "06",
            "jul": "07",
            "july": "07",
            "aug": "08",
            "august": "08",
            "sep": "09",
            "september": "09",
            "oct": "10",
            "october": "10",
            "nov": "11",
            "november": "11",
            "dec": "12",
            "december": "12",
        }

        # Known venue mappings
        self.venue_mapping = {
            "ap_k": "AP_K",
            "gee": "GEE",
            "rich": "RICH",
            "dapt": "DAPT",
            "bal": "BAL",
            "ben": "BEN",
            "hea": "HEA",
            "war": "WAR",
            "san": "SAN",
            "mount": "MOUNT",
            "murr": "MURR",
            "sal": "SAL",
            "hor": "HOR",
            "cann": "CANN",
            "w_pk": "W_PK",
        }

    def extract_race_info(self, filename: str) -> Dict[str, Any]:
        """
        Extract race information from filename with improved parsing

        Args:
            filename: The race file name to parse

        Returns:
            Dictionary containing race information with fallback values
        """
        self.logger.debug(f"Parsing filename: {filename}")

        # Initialize result with safe defaults
        result = {
            "filename": filename,
            "venue": None,
            "race_number": None,
            "date_str": None,
            "parse_confidence": 0.0,
            "parse_method": None,
            "parse_warnings": [],
        }

        # Try each parsing pattern
        for pattern_config in self.filename_patterns:
            match = re.search(pattern_config["pattern"], filename, re.IGNORECASE)
            if match:
                try:
                    parsed_info = self._process_match(match, pattern_config)
                    if parsed_info:
                        result.update(parsed_info)
                        result["parse_method"] = pattern_config["format"]
                        result["parse_confidence"] = self._calculate_confidence(result)
                        break
                except Exception as e:
                    self.logger.warning(
                        f"Error processing pattern {pattern_config['format']}: {e}"
                    )
                    result["parse_warnings"].append(
                        f"Pattern {pattern_config['format']} failed: {e}"
                    )

        # If no pattern matched, try fallback methods
        if result["venue"] is None:
            result = self._try_fallback_parsing(filename, result)

        # Validate and clean the results
        result = self._validate_and_clean(result)

        self.logger.info(
            f"Parsed {filename} -> venue: {result['venue']}, race: {result['race_number']}, confidence: {result['parse_confidence']:.2f}"
        )

        return result

    def _process_match(self, match, pattern_config):
        """Process a regex match according to the pattern configuration"""
        groups = match.groups()
        group_names = pattern_config["groups"]

        if len(groups) != len(group_names):
            raise ValueError(
                f"Group count mismatch: {len(groups)} vs {len(group_names)}"
            )

        result = {}
        for i, group_name in enumerate(group_names):
            value = groups[i].strip()

            if group_name == "venue":
                result["venue"] = self._normalize_venue(value)
            elif group_name == "race_number":
                result["race_number"] = self._parse_race_number(value)
            elif group_name == "date_str":
                result["date_str"] = self._normalize_date(value)

        return result

    def _normalize_venue(self, venue_str: str) -> str:
        """Normalize venue name to standard format"""
        if not venue_str:
            return None

        venue_clean = venue_str.lower().strip().replace(" ", "_")

        # Check direct mapping
        if venue_clean in self.venue_mapping:
            return self.venue_mapping[venue_clean]

        # Check if it's already in standard format
        if venue_str.isupper() and len(venue_str) <= 6:
            return venue_str

        # Try to find partial matches
        for key, value in self.venue_mapping.items():
            if key in venue_clean or venue_clean in key:
                return value

        # Return uppercase version as fallback
        return venue_str.upper()

    def _parse_race_number(self, race_str: str) -> Optional[int]:
        """Parse race number from string"""
        if not race_str:
            return None

        # Extract digits
        digits = re.findall(r"\d+", race_str)
        if digits:
            race_num = int(digits[0])
            if 1 <= race_num <= 20:  # Reasonable race number range
                return race_num

        return None

    def _normalize_date(self, date_str: str) -> str:
        """Normalize date string to consistent format"""
        if not date_str:
            return None

        date_clean = date_str.strip().replace("_", " ")

        # Try each date pattern
        for pattern in self.date_patterns:
            match = re.search(pattern, date_clean, re.IGNORECASE)
            if match:
                try:
                    return self._format_date_from_match(match, pattern, date_clean)
                except Exception as e:
                    self.logger.debug(f"Date parsing error with pattern {pattern}: {e}")

        # Return original if no pattern matched
        return date_str

    def _format_date_from_match(self, match, pattern, original):
        """Format date from regex match"""
        groups = match.groups()

        if pattern == r"(\d{1,2})\s+(\w+)\s+(\d{4})":  # "22 July 2025"
            day, month_name, year = groups
            month = self.month_mapping.get(month_name.lower())
            if month:
                return f"{year}-{month}-{day.zfill(2)}"

        elif pattern == r"(\d{4})-(\d{2})-(\d{2})":  # "2025-07-22"
            return f"{groups[0]}-{groups[1]}-{groups[2]}"

        elif pattern == r"(\w+)(\d{1,2})":  # "july22"
            month_name, day = groups
            month = self.month_mapping.get(month_name.lower())
            if month:
                year = datetime.now().year  # Assume current year
                return f"{year}-{month}-{day.zfill(2)}"

        return original

    def _try_fallback_parsing(self, filename: str, current_result: Dict) -> Dict:
        """Try fallback parsing methods for difficult filenames"""
        result = current_result.copy()

        # Try to extract any venue-like strings (2-6 uppercase letters)
        venue_matches = re.findall(r"\b[A-Z]{2,6}\b", filename)
        if venue_matches:
            # Use the first match that looks like a venue
            for venue in venue_matches:
                if (
                    venue in self.venue_mapping.values()
                    or venue.lower() in self.venue_mapping
                ):
                    result["venue"] = venue
                    result["parse_confidence"] = 0.3
                    result["parse_method"] = "fallback_venue_extraction"
                    break

        # Try to extract race numbers
        race_matches = re.findall(r"\b(\d{1,2})\b", filename)
        if race_matches:
            for race_num_str in race_matches:
                race_num = int(race_num_str)
                if 1 <= race_num <= 20:
                    result["race_number"] = race_num
                    if result["parse_confidence"] < 0.3:
                        result["parse_confidence"] = 0.3
                        result["parse_method"] = "fallback_number_extraction"
                    break

        # If still no venue found, try CSV content parsing
        if not result["venue"]:
            result = self._try_content_based_parsing(filename, result)

        return result

    def _try_content_based_parsing(self, filename: str, current_result: Dict) -> Dict:
        """Try to parse race info from CSV content as last resort"""
        result = current_result.copy()

        try:
            # Check if file exists and try to read it
            from pathlib import Path

            file_path = Path(filename)

            # Try different possible paths
            possible_paths = [
                file_path,
                Path("form_guides") / filename,
                Path("form_guides/downloaded") / filename,
                Path("upcoming") / filename,
            ]

            for path in possible_paths:
                if path.exists():
                    try:
                        df = pd.read_csv(
                            path, nrows=5, sep="|"
                        )  # Read first few rows only

                        # Look for track information in the data
                        if "TRACK" in df.columns:
                            tracks = df["TRACK"].dropna().unique()
                            if len(tracks) > 0:
                                track = str(tracks[0]).strip()
                                result["venue"] = self._normalize_venue(track)
                                result["parse_confidence"] = 0.4
                                result["parse_method"] = "content_based_track"
                                break

                        # Look for venue info in dog names or other columns
                        for col in df.columns:
                            if col in ["Dog Name", "DOG", "TRACK"]:
                                values = df[col].dropna().astype(str)
                                for value in values:
                                    venue_matches = re.findall(r"\b[A-Z]{2,6}\b", value)
                                    for venue in venue_matches:
                                        if venue in self.venue_mapping.values():
                                            result["venue"] = venue
                                            result["parse_confidence"] = 0.3
                                            result["parse_method"] = (
                                                "content_based_scanning"
                                            )
                                            break
                                    if result["venue"]:
                                        break
                            if result["venue"]:
                                break

                    except Exception as e:
                        self.logger.debug(f"Content parsing failed for {path}: {e}")
                        continue

                    break

        except Exception as e:
            self.logger.debug(f"Content-based parsing failed: {e}")

        return result

    def _validate_and_clean(self, result: Dict) -> Dict:
        """Validate and clean the parsing results"""
        # Ensure we have at least some information
        if not result["venue"] and not result["race_number"]:
            result["parse_warnings"].append(
                "No venue or race number could be extracted"
            )
            result["parse_confidence"] = 0.0

        # Set defaults for missing critical information
        if not result["venue"]:
            result["venue"] = "UNKNOWN"
            result["parse_warnings"].append("Venue defaulted to UNKNOWN")

        if not result["race_number"]:
            result["race_number"] = 0
            result["parse_warnings"].append("Race number defaulted to 0")

        if not result["date_str"]:
            result["date_str"] = datetime.now().strftime("%Y-%m-%d")
            result["parse_warnings"].append("Date defaulted to current date")

        # Generate race_id
        venue = result["venue"] if result["venue"] != "UNKNOWN" else "UNK"
        race_num = result["race_number"] if result["race_number"] > 0 else 0
        date_part = result["date_str"].replace(" ", "_").replace("-", "_")
        result["race_id"] = f"{venue}_{race_num}_{date_part}"

        return result

    def _calculate_confidence(self, result: Dict) -> float:
        """Calculate confidence score for parsing results"""
        confidence = 0.0

        # Base confidence for having venue and race number
        if result["venue"] and result["venue"] != "UNKNOWN":
            confidence += 0.4
        if result["race_number"] and result["race_number"] > 0:
            confidence += 0.3
        if result["date_str"]:
            confidence += 0.2

        # Bonus for known venues
        if result["venue"] in self.venue_mapping.values():
            confidence += 0.1

        return min(confidence, 1.0)


# Monkey patch the existing function for immediate improvement
def patch_race_predictor():
    """Patch the existing race predictor with improved parsing"""
    try:
        import upcoming_race_predictor

        parser = EnhancedRaceParser()

        # Replace the extract_race_info method
        def enhanced_extract_race_info(self, filename):
            return parser.extract_race_info(filename)

        upcoming_race_predictor.UpcomingRacePredictor.extract_race_info = (
            enhanced_extract_race_info
        )
        print("✅ Patched race predictor with enhanced parsing")

    except ImportError:
        print("⚠️  Could not patch race predictor - module not found")


if __name__ == "__main__":
    # Test the parser
    parser = EnhancedRaceParser()

    test_files = [
        "Race 5 - GEE - 22 July 2025.csv",
        "Race 3 - RICH - 20 July 2025.csv",
        "Race 2 - DAPT - 22 August 2025.csv",
        "gee_2025-07-22_5.csv",
        "unknown_race_file.csv",
    ]

    print("🧪 Testing Enhanced Race Parser")
    print("=" * 50)

    for filename in test_files:
        result = parser.extract_race_info(filename)
        print(f"\n📁 {filename}")
        print(f"   🏟️  Venue: {result['venue']}")
        print(f"   🏃 Race: {result['race_number']}")
        print(f"   📅 Date: {result['date_str']}")
        print(f"   🆔 Race ID: {result['race_id']}")
        print(f"   📊 Confidence: {result['parse_confidence']:.2f}")
        print(f"   🔧 Method: {result['parse_method']}")
        if result["parse_warnings"]:
            print(f"   ⚠️  Warnings: {', '.join(result['parse_warnings'])}")
