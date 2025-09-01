#!/usr/bin/env python3
"""
Fix TGR Scraper
===============

This script fixes the TGR scraper by implementing proper HTML parsing
that matches the actual TGR website structure.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_tgr_parsing():
    """Fix the TGR parsing methods to work with real HTML structure."""

    import sqlite3

    from src.collectors.the_greyhound_recorder_scraper import (
        TheGreyhoundRecorderScraper,
    )

    # Create a custom scraper with fixed parsing methods
    class FixedTGRScraper(TheGreyhoundRecorderScraper):
        """TGR Scraper with corrected HTML parsing logic."""

        def _fetch_race_details_fixed(self, race_url: str) -> Dict[str, Any]:
            """Fixed race details extraction that works with actual TGR HTML."""

            if not race_url.startswith("http"):
                race_url = f"https://www.thegreyhoundrecorder.com.au{race_url}"

            soup = self._get(race_url)
            if not soup:
                return {}

            race_details = {
                "url": race_url,
                "date": None,
                "venue": None,
                "race_number": None,
                "grade": None,
                "distance": None,
                "field_size": 0,
                "dogs": [],
                "race_result": {},
                "expert_comments": [],
            }

            try:
                # Extract race title/heading
                heading = soup.find(class_="form-guide-meeting__heading")
                if heading:
                    heading_text = heading.get_text(strip=True)
                    logger.debug(f"Found heading: {heading_text}")

                    # Parse venue, date, race number from heading
                    # Format: "Ballarat Form Guide(Race 1)- 30th Jul 2025"
                    if "Form Guide" in heading_text and "Race" in heading_text:
                        parts = heading_text.split("Form Guide")
                        if len(parts) >= 2:
                            race_details["venue"] = parts[0].strip()

                            race_part = parts[1]
                            if "Race" in race_part and ")" in race_part:
                                # Extract race number
                                race_num_text = (
                                    race_part.split("Race")[1].split(")")[0].strip()
                                )
                                try:
                                    race_details["race_number"] = int(race_num_text)
                                except:
                                    pass

                                # Extract date
                                if "-" in race_part:
                                    date_part = race_part.split("-")[-1].strip()
                                    race_details["date"] = date_part

                # Find the main dog data table
                # Look for tables with racing data structure
                tables = soup.find_all("table")
                main_table = None

                for table in tables:
                    rows = table.find_all("tr")
                    if len(rows) >= 2:  # At least header + 1 data row
                        headers = rows[0].find_all(["th", "td"])
                        if headers:
                            header_text = [
                                h.get_text(strip=True).lower() for h in headers
                            ]

                            # Check if this looks like the main form table
                            if any(
                                term in " ".join(header_text)
                                for term in ["greyhound", "rug", "form", "comment"]
                            ):
                                main_table = table
                                logger.debug(
                                    f"Found main table with headers: {[h.get_text(strip=True) for h in headers]}"
                                )
                                break

                if main_table:
                    rows = main_table.find_all("tr")[1:]  # Skip header

                    for row in rows:
                        dog_entry = self._parse_dog_entry_row_fixed(row)
                        if dog_entry:
                            race_details["dogs"].append(dog_entry)
                            race_details["field_size"] += 1
                            logger.debug(f"Parsed dog: {dog_entry.get('dog_name')}")

                # Extract any expert comments from the page
                comment_elements = soup.find_all(
                    ["div", "p"],
                    class_=lambda x: x
                    and any(
                        word in str(x).lower()
                        for word in ["comment", "analysis", "preview"]
                    ),
                )

                for elem in comment_elements:
                    comment_text = elem.get_text(strip=True)
                    if len(comment_text) > 30:  # Filter out short text
                        race_details["expert_comments"].append(comment_text)

                logger.info(
                    f"Extracted {race_details['field_size']} dogs and {len(race_details['expert_comments'])} comments from {race_url}"
                )

            except Exception as e:
                logger.error(f"Error parsing race details from {race_url}: {e}")

            return race_details

        def _parse_dog_entry_row_fixed(self, row) -> Optional[Dict[str, Any]]:
            """Fixed dog entry parsing that works with actual TGR table structure."""

            try:
                cells = row.find_all(["td", "th"])
                if len(cells) < 3:  # Need at least dog name + some data
                    return None

                dog_entry = {
                    "box_number": None,
                    "dog_name": None,
                    "trainer": None,
                    "weight": None,
                    "recent_form": [],
                    "last_start": {},
                    "comments": None,
                    "odds": None,
                    "rating": None,
                    "early_speed": None,
                }

                # Based on the table structure we found:
                # ['Rug/Greyhound (box)', 'Form', 'Comment', 'Early Speed', 'Rtg']

                for i, cell in enumerate(cells):
                    cell_text = cell.get_text(strip=True)

                    if i == 0:  # Dog name (and possibly box)
                        # Could be "Dog Name" or "Dog Name (box)"
                        if "(" in cell_text and ")" in cell_text:
                            # Extract dog name and box number
                            parts = cell_text.split("(")
                            dog_entry["dog_name"] = parts[0].strip()

                            box_part = parts[1].replace(")", "").strip()
                            try:
                                dog_entry["box_number"] = int(box_part)
                            except:
                                pass
                        else:
                            dog_entry["dog_name"] = cell_text

                    elif i == 1:  # Form
                        # Form could be like "3105" or "1234"
                        if cell_text.isdigit() and len(cell_text) >= 2:
                            # Convert form string to list of positions
                            dog_entry["recent_form"] = list(cell_text)

                    elif i == 2:  # Comment
                        if cell_text and len(cell_text) > 3:
                            dog_entry["comments"] = cell_text

                    elif i == 3:  # Early Speed
                        try:
                            dog_entry["early_speed"] = float(cell_text)
                        except:
                            pass

                    elif i == 4:  # Rating
                        try:
                            dog_entry["rating"] = int(cell_text)
                        except:
                            pass

                # Look for additional data in nested elements
                links = row.find_all("a")
                for link in links:
                    href = link.get("href", "")
                    if "/greyhound/" in href:
                        dog_entry["profile_url"] = href

                return dog_entry if dog_entry["dog_name"] else None

            except Exception as e:
                logger.debug(f"Error parsing dog entry row: {e}")
                return None

        def fetch_enhanced_dog_data_fixed(
            self, dog_name: str, limit_days: int = 365
        ) -> Dict[str, Any]:
            """Fixed enhanced dog data collection using proper parsing."""

            logger.info(f"Fetching enhanced TGR data for dog: {dog_name}")

            enhanced_data = {
                "dog_name": dog_name,
                "form_entries": [],
                "performance_summary": {},
                "venue_analysis": {},
                "distance_analysis": {},
                "recent_comments": [],
                "expert_insights": [],
            }

            try:
                # Get form guides
                form_guides = self.fetch_form_guides()

                # Process each meeting to find races with this dog
                for meeting in form_guides.get("meetings", [])[
                    :10
                ]:  # Limit to first 10 meetings
                    if meeting.get("long_form_url"):
                        try:
                            race_data = self._fetch_race_details_fixed(
                                meeting["long_form_url"]
                            )
                            dog_entries = self._extract_dog_entries_fixed(
                                race_data, dog_name
                            )
                            enhanced_data["form_entries"].extend(dog_entries)

                        except Exception as e:
                            logger.debug(
                                f"Error processing meeting {meeting.get('meeting_title')}: {e}"
                            )
                            continue

                # Calculate performance metrics if we found entries
                if enhanced_data["form_entries"]:
                    enhanced_data["performance_summary"] = (
                        self._calculate_performance_metrics(
                            enhanced_data["form_entries"]
                        )
                    )

                    enhanced_data["venue_analysis"] = self._analyze_venue_performance(
                        enhanced_data["form_entries"]
                    )

                    enhanced_data["distance_analysis"] = (
                        self._analyze_distance_performance(
                            enhanced_data["form_entries"]
                        )
                    )

                    enhanced_data["recent_comments"] = self._extract_recent_comments(
                        enhanced_data["form_entries"]
                    )

                logger.info(
                    f"Enhanced data collected for {dog_name}: {len(enhanced_data['form_entries'])} races"
                )

            except Exception as e:
                logger.error(f"Failed to fetch enhanced dog data for {dog_name}: {e}")

            return enhanced_data

        def _calculate_performance_metrics(
            self, form_entries: List[Dict[str, Any]]
        ) -> Dict[str, Any]:
            """Calculate performance metrics from form entries."""

            if not form_entries:
                return {}

            metrics = {
                "total_starts": len(form_entries),
                "wins": 0,
                "places": 0,
                "win_percentage": 0.0,
                "place_percentage": 0.0,
                "average_rating": 0.0,
                "average_early_speed": 0.0,
            }

            ratings = []
            early_speeds = []

            for entry in form_entries:
                # Count wins and places from recent form
                recent_form = entry.get("recent_form", [])
                for position in recent_form:
                    if position == "1":
                        metrics["wins"] += 1
                    elif position in ["2", "3"]:
                        metrics["places"] += 1

                # Collect ratings and early speeds
                if entry.get("rating"):
                    ratings.append(entry["rating"])
                if entry.get("early_speed"):
                    early_speeds.append(entry["early_speed"])

            if metrics["total_starts"] > 0:
                metrics["win_percentage"] = (
                    metrics["wins"] / metrics["total_starts"]
                ) * 100
                metrics["place_percentage"] = (
                    (metrics["wins"] + metrics["places"]) / metrics["total_starts"]
                ) * 100

            if ratings:
                metrics["average_rating"] = sum(ratings) / len(ratings)

            if early_speeds:
                metrics["average_early_speed"] = sum(early_speeds) / len(early_speeds)

            return metrics

        def _analyze_venue_performance(
            self, form_entries: List[Dict[str, Any]]
        ) -> Dict[str, Any]:
            """Analyze performance by venue."""

            venue_stats = {}

            for entry in form_entries:
                venue = entry.get("venue", "Unknown")
                if venue not in venue_stats:
                    venue_stats[venue] = {
                        "starts": 0,
                        "wins": 0,
                        "places": 0,
                        "ratings": [],
                        "early_speeds": [],
                    }

                venue_stats[venue]["starts"] += 1

                # Process form for wins/places
                recent_form = entry.get("recent_form", [])
                for position in recent_form:
                    if position == "1":
                        venue_stats[venue]["wins"] += 1
                    elif position in ["2", "3"]:
                        venue_stats[venue]["places"] += 1

                if entry.get("rating"):
                    venue_stats[venue]["ratings"].append(entry["rating"])
                if entry.get("early_speed"):
                    venue_stats[venue]["early_speeds"].append(entry["early_speed"])

            # Calculate percentages and averages
            for venue, stats in venue_stats.items():
                if stats["starts"] > 0:
                    stats["win_percentage"] = (stats["wins"] / stats["starts"]) * 100
                    stats["place_percentage"] = (
                        (stats["wins"] + stats["places"]) / stats["starts"]
                    ) * 100

                if stats["ratings"]:
                    stats["average_rating"] = sum(stats["ratings"]) / len(
                        stats["ratings"]
                    )

                if stats["early_speeds"]:
                    stats["average_early_speed"] = sum(stats["early_speeds"]) / len(
                        stats["early_speeds"]
                    )

            return venue_stats

        def _analyze_distance_performance(
            self, form_entries: List[Dict[str, Any]]
        ) -> Dict[str, Any]:
            """Analyze performance by race distance."""

            distance_stats = {}

            for entry in form_entries:
                distance = entry.get("distance", "Unknown")
                if distance not in distance_stats:
                    distance_stats[distance] = {
                        "starts": 0,
                        "wins": 0,
                        "places": 0,
                        "ratings": [],
                        "early_speeds": [],
                    }

                distance_stats[distance]["starts"] += 1

                # Process form for wins/places
                recent_form = entry.get("recent_form", [])
                for position in recent_form:
                    if position == "1":
                        distance_stats[distance]["wins"] += 1
                    elif position in ["2", "3"]:
                        distance_stats[distance]["places"] += 1

                if entry.get("rating"):
                    distance_stats[distance]["ratings"].append(entry["rating"])
                if entry.get("early_speed"):
                    distance_stats[distance]["early_speeds"].append(
                        entry["early_speed"]
                    )

            # Calculate percentages and averages
            for distance, stats in distance_stats.items():
                if stats["starts"] > 0:
                    stats["win_percentage"] = (stats["wins"] / stats["starts"]) * 100
                    stats["place_percentage"] = (
                        (stats["wins"] + stats["places"]) / stats["starts"]
                    ) * 100

                if stats["ratings"]:
                    stats["average_rating"] = sum(stats["ratings"]) / len(
                        stats["ratings"]
                    )

                if stats["early_speeds"]:
                    stats["average_early_speed"] = sum(stats["early_speeds"]) / len(
                        stats["early_speeds"]
                    )

            return distance_stats

        def _extract_recent_comments(
            self, form_entries: List[Dict[str, Any]]
        ) -> List[str]:
            """Extract recent comments from form entries."""

            comments = []

            for entry in form_entries:
                # Add race comments
                if entry.get("comments"):
                    comments.append(entry["comments"])

                # Add expert comments from the race
                expert_comments = entry.get("expert_comments", [])
                comments.extend(expert_comments)

            # Remove duplicates and filter short comments
            unique_comments = list(
                set(comment for comment in comments if len(comment) > 10)
            )

            return unique_comments[:20]  # Limit to 20 most recent

        def _extract_dog_entries_fixed(
            self, race_data: Dict[str, Any], target_dog_name: str
        ) -> List[Dict[str, Any]]:
            """Extract entries for a specific dog using fixed parsing."""

            entries = []
            target_name_clean = target_dog_name.upper().strip()

            for dog in race_data.get("dogs", []):
                dog_name = dog.get("dog_name", "").upper().strip()
                if dog_name == target_name_clean or target_name_clean in dog_name:
                    # Enhance with race metadata
                    enhanced_entry = {
                        **dog,
                        "race_date": race_data.get("date"),
                        "venue": race_data.get("venue"),
                        "race_number": race_data.get("race_number"),
                        "grade": race_data.get("grade"),
                        "distance": race_data.get("distance"),
                        "field_size": race_data.get("field_size"),
                        "race_url": race_data.get("url"),
                        "expert_comments": race_data.get("expert_comments", []),
                    }
                    entries.append(enhanced_entry)

            return entries

    return FixedTGRScraper


def test_fixed_tgr_scraper():
    """Test the fixed TGR scraper."""

    logger.info("🧪 Testing Fixed TGR Scraper...")

    FixedTGRScraper = fix_tgr_parsing()
    scraper = FixedTGRScraper(rate_limit=2.0, use_cache=True)

    # Test with a known TGR URL
    test_url = "/form-guides/ballarat/long-form/244740/1/"

    logger.info(f"Testing race details extraction: {test_url}")
    race_data = scraper._fetch_race_details_fixed(test_url)

    logger.info(f"Results:")
    logger.info(f"  Dogs found: {len(race_data.get('dogs', []))}")
    logger.info(f"  Venue: {race_data.get('venue')}")
    logger.info(f"  Race number: {race_data.get('race_number')}")
    logger.info(f"  Date: {race_data.get('date')}")

    if race_data.get("dogs"):
        logger.info(f"  Sample dog: {race_data['dogs'][0]}")

    # Test enhanced dog data collection
    if race_data.get("dogs"):
        first_dog_name = race_data["dogs"][0].get("dog_name")
        if first_dog_name:
            logger.info(f"\\nTesting enhanced data collection for: {first_dog_name}")
            enhanced_data = scraper.fetch_enhanced_dog_data_fixed(first_dog_name)

            logger.info(f"Enhanced data results:")
            logger.info(f"  Form entries: {len(enhanced_data.get('form_entries', []))}")
            logger.info(f"  Comments: {len(enhanced_data.get('recent_comments', []))}")
            logger.info(
                f"  Performance summary: {bool(enhanced_data.get('performance_summary'))}"
            )

    return len(race_data.get("dogs", [])) > 0


def main():
    """Main function to test and demonstrate the fixed TGR scraper."""

    logger.info("🚀 Fixing TGR Scraper Issues...")

    # Test the fixed scraper
    success = test_fixed_tgr_scraper()

    if success:
        logger.info("✅ Fixed TGR scraper is working correctly!")
        logger.info("💡 Next steps:")
        logger.info("  1. Replace the parsing methods in the main TGR scraper")
        logger.info("  2. Run bulk TGR data collection with the fixed parser")
        logger.info("  3. Update existing placeholder records with real data")
    else:
        logger.error("❌ Fixed TGR scraper still has issues")
        logger.info("🔍 Additional debugging needed")


if __name__ == "__main__":
    main()
