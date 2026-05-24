#!/usr/bin/env python3
"""
Enhanced TGR Scraper
===================

This script fixes the TGR scraper to extract the detailed racing history
that appears under each dog in the form guides, not just the current race entry.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_enhanced_tgr_scraper():
    """Create an enhanced TGR scraper that extracts individual dog racing histories."""

    from src.collectors.the_greyhound_recorder_scraper import (
        TheGreyhoundRecorderScraper,
    )

    class EnhancedTGRScraper(TheGreyhoundRecorderScraper):
        """Enhanced TGR Scraper that extracts detailed dog racing histories."""

        def fetch_enhanced_dog_data(
            self, dog_name: str, limit_days: int = 365
        ) -> Dict[str, Any]:
            """Enhanced dog data collection that extracts full racing history."""

            logger.info(f"🐕 Fetching comprehensive TGR data for: {dog_name}")

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
                processed_meetings = 0
                for meeting in form_guides.get("meetings", [])[
                    :15
                ]:  # Check more meetings
                    if meeting.get("long_form_url") and processed_meetings < 10:
                        try:
                            logger.debug(
                                f"Processing meeting: {meeting.get('meeting_title', meeting['long_form_url'])}"
                            )

                            # Extract dog histories from this meeting's form guide
                            dog_histories = self._extract_dog_histories_from_meeting(
                                meeting["long_form_url"], dog_name
                            )

                            if dog_histories:
                                enhanced_data["form_entries"].extend(dog_histories)
                                logger.info(
                                    f"Found {len(dog_histories)} race entries for {dog_name} in meeting"
                                )

                            processed_meetings += 1

                        except Exception as e:
                            logger.debug(
                                f"Error processing meeting {meeting.get('meeting_title')}: {e}"
                            )
                            continue

                # Calculate performance metrics if we found entries
                if enhanced_data["form_entries"]:
                    logger.info(
                        f"Calculating performance metrics for {len(enhanced_data['form_entries'])} race entries"
                    )

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
                    f"✅ Enhanced data collected for {dog_name}: {len(enhanced_data['form_entries'])} total race entries"
                )

            except Exception as e:
                logger.error(
                    f"❌ Failed to fetch enhanced dog data for {dog_name}: {e}"
                )

            return enhanced_data

        def _extract_dog_histories_from_meeting(
            self, meeting_url: str, target_dog_name: str
        ) -> List[Dict[str, Any]]:
            """Extract detailed racing history for a specific dog from a meeting form guide."""

            if not meeting_url.startswith("http"):
                meeting_url = f"https://www.thegreyhoundrecorder.com.au{meeting_url}"

            soup = self._get(meeting_url)
            if not soup:
                return []

            dog_histories = []
            target_name_clean = target_dog_name.upper().strip()

            try:
                # Look for dog-specific sections in the form guide
                # These often have class patterns like 'dog-section', 'form-section', etc.

                # Method 1: Look for dog names as headers/section dividers
                dog_sections = self._find_dog_sections(soup, target_name_clean)

                for dog_section in dog_sections:
                    logger.debug(f"Processing dog section for: {target_dog_name}")

                    # Extract the racing history that appears under this dog
                    race_entries = self._extract_race_entries_from_dog_section(
                        dog_section, target_dog_name
                    )
                    dog_histories.extend(race_entries)

                # Method 2: Look for table-based racing histories
                if not dog_histories:
                    dog_histories = self._extract_from_racing_tables(
                        soup, target_name_clean
                    )

                # Method 3: Look for list-based racing histories
                if not dog_histories:
                    dog_histories = self._extract_from_racing_lists(
                        soup, target_name_clean
                    )

            except Exception as e:
                logger.debug(f"Error extracting dog histories from {meeting_url}: {e}")

            return dog_histories

        def _find_dog_sections(self, soup, target_dog_name: str) -> List:
            """Find sections of the page that contain racing data for the target dog."""

            dog_sections = []

            # Look for various patterns where dog names might appear as section headers
            potential_patterns = [
                # Dog name in headings
                soup.find_all(
                    ["h1", "h2", "h3", "h4", "h5"],
                    string=lambda text: text
                    and target_dog_name.lower() in text.lower(),
                ),
                # Dog name in strong/bold text
                soup.find_all(
                    "strong",
                    string=lambda text: text
                    and target_dog_name.lower() in text.lower(),
                ),
                # Dog name in spans or divs
                soup.find_all(
                    ["span", "div"],
                    string=lambda text: text
                    and target_dog_name.lower() in text.lower(),
                ),
                # Dog name in table cells (might be section headers)
                soup.find_all(
                    ["th", "td"],
                    string=lambda text: text
                    and target_dog_name.lower() in text.lower(),
                ),
            ]

            for pattern_results in potential_patterns:
                for element in pattern_results:
                    # Find the parent section that contains racing data
                    section = self._find_racing_data_section(element)
                    if section and section not in dog_sections:
                        dog_sections.append(section)

            return dog_sections

        def _find_racing_data_section(self, dog_name_element) -> Optional:
            """Find the section containing racing data starting from a dog name element."""

            # Look for the next sibling or parent that contains racing data
            current = dog_name_element

            # Check next siblings
            for sibling in dog_name_element.find_next_siblings():
                if self._contains_racing_data(sibling):
                    return sibling

                # Stop if we hit another dog name (start of next section)
                if self._looks_like_dog_name_header(sibling):
                    break

            # Check parent elements
            parent = dog_name_element.parent
            if parent and self._contains_racing_data(parent):
                return parent

            return None

        def _contains_racing_data(self, element) -> bool:
            """Check if an element contains racing data (dates, venues, times, etc.)."""

            if not element:
                return False

            text = element.get_text().lower()

            # Look for racing-related indicators
            racing_indicators = [
                "ballarat",
                "bendigo",
                "sandown",
                "melbourne",
                "traralgon",  # venues
                "jan",
                "feb",
                "mar",
                "apr",
                "may",
                "jun",
                "jul",
                "aug",
                "sep",
                "oct",
                "nov",
                "dec",  # months
                "m ",
                "metres",
                "meter",
                "distance",  # distances
                "time:",
                "box",
                "weight",
                "grade",  # racing terms
                "1st",
                "2nd",
                "3rd",
                "4th",
                "5th",
                "6th",
                "7th",
                "8th",  # positions
            ]

            # Check if element contains tables, lists, or structured data
            has_structure = bool(element.find_all(["table", "tr", "ul", "ol", "li"]))

            # Check for racing content
            has_racing_content = any(
                indicator in text for indicator in racing_indicators
            )

            return has_structure and has_racing_content

        def _looks_like_dog_name_header(self, element) -> bool:
            """Check if element looks like it might be another dog's name header."""

            if not element:
                return False

            text = element.get_text().strip()

            # Heuristics for dog names
            # - Usually 2-3 words
            # - Contains letters
            # - Not too long
            # - Often in header tags or bold

            if len(text) > 50 or len(text) < 3:
                return False

            words = text.split()
            if len(words) > 4 or len(words) < 1:
                return False

            # Check if it's in a header-like element
            is_header_like = (
                element.name in ["h1", "h2", "h3", "h4", "h5", "strong", "b"]
                or "header" in str(element.get("class", [])).lower()
                or "title" in str(element.get("class", [])).lower()
            )

            return is_header_like and text.isalpha()

        def _extract_race_entries_from_dog_section(
            self, section, dog_name: str
        ) -> List[Dict[str, Any]]:
            """Extract individual race entries from a dog's section."""

            race_entries = []

            try:
                # Look for different patterns of race data presentation

                # Pattern 1: Table rows with race data
                tables = section.find_all("table")
                for table in tables:
                    rows = table.find_all("tr")[1:]  # Skip header
                    for row in rows:
                        race_entry = self._parse_race_history_row(row, dog_name)
                        if race_entry:
                            race_entries.append(race_entry)

                # Pattern 2: List items with race data
                lists = section.find_all(["ul", "ol"])
                for list_elem in lists:
                    items = list_elem.find_all("li")
                    for item in items:
                        race_entry = self._parse_race_history_item(item, dog_name)
                        if race_entry:
                            race_entries.append(race_entry)

                # Pattern 3: Div sections with race data
                divs = section.find_all("div")
                for div in divs:
                    if self._contains_racing_data(div):
                        race_entry = self._parse_race_history_div(div, dog_name)
                        if race_entry:
                            race_entries.append(race_entry)

            except Exception as e:
                logger.debug(f"Error extracting race entries from section: {e}")

            return race_entries

        def _extract_from_racing_tables(
            self, soup, target_dog_name: str
        ) -> List[Dict[str, Any]]:
            """Extract racing history from table structures."""

            race_entries = []

            # Find tables that might contain racing data
            tables = soup.find_all("table")

            for table in tables:
                rows = table.find_all("tr")
                if len(rows) < 2:
                    continue

                # Check if this table has racing data structure
                headers = rows[0].find_all(["th", "td"])
                if headers:
                    header_text = [h.get_text(strip=True).lower() for h in headers]

                    # Look for racing-related headers
                    racing_headers = [
                        "date",
                        "venue",
                        "distance",
                        "time",
                        "position",
                        "box",
                        "grade",
                    ]
                    if any(rh in " ".join(header_text) for rh in racing_headers):

                        # Process each row looking for our dog
                        for row in rows[1:]:
                            cells = row.find_all(["td", "th"])

                            # Check if this row mentions our target dog
                            row_text = row.get_text().upper()
                            if target_dog_name in row_text:
                                race_entry = self._parse_race_history_row(
                                    row, target_dog_name
                                )
                                if race_entry:
                                    race_entries.append(race_entry)

            return race_entries

        def _extract_from_racing_lists(
            self, soup, target_dog_name: str
        ) -> List[Dict[str, Any]]:
            """Extract racing history from list structures."""

            race_entries = []

            # Find lists that might contain racing data
            lists = soup.find_all(["ul", "ol"])

            for list_elem in lists:
                items = list_elem.find_all("li")

                for item in items:
                    item_text = item.get_text().upper()
                    if target_dog_name in item_text and self._contains_racing_data(
                        item
                    ):
                        race_entry = self._parse_race_history_item(
                            item, target_dog_name
                        )
                        if race_entry:
                            race_entries.append(race_entry)

            return race_entries

        def _parse_race_history_row(
            self, row, dog_name: str
        ) -> Optional[Dict[str, Any]]:
            """Parse a table row containing race history data from TGR form guides."""

            try:
                cells = row.find_all(["td", "th"])
                if len(cells) < 3:
                    return None

                race_entry = {
                    "dog_name": dog_name,
                    "race_date": None,
                    "venue": None,
                    "distance": None,
                    "box_number": None,
                    "finish_position": None,
                    "individual_time": None,  # This dog's actual time
                    "winning_time": None,  # The winning time for this race
                    "grade": None,
                    "weight": None,
                    "margin": None,
                    "starting_price": None,
                    "comments": None,
                    "sectional_times": [],
                }

                # Based on typical TGR table structure, try to parse by position:
                # Common columns: Date, Venue, Distance, Box, Fin, Time, Win T, etc.

                for i, cell in enumerate(cells):
                    cell_text = cell.get_text(strip=True)

                    if not cell_text:
                        continue

                    # Column 0-1: Often date
                    if i <= 1 and self._is_date(cell_text):
                        race_entry["race_date"] = cell_text

                    # Early columns: Venue detection
                    elif i <= 2 and self._is_venue(cell_text):
                        race_entry["venue"] = cell_text

                    # Distance detection (e.g., "515m", "725m")
                    elif self._is_distance(cell_text):
                        race_entry["distance"] = cell_text

                    # Box number (usually single digit 1-8)
                    elif self._is_box_number(cell_text):
                        race_entry["box_number"] = int(cell_text)

                    # Finish position in "Fin" column (numbers 1-8, sometimes with suffix)
                    elif self._is_finish_position(cell_text):
                        race_entry["finish_position"] = cell_text

                    # Individual time (this dog's time)
                    elif (
                        self._is_race_time(cell_text)
                        and not race_entry["individual_time"]
                    ):
                        race_entry["individual_time"] = cell_text

                    # Winning time (second time field, often in "Win T" column)
                    elif (
                        self._is_race_time(cell_text)
                        and race_entry["individual_time"]
                        and not race_entry["winning_time"]
                    ):
                        race_entry["winning_time"] = cell_text

                    # Grade detection
                    elif self._is_grade(cell_text):
                        race_entry["grade"] = cell_text

                # Also try to extract data based on column headers if available
                # Look at the parent table to find headers
                table = row.find_parent("table")
                if table:
                    header_row = table.find("tr")
                    if header_row:
                        headers = [
                            h.get_text(strip=True).lower()
                            for h in header_row.find_all(["th", "td"])
                        ]

                        # Map cells to columns based on headers
                        for i, cell in enumerate(cells):
                            if i >= len(headers):
                                break

                            header = headers[i]
                            cell_text = cell.get_text(strip=True)

                            if "date" in header and self._is_date(cell_text):
                                race_entry["race_date"] = cell_text
                            elif "venue" in header or "track" in header:
                                race_entry["venue"] = cell_text
                            elif "dist" in header and self._is_distance(cell_text):
                                race_entry["distance"] = cell_text
                            elif "box" in header and cell_text.isdigit():
                                race_entry["box_number"] = int(cell_text)
                            elif "fin" in header:
                                race_entry["finish_position"] = cell_text
                            elif "time" in header and "win" not in header:
                                # Individual time column
                                if self._is_race_time(cell_text):
                                    race_entry["individual_time"] = cell_text
                            elif "win" in header and "time" in header:
                                # Winning time column
                                if self._is_race_time(cell_text):
                                    race_entry["winning_time"] = cell_text
                            elif "grade" in header:
                                race_entry["grade"] = cell_text

                # Only return if we found meaningful racing data
                if (
                    race_entry["race_date"]
                    or race_entry["venue"]
                    or race_entry["finish_position"]
                    or race_entry["individual_time"]
                ):
                    return race_entry

            except Exception as e:
                logger.debug(f"Error parsing race history row: {e}")

            return None

        def _parse_race_history_item(
            self, item, dog_name: str
        ) -> Optional[Dict[str, Any]]:
            """Parse a list item containing race history data."""

            try:
                text = item.get_text(strip=True)

                race_entry = {
                    "dog_name": dog_name,
                    "race_date": None,
                    "venue": None,
                    "distance": None,
                    "finish_position": None,
                    "individual_time": None,
                    "winning_time": None,
                    "comments": text,  # Store full text as comments
                }

                # Use regex to extract structured data from text

                # Extract date patterns (DD/MM/YY format common in TGR)
                date_match = re.search(r"(\d{2}/\d{2}/\d{2})", text)
                if not date_match:
                    date_match = re.search(
                        r"(\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec))",
                        text,
                        re.IGNORECASE,
                    )
                if date_match:
                    race_entry["race_date"] = date_match.group(1)

                # Extract venue patterns
                venues = [
                    "ballarat",
                    "bendigo",
                    "sandown",
                    "melbourne",
                    "traralgon",
                    "cranbourne",
                    "sale",
                    "warragul",
                    "richmond",
                    "horsham",
                    "geelong",
                    "shepparton",
                    "mildura",
                    "ascot park",
                ]
                for venue in venues:
                    if venue in text.lower():
                        race_entry["venue"] = venue.title()
                        break

                # Extract distance patterns
                distance_match = re.search(r"(\d{3,4}m)", text)
                if distance_match:
                    race_entry["distance"] = distance_match.group(1)

                # Extract finish position (just numbers, not ordinal)
                position_match = re.search(r"\bfin[:\s]*(\d+)", text, re.IGNORECASE)
                if not position_match:
                    position_match = re.search(r"\b(\d+)(?:st|nd|rd|th)\b", text)
                if position_match:
                    race_entry["finish_position"] = position_match.group(1)

                # Extract time patterns - look for individual time
                time_matches = re.findall(r"(\d{2}\.\d{2})", text)
                if time_matches:
                    # First time is usually individual time
                    race_entry["individual_time"] = time_matches[0]
                    # Second time (if exists) might be winning time
                    if len(time_matches) > 1:
                        race_entry["winning_time"] = time_matches[1]

                # Only return if we found meaningful data
                if (
                    race_entry["race_date"]
                    or race_entry["venue"]
                    or race_entry["finish_position"]
                    or race_entry["individual_time"]
                ):
                    return race_entry

            except Exception as e:
                logger.debug(f"Error parsing race history item: {e}")

            return None

        def _parse_race_history_div(
            self, div, dog_name: str
        ) -> Optional[Dict[str, Any]]:
            """Parse a div containing race history data."""

            # Similar to _parse_race_history_item but for div elements
            return self._parse_race_history_item(div, dog_name)

        def _is_date(self, text: str) -> bool:
            """Check if text looks like a date."""
            date_patterns = [
                r"\d{1,2}/\d{1,2}/\d{2,4}",  # 12/25/2023
                r"\d{1,2}-\d{1,2}-\d{2,4}",  # 12-25-2023
                r"\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",  # 25 Jan
            ]
            return any(re.search(pattern, text.lower()) for pattern in date_patterns)

        def _is_venue(self, text: str) -> bool:
            """Check if text looks like a race venue."""
            venues = [
                "ballarat",
                "bendigo",
                "sandown",
                "melbourne",
                "traralgon",
                "cranbourne",
                "sale",
                "warragul",
                "richmond",
                "horsham",
                "geelong",
                "shepparton",
                "mildura",
            ]
            return any(venue in text.lower() for venue in venues)

        def _is_distance(self, text: str) -> bool:
            """Check if text looks like a race distance."""
            return bool(re.search(r"\d{3,4}m", text))

        def _is_position(self, text: str) -> bool:
            """Check if text looks like a finishing position."""
            return bool(re.search(r"\d{1}(?:st|nd|rd|th)", text))

        def _is_finish_position(self, text: str) -> bool:
            """Check if text looks like a finishing position (including plain numbers)."""
            # TGR uses plain numbers (1, 2, 3, etc.) in the Fin column
            if text.isdigit() and 1 <= int(text) <= 8:
                return True
            # Also check for ordinal positions
            return bool(re.search(r"\d{1}(?:st|nd|rd|th)", text))

        def _is_race_time(self, text: str) -> bool:
            """Check if text looks like a race time."""
            return bool(re.search(r"\d{2}\.\d{2}", text))

        def _is_box_number(self, text: str) -> bool:
            """Check if text looks like a box number."""
            return text.isdigit() and 1 <= int(text) <= 8

        def _is_grade(self, text: str) -> bool:
            """Check if text looks like a race grade."""
            return "grade" in text.lower() or re.search(r"[g|G]\d", text)

        # Include the performance analysis methods from the previous version
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
                "average_time": 0.0,
                "best_time": None,
                "average_position": 0.0,
            }

            times = []
            positions = []

            for entry in form_entries:
                # Count wins and places - use correct position parsing
                position_str = str(entry.get("finish_position", ""))
                position_num = None

                # Extract numeric position
                if position_str.isdigit():
                    position_num = int(position_str)
                else:
                    # Try to extract from ordinal (1st, 2nd, etc.)
                    match = re.search(r"(\d+)", position_str)
                    if match:
                        position_num = int(match.group(1))

                if position_num:
                    positions.append(position_num)
                    if position_num == 1:
                        metrics["wins"] += 1
                    elif position_num in [2, 3]:
                        metrics["places"] += 1

                # Collect individual race times (not winning times)
                individual_time = entry.get("individual_time")
                if individual_time:
                    try:
                        time_float = float(individual_time)
                        times.append(time_float)
                    except:
                        pass

            if metrics["total_starts"] > 0:
                metrics["win_percentage"] = (
                    metrics["wins"] / metrics["total_starts"]
                ) * 100
                metrics["place_percentage"] = (
                    (metrics["wins"] + metrics["places"]) / metrics["total_starts"]
                ) * 100

                if positions:
                    metrics["average_position"] = sum(positions) / len(positions)

            if times:
                metrics["average_time"] = sum(times) / len(times)
                metrics["best_time"] = min(times)

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
                        "times": [],
                        "positions": [],
                    }

                venue_stats[venue]["starts"] += 1

                # Count wins and places using corrected logic
                position_str = str(entry.get("finish_position", ""))
                position_num = None

                if position_str.isdigit():
                    position_num = int(position_str)
                else:
                    match = re.search(r"(\d+)", position_str)
                    if match:
                        position_num = int(match.group(1))

                if position_num:
                    venue_stats[venue]["positions"].append(position_num)
                    if position_num == 1:
                        venue_stats[venue]["wins"] += 1
                    elif position_num in [2, 3]:
                        venue_stats[venue]["places"] += 1

                # Collect individual times
                individual_time = entry.get("individual_time")
                if individual_time:
                    try:
                        venue_stats[venue]["times"].append(float(individual_time))
                    except:
                        pass

            # Calculate percentages and averages
            for venue, stats in venue_stats.items():
                if stats["starts"] > 0:
                    stats["win_percentage"] = (stats["wins"] / stats["starts"]) * 100
                    stats["place_percentage"] = (
                        (stats["wins"] + stats["places"]) / stats["starts"]
                    ) * 100

                    if stats["positions"]:
                        stats["average_position"] = sum(stats["positions"]) / len(
                            stats["positions"]
                        )

                if stats["times"]:
                    stats["average_time"] = sum(stats["times"]) / len(stats["times"])
                    stats["best_time"] = min(stats["times"])

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
                        "times": [],
                    }

                distance_stats[distance]["starts"] += 1

                # Count wins and places
                position = entry.get("finish_position", "")
                if "1st" in str(position) or position == "1":
                    distance_stats[distance]["wins"] += 1
                elif any(place in str(position) for place in ["2nd", "3rd", "2", "3"]):
                    distance_stats[distance]["places"] += 1

                # Collect times
                race_time = entry.get("race_time")
                if race_time:
                    try:
                        distance_stats[distance]["times"].append(float(race_time))
                    except:
                        pass

            # Calculate percentages and averages
            for distance, stats in distance_stats.items():
                if stats["starts"] > 0:
                    stats["win_percentage"] = (stats["wins"] / stats["starts"]) * 100
                    stats["place_percentage"] = (
                        (stats["wins"] + stats["places"]) / stats["starts"]
                    ) * 100

                if stats["times"]:
                    stats["average_time"] = sum(stats["times"]) / len(stats["times"])
                    stats["best_time"] = min(stats["times"])

            return distance_stats

        def _extract_recent_comments(
            self, form_entries: List[Dict[str, Any]]
        ) -> List[str]:
            """Extract recent comments from form entries."""

            comments = []

            for entry in form_entries:
                if entry.get("comments"):
                    comments.append(entry["comments"])

            # Remove duplicates and filter short comments
            unique_comments = list(
                set(comment for comment in comments if len(comment) > 10)
            )

            return unique_comments[:20]  # Limit to 20 most recent

    return EnhancedTGRScraper


def test_enhanced_scraper():
    """Test the enhanced TGR scraper with detailed dog history extraction."""

    logger.info("🧪 Testing Enhanced TGR Scraper with Dog History Extraction...")

    EnhancedTGRScraper = create_enhanced_tgr_scraper()
    scraper = EnhancedTGRScraper(rate_limit=2.0, use_cache=True)

    # Test with known dogs from Ascot Park
    test_dogs = ["Mayfield Star", "Salted Caramel"]

    for dog_name in test_dogs:
        logger.info(f"\n🐕 Testing enhanced data collection for: {dog_name}")
        enhanced_data = scraper.fetch_enhanced_dog_data(dog_name)

        logger.info(f"Results for {dog_name}:")
        logger.info(
            f"  Total race entries found: {len(enhanced_data.get('form_entries', []))}"
        )

        if enhanced_data.get("form_entries"):
            logger.info(f"  Sample race entry: {enhanced_data['form_entries'][0]}")

            # Show performance summary
            performance = enhanced_data.get("performance_summary", {})
            logger.info(f"  Performance summary:")
            logger.info(f"    Total starts: {performance.get('total_starts', 0)}")
            logger.info(f"    Wins: {performance.get('wins', 0)}")
            logger.info(f"    Win %: {performance.get('win_percentage', 0):.1f}%")
            logger.info(f"    Best time: {performance.get('best_time', 'N/A')}")

            # Show venue analysis
            venue_analysis = enhanced_data.get("venue_analysis", {})
            if venue_analysis:
                logger.info(f"  Venue analysis:")
                for venue, stats in list(venue_analysis.items())[:3]:  # Show top 3
                    logger.info(
                        f"    {venue}: {stats.get('starts', 0)} starts, {stats.get('win_percentage', 0):.1f}% win rate"
                    )

        logger.info(
            f"  Comments collected: {len(enhanced_data.get('recent_comments', []))}"
        )

        # Stop after first successful test to avoid rate limiting
        if enhanced_data.get("form_entries"):
            break


def main():
    """Main function to test the enhanced TGR scraper."""

    logger.info("🚀 Creating Enhanced TGR Scraper with Dog History Extraction...")

    test_enhanced_scraper()

    logger.info("\n✅ Enhanced TGR scraper testing complete!")
    logger.info("💡 Next steps:")
    logger.info("  1. Replace the main TGR scraper with this enhanced version")
    logger.info("  2. Run comprehensive data collection on all unprocessed races")
    logger.info("  3. Verify the detailed racing histories are being stored correctly")


if __name__ == "__main__":
    main()
