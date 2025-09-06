#!/usr/bin/env python3
"""
Fixed TGR Implementation
=======================

This file contains the corrected methods for the TGR scraper based on
the actual HTML structure observed in TGR form guide pages.
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def create_fixed_fetch_race_details():
    """Return a fixed _fetch_race_details method."""

    def _fetch_race_details(self, race_url: str) -> Dict[str, Any]:
        """Extract race details and individual dog racing histories from TGR."""

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
            # Extract race title/heading from the meeting heading
            heading = soup.find(class_="form-guide-meeting__heading")
            if heading:
                heading_text = heading.get_text(strip=True)
                self.logger.debug(f"Found heading: {heading_text}")

                # Parse venue, date, race number from heading
                # Format: "Murray Bridge Form Guide (Race 1) - 3rd Aug 2025"
                if "Form Guide" in heading_text:
                    parts = heading_text.split("Form Guide")
                    if len(parts) >= 2:
                        race_details["venue"] = parts[0].strip()

                        race_part = parts[1]
                        if "Race" in race_part and ")" in race_part:
                            # Extract race number from (Race X)
                            race_match = re.search(r"Race (\d+)", race_part)
                            if race_match:
                                race_details["race_number"] = int(race_match.group(1))

                            # Extract date from the end
                            if "-" in race_part:
                                date_part = race_part.split("-")[-1].strip()
                                race_details["date"] = date_part

            # Extract individual dog racing histories from form-guide-long-form-selection sections
            dog_sections = soup.find_all(class_="form-guide-long-form-selection")

            for section in dog_sections:
                # Skip vacant boxes
                if "form-guide-long-form-selection--vacant" in section.get("class", []):
                    continue

                # Extract dog name from the header
                header = section.find(class_="form-guide-long-form-selection__header")
                if not header:
                    continue

                dog_name_elem = header.find(
                    class_="form-guide-long-form-selection__header-name"
                )
                if not dog_name_elem:
                    continue

                dog_name = dog_name_elem.get_text(strip=True)
                self.logger.debug(f"Processing dog: {dog_name}")

                # Extract the racing history table for this dog
                history_table = section.find(
                    "table", class_="form-guide-selection-results"
                )
                if history_table:
                    dog_history = self._extract_dog_racing_history(
                        history_table, dog_name
                    )
                    if dog_history:
                        # Add the dog with their racing history
                        dog_entry = {
                            "dog_name": dog_name,
                            "racing_history": dog_history,
                            "total_races": len(dog_history),
                        }
                        race_details["dogs"].append(dog_entry)
                        race_details["field_size"] += 1

                        self.logger.debug(
                            f"Extracted {len(dog_history)} races for {dog_name}"
                        )

            self.logger.debug(
                f"Extracted {race_details['field_size']} dogs with racing histories"
            )

        except Exception as e:
            self.logger.error(f"Error parsing race details from {race_url}: {e}")

        return race_details

    return _fetch_race_details


def create_extract_dog_racing_history():
    """Return the _extract_dog_racing_history method."""

    def _extract_dog_racing_history(self, table, dog_name: str) -> List[Dict[str, Any]]:
        """Extract individual race history from a dog's history table."""

        racing_history = []

        try:
            # Get table headers to understand column structure
            headers = []
            header_row = table.find("thead")
            if header_row:
                header_cells = header_row.find_all("th")
                headers = [cell.get_text(strip=True) for cell in header_cells]

            self.logger.debug(f"Table headers for {dog_name}: {headers}")

            # Process each race row in the table
            tbody = table.find("tbody")
            if tbody:
                rows = tbody.find_all("tr")

                for row in rows:
                    cells = row.find_all("td")
                    if len(cells) < 5:  # Need minimum data
                        continue

                    race_entry = {
                        "dog_name": dog_name,
                        "race_date": None,
                        "finish_position": None,
                        "box_number": None,
                        "margin": None,
                        "track": None,
                        "distance": None,
                        "grade": None,
                        "individual_time": None,
                        "winning_time": None,
                        "best_time_of_night": None,
                        "sectional_time": None,
                        "in_run": None,
                        "weight": None,
                        "starting_price": None,
                        "winner_second": None,
                    }

                    # Map cells to fields based on typical TGR structure:
                    # Date, Fin, Box, Mgn, Trk, Dis, Grd, Time, Win T, Bon, Sect, In Run, Weight, SP, Winner/Second

                    for i, cell in enumerate(cells):
                        cell_text = cell.get_text(strip=True)

                        if i == 0:  # Date
                            race_entry["race_date"] = cell_text
                        elif i == 1:  # Fin (Finishing position)
                            # Extract numeric position from text like "1st", "2nd", etc.
                            pos_match = re.search(r"(\d+)", cell_text)
                            if pos_match:
                                race_entry["finish_position"] = int(pos_match.group(1))
                        elif i == 2:  # Box
                            # Remove parentheses from "(1)" format
                            box_text = cell_text.replace("(", "").replace(")", "")
                            if box_text.isdigit():
                                race_entry["box_number"] = int(box_text)
                        elif i == 3:  # Margin
                            try:
                                race_entry["margin"] = float(cell_text)
                            except:
                                race_entry["margin"] = cell_text
                        elif i == 4:  # Track
                            race_entry["track"] = cell_text
                        elif i == 5:  # Distance
                            race_entry["distance"] = cell_text
                        elif i == 6:  # Grade
                            race_entry["grade"] = cell_text
                        elif i == 7:  # Time (individual time)
                            try:
                                race_entry["individual_time"] = float(cell_text)
                            except:
                                pass
                        elif i == 8:  # Win T (winning time)
                            try:
                                race_entry["winning_time"] = float(cell_text)
                            except:
                                pass
                        elif i == 9:  # BON (Best of night)
                            try:
                                race_entry["best_time_of_night"] = float(cell_text)
                            except:
                                pass
                        elif i == 10:  # Sect (sectional)
                            try:
                                race_entry["sectional_time"] = float(cell_text)
                            except:
                                pass
                        elif i == 11:  # In Run
                            race_entry["in_run"] = cell_text
                        elif i == 15:  # Weight (usually around position 15)
                            try:
                                race_entry["weight"] = float(cell_text)
                            except:
                                pass
                        elif i == 16:  # SP (Starting Price)
                            race_entry["starting_price"] = cell_text
                        elif i == 17:  # Winner/Second
                            race_entry["winner_second"] = cell_text

                    # Only add race if we have meaningful data
                    if race_entry["race_date"] and race_entry["finish_position"]:
                        racing_history.append(race_entry)

        except Exception as e:
            self.logger.debug(f"Error extracting racing history for {dog_name}: {e}")

        return racing_history

    return _extract_dog_racing_history


def create_fixed_extract_dog_entries():
    """Return a fixed _extract_dog_entries method."""

    def _extract_dog_entries(
        self, race_data: Dict[str, Any], target_dog_name: str
    ) -> List[Dict[str, Any]]:
        """Extract entries for a specific dog from race data - now handles racing history."""

        entries = []
        target_name_clean = target_dog_name.upper().strip()

        for dog in race_data.get("dogs", []):
            dog_name = dog.get("dog_name", "").upper().strip()
            if dog_name == target_name_clean:
                # Now we have racing history for this dog
                racing_history = dog.get("racing_history", [])

                # Convert each race in the history to the expected format
                for race in racing_history:
                    enhanced_entry = {
                        "dog_name": target_dog_name,
                        "race_date": race.get("race_date"),
                        "venue": race.get("track"),  # track -> venue
                        "finish_position": race.get("finish_position"),
                        "box_number": race.get("box_number"),
                        "margin": race.get("margin"),
                        "distance": race.get("distance"),
                        "grade": race.get("grade"),
                        "individual_time": race.get("individual_time"),
                        "winning_time": race.get("winning_time"),
                        "best_time_of_night": race.get("best_time_of_night"),
                        "sectional_time": race.get("sectional_time"),
                        "weight": race.get("weight"),
                        "starting_price": race.get("starting_price"),
                        "winner_second": race.get("winner_second"),
                        "in_run": race.get("in_run"),
                        # Add race metadata from the page
                        "race_url": race_data.get("url"),
                        "expert_comments": race_data.get("expert_comments", []),
                    }
                    entries.append(enhanced_entry)

        return entries

    return _extract_dog_entries


def test_fixed_implementation():
    """Test the fixed implementation."""

    from src.collectors.the_greyhound_recorder_scraper import (
        TheGreyhoundRecorderScraper,
    )

    # Create a test scraper
    scraper = TheGreyhoundRecorderScraper(rate_limit=2.0, use_cache=True)

    # Replace methods with fixed versions
    scraper._fetch_race_details = create_fixed_fetch_race_details().__get__(
        scraper, type(scraper)
    )
    scraper._extract_dog_racing_history = create_extract_dog_racing_history().__get__(
        scraper, type(scraper)
    )
    scraper._extract_dog_entries = create_fixed_extract_dog_entries().__get__(
        scraper, type(scraper)
    )

    # Test with known TGR URL
    test_url = "/form-guides/murray-bridge/long-form/244836/1/"

    logger.info(f"🧪 Testing fixed TGR implementation with: {test_url}")
    race_data = scraper._fetch_race_details(test_url)

    logger.info(f"📊 Results:")
    logger.info(f"  Dogs found: {len(race_data.get('dogs', []))}")
    logger.info(f"  Venue: {race_data.get('venue')}")
    logger.info(f"  Date: {race_data.get('date')}")
    logger.info(f"  Race number: {race_data.get('race_number')}")

    if race_data.get("dogs"):
        sample_dog = race_data["dogs"][0]
        logger.info(f"  Sample dog: {sample_dog['dog_name']}")
        logger.info(f"  Racing history entries: {sample_dog.get('total_races', 0)}")

        if sample_dog.get("racing_history"):
            sample_race = sample_dog["racing_history"][0]
            logger.info(f"  Sample race entry: {sample_race}")

    # Test enhanced dog data collection with a specific dog
    if race_data.get("dogs"):
        test_dog_name = race_data["dogs"][0]["dog_name"]
        logger.info(f"\\n🐕 Testing enhanced data collection for: {test_dog_name}")

        enhanced_data = scraper.fetch_enhanced_dog_data(test_dog_name)
        logger.info(
            f"  Total race entries found: {len(enhanced_data.get('form_entries', []))}"
        )

        if enhanced_data.get("form_entries"):
            logger.info(f"  Sample enhanced entry: {enhanced_data['form_entries'][0]}")


def main():
    """Main function to test the fixed implementation."""
    logging.basicConfig(level=logging.INFO)
    test_fixed_implementation()


if __name__ == "__main__":
    main()
