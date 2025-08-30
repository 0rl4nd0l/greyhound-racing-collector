#!/usr/bin/env python3
"""
Upcoming Race Browser for thedogs.com.au
========================================

This script browses upcoming greyhound races and allows selective downloading
of form guides for analysis and prediction.

Author: AI Assistant
Date: July 23, 2025
"""

import os
import random
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup


class UpcomingRaceBrowser:
    def __init__(self):
        self.base_url = "https://www.thedogs.com.au"
        # Honor configured UPCOMING_RACES_DIR if provided; default to ./upcoming_races
        self.upcoming_dir = os.getenv("UPCOMING_RACES_DIR", "./upcoming_races")

        # Create directories
        os.makedirs(self.upcoming_dir, exist_ok=True)

        # Setup session
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )

        # Venue mapping
        self.venue_map = {
            "angle-park": "AP_K",
            "sandown": "SAN",
            "warrnambool": "WAR",
            "bendigo": "BEN",
            "geelong": "GEE",
            "ballarat": "BAL",
            "horsham": "HOR",
            "traralgon": "TRA",
            "dapto": "DAPT",
            "wentworth-park": "WPK",
            "albion-park": "ALBION",
            "cannington": "CANN",
            "the-meadows": "MEA",
            "healesville": "HEA",
            "sale": "SAL",
            "richmond": "RICH",
            "murray-bridge": "MURR",
            "gawler": "GAWL",
            "mount-gambier": "MOUNT",
            "northam": "NOR",
            "mandurah": "MAND",
            "gosford": "GOSF",
            "hobart": "HOBT",
            "the-gardens": "GRDN",
            "darwin": "DARW",
            "casino": "CASO",
        }

        print("🏁 Upcoming Race Browser initialized")
        print(f"📂 Upcoming races directory: {self.upcoming_dir}")

    def get_upcoming_races(self, days_ahead=0):
        """Get upcoming races for the next specified days"""
        races = []
        today = datetime.now().date()

        print(f"🔍 Fetching upcoming races for the next {days_ahead} days...")

        # Prioritize live scraping for real race times
        for i in range(days_ahead + 1):  # Include today
            check_date = today + timedelta(days=i)
            date_str = check_date.strftime("%Y-%m-%d")

            try:
                # Always try to scrape live data first for real times
                print(
                    f"   🌐 Scraping live data for {date_str} to get real race times..."
                )
                date_races = self.get_races_for_date(check_date)

                if date_races:
                    races.extend(date_races)
                    print(f"   ✅ Found {len(date_races)} live races for {date_str}")
                else:
                    # Fallback to cached races if live scraping fails
                    cached_races = self._get_cached_races_for_date(date_str)
                    if cached_races:
                        print(
                            f"   📋 Fallback: Using {len(cached_races)} cached races for {date_str}"
                        )
                        races.extend(cached_races)
                    else:
                        print(f"   ⚪ No races found for {date_str}")

                # Rate limiting for respectful scraping
                if i < days_ahead:
                    time.sleep(0.5)  # Slightly longer delay for live scraping

            except Exception as e:
                print(f"   ⚠️ Error scraping {date_str}: {e}")
                # Fallback to cached data on error
                cached_races = self._get_cached_races_for_date(date_str)
                if cached_races:
                    print(
                        f"   📋 Error fallback: Using {len(cached_races)} cached races for {date_str}"
                    )
                    races.extend(cached_races)
                continue

        # Sort races chronologically by date and time (robust to AM/PM or 24h times)
        def _minutes_from_time(time_str: str, race_num: int) -> int:
            """Return minutes since midnight. If missing/unparseable, estimate from race number."""
            try:
                if not time_str:
                    raise ValueError('no time')
                t = str(time_str).strip().upper()
                # 12-hour format with AM/PM
                if re.match(r"^\d{1,2}:\d{2}\s*[AP]M$", t):
                    dt = datetime.strptime(t.replace(" ", ""), "%I:%M%p")
                    return dt.hour * 60 + dt.minute
                # 24-hour HH:MM
                if re.match(r"^\d{1,2}:\d{2}$", t):
                    h, m = t.split(":")
                    return int(h) * 60 + int(m)
                # Fallback to estimate
                raise ValueError('unsupported format')
            except Exception:
                base_minutes = 13 * 60  # 1 PM baseline
                total = base_minutes + max(0, int(race_num) - 1) * 25
                return total

        def _sort_key(race: dict):
            date_str = race.get("date", "9999-12-31")
            rn = int(race.get("race_number", 999))
            mins = _minutes_from_time(race.get("race_time", ""), rn)
            # Sort by date asc, time (mins) asc, race_number asc
            return (date_str, mins, rn)

        races.sort(key=_sort_key)

        # Filter out past races (races that have already started)
        now = datetime.now()
        current_time = now.time()
        current_date = now.date().strftime("%Y-%m-%d")

        future_races = []
        for race in races:
            race_date = race.get("date", "")
            race_time_str = race.get("race_time", "")

            # Only filter for today's races
            if race_date == current_date and race_time_str:
                try:
                    # Parse race time
                    clean_time = race_time_str.strip().upper()
                    if "PM" in clean_time or "AM" in clean_time:
                        race_time_obj = datetime.strptime(clean_time, "%I:%M %p").time()

                        # Add 30 minute buffer (race might still be running)
                        race_end_time = datetime.combine(
                            now.date(), race_time_obj
                        ) + timedelta(minutes=30)

                        # Only include if race hasn't finished yet
                        if datetime.combine(now.date(), current_time) < race_end_time:
                            future_races.append(race)
                        else:
                            print(
                                f"   ⏰ Skipping past race: {race.get('title', 'Unknown')} at {race_time_str}"
                            )
                    else:
                        # If we can't parse time, include it to be safe
                        future_races.append(race)
                except Exception as e:
                    # If time parsing fails, include the race to be safe
                    future_races.append(race)
            else:
                # Include races from other dates
                future_races.append(race)

        print(
            f"✅ Found {len(future_races)} upcoming races (filtered past races, sorted by time)"
        )
        return future_races

    def generate_csv_link(self, venue, date, race_number):
        """Generate CSV download link for given race venue, date, and number"""
        try:
            return f"https://www.thedogs.com.au/Racing/{venue}/{date}/{race_number}/Form-Guide"
        except Exception as e:
            print(f"❌ Error generating CSV link: {e}")
            return None

    def get_races_for_date(self, date):
        """Get races for a specific date by scraping live data from thedogs.com"""
        date_str = date.strftime("%Y-%m-%d")

        print(f"🔍 Scraping live races for {date_str}...")

        try:
            races = []

            # First get cached CSV files (these are confirmed races) - PRIORITIZE THESE
            cached_races = self._get_cached_races_for_date(date_str)
            if cached_races:
                print(f"   📋 Found {len(cached_races)} cached races for {date_str}")
                races.extend(cached_races)
                print(f"   ✅ Added {len(cached_races)} cached races to results")

                # For cached races, try to enhance with live times from individual race pages
                print(f"   🔄 Enhancing cached races with live times...")
                for i, cached_race in enumerate(
                    cached_races[:5]
                ):  # Limit to first 5 for performance
                    if not cached_race.get("race_time") or cached_race.get(
                        "race_time"
                    ) in ["1:00 PM", "1:25 PM", "1:50 PM"]:
                        # These look like estimated times, try to get real times
                        real_race_time = self._scrape_race_time_from_page(
                            cached_race["url"]
                        )
                        if real_race_time:
                            cached_race["race_time"] = real_race_time
                            cached_race["time_source"] = "live_scraped"
                            print(
                                f"     ✅ Updated {cached_race['title']} with real time: {real_race_time}"
                            )

            # Try to scrape live data from main racing page (add to cached races)
            live_races = self._scrape_live_races_for_date(date_str)
            if live_races:
                print(
                    f"   🌐 Found {len(live_races)} live races from main page for {date_str}"
                )

                # Only add live races that aren't already in cached races
                existing_race_keys = set()
                for race in races:
                    race_key = f"{race.get('venue', '')}_{race.get('race_number', '')}_{race.get('date', '')}"
                    existing_race_keys.add(race_key)

                added_live_count = 0
                for live_race in live_races:
                    race_key = f"{live_race.get('venue', '')}_{live_race.get('race_number', '')}_{live_race.get('date', '')}"
                    if race_key not in existing_race_keys:
                        races.append(live_race)
                        existing_race_keys.add(race_key)
                        added_live_count += 1

                print(
                    f"   ➕ Added {added_live_count} additional live races to cached races"
                )

            print(
                f"   📊 Total races found: {len(races)} (Cached: {len(cached_races)}, Live: {len(live_races) if live_races else 0})"
            )

            # If we have cached races but main page didn't find all venues,
            # try to get live times for cached races
            if cached_races and len(live_races) < len(cached_races):
                print(f"   🔄 Enhancing cached races with live times...")
                for cached_race in cached_races:
                    # Find if this race already has live data
                    found_live = False
                    for race in races:
                        if (
                            race.get("venue") == cached_race.get("venue")
                            and race.get("race_number")
                            == cached_race.get("race_number")
                            and race.get("time_source") == "live_scraped"
                        ):
                            found_live = True
                            break

                    # If no live data found, try to get real race time
                    if not found_live:
                        real_race_time = self._scrape_race_time_from_page(
                            cached_race["url"]
                        )
                        if real_race_time:
                            # Update the cached race with real time
                            for race in races:
                                if race.get("venue") == cached_race.get(
                                    "venue"
                                ) and race.get("race_number") == cached_race.get(
                                    "race_number"
                                ):
                                    race["race_time"] = real_race_time
                                    race["time_source"] = "live_scraped"
                                    print(
                                        f"     ✅ Updated {race['title']} with real time: {real_race_time}"
                                    )
                                    break

            if not races:
                print(f"   ⚪ No races found for {date_str}")

            # Sort by time within the date (earliest first); tie-breaker by race number
            def _minutes_from_time(time_str: str, race_num: int) -> int:
                try:
                    if not time_str:
                        raise ValueError('no time')
                    t = str(time_str).strip().upper()
                    if re.match(r"^\d{1,2}:\d{2}\s*[AP]M$", t):
                        dt = datetime.strptime(t.replace(" ", ""), "%I:%M%p")
                        return dt.hour * 60 + dt.minute
                    if re.match(r"^\d{1,2}:\d{2}$", t):
                        h, m = t.split(":")
                        return int(h) * 60 + int(m)
                    raise ValueError('unsupported format')
                except Exception:
                    base_minutes = 13 * 60
                    return base_minutes + (max(0, race_num - 1) * 25)

            races.sort(key=lambda x: (_minutes_from_time(x.get("race_time", ""), int(x.get("race_number", 999))), int(x.get("race_number", 999))))

            return races

        except Exception as e:
            print(f"   ❌ Error checking {date_str}: {e}")
            return []

    def extract_race_info_from_csv_filename(self, filename, date_str):
        """Extract race information from CSV filename (e.g., 'Race 1 - BROKEN-HILL - 2025-07-27.csv')"""
        try:
            # Pattern: Race {number} - {venue} - {date}.csv
            pattern = r"Race (\d+) - ([A-Z-]+) - (\d{4}-\d{2}-\d{2})\.csv"
            match = re.match(pattern, filename)

            if not match:
                return None

            race_number = int(match.group(1))
            venue_code = match.group(2)
            file_date = match.group(3)

            # Map venue code to display name
            venue_name_map = {
                "AP_K": "Angle Park",
                "SAN": "Sandown",
                "WAR": "Warrnambool",
                "BEN": "Bendigo",
                "GEE": "Geelong",
                "BAL": "Ballarat",
                "HOR": "Horsham",
                "TRA": "Traralgon",
                "DAPT": "Dapto",
                "WPK": "Wentworth Park",
                "ALBION": "Albion Park",
                "CANN": "Cannington",
                "MEA": "The Meadows",
                "HEA": "Healesville",
                "SAL": "Sale",
                "RICH": "Richmond",
                "MURR": "Murray Bridge",
                "GAWL": "Gawler",
                "MOUNT": "Mount Gambier",
                "NOR": "Northam",
                "MAND": "Mandurah",
                "GOSF": "Gosford",
                "HOBT": "Hobart",
                "GRDN": "The Gardens",
                "DARW": "Darwin",
                "BROKEN-HILL": "Broken Hill",
                "CAPALABA": "Capalaba",
                "TAREE": "Taree",
            }

            venue_name = venue_name_map.get(
                venue_code, venue_code.replace("-", " ").title()
            )

            # Generate estimated race time based on race number
            # Most races start around 1 PM and run every 25 minutes
            base_hour = 13  # 1 PM
            base_minute = 0

            # Add 25 minutes per race number (typical spacing)
            total_minutes = base_minute + ((race_number - 1) * 25)
            hour = base_hour + (total_minutes // 60)
            minute = total_minutes % 60

            # Convert to 12-hour format
            if hour > 12:
                race_time = f"{hour - 12}:{minute:02d} PM"
            elif hour == 12:
                race_time = f"12:{minute:02d} PM"
            else:
                race_time = f"{hour}:{minute:02d} AM"

            # Create race URL (construct from venue mapping)
            venue_slug = None
            for slug, code in self.venue_map.items():
                if code == venue_code:
                    venue_slug = slug
                    break

            if not venue_slug:
                venue_slug = venue_code.lower().replace("_", "-")

            race_url = f"{self.base_url}/racing/{venue_slug}/{file_date}/{race_number}"

            # Try to extract additional info from CSV if it exists
            csv_path = os.path.join(self.upcoming_dir, filename)
            distance = None
            grade = None

            if os.path.exists(csv_path):
                try:
                    import csv

                    with open(csv_path, "r") as f:
                        reader = csv.DictReader(f)
                        first_row = next(reader, None)
                        if first_row:
                            # Try to get distance from DIST column
                            distance = first_row.get("DIST", None)
                            # Try to get grade from G column
                            grade = first_row.get("G", None)
                except Exception:
                    pass

            return {
                "date": file_date,
                "venue": venue_code,
                "venue_name": venue_name,
                "race_number": race_number,
                "race_time": race_time,
                "distance": distance,
                "grade": grade,
                "race_name": None,
                "url": race_url,
                "title": f"Race {race_number} - {venue_name} - {file_date}",
                "description": (
                    f"🕐 {race_time} | 🏁 {distance}m | 🏆 {grade}"
                    if distance and grade
                    else f"🕐 {race_time}"
                ),
            }

        except Exception as e:
            print(f"   ❌ Error extracting info from {filename}: {e}")
            return None

    def extract_race_info_from_element(self, element, date_str):
        """Extract race information from a DOM element"""
        try:
            # Try to find race number, venue, and other details
            text = element.get_text().strip()

            # Look for race number pattern
            race_num_match = re.search(r"race\s*(\d+)", text, re.I)
            if not race_num_match:
                return None

            race_number = race_num_match.group(1)

            # Look for venue information
            venue = None
            venue_text = None

            # Check for venue in text or nearby elements
            for venue_key, venue_code in self.venue_map.items():
                venue_name = venue_key.replace("-", " ").title()
                if venue_name.lower() in text.lower() or venue_code in text:
                    venue = venue_code
                    venue_text = venue_name
                    break

            if not venue:
                return None

            # Try to find a link to the race page
            link_element = element if element.name == "a" else element.find("a")
            race_url = None
            race_name = None

            if link_element:
                href = link_element.get("href")
                if href:
                    race_url = (
                        href if href.startswith("http") else f"{self.base_url}{href}"
                    )
                    # Try to extract race name from URL
                    url_parts = href.strip("/").split("/")
                    if len(url_parts) > 4 and "racing" in url_parts:
                        try:
                            racing_index = url_parts.index("racing")
                            if len(url_parts) > racing_index + 4:
                                race_name = (
                                    url_parts[racing_index + 4]
                                    .replace("-", " ")
                                    .title()
                                )
                        except (ValueError, IndexError):
                            pass

            if not race_url:
                # Construct URL based on date, venue, and race number
                venue_slug = next(
                    (k for k, v in self.venue_map.items() if v == venue), venue.lower()
                )
                race_url = (
                    f"{self.base_url}/racing/{venue_slug}/{date_str}/{race_number}"
                )

            # Enhanced extraction of race details with better patterns
            race_time = None
            distance = None
            grade = None

            # Look in surrounding elements for more context
            search_text = text
            if element.parent:
                parent_text = element.parent.get_text().strip()
                search_text = f"{text} {parent_text}"

            # Extract time with various patterns
            time_patterns = [
                r"(\d{1,2}:\d{2}\s*(?:AM|PM))",
                r"(\d{1,2}:\d{2})",
                r"Start\s*(\d{1,2}:\d{2})",
                r"Time\s*(\d{1,2}:\d{2})",
            ]

            for pattern in time_patterns:
                time_match = re.search(pattern, search_text, re.I)
                if time_match:
                    race_time = time_match.group(1)
                    break

            # Extract distance with various patterns
            distance_patterns = [
                r"(\d{3,4})m",
                r"(\d{3,4})\s*metre",
                r"Distance[:\s]*(\d{3,4})m?",
                r"(\d{3,4})\s*meter",
            ]

            for pattern in distance_patterns:
                distance_match = re.search(pattern, search_text, re.I)
                if distance_match:
                    distance = distance_match.group(1)
                    break

            # Extract grade with comprehensive patterns
            grade_patterns = [
                r"(Grade\s*\d+)",
                r"(G\d+)",
                r"(Maiden)",
                r"(Open)",
                r"(Novice)",
                r"(Final)",
                r"(Heat)",
                r"(Restricted)",
                r"(Mixed)",
                r"(Free For All)",
                r"(Qualifying)",
                r"(Provincial)",
                r"(Metropolitan)",
            ]

            for pattern in grade_patterns:
                grade_match = re.search(pattern, search_text, re.I)
                if grade_match:
                    grade = grade_match.group(1)
                    break

            # Create rich description
            description_parts = []
            if race_time:
                description_parts.append(f"🕐 {race_time}")
            if distance:
                description_parts.append(f"🏁 {distance}m")
            if grade:
                description_parts.append(f"🏆 {grade}")
            if race_name:
                description_parts.append(f"📋 {race_name}")

            description = (
                " | ".join(description_parts) if description_parts else text[:100]
            )
            if len(description) > 100:
                description = description[:97] + "..."

            return {
                "date": date_str,
                "venue": venue,
                "venue_name": venue_text,
                "race_number": race_number,
                "race_time": race_time,
                "distance": distance,
                "grade": grade,
                "race_name": race_name,
                "url": race_url,
                "title": f"Race {race_number} - {venue_text} - {date_str}",
                "description": description,
            }

        except Exception as e:
            return None

    def extract_race_info_from_link(self, link_element, href, date_str):
        """Extract race information from a race link"""
        try:
            # Parse URL to get venue and race number
            url_parts = href.strip("/").split("/")
            if len(url_parts) < 4:
                return None

            # URL format: /racing/{venue}/{date}/{race_number}/{optional_race_name}
            if "racing" not in url_parts:
                return None

            racing_index = url_parts.index("racing")
            if len(url_parts) <= racing_index + 3:
                return None

            venue_slug = url_parts[racing_index + 1]
            race_date = url_parts[racing_index + 2]
            race_number = url_parts[racing_index + 3]

            # Validate race number is numeric
            if not race_number.isdigit():
                return None

            # Map venue slug to venue code
            venue_code = self.venue_map.get(venue_slug, venue_slug.upper())
            venue_name = venue_slug.replace("-", " ").title()

            # Get link text and surrounding elements for additional information
            link_text = link_element.get_text().strip()

            # Try to get race name from URL if available
            race_name = None
            if len(url_parts) > racing_index + 4:
                race_name_part = url_parts[racing_index + 4]
                # Clean up race name
                race_name = race_name_part.replace("-", " ").title()

            # Look for race conditions in the link text and surrounding elements
            race_time = None
            distance = None
            grade = None

            # Check parent elements for more detailed race info
            parent_element = link_element.parent
            if parent_element:
                parent_text = parent_element.get_text().strip()
                combined_text = f"{link_text} {parent_text}"
            else:
                combined_text = link_text

            # Extract time (format like "7:45 PM" or "19:45" or valid 4-digit HHMM)
            time_patterns = [
                r"(\d{1,2}:\d{2}\s*(?:AM|PM))",   # 12-hour with AM/PM
                r"(\d{1,2}:\d{2})",                # 24-hour with colon
                r"(\d{4})",                         # 24-hour HHMM (validate)
            ]

            for pattern in time_patterns:
                time_match = re.search(pattern, combined_text, re.I)
                if not time_match:
                    continue
                raw_t = time_match.group(1).strip()
                # Normalize the captured time
                try:
                    if re.match(r"^\d{1,2}:\d{2}\s*(?:AM|PM)$", raw_t, re.I):
                        # Already in 12-hour format
                        race_time = raw_t.upper().replace(" ", " ")
                        break
                    elif re.match(r"^\d{1,2}:\d{2}$", raw_t):
                        # 24-hour with colon; keep as HH:MM
                        # Optionally convert to 12-hour for display consistency
                        h, m = map(int, raw_t.split(":"))
                        if 0 <= h <= 23 and 0 <= m <= 59:
                            from datetime import datetime as _dt
                            race_time = _dt.strptime(f"{h:02d}:{m:02d}", "%H:%M").strftime("%I:%M %p").lstrip("0")
                            break
                        else:
                            continue
                    elif re.match(r"^\d{4}$", raw_t):
                        # Raw HHMM digits; validate bounds
                        h = int(raw_t[:2])
                        m = int(raw_t[2:])
                        if 0 <= h <= 23 and 0 <= m <= 59:
                            from datetime import datetime as _dt
                            race_time = _dt.strptime(f"{h:02d}:{m:02d}", "%H:%M").strftime("%I:%M %p").lstrip("0")
                            break
                        else:
                            # Ignore invalid 4-digit times like 7215
                            continue
                except Exception:
                    continue

            # Extract distance (format like "520m", "715m")
            distance_patterns = [
                r"(\d{3,4})m",
                r"(\d{3,4})\s*metre",
                r"(\d{3,4})\s*meter",
            ]

            for pattern in distance_patterns:
                distance_match = re.search(pattern, combined_text, re.I)
                if distance_match:
                    distance = distance_match.group(1)
                    break

            # Extract grade (more comprehensive patterns)
            grade_patterns = [
                r"(Grade\s*\d+)",
                r"(G\d+)",
                r"(Maiden)",
                r"(Open)",
                r"(Novice)",
                r"(Final)",
                r"(Heat)",
                r"(Restricted)",
                r"(Mixed)",
                r"(Free For All)",
            ]

            for pattern in grade_patterns:
                grade_match = re.search(pattern, combined_text, re.I)
                if grade_match:
                    grade = grade_match.group(1)
                    break

            # If we didn't find grade in text, try to extract from race name
            if not grade and race_name:
                for pattern in grade_patterns:
                    grade_match = re.search(pattern, race_name, re.I)
                    if grade_match:
                        grade = grade_match.group(1)
                        break

            race_url = href if href.startswith("http") else f"{self.base_url}{href}"

            # Create description with available information
            description_parts = []
            if race_time:
                description_parts.append(f"Time: {race_time}")
            if distance:
                description_parts.append(f"Distance: {distance}m")
            if grade:
                description_parts.append(f"Grade: {grade}")
            if race_name:
                description_parts.append(f"Race: {race_name}")

            description = (
                " • ".join(description_parts) if description_parts else link_text[:100]
            )

            return {
                "date": race_date,
                "venue": venue_code,
                "venue_name": venue_name,
                "race_number": race_number,
                "race_time": race_time,
                "distance": distance,
                "grade": grade,
                "race_name": race_name,
                "url": race_url,
                "title": f"Race {race_number} - {venue_name} - {race_date}",
                "description": description,
            }

        except Exception as e:
            return None

    def download_race_csv(self, race_url):
        """Download CSV form guide for a specific race"""
        try:
            print(f"🔄 Downloading CSV for: {race_url}")

            # Get race page
            response = self.session.get(race_url, timeout=30)

            if response.status_code == 404:
                return {
                    "success": False,
                    "error": "Race page not found (404). Please check the race URL or try again later.",
                }
            elif response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Failed to access race page: {response.status_code}",
                }

            soup = BeautifulSoup(response.content, "html.parser")

            # Extract race information for filename
            race_info = self.extract_detailed_race_info(soup, race_url)

            if not race_info:
                return {"success": False, "error": "Could not extract race information"}

            # Generate filename
            filename = f"Race {race_info['race_number']} - {race_info['venue']} - {race_info['date']}.csv"
            filepath = os.path.join(self.upcoming_dir, filename)

            # Check if already exists
            if os.path.exists(filepath):
                return {"success": False, "error": f"File already exists: {filename}"}

            # Find CSV download link
            csv_info = self.find_csv_download_link(soup, race_url)

            if not csv_info:
                return {"success": False, "error": "No CSV download link found"}

            # Download CSV
            if isinstance(csv_info, dict) and csv_info.get("type") == "form_post":
                # Handle form POST request
                csv_response = self.session.post(
                    csv_info["url"], data=csv_info["data"], timeout=30
                )
            else:
                # Handle direct URL request
                if isinstance(csv_info, str):
                    csv_url = csv_info
                elif isinstance(csv_info, dict):
                    csv_url = csv_info.get("url")
                else:
                    # csv_info is None or unexpected type
                    return {"success": False, "error": "Invalid CSV info returned from link finder"}
                
                if not csv_url:
                    return {"success": False, "error": "No valid CSV URL found"}
                
                csv_response = self.session.get(csv_url, timeout=30)

            if csv_response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Failed to download CSV: {csv_response.status_code}",
                }

            # Validate CSV content
            content = csv_response.text
            if not content.strip():
                return {"success": False, "error": "Empty CSV content"}

            lines = content.strip().split("\n")
            if len(lines) < 2:
                return {"success": False, "error": "CSV has insufficient data"}

            # Check for expected headers
            first_line = lines[0].lower()
            if not any(
                header in first_line for header in ["dog name", "dog", "runner", "name"]
            ):
                return {
                    "success": False,
                    "error": "CSV doesn't appear to be a form guide",
                }

            # Save file
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

            print(f"   ✅ Downloaded: {filename}")

            return {"success": True, "filename": filename, "filepath": filepath}

        except Exception as e:
            return {"success": False, "error": f"Error downloading race CSV: {str(e)}"}

    def extract_detailed_race_info(self, soup, race_url):
        """Extract detailed race information from race page"""
        try:
            race_number = None
            venue = None
            date = None

            # Try to extract from page title or headers
            title_element = soup.find("title")
            if title_element:
                title_text = title_element.get_text()

                # Look for race number
                race_match = re.search(r"race\s*(\d+)", title_text, re.I)
                if race_match:
                    race_number = race_match.group(1)

            # Try to extract from URL
            url_parts = race_url.strip("/").split("/")
            if len(url_parts) >= 4:
                try:
                    racing_index = url_parts.index("racing")
                    if len(url_parts) > racing_index + 3:
                        venue_slug = url_parts[racing_index + 1]
                        url_date = url_parts[racing_index + 2]
                        if not race_number:
                            race_number = url_parts[racing_index + 3]

                        # Map venue slug to code
                        venue = self.venue_map.get(venue_slug, venue_slug.upper())

                        # Convert date from YYYY-MM-DD to YYYY-MM-DD for consistency
                        if url_date and re.match(r"\d{4}-\d{2}-\d{2}", url_date):
                            date = url_date  # Keep as YYYY-MM-DD format
                        else:
                            date = url_date
                except ValueError:
                    pass

            # Try to extract venue from page content
            if not venue:
                for element in soup.find_all(["h1", "h2", "h3", "div", "span"]):
                    text = element.get_text(strip=True)
                    for venue_key, venue_code in self.venue_map.items():
                        if venue_key.replace("-", " ").lower() in text.lower():
                            venue = venue_code
                            break
                    if venue:
                        break

            # Try to extract date from page content
            if not date:
                date_elements = soup.find_all(
                    string=re.compile(
                        r"\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}\s+\w+\s+\d{4}"
                    )
                )
                for date_str in date_elements:
                    # Try to parse date
                    try:
                        if "/" in date_str:
                            date_obj = datetime.strptime(date_str.strip(), "%d/%m/%Y")
                        elif "-" in date_str:
                            date_obj = datetime.strptime(date_str.strip(), "%Y-%m-%d")
                        else:
                            date_obj = datetime.strptime(date_str.strip(), "%d %B %Y")

                        date = date_obj.strftime(
                            "%Y-%m-%d"
                        )  # Keep consistent YYYY-MM-DD format
                        break
                    except ValueError:
                        continue

            if race_number and venue and date:
                return {"race_number": race_number, "venue": venue, "date": date}

            return None

        except Exception as e:
            print(f"   ❌ Error extracting race info: {e}")
            return None

    def find_csv_download_link(self, soup, race_url):
        """Find CSV download link on the race page"""
        try:
            # Try the expert-form page method first
            base_race_url = race_url.split("?")[0].rstrip("/")
            expert_form_url = f"{base_race_url}/expert-form"

            print(f"   🔍 Trying expert-form URL: {expert_form_url}")
            response = self.session.get(expert_form_url, timeout=15)

            if response.status_code == 200:
                expert_soup = BeautifulSoup(response.content, "html.parser")

                # Method 1: Look for direct CSV download links
                csv_links = expert_soup.find_all("a", href=True)
                for link in csv_links:
                    href = link.get("href")
                    link_text = link.get_text().strip().lower()

                    if (
                        any(
                            keyword in link_text
                            for keyword in ["csv", "export", "download"]
                        )
                        or "csv" in href.lower()
                    ):
                        if href.startswith("/"):
                            csv_url = f"{self.base_url}{href}"
                        elif href.startswith("http"):
                            csv_url = href
                        else:
                            csv_url = f"{self.base_url}/{href}"

                        print(f"   ✅ Found CSV link: {csv_url}")
                        return csv_url

                # Method 2: Look for any download buttons or links
                download_elements = expert_soup.find_all(
                    ["a", "button", "input"],
                    string=re.compile("download|export|csv", re.I),
                )
                for element in download_elements:
                    if element.name == "a" and element.get("href"):
                        href = element.get("href")
                        if href.startswith("/"):
                            csv_url = f"{self.base_url}{href}"
                        elif href.startswith("http"):
                            csv_url = href
                        else:
                            csv_url = f"{self.base_url}/{href}"

                        print(f"   ✅ Found download link: {csv_url}")
                        return csv_url

                # Method 3: Look for forms with CSV export functionality
                forms = expert_soup.find_all("form")
                for form in forms:
                    # Check for CSV export buttons or inputs
                    csv_elements = form.find_all(
                        ["input", "button"],
                        attrs={"name": re.compile("csv|export", re.I)},
                    )

                    if not csv_elements:
                        # Also check for elements with CSV in their text or value
                        csv_elements = form.find_all(
                            ["input", "button"], string=re.compile("csv|export", re.I)
                        )

                    # Also check for any submit buttons in forms that might export data
                    if not csv_elements:
                        submit_buttons = form.find_all(
                            ["input", "button"], attrs={"type": "submit"}
                        )
                        for btn in submit_buttons:
                            btn_text = (
                                btn.get_text().strip().lower() if btn.get_text() else ""
                            )
                            btn_value = btn.get("value", "").lower()
                            if any(
                                keyword in f"{btn_text} {btn_value}"
                                for keyword in ["download", "export", "csv"]
                            ):
                                csv_elements.append(btn)

                    if csv_elements:
                        print(f"   📋 Found form with CSV export elements")

                        # Extract form data
                        form_action = form.get("action", "")
                        form_method = form.get("method", "GET").upper()

                        form_data = {}

                        # Get all form inputs (match the working scraper exactly)
                        for input_elem in form.find_all(
                            ["input", "select", "textarea"]
                        ):
                            name = input_elem.get("name")
                            if name:
                                input_type = input_elem.get("type", "text")

                                if input_type == "checkbox":
                                    if input_elem.get("checked"):
                                        form_data[name] = input_elem.get("value", "on")
                                elif input_type == "radio":
                                    if input_elem.get("checked"):
                                        form_data[name] = input_elem.get("value", "")
                                elif input_type == "submit":
                                    pass  # Don't include submit buttons in form data
                                elif input_type == "hidden":
                                    form_data[name] = input_elem.get("value", "")
                                else:
                                    form_data[name] = input_elem.get("value", "")

                        # Add the CSV export parameter (exactly like working scraper)
                        form_data["export_csv"] = "true"
                        print(f"   📋 Form data prepared: {form_data}")

                        # Determine target URL
                        if form_action:
                            if form_action.startswith("/"):
                                target_url = f"{self.base_url}{form_action}"
                            elif form_action.startswith("http"):
                                target_url = form_action
                            else:
                                target_url = f"{self.base_url}/{form_action}"
                        else:
                            target_url = expert_form_url

                        print(f"   📤 Submitting form to: {target_url}")
                        print(f"   📝 Form data: {form_data}")

                        # Submit form
                        try:
                            if form_method == "POST":
                                form_response = self.session.post(
                                    target_url, data=form_data, timeout=15
                                )
                            else:
                                form_response = self.session.get(
                                    target_url, params=form_data, timeout=15
                                )

                            if form_response.status_code == 200:
                                # The response should contain the actual download URL (like the working scraper)
                                download_url = form_response.text.strip()

                                print(
                                    f"   📄 Response length: {len(form_response.content)} bytes"
                                )
                                print(f"   📄 Response content: {download_url[:200]}")

                                if download_url.startswith("http"):
                                    print(f"   ✅ Got CSV download URL: {download_url}")
                                    return download_url
                                else:
                                    print(
                                        f"   ⚠️ Unexpected response format: {download_url[:100]}"
                                    )
                            else:
                                print(
                                    f"   ❌ Form submission failed with status: {form_response.status_code}"
                                )

                        except Exception as e:
                            print(f"   ⚠️ Error submitting form: {e}")
                            continue

                # Method 4: Look for JavaScript-generated CSV URLs
                script_tags = expert_soup.find_all("script")
                for script in script_tags:
                    script_text = script.get_text()
                    if "csv" in script_text.lower():
                        # Look for URL patterns in JavaScript
                        url_matches = re.findall(
                            r'["\']([^"\'\n]*csv[^"\'\n]*)["\']', script_text, re.I
                        )
                        for match in url_matches:
                            if match.startswith("/"):
                                csv_url = f"{self.base_url}{match}"
                            elif match.startswith("http"):
                                csv_url = match
                            else:
                                continue

                            print(f"   ✅ Found CSV URL in JavaScript: {csv_url}")
                            return csv_url

            else:
                print(f"   ❌ Expert-form page not accessible: {response.status_code}")

            # Try direct CSV URLs on expert-form page
            direct_csv_urls = [
                f"{expert_form_url}?export=csv",
                f"{expert_form_url}?format=csv",
                f"{expert_form_url}?export_csv=1",
                f"{expert_form_url}?download=csv",
            ]

            for csv_url in direct_csv_urls:
                try:
                    print(f"   🔍 Trying direct CSV URL: {csv_url}")
                    response = self.session.get(csv_url, timeout=10)
                    if response.status_code == 200:
                        content_type = response.headers.get("content-type", "").lower()
                        if "csv" in content_type or (
                            "text" in content_type and len(response.content) > 100
                        ):
                            # Validate it looks like CSV content
                            content_sample = response.text[:300].lower()
                            if any(
                                indicator in content_sample
                                for indicator in [
                                    "dog name",
                                    "runner",
                                    "barrier",
                                    "trainer",
                                    "box",
                                    "form",
                                ]
                            ) or ("," in content_sample and "\n" in content_sample):
                                print(f"   ✅ Direct CSV URL worked: {csv_url}")
                                return csv_url
                except Exception as e:
                    print(f"   ⚠️ Error with direct CSV URL {csv_url}: {e}")
                    continue

        except Exception as e:
            print(f"   ⚠️ Error with expert-form method: {e}")

        # Fallback 1: Look for CSV links on the main race page
        print(f"   🔍 Checking main race page for CSV links...")
        main_csv_selectors = [
            'a[href*="csv"]',
            'a[href*="export"]',
            'a[href*="download"]',
            ".csv-download",
            ".export-csv",
            ".download-csv",
        ]

        for selector in main_csv_selectors:
            elements = soup.select(selector)
            for element in elements:
                href = element.get("href")
                element_text = element.get_text().strip().lower()
                if href and ("csv" in href.lower() or "csv" in element_text):
                    if href.startswith("/"):
                        href = f"{self.base_url}{href}"
                    elif not href.startswith("http"):
                        href = f"{self.base_url}/{href}"

                    print(f"   ✅ Found CSV link on main page: {href}")
                    return href

        # Fallback 2: Try direct CSV URLs on the main race page
        base_race_url = race_url.split("?")[0]  # Remove query parameters
        direct_main_urls = [
            f"{base_race_url}?format=csv",
            f"{base_race_url}?export=csv",
            f"{base_race_url}/export/csv",
            f"{base_race_url}/download/csv",
            f"{base_race_url}/csv",
        ]

        for csv_url in direct_main_urls:
            try:
                print(f"   🔍 Trying direct main page CSV URL: {csv_url}")
                response = self.session.get(csv_url, timeout=10)
                if response.status_code == 200:
                    content_type = response.headers.get("content-type", "").lower()
                    if "csv" in content_type or (
                        "text" in content_type and len(response.content) > 100
                    ):
                        # Validate it looks like CSV content
                        content_sample = response.text[:300].lower()
                        if any(
                            indicator in content_sample
                            for indicator in [
                                "dog name",
                                "runner",
                                "barrier",
                                "trainer",
                                "box",
                                "form",
                            ]
                        ) or ("," in content_sample and "\n" in content_sample):
                            print(f"   ✅ Direct main page CSV URL worked: {csv_url}")
                            return csv_url
            except Exception as e:
                continue

        # Try common CSV URL patterns
        common_patterns = [
            f"{race_url}/csv",
            f"{race_url}/export",
            f"{race_url}/download",
            f"{race_url}.csv",
            f"{race_url}/form-guide.csv",
        ]

        for pattern in common_patterns:
            try:
                response = self.session.head(pattern, timeout=10)
                if response.status_code == 200:
                    content_type = response.headers.get("content-type", "").lower()
                    if "csv" in content_type or "text" in content_type:
                        return pattern
            except:
                continue

        return None

    def _get_cached_races_for_date(self, date_str):
        """Get races from cached CSV files in upcoming_races directory"""
        races = []

        try:
            if os.path.exists(self.upcoming_dir):
                for filename in os.listdir(self.upcoming_dir):
                    if filename.endswith(".csv") and date_str in filename:
                        race_info = self.extract_race_info_from_csv_filename(
                            filename, date_str
                        )
                        if race_info:
                            races.append(race_info)
        except Exception as e:
            print(f"   ⚠️ Error reading cached races: {e}")

        return races

    def _scrape_race_time_from_page(self, race_url, max_retries=3):
        """Scrape actual race time from individual race page"""
        try:
            retry_count = 0
            while retry_count < max_retries:
                try:
                    print(
                        f"     🕐 Scraping race time from: {race_url} (attempt {retry_count + 1})"
                    )
                    response = self.session.get(race_url, timeout=15)

                    if response.status_code == 429:  # Too Many Requests
                        retry_delay = int(response.headers.get("Retry-After", 30))
                        print(f"     ⏳ Rate limited, waiting {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_count += 1
                        continue

                    if response.status_code == 404:
                        print(f"     ⚠️ Race page not found (404)")
                        return None

                    if response.status_code != 200:
                        print(
                            f"     ⚠️ Failed to access race page: {response.status_code}"
                        )
                        retry_count += 1
                        time.sleep(5)  # Wait 5 seconds before retry
                        continue

                    break  # Success - exit retry loop

                except (
                    requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError,
                ) as e:
                    print(f"     ⚠️ Network error: {e}")
                    if retry_count >= max_retries - 1:
                        return None
                    retry_count += 1
                    time.sleep(5)  # Wait 5 seconds before retry
                    continue

            if response.status_code != 200:
                print(f"     ⚠️ Failed to access race page: {response.status_code}")
                return None

            soup = BeautifulSoup(response.content, "html.parser")

            # Multiple strategies to find race time
            race_time = None

            # Strategy 1: Look for common race time selectors
            time_selectors = [
                ".race-time",
                ".start-time",
                ".race-start-time",
                '[class*="time"]',
                '[class*="start"]',
            ]

            for selector in time_selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text(strip=True)
                    # Look for time patterns in the text
                    time_match = re.search(r"(\d{1,2}:\d{2}\s*(?:AM|PM))", text, re.I)
                    if time_match:
                        race_time = time_match.group(1)
                        print(f"     ✅ Found race time: {race_time} (from {selector})")
                        return race_time

            # Strategy 2: Search entire page text for time patterns
            page_text = soup.get_text()

            # Look for structured data or JSON with race times
            json_patterns = [
                r'"startTime"\s*:\s*"([^"]+)"',
                r'"raceTime"\s*:\s*"([^"]+)"',
                r'"time"\s*:\s*"([^"]+)"',
                r'"start"\s*:\s*"([^"]+)"',
            ]

            for pattern in json_patterns:
                matches = re.findall(pattern, page_text, re.I)
                for match in matches:
                    # Try to parse as time
                    time_match = re.search(r"(\d{1,2}:\d{2}\s*(?:AM|PM))", match, re.I)
                    if time_match:
                        race_time = time_match.group(1)
                        print(f"     ✅ Found race time: {race_time} (from JSON data)")
                        return race_time

            # Enhanced time patterns for various formats
            time_patterns = [
                # Standard patterns
                r"Start[:\s]*(\d{1,2}:\d{2}\s*(?:AM|PM))",
                r"Time[:\s]*(\d{1,2}:\d{2}\s*(?:AM|PM))",
                r"Race\s*Time[:\s]*(\d{1,2}:\d{2}\s*(?:AM|PM))",
                # More flexible patterns
                r"(\d{1,2}:\d{2}\s*(?:AM|PM))(?:\s*(?:AEDT|AEST|EST|EDT))?",
                r"(?:at|@)\s*(\d{1,2}:\d{2}\s*(?:AM|PM))",
                r"(?:starts?|begins?)\s*(?:at)?\s*(\d{1,2}:\d{2}\s*(?:AM|PM))",
                # 24-hour format patterns
                r"(\d{2}:\d{2})(?:\s*hrs?)?",
                r"Start[:\s]*(\d{2}:\d{2})",
                r"Time[:\s]*(\d{2}:\d{2})",
                # Look for times in specific contexts
                r"Post\s*Time[:\s]*(\d{1,2}:\d{2}\s*(?:AM|PM))",
                r"Jump\s*Time[:\s]*(\d{1,2}:\d{2}\s*(?:AM|PM))",
                r"Off\s*Time[:\s]*(\d{1,2}:\d{2}\s*(?:AM|PM))",
            ]

            for pattern in time_patterns:
                matches = re.findall(pattern, page_text, re.I)
                if matches:
                    # Take the first reasonable match
                    for match in matches:
                        match = match.strip()

                        # Try to validate and format the time
                        try:
                            # Handle AM/PM format
                            if re.search(r"(AM|PM)", match, re.I):
                                time_obj = datetime.strptime(match.upper(), "%I:%M %p")
                                hour = time_obj.hour
                                # Race times typically between 8 AM and 11 PM
                                if 8 <= hour <= 23:
                                    race_time = match.upper()
                                    print(
                                        f"     ✅ Found race time: {race_time} (from page text - AM/PM)"
                                    )
                                    return race_time

                            # Handle 24-hour format
                            elif re.match(r"^\d{2}:\d{2}$", match):
                                hour, minute = map(int, match.split(":"))
                                if 8 <= hour <= 23:  # Reasonable race hours
                                    # Convert to 12-hour format
                                    time_obj = datetime.strptime(match, "%H:%M")
                                    formatted_time = time_obj.strftime(
                                        "%I:%M %p"
                                    ).lstrip("0")
                                    print(
                                        f"     ✅ Found race time: {formatted_time} (from page text - 24hr)"
                                    )
                                    return formatted_time

                        except ValueError:
                            continue

            # Strategy 2.5: Look for times in script tags or data attributes
            script_tags = soup.find_all("script")
            for script in script_tags:
                script_text = script.get_text()
                # Look for time data in JavaScript
                js_time_patterns = [
                    r'startTime["\']?\s*[=:]\s*["\']([^"\')]+)["\']',
                    r'raceTime["\']?\s*[=:]\s*["\']([^"\')]+)["\']',
                    r'time["\']?\s*[=:]\s*["\']([^"\')]+)["\']',
                ]

                for pattern in js_time_patterns:
                    matches = re.findall(pattern, script_text, re.I)
                    for match in matches:
                        time_match = re.search(
                            r"(\d{1,2}:\d{2}\s*(?:AM|PM))", match, re.I
                        )
                        if time_match:
                            race_time = time_match.group(1)
                            print(
                                f"     ✅ Found race time: {race_time} (from JavaScript)"
                            )
                            return race_time

            # Strategy 3: Look in meta tags or structured data
            meta_elements = soup.find_all("meta")
            for meta in meta_elements:
                content = meta.get("content", "")
                if content:
                    time_match = re.search(
                        r"(\d{1,2}:\d{2}\s*(?:AM|PM))", content, re.I
                    )
                    if time_match:
                        race_time = time_match.group(1)
                        print(f"     ✅ Found race time: {race_time} (from meta tag)")
                        return race_time

            print(f"     ⚠️ No race time found on page")
            return None

        except Exception as e:
            print(f"     ❌ Error scraping race time: {e}")
            return None

    def _scrape_live_races_for_date(self, date_str):
        """Scrape live races from thedogs.com for a specific date (optimized)"""
        races = []

        try:
            # Format date for URL (thedogs.com uses YYYY-MM-DD format)
            date_url = f"{self.base_url}/racing/{date_str}"

            print(f"   🌐 Fetching: {date_url}")
            # Reduced timeout for faster failure
            response = self.session.get(date_url, timeout=10)

            if response.status_code != 200:
                print(f"   ⚠️ Failed to access racing page: {response.status_code}")
                return races

            soup = BeautifulSoup(response.content, "html.parser")

            # Find race links using optimized strategy
            race_links = self._find_race_links_fast(soup, date_str)

            if not race_links:
                print(f"   ⚠️ No race links found on page")
                return races

            print(f"   🔍 Found {len(race_links)} potential race links")

            # Group race links by venue for more efficient processing
            venue_links = {}
            for link_element, href in race_links:
                try:
                    url_parts = href.strip("/").split("/")
                    racing_index = url_parts.index("racing")
                    if len(url_parts) > racing_index + 1:
                        venue = url_parts[racing_index + 1]
                        if venue not in venue_links:
                            venue_links[venue] = []
                        venue_links[venue].append((link_element, href))
                except (ValueError, IndexError):
                    continue

            print(f"   📊 Processing races for {len(venue_links)} venues")

            # Process venues in parallel using up to 3 concurrent threads
            from concurrent.futures import ThreadPoolExecutor
            from itertools import chain

            def process_venue_races(venue_data):
                venue, links = venue_data
                venue_races = []
                print(f"     🏟️ Processing {len(links)} races for {venue}")

                # Get the first race time to establish pattern
                first_link = links[0]
                first_race = self.extract_race_info_from_link(
                    first_link[0], first_link[1], date_str
                )
                if first_race:
                    first_time = self._scrape_race_time_from_page(first_race["url"])
                    if first_time:
                        first_race["race_time"] = first_time
                        first_race["time_source"] = "live_scraped"
                        venue_races.append(first_race)

                        # Estimate remaining race times (usually 20-25 min apart)
                        try:
                            from datetime import datetime, timedelta

                            base_time = datetime.strptime(first_time, "%I:%M %p")

                            # Process remaining races with estimated times
                            for link_element, href in links[1:]:
                                race_info = self.extract_race_info_from_link(
                                    link_element, href, date_str
                                )
                                if race_info:
                                    # Estimate time based on race number difference
                                    race_num_diff = int(race_info["race_number"]) - int(
                                        first_race["race_number"]
                                    )
                                    estimated_time = base_time + timedelta(
                                        minutes=race_num_diff * 22
                                    )
                                    race_info["race_time"] = estimated_time.strftime(
                                        "%I:%M %p"
                                    ).lstrip("0")
                                    race_info["time_source"] = "estimated"
                                    venue_races.append(race_info)
                        except Exception as e:
                            print(f"     ⚠️ Error estimating times for {venue}: {e}")

                            # Fallback: Get real times for all races
                            for link_element, href in links[1:]:
                                try:
                                    race_info = self.extract_race_info_from_link(
                                        link_element, href, date_str
                                    )
                                    if race_info:
                                        real_time = self._scrape_race_time_from_page(
                                            race_info["url"]
                                        )
                                        if real_time:
                                            race_info["race_time"] = real_time
                                            race_info["time_source"] = "live_scraped"
                                            venue_races.append(race_info)
                                except Exception as e:
                                    print(f"     ⚠️ Error processing race: {e}")
                                time.sleep(0.5 + random.random() * 0.5)
                return venue_races

            # Process venues concurrently
            with ThreadPoolExecutor(max_workers=3) as executor:
                all_venue_races = list(
                    chain.from_iterable(
                        executor.map(process_venue_races, venue_links.items())
                    )
                )

            races.extend(all_venue_races)

            # Sort races by venue and race number
            races.sort(key=lambda x: (x.get("venue", ""), int(x.get("race_number", 0))))

            print(
                f"   📊 Found {len(races)} total races across {len(venue_links)} venues"
            )

        except Exception as e:
            print(f"   ❌ Error scraping live races: {e}")

        return races

    def _find_race_links_fast(self, soup, date_str):
        """Find race links on the racing page using optimized strategy"""
        race_links = []

        # Strategy 1: Look for direct race links with racing pattern (fastest)
        race_selectors = [
            f'a[href*="/racing/"][href*="{date_str}"]',
            'a[href*="/racing/"]',
        ]

        for selector in race_selectors:
            try:
                links = soup.select(selector)
                for link in links:
                    href = link.get("href")
                    if href and self._is_valid_race_link(href, date_str):
                        race_links.append((link, href))
                        # Remove the limit to find ALL venues, not just first 10
                        # if len(race_links) >= 10:  # Limit for performance
                        #     break
                if race_links:
                    break  # Use first successful selector
            except Exception as e:
                continue

        # Remove duplicates based on href
        seen_hrefs = set()
        unique_links = []
        for link, href in race_links:
            if href not in seen_hrefs:
                seen_hrefs.add(href)
                unique_links.append((link, href))

        return unique_links

    def _find_race_links(self, soup, date_str):
        """Find race links on the racing page using multiple strategies"""
        race_links = []

        # Strategy 1: Look for direct race links with racing pattern
        race_selectors = [
            f'a[href*="/racing/"][href*="{date_str}"]',
            'a[href*="/racing/"]',
            ".race-link a",
            ".race-card a",
            'a[class*="race"]',
        ]

        for selector in race_selectors:
            try:
                links = soup.select(selector)
                for link in links:
                    href = link.get("href")
                    if href and self._is_valid_race_link(href, date_str):
                        race_links.append((link, href))
            except Exception as e:
                continue

        # Strategy 2: Look for links that contain race numbers and venue names
        if not race_links:
            all_links = soup.find_all("a", href=True)
            for link in all_links:
                href = link.get("href")
                if href and self._is_valid_race_link(href, date_str):
                    race_links.append((link, href))

        # Remove duplicates based on href
        seen_hrefs = set()
        unique_links = []
        for link, href in race_links:
            if href not in seen_hrefs:
                seen_hrefs.add(href)
                unique_links.append((link, href))

        return unique_links

    def _is_valid_race_link(self, href, date_str):
        """Check if a link is a valid race link"""
        try:
            # Must contain racing path
            if "/racing/" not in href:
                return False

            # Parse URL parts
            url_parts = href.strip("/").split("/")
            if "racing" not in url_parts:
                return False

            racing_index = url_parts.index("racing")

            # Must have at least venue, date, race_number after racing
            if len(url_parts) <= racing_index + 3:
                return False

            # Check if race number is numeric
            race_number_part = url_parts[racing_index + 3]
            if not race_number_part.isdigit():
                return False

            # Check if date matches (optional - some links might not have date)
            date_part = url_parts[racing_index + 2]
            if (
                date_str in href
                or date_part == date_str.replace("-", "/")
                or date_part == date_str
            ):
                return True

            # Also accept if it's a valid racing URL structure even without exact date match
            return True

        except Exception:
            return False


def test_csv_download(race_url=None):
    """Test CSV download for a specific race"""
    browser = UpcomingRaceBrowser()

    if not race_url:
        # Get a test race
        upcoming_races = browser.get_upcoming_races(days_ahead=1)
        if not upcoming_races:
            print("❌ No upcoming races found for testing")
            return
        race_url = upcoming_races[0]["url"]

    print(f"🧪 Testing CSV download for: {race_url}")
    result = browser.download_race_csv(race_url)

    if result["success"]:
        print(f"✅ Successfully downloaded: {result['filename']}")
        print(f"📁 File saved to: {result['filepath']}")
    else:
        print(f"❌ Download failed: {result['error']}")

    return result


def main():
    """Main function for testing"""
    browser = UpcomingRaceBrowser()
    upcoming_races = browser.get_upcoming_races(days_ahead=0)

    print(f"\n🎯 Found {len(upcoming_races)} upcoming races:")
    # Add links to thedogs.com.au CSV downloads
    for race in upcoming_races:
        csv_url = browser.generate_csv_link(
            race["venue"], race["date"], race["race_number"]
        )
        print(f"   📅 {race['title']} - {race.get('race_time', 'TBA')}")
        print(f"   🔗 CSV Download: {csv_url}")

    if upcoming_races:
        print(f"\n💡 To download a race, use the Flask app at /upcoming")
        print(
            f"💡 To test CSV download: python3 -c 'from upcoming_race_browser import test_csv_download; test_csv_download()'"
        )


if __name__ == "__main__":
    main()
