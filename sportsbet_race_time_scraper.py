#!/usr/bin/env python3
"""
Sportsbet Race Time Scraper
===========================

This script scrapes accurate race start times from Sportsbet for organizing
upcoming greyhound races in chronological order.

Usage: python3 sportsbet_race_time_scraper.py
"""

import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


class SportsbetRaceTimeScraper:
    """Scraper for getting accurate race times from Sportsbet"""

    def __init__(self):
        self.base_url = "https://www.sportsbet.com.au"
        self.greyhound_url = f"{self.base_url}/betting/greyhound-racing"
        self.driver = None
        self.race_times = {}

    def setup_driver(self):
        """Setup Chrome driver for web scraping"""
        if self.driver:
            return True

        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
            self.driver.execute_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )
            print("✅ Chrome driver setup successful")
            return True
        except Exception as e:
            print(f"❌ Chrome driver setup failed: {e}")
            self.driver = None
            return False

    def close_driver(self):
        """Close the web driver"""
        if self.driver:
            self.driver.quit()
            self.driver = None

    def scrape_race_times(self) -> Dict[str, List[Dict]]:
        """Scrape race times from Sportsbet greyhound racing page"""
        if not self.setup_driver():
            return {}

        try:
            print("🏁 Scraping race times from Sportsbet...")
            print(f"🌐 Navigating to: {self.greyhound_url}")

            self.driver.get(self.greyhound_url)

            # Wait for page to load
            try:
                WebDriverWait(self.driver, 15).until(
                    lambda driver: driver.execute_script("return document.readyState")
                    == "complete"
                )
                print("📄 Page loaded successfully")
            except TimeoutException:
                print("⚠️ Timeout waiting for page to load")

            # Give additional time for dynamic content
            time.sleep(5)

            # Extract race information
            races = self._extract_races_from_page()

            return races

        except Exception as e:
            print(f"❌ Error scraping race times: {e}")
            return {}
        finally:
            self.close_driver()

    def _extract_races_from_page(self) -> Dict[str, List[Dict]]:
        """Extract race information from the Sportsbet page"""
        races_by_venue = {}

        try:
            print("🔍 Looking for race elements...")

            # Strategy 1: Look for race cards or meeting elements
            race_selectors = [
                "[data-automation-id*='race']",
                "[class*='race-card']",
                "[class*='meeting']",
                "a[href*='greyhound-racing']",
                ".upcoming-races",
                "[data-testid*='race']",
            ]

            race_elements = []
            for selector in race_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        race_elements.extend(elements)
                        print(
                            f"  📊 Found {len(elements)} elements with selector: {selector}"
                        )
                except:
                    continue

            if not race_elements:
                print("⚠️ No specific race elements found, trying broader search...")
                race_elements = self._find_races_by_content()

            print(f"📋 Processing {len(race_elements)} potential race elements...")

            # Process each element
            for i, element in enumerate(race_elements[:20]):  # Limit to first 20
                try:
                    race_info = self._extract_race_info_from_element(element, i)
                    if race_info:
                        venue = race_info["venue"]
                        if venue not in races_by_venue:
                            races_by_venue[venue] = []
                        races_by_venue[venue].append(race_info)
                        print(
                            f"  ✅ Added: {race_info['venue']} Race {race_info.get('race_number', '?')} at {race_info['race_time']}"
                        )
                except Exception as e:
                    print(f"  ⚠️ Error processing element {i}: {e}")
                    continue

            return races_by_venue

        except Exception as e:
            print(f"❌ Error extracting races from page: {e}")
            return {}

    def _find_races_by_content(self) -> List:
        """Find race elements by analyzing page content"""
        try:
            # Look for elements containing racing-related text
            all_elements = self.driver.find_elements(
                By.CSS_SELECTOR, "div, section, article, li"
            )
            race_elements = []

            for element in all_elements:
                try:
                    text = element.text.strip().lower()
                    # Look for elements that might contain race info
                    if (
                        ("race" in text or "r1" in text or "r2" in text)
                        and ("pm" in text or "am" in text or ":" in text)
                        and len(text) > 10
                        and len(text) < 200
                    ):
                        race_elements.append(element)
                        if len(race_elements) >= 20:
                            break
                except:
                    continue

            print(f"📊 Found {len(race_elements)} elements by content analysis")
            return race_elements

        except Exception as e:
            print(f"⚠️ Error in content-based search: {e}")
            return []

    def _extract_race_info_from_element(self, element, index: int) -> Optional[Dict]:
        """Extract race information from a single element"""
        try:
            text = element.text.strip()
            if not text or len(text) < 5:
                return None

            # Try to get link URL for more context
            race_url = None
            try:
                if element.tag_name == "a":
                    race_url = element.get_attribute("href")
                else:
                    link = element.find_element(By.CSS_SELECTOR, "a")
                    race_url = link.get_attribute("href")
            except:
                pass

            # Extract venue name
            venue = self._extract_venue_from_text(text, race_url)
            if not venue:
                return None

            # Extract race time
            race_time = self._extract_race_time_from_text(text)
            if not race_time:
                return None

            # Extract race number if possible
            race_number = self._extract_race_number_from_text(text, race_url)

            # Create race info
            race_info = {
                "venue": venue,
                "race_number": race_number or 1,
                "race_time": race_time,
                "source": "sportsbet",
                "raw_text": text[:100],  # First 100 chars for debugging
                "url": race_url,
            }

            return race_info

        except Exception as e:
            print(f"    ⚠️ Error extracting info from element: {e}")
            return None

    def _extract_venue_from_text(
        self, text: str, url: Optional[str] = None
    ) -> Optional[str]:
        """Extract venue name from text or URL"""
        # Known venue patterns
        venue_patterns = [
            "angle park",
            "sandown",
            "warrnambool",
            "bendigo",
            "geelong",
            "ballarat",
            "healesville",
            "sale",
            "richmond",
            "murray bridge",
            "mount gambier",
            "capalaba",
            "rockhampton",
            "broken hill",
            "grafton",
            "darwin",
            "the meadows",
            "wentworth park",
            "cannington",
            "northam",
            "mandurah",
        ]

        text_lower = text.lower()

        # Try to find venue in text
        for venue in venue_patterns:
            if venue in text_lower:
                return venue.title()

        # Try to extract from URL if available
        if url:
            for venue in venue_patterns:
                venue_slug = venue.replace(" ", "-")
                if venue_slug in url.lower():
                    return venue.title()

        # Look for patterns like "R1 VeneName" or "VenueName Australia"
        venue_match = re.search(
            r"(?:R\d+\s+)?([A-Za-z\s]+?)(?:\s+Australia|\s+\d|$)", text
        )
        if venue_match:
            potential_venue = venue_match.group(1).strip()
            if len(potential_venue) > 3 and len(potential_venue) < 20:
                return potential_venue.title()

        return None

    def _extract_race_time_from_text(self, text: str) -> Optional[str]:
        """Extract race time from text"""
        # Look for time patterns
        time_patterns = [
            r"(\d{1,2}:\d{2}\s*(?:AM|PM))",  # 7:45 PM
            r"(\d{1,2}:\d{2})",  # 19:45
            r"(\d{4})",  # 1945
        ]

        for pattern in time_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                time_str = match.group(1)
                # Validate and format time
                try:
                    if "AM" in time_str.upper() or "PM" in time_str.upper():
                        # Already in 12-hour format
                        return time_str.upper()
                    elif ":" in time_str:
                        # 24-hour format like "19:45"
                        hour, minute = time_str.split(":")
                        hour = int(hour)
                        minute = int(minute)
                        if 0 <= hour <= 23 and 0 <= minute <= 59:
                            if hour > 12:
                                return f"{hour-12}:{minute:02d} PM"
                            elif hour == 12:
                                return f"12:{minute:02d} PM"
                            elif hour == 0:
                                return f"12:{minute:02d} AM"
                            else:
                                return f"{hour}:{minute:02d} AM"
                    elif len(time_str) == 4 and time_str.isdigit():
                        # Format like "1945"
                        hour = int(time_str[:2])
                        minute = int(time_str[2:])
                        if 0 <= hour <= 23 and 0 <= minute <= 59:
                            if hour > 12:
                                return f"{hour-12}:{minute:02d} PM"
                            elif hour == 12:
                                return f"12:{minute:02d} PM"
                            elif hour == 0:
                                return f"12:{minute:02d} AM"
                            else:
                                return f"{hour}:{minute:02d} AM"
                except:
                    continue

        # Look for countdown timers (like "17m", "32m")
        countdown_match = re.search(r"(\d+)m(?:\s|$)", text)
        if countdown_match:
            minutes = int(countdown_match.group(1))
            if minutes < 120:  # Within 2 hours
                race_datetime = datetime.now() + timedelta(minutes=minutes)
                hour = race_datetime.hour
                minute = race_datetime.minute
                if hour > 12:
                    return f"{hour-12}:{minute:02d} PM"
                elif hour == 12:
                    return f"12:{minute:02d} PM"
                elif hour == 0:
                    return f"12:{minute:02d} AM"
                else:
                    return f"{hour}:{minute:02d} AM"

        return None

    def _extract_race_number_from_text(
        self, text: str, url: Optional[str] = None
    ) -> Optional[int]:
        """Extract race number from text or URL"""
        # Try URL first
        if url:
            race_match = re.search(r"/race-(\d+)-", url)
            if race_match:
                return int(race_match.group(1))

        # Try text patterns
        race_patterns = [
            r"R(\d+)",  # R1, R2, etc.
            r"Race\s*(\d+)",  # Race 1, Race 2
            r"#(\d+)",  # #1, #2
        ]

        for pattern in race_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                race_num = int(match.group(1))
                if 1 <= race_num <= 20:  # Reasonable race number
                    return race_num

        return None


def organize_races_with_sportsbet_times():
    """Main function to organize races using Sportsbet times"""
    print("🏁 Organizing upcoming races using Sportsbet race times...")

    # Scrape race times from Sportsbet
    scraper = SportsbetRaceTimeScraper()
    sportsbet_races = scraper.scrape_race_times()

    if not sportsbet_races:
        print(
            "❌ No race times found from Sportsbet, falling back to file-based organization..."
        )
        return organize_races_from_files()

    # Organize all races by time
    all_races = []
    for venue, races in sportsbet_races.items():
        for race in races:
            # Convert time to sortable format
            time_str = race["race_time"]
            try:
                if "AM" in time_str or "PM" in time_str:
                    time_obj = datetime.strptime(time_str, "%I:%M %p")
                else:
                    time_obj = datetime.strptime(time_str, "%H:%M")

                race["sort_time"] = time_obj.hour * 60 + time_obj.minute
                all_races.append(race)
            except:
                continue

    # Sort by time
    all_races.sort(key=lambda x: x["sort_time"])

    # Display results
    print(
        f"\n🕐 Today's races organized by Sportsbet start times ({len(all_races)} races):"
    )
    print("=" * 80)

    for i, race in enumerate(all_races, 1):
        venue = race["venue"]
        race_num = race["race_number"]
        race_time = race["race_time"]

        print(f"{i:2d}. {race_time:>8} 🌐 - Race {race_num:2d} at {venue}")

        if race.get("url"):
            print(f"     🔗 {race['url']}")

        print()

    print("Legend:")
    print("🌐 = Actual time scraped from Sportsbet")
    print("📅 = Estimated time based on race number")

    return all_races


def organize_races_from_files():
    """Fallback: organize races from existing CSV files"""
    upcoming_races_dir = Path("./upcoming_races")

    if not upcoming_races_dir.exists():
        print("❌ upcoming_races directory not found")
        return []

    races = []

    # Read existing files
    for file_path in upcoming_races_dir.glob("*.csv"):
        filename = file_path.name
        if filename.lower().startswith("readme"):
            continue

        # Extract race info from filename
        pattern = r"Race (\d+) - ([A-Z-]+) - (\d{4}-\d{2}-\d{2})\.csv"
        match = re.match(pattern, filename)

        if match:
            race_number = int(match.group(1))
            venue = match.group(2).replace("-", " ").replace("_", " ")

            # Estimate time
            base_hour = 13  # 1 PM
            total_minutes = (race_number - 1) * 25
            hour = base_hour + (total_minutes // 60)
            minute = total_minutes % 60

            if hour > 12:
                race_time = f"{hour - 12}:{minute:02d} PM"
            elif hour == 12:
                race_time = f"12:{minute:02d} PM"
            else:
                race_time = f"{hour}:{minute:02d} AM"

            races.append(
                {
                    "venue": venue,
                    "race_number": race_number,
                    "race_time": race_time,
                    "sort_time": hour * 60 + minute,
                    "source": "file_estimate",
                    "filename": filename,
                }
            )

    # Sort and display
    races.sort(key=lambda x: x["sort_time"])

    print(f"\n🕐 Races organized by estimated start times ({len(races)} races):")
    print("=" * 80)

    for i, race in enumerate(races, 1):
        print(
            f"{i:2d}. {race['race_time']:>8} 📅 - Race {race['race_number']:2d} at {race['venue']}"
        )
        print(f"     📁 {race['filename']}")
        print()

    return races


def main():
    """Main function"""
    try:
        organize_races_with_sportsbet_times()
    except KeyboardInterrupt:
        print("\n\n⚠️ Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
