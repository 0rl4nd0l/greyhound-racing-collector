#!/usr/bin/env python3
"""
Form Guide CSV Scraper for thedogs.com.au
=========================================

This script downloads CSV form guides from race pages on thedogs.com.au
and saves them to the unprocessed folder for further analysis.

The CSV files contain individual greyhound form data including:
- Dog Name, Sex, Placing, Box, Weight, Distance, Date, Track, Grade
- Time, Win Time, Bonus, First Split, Margin, PIR, Starting Price

Usage: python3 form_guide_csv_scraper.py

Author: AI Assistant
Date: July 31, 2025
Version: 3.0.1 - Fixed regex patterns and centralized date parsing
"""

import os
import random
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from src.parsers.csv_ingestion import CsvIngestion
from utils.date_parsing import parse_date_flexible
from utils.http_client import get_shared_session
from utils.race_file_utils import RaceFileManager


class StatisticsTracker:
    """Lightweight statistics tracking utility using collections.Counter wrapper"""

    def __init__(self):
        """Initialize the statistics tracker with predefined counters"""
        from collections import Counter

        self.stats = Counter(
            {
                "races_requested": 0,
                "cache_hits": 0,
                "fetches_attempted": 0,
                "fetches_failed": 0,
                "successful_saves": 0,
            }
        )

    def increment(self, key, amount=1):
        """Increment a statistic counter by the specified amount"""
        if key in self.stats:
            self.stats[key] += amount
        else:
            # Allow tracking of additional stats not in the predefined set
            self.stats[key] += amount

    def get(self, key):
        """Get the current value of a statistic"""
        return self.stats.get(key, 0)

    def log_summary(self):
        """Log a summary of all statistics"""
        print("\n📊 Statistics Summary:")
        print(f"   🏁 Races requested: {self.stats['races_requested']}")
        print(f"   ⚡ Cache hits: {self.stats['cache_hits']}")
        print(f"   🌐 Fetches attempted: {self.stats['fetches_attempted']}")
        print(f"   ❌ Fetches failed: {self.stats['fetches_failed']}")
        print(f"   ✅ Successful saves: {self.stats['successful_saves']}")

        # Calculate derived metrics
        if self.stats["fetches_attempted"] > 0:
            success_rate = (
                (self.stats["fetches_attempted"] - self.stats["fetches_failed"])
                / self.stats["fetches_attempted"]
                * 100
            )
            print(f"   📈 Fetch success rate: {success_rate:.1f}%")

        if self.stats["races_requested"] > 0:
            cache_hit_rate = (
                self.stats["cache_hits"] / self.stats["races_requested"] * 100
            )
            print(f"   ⚡ Cache hit rate: {cache_hit_rate:.1f}%")

    def reset(self):
        """Reset all statistics to zero"""
        for key in self.stats:
            self.stats[key] = 0

    def to_dict(self):
        """Return statistics as a dictionary"""
        return dict(self.stats)


class FormGuideCsvScraper:
    def __init__(self, historical=False, verbose_fetch=False):
        self.base_url = "https://www.thedogs.com.au"
        self.unprocessed_dir = "./unprocessed"
        self.download_dir = "./form_guides/downloaded"
        self.database_path = "./databases/greyhound_racing.db"

        # Create directories
        os.makedirs(self.unprocessed_dir, exist_ok=True)
        os.makedirs(self.download_dir, exist_ok=True)

        # Initialize statistics tracker
        self.stats = StatisticsTracker()

        # Initialize shared race file manager
        self.race_file_manager = RaceFileManager(self.database_path)

        # Use manager's cached data
        self.collected_races = self.race_file_manager.collected_races
        self.existing_files = self.race_file_manager.existing_files
        self.processed_hashes = self.race_file_manager.processed_hashes

        # Additional tracking for scraper-specific functionality
        self.completed_dates = set()  # Stores dates that have been fully downloaded

        # Store CLI flags for global access
        self.historical = historical
        self.verbose_fetch = verbose_fetch

        # Note: We collect all historical races (previous day or earlier) for training data

        # Setup session
        self.session = get_shared_session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )

        # Comprehensive venue mapping for all Australian greyhound tracks
        self.venue_map = {
            # Major metropolitan tracks
            "angle-park": "AP_K",
            "sandown": "SAN",
            "warrnambool": "WAR",
            "bendigo": "BEN",
            "geelong": "GEE",
            "ballarat": "BAL",
            "horsham": "HOR",
            "traralgon": "TRA",
            "dapto": "DAPT",
            "wentworth-park": "W_PK",
            "albion-park": "APWE",
            "cannington": "CANN",
            "the-meadows": "MEA",
            "meadows": "MEA",
            "healesville": "HEA",
            "sale": "SAL",
            "richmond": "RICH",
            "richmond-straight": "RICH_S",
            "murray-bridge": "MURR",
            "gawler": "GAWL",
            "mount-gambier": "MOUNT",
            "northam": "NOR",
            "mandurah": "MAND",
            # NSW tracks
            "the-gardens": "GARD",
            "casino": "CASINO",
            "wagga": "WAG",
            "goulburn": "GOUL",
            "taree": "TAR",
            "dubbo": "DUB",
            "grafton": "GRAF",
            "broken-hill": "BH",
            "lismore": "LIS",
            "nowra": "NOW",
            "temora": "TEM",
            "young": "YOU",
            "orange": "ORA",
            "mudgee": "MUD",
            "cowra": "COW",
            "bathurst": "BAT",
            "katoomba": "KAT",
            "wollongong": "WOL",
            "ingle-farm": "INF",
            "bulli": "BUL",
            "raymond-terrace": "RAY",
            "maitland": "MAIT",
            # QLD tracks
            "ladbrokes-q1-lakeside": "Q1L",
            "ladbrokes-q-straight": "QST",
            "townsville": "TWN",
            "capalaba": "CAP",
            "ipswich": "IPS",
            "rockhampton": "ROCK",
            "bundaberg": "BUN",
            "cairns": "CAI",
            "mackay": "MAC",
            "toowoomba": "TOO",
            "gold-coast": "GC",
            "caloundra": "CAL",
            "maroochy": "MAR",
            # VIC tracks
            "shepparton": "SHEP",
            "warragul": "WRGL",
            "cranbourne": "CRAN",
            "moe": "MOE",
            "pakenham": "PAK",
            "colac": "COL",
            "hamilton": "HAM",
            "portland": "PORT",
            "ararat": "ARA",
            "stawell": "STA",
            "swan-hill": "SH",
            "mildura": "MIL",
            "echuca": "ECH",
            "seymour": "SEY",
            "kilmore": "KIL",
            "wodonga": "WOD",
            "wodonga-gvgrc": "WOD_G",
            # SA tracks
            "virginia": "VIR",
            "strathalbyn": "STR",
            "whyalla": "WHY",
            "port-augusta": "PA",
            "port-pirie": "PP",
            "glenelg": "GLE",
            # WA tracks
            "albany": "ALB",
            "geraldton": "GER",
            "kalgoorlie": "KAL",
            "bunbury": "BUNB",
            "esperance": "ESP",
            "broome": "BRO",
            "karratha": "KAR",
            "port-hedland": "PH",
            "kununurra": "KUN",
            # TAS tracks
            "hobart": "HOB",
            "launceston": "LAU",
            "devonport": "DEV",
            # NT tracks
            "darwin": "DAR",
            "alice-springs": "AS",
            # ACT tracks
            "canberra": "CANB",
        }

        print("🏁 Form Guide CSV Scraper initialized")
        print(f"📂 Unprocessed directory: {self.unprocessed_dir}")
        print(f"📂 Download directory: {self.download_dir}")
        print(
            f"🎯 Target: Historical races (previous day or earlier) for training data"
        )

    def load_collected_races(self):
        """Load all collected races from all directories to avoid re-downloading - delegate to manager"""
        # Delegate to the shared race file manager
        self.race_file_manager.reload_cache()

        # Update local references
        self.collected_races = self.race_file_manager.collected_races
        self.existing_files = self.race_file_manager.existing_files
        self.processed_hashes = self.race_file_manager.processed_hashes

    def _ensure_database_tables(self):
        """Ensure the processed_race_files table exists"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        # Create table if it doesn't exist
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_race_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash TEXT UNIQUE NOT NULL,
                race_date DATE NOT NULL,
                venue TEXT NOT NULL, 
                race_no INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER,
                processed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'processed',
                error_message TEXT
            )
        """
        )

        # Create indexes for performance
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_processed_files_hash ON processed_race_files(file_hash)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_processed_files_race_key ON processed_race_files(race_date, venue, race_no)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_processed_files_file_path ON processed_race_files(file_path)"
        )

        conn.commit()
        conn.close()

    def parse_filename_to_race_id(self, filename):
        """Parse filename to extract race_id tuple (date, venue, race_number) - delegate to manager"""
        return self.race_file_manager.parse_filename_to_race_id(filename)

    def compute_file_hash(self, file_path):
        """Compute SHA-256 hash of a file - delegate to manager"""
        return self.race_file_manager.compute_file_hash(file_path)

    def parse_csv_with_ingestion(self, file_path, force=False):
        """Parse CSV using CsvIngestion module with caching and de-duplication"""
        file_hash = self.compute_file_hash(file_path)

        # Check if file already processed (cache hit)
        if file_hash in self.processed_hashes and not force:
            self.stats.increment("cache_hits")
            print(f"⚠️ Cache HIT: Skipping already processed file: {file_path}")
            return "hit"

        print(f"🔄 Cache MISS: Processing file: {file_path}")

        # Extract race metadata from filename for database recording
        race_info = self.parse_filename_to_race_id(os.path.basename(file_path))

        try:
            # Parse CSV using ingestion module
            try:
                ingestion = CsvIngestion(file_path)
                parsed_race, validation_report = ingestion.parse_csv()
            except NameError as name_error:
                if "ParsedRace" in str(name_error):
                    # Fallback: Just validate that file is readable CSV
                    import csv

                    with open(file_path, "r", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        headers = next(reader)
                        row_count = sum(1 for row in reader)
                    print(f"📋 Parsed CSV: {len(headers)} columns, {row_count} rows")
                    parsed_race = {"headers": headers, "row_count": row_count}
                    validation_report = {"errors": []}
                else:
                    raise

            # Get file size
            file_size = os.path.getsize(file_path)

            # Record processing outcome in database
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            if race_info:
                race_date, venue, race_no = race_info
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO processed_race_files 
                    (file_hash, race_date, venue, race_no, file_path, file_size, status) 
                    VALUES (?, ?, ?, ?, ?, ?, 'processed')
                """,
                    (file_hash, race_date, venue, race_no, file_path, file_size),
                )
            else:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO processed_race_files 
                    (file_hash, race_date, venue, race_no, file_path, file_size, status) 
                    VALUES (?, 'unknown', 'unknown', 0, ?, ?, 'processed')
                """,
                    (file_hash, file_path, file_size),
                )

            conn.commit()
            conn.close()

            # Add to processed hashes to avoid re-processing in same session
            self.processed_hashes.add(file_hash)

            # Track successful save
            self.stats.increment("successful_saves")

            print(f"✅ Successfully processed and cached: {file_path}")
            return "miss"

        except Exception as e:
            print(f"❌ Error processing file {file_path}: {e}")

            # Record error in database
            try:
                conn = sqlite3.connect(self.database_path)
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO processed_race_files 
                    (file_hash, race_date, venue, race_no, file_path, file_size, status, error_message) 
                    VALUES (?, 'error', 'error', 0, ?, ?, 'failed', ?)
                """,
                    (
                        file_hash,
                        file_path,
                        os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                        str(e),
                    ),
                )
                conn.commit()
                conn.close()
            except Exception as db_error:
                print(f"❌ Error recording failure to database: {db_error}")

            return "error"

    def load_processed_races(self):
        """Load processed races to avoid re-processing"""
        processed_races = set()
        processed_dir = "./processed"

        if os.path.exists(processed_dir):
            for filename in os.listdir(processed_dir):
                if filename.endswith(".csv"):
                    # Extract race info from processed filename
                    match = re.match(
                        r"(\d{4}-\d{2}-\d{2})_([A-Z_]+)_Race_(\d+)_processed\.csv",
                        filename,
                    )
                    if match:
                        race_date, venue, race_number = match.groups()
                        try:
                            formatted_date = parse_date_flexible(race_date)
                            race_id = (formatted_date, venue, race_number)
                            processed_races.add(race_id)
                        except ValueError:
                            continue

        return processed_races

    def download_csv_from_race_page(self, race_info, max_retries=3):
        """Download CSV file from a race page with HTTP status code tracking"""
        last_http_status = None

        for attempt in range(max_retries):
            try:
                race_url = race_info["url"]
                print(f"🏁 Downloading CSV from race: {race_url}")

                response = None
                try:
                    response = self.session.get(race_url, timeout=30)
                    last_http_status = response.status_code
                    response.raise_for_status()

                    soup = BeautifulSoup(response.content, "html.parser")
                finally:
                    if response is not None:
                        try:
                            response.close()
                        except Exception:
                            pass

                # Method 1: Look for CSV download links using multiple selectors
                csv_selectors = [
                    'a[href*="csv"]',
                    'a[href*="form-guide"]',
                    'a[href*="export"]',
                    'a[href*="download"]',
                    'a[download*="csv"]',
                    ".csv-download",
                    ".form-guide-download",
                    ".export-csv",
                    ".download-csv",
                ]

                csv_url = None
                for selector in csv_selectors:
                    elements = soup.select(selector)
                    for element in elements:
                        href = element.get("href")
                        if href and (
                            "csv" in href.lower()
                            or "form" in href.lower()
                            or "export" in href.lower()
                        ):
                            csv_url = href
                            if self.verbose_fetch:
                                print(
                                    f"   🔍 Found CSV link via selector '{selector}': {href}"
                                )
                            break
                    if csv_url:
                        break

                # Method 2: Fallback to regex-based search
                if not csv_url:
                    csv_link = soup.find(
                        "a", href=re.compile(r".*\.csv.*", re.IGNORECASE)
                    )
                    if csv_link:
                        csv_url = csv_link.get("href")
                        if self.verbose_fetch:
                            print(f"   🔍 Found CSV link via regex: {csv_url}")

                # Method 3: Try direct CSV URL construction
                if not csv_url:
                    direct_csv_url = f"{race_url.rstrip('/')}/form-guide.csv"
                    if self.verbose_fetch:
                        print(f"   🔍 Trying direct CSV URL: {direct_csv_url}")

                    # Test if direct URL works
                    try:
                        test_response = None
                        try:
                            test_response = self.session.head(
                                direct_csv_url, timeout=10
                            )
                            if test_response.status_code == 200:
                                csv_url = direct_csv_url
                                if self.verbose_fetch:
                                    print(
                                        f"   ✅ Direct CSV URL works: {direct_csv_url}"
                                    )
                        finally:
                            if test_response is not None:
                                try:
                                    test_response.close()
                                except Exception:
                                    pass
                    except:
                        pass

                # Method 4: Try alternative direct URLs
                if not csv_url:
                    alternative_urls = [
                        f"{race_url.rstrip('/')}.csv",
                        f"{race_url.rstrip('/')}/export.csv",
                        f"{race_url.rstrip('/')}/download.csv",
                    ]

                    for alt_url in alternative_urls:
                        try:
                            test_response = None
                            try:
                                test_response = self.session.head(alt_url, timeout=5)
                                if test_response.status_code == 200:
                                    csv_url = alt_url
                                    if self.verbose_fetch:
                                        print(
                                            f"   ✅ Alternative CSV URL works: {alt_url}"
                                        )
                                    break
                            finally:
                                if test_response is not None:
                                    try:
                                        test_response.close()
                                    except Exception:
                                        pass
                        except:
                            continue

                if not csv_url:
                    if self.verbose_fetch:
                        print(f"   🔍 Available links on page:")
                        all_links = soup.find_all("a", href=True)
                        for i, link in enumerate(all_links[:10]):  # Show first 10 links
                            href = link.get("href")
                            text = link.get_text(strip=True)[:50]
                            print(f'     {i+1}. {href} - "{text}"')
                    print(f"❌ No CSV link found on race page: {race_url}")
                    return False, last_http_status

                # Make URL absolute
                if not csv_url.startswith("http"):
                    if csv_url.startswith("/"):
                        csv_url = self.base_url + csv_url
                    else:
                        csv_url = f"{self.base_url}/{csv_url}"

                # Download CSV content
                csv_response = None
                try:
                    csv_response = self.session.get(csv_url, timeout=30)
                    last_http_status = (
                        csv_response.status_code
                    )  # Update with CSV download status
                    csv_response.raise_for_status()
                finally:
                    # We will read csv_response.text after ensuring object exists; keep content before closing
                    csv_text = csv_response.text if csv_response is not None else ""
                    if csv_response is not None:
                        try:
                            csv_response.close()
                        except Exception:
                            pass

                # Use centralized date parsing for consistent formatting
                try:
                    formatted_date = parse_date_flexible(race_info["date"])
                except ValueError as e:
                    print(f"❌ Date parsing error for {race_info['date']}: {e}")
                    return False, last_http_status

                # Generate filename with consistent date format
                filename = f"Race {race_info['race_number']} - {race_info['venue']} - {formatted_date}.csv"
                filepath = os.path.join(self.unprocessed_dir, filename)

                # Save CSV file
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(csv_text)

                print(f"✅ Downloaded: {filename}")
                return True, last_http_status

            except requests.exceptions.RequestException as e:
                # Capture HTTP status from requests exceptions
                if hasattr(e, "response") and e.response is not None:
                    last_http_status = e.response.status_code
                print(f"❌ HTTP error downloading CSV (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff
                continue
            except Exception as e:
                print(f"❌ Error downloading CSV (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff
                continue

        return False, last_http_status

    def extract_race_info(self, race_element):
        """Extract race information from race element with improved date parsing"""
        try:
            # Extract race number
            race_number_elem = race_element.find("span", class_="race-number")
            if not race_number_elem:
                return None
            race_number = race_number_elem.text.strip().replace("R", "")

            # Extract race URL
            link_elem = race_element.find("a")
            if not link_elem:
                return None
            race_url = link_elem.get("href")
            if not race_url.startswith("http"):
                race_url = self.base_url + race_url

            # Extract venue from URL or race element
            venue = None
            for url_venue, code in self.venue_map.items():
                if url_venue in race_url:
                    venue = code
                    break

            if not venue:
                print(f"⚠️ Could not determine venue for URL: {race_url}")
                return None

            # Extract date from URL or page context
            date_match = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", race_url)
            if date_match:
                year, month, day = date_match.groups()
                date_str = f"{year}-{month}-{day}"
            else:
                # Try to extract from other sources
                date_str = None

            if not date_str:
                print(f"⚠️ Could not extract date from URL: {race_url}")
                return None

            # Use centralized date parsing
            try:
                formatted_date = parse_date_flexible(date_str)
            except ValueError as e:
                print(f"❌ Date parsing error for {date_str}: {e}")
                return None

            return {
                "race_number": race_number,
                "venue": venue,
                "date": formatted_date,
                "url": race_url,
            }

        except Exception as e:
            print(f"❌ Error extracting race info: {e}")
            return None

    def download_csv_file(self, race_info):
        """Download and save CSV file with improved date parsing and enhanced logging"""
        from datetime import datetime

        from logger import logger

        try:
            # Track race request at start
            self.stats.increment("races_requested")
            if self.verbose_fetch:
                print(
                    f"📊 Statistics: races_requested += 1 (now {self.stats.get('races_requested')})"
                )

            # Use centralized date parsing to ensure consistency
            try:
                formatted_date = parse_date_flexible(race_info["date"])
            except ValueError as e:
                self.stats.increment("fetches_failed")

                # Enhanced logging for date parsing errors
                logger.log_race_operation(
                    race_date=race_info.get("date", "unknown"),
                    venue=race_info.get("venue", "unknown"),
                    race_number=str(race_info.get("race_number", "0")),
                    operation="SKIP",
                    reason=f"Date parsing error: {e}",
                    verbose_fetch=self.verbose_fetch,
                    level="ERROR",
                )

                if self.verbose_fetch:
                    print(
                        f"📊 Statistics: fetches_failed += 1 (date parsing error, now {self.stats.get('fetches_failed')})"
                    )
                return False

            # Day-level optimization: Check if this date is mostly complete before processing individual race
            try:
                race_date_obj = datetime.strptime(formatted_date, "%Y-%m-%d").date()
                date_str = race_date_obj.strftime("%Y-%m-%d")

                # Skip if date is already marked as complete
                if date_str in self.completed_dates:
                    self.stats.increment("cache_hits")
                    if self.verbose_fetch:
                        print(
                            f"⚡ Skipping race on {date_str} - day marked as complete (day-level cache)"
                        )

                    logger.log_race_operation(
                        race_date=formatted_date,
                        venue=race_info["venue"],
                        race_number=str(race_info["race_number"]),
                        operation="CACHE",
                        reason="Day marked as complete",
                        verbose_fetch=self.verbose_fetch,
                    )
                    return True

                # Check if date appears mostly complete and mark it
                if self.is_date_mostly_complete(race_date_obj):
                    self.mark_date_as_complete(race_date_obj)
                    self.stats.increment("cache_hits")
                    if self.verbose_fetch:
                        print(
                            f"⚡ Marking {date_str} as complete and skipping remaining races"
                        )

                    logger.log_race_operation(
                        race_date=formatted_date,
                        venue=race_info["venue"],
                        race_number=str(race_info["race_number"]),
                        operation="CACHE",
                        reason="Day appears complete, marked for skipping",
                        verbose_fetch=self.verbose_fetch,
                    )
                    return True

            except (ValueError, AttributeError) as date_error:
                if self.verbose_fetch:
                    print(
                        f"⚠️ Could not parse date for day-level optimization: {date_error}"
                    )

            race_id = (formatted_date, race_info["venue"], race_info["race_number"])

            # Check if already collected (cache hit)
            if race_id in self.collected_races:
                self.stats.increment("cache_hits")

                # Enhanced logging for cache hits
                logger.log_race_operation(
                    race_date=formatted_date,
                    venue=race_info["venue"],
                    race_number=str(race_info["race_number"]),
                    operation="CACHE",
                    reason="Race already collected",
                    verbose_fetch=self.verbose_fetch,
                )

                if self.verbose_fetch:
                    print(
                        f"📊 Statistics: cache_hits += 1 (now {self.stats.get('cache_hits')}) - Race already collected: {race_id}"
                    )
                return True

            # Before HTTP call - track fetch attempt
            self.stats.increment("fetches_attempted")
            if self.verbose_fetch:
                print(
                    f"📊 Statistics: fetches_attempted += 1 (now {self.stats.get('fetches_attempted')}) - Attempting to fetch: {race_id}"
                )

            # Download the CSV with HTTP status tracking
            success, http_status = self.download_csv_from_race_page(race_info)

            if success:
                self.collected_races.add(race_id)
                self.stats.increment("successful_saves")

                # Enhanced logging for successful fetches
                logger.log_race_operation(
                    race_date=formatted_date,
                    venue=race_info["venue"],
                    race_number=str(race_info["race_number"]),
                    operation="FETCH",
                    reason="CSV downloaded successfully",
                    http_status=http_status,
                    verbose_fetch=self.verbose_fetch,
                )

                if self.verbose_fetch:
                    print(
                        f"📊 Statistics: successful_saves += 1 (now {self.stats.get('successful_saves')}) - Successfully saved: {race_id}"
                    )
                return True
            else:
                # On HTTP failure or missing link
                self.stats.increment("fetches_failed")

                # Enhanced logging for fetch failures
                logger.log_race_operation(
                    race_date=formatted_date,
                    venue=race_info["venue"],
                    race_number=str(race_info["race_number"]),
                    operation="FETCH",
                    reason="CSV download failed or no CSV link found",
                    http_status=http_status,
                    verbose_fetch=self.verbose_fetch,
                    level="WARNING",
                )

                if self.verbose_fetch:
                    print(
                        f"📊 Statistics: fetches_failed += 1 (download failed, now {self.stats.get('fetches_failed')}) - Failed to download: {race_id}"
                    )

            return False

        except Exception as e:
            self.stats.increment("fetches_failed")

            # Enhanced logging for exceptions
            logger.log_race_operation(
                race_date=race_info.get("date", "unknown"),
                venue=race_info.get("venue", "unknown"),
                race_number=str(race_info.get("race_number", "0")),
                operation="FETCH",
                reason=f"Exception during download: {str(e)}",
                verbose_fetch=self.verbose_fetch,
                level="ERROR",
            )

            if self.verbose_fetch:
                print(
                    f"📊 Statistics: fetches_failed += 1 (exception, now {self.stats.get('fetches_failed')})"
                )
            print(f"❌ Error downloading CSV file: {e}")
            return False

    def get_driver(self):
        """Set up Chrome driver with options"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument(
            "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )

        try:
            driver = webdriver.Chrome(options=chrome_options)
            return driver
        except Exception as e:
            print(f"❌ Error setting up Chrome driver: {e}")
            return None

    def test_single_race_download(self, race_url):
        """Test mode: Download CSV from a single race URL"""
        print(f"🧪 Test Mode: Downloading CSV from single race URL")
        print(f"📍 URL: {race_url}")

        try:
            # Extract race info from URL
            race_info = self.extract_race_info_from_url(race_url)
            if not race_info:
                print(f"❌ Could not extract race information from URL")
                return False

            print(f"📊 Race Info: {race_info}")

            # Download the CSV
            success = self.download_csv_file(race_info)
            if success:
                print(f"✅ Successfully downloaded race CSV!")
                return True
            else:
                print(f"❌ Failed to download race CSV")
                return False

        except Exception as e:
            print(f"❌ Error in test mode: {e}")
            return False

    def extract_race_info_from_url(self, race_url):
        """Extract race information from a race URL"""
        try:
            # Parse URL to extract venue, date, and race number
            # Handle both formats:
            # - https://www.thedogs.com.au/racing/venue/YYYY/MM/DD/race_number
            # - https://www.thedogs.com.au/racing/venue/YYYY-MM-DD/race_number/...

            # Remove query parameters
            base_url = race_url.split("?")[0]

            # Try format 1: /racing/venue/YYYY-MM-DD/race_number/...
            url_match = re.match(
                r"https://www\.thedogs\.com\.au/racing/([^/]+)/(\d{4})-(\d{2})-(\d{2})/(\d+)",
                base_url,
            )
            if url_match:
                venue_slug, year, month, day, race_number = url_match.groups()
            else:
                # Try format 2: /racing/venue/YYYY/MM/DD/race_number
                url_match = re.match(
                    r"https://www\.thedogs\.com\.au/racing/([^/]+)/(\d{4})/(\d{2})/(\d{2})/(\d+)",
                    base_url,
                )
                if url_match:
                    venue_slug, year, month, day, race_number = url_match.groups()
                else:
                    print(f"⚠️ URL format not recognized: {race_url}")
                    return None

            # Map venue slug to venue code
            venue = None
            for url_venue, code in self.venue_map.items():
                if url_venue == venue_slug:
                    venue = code
                    break

            if not venue:
                print(f"⚠️ Unknown venue: {venue_slug}")
                venue = venue_slug.upper().replace("-", "_")

            # Format date
            date_str = f"{year}-{month}-{day}"

            return {
                "race_number": race_number,
                "venue": venue,
                "date": date_str,
                "url": race_url,
            }

        except Exception as e:
            print(f"❌ Error extracting race info from URL: {e}")
            return None

    def get_processed_filenames(self, directory: str) -> set:
        """Get set of processed filenames from specified directory for O(1) membership tests - delegate to manager"""
        return self.race_file_manager.get_processed_filenames(directory)

    def is_historical(self, date_obj):
        """Check if a race date is historical (previous day or earlier).

        Args:
            date_obj: datetime.date object representing the race date

        Returns:
            bool: True if the date is strictly before today (historical)
        """
        from datetime import date

        today = date.today()
        return date_obj < today

    def is_date_mostly_complete(self, target_date, completion_threshold=0.8):
        """Check if a date has most races already collected to enable day-level skipping.

        Args:
            target_date: datetime.date object for the target date
            completion_threshold: Fraction of races that must be collected (default: 0.8 = 80%)

        Returns:
            bool: True if the date is mostly complete and should be skipped
        """
        try:
            # Convert date to string format for race_id matching
            date_str = target_date.strftime("%Y-%m-%d")

            # Count races already collected for this date
            collected_for_date = 0
            total_races_found = 0

            # Count collected races for this date
            for race_id in self.collected_races:
                race_date, venue, race_number = race_id
                if race_date == date_str:
                    collected_for_date += 1

            # If we have no collected races for this date, it's not complete
            if collected_for_date == 0:
                return False

            # For day-level optimization, we need to estimate total races for the date
            # Use a reasonable heuristic: if we have collected a good number of races
            # (e.g., 8+) for a date, it's likely mostly complete
            if collected_for_date >= 8:
                # Consider it mostly complete if we have 8+ races (typical day has 10-20 races)
                return True

            # For smaller numbers, we could attempt to discover actual race count,
            # but that would require an HTTP request which defeats the optimization purpose.
            # Instead, use a conservative approach: only skip if we have many races
            return False

        except Exception as e:
            if self.verbose_fetch:
                print(f"⚠️ Error checking date completion for {target_date}: {e}")
            return False

    def mark_date_as_complete(self, target_date):
        """Mark a date as fully processed to enable day-level skipping.

        Args:
            target_date: datetime.date object to mark as complete
        """
        date_str = target_date.strftime("%Y-%m-%d")
        self.completed_dates.add(date_str)
        if self.verbose_fetch:
            print(f"📅 Marked {date_str} as complete for future day-level skipping")

    def discover_races_for_date(self, target_date):
        """Discover all race URLs for a specific date.

        Args:
            target_date: datetime.date object for the target date

        Returns:
            list: List of race URLs for the specified date
        """
        try:
            # Format date for URL
            date_str = target_date.strftime("%Y-%m-%d")

            # Get the main racing page for the date
            racing_url = f"{self.base_url}/racing/{date_str}"

            if self.verbose_fetch:
                print(f"📅 Discovering races for {date_str} from {racing_url}")

            response = None
            try:
                response = self.session.get(racing_url, timeout=30)
                if response.status_code != 200:
                    if self.verbose_fetch:
                        print(
                            f"⚠️ Failed to get racing page for {date_str}: HTTP {response.status_code}"
                        )
                    return []

                soup = BeautifulSoup(response.content, "html.parser")
            finally:
                if response is not None:
                    try:
                        response.close()
                    except Exception:
                        pass

            # Find race links - they typically follow patterns like:
            # /racing/venue/YYYY-MM-DD/race_number/race-name
            race_urls = []

            # Look for links that match the race URL pattern
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if (
                    "/racing/" in href and date_str in href and href.count("/") >= 5
                ):  # At least /racing/venue/date/race/name

                    # Build full URL if it's relative
                    if href.startswith("/"):
                        full_url = self.base_url + href
                    else:
                        full_url = href

                    # Remove query parameters and add to list if not already present
                    clean_url = full_url.split("?")[0]
                    if clean_url not in race_urls:
                        race_urls.append(clean_url)

            if self.verbose_fetch:
                print(f"📋 Found {len(race_urls)} race URLs for {date_str}")
                if race_urls and len(race_urls) <= 5:
                    print(f"   Sample URLs: {race_urls}")
                elif race_urls:
                    print(f"   Sample URLs: {race_urls[:3]}")

            return race_urls

        except Exception as e:
            print(f"❌ Error discovering races for {target_date}: {e}")
            return []

    def run_historical_batch_scraping(self, days_back=7):
        """Run batch scraping for historical races.

        Args:
            days_back: Number of days back from today to scrape (default: 7)
        """
        from datetime import date, timedelta

        print(f"🎯 Starting historical batch scraping for {days_back} days back...")

        end_date = date.today() - timedelta(days=1)  # Start from yesterday
        start_date = end_date - timedelta(days=days_back - 1)

        print(f"📅 Date range: {start_date} to {end_date}")

        total_races_processed = 0

        # Process each date
        current_date = start_date
        while current_date <= end_date:
            if self.verbose_fetch:
                print(f"\n🔍 Processing date: {current_date}")

            # Day-level optimization: Skip dates that are mostly complete
            date_str = current_date.strftime("%Y-%m-%d")
            if date_str in self.completed_dates:
                if self.verbose_fetch:
                    print(
                        f"   ⚡ Skipping {current_date} - marked as complete (day-level cache)"
                    )
                current_date += timedelta(days=1)
                continue

            # Check if date is mostly complete based on existing collected races
            if self.is_date_mostly_complete(current_date):
                if self.verbose_fetch:
                    print(
                        f"   ⚡ Skipping {current_date} - appears mostly complete ({len([r for r in self.collected_races if r[0] == date_str])} races already collected)"
                    )
                self.mark_date_as_complete(current_date)
                current_date += timedelta(days=1)
                continue

            # Use UpcomingRaceBrowser for race discovery (it has working race discovery logic)
            try:
                from upcoming_race_browser import UpcomingRaceBrowser

                browser = UpcomingRaceBrowser()
                race_data_list = browser.get_races_for_date(current_date)

                if not race_data_list:
                    if self.verbose_fetch:
                        print(f"   ⚠️ No races found for {current_date}")
                    current_date += timedelta(days=1)
                    continue

                if self.verbose_fetch:
                    print(f"   📋 Found {len(race_data_list)} races for {current_date}")

                # Process each race from the browser results
                for race_data in race_data_list:
                    try:
                        # Convert browser race data to scraper format
                        race_info = {
                            "race_number": str(race_data.get("race_number", "1")),
                            "venue": race_data.get("venue", "UNKNOWN"),
                            "date": current_date.strftime("%Y-%m-%d"),
                            "url": race_data.get("url", ""),
                        }

                        # Verify it's actually historical
                        if self.is_historical(current_date):
                            self.stats.increment("races_requested")
                            if self.verbose_fetch:
                                print(
                                    f"   🏁 Processing: {race_info['venue']} Race {race_info['race_number']} ({race_info['date']})"
                                )

                            success = self.download_csv_file(race_info)
                            if success:
                                total_races_processed += 1
                                if self.verbose_fetch:
                                    print(
                                        f"   ✅ Successfully downloaded: {race_info['venue']} Race {race_info['race_number']}"
                                    )
                            else:
                                if self.verbose_fetch:
                                    print(
                                        f"   ❌ Failed to download: {race_info['venue']} Race {race_info['race_number']}"
                                    )
                        else:
                            if self.verbose_fetch:
                                print(
                                    f"   ⚪ Skipping non-historical race: {race_info['venue']} Race {race_info['race_number']} on {race_info['date']}"
                                )

                    except Exception as race_error:
                        if self.verbose_fetch:
                            print(f"   ⚠️ Error processing race: {race_error}")
                        continue

                    # Small delay to be polite to the server
                    time.sleep(0.5)

            except Exception as browser_error:
                print(
                    f"   ❌ Error using UpcomingRaceBrowser for {current_date}: {browser_error}"
                )
                # Fallback to old method if browser fails
                race_urls = self.discover_races_for_date(current_date)

                if not race_urls:
                    if self.verbose_fetch:
                        print(
                            f"   ⚠️ No races found for {current_date} (fallback method)"
                        )
                    current_date += timedelta(days=1)
                    continue

                # Process each race URL using old method
                for race_url in race_urls:
                    race_info = self.extract_race_info_from_url(race_url)
                    if race_info:
                        # Verify it's actually historical
                        try:
                            race_date = datetime.strptime(
                                race_info["date"], "%Y-%m-%d"
                            ).date()
                            if self.is_historical(race_date):
                                self.stats.increment("races_requested")
                                success = self.download_csv_file(race_info)
                                if success:
                                    total_races_processed += 1
                            else:
                                if self.verbose_fetch:
                                    print(
                                        f"   ⚪ Skipping non-historical race: {race_info['venue']} Race {race_info['race_number']} on {race_info['date']}"
                                    )
                        except ValueError as e:
                            if self.verbose_fetch:
                                print(f"   ⚠️ Date parsing error for {race_url}: {e}")

                    # Small delay to be polite to the server
                    time.sleep(0.5)

            current_date += timedelta(days=1)

        print(f"\n✅ Historical batch scraping completed!")
        print(f"📊 Total races processed: {total_races_processed}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Form Guide CSV Scraper with robust caching"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of all files, ignoring cache",
    )
    parser.add_argument(
        "--test-file", type=str, help="Test the caching system with a specific CSV file"
    )
    parser.add_argument(
        "--stats", action="store_true", help="Show cache statistics and exit"
    )
    parser.add_argument(
        "--test-url", type=str, help="Test mode: Download CSV from a single race URL"
    )
    parser.add_argument(
        "--historical",
        action="store_true",
        help="Download races from previous day or earlier",
    )
    parser.add_argument(
        "--verbose-fetch", action="store_true", help="Enable per-race fetch logging"
    )

    args = parser.parse_args()

    scraper = FormGuideCsvScraper(
        historical=args.historical, verbose_fetch=args.verbose_fetch
    )
    print(f"📊 Loaded {len(scraper.collected_races)} existing races")
    print(f"📁 Found {len(scraper.existing_files)} existing files")

    if args.stats:
        # Show detailed cache statistics
        conn = sqlite3.connect(scraper.database_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM processed_race_files")
        total_processed = cursor.fetchone()[0]

        cursor.execute(
            "SELECT status, COUNT(*) FROM processed_race_files GROUP BY status"
        )
        status_counts = cursor.fetchall()

        cursor.execute(
            "SELECT COUNT(DISTINCT venue) FROM processed_race_files WHERE venue != 'unknown'"
        )
        unique_venues = cursor.fetchone()[0]

        cursor.execute(
            "SELECT MIN(processed_at), MAX(processed_at) FROM processed_race_files"
        )
        date_range = cursor.fetchone()

        print(f"\n📈 Cache Statistics:")
        print(f"   Total processed files: {total_processed}")
        print(f"   Unique venues: {unique_venues}")
        for status, count in status_counts:
            print(f"   {status.capitalize()}: {count}")
        if date_range[0] and date_range[1]:
            print(f"   Date range: {date_range[0]} to {date_range[1]}")

        conn.close()

        # Log summary and failure guard before exit
        scraper.stats.log_summary()
        if (
            scraper.stats.get("fetches_attempted") > 0
            and scraper.stats.get("successful_saves") == 0
        ):
            raise RuntimeError("All downloads failed")
        sys.exit(0)

    if args.test_file:
        if not os.path.exists(args.test_file):
            print(f"❌ Test file not found: {args.test_file}")
            sys.exit(1)

        print(f"\n🧪 Testing caching with file: {args.test_file}")

        # Test without force
        result1 = scraper.parse_csv_with_ingestion(args.test_file, force=False)
        print(f"First attempt (no force): {result1}")

        # Test with force
        result2 = scraper.parse_csv_with_ingestion(args.test_file, force=args.force)
        print(f"Second attempt (force={args.force}): {result2}")

        # Show statistics from the test
        scraper.stats.log_summary()

        # Check for failure guard condition before exit
        if (
            scraper.stats.get("fetches_attempted") > 0
            and scraper.stats.get("successful_saves") == 0
        ):
            raise RuntimeError("All downloads failed")
        sys.exit(0)

    if args.test_url:
        print(f"\n🧪 Test Mode: Downloading single race CSV")
        success = scraper.test_single_race_download(args.test_url)
        if success:
            print(f"\n🎉 Test completed successfully!")
        else:
            print(f"\n💥 Test failed!")
        # Log summary before exit
        scraper.stats.log_summary()
        # Check for failure guard condition
        if (
            scraper.stats.get("fetches_attempted") > 0
            and scraper.stats.get("successful_saves") == 0
        ):
            raise RuntimeError("All downloads failed")
        sys.exit(0)

    # Run historical batch scraping if --historical flag is provided
    if args.historical:
        print(f"\n🎯 Historical mode activated - starting batch scraping...")
        scraper.run_historical_batch_scraping(days_back=7)  # Default to 7 days back
    else:
        print(f"\n✨ Caching system ready! Use --help for options.")
        print(f"💡 Tips:")
        print(f"   - Use --test-file <path> to test caching with a specific file")
        print(f"   - Use --force to ignore cache and reprocess files")
        print(f"   - Use --stats to see cache statistics")
        print(f"   - Use --historical to run batch scraping of recent races")

    # Final summary and failure guard for main execution path
    scraper.stats.log_summary()

    # If any fetches were attempted but no saves succeeded, raise error for CI/batch jobs
    if (
        scraper.stats.get("fetches_attempted") > 0
        and scraper.stats.get("successful_saves") == 0
    ):
        raise RuntimeError("All downloads failed")

    # Show completion message
    if args.historical and scraper.stats.get("races_requested") > 0:
        print(f"🏁 Historical scraping completed successfully!")

    # Allow zero eligible races case to exit normally (no error raised)
