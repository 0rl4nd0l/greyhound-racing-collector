#!/usr/bin/env python3
"""
Enhanced Comprehensive Greyhound Race Data Collector
===================================================

This system collects comprehensive racing data including:
- Complete odds data for all dogs
- Detailed race conditions and metadata
- Individual dog performance metrics
- Trainer and form information
- Track records and timing data
- Weather and track conditions
- In-running positions and sectionals
- Prize money and race classifications

Designed for maximum analysis capability and prediction accuracy.
"""

import json
import re
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


class EnhancedGreyhoundDataCollector:
    """Comprehensive greyhound racing data collector with complete information extraction"""

    def __init__(self, db_path="comprehensive_greyhound_data.db"):
        self.db_path = db_path
        self.setup_database()
        self.setup_driver()

    def setup_database(self):
        """Initialize comprehensive SQLite database for all racing data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Race metadata table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS race_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT UNIQUE,
                venue TEXT,
                race_number INTEGER,
                race_date DATE,
                race_name TEXT,
                grade TEXT,
                distance TEXT,
                track_condition TEXT,
                weather TEXT,
                track_record TEXT,
                prize_money_total REAL,
                prize_money_breakdown TEXT,
                race_time TEXT,
                field_size INTEGER,
                url TEXT,
                extraction_timestamp DATETIME,
                UNIQUE(race_id)
            )
        """
        )

        # Comprehensive dog data table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS dog_race_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT,
                dog_name TEXT,
                dog_clean_name TEXT,
                dog_id INTEGER,
                box_number INTEGER,
                finish_position TEXT,
                trainer_name TEXT,
                trainer_id INTEGER,
                weight REAL,
                running_style TEXT,
                odds_decimal REAL,
                odds_fractional TEXT,
                individual_time TEXT,
                sectional_1st TEXT,
                sectional_2nd TEXT,
                sectional_3rd TEXT,
                margin TEXT,
                beaten_margin REAL,
                was_scratched BOOLEAN DEFAULT FALSE,
                blackbook_link TEXT,
                extraction_timestamp DATETIME,
                FOREIGN KEY (race_id) REFERENCES race_metadata (race_id)
            )
        """
        )

        # In-running positions table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS in_running_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT,
                dog_clean_name TEXT,
                box_number INTEGER,
                section_1_position INTEGER,
                section_2_position INTEGER,
                section_3_position INTEGER,
                section_4_position INTEGER,
                final_position INTEGER,
                extraction_timestamp DATETIME,
                FOREIGN KEY (race_id) REFERENCES race_metadata (race_id)
            )
        """
        )

        # Race sectional times table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS race_sectionals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT,
                section_name TEXT,
                cumulative_time REAL,
                sectional_time REAL,
                section_order INTEGER,
                extraction_timestamp DATETIME,
                FOREIGN KEY (race_id) REFERENCES race_metadata (race_id)
            )
        """
        )

        # Historical odds snapshots (for multiple collections over time)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS odds_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT,
                dog_clean_name TEXT,
                odds_decimal REAL,
                odds_fractional TEXT,
                source TEXT DEFAULT 'thedogs',
                snapshot_timestamp DATETIME,
                is_current BOOLEAN DEFAULT TRUE,
                FOREIGN KEY (race_id) REFERENCES race_metadata (race_id)
            )
        """
        )

        conn.commit()
        conn.close()

    def setup_driver(self):
        """Setup Chrome driver for web scraping"""
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
            print("✅ Chrome driver setup successful")
        except Exception as e:
            print(f"⚠️  Chrome driver setup failed: {e}")
            self.driver = None

    def extract_race_metadata(self, driver, race_url):
        """Extract comprehensive race metadata"""
        metadata = {"url": race_url, "extraction_timestamp": datetime.now()}

        # Extract race ID from URL
        metadata["race_id"] = self.extract_race_id_from_url(race_url)

        # Race name
        try:
            race_name_elem = driver.find_element(
                By.CSS_SELECTOR, ".race-header__info__name"
            )
            metadata["race_name"] = race_name_elem.text.strip()
        except:
            metadata["race_name"] = None

        # Grade and distance (from "5th Grade 530m")
        try:
            grade_elem = driver.find_element(
                By.CSS_SELECTOR, ".race-header__info__grade"
            )
            grade_text = grade_elem.text.strip()

            # Parse grade and distance
            grade_match = re.search(r"(\d+)(?:st|nd|rd|th)?\s*Grade", grade_text)
            distance_match = re.search(r"(\d+m)", grade_text)

            metadata["grade"] = grade_match.group(1) if grade_match else None
            metadata["distance"] = distance_match.group(1) if distance_match else None
        except:
            metadata["grade"] = None
            metadata["distance"] = None

        # Track record
        try:
            track_record_elem = driver.find_element(
                By.CSS_SELECTOR, ".race-header__record th:last-child"
            )
            metadata["track_record"] = track_record_elem.text.strip()
        except:
            metadata["track_record"] = None

        # Prize money
        try:
            prize_total_elem = driver.find_element(
                By.CSS_SELECTOR, ".race-header__prize__total"
            )
            metadata["prize_money_total"] = float(
                prize_total_elem.text.strip().replace("$", "").replace(",", "")
            )

            prize_breakdown_elem = driver.find_element(
                By.CSS_SELECTOR, ".race-header__prize__places"
            )
            metadata["prize_money_breakdown"] = prize_breakdown_elem.text.strip()
        except:
            metadata["prize_money_total"] = None
            metadata["prize_money_breakdown"] = None

        # Weather (from weather icon)
        try:
            weather_elem = driver.find_element(By.CSS_SELECTOR, '[name*="weather"]')
            weather_name = weather_elem.get_attribute("name")
            metadata["weather"] = (
                weather_name.split("weather_")[-1]
                if "weather_" in weather_name
                else None
            )
        except:
            metadata["weather"] = None

        # Race time (winning time)
        try:
            race_time_elem = driver.find_element(
                By.CSS_SELECTOR, ".race__times__time:last-child"
            )
            metadata["race_time"] = (
                race_time_elem.text.strip().replace("(", "").replace(")", "")
            )
        except:
            metadata["race_time"] = None

        # Extract venue and date from URL or page
        url_parts = race_url.split("/")
        try:
            for i, part in enumerate(url_parts):
                if re.match(r"\d{4}-\d{2}-\d{2}", part):
                    metadata["race_date"] = part
                    if i > 0:
                        metadata["venue"] = url_parts[i - 1]
                    if i < len(url_parts) - 1:
                        try:
                            metadata["race_number"] = int(url_parts[i + 1])
                        except:
                            pass
                    break
        except:
            pass

        return metadata

    def extract_dog_data(self, driver):
        """Extract comprehensive data for each dog"""
        dogs_data = []

        try:
            # Find all runner rows
            runners = driver.find_elements(By.CSS_SELECTOR, ".race-runner")

            for runner in runners:
                dog_data = {}

                # Finish position
                try:
                    position_elem = runner.find_element(
                        By.CSS_SELECTOR, ".race-runners__finish-position"
                    )
                    dog_data["finish_position"] = position_elem.text.strip()
                    dog_data["was_scratched"] = "SCR" in dog_data["finish_position"]
                except:
                    dog_data["finish_position"] = None
                    dog_data["was_scratched"] = False

                # Box number (from rug color)
                try:
                    box_elem = runner.find_element(
                        By.CSS_SELECTOR, '.race-runners__box [name^="rug_"]'
                    )
                    rug_name = box_elem.get_attribute("name")
                    dog_data["box_number"] = (
                        int(rug_name.split("_")[-1]) if "rug_" in rug_name else None
                    )
                except:
                    dog_data["box_number"] = None

                # Dog name and ID
                try:
                    dog_link = runner.find_element(
                        By.CSS_SELECTOR, ".race-runners__name__dog a"
                    )
                    dog_data["dog_name"] = dog_link.text.strip()
                    dog_data["dog_clean_name"] = self.clean_dog_name(
                        dog_data["dog_name"]
                    )

                    # Extract dog ID from URL
                    dog_url = dog_link.get_attribute("href")
                    dog_id_match = re.search(r"/dogs/(\d+)/", dog_url)
                    dog_data["dog_id"] = (
                        int(dog_id_match.group(1)) if dog_id_match else None
                    )
                except:
                    dog_data["dog_name"] = None
                    dog_data["dog_clean_name"] = None
                    dog_data["dog_id"] = None

                # Individual time
                try:
                    time_elem = runner.find_element(
                        By.CSS_SELECTOR, ".race-runners__name__time"
                    )
                    dog_data["individual_time"] = time_elem.text.strip()
                except:
                    dog_data["individual_time"] = None

                # Running style (R/T)
                try:
                    style_elem = runner.find_element(
                        By.CSS_SELECTOR, ".race-runners__track-sa-trait"
                    )
                    dog_data["running_style"] = style_elem.text.strip()
                except:
                    dog_data["running_style"] = None

                # Trainer information
                try:
                    trainer_link = runner.find_element(
                        By.CSS_SELECTOR, ".race-runners__trainer a"
                    )
                    dog_data["trainer_name"] = trainer_link.text.strip()

                    # Extract trainer ID
                    trainer_url = trainer_link.get_attribute("href")
                    trainer_id_match = re.search(r"/trainers/(\d+)/", trainer_url)
                    dog_data["trainer_id"] = (
                        int(trainer_id_match.group(1)) if trainer_id_match else None
                    )
                except:
                    dog_data["trainer_name"] = None
                    dog_data["trainer_id"] = None

                # Weight
                try:
                    weight_elem = runner.find_element(
                        By.CSS_SELECTOR, ".race-runners__weight"
                    )
                    weight_text = weight_elem.text.strip()
                    if weight_text:
                        dog_data["weight"] = float(weight_text)
                except:
                    dog_data["weight"] = None

                # Sectional times
                try:
                    sectional_elems = runner.find_elements(
                        By.CSS_SELECTOR, ".race-runners__sectional"
                    )
                    if len(sectional_elems) >= 2:
                        dog_data["sectional_1st"] = sectional_elems[0].text.strip()
                        dog_data["sectional_2nd"] = sectional_elems[1].text.strip()
                    if len(sectional_elems) >= 3:
                        dog_data["sectional_3rd"] = sectional_elems[2].text.strip()
                except:
                    dog_data["sectional_1st"] = None
                    dog_data["sectional_2nd"] = None
                    dog_data["sectional_3rd"] = None

                # Final time
                try:
                    time_elem = runner.find_element(
                        By.CSS_SELECTOR, ".race-runners__time"
                    )
                    dog_data["final_time"] = time_elem.text.strip()
                except:
                    dog_data["final_time"] = None

                # Margin
                try:
                    margin_elem = runner.find_element(
                        By.CSS_SELECTOR, ".race-runners__margin"
                    )
                    margin_text = margin_elem.text.strip()
                    dog_data["margin"] = margin_text

                    # Convert margin to numeric if possible
                    if margin_text and margin_text != "—":
                        try:
                            dog_data["beaten_margin"] = float(margin_text)
                        except:
                            dog_data["beaten_margin"] = None
                    else:
                        dog_data["beaten_margin"] = 0.0  # Winner
                except:
                    dog_data["margin"] = None
                    dog_data["beaten_margin"] = None

                # Starting price (odds)
                try:
                    odds_elem = runner.find_element(
                        By.CSS_SELECTOR, ".race-runners__starting-price"
                    )
                    odds_text = odds_elem.text.strip()
                    dog_data["odds_fractional"] = odds_text
                    dog_data["odds_decimal"] = self.parse_odds_to_decimal(odds_text)
                except:
                    dog_data["odds_fractional"] = None
                    dog_data["odds_decimal"] = None

                # Blackbook link
                try:
                    blackbook_elem = runner.find_element(
                        By.CSS_SELECTOR, "blackbook-dog"
                    )
                    dog_data["blackbook_link"] = blackbook_elem.get_attribute(
                        "data-dog-id"
                    )
                except:
                    dog_data["blackbook_link"] = None

                dog_data["extraction_timestamp"] = datetime.now()

                if dog_data.get("dog_name"):  # Only add if we have a dog name
                    dogs_data.append(dog_data)

        except Exception as e:
            print(f"⚠️  Error extracting dog data: {e}")

        return dogs_data

    def extract_in_running_positions(self, driver):
        """Extract in-running positions throughout the race"""
        positions_data = []

        try:
            # Find in-running table
            in_running_table = driver.find_element(By.CSS_SELECTOR, ".race__in-running")
            rows = in_running_table.find_elements(By.CSS_SELECTOR, "tr")[
                1:
            ]  # Skip header

            section_names = ["1st Section", "2nd Section", "3rd Section", "4th Section"]

            for i, row in enumerate(rows):
                if i < len(section_names):
                    section_positions = []
                    position_cells = row.find_elements(
                        By.CSS_SELECTOR, ".race__in-running__box"
                    )[
                        1:
                    ]  # Skip title

                    for pos, cell in enumerate(position_cells):
                        try:
                            rug_elem = cell.find_element(
                                By.CSS_SELECTOR, '[name^="rug_"]'
                            )
                            rug_name = rug_elem.get_attribute("name")
                            box_number = (
                                int(rug_name.split("_")[-1])
                                if "rug_" in rug_name
                                else None
                            )

                            positions_data.append(
                                {
                                    "section_name": section_names[i],
                                    "section_order": i + 1,
                                    "position": pos + 1,
                                    "box_number": box_number,
                                    "extraction_timestamp": datetime.now(),
                                }
                            )
                        except:
                            continue

        except Exception as e:
            print(f"⚠️  Error extracting in-running positions: {e}")

        return positions_data

    def extract_race_sectionals(self, driver):
        """Extract race sectional times"""
        sectionals_data = []

        try:
            # Find race times table
            times_table = driver.find_element(By.CSS_SELECTOR, ".race__times")
            rows = times_table.find_elements(By.CSS_SELECTOR, "tr")

            for row in rows:
                cells = row.find_elements(By.CSS_SELECTOR, "td")
                if len(cells) > 1:
                    row_type = cells[0].text.strip()

                    if "Race Time" in row_type:
                        # Cumulative times
                        for i, cell in enumerate(cells[1:]):
                            time_text = (
                                cell.text.strip().replace("(", "").replace(")", "")
                            )
                            if time_text:
                                try:
                                    sectionals_data.append(
                                        {
                                            "section_name": f"Cumulative_{i+1}",
                                            "cumulative_time": float(time_text),
                                            "sectional_time": None,
                                            "section_order": i + 1,
                                            "extraction_timestamp": datetime.now(),
                                        }
                                    )
                                except:
                                    pass

                    elif "Sectional Time" in row_type:
                        # Individual section times
                        for i, cell in enumerate(cells[1:]):
                            time_text = cell.text.strip()
                            if time_text:
                                try:
                                    sectionals_data.append(
                                        {
                                            "section_name": f"Section_{i+1}",
                                            "cumulative_time": None,
                                            "sectional_time": float(time_text),
                                            "section_order": i + 1,
                                            "extraction_timestamp": datetime.now(),
                                        }
                                    )
                                except:
                                    pass

        except Exception as e:
            print(f"⚠️  Error extracting race sectionals: {e}")

        return sectionals_data

    def collect_comprehensive_race_data(self, race_url):
        """Collect all comprehensive race data from a single race URL"""
        if not self.driver:
            return None

        try:
            print(f"🏁 Collecting comprehensive data from: {race_url}")
            self.driver.get(race_url)
            time.sleep(5)  # Wait for page load

            # Extract all data components
            race_metadata = self.extract_race_metadata(self.driver, race_url)
            dogs_data = self.extract_dog_data(self.driver)
            in_running_data = self.extract_in_running_positions(self.driver)
            sectionals_data = self.extract_race_sectionals(self.driver)

            # Count field size
            race_metadata["field_size"] = len(
                [d for d in dogs_data if not d.get("was_scratched", False)]
            )

            # Store all data
            self.store_comprehensive_data(
                race_metadata, dogs_data, in_running_data, sectionals_data
            )

            print(
                f"✅ Collected data for {len(dogs_data)} dogs in race {race_metadata.get('race_id')}"
            )
            return {
                "race_metadata": race_metadata,
                "dogs_data": dogs_data,
                "in_running_data": in_running_data,
                "sectionals_data": sectionals_data,
            }

        except Exception as e:
            print(f"❌ Error collecting race data from {race_url}: {e}")
            return None

    def store_comprehensive_data(
        self, race_metadata, dogs_data, in_running_data, sectionals_data
    ):
        """Store all collected data in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Store race metadata (use INSERT OR IGNORE to prevent overwriting)
            cursor.execute(
                """
                INSERT OR IGNORE INTO race_metadata 
                (race_id, venue, race_number, race_date, race_name, grade, distance, 
                 track_condition, weather, track_record, prize_money_total, 
                 prize_money_breakdown, race_time, field_size, url, extraction_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    race_metadata.get("race_id"),
                    race_metadata.get("venue"),
                    race_metadata.get("race_number"),
                    race_metadata.get("race_date"),
                    race_metadata.get("race_name"),
                    race_metadata.get("grade"),
                    race_metadata.get("distance"),
                    race_metadata.get("track_condition"),
                    race_metadata.get("weather"),
                    race_metadata.get("track_record"),
                    race_metadata.get("prize_money_total"),
                    race_metadata.get("prize_money_breakdown"),
                    race_metadata.get("race_time"),
                    race_metadata.get("field_size"),
                    race_metadata.get("url"),
                    race_metadata.get("extraction_timestamp"),
                ),
            )

            # Store dog data (use INSERT OR IGNORE to prevent overwriting)
            for dog in dogs_data:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO dog_race_data 
                    (race_id, dog_name, dog_clean_name, dog_id, box_number, finish_position,
                     trainer_name, trainer_id, weight, running_style, odds_decimal, odds_fractional,
                     individual_time, sectional_1st, sectional_2nd, sectional_3rd, margin,
                     beaten_margin, was_scratched, blackbook_link, extraction_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        race_metadata.get("race_id"),
                        dog.get("dog_name"),
                        dog.get("dog_clean_name"),
                        dog.get("dog_id"),
                        dog.get("box_number"),
                        dog.get("finish_position"),
                        dog.get("trainer_name"),
                        dog.get("trainer_id"),
                        dog.get("weight"),
                        dog.get("running_style"),
                        dog.get("odds_decimal"),
                        dog.get("odds_fractional"),
                        dog.get("individual_time"),
                        dog.get("sectional_1st"),
                        dog.get("sectional_2nd"),
                        dog.get("sectional_3rd"),
                        dog.get("margin"),
                        dog.get("beaten_margin"),
                        dog.get("was_scratched"),
                        dog.get("blackbook_link"),
                        dog.get("extraction_timestamp"),
                    ),
                )

                # Store odds snapshot
                if dog.get("odds_decimal"):
                    cursor.execute(
                        """
                        INSERT INTO odds_snapshots 
                        (race_id, dog_clean_name, odds_decimal, odds_fractional, snapshot_timestamp)
                        VALUES (?, ?, ?, ?, ?)
                    """,
                        (
                            race_metadata.get("race_id"),
                            dog.get("dog_clean_name"),
                            dog.get("odds_decimal"),
                            dog.get("odds_fractional"),
                            dog.get("extraction_timestamp"),
                        ),
                    )

            # Store in-running positions
            for position in in_running_data:
                cursor.execute(
                    """
                    INSERT INTO in_running_positions 
                    (race_id, box_number, section_1_position, section_2_position, 
                     section_3_position, section_4_position, extraction_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        race_metadata.get("race_id"),
                        position.get("box_number"),
                        (
                            position.get("position")
                            if position.get("section_order") == 1
                            else None
                        ),
                        (
                            position.get("position")
                            if position.get("section_order") == 2
                            else None
                        ),
                        (
                            position.get("position")
                            if position.get("section_order") == 3
                            else None
                        ),
                        (
                            position.get("position")
                            if position.get("section_order") == 4
                            else None
                        ),
                        position.get("extraction_timestamp"),
                    ),
                )

            # Store race sectionals
            for sectional in sectionals_data:
                cursor.execute(
                    """
                    INSERT INTO race_sectionals 
                    (race_id, section_name, cumulative_time, sectional_time, 
                     section_order, extraction_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        race_metadata.get("race_id"),
                        sectional.get("section_name"),
                        sectional.get("cumulative_time"),
                        sectional.get("sectional_time"),
                        sectional.get("section_order"),
                        sectional.get("extraction_timestamp"),
                    ),
                )

            conn.commit()

        except Exception as e:
            print(f"⚠️  Error storing data: {e}")
            conn.rollback()

        finally:
            conn.close()

    def collect_from_race_results(self):
        """Collect comprehensive data from existing race result URLs"""
        print("🚀 Starting comprehensive data collection from race results...")

        # Load existing race results
        results_file = "./form_guides/navigator_race_results.csv"
        if not Path(results_file).exists():
            print("❌ No race results file found")
            return

        results = pd.read_csv(results_file)

        collected_races = 0
        total_dogs = 0

        # Start from race 28 to avoid duplicates (we already have 27 races)
        start_index = 27
        print(f"Starting from race {start_index + 1} to avoid duplicates...")

        for index, race in results.iterrows():
            if index < start_index:
                continue  # Skip already processed races

            race_url = race.get("source_url")
            if race_url and "thedogs.com.au" in race_url:
                print(
                    f"Processing race {index + 1}/{len(results)}: {race.get('race_id')} at {race.get('venue')}"
                )
                race_data = self.collect_comprehensive_race_data(race_url)

                if race_data:
                    collected_races += 1
                    total_dogs += len(race_data["dogs_data"])
                    print(f"✅ Successfully processed race {index + 1}")
                else:
                    print(f"❌ Failed to process race {index + 1}")

                # Rate limiting
                time.sleep(3)

        print(f"✅ Comprehensive collection complete!")
        print(f"   Races processed: {collected_races}")
        print(f"   Total dogs: {total_dogs}")
        print(f"   Database: {self.db_path}")

    def generate_data_summary(self):
        """Generate comprehensive summary of collected data"""
        conn = sqlite3.connect(self.db_path)

        summary = {}

        # Race metadata summary
        race_stats = pd.read_sql_query(
            """
            SELECT 
                COUNT(*) as total_races,
                COUNT(DISTINCT venue) as unique_venues,
                COUNT(DISTINCT grade) as unique_grades,
                AVG(field_size) as avg_field_size,
                AVG(prize_money_total) as avg_prize_money
            FROM race_metadata
        """,
            conn,
        )

        # Dog data summary
        dog_stats = pd.read_sql_query(
            """
            SELECT 
                COUNT(*) as total_dog_entries,
                COUNT(DISTINCT dog_clean_name) as unique_dogs,
                COUNT(DISTINCT trainer_name) as unique_trainers,
                AVG(weight) as avg_weight,
                COUNT(CASE WHEN was_scratched THEN 1 END) as total_scratched
            FROM dog_race_data
        """,
            conn,
        )

        # Odds data summary
        odds_stats = pd.read_sql_query(
            """
            SELECT 
                COUNT(*) as total_odds_entries,
                AVG(odds_decimal) as avg_odds,
                MIN(odds_decimal) as min_odds,
                MAX(odds_decimal) as max_odds
            FROM odds_snapshots
        """,
            conn,
        )

        summary = {
            "race_statistics": (
                race_stats.to_dict("records")[0] if len(race_stats) > 0 else {}
            ),
            "dog_statistics": (
                dog_stats.to_dict("records")[0] if len(dog_stats) > 0 else {}
            ),
            "odds_statistics": (
                odds_stats.to_dict("records")[0] if len(odds_stats) > 0 else {}
            ),
            "generation_time": datetime.now().isoformat(),
        }

        conn.close()
        return summary

    # Utility methods
    def parse_odds_to_decimal(self, odds_text):
        """Convert various odds formats to decimal"""
        if not odds_text:
            return None

        odds_text = odds_text.strip().replace("$", "").replace(",", "")

        # Decimal odds (e.g., "3.50")
        if re.match(r"^\d+\.?\d*$", odds_text):
            return float(odds_text)

        # Fractional odds (e.g., "5/2", "7/4")
        frac_match = re.match(r"^(\d+)/(\d+)$", odds_text)
        if frac_match:
            numerator, denominator = frac_match.groups()
            return (float(numerator) / float(denominator)) + 1.0

        return None

    def clean_dog_name(self, name):
        """Clean dog name for matching"""
        return (
            "".join(c for c in name if c.isalnum()).lower()
            if isinstance(name, str)
            else name
        )

    def extract_race_id_from_url(self, url):
        """Extract race identifier from URL"""
        patterns = [
            r"/([^/]+)/([\d-]+)/(\d+)/",  # venue/date/race_number
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return "_".join(match.groups())

        return str(hash(url))[-8:]

    def cleanup(self):
        """Clean up resources"""
        if self.driver:
            self.driver.quit()


def main():
    """Main execution function"""
    collector = EnhancedGreyhoundDataCollector()

    try:
        # Collect comprehensive data
        collector.collect_from_race_results()

        # Generate summary
        summary = collector.generate_data_summary()

        print("\n📊 Comprehensive Data Collection Summary:")
        print(f"  Total races: {summary['race_statistics'].get('total_races', 0)}")
        print(
            f"  Total dog entries: {summary['dog_statistics'].get('total_dog_entries', 0)}"
        )
        print(f"  Unique dogs: {summary['dog_statistics'].get('unique_dogs', 0)}")
        print(
            f"  Unique trainers: {summary['dog_statistics'].get('unique_trainers', 0)}"
        )
        print(
            f"  Total odds entries: {summary['odds_statistics'].get('total_odds_entries', 0)}"
        )
        print(
            f"  Average field size: {summary['race_statistics'].get('avg_field_size', 0):.1f}"
        )

        # Save summary
        with open("comprehensive_data_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print("✅ Comprehensive data collection complete!")

    except Exception as e:
        print(f"❌ Error in comprehensive collection: {e}")

    finally:
        collector.cleanup()


if __name__ == "__main__":
    main()
