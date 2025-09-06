#!/usr/bin/env python3
"""
Comprehensive Form Data Collector
=================================

Integrates multiple data sources to gather comprehensive form data:
1. Greyhound Recorder scraper for detailed form guides
2. Individual dog profile scraping for career statistics
3. Parallel data sources for complete coverage
4. Enhanced data processing for full career statistics

This addresses the limitation of only having 4-5 races per dog by implementing
comprehensive career data collection.

Author: AI Assistant
Date: July 31, 2025
"""

import json
import logging
import os
import re
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# Import existing components
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# NOTE: Do not import TheGreyhoundRecorderScraper at module level to avoid heavy deps
# It will be imported lazily when first needed using importlib.import_module

try:
    from bs4 import BeautifulSoup

    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DogProfile:
    """Comprehensive dog profile data structure"""

    name: str
    career_races: int
    career_wins: int
    career_places: int
    career_earnings: float
    best_time: float
    average_time: float
    track_preferences: Dict[str, float]
    distance_preferences: Dict[str, float]
    grade_performance: Dict[str, float]
    trainer_history: List[Dict[str, Any]]
    recent_form: List[Dict[str, Any]]
    sectional_data: List[Dict[str, Any]]
    injury_history: List[Dict[str, Any]]
    breeding_info: Dict[str, Any]


class ComprehensiveFormDataCollector:
    """
    Main collector that orchestrates comprehensive form data gathering
    from multiple sources with parallel processing and intelligent caching.
    """

    def __init__(self, db_path="greyhound_racing_data.db", max_workers=4):
        self.db_path = db_path
        self.max_workers = max_workers
        self.cache_dir = Path("./comprehensive_form_cache")
        self.cache_dir.mkdir(exist_ok=True)

        # Initialize data sources lazily (defer heavy import and instantiation)
        self.greyhound_recorder = None

        # Initialize web driver for additional scraping
        self.driver = None
        self._setup_driver()

        # Performance tracking
        self.stats = {
            "dogs_processed": 0,
            "profiles_enhanced": 0,
            "errors": 0,
            "cache_hits": 0,
            "api_calls": 0,
        }

        # Initialize database enhancements
        self._enhance_database_schema()

        logger.info("🚀 Comprehensive Form Data Collector initialized")
        logger.info(f"   Max workers: {max_workers}")
        logger.info(f"   Cache directory: {self.cache_dir}")
        logger.info(f"   Database: {db_path}")

    def _load_greyhound_recorder(self):
        """Lazily import and instantiate TheGreyhoundRecorderScraper when needed."""
        if self.greyhound_recorder is not None:
            return self.greyhound_recorder
        try:
            import importlib
            import sys as _sys

            # Ensure collector path is available for import by module name used in repo
            collectors_path = "src/collectors"
            if collectors_path not in _sys.path:
                _sys.path.append(collectors_path)
            tgr_module = importlib.import_module("the_greyhound_recorder_scraper")
            Scraper = getattr(tgr_module, "TheGreyhoundRecorderScraper")
            self.greyhound_recorder = Scraper(
                rate_limit=1.5,
                cache_dir=str(self.cache_dir / "tgr_cache"),
                use_cache=True,
            )
            logger.info("✅ TheGreyhoundRecorderScraper lazily loaded")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load TheGreyhoundRecorderScraper lazily: {e}")
            self.greyhound_recorder = None
        return self.greyhound_recorder

    def _setup_driver(self):
        """Setup Chrome driver for additional scraping"""
        try:
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
            options.add_argument(
                "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            )

            self.driver = webdriver.Chrome(options=options)
            logger.info("✅ Chrome driver initialized for additional scraping")
        except Exception as e:
            logger.warning(f"⚠️ Chrome driver setup failed: {e}")
            self.driver = None

    def _enhance_database_schema(self):
        """Add comprehensive form data tables to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create comprehensive dog profiles table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS comprehensive_dog_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dog_name TEXT UNIQUE,
                dog_clean_name TEXT,
                career_races INTEGER DEFAULT 0,
                career_wins INTEGER DEFAULT 0,
                career_places INTEGER DEFAULT 0,
                career_earnings REAL DEFAULT 0,
                best_time REAL,
                average_time REAL,
                win_percentage REAL,
                place_percentage REAL,
                track_preferences TEXT,  -- JSON
                distance_preferences TEXT,  -- JSON
                grade_performance TEXT,  -- JSON
                trainer_history TEXT,  -- JSON
                recent_form_extended TEXT,  -- JSON - 20+ races
                sectional_data TEXT,  -- JSON
                injury_history TEXT,  -- JSON
                breeding_info TEXT,  -- JSON
                last_updated DATETIME,
                data_completeness_score REAL,
                profile_source TEXT,
                UNIQUE(dog_clean_name)
            )
        """
        )

        # Create detailed race history table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS detailed_race_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dog_name TEXT,
                race_date DATE,
                venue TEXT,
                race_number INTEGER,
                distance INTEGER,
                grade TEXT,
                track_condition TEXT,
                weather TEXT,
                box_number INTEGER,
                finish_position INTEGER,
                race_time REAL,
                sectional_times TEXT,  -- JSON
                margin REAL,
                starting_odds REAL,
                prize_money REAL,
                field_size INTEGER,
                trainer_name TEXT,
                weight REAL,
                race_class TEXT,
                track_record BOOLEAN,
                data_source TEXT,
                extraction_timestamp DATETIME,
                FOREIGN KEY (dog_name) REFERENCES comprehensive_dog_profiles (dog_clean_name)
            )
        """
        )

        # Create trainer performance table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trainer_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trainer_name TEXT,
                total_races INTEGER,
                total_wins INTEGER,
                win_percentage REAL,
                speciality_tracks TEXT,  -- JSON
                speciality_distances TEXT,  -- JSON
                recent_form TEXT,  -- JSON
                last_updated DATETIME,
                UNIQUE(trainer_name)
            )
        """
        )

        conn.commit()
        conn.close()

        logger.info("✅ Enhanced database schema for comprehensive form data")

    def collect_comprehensive_form_data(
        self, race_file_path: str = None, target_dogs: List[str] = None
    ) -> Dict[str, Any]:
        """
        Main method to collect comprehensive form data.

        Args:
            race_file_path: Optional specific race file to process
            target_dogs: Optional list of specific dogs to enhance

        Returns:
            Dict with collection results and statistics
        """
        logger.info("🚀 Starting comprehensive form data collection...")
        start_time = time.time()

        # Step 1: Identify target dogs
        if target_dogs is None:
            target_dogs = self._identify_dogs_needing_enhancement(race_file_path)

        logger.info(f"🎯 Target dogs for enhancement: {len(target_dogs)}")

        # Step 2: Collect from Greyhound Recorder (ensure lazy loader is initialized)
        if self.greyhound_recorder is None:
            self._load_greyhound_recorder()
        recorder_data = self._collect_from_greyhound_recorder()

        # Step 3: Enhanced dog profile scraping with parallel processing
        profile_results = self._collect_dog_profiles_parallel(target_dogs)

        # Step 4: Integrate and store enhanced data
        integration_results = self._integrate_and_store_data(
            profile_results, recorder_data
        )

        # Step 5: Generate comprehensive statistics
        end_time = time.time()

        results = {
            "success": True,
            "execution_time": end_time - start_time,
            "dogs_processed": len(target_dogs),
            "profiles_enhanced": self.stats["profiles_enhanced"],
            "data_sources_used": self._get_active_data_sources(),
            "collection_stats": self.stats,
            "integration_results": integration_results,
            "cache_efficiency": self.stats["cache_hits"]
            / max(1, self.stats["api_calls"]),
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"✅ Comprehensive form data collection completed in {end_time - start_time:.2f}s"
        )
        logger.info(f"   Dogs processed: {results['dogs_processed']}")
        logger.info(f"   Profiles enhanced: {results['profiles_enhanced']}")
        logger.info(f"   Cache efficiency: {results['cache_efficiency']:.2%}")

        return results

    def _identify_dogs_needing_enhancement(
        self, race_file_path: str = None
    ) -> List[str]:
        """Identify dogs that need comprehensive form data enhancement"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if race_file_path:
            # Get dogs from specific race file
            import pandas as pd

            try:
                df = pd.read_csv(race_file_path)
                dogs = []
                for _, row in df.iterrows():
                    dog_name = str(row.get("Dog Name", "")).strip()
                    if dog_name and dog_name != "":
                        # Clean dog name
                        clean_name = re.sub(r"^\d+\.\s*", "", dog_name)
                        dogs.append(clean_name)
                return list(set(dogs))
            except Exception as e:
                logger.error(f"Error reading race file {race_file_path}: {e}")

        # Get dogs with limited historical data (current limitation)
        cursor.execute(
            """
            SELECT DISTINCT dog_clean_name 
            FROM dog_race_data 
            WHERE dog_clean_name NOT IN (
                SELECT DISTINCT dog_clean_name 
                FROM comprehensive_dog_profiles 
                WHERE data_completeness_score > 0.8
                AND last_updated > datetime('now', '-7 days')
            )
            AND dog_clean_name IS NOT NULL 
            AND dog_clean_name != ''
            LIMIT 100
        """
        )

        dogs = [row[0] for row in cursor.fetchall()]
        conn.close()

        return dogs

    def _collect_from_greyhound_recorder(self):
        """Collect comprehensive form guides from Greyhound Recorder"""
        logger.info("📋 Collecting form guides from Greyhound Recorder...")

        try:
            # Ensure scraper is available via lazy loader
            if self.greyhound_recorder is None:
                self._load_greyhound_recorder()
            if self.greyhound_recorder is None:
                logger.warning(
                    "Greyhound Recorder scraper unavailable; skipping this source"
                )
                return {"success": False, "error": "scraper_unavailable"}

            # Get current form guides
            form_guides = self.greyhound_recorder.fetch_form_guides()

            # Get detailed race data
            detailed_meetings = self.greyhound_recorder.fetch_all_meetings_with_races()

            recorder_data = {
                "form_guides": form_guides,
                "detailed_meetings": detailed_meetings,
                "collection_timestamp": datetime.now().isoformat(),
                "success": True,
            }

            logger.info(
                f"✅ Collected {len(form_guides.get('meetings', []))} meetings from Greyhound Recorder"
            )

            return recorder_data

        except Exception as e:
            logger.error(f"❌ Error collecting from Greyhound Recorder: {e}")
            return {"success": False, "error": str(e)}

    def _collect_dog_profiles_parallel(self, target_dogs: List[str]) -> Dict[str, Any]:
        """Collect individual dog profiles using parallel processing"""
        logger.info(
            f"🐕 Collecting comprehensive profiles for {len(target_dogs)} dogs..."
        )

        results = {
            "successful_profiles": {},
            "failed_profiles": [],
            "cache_hits": 0,
            "api_calls": 0,
        }

        # Process dogs in parallel batches
        batch_size = min(self.max_workers, len(target_dogs))

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Submit all dog profile collection tasks
            future_to_dog = {
                executor.submit(self._collect_single_dog_profile, dog_name): dog_name
                for dog_name in target_dogs
            }

            # Process completed tasks
            for future in as_completed(future_to_dog):
                dog_name = future_to_dog[future]
                try:
                    profile_data = future.result(
                        timeout=30
                    )  # 30 second timeout per dog
                    if profile_data and profile_data.get("success"):
                        results["successful_profiles"][dog_name] = profile_data
                        self.stats["profiles_enhanced"] += 1
                    else:
                        results["failed_profiles"].append(
                            {
                                "dog_name": dog_name,
                                "error": profile_data.get("error", "Unknown error"),
                            }
                        )
                        self.stats["errors"] += 1

                except Exception as e:
                    logger.error(f"Error processing {dog_name}: {e}")
                    results["failed_profiles"].append(
                        {"dog_name": dog_name, "error": str(e)}
                    )
                    self.stats["errors"] += 1

                self.stats["dogs_processed"] += 1

                # Progress logging
                if self.stats["dogs_processed"] % 10 == 0:
                    logger.info(
                        f"   Progress: {self.stats['dogs_processed']}/{len(target_dogs)} dogs processed"
                    )

        logger.info(f"✅ Dog profile collection completed")
        logger.info(f"   Successful: {len(results['successful_profiles'])}")
        logger.info(f"   Failed: {len(results['failed_profiles'])}")

        return results

    def _collect_single_dog_profile(self, dog_name: str) -> Dict[str, Any]:
        """Collect comprehensive profile data for a single dog"""
        try:
            # Check cache first
            cache_key = self._generate_cache_key(dog_name)
            cached_data = self._get_cached_profile(cache_key)

            if cached_data:
                self.stats["cache_hits"] += 1
                return cached_data

            self.stats["api_calls"] += 1

            # Collect from multiple sources
            profile_data = {
                "dog_name": dog_name,
                "collection_timestamp": datetime.now().isoformat(),
                "data_sources": [],
                "career_statistics": {},
                "detailed_history": [],
                "performance_metrics": {},
                "success": False,
            }

            # Source 1: Greyhound Recorder individual dog page
            recorder_profile = self._scrape_dog_from_recorder(dog_name)
            if recorder_profile:
                profile_data["data_sources"].append("greyhound_recorder")
                profile_data["career_statistics"].update(
                    recorder_profile.get("career_stats", {})
                )
                profile_data["detailed_history"].extend(
                    recorder_profile.get("race_history", [])
                )

            # Source 2: FastTrack (if available)
            fasttrack_profile = self._scrape_dog_from_fasttrack(dog_name)
            if fasttrack_profile:
                profile_data["data_sources"].append("fasttrack")
                profile_data["performance_metrics"].update(
                    fasttrack_profile.get("performance", {})
                )

            # Source 3: Additional racing sites
            additional_data = self._scrape_dog_from_additional_sources(dog_name)
            if additional_data:
                profile_data["data_sources"].extend(additional_data.get("sources", []))
                profile_data["detailed_history"].extend(
                    additional_data.get("races", [])
                )

            # Calculate comprehensive metrics
            if profile_data["detailed_history"]:
                profile_data["performance_metrics"] = (
                    self._calculate_comprehensive_metrics(
                        profile_data["detailed_history"]
                    )
                )
                profile_data["success"] = True

                # Cache the successful result
                self._cache_profile(cache_key, profile_data)

            return profile_data

        except Exception as e:
            logger.error(f"Error collecting profile for {dog_name}: {e}")
            return {"success": False, "error": str(e), "dog_name": dog_name}

    def _scrape_dog_from_recorder(self, dog_name: str) -> Optional[Dict[str, Any]]:
        """Scrape individual dog profile from Greyhound Recorder"""
        try:
            # This would need to be implemented based on Greyhound Recorder's individual dog pages
            # For now, return placeholder structure
            logger.debug(f"Scraping {dog_name} from Greyhound Recorder...")

            # TODO: Implement actual scraping logic for individual dog profiles
            # This would involve:
            # 1. Searching for the dog by name
            # 2. Navigating to their profile page
            # 3. Extracting comprehensive career statistics
            # 4. Collecting detailed race history (20-50+ races)

            return None

        except Exception as e:
            logger.error(f"Error scraping {dog_name} from Greyhound Recorder: {e}")
            return None

    def _scrape_dog_from_fasttrack(self, dog_name: str) -> Optional[Dict[str, Any]]:
        """Scrape dog data from FastTrack racing database"""
        try:
            if not self.driver:
                return None

            # Implement FastTrack scraping
            # This is a placeholder for actual FastTrack integration
            logger.debug(f"Scraping {dog_name} from FastTrack...")

            # TODO: Implement FastTrack scraping
            return None

        except Exception as e:
            logger.error(f"Error scraping {dog_name} from FastTrack: {e}")
            return None

    def _scrape_dog_from_additional_sources(
        self, dog_name: str
    ) -> Optional[Dict[str, Any]]:
        """Scrape dog data from additional racing websites"""
        try:
            # Implement additional source scraping
            # This could include:
            # - thedogs.com.au individual dog pages
            # - punters.com.au
            # - Racing and Sports
            # - etc.

            additional_data = {"sources": [], "races": []}

            # Placeholder implementation
            return additional_data if additional_data["sources"] else None

        except Exception as e:
            logger.error(f"Error scraping {dog_name} from additional sources: {e}")
            return None

    def _calculate_comprehensive_metrics(
        self, race_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics from race history"""
        if not race_history:
            return {}

        try:
            total_races = len(race_history)
            wins = sum(1 for race in race_history if race.get("position") == 1)
            places = sum(1 for race in race_history if race.get("position", 0) <= 3)

            # Calculate performance metrics
            metrics = {
                "total_races": total_races,
                "total_wins": wins,
                "total_places": places,
                "win_percentage": (wins / total_races) * 100 if total_races > 0 else 0,
                "place_percentage": (
                    (places / total_races) * 100 if total_races > 0 else 0
                ),
                "average_position": (
                    sum(race.get("position", 8) for race in race_history) / total_races
                    if total_races > 0
                    else 8
                ),
            }

            # Calculate time-based metrics
            times = [race.get("time") for race in race_history if race.get("time")]
            if times:
                metrics["best_time"] = min(times)
                metrics["average_time"] = sum(times) / len(times)
                metrics["time_consistency"] = np.std(times) if len(times) > 1 else 0

            # Track preferences
            track_performance = {}
            for race in race_history:
                track = race.get("track", "Unknown")
                if track not in track_performance:
                    track_performance[track] = {"races": 0, "wins": 0}
                track_performance[track]["races"] += 1
                if race.get("position") == 1:
                    track_performance[track]["wins"] += 1

            metrics["track_preferences"] = {
                track: stats["wins"] / stats["races"] if stats["races"] > 0 else 0
                for track, stats in track_performance.items()
            }

            return metrics

        except Exception as e:
            logger.error(f"Error calculating comprehensive metrics: {e}")
            return {}

    def _integrate_and_store_data(
        self, profile_results: Dict[str, Any], recorder_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Integrate collected data and store in enhanced database schema"""
        logger.info("🔧 Integrating and storing comprehensive form data...")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        integration_stats = {
            "profiles_stored": 0,
            "race_records_added": 0,
            "trainer_records_updated": 0,
            "errors": 0,
        }

        try:
            # Store comprehensive dog profiles
            for dog_name, profile_data in profile_results.get(
                "successful_profiles", {}
            ).items():
                try:
                    metrics = profile_data.get("performance_metrics", {})

                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO comprehensive_dog_profiles (
                            dog_name, dog_clean_name, career_races, career_wins, career_places,
                            career_earnings, best_time, average_time, win_percentage, place_percentage,
                            track_preferences, distance_preferences, grade_performance,
                            recent_form_extended, last_updated, data_completeness_score, profile_source
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            dog_name,
                            dog_name,  # Clean name
                            metrics.get("total_races", 0),
                            metrics.get("total_wins", 0),
                            metrics.get("total_places", 0),
                            metrics.get("career_earnings", 0),
                            metrics.get("best_time"),
                            metrics.get("average_time"),
                            metrics.get("win_percentage", 0),
                            metrics.get("place_percentage", 0),
                            json.dumps(metrics.get("track_preferences", {})),
                            json.dumps(metrics.get("distance_preferences", {})),
                            json.dumps(metrics.get("grade_performance", {})),
                            json.dumps(profile_data.get("detailed_history", [])),
                            datetime.now(),
                            self._calculate_completeness_score(profile_data),
                            ",".join(profile_data.get("data_sources", [])),
                        ),
                    )

                    # Store detailed race history
                    for race in profile_data.get("detailed_history", []):
                        cursor.execute(
                            """
                            INSERT OR IGNORE INTO detailed_race_history (
                                dog_name, race_date, venue, race_number, distance, grade,
                                track_condition, weather, box_number, finish_position,
                                race_time, margin, starting_odds, field_size, trainer_name,
                                weight, data_source, extraction_timestamp
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                dog_name,
                                race.get("date"),
                                race.get("venue"),
                                race.get("race_number"),
                                race.get("distance"),
                                race.get("grade"),
                                race.get("track_condition"),
                                race.get("weather"),
                                race.get("box"),
                                race.get("position"),
                                race.get("time"),
                                race.get("margin"),
                                race.get("odds"),
                                race.get("field_size"),
                                race.get("trainer"),
                                race.get("weight"),
                                "comprehensive_collector",
                                datetime.now(),
                            ),
                        )

                        integration_stats["race_records_added"] += 1

                    integration_stats["profiles_stored"] += 1

                except Exception as e:
                    logger.error(f"Error storing profile for {dog_name}: {e}")
                    integration_stats["errors"] += 1

            conn.commit()

        except Exception as e:
            logger.error(f"Error during data integration: {e}")
            conn.rollback()
            integration_stats["errors"] += 1
        finally:
            conn.close()

        logger.info(f"✅ Data integration completed")
        logger.info(f"   Profiles stored: {integration_stats['profiles_stored']}")
        logger.info(f"   Race records added: {integration_stats['race_records_added']}")

        return integration_stats

    def _calculate_completeness_score(self, profile_data: Dict[str, Any]) -> float:
        """Calculate data completeness score (0.0 to 1.0)"""
        score = 0.0
        max_score = 5.0

        # Check data sources (0.2 points per source, max 1.0)
        sources = len(profile_data.get("data_sources", []))
        score += min(sources * 0.2, 1.0)

        # Check race history depth (up to 1.0)
        races = len(profile_data.get("detailed_history", []))
        score += min(races / 20.0, 1.0)  # Full score at 20+ races

        # Check performance metrics completeness (up to 1.0)
        metrics = profile_data.get("performance_metrics", {})
        required_metrics = [
            "total_races",
            "win_percentage",
            "best_time",
            "track_preferences",
        ]
        metrics_present = sum(1 for metric in required_metrics if metric in metrics)
        score += metrics_present / len(required_metrics)

        # Check career statistics (up to 1.0)
        career_stats = profile_data.get("career_statistics", {})
        if career_stats:
            score += 1.0

        # Check recent form depth (up to 1.0)
        recent_form = profile_data.get("detailed_history", [])
        if len(recent_form) >= 10:
            score += 1.0
        elif len(recent_form) >= 5:
            score += 0.5

        return min(score / max_score, 1.0)

    def _get_active_data_sources(self) -> List[str]:
        """Get list of active data sources"""
        sources = ["greyhound_recorder"]

        if self.driver:
            sources.append("selenium_scraper")

        return sources

    def _generate_cache_key(self, dog_name: str) -> str:
        """Generate cache key for dog profile"""
        import hashlib

        return hashlib.md5(f"dog_profile_{dog_name}".encode()).hexdigest()

    def _get_cached_profile(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached dog profile if available and recent"""
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                # Check if cache is recent (within 24 hours)
                cache_age = time.time() - cache_file.stat().st_mtime
                if cache_age < 86400:  # 24 hours
                    with open(cache_file, "r") as f:
                        return json.load(f)
            except Exception as e:
                logger.debug(f"Cache read error: {e}")

        return None

    def _cache_profile(self, cache_key: str, profile_data: Dict[str, Any]):
        """Cache dog profile data"""
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            with open(cache_file, "w") as f:
                json.dump(profile_data, f, indent=2, default=str)
        except Exception as e:
            logger.debug(f"Cache write error: {e}")

    def get_collection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive collection statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get profile statistics
        cursor.execute(
            """
            SELECT 
                COUNT(*) as total_profiles,
                AVG(career_races) as avg_career_races,
                AVG(data_completeness_score) as avg_completeness,
                COUNT(CASE WHEN data_completeness_score > 0.8 THEN 1 END) as high_quality_profiles
            FROM comprehensive_dog_profiles
        """
        )

        profile_stats = cursor.fetchone()

        # Get race history statistics
        cursor.execute(
            """
            SELECT 
                COUNT(*) as total_race_records,
                COUNT(DISTINCT dog_name) as dogs_with_history,
                AVG(race_time) as avg_race_time
            FROM detailed_race_history
        """
        )

        history_stats = cursor.fetchone()

        conn.close()

        return {
            "profile_statistics": {
                "total_profiles": profile_stats[0],
                "average_career_races": profile_stats[1] or 0,
                "average_completeness_score": profile_stats[2] or 0,
                "high_quality_profiles": profile_stats[3],
            },
            "race_history_statistics": {
                "total_race_records": history_stats[0],
                "dogs_with_detailed_history": history_stats[1],
                "average_race_time": history_stats[2] or 0,
            },
            "collection_stats": self.stats,
            "timestamp": datetime.now().isoformat(),
        }

    def cleanup(self):
        """Cleanup resources"""
        if self.driver:
            self.driver.quit()
        logger.info("🧹 Comprehensive Form Data Collector cleanup completed")


def main():
    """Main function for testing the comprehensive form data collector"""
    collector = ComprehensiveFormDataCollector(max_workers=2)

    try:
        # Test collection on a small set
        results = collector.collect_comprehensive_form_data()

        print("🎉 Collection Results:")
        print(f"   Execution time: {results['execution_time']:.2f}s")
        print(f"   Dogs processed: {results['dogs_processed']}")
        print(f"   Profiles enhanced: {results['profiles_enhanced']}")
        print(f"   Cache efficiency: {results['cache_efficiency']:.2%}")

        # Get statistics
        stats = collector.get_collection_statistics()
        print(f"\n📊 Collection Statistics:")
        print(f"   Total profiles: {stats['profile_statistics']['total_profiles']}")
        print(
            f"   Average career races: {stats['profile_statistics']['average_career_races']:.1f}"
        )
        print(
            f"   Average completeness: {stats['profile_statistics']['average_completeness_score']:.2f}"
        )

    finally:
        collector.cleanup()


if __name__ == "__main__":
    main()
