#!/usr/bin/env python3
"""
Enhanced Database Manager for Greyhound Racing System
======================================================

This module provides an optimized database interface that:
1. Uses the new data integrity system for all operations
2. Provides optimized queries with proper formatting
3. Integrates with the safe data ingestion system
4. Ensures all endpoints benefit from duplicate prevention

Author: AI Assistant
Date: 2025-01-27
"""

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

# Import our data integrity systems
from data_integrity_system import DataIntegrityManager
from safe_data_ingestion import SafeDataIngestion


class EnhancedDatabaseManager:
    """Enhanced database manager with integrity protection and optimization"""

    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path
        self.integrity_manager = DataIntegrityManager(db_path)
        self.safe_ingestion = SafeDataIngestion(db_path)
        self.setup_logging()

        # Optimized venue mapping with proper formatting
        self.venue_url_map = {
            "AP_K": "angle-park",
            "HOBT": "hobart",
            "GOSF": "gosford",
            "DAPT": "dapto",
            "SAN": "sandown",
            "MEA": "the-meadows",
            "WPK": "wentworth-park",
            "CANN": "cannington",
            "BAL": "ballarat",
            "BEN": "bendigo",
            "GEE": "geelong",
            "WAR": "warrnambool",
            "NOR": "northam",
            "MAND": "mandurah",
            "MURR": "murray-bridge",
            "GAWL": "gawler",
            "MOUNT": "mount-gambier",
            "TRA": "traralgon",
            "SAL": "sale",
            "RICH": "richmond",
            "HEA": "healesville",
            "CASO": "casino",
            "GRDN": "the-gardens",
            "DARW": "darwin",
            "ALBION": "albion-park",
            "HOR": "horsham",
        }

        # Cache for improved performance
        self._cache = {}
        self._cache_timeout = 300  # 5 minutes

    def setup_logging(self):
        """Setup logging for database operations"""
        os.makedirs("logs", exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - DATABASE - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("logs/enhanced_database.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    @contextmanager
    def get_connection(self):
        """Get database connection with proper error handling and optimization"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            # Enable foreign keys and optimize performance
            try:
                conn.execute("PRAGMA foreign_keys = ON")
            except Exception:
                pass

            # Disable WAL and heavy PRAGMAs when running tests or when explicitly requested
            disable_wal = False
            try:
                if os.environ.get("SQLITE_DISABLE_WAL", "0").lower() in ("1", "true", "yes"):
                    disable_wal = True
                if os.environ.get("PYTEST_CURRENT_TEST"):
                    disable_wal = True
                if os.environ.get("FLASK_ENV", "").lower() == "testing" or os.environ.get("TESTING", "0").lower() in ("1", "true", "yes"):
                    disable_wal = True
            except Exception:
                disable_wal = False

            try:
                if disable_wal:
                    conn.execute("PRAGMA journal_mode = DELETE")
                    conn.execute("PRAGMA synchronous = OFF")
                else:
                    conn.execute("PRAGMA journal_mode = WAL")
                    conn.execute("PRAGMA synchronous = NORMAL")
                    conn.execute("PRAGMA cache_size = 10000")
                    conn.execute("PRAGMA temp_store = MEMORY")
                    try:
                        conn.execute("PRAGMA mmap_size = 268435456")
                    except Exception:
                        pass
            except Exception:
                # Fallback-safe settings
                try:
                    conn.execute("PRAGMA journal_mode = DELETE")
                    conn.execute("PRAGMA synchronous = OFF")
                except Exception:
                    pass

            # Set Row factory for easier data handling
            conn.row_factory = sqlite3.Row

            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def safe_insert_record(
        self, table_name: str, record: Dict
    ) -> Tuple[bool, List[str]]:
        """Safely insert a record using the data integrity system"""
        return self.safe_ingestion.insert_single_record(record, table_name)

    def safe_insert_batch(self, table_name: str, records: List[Dict]) -> Dict:
        """Safely insert a batch of records"""
        return self.safe_ingestion.insert_batch_records(records, table_name)

    def get_cache_key(self, method_name: str, params: tuple) -> str:
        """Generate cache key for method and parameters"""
        return f"{method_name}:{hash(params)}"

    def is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid"""
        if cache_key not in self._cache:
            return False

        cache_time = self._cache[cache_key].get("timestamp", 0)
        return (datetime.now().timestamp() - cache_time) < self._cache_timeout

    def set_cache(self, cache_key: str, data: Any):
        """Set cache entry with timestamp"""
        self._cache[cache_key] = {"data": data, "timestamp": datetime.now().timestamp()}

        # Clean old cache entries (keep only last 100)
        if len(self._cache) > 100:
            oldest_key = min(
                self._cache.keys(), key=lambda k: self._cache[k]["timestamp"]
            )
            del self._cache[oldest_key]

    def get_cache(self, cache_key: str) -> Any:
        """Get cached data"""
        return self._cache[cache_key]["data"] if cache_key in self._cache else None

    def format_race_data(self, race_row: sqlite3.Row) -> Dict:
        """Format race data with proper null handling and URL generation"""
        race_dict = dict(race_row)

        # Generate race URL if missing
        if not race_dict.get("url"):
            race_dict["url"] = self.generate_race_url(
                race_dict.get("venue"),
                race_dict.get("race_date"),
                race_dict.get("race_number"),
            )

        # Format extraction timestamp
        extraction_time = race_dict.get("extraction_timestamp")
        if extraction_time:
            try:
                dt = datetime.fromisoformat(extraction_time.replace("Z", "+00:00"))
                race_dict["formatted_extraction_time"] = dt.strftime("%Y-%m-%d %H:%M")
            except:
                race_dict["formatted_extraction_time"] = extraction_time
        else:
            race_dict["formatted_extraction_time"] = "Unknown"

        # Clean null values and handle non-serializable types
        for key, value in race_dict.items():
            if value is None or (
                isinstance(value, str) and value.lower() in ["nan", "null", "none", ""]
            ):
                race_dict[key] = None
            elif isinstance(value, bytes):
                # Convert bytes to string for JSON serialization
                try:
                    race_dict[key] = value.decode('utf-8')
                except UnicodeDecodeError:
                    race_dict[key] = str(value)

        return race_dict

    def format_dog_data(self, dog_row: sqlite3.Row) -> Dict:
        """Format dog data with proper null handling and data cleaning"""
        dog_dict = dict(dog_row)

        # Clean up the dog data
        for key, value in dog_dict.items():
            if value == "nan" or value is None:
                dog_dict[key] = None
            elif isinstance(value, str) and value.lower() == "nan":
                dog_dict[key] = None
            elif isinstance(value, bytes):
                # Convert bytes to string for JSON serialization
                try:
                    dog_dict[key] = value.decode('utf-8')
                except UnicodeDecodeError:
                    dog_dict[key] = str(value)

        # Parse historical_records JSON if it exists
        if dog_dict.get("historical_records"):
            try:
                dog_dict["historical_data"] = json.loads(dog_dict["historical_records"])
            except (json.JSONDecodeError, TypeError):
                dog_dict["historical_data"] = []

        return dog_dict

    def generate_race_url(
        self, venue: str, race_date: str, race_number: int
    ) -> Optional[str]:
        """Generate race URL with proper error handling"""
        try:
            if not all([venue, race_date, race_number]):
                return None

            venue_slug = self.venue_url_map.get(venue, venue.lower() if venue else "")
            return f"https://www.thedogs.com.au/racing/{venue_slug}/{race_date}/{race_number}"
        except Exception as e:
            self.logger.warning(f"Error generating race URL: {e}")
            return None

    def get_recent_races(self, limit: int = 10) -> List[Dict]:
        """Get recent races with caching and optimization"""
        cache_key = self.get_cache_key("get_recent_races", (limit,))

        if self.is_cache_valid(cache_key):
            return self.get_cache(cache_key)

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Optimized query with proper indexing
                query = """
                SELECT 
                    race_id, venue, race_number, race_date, race_name,
                    grade, distance, field_size, winner_name, winner_odds,
                    winner_margin, url, extraction_timestamp, track_condition,
                    weather, temperature, humidity
                FROM race_metadata 
                WHERE winner_name IS NOT NULL 
                    AND winner_name != '' 
                    AND winner_name != 'nan'
                ORDER BY 
                    CASE 
                        WHEN extraction_timestamp IS NOT NULL THEN datetime(extraction_timestamp) 
                        ELSE datetime('1900-01-01') 
                    END DESC,
                    race_date DESC,
                    race_number DESC
                LIMIT ?
                """

                cursor.execute(query, (limit,))
                races = cursor.fetchall()

                result = [self.format_race_data(race) for race in races]

                # Cache the result
                self.set_cache(cache_key, result)

                self.logger.info(f"Retrieved {len(result)} recent races")
                return result

        except Exception as e:
            self.logger.error(f"Error getting recent races: {e}")
            return []

    def get_paginated_races(
        self, page: int = 1, per_page: int = 20, filters: Dict = None
    ) -> Dict:
        """Get paginated races with advanced filtering and optimization"""
        cache_key = self.get_cache_key(
            "get_paginated_races", (page, per_page, str(filters or {}))
        )

        if self.is_cache_valid(cache_key):
            return self.get_cache(cache_key)

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Calculate offset
                offset = (page - 1) * per_page

                # Build WHERE clause for filters
                where_conditions = []
                params = []

                if filters:
                    if filters.get("venue"):
                        where_conditions.append("venue = ?")
                        params.append(filters["venue"])

                    if filters.get("date_from"):
                        where_conditions.append("race_date >= ?")
                        params.append(filters["date_from"])

                    if filters.get("date_to"):
                        where_conditions.append("race_date <= ?")
                        params.append(filters["date_to"])

                    if filters.get("has_winner"):
                        where_conditions.append(
                            "winner_name IS NOT NULL AND winner_name != '' AND winner_name != 'nan'"
                        )

                where_clause = (
                    "WHERE " + " AND ".join(where_conditions)
                    if where_conditions
                    else ""
                )

                # Get total count for pagination
                count_query = f"SELECT COUNT(*) FROM race_metadata {where_clause}"
                cursor.execute(count_query, params)
                total_races = cursor.fetchone()[0]

                # Main query with optimization
                main_query = f"""
                SELECT 
                    race_id, venue, race_number, race_date, race_name,
                    grade, distance, field_size, winner_name, winner_odds,
                    winner_margin, url, extraction_timestamp, track_condition,
                    weather, temperature, humidity, data_quality_note
                FROM race_metadata 
                {where_clause}
                ORDER BY 
                    CASE 
                        WHEN extraction_timestamp IS NOT NULL THEN datetime(extraction_timestamp) 
                        ELSE datetime('1900-01-01') 
                    END DESC,
                    race_date DESC,
                    race_number DESC
                LIMIT ? OFFSET ?
                """

                params.extend([per_page, offset])
                cursor.execute(main_query, params)
                races = cursor.fetchall()

                result = {
                    "races": [self.format_race_data(race) for race in races],
                    "pagination": {
                        "current_page": page,
                        "per_page": per_page,
                        "total_races": total_races,
                        "total_pages": (total_races + per_page - 1) // per_page,
                        "has_more": (offset + per_page) < total_races,
                        "has_previous": page > 1,
                    },
                    "filters_applied": filters or {},
                }

                # Cache the result
                self.set_cache(cache_key, result)

                self.logger.info(
                    f"Retrieved {len(result['races'])} paginated races (page {page})"
                )
                return result

        except Exception as e:
            self.logger.error(f"Error getting paginated races: {e}")
            return {
                "races": [],
                "pagination": {
                    "current_page": 1,
                    "per_page": per_page,
                    "total_races": 0,
                    "total_pages": 0,
                    "has_more": False,
                    "has_previous": False,
                },
                "filters_applied": filters or {},
            }

    def get_race_details(self, race_id: str) -> Optional[Dict]:
        """Get detailed race information with optimization"""
        cache_key = self.get_cache_key("get_race_details", (race_id,))

        if self.is_cache_valid(cache_key):
            return self.get_cache(cache_key)

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Get race metadata with all available fields
                race_query = """
                SELECT * FROM race_metadata WHERE race_id = ?
                """
                cursor.execute(race_query, (race_id,))
                race_data = cursor.fetchone()

                if not race_data:
                    return None

                race_info = self.format_race_data(race_data)

                # Get dog data for this race with proper filtering
                dogs_query = """
                SELECT * FROM dog_race_data 
                WHERE race_id = ? 
                    AND dog_name IS NOT NULL 
                    AND dog_name != 'nan' 
                    AND dog_name != ''
                ORDER BY 
                    CASE WHEN box_number IS NOT NULL THEN box_number ELSE 999 END,
                    dog_name
                """
                cursor.execute(dogs_query, (race_id,))
                dogs_data = cursor.fetchall()

                dogs = [self.format_dog_data(dog) for dog in dogs_data]

                # Get enhanced expert data if available
                enhanced_query = """
                SELECT * FROM enhanced_expert_data 
                WHERE race_id = ?
                ORDER BY 
                    CASE WHEN position IS NOT NULL THEN position ELSE 999 END,
                    dog_clean_name
                """
                cursor.execute(enhanced_query, (race_id,))
                enhanced_data = cursor.fetchall()

                enhanced_dogs = [dict(row) for row in enhanced_data]

                result = {
                    "race_info": race_info,
                    "dogs": dogs,
                    "enhanced_data": enhanced_dogs,
                    "metadata": {
                        "total_dogs": len(dogs),
                        "has_enhanced_data": len(enhanced_dogs) > 0,
                        "has_winner": bool(race_info.get("winner_name")),
                        "data_quality": race_info.get("data_quality_note", "Unknown"),
                    },
                }

                # Cache the result
                self.set_cache(cache_key, result)

                self.logger.info(f"Retrieved race details for {race_id}")
                return result

        except Exception as e:
            self.logger.error(f"Error getting race details for {race_id}: {e}")
            return None

    def get_database_stats(self) -> Dict:
        """Get comprehensive database statistics with caching"""
        cache_key = self.get_cache_key("get_database_stats", ())

        if self.is_cache_valid(cache_key):
            return self.get_cache(cache_key)

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                stats = {}

                # Basic counts with optimization
                queries = {
                    "total_races": "SELECT COUNT(*) FROM race_metadata",
                    "completed_races": "SELECT COUNT(*) FROM race_metadata WHERE winner_name IS NOT NULL AND winner_name != '' AND winner_name != 'nan'",
                    "total_entries": "SELECT COUNT(*) FROM dog_race_data",
                    "unique_dogs": "SELECT COUNT(DISTINCT dog_clean_name) FROM dog_race_data WHERE dog_clean_name IS NOT NULL",
                    "venues": "SELECT COUNT(DISTINCT venue) FROM race_metadata WHERE venue IS NOT NULL",
                    "enhanced_records": "SELECT COUNT(*) FROM enhanced_expert_data",
                }

                for stat_name, query in queries.items():
                    cursor.execute(query)
                    stats[stat_name] = cursor.fetchone()[0]

                # Date ranges
                cursor.execute(
                    "SELECT MIN(race_date), MAX(race_date) FROM race_metadata WHERE race_date IS NOT NULL"
                )
                date_range = cursor.fetchone()
                stats["earliest_race_date"] = date_range[0]
                stats["latest_race_date"] = date_range[1]

                # Data quality metrics
                cursor.execute(
                    """
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN winner_name IS NOT NULL AND winner_name != '' AND winner_name != 'nan' THEN 1 ELSE 0 END) as with_winners,
                        SUM(CASE WHEN track_condition IS NOT NULL AND track_condition != '' THEN 1 ELSE 0 END) as with_conditions,
                        SUM(CASE WHEN weather IS NOT NULL AND weather != '' THEN 1 ELSE 0 END) as with_weather
                    FROM race_metadata
                """
                )
                quality_data = cursor.fetchone()

                if quality_data[0] > 0:
                    stats["data_completeness"] = {
                        "winner_completion_rate": round(
                            (quality_data[1] / quality_data[0]) * 100, 2
                        ),
                        "track_condition_rate": round(
                            (quality_data[2] / quality_data[0]) * 100, 2
                        ),
                        "weather_data_rate": round(
                            (quality_data[3] / quality_data[0]) * 100, 2
                        ),
                    }
                else:
                    stats["data_completeness"] = {
                        "winner_completion_rate": 0,
                        "track_condition_rate": 0,
                        "weather_data_rate": 0,
                    }

                # Recent activity
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM race_metadata 
                    WHERE extraction_timestamp >= datetime('now', '-7 days')
                """
                )
                stats["recent_races_7_days"] = cursor.fetchone()[0]

                # Cache the result
                self.set_cache(cache_key, stats)

                self.logger.info("Retrieved database statistics")
                return stats

        except Exception as e:
            self.logger.error(f"Error getting database stats: {e}")
            return {}

    def search_races(
        self, query: str, filters: Dict = None, limit: int = 50
    ) -> List[Dict]:
        """Search races with full-text search and filtering"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Build search conditions
                search_conditions = []
                params = []

                if query and query.strip():
                    # Search in multiple fields
                    search_term = f"%{query.strip().lower()}%"
                    search_conditions.append(
                        """
                        (LOWER(race_name) LIKE ? 
                         OR LOWER(venue) LIKE ? 
                         OR LOWER(winner_name) LIKE ?
                         OR LOWER(grade) LIKE ?)
                    """
                    )
                    params.extend([search_term, search_term, search_term, search_term])

                # Add filters
                if filters:
                    if filters.get("venue"):
                        search_conditions.append("venue = ?")
                        params.append(filters["venue"])

                    if filters.get("grade"):
                        search_conditions.append("grade = ?")
                        params.append(filters["grade"])

                    if filters.get("distance"):
                        search_conditions.append("distance = ?")
                        params.append(filters["distance"])

                where_clause = (
                    "WHERE " + " AND ".join(search_conditions)
                    if search_conditions
                    else ""
                )

                search_query = f"""
                SELECT 
                    race_id, venue, race_number, race_date, race_name,
                    grade, distance, field_size, winner_name, winner_odds,
                    winner_margin, url, extraction_timestamp, track_condition
                FROM race_metadata 
                {where_clause}
                ORDER BY 
                    CASE WHEN extraction_timestamp IS NOT NULL THEN datetime(extraction_timestamp) ELSE datetime('1900-01-01') END DESC
                LIMIT ?
                """

                params.append(limit)
                cursor.execute(search_query, params)
                races = cursor.fetchall()

                result = [self.format_race_data(race) for race in races]

                self.logger.info(
                    f"Search returned {len(result)} races for query: {query}"
                )
                return result

        except Exception as e:
            self.logger.error(f"Error searching races: {e}")
            return []

    def get_venue_statistics(self) -> Dict:
        """Get statistics grouped by venue"""
        cache_key = self.get_cache_key("get_venue_statistics", ())

        if self.is_cache_valid(cache_key):
            return self.get_cache(cache_key)

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                venue_query = """
                SELECT 
                    venue,
                    COUNT(*) as total_races,
                    SUM(CASE WHEN winner_name IS NOT NULL AND winner_name != '' AND winner_name != 'nan' THEN 1 ELSE 0 END) as completed_races,
                    AVG(field_size) as avg_field_size,
                    MIN(race_date) as first_race,
                    MAX(race_date) as last_race
                FROM race_metadata 
                WHERE venue IS NOT NULL
                GROUP BY venue
                ORDER BY total_races DESC
                """

                cursor.execute(venue_query)
                venues = cursor.fetchall()

                result = {}
                for venue in venues:
                    venue_dict = dict(venue)
                    venue_dict["completion_rate"] = (
                        round(
                            (venue_dict["completed_races"] / venue_dict["total_races"])
                            * 100,
                            2,
                        )
                        if venue_dict["total_races"] > 0
                        else 0
                    )
                    result[venue_dict["venue"]] = venue_dict

                # Cache the result
                self.set_cache(cache_key, result)

                self.logger.info(f"Retrieved statistics for {len(result)} venues")
                return result

        except Exception as e:
            self.logger.error(f"Error getting venue statistics: {e}")
            return {}

    def run_integrity_check(self) -> Dict:
        """Run integrity check using a thread-safe data integrity system"""
        try:
            # Create a new integrity manager for this thread to avoid SQLite threading issues
            from data_integrity_system import DataIntegrityManager

            thread_integrity_manager = DataIntegrityManager(self.db_path)

            with thread_integrity_manager:
                report = thread_integrity_manager.run_comprehensive_integrity_check()

            self.logger.info("Completed integrity check")
            return report
        except Exception as e:
            self.logger.error(f"Error running integrity check: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "statistics": {},
            }

    def create_backup(self) -> str:
        """Create database backup using the integrity system"""
        try:
            with self.integrity_manager:
                backup_path = self.integrity_manager.create_backup()

            self.logger.info(f"Created database backup: {backup_path}")
            return backup_path
        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
            raise

    def clear_cache(self):
        """Clear the internal cache"""
        self._cache.clear()
        self.logger.info("Cache cleared")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            "entries": len(self._cache),
            "timeout_seconds": self._cache_timeout,
            "oldest_entry": min(
                [entry["timestamp"] for entry in self._cache.values()], default=0
            ),
            "newest_entry": max(
                [entry["timestamp"] for entry in self._cache.values()], default=0
            ),
        }

    def get_training_data_stats(self) -> Dict:
        """Get training data statistics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                stats = {}

                # Basic training data stats
                cursor.execute(
                    "SELECT COUNT(*) FROM race_metadata WHERE winner_name IS NOT NULL"
                )
                stats["total_training_races"] = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM dog_race_data")
                stats["total_training_entries"] = cursor.fetchone()[0]

                # Recent data availability
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM race_metadata 
                    WHERE race_date >= date('now', '-30 days') 
                    AND winner_name IS NOT NULL
                """
                )
                stats["recent_races_30_days"] = cursor.fetchone()[0]

                cursor.execute(
                    """
                    SELECT COUNT(*) FROM race_metadata 
                    WHERE race_date >= date('now', '-90 days') 
                    AND winner_name IS NOT NULL
                """
                )
                stats["recent_races_90_days"] = cursor.fetchone()[0]

                # Data quality for training
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM race_metadata 
                    WHERE winner_name IS NOT NULL 
                    AND track_condition IS NOT NULL
                    AND distance IS NOT NULL
                """
                )
                stats["high_quality_training_races"] = cursor.fetchone()[0]

                return stats

        except Exception as e:
            self.logger.error(f"Error getting training data stats: {e}")
            return {}

    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """Execute a custom SQL query and return results"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)

                results = cursor.fetchall()
                return [dict(row) for row in results]

        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            return []


# Factory function for easy integration
def create_enhanced_database_manager(
    db_path: str = "greyhound_racing_data.db",
) -> EnhancedDatabaseManager:
    """Factory function to create an enhanced database manager"""
    return EnhancedDatabaseManager(db_path)
