import json
import logging
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)


class FastTrackDBAdapter:
    """
    Adapter for inserting FastTrack data into the SQLite database.
    """

    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path
        self.conn = None

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()

    def _get_existing_id(self, table: str, where_col: str, where_val) -> Optional[int]:
        """Helper to get an existing ID from a table."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                f"SELECT id FROM {table} WHERE {where_col} = ?", (where_val,)
            )
            result = cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Error checking for existing ID in {table}: {e}")
            return None

    def adapt_and_load_race(self, race_data: Dict[str, Any]):
        """
        Adapts and loads a single scraped race and its associated dog performances into the database.
        """
        try:
            # Get the main race_id from the base race_metadata table
            # This assumes that the base race data has already been scraped
            race_id = self._get_existing_id(
                "race_metadata", "race_name", race_data.get("race_name")
            )
            if not race_id:
                logger.warning(f"Base race not found for: {race_data.get('race_name')}")
                return

            # Map and insert the race_ft_extra record
            race_extra = {
                "race_id": race_id,
                "track_rating": race_data.get("track", {}).get("rating"),
                "weather_description": race_data.get("weather", {}).get("description"),
                "race_comment": race_data.get("race_comment"),
            }
            self._insert_or_update("races_ft_extra", race_extra, {"race_id": race_id})

            # Map and insert dog performance data
            for dog_perf in race_data.get("results", []):
                # Get dog_id from the dogs table
                dog_id = self._get_existing_id(
                    "dogs", "dog_name", dog_perf.get("dog_name")
                )
                if not dog_id:
                    logger.warning(f"Dog not found: {dog_perf.get('dog_name')}")
                    continue

                # Get performance_id from the dog_performances table
                performance_id = self._get_existing_performance_id(race_id, dog_id)
                if not performance_id:
                    logger.warning(
                        f"Performance not found for race {race_id}, dog {dog_id}"
                    )
                    continue

                perf_extra = {
                    "performance_id": performance_id,
                    "pir_rating": dog_perf.get("pir"),
                    "split_1_time": dog_perf.get("split1"),
                    "run_home_time": dog_perf.get("run_home"),
                    "beaten_margin": dog_perf.get("margin"),
                    "in_race_comment": dog_perf.get("comment"),
                    "fixed_odds_sp": dog_perf.get("sp"),
                }
                self._insert_or_update(
                    "dog_performance_ft_extra",
                    perf_extra,
                    {"performance_id": performance_id},
                )

            self.conn.commit()
            logger.info(f"Successfully loaded FastTrack data for race {race_id}")

        except Exception as e:
            logger.error(f"Error loading race data: {e}")
            self.conn.rollback()

    def _get_existing_performance_id(self, race_id: int, dog_id: int) -> Optional[int]:
        """Get existing performance ID from dog_performances table."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT id FROM dog_performances WHERE race_id = ? AND dog_id = ?",
                (race_id, dog_id),
            )
            result = cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Error getting performance ID: {e}")
            return None

    def _insert_or_update(
        self, table: str, data: Dict[str, Any], lookup: Dict[str, Any]
    ):
        """Insert a record or update it if it exists, based on a lookup dictionary."""
        cursor = self.conn.cursor()

        # Build lookup clause
        lookup_cols = list(lookup.keys())
        lookup_vals = list(lookup.values())
        where_clause = " AND ".join([f"{col} = ?" for col in lookup_cols])

        # Check if record exists
        cursor.execute(f"SELECT id FROM {table} WHERE {where_clause}", lookup_vals)
        existing = cursor.fetchone()

        if existing:
            # Update existing record
            update_cols = list(data.keys())
            set_clause = ", ".join([f"{col} = ?" for col in update_cols])
            update_vals = list(data.values()) + lookup_vals

            cursor.execute(
                f"UPDATE {table} SET {set_clause} WHERE {where_clause}", update_vals
            )
            logger.debug(f"Updated record in {table} where {where_clause}")
        else:
            # Insert new record
            all_data = {**data, **lookup}
            cols = ", ".join(all_data.keys())
            placeholders = ", ".join(["?" for _ in all_data])
            vals = list(all_data.values())

            cursor.execute(
                f"INSERT INTO {table} ({cols}) VALUES ({placeholders})", vals
            )
            logger.debug(f"Inserted new record into {table}")
