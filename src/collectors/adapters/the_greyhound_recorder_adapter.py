#!/usr/bin/env python3
"""
Database adapter for The Greyhound Recorder data.
"""

import sqlite3
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class TheGreyhoundRecorderDBAdapter:
    def __init__(self, db_path: str = 'greyhound_racing_data.db'):
        self.db_path = db_path
        self.conn = None

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
            
    def adapt_and_load_meeting(self, meeting_data: Dict[str, Any]):
        """Adapts and loads a single meeting into the database."""
        try:
            cursor = self.conn.cursor()
            
            # Create a basic race_metadata record if it doesn't exist
            # This assumes that the primary data source may have missed this race
            cursor.execute(
                "INSERT OR IGNORE INTO race_metadata (race_name, venue, race_date) VALUES (?, ?, ?)",
                (meeting_data.get('meeting_title'), meeting_data.get('venue'), meeting_data.get('date'))
            )
            
            # Get the race_id
            cursor.execute("SELECT id FROM race_metadata WHERE race_name = ?", (meeting_data.get('meeting_title'),))
            race_id = cursor.fetchone()[0]
            
            # Create the recorder_ft_extra record
            recorder_extra = {
                'race_id': race_id,
                'meeting_id': meeting_data.get('meeting_id'),
                'long_form_url': meeting_data.get('long_form_url'),
                'short_form_url': meeting_data.get('short_form_url'),
                'fields_url': meeting_data.get('fields_url'),
                'data_source': 'the_greyhound_recorder'
            }
            
            self._insert_or_update('races_gr_extra', recorder_extra, {'race_id': race_id})
            
            self.conn.commit()
            logger.info(f"Successfully loaded The Greyhound Recorder data for meeting {meeting_data.get('meeting_title')}")

        except Exception as e:
            logger.error(f"Error loading meeting data: {e}")
            self.conn.rollback()
            
    def _insert_or_update(self, table: str, data: Dict[str, Any], lookup: Dict[str, Any]):
        """Insert a record or update it if it exists, based on a lookup dictionary."""
        cursor = self.conn.cursor()

        lookup_cols = list(lookup.keys())
        lookup_vals = list(lookup.values())
        where_clause = " AND ".join([f"{col} = ?" for col in lookup_cols])

        cursor.execute(f"SELECT id FROM {table} WHERE {where_clause}", lookup_vals)
        existing = cursor.fetchone()

        if existing:
            update_cols = list(data.keys())
            set_clause = ", ".join([f"{col} = ?" for col in update_cols])
            update_vals = list(data.values()) + lookup_vals
            
            cursor.execute(f"UPDATE {table} SET {set_clause} WHERE {where_clause}", update_vals)
            logger.debug(f"Updated record in {table} where {where_clause}")
        else:
            all_data = {**data, **lookup}
            cols = ", ".join(all_data.keys())
            placeholders = ", ".join(["?" for _ in all_data])
            vals = list(all_data.values())
            
            cursor.execute(f"INSERT INTO {table} ({cols}) VALUES ({placeholders})", vals)
            logger.debug(f"Inserted new record into {table}")

