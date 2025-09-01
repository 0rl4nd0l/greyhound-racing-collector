#!/usr/bin/env python3
"""
CLI to stage CSV-embedded dog histories and upsert into canonical tables.

Usage:
    python scripts/ingest_csv_history.py --csv "Race 7 - MURR - 2025-08-24.csv" --db "$GREYHOUND_DB_PATH"

Behavior:
- Parses the CSV (robust to multiple header/date formats, venue slash normalization)
- Ensures staging tables exist
- Writes race- and dog-level records to staging tables
- Performs safe upserts into:
  - race_metadata (ON CONFLICT(race_id) DO UPDATE ... with COALESCE)
  - dog_race_data (ON CONFLICT(race_id, dog_clean_name, box_number) DO UPDATE ...)
- Preserves raw row JSON in dog_race_data.form_guide_json for audit

Notes:
- If --db is not provided, uses GREYHOUND_DB_PATH env var, else falls back to 'greyhound_racing_data.db'
- Designed to align with existing schema (see current_schema.sql)
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from typing import List, Dict, Optional
from scripts.db_utils import open_sqlite_writable

from ingestion.staging_writer import parse_race_csv_for_staging, RaceMeta
from scripts.db_guard import db_guard


CREATE_STAGING_SQL = {
    "csv_race_metadata_staging": """
    CREATE TABLE IF NOT EXISTS csv_race_metadata_staging (
        race_id TEXT PRIMARY KEY,
        venue TEXT,
        race_number INTEGER,
        race_date TEXT,
        race_name TEXT,
        grade TEXT,
        distance TEXT,
        field_size INTEGER,
        extraction_timestamp TEXT,
        data_source TEXT
    );
    """,
    "csv_dog_history_staging": """
    CREATE TABLE IF NOT EXISTS csv_dog_history_staging (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        race_id TEXT,
        venue TEXT,
        race_number INTEGER,
        race_date TEXT,
        dog_name TEXT,
        dog_clean_name TEXT,
        box_number INTEGER,
        finish_position INTEGER,
        weight REAL,
        starting_price REAL,
        individual_time TEXT,
        sectional_1st TEXT,
        margin REAL,
        trainer_name TEXT,
        extraction_timestamp TEXT,
        data_source TEXT,
        raw_row_json TEXT,
        UNIQUE(race_id, dog_clean_name, box_number)
    );
    """,
}


def ensure_staging_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    for sql in CREATE_STAGING_SQL.values():
        cur.executescript(sql)
    conn.commit()


def upsert_race_metadata(conn: sqlite3.Connection, meta: RaceMeta, field_size: int) -> None:
    cur = conn.cursor()
    # Insert staging
    cur.execute(
        """
        INSERT OR REPLACE INTO csv_race_metadata_staging (
            race_id, venue, race_number, race_date, race_name, grade, distance, field_size, extraction_timestamp, data_source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), 'csv_stage')
        """,
        (
            meta.race_id,
            meta.venue,
            meta.race_number,
            meta.race_date,
            meta.race_name,
            meta.grade,
            meta.distance,
            field_size,
        ),
    )

    # Upsert into canonical table; COALESCE to avoid overwriting non-null with null
    cur.execute(
        """
        INSERT INTO race_metadata (
            race_id, venue, race_number, race_date, race_name, grade, distance, field_size, extraction_timestamp, data_source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), 'csv_stage')
        ON CONFLICT(race_id) DO UPDATE SET
            venue              = excluded.venue,
            race_number        = excluded.race_number,
            race_date          = excluded.race_date,
            race_name          = COALESCE(excluded.race_name, race_metadata.race_name),
            grade              = COALESCE(excluded.grade, race_metadata.grade),
            distance           = COALESCE(excluded.distance, race_metadata.distance),
            field_size         = excluded.field_size,
            extraction_timestamp = excluded.extraction_timestamp,
            data_source        = COALESCE(excluded.data_source, race_metadata.data_source)
        """,
        (
            meta.race_id,
            meta.venue,
            meta.race_number,
            meta.race_date,
            meta.race_name,
            meta.grade,
            meta.distance,
            field_size,
        ),
    )
    conn.commit()


def upsert_dogs(conn: sqlite3.Connection, dogs: List[Dict[str, object]]) -> None:
    if not dogs:
        return
    cur = conn.cursor()

    # Stage all dogs (insert or replace for staging consistency)
    cur.executemany(
        """
        INSERT OR REPLACE INTO csv_dog_history_staging (
            race_id, venue, race_number, race_date, dog_name, dog_clean_name, box_number, finish_position,
            weight, starting_price, individual_time, sectional_1st, margin, trainer_name, extraction_timestamp,
            data_source, raw_row_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                d.get("race_id"),
                d.get("venue"),
                d.get("race_number"),
                d.get("race_date"),
                d.get("dog_name"),
                d.get("dog_clean_name"),
                d.get("box_number"),
                d.get("finish_position"),
                d.get("weight"),
                d.get("starting_price"),
                d.get("individual_time"),
                d.get("sectional_1st"),
                d.get("margin"),
                d.get("trainer_name"),
                d.get("extraction_timestamp"),
                d.get("data_source"),
                d.get("raw_row_json"),
            )
            for d in dogs
        ],
    )

    # Upsert into canonical dog_race_data
    cur.executemany(
        """
        INSERT INTO dog_race_data (
            race_id, dog_name, dog_clean_name, box_number, finish_position, weight, starting_price,
            individual_time, sectional_1st, margin, extraction_timestamp, data_source, form_guide_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(race_id, dog_clean_name, box_number) DO UPDATE SET
            finish_position      = COALESCE(excluded.finish_position, dog_race_data.finish_position),
            weight               = COALESCE(excluded.weight, dog_race_data.weight),
            starting_price       = COALESCE(excluded.starting_price, dog_race_data.starting_price),
            individual_time      = COALESCE(excluded.individual_time, dog_race_data.individual_time),
            sectional_1st        = COALESCE(excluded.sectional_1st, dog_race_data.sectional_1st),
            margin               = COALESCE(excluded.margin, dog_race_data.margin),
            extraction_timestamp = excluded.extraction_timestamp,
            data_source          = COALESCE(excluded.data_source, dog_race_data.data_source),
            form_guide_json      = COALESCE(excluded.form_guide_json, dog_race_data.form_guide_json)
        """,
        [
            (
                d.get("race_id"),
                d.get("dog_name"),
                d.get("dog_clean_name"),
                d.get("box_number"),
                d.get("finish_position"),
                d.get("weight"),
                d.get("starting_price"),
                d.get("individual_time"),
                d.get("sectional_1st"),
                d.get("margin"),
                d.get("extraction_timestamp"),
                d.get("data_source"),
                d.get("raw_row_json"),
            )
            for d in dogs
        ],
    )

    conn.commit()


def pick_db_path(cli_db: Optional[str]) -> str:
    if cli_db:
        return cli_db
    # Prefer staging DB for writers
    env_db = os.getenv("STAGING_DB_PATH") or os.getenv("GREYHOUND_DB_PATH")
    if env_db and env_db.strip():
        return env_db
    return "greyhound_racing_data_stage.db"


def main():
    ap = argparse.ArgumentParser(description="Stage CSV dog histories and upsert into database")
    ap.add_argument("--csv", required=True, help="Path to race CSV (e.g., 'Race 7 - MURR - 2025-08-24.csv')")
    ap.add_argument("--db", required=False, help="Path to SQLite DB (defaults to $GREYHOUND_DB_PATH or greyhound_racing_data.db)")
    args = ap.parse_args()

    db_path = pick_db_path(args.db)

    # Parse
    meta, dogs = parse_race_csv_for_staging(args.csv)

    # Guarded DB operation (pre-backup, post-validate)
    with db_guard(db_path=db_path, label="ingest_csv_history") as guard:
        guard.expect_table_growth("race_metadata", min_delta=0)
        guard.expect_table_growth("dog_race_data", min_delta=0)
        # Connect and stage/upsert (use writable/staging DB)
        conn = open_sqlite_writable(db_path)
        try:
            ensure_staging_tables(conn)
            upsert_race_metadata(conn, meta, field_size=len(dogs))
            upsert_dogs(conn, dogs)
            print(f"✅ Ingested {args.csv}: race_id={meta.race_id}, dogs={len(dogs)} -> DB={db_path}")
        finally:
            conn.close()


if __name__ == "__main__":
    main()

