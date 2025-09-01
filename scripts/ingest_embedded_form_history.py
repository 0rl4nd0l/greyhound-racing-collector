#!/usr/bin/env python3
"""
Ingest embedded historical rows from a race CSV (form guide style) into dog_race_data.

Context
- Some upcoming race CSVs include each participant followed by blank-name rows that
  encode that dog's historical runs (DATE, TRACK, DIST, PLC, TIME, etc.).
- The UI derives historical stats from dog_race_data (see app.py _derive_stats_from_db),
  not from dog_performances, so we write into dog_race_data.
- We intentionally DO NOT write to race_metadata here to avoid polluting training joins.
  Synthetic race_ids are generated per (venue_code, date, distance, grade) to group rows.

Usage
  python scripts/ingest_embedded_form_history.py --csv "processed/excluded/Race 7 - DARW - 2025-08-24.csv" \
      --db greyhound_racing_data.db

Notes
- Idempotent-ish: before insert, checks if a row with (race_id, dog_clean_name) exists.
- Name normalization: Title Case for display consistency; a separate normalized key is
  also inserted via Python-side cleaning to match app.py lookups.
- Venue normalization uses config.venue_mapping.normalize_venue.

Limitations
- race_id is synthetic as form guides typically omit race numbers for historical rows.
  This is OK for UI history (no race_metadata joins). We avoid using this for training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sqlite3
from dataclasses import dataclass
from scripts.db_utils import open_sqlite_writable
from datetime import datetime
from typing import Dict, List, Optional

from scripts.db_guard import db_guard

# Local import for venue normalization
try:
    from config.venue_mapping import normalize_venue
except Exception:
    # Fallback: basic normalization
    def normalize_venue(v: str) -> str:
        v = (v or "").strip().upper()
        v = v.replace("/", "_")
        return re.sub(r"[^A-Z0-9_\-]", "", v)


@dataclass
class HistoryRow:
    dog_clean_name: str
    venue: str
    race_date: str
    distance: Optional[int]
    grade: Optional[str]
    finish_position: Optional[int]
    individual_time: Optional[float]
    weight: Optional[float]
    sectional_1st: Optional[float]
    margin: Optional[float]
    starting_price: Optional[float]


def _to_int(val) -> Optional[int]:
    try:
        if val is None:
            return None
        s = str(val).strip()
        if s == "":
            return None
        return int(float(s))
    except Exception:
        return None


def _to_float(val) -> Optional[float]:
    try:
        if val is None:
            return None
        s = str(val).strip().replace("s", "")
        if s == "":
            return None
        f = float(s)
        return f
    except Exception:
        return None


def _parse_date(val: str) -> Optional[str]:
    if not val:
        return None
    s = str(val).strip()
    # try ISO first
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    # already YYYY-MM-DD?
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s
    return None


def _clean_participant_name(raw: str) -> str:
    s = (raw or "").strip().replace('"', '')
    # Remove leading numeric index like "1. Name"
    if "." in s and s.split(".")[0].strip().isdigit():
        s = s.split(".", 1)[1].strip()
    # Collapse whitespace, title case for display consistency
    s = re.sub(r"\s+", " ", s).strip()
    return s.title()


def _norm_key_for_lookup(name: str) -> str:
    # Mirror app.py normalization logic used in SQL layer
    return re.sub(r"[^A-Za-z0-9]", "", (name or "").strip()).upper()


def parse_embedded_history(csv_path: str) -> List[HistoryRow]:
    rows: List[HistoryRow] = []
    current_dog: Optional[str] = None

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        # Expect columns: Dog Name, PLC, BOX, WGT, DIST, DATE, TRACK, G, TIME, 1 SEC, MGN, SP, etc.
        for raw in reader:
            dog_name = (raw.get("Dog Name") or raw.get("dog_name") or "").strip()
            if dog_name and not dog_name == "\"\"":
                # Participant row
                cleaned = _clean_participant_name(dog_name)
                if cleaned:
                    current_dog = cleaned
                else:
                    current_dog = None
                continue

            # Historical row for current dog
            if not current_dog:
                continue

            date = _parse_date(raw.get("DATE"))
            track = normalize_venue(raw.get("TRACK") or raw.get("Venue") or raw.get("venue") or "")
            dist = _to_int(raw.get("DIST"))
            grade = (raw.get("G") or raw.get("Grade") or "").strip().upper() or None
            plc = _to_int(raw.get("PLC") or raw.get("Plc"))
            t = _to_float(raw.get("TIME"))
            wgt = _to_float(raw.get("WGT") or raw.get("Weight"))
            sec1 = _to_float(raw.get("1 SEC") or raw.get("First Sectional") or raw.get("sectional_1st"))
            mgn = _to_float(raw.get("MGN") or raw.get("Margin"))
            sp = _to_float(raw.get("SP") or raw.get("Starting Price") or raw.get("Odds Decimal") or raw.get("odds_decimal"))

            if not date or not track:
                # Need at least venue/date to build a synthetic race_id; skip otherwise
                continue

            rows.append(
                HistoryRow(
                    dog_clean_name=current_dog,
                    venue=track,
                    race_date=date,
                    distance=dist,
                    grade=grade,
                    finish_position=plc,
                    individual_time=t,
                    weight=wgt,
                    sectional_1st=sec1,
                    margin=mgn,
                    starting_price=sp,
                )
            )
    return rows


def make_synthetic_race_id(hr: HistoryRow) -> str:
    # Build a stable race_id from venue, date, distance, and grade (no race number available)
    base = f"{hr.venue}_{hr.race_date}"
    if hr.distance:
        base += f"_{int(hr.distance)}m"
    if hr.grade:
        base += f"_{hr.grade.replace('/', '_')}"
    # Keep base short and safe
    return base


def ensure_tables(conn: sqlite3.Connection) -> None:
    # dog_race_data exists in this repo, but be defensive in case of custom DBs
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS dog_race_data (
            id INTEGER PRIMARY KEY,
            race_id TEXT,
            dog_name TEXT,
            dog_clean_name TEXT,
            box_number INTEGER,
            trainer_name TEXT,
            weight REAL,
            starting_price REAL,
            individual_time REAL,
            sectional_1st REAL,
            margin REAL,
            finish_position INTEGER,
            extraction_timestamp TEXT,
            data_source TEXT
        )
        """
    )
    conn.commit()


def upsert_embedded_history(db_path: str, csv_path: str) -> Dict[str, int]:
    rows = parse_embedded_history(csv_path)
    inserted = 0
    skipped = 0

    if not rows:
        return {"inserted": 0, "skipped": 0}

    conn = open_sqlite_writable(db_path)
    try:
        ensure_tables(conn)
        cur = conn.cursor()
        for hr in rows:
            dog_name = hr.dog_clean_name
            dog_title = dog_name  # already title-cased
            dog_norm = _norm_key_for_lookup(dog_title)

            race_id = make_synthetic_race_id(hr)

            # Existence check: avoid duplicates on (race_id, dog_clean_name or dog_name)
            # Keep the check simple to avoid SQLite string-normalization pitfalls.
            cur.execute(
                "SELECT 1 FROM dog_race_data WHERE race_id=? AND (dog_clean_name=? OR dog_name=?) LIMIT 1",
                (race_id, dog_title, dog_title),
            )
            if cur.fetchone():
                skipped += 1
                continue

            cur.execute(
                """
                INSERT INTO dog_race_data (
                    race_id, dog_name, dog_clean_name, box_number, trainer_name, weight, starting_price,
                    individual_time, sectional_1st, margin, finish_position, extraction_timestamp, data_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    race_id,
                    dog_title,
                    dog_title,
                    None,
                    None,
                    hr.weight,
                    hr.starting_price,
                    hr.individual_time,
                    hr.sectional_1st,
                    hr.margin,
                    hr.finish_position,
                    datetime.utcnow().isoformat(timespec="seconds"),
                    "embedded_form_guide",
                ),
            )
            inserted += 1
        conn.commit()
    finally:
        conn.close()

    return {"inserted": inserted, "skipped": skipped}


def main():
    ap = argparse.ArgumentParser(description="Ingest embedded historical rows into dog_race_data")
    ap.add_argument("--csv", required=True, help="Path to race CSV file (with embedded historical rows)")
    ap.add_argument("--db", required=False, help="SQLite DB path (defaults to $GREYHOUND_DB_PATH or greyhound_racing_data.db)")
    args = ap.parse_args()

    # Prefer staging DB for writers
    db_path = args.db or os.getenv("STAGING_DB_PATH") or os.getenv("GREYHOUND_DB_PATH") or "greyhound_racing_data_stage.db"
    # Guarded write (pre-backup, post-validate)
    with db_guard(db_path=db_path, label="ingest_embedded_form_history") as guard:
        guard.expect_table_growth("dog_race_data", min_delta=0)
        stats = upsert_embedded_history(db_path, args.csv)
        print(f"✅ Ingested embedded history from {args.csv} -> DB={db_path} | inserted={stats['inserted']} skipped={stats['skipped']}")


if __name__ == "__main__":
    main()
