#!/usr/bin/env python3
"""
Populate race_metadata from upcoming CSVs.

- Scans UPCOMING_RACES_DIR for race CSVs
- Extracts race_date, venue code, race_number using robust parsers
- Computes canonical race_id = VENUECODE_YYYY-MM-DD_RN (code, not long name)
- Upserts minimal metadata into race_metadata so downstream joins (predictions ↔ odds) resolve

Safety and routing
- Writes go via scripts.db_utils.open_sqlite_writable (respects STAGING_DB_PATH or GREYHOUND_DB_PATH)
- Idempotent: uses ON CONFLICT(race_id) DO UPDATE with conservative field updates

Usage examples
  # Default UPCOMING_RACES_DIR and staging DB
  python scripts/populate_race_metadata_from_upcoming.py

  # Custom directory and DB
  UPCOMING_RACES_DIR=./upcoming_races_temp \
  STAGING_DB_PATH=./greyhound_racing_data_writable.db \
  python scripts/populate_race_metadata_from_upcoming.py --limit 200

Notes
- Keeps venue field in DB as a human-friendly standardized name (e.g., BALLARAT) to align with API overlay
- Canonical race_id uses venue CODE to be consistent with internal pipelines
"""
from __future__ import annotations

import argparse
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
import re

# Routing helpers
try:
    from scripts.db_utils import open_sqlite_writable
except Exception:  # fallback
    def open_sqlite_writable(db_path: str | None = None):
        import os as _os, sqlite3 as _sqlite3
        path = db_path or _os.getenv("STAGING_DB_PATH") or _os.getenv("GREYHOUND_DB_PATH") or "greyhound_racing_data.db"
        return _sqlite3.connect(str(Path(path).resolve()))

# Metadata parsers (prefer ingestion.staging_writer for robust code/number/date)
try:
    from ingestion.staging_writer import (
        sniff_dialect_and_headers as _sniff_dialect_and_headers,
        extract_meta_from_csv as _extract_meta_from_csv,
        extract_meta_from_filename as _extract_meta_from_filename,
    )
except Exception as _e:
    _sniff_dialect_and_headers = None
    _extract_meta_from_csv = None
    _extract_meta_from_filename = None

# Lightweight enrichment (distance/grade/field_size + venue long-name)
try:
    from utils.csv_metadata import parse_race_csv_meta as _parse_race_csv_meta
    from utils.csv_metadata import standardize_venue_name as _std_venue
except Exception:
    _parse_race_csv_meta = None
    def _std_venue(v: str) -> str:
        return (v or "").strip().upper()


@dataclass
class ParsedMeta:
    race_date: str   # YYYY-MM-DD
    venue_code: str  # e.g., BAL, AP_K (uppercase code form)
    race_number: int

    @property
    def race_id(self) -> str:
        return f"{self.venue_code}_{self.race_date}_{self.race_number}"


def _parse_human_date_str(s: str) -> Optional[str]:
    """Parse strings like '25 August 2025', '25 Aug 2025', or 'Mon 25 Aug 2025' -> 'YYYY-MM-DD'."""
    try:
        s2 = s.strip().replace("_", " ")
        # Strip leading day-of-week tokens like 'Mon', 'Monday', possibly with a comma
        s2 = re.sub(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+",
                    "", s2, flags=re.IGNORECASE)
        for fmt in ("%d %B %Y", "%d %b %Y"):
            try:
                return datetime.strptime(s2, fmt).strftime("%Y-%m-%d")
            except Exception:
                continue
        # If already ISO
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s2):
            return s2
    except Exception:
        pass
    return None


def _extract_meta(path: Path) -> Optional[ParsedMeta]:
    """Extract race_date, venue code, race_number from CSV or filename.
    Strategy (best-effort):
      1) ingestion.staging_writer: CSV headers → fallback to filename
      2) utils.csv_metadata: lightweight parse (may return human venue)
      3) Raw filename regex: "Race N - VENUE - YYYY-MM-DD" OR human dates
    """
    try:
        meta = None
        # 1) Use robust staging_writer helpers when available
        if _sniff_dialect_and_headers and _extract_meta_from_csv and _extract_meta_from_filename:
            try:
                # Prefer filename parse first (cheap and stable for our naming convention)
                fm = _extract_meta_from_filename(path)
                if not fm:
                    dialect, _headers = _sniff_dialect_and_headers(path)
                    fm = _extract_meta_from_csv(path, dialect)
                meta = fm
            except Exception:
                meta = None
        # 2) csv_metadata fallback (may return long-form venue). Prefer filename date if both available.
        filename_date = None
        try:
            # ISO date
            m = re.search(r"Race\s*(?P<rno>\d+)\s*[-_]\s*(?P<venue>[A-Za-z0-9_\-/]+)\s*[-_]\s*(?P<date>\d{4}-\d{2}-\d{2})", path.name, re.IGNORECASE)
            if m:
                filename_date = m.group("date")
            # Human date (e.g., 25 August 2025 / 25 Aug 2025)
            if not filename_date:
                hm = re.search(r"Race\s*(?P<rno>\d+)\s*[-_]\s*(?P<venue>[A-Za-z0-9_\-/]+)\s*[-_]\s*(?P<hdate>\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})", path.name, re.IGNORECASE)
                if hm:
                    parsed = _parse_human_date_str(hm.group("hdate"))
                    if parsed:
                        filename_date = parsed
        except Exception:
            filename_date = None
        if meta is None and _parse_race_csv_meta is not None:
            info = _parse_race_csv_meta(str(path))
            if isinstance(info, dict) and info.get("status") == "success":
                race_date = str(info.get("race_date") or "").strip()
                venue = str(info.get("venue") or "").strip().upper()
                race_number = info.get("race_number")
                # Prefer the date from filename when present
                if filename_date:
                    race_date = filename_date
                if race_date and venue and race_number:
                    return ParsedMeta(race_date=race_date, venue_code=venue.replace("/", "_"), race_number=int(race_number))
        # 3) Raw filename regex fallback (ISO or human date)
        try:
            # ISO date
            m = re.search(r"Race\s*(?P<rno>\d+)\s*[-_]\s*(?P<venue>[A-Za-z0-9_\-/]+)\s*[-_]\s*(?P<date>\d{4}-\d{2}-\d{2})", path.name, re.IGNORECASE)
            if m:
                rno = int(m.group("rno"))
                venue = m.group("venue").replace("/", "_").upper()
                date = m.group("date")
                return ParsedMeta(race_date=date, venue_code=venue, race_number=rno)
            # Human date
            hm = re.search(r"Race\s*(?P<rno>\d+)\s*[-_]\s*(?P<venue>[A-Za-z0-9_\-/]+)\s*[-_]\s*(?P<hdate>\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})", path.name, re.IGNORECASE)
            if hm:
                rno = int(hm.group("rno"))
                venue = hm.group("venue").replace("/", "_").upper()
                hdate = _parse_human_date_str(hm.group("hdate"))
                if hdate:
                    return ParsedMeta(race_date=hdate, venue_code=venue, race_number=rno)
        except Exception:
            pass
        if meta is None:
            return None
        # ingestion.staging_writer RaceMeta has .race_date, .venue (code), .race_number
        return ParsedMeta(race_date=meta.race_date, venue_code=str(meta.venue).upper(), race_number=int(meta.race_number))
    except Exception:
        return None


def upsert_race_meta(conn: sqlite3.Connection, pm: ParsedMeta, enrich: Dict[str, object]) -> Tuple[bool, bool]:
    """Upsert minimal race_metadata. Returns (inserted, updated)."""
    # Human-friendly venue for joins (align with API overlay and persistence script expectations)
    venue_name = _std_venue(pm.venue_code)

    # Prepare minimal columns
    cols = {
        "race_id": pm.race_id,
        "venue": venue_name,  # human-friendly name for easier joins
        "race_number": pm.race_number,
        "race_date": pm.race_date,
        "grade": (enrich.get("grade") if enrich else None),
        "distance": (str(enrich.get("distance")) if enrich and enrich.get("distance") else None),
        "field_size": (int(enrich.get("field_size")) if enrich and isinstance(enrich.get("field_size"), (int, float)) else None),
        "extraction_timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_source": "upcoming_csv",
    }

    # Build INSERT with ON CONFLICT(race_id) DO UPDATE for selected mutable fields
    cur = conn.cursor()
    inserted = False
    updated = False
    try:
        cur.execute(
            """
            INSERT INTO race_metadata (
                race_id, venue, race_number, race_date, grade, distance, field_size, extraction_timestamp, data_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(race_id) DO UPDATE SET
                venue = COALESCE(excluded.venue, race_metadata.venue),
                race_number = COALESCE(excluded.race_number, race_metadata.race_number),
                race_date = COALESCE(excluded.race_date, race_metadata.race_date),
                grade = COALESCE(excluded.grade, race_metadata.grade),
                distance = COALESCE(excluded.distance, race_metadata.distance),
                field_size = COALESCE(excluded.field_size, race_metadata.field_size),
                extraction_timestamp = excluded.extraction_timestamp,
                data_source = COALESCE(race_metadata.data_source, excluded.data_source)
            """,
            (
                cols["race_id"],
                cols["venue"],
                cols["race_number"],
                cols["race_date"],
                cols["grade"],
                cols["distance"],
                cols["field_size"],
                cols["extraction_timestamp"],
                cols["data_source"],
            ),
        )
        # Detect insert vs update via changes()
        try:
            # lastrowid is not reliable on UPSERT; use changes() heuristic
            ch = cur.execute("SELECT changes()").fetchone()[0]
            if ch == 1:
                inserted = True
            elif ch == 0:
                # No-op
                pass
            else:
                updated = True
        except Exception:
            pass
        return inserted, updated
    finally:
        cur.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Populate race_metadata from upcoming CSVs")
    ap.add_argument("--dir", dest="upcoming_dir", default=os.getenv("UPCOMING_RACES_DIR", "./upcoming_races_temp"))
    ap.add_argument("--db", dest="db_path", default=os.getenv("STAGING_DB_PATH") or os.getenv("GREYHOUND_DB_PATH") or None)
    ap.add_argument("--limit", type=int, default=0, help="Limit number of files processed (0 = no limit)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ns = ap.parse_args()

    upc = Path(ns.upcoming_dir).expanduser().resolve()
    if not upc.exists() or not upc.is_dir():
        print(f"❌ Upcoming dir not found: {upc}")
        return 2

    # Gather candidate CSVs
    files = [p for p in upc.iterdir() if p.is_file() and p.suffix.lower() == ".csv" and not p.name.startswith(".")]
    files.sort()
    if ns.limit and ns.limit > 0:
        files = files[: ns.limit]

    # Open DB connection via routing (writeable)
    conn = open_sqlite_writable(ns.db_path)
    try:
        # PRAGMAs for safer, faster batch upsert
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        except Exception:
            pass

        inserted = 0
        updated = 0
        skipped = 0
        errors = 0

        for p in files:
            # Extract core meta
            pm = _extract_meta(p)
            if pm is None:
                skipped += 1
                if ns.verbose:
                    print(f"[skip] Unable to extract meta from {p.name} (no CSV/filename match)")
                continue

            # Enrich with distance/grade/field_size when available
            enrich: Dict[str, object] = {}
            try:
                if _parse_race_csv_meta is not None:
                    info = _parse_race_csv_meta(str(p))
                    if isinstance(info, dict) and info.get("status") == "success":
                        # Only copy selected keys if present
                        for k in ("distance", "grade", "field_size"):
                            v = info.get(k)
                            if v is not None and v != "Unknown":
                                enrich[k] = v
            except Exception:
                pass

            if ns.dry_run:
                if ns.verbose:
                    print(f"[dry-run] Would upsert: race_id={pm.race_id} venue_code={pm.venue_code} date={pm.race_date} R{pm.race_number} enrich={enrich}")
                continue

            try:
                ins, upd = upsert_race_meta(conn, pm, enrich)
                if ins:
                    inserted += 1
                    if ns.verbose:
                        print(f"[insert] {pm.race_id} ← {p.name}")
                elif upd:
                    updated += 1
                    if ns.verbose:
                        print(f"[update] {pm.race_id} ← {p.name}")
                else:
                    # No-op (already present with same values)
                    if ns.verbose:
                        print(f"[noop] {pm.race_id}")
            except Exception as e:
                errors += 1
                print(f"[error] {pm.race_id} from {p.name}: {e}")

        if not ns.dry_run:
            try:
                conn.commit()
            except Exception:
                pass

        print(
            f"Done. files={len(files)} inserted={inserted} updated={updated} skipped={skipped} errors={errors}"
        )
        # Guidance for persistence joins
        print(
            "Hint: Now rerun scripts/persist_predictions_from_json.py to map predictions to standardized race_id if needed."
        )
        return 0 if errors == 0 else 1
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

