#!/usr/bin/env python3
"""
Normalize live_odds.race_id to canonical form (VENUECODE_YYYY-MM-DD_RN) and upsert race_metadata rows.

Usage:
  python scripts/normalize_live_odds_race_ids.py [--db ./greyhound_racing_data_writable.db] [--dry-run]

Behavior:
- Reads distinct (race_id, venue, race_number, race_date) from live_odds
- Computes canonical race_id using config.venue_mapping.normalize_venue and ISO date
- Updates live_odds rows to canonical race_id when different
- Inserts/updates race_metadata for the canonical race_id (venue, race_number, race_date)
- Non-destructive and idempotent
"""
from __future__ import annotations

import argparse
import os
import sqlite3
from datetime import datetime
from pathlib import Path

try:
    from config.venue_mapping import normalize_venue  # type: ignore
except Exception:
    def normalize_venue(v: str) -> str:  # type: ignore
        return (v or "").strip().upper()


def to_iso_date(s: str | None) -> str | None:
    if not s:
        return None
    ss = str(s).strip()
    # Already ISO
    if len(ss) >= 10 and ss[4] == "-" and ss[7] == "-":
        return ss[:10]
    for fmt in ("%d %B %Y", "%d %b %Y", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(ss, fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    return ss or None


def canonical_race_id(venue: str | None, race_date: str | None, race_number) -> str | None:
    try:
        code = normalize_venue(venue or "")
        date_iso = to_iso_date(race_date)
        rn = int(race_number) if race_number is not None else None
        if code and date_iso and rn:
            return f"{code}_{date_iso}_{rn}"
    except Exception:
        return None
    return None


def resolve_db(cli_db: str | None) -> str:
    if cli_db:
        return cli_db
    for env in ("STAGING_DB_PATH", "GREYHOUND_DB_PATH", "DATABASE_PATH"):
        v = os.getenv(env)
        if v:
            return v
    return "./greyhound_racing_data_writable.db"


def main() -> int:
    ap = argparse.ArgumentParser(description="Normalize live_odds race_id to canonical form")
    ap.add_argument("--db", dest="db_path", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    db_path = resolve_db(args.db_path)
    conn = sqlite3.connect(db_path)
    try:
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        except Exception:
            pass
        cur = conn.cursor()
        # Fetch distinct race keys from live_odds
        cur.execute(
            """
            SELECT DISTINCT COALESCE(race_id,''), COALESCE(venue,''), race_number, COALESCE(race_date,''), COALESCE(race_time,'')
            FROM live_odds
            """
        )
        rows = cur.fetchall() or []
        examined = len(rows)
        updated_lo = 0
        upsert_rm = 0
        for rid, venue, race_number, race_date, race_time in rows:
            can = canonical_race_id(venue, race_date, race_number)
            if not can:
                continue
            if can != rid and not args.dry_run:
                # Update live_odds to canonical id for this race key
                try:
                    cur.execute(
                        """
                        UPDATE live_odds
                        SET race_id = ?
                        WHERE race_id = ? OR (
                            venue = ? AND race_number = ? AND date(COALESCE(race_date,'')) = date(?)
                        )
                        """,
                        (can, rid, venue, race_number, race_date),
                    )
                    if cur.rowcount > 0:
                        updated_lo += cur.rowcount
                except Exception:
                    pass
            # Upsert minimal race_metadata
            if not args.dry_run:
                try:
                    cur.execute(
                        """
                        INSERT INTO race_metadata (race_id, venue, race_number, race_date, race_time, extraction_timestamp, data_source)
                        VALUES (?, ?, ?, ?, ?, ?, 'odds_normalizer')
                        ON CONFLICT(race_id) DO UPDATE SET
                            venue = COALESCE(excluded.venue, race_metadata.venue),
                            race_number = COALESCE(excluded.race_number, race_metadata.race_number),
                            race_date = COALESCE(excluded.race_date, race_metadata.race_date),
                            race_time = COALESCE(excluded.race_time, race_metadata.race_time),
                            extraction_timestamp = excluded.extraction_timestamp
                        """,
                        (
                            can,
                            normalize_venue(venue),
                            int(race_number) if race_number is not None else None,
                            to_iso_date(race_date),
                            race_time or None,
                            datetime.now().isoformat(timespec="seconds"),
                        ),
                    )
                    upsert_rm += 1
                except Exception:
                    pass
        if not args.dry_run:
            try:
                conn.commit()
            except Exception:
                pass
        print(
            f"Done. examined={examined} updated_live_odds_rows~={updated_lo} upserted_race_metadata={upsert_rm} dry_run={args.dry_run} db={db_path}"
        )
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())

