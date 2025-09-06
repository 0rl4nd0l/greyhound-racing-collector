#!/usr/bin/env python3
"""
Backfill TGR enhanced dog histories from long-form page(s) by race date
=====================================================================

This script backfills enhanced TGR dog histories for all dogs in a specific race
by fetching the TGR form-guides page for that date, visiting each long-form meeting
page, extracting per-dog racing histories, filtering to only the race's participant
dogs, and persisting only history entries that occurred before the race date (to
avoid temporal leakage).

Usage:
  python scripts/tgr_backfill_from_longform_by_date.py \
    --db databases/comprehensive_greyhound_data.db \
    --race-id ap_k_2025-02-18_2 \
    [--rate-limit 2.0] [--no-cache]

Notes:
- Uses TheGreyhoundRecorderScraper._get and _parse_form_guides/_fetch_race_details
  to reuse existing parsing logic that knows the class names
- Persists into tgr_enhanced_dog_form with ISO-formatted race_date
- Does NOT populate legacy gr_dog_form/gr_dog_entries; ML uses enhanced fallback
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from typing import Any, Dict, List, Optional, Tuple

from scripts.db_utils import open_sqlite_writable

# Ensure project root on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.collectors.the_greyhound_recorder_scraper import (
    BASE_URL,
    TheGreyhoundRecorderScraper,
)


def get_race_info(db_path: str, race_id: str) -> Tuple[str, List[str]]:
    """Return (race_date_iso, participants) for race_id."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT race_date FROM race_metadata WHERE race_id = ? LIMIT 1",
            [race_id],
        )
        row = cur.fetchone()
        if not row or not row[0]:
            raise RuntimeError(f"No race_date found for race_id={race_id}")
        race_date_iso = str(row[0]).strip()

        cur.execute(
            """
            SELECT DISTINCT dog_clean_name
            FROM dog_race_data
            WHERE race_id = ? AND dog_clean_name IS NOT NULL AND TRIM(dog_clean_name) != ''
            ORDER BY box_number ASC
            """,
            [race_id],
        )
        dogs = [r[0] for r in cur.fetchall()]
        if not dogs:
            raise RuntimeError(f"No participants found for race_id={race_id}")
        return race_date_iso, dogs
    finally:
        conn.close()


def norm_name(s: str) -> str:
    if s is None:
        return ""
    # Uppercase, remove quotes/apostrophes/backticks and compress whitespace
    s2 = (
        str(s)
        .upper()
        .replace('"', "")
        .replace("'", "")
        .replace("`", "")
        .replace("’", "")
        .replace("“", "")
        .replace("”", "")
    )
    return re.sub(r"\s+", " ", s2).strip()


def parse_any_date_to_iso(text: str) -> Optional[str]:
    """Parse diverse TGR date strings into ISO YYYY-MM-DD or return None."""
    if not text:
        return None
    s = str(text).strip()
    # Remove ordinal suffixes (1st -> 1)
    s = re.sub(r"(\d{1,2})(st|nd|rd|th)\b", r"\1", s, flags=re.IGNORECASE)

    # Try several formats
    fmts = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d/%m/%y",
        "%d %b %Y",
        "%d %B %Y",
        "%d-%m-%Y",
        "%d-%m-%y",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue

    # As a last resort: try to extract dd/mm/yy or dd Mon YYYY with regex
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", s)
    if m:
        d, mo, y = m.groups()
        y = ("20" + y) if len(y) == 2 else y
        try:
            dt = datetime(int(y), int(mo), int(d))
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return None
    # e.g., 3 Aug 2025 or 3 August 2025
    m2 = re.search(r"(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})", s)
    if m2:
        d, mon, y = m2.groups()
        try:
            dt = datetime.strptime(f"{d} {mon} {y}", "%d %b %Y")
            return dt.strftime("%Y-%m-%d")
        except Exception:
            try:
                dt = datetime.strptime(f"{d} {mon} {y}", "%d %B %Y")
                return dt.strftime("%Y-%m-%d")
            except Exception:
                return None
    return None


def safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        s = str(v).strip().replace("$", "")
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def backfill_from_longform_for_date(
    db_path: str, race_id: str, rate_limit: float, use_cache: bool, scan_days: int = 7
) -> Dict[str, Any]:
    race_date_iso, dogs = get_race_info(db_path, race_id)
    print(f"📅 Race date: {race_date_iso}")
    print(f"🐕 Participants ({len(dogs)}):")
    for i, d in enumerate(dogs, 1):
        print(f"  {i}. {d}")

    target_date_iso = race_date_iso

    scraper = TheGreyhoundRecorderScraper(rate_limit=rate_limit, use_cache=use_cache)

    # Build a set of meetings across a window of dates [-scan_days .. 0]
    from datetime import timedelta

    base_dt = datetime.strptime(target_date_iso, "%Y-%m-%d")
    all_meetings: List[Dict[str, Any]] = []
    seen_urls = set()

    for delta in range(-abs(scan_days), 1):
        dt = base_dt + timedelta(days=delta)
        d_iso = dt.strftime("%Y-%m-%d")
        url = f"{BASE_URL}/form-guides?date={d_iso}"
        soup = scraper._get(url)
        if not soup:
            continue
        fg = scraper._parse_form_guides(soup)
        meetings = fg.get("meetings", [])
        for m in meetings:
            lf = m.get("long_form_url")
            if lf and lf not in seen_urls:
                seen_urls.add(lf)
                all_meetings.append(m)
    if not all_meetings:
        return {
            "success": False,
            "error": f"No meetings found in window for {target_date_iso}",
        }

    print(f"📋 Meetings scanned in window [-{scan_days}..0]: {len(all_meetings)}")

    # Normalize participant names for matching
    dog_set = {norm_name(d) for d in dogs}

    inserted = 0
    matched_dogs = set()
    errors: List[str] = []

    # Prepare DB
    conn = open_sqlite_writable(db_path)
    cur = conn.cursor()

    def insert_entry(
        dog_name: str, entry: Dict[str, Any], meeting_meta: Dict[str, Any]
    ):
        nonlocal inserted
        try:
            race_date_raw = entry.get("race_date")
            race_date_iso_entry = parse_any_date_to_iso(race_date_raw)
            if not race_date_iso_entry:
                return
            # Respect temporal cutoff
            if race_date_iso_entry >= target_date_iso:
                return

            venue = entry.get("track") or meeting_meta.get("venue")
            grade = entry.get("grade")
            distance = entry.get("distance")
            box_number = entry.get("box_number")
            weight = entry.get("weight")
            comments_parts = []
            if entry.get("in_run"):
                comments_parts.append(f"In run: {entry.get('in_run')}")
            if entry.get("winner_second"):
                comments_parts.append(f"W/S: {entry.get('winner_second')}")
            comments = "; ".join(comments_parts) if comments_parts else None
            odds = safe_float(entry.get("starting_price"))
            odds_text = (
                entry.get("starting_price")
                if isinstance(entry.get("starting_price"), str)
                else None
            )
            race_time = entry.get("individual_time")
            split_times = (
                {"sectional": entry.get("sectional_time")}
                if entry.get("sectional_time") is not None
                else {}
            )
            margin = entry.get("margin")
            field_size = meeting_meta.get("field_size")
            race_number = meeting_meta.get("race_number")
            race_url = meeting_meta.get("url")

            cur.execute(
                """
                INSERT OR REPLACE INTO tgr_enhanced_dog_form
                (dog_name, race_date, venue, grade, distance, box_number,
                 recent_form, weight, comments, odds, odds_text, trainer,
                 profile_url, race_url, field_size, race_number, expert_comments,
                 finishing_position, race_time, split_times, margin)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    dog_name,
                    race_date_iso_entry,
                    venue,
                    grade,
                    distance,
                    box_number,
                    json.dumps([]),
                    weight,
                    comments,
                    odds,
                    odds_text,
                    None,  # trainer
                    None,  # profile_url
                    race_url,
                    field_size,
                    race_number,
                    json.dumps([]),
                    entry.get("finish_position"),
                    str(race_time) if race_time is not None else None,
                    json.dumps(split_times),
                    str(margin) if margin is not None else None,
                ],
            )
            inserted += 1
        except Exception as e:
            errors.append(f"Insert failed for {dog_name}: {e}")

    # Iterate meetings (long-form pages)
    for m in all_meetings:
        long_form_url = m.get("long_form_url")
        if not long_form_url:
            continue
        details = scraper._fetch_race_details(long_form_url)
        if not details or not details.get("dogs"):
            continue
        # Build meeting meta
        meeting_meta = {
            "venue": details.get("venue"),
            "date": details.get("date"),
            "race_number": details.get("race_number"),
            "field_size": details.get("field_size"),
            "url": details.get("url")
            or (
                BASE_URL + long_form_url
                if long_form_url.startswith("/")
                else long_form_url
            ),
        }
        # For each dog block on this page, see if it matches race participants
        for dog_block in details.get("dogs", []):
            dog_name_raw = dog_block.get("dog_name")
            if not dog_name_raw:
                continue
            if norm_name(dog_name_raw) not in dog_set:
                continue
            matched_dogs.add(dog_name_raw)
            history = dog_block.get("racing_history") or []
            for entry in history:
                insert_entry(dog_name_raw, entry, meeting_meta)

        # Commit periodically to avoid large transactions
        conn.commit()

    conn.commit()
    conn.close()

    return {
        "success": True,
        "inserted": inserted,
        "matched_dogs": sorted(list(matched_dogs)),
        "errors": errors[:10],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Backfill TGR enhanced dog histories by date from long-form page(s)"
    )
    parser.add_argument("--db", dest="db_path", required=True, help="Path to SQLite DB")
    parser.add_argument(
        "--race-id", dest="race_id", required=True, help="Race ID to backfill"
    )
    parser.add_argument(
        "--rate-limit",
        dest="rate_limit",
        default="2.0",
        help="Rate limit seconds between requests",
    )
    parser.add_argument(
        "--no-cache",
        dest="no_cache",
        action="store_true",
        help="Disable cache for fresh fetch",
    )
    args = parser.parse_args()

    # Configure scraper env
    os.environ.setdefault("TGR_RATE_LIMIT", str(args.rate_limit))
    os.environ["TGR_DISABLE_CACHE"] = "1" if args.no_cache else "0"

    result = backfill_from_longform_for_date(
        db_path=args.db_path,
        race_id=args.race_id,
        rate_limit=float(args.rate_limit),
        use_cache=not args.no_cache,
        scan_days=7,
    )

    if not result.get("success"):
        print(f"❌ Backfill failed: {result.get('error')}")
        sys.exit(1)

    print("\n📊 Backfill from long-form summary:")
    print(f"  Inserted rows: {result.get('inserted', 0)}")
    print(f"  Matched dogs: {', '.join(result.get('matched_dogs') or [])}")
    errs = result.get("errors") or []
    if errs:
        print(f"  Errors: {len(errs)} (showing up to 10):")
        for e in errs:
            print(f"    - {e}")
    print("\n✅ Done")


if __name__ == "__main__":
    main()
