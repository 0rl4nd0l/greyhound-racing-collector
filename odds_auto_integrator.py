#!/usr/bin/env python3
"""Just-in-time Sportsbet odds loader used by prediction endpoints.

The Flask app imports ``ensure_odds_for_target_race`` behind the explicit
``ENABLE_AUTO_SCRAPE_ODDS`` flag. This module stays side-effect free until that
function is called.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import date, datetime, timedelta
from typing import Any


VENUE_NAME_HINTS = {
    "AP_K": "Angle Park",
    "AP/K": "Angle Park",
    "APWE": "Angle Park",
    "WAR": "Warrnambool",
    "WRGL": "Warragul",
    "WPK": "Wentworth Park",
    "MEA": "The Meadows",
    "MOUNT": "Mount Gambier",
    "MT_G": "Mount Gambier",
    "MAND": "Mandurah",
    "HOBT": "Hobart",
    "GRDN": "The Gardens",
    "CASO": "Casino",
    "NOWRA": "Nowra",
    "SHEP": "Shepparton",
    "QOT": "Ladbrokes Q",
}


def _norm(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _iso_date(value: Any) -> str:
    if value is None:
        return date.today().isoformat()
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        return date.today().isoformat()
    try:
        return datetime.fromisoformat(text[:10]).date().isoformat()
    except Exception:
        return text[:10]


def _target_names(venue: Any) -> set[str]:
    raw = str(venue or "").strip()
    values = {raw}
    hint = VENUE_NAME_HINTS.get(raw.upper())
    if hint:
        values.add(hint)
    try:
        from config.venue_mapping import VENUE_CODE_TO_NAME, normalize_venue

        code = normalize_venue(raw)
        values.add(code)
        if code in VENUE_CODE_TO_NAME:
            values.add(VENUE_CODE_TO_NAME[code])
    except Exception:
        pass
    return {_norm(v) for v in values if v}


def _parse_anchor(text: str, href: str, target_date: str) -> dict[str, Any] | None:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return None
    match = re.match(r"^R(\d+)\s+(.+)$", lines[0], re.IGNORECASE)
    if not match:
        return None
    race_number = int(match.group(1))
    venue = match.group(2).strip()
    countdown = lines[1] if len(lines) > 1 else ""
    start_dt = None
    race_time = "Unknown"
    minutes_match = re.search(r"(\d+)m", countdown)
    seconds_match = re.search(r"(\d+)s", countdown)
    if minutes_match or seconds_match:
        minutes = int(minutes_match.group(1)) if minutes_match else 0
        seconds = int(seconds_match.group(1)) if seconds_match else 0
        start_dt = datetime.now() + timedelta(minutes=minutes, seconds=seconds)
        race_time = start_dt.strftime("%H:%M")
    venue_slug = ""
    try:
        parts = [p for p in href.split("/") if p]
        if "australia-nz" in parts:
            idx = parts.index("australia-nz")
            if idx + 1 < len(parts):
                venue_slug = parts[idx + 1]
    except Exception:
        venue_slug = ""
    return {
        "race_id": f"{venue_slug or _norm(venue).lower()}_{race_number}_{target_date.replace('-', '')}",
        "venue": venue,
        "venue_slug": venue_slug or venue.lower().replace(" ", "-"),
        "race_number": race_number,
        "race_date": target_date,
        "race_time": race_time,
        "start_datetime": start_dt,
        "venue_url": href,
        "odds_data": [],
    }


def _alias_race_id(race_number: int, venue: Any, target_date: str) -> str:
    return f"Race {int(race_number)} - {str(venue).strip()} - {target_date}"


def _copy_current_odds_to_alias(
    db_path: str,
    source_race_id: str,
    alias_race_id: str,
    venue: Any,
    race_number: int,
    race_date: str,
) -> int:
    if not source_race_id or not alias_race_id or source_race_id == alias_race_id:
        return 0
    from sportsbet_odds_integrator import safe_upsert_race_metadata

    conn = sqlite3.connect(db_path)
    try:
        try:
            conn.execute("PRAGMA busy_timeout=2000")
        except Exception:
            pass
        cur = conn.cursor()
        conn.execute("BEGIN")
        cur.execute(
            "UPDATE live_odds SET is_current = 0 WHERE race_id = ? AND is_current = 1",
            (alias_race_id,),
        )
        cur.execute(
            """
            INSERT INTO live_odds (
                race_id, venue, race_number, race_date, race_time, dog_name,
                dog_clean_name, box_number, odds_decimal, odds_fractional,
                market_type, source, is_current, topN
            )
            SELECT
                ?, ?, ?, ?, race_time, dog_name, dog_clean_name, box_number,
                odds_decimal, odds_fractional, market_type, source, 1, topN
            FROM live_odds
            WHERE race_id = ? AND is_current = 1
            """,
            (
                alias_race_id,
                str(venue or ""),
                int(race_number),
                race_date,
                source_race_id,
            ),
        )
        inserted = int(cur.rowcount or 0)

        cur.execute("PRAGMA table_info(race_metadata)")
        columns = {str(row[1]) for row in cur.fetchall()}
        source_fields = [
            column
            for column in ("race_time", "sportsbet_url", "url", "venue_slug", "start_datetime")
            if column in columns
        ]
        copied_metadata: dict[str, Any] = {}
        if source_fields:
            cur.execute(
                f"SELECT {', '.join(source_fields)} FROM race_metadata WHERE race_id = ?",
                (source_race_id,),
            )
            row = cur.fetchone()
            if row:
                copied_metadata.update(dict(zip(source_fields, row)))

        safe_upsert_race_metadata(
            cur,
            alias_race_id,
            {
                "venue": str(venue or ""),
                "race_number": int(race_number),
                "race_date": race_date,
                **copied_metadata,
            },
        )
        conn.commit()
        return inserted
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        conn.close()


def _auto_scrape_odds_allowed(
    allow_auto_scrape_odds: bool | None = None,
) -> tuple[bool, str]:
    if allow_auto_scrape_odds is not None:
        return bool(allow_auto_scrape_odds), "explicit argument allow_auto_scrape_odds"
    try:
        from utils.feature_flags import auto_scrape_odds_enabled, load_flags

        _flags, sources = load_flags()
        enabled = auto_scrape_odds_enabled()
        source = sources.get("ENABLE_AUTO_SCRAPE_ODDS", "default")
        return enabled, f"ENABLE_AUTO_SCRAPE_ODDS from {source}"
    except Exception as exc:
        return False, f"feature flag unavailable: {exc}"


def ensure_odds_for_target_race(
    db_path: str,
    venue: Any,
    race_number: int | None,
    race_date: Any = None,
    allow_auto_scrape_odds: bool | None = None,
    append_only: bool = False,
) -> dict[str, Any]:
    """Fetch and persist current Sportsbet odds for a target race if visible."""

    allowed, opt_in_source = _auto_scrape_odds_allowed(allow_auto_scrape_odds)
    summary: dict[str, Any] = {
        "success": False,
        "win_count": 0,
        "place_count": 0,
        "warnings": [],
        "race_id": None,
        "alias_race_id": None,
        "opt_in_source": opt_in_source,
        "append_only": bool(append_only),
        "capture_reports": [],
        "captured_rows": 0,
    }
    if not allowed:
        summary["warnings"].append(f"auto odds scraping disabled; {opt_in_source}")
        return summary
    if not race_number:
        summary["warnings"].append("race_number missing")
        return summary

    target_date = _iso_date(race_date)
    target_names = _target_names(venue)

    from sportsbet_odds_integrator import SportsbetOddsIntegrator

    print(
        "🔄 Auto odds scraping enabled "
        f"because {opt_in_source}; target={venue} R{race_number} {target_date}"
    )
    integrator = SportsbetOddsIntegrator(
        db_path,
        allow_auto_scrape_odds=True,
    )
    try:
        if not integrator.setup_driver():
            summary["warnings"].append("selenium driver unavailable")
            return summary
        driver = integrator.driver
        driver.get(integrator.greyhound_url)
        import time

        time.sleep(5)
        anchors = driver.find_elements("css selector", "a[href*='greyhound-racing']")
        selected = None
        for anchor in anchors:
            href = anchor.get_attribute("href") or ""
            text = (anchor.text or "").strip()
            parsed = _parse_anchor(text, href, target_date)
            if not parsed:
                continue
            if int(parsed["race_number"]) != int(race_number):
                continue
            if target_names and _norm(parsed["venue"]) not in target_names:
                continue
            selected = parsed
            break
        if not selected:
            summary["warnings"].append(
                f"target race not visible on Sportsbet landing: venue={venue} race={race_number}"
            )
            return summary

        enhanced = integrator.get_race_odds_from_page(selected)
        source_race_id = integrator._canonical_race_id(
            enhanced.get("venue"), enhanced.get("race_date"), enhanced.get("race_number")
        ) or enhanced.get("race_id")
        summary["race_id"] = source_race_id
        summary["win_count"] = len(enhanced.get("odds_data") or [])
        summary["place_count"] = len(enhanced.get("odds_data_place") or [])
        alias = _alias_race_id(int(race_number), venue, target_date)
        summary["alias_race_id"] = alias
        if append_only:
            canonical_info = dict(enhanced)
            canonical_info["race_id"] = source_race_id
            canonical_report = integrator.append_pre_jump_odds_snapshot(
                canonical_info,
                enhanced.get("odds_data") or [],
                capture_mode="opt_in_live_pre_jump_snapshot",
            )
            alias_info = dict(enhanced)
            alias_info.update(
                {
                    "race_id": alias,
                    "venue": venue,
                    "race_number": int(race_number),
                    "race_date": target_date,
                    "preserve_race_id": True,
                }
            )
            alias_report = integrator.append_pre_jump_odds_snapshot(
                alias_info,
                enhanced.get("odds_data") or [],
                capture_mode="opt_in_live_pre_jump_snapshot_alias",
            )
            summary["capture_reports"] = [canonical_report, alias_report]
            summary["alias_rows"] = alias_report.get("inserted_rows", 0)
            summary["win_count"] = int(canonical_report.get("inserted_rows") or 0)
            summary["place_count"] = 0
            summary["captured_rows"] = sum(
                int(report.get("inserted_rows") or 0)
                for report in summary["capture_reports"]
            )
            for report in summary["capture_reports"]:
                summary["warnings"].extend(report.get("warnings") or [])
        else:
            integrator.save_odds_to_database(enhanced)
            try:
                summary["alias_rows"] = _copy_current_odds_to_alias(
                    db_path, source_race_id, alias, venue, int(race_number), target_date
                )
            except Exception as exc:
                summary["warnings"].append(f"alias odds copy failed: {exc}")
        summary["success"] = (
            int(summary.get("captured_rows") or 0) > 0
            if append_only
            else summary["win_count"] > 0
        )
        if not summary["success"]:
            summary["warnings"].append("race found but no win odds extracted")
        return summary
    finally:
        integrator.close_driver()
