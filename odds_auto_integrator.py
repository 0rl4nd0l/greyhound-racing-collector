#!/usr/bin/env python3
"""Just-in-time Sportsbet odds loader used by prediction endpoints.

The Flask app imports ``ensure_odds_for_target_race`` behind the explicit
``ENABLE_AUTO_SCRAPE_ODDS`` flag. This module stays side-effect free until that
function is called.
"""

from __future__ import annotations

import re
import sqlite3
import time
from datetime import date, datetime, timedelta
from typing import Any
from urllib.parse import urljoin, urlparse


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
    "QOT": ("Ladbrokes Q", "Q1 Lakeside", "Q2 Parklands"),
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


def _hint_values(raw: str) -> set[str]:
    hint = VENUE_NAME_HINTS.get(raw.upper())
    if not hint:
        return set()
    if isinstance(hint, str):
        return {hint}
    return {str(value) for value in hint if value}


def _target_display_names(venue: Any) -> set[str]:
    raw = str(venue or "").strip()
    values = {raw}
    values.update(_hint_values(raw))
    try:
        from config.venue_mapping import (
            VENUE_CODE_TO_NAME,
            VENUE_MAPPING,
            normalize_venue,
        )

        code = normalize_venue(raw)
        values.add(code)
        if code in VENUE_CODE_TO_NAME:
            values.add(VENUE_CODE_TO_NAME[code])
            values.update(_hint_values(code))
        for alias, mapped_code in VENUE_MAPPING.items():
            if mapped_code == code:
                values.add(alias.replace("_", " ").replace("-", " "))
    except Exception:
        pass
    return {v for v in values if v}


def _target_names(venue: Any) -> set[str]:
    return {_norm(v) for v in _target_display_names(venue) if v}


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")


def _target_slugs(venue: Any) -> set[str]:
    return {slug for slug in (_slug(value) for value in _target_display_names(venue)) if slug}


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


def _candidate_meeting_slug(href: str) -> str:
    try:
        parts = [part for part in urlparse(href).path.split("/") if part]
    except Exception:
        return ""
    if "greyhound-racing" not in parts:
        return ""
    idx = parts.index("greyhound-racing")
    if idx + 2 >= len(parts):
        return ""
    slug = parts[idx + 2]
    if slug.startswith("race-") or "meeting-" in slug:
        return ""
    return slug


def _name_matches_target(candidate: str, target_norms: set[str]) -> bool:
    candidate_norm = _norm(candidate)
    if not candidate_norm:
        return False
    for target_norm in target_norms:
        if len(target_norm) < 3:
            continue
        if candidate_norm == target_norm:
            return True
        if len(candidate_norm) > 3 and (
            candidate_norm in target_norm or target_norm in candidate_norm
        ):
            return True
    return False


def _slug_matches_target(candidate_slug: str, target_slugs: set[str]) -> bool:
    if not candidate_slug:
        return False
    for target_slug in target_slugs:
        if len(target_slug) < 3:
            continue
        if candidate_slug == target_slug:
            return True
        if len(candidate_slug) > 3 and (
            candidate_slug in target_slug or target_slug in candidate_slug
        ):
            return True
    return False


def _region_for_target(venue: Any) -> str:
    try:
        from config.venue_mapping import get_venue_state, normalize_venue

        code = normalize_venue(str(venue or ""))
        state = get_venue_state(code)
        if state and state != "UNKNOWN":
            return "australia-nz"
    except Exception:
        pass
    return "australia-nz"


def _find_meeting_url_for_target(driver: Any, base_url: str, venue: Any) -> str | None:
    region = _region_for_target(venue)
    region_url = urljoin(base_url, f"/greyhound-racing/{region}")
    target_norms = _target_names(venue)
    target_slugs = _target_slugs(venue)
    driver.get(region_url)
    try:
        driver.execute_script("return document.readyState")
    except Exception:
        pass
    time.sleep(1)
    try:
        anchors = driver.find_elements(
            "css selector",
            f"a[href*='/greyhound-racing/{region}/']",
        )
    except Exception:
        anchors = []
    for anchor in anchors:
        try:
            href = anchor.get_attribute("href") or ""
        except Exception:
            continue
        if "/race-" in href:
            continue
        meeting_slug = _candidate_meeting_slug(href)
        if not meeting_slug:
            continue
        text = (getattr(anchor, "text", "") or "").strip()
        if not text:
            try:
                text = anchor.get_attribute("aria-label") or ""
            except Exception:
                text = ""
        if _slug_matches_target(meeting_slug, target_slugs) or _name_matches_target(
            text, target_norms
        ):
            return href
    return None


def _resolve_target_race_from_meeting(
    integrator: Any,
    driver: Any,
    venue: Any,
    race_number: int,
    target_date: str,
) -> dict[str, Any] | None:
    meeting_url = _find_meeting_url_for_target(driver, integrator.base_url, venue)
    if not meeting_url:
        return None
    race_url = integrator.find_specific_race_from_meeting(
        meeting_url,
        int(race_number),
        expected_venue=venue,
    )
    if not race_url:
        return None
    meeting_slug = _candidate_meeting_slug(meeting_url) or _slug(venue)
    display_name = sorted(_target_display_names(venue), key=len, reverse=True)[0]
    return {
        "race_id": f"{meeting_slug}_{int(race_number)}_{target_date.replace('-', '')}",
        "venue": display_name,
        "venue_slug": meeting_slug,
        "race_number": int(race_number),
        "race_date": target_date,
        "race_time": "Unknown",
        "start_datetime": None,
        "venue_url": race_url,
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
        cur.execute("PRAGMA table_info(live_odds)")
        live_columns = {str(row[1]) for row in cur.fetchall()}
        optional_columns = [
            column
            for column in (
                "source_url",
                "capture_timestamp",
                "capture_mode",
                "odds_level",
                "sportsbet_box_source",
                "sportsbet_list_position",
                "sportsbet_raw_runner_text",
            )
            if column in live_columns
        ]
        insert_columns = [
            "race_id",
            "venue",
            "race_number",
            "race_date",
            "race_time",
            "dog_name",
            "dog_clean_name",
            "box_number",
            "odds_decimal",
            "odds_fractional",
            "market_type",
            "source",
            "is_current",
            "topN",
            *optional_columns,
        ]
        select_columns = [
            "?",
            "?",
            "?",
            "?",
            "race_time",
            "dog_name",
            "dog_clean_name",
            "box_number",
            "odds_decimal",
            "odds_fractional",
            "market_type",
            "source",
            "1",
            "topN",
            *optional_columns,
        ]
        cur.execute(
            f"""
            INSERT INTO live_odds ({', '.join(insert_columns)})
            SELECT {', '.join(select_columns)}
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


def fetch_odds_for_target_race(
    db_path: str,
    venue: Any,
    race_number: int | None,
    race_date: Any = None,
    allow_auto_scrape_odds: bool | None = None,
) -> dict[str, Any]:
    """Fetch current Sportsbet odds for a target race without writing DB rows."""

    allowed, opt_in_source = _auto_scrape_odds_allowed(allow_auto_scrape_odds)
    summary: dict[str, Any] = {
        "success": False,
        "win_count": 0,
        "place_count": 0,
        "warnings": [],
        "race_id": None,
        "alias_race_id": None,
        "opt_in_source": opt_in_source,
        "discovery_method": None,
        "race_info": None,
        "odds_data": [],
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
        "🔄 Auto odds fetch enabled "
        f"because {opt_in_source}; target={venue} R{race_number} {target_date}"
    )
    integrator = SportsbetOddsIntegrator(
        db_path,
        allow_auto_scrape_odds=True,
        setup_database=False,
    )
    try:
        if not integrator.setup_driver():
            summary["warnings"].append("selenium driver unavailable")
            return summary
        driver = integrator.driver
        driver.get(integrator.greyhound_url)

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
            selected = _resolve_target_race_from_meeting(
                integrator,
                driver,
                venue,
                int(race_number),
                target_date,
            )
            if selected:
                summary["discovery_method"] = "sportsbet_meeting_exact_race"
            else:
                summary["warnings"].append(
                    f"target race not visible on Sportsbet landing or meeting pages: venue={venue} race={race_number}"
                )
                return summary
        else:
            summary["discovery_method"] = "sportsbet_landing"

        enhanced = integrator.get_race_odds_from_page(selected)
        source_race_id = integrator._canonical_race_id(
            enhanced.get("venue"), enhanced.get("race_date"), enhanced.get("race_number")
        ) or enhanced.get("race_id")
        alias = _alias_race_id(int(race_number), venue, target_date)
        odds_data = list(enhanced.get("odds_data") or [])
        summary.update(
            {
                "success": bool(odds_data),
                "race_id": source_race_id,
                "alias_race_id": alias,
                "win_count": len(odds_data),
                "place_count": len(enhanced.get("odds_data_place") or []),
                "race_info": enhanced,
                "odds_data": odds_data,
            }
        )
        if not summary["success"]:
            summary["warnings"].append("race found but no win odds extracted")
        return summary
    finally:
        integrator.close_driver()


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
            selected = _resolve_target_race_from_meeting(
                integrator,
                driver,
                venue,
                int(race_number),
                target_date,
            )
            if selected:
                summary["discovery_method"] = "sportsbet_meeting_exact_race"
            else:
                summary["warnings"].append(
                    f"target race not visible on Sportsbet landing or meeting pages: venue={venue} race={race_number}"
                )
                return summary
        else:
            summary["discovery_method"] = "sportsbet_landing"

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
