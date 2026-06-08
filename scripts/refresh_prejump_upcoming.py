#!/usr/bin/env python3
"""Refresh current pre-jump TheDogs form guides into an upcoming directory.

The script writes only local upcoming CSV artifacts, raw exports, sidecars, and
quarantine files produced by the existing UpcomingRaceBrowser/export path. It
does not persist prediction snapshots, capture odds, ingest results, write
labels, retrain, promote, or bet.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from utils.race_lifecycle import melbourne_now  # noqa: E402


VENUE_EXCLUSION_ALIAS_GROUPS = [
    {"QOT", "LADBROKES-Q-STRAIGHT", "LADBROKES_Q_STRAIGHT", "LADBROKES Q STRAIGHT"},
    {"LCTN", "LAUNCESTON"},
    {"AP_K", "ANGLE-PARK", "ANGLE_PARK", "ANGLE PARK"},
    {"SAN", "SANDOWN"},
    {"BAL", "BALLARAT"},
    {"WAR", "WARRNAMBOOL"},
    {"TRA", "TRARALGON"},
    {"NOR", "NORTHAM"},
    {"BEN", "BENDIGO"},
    {"GEE", "GEELONG"},
    {"GOSF", "GOSFORD"},
    {"GRAF", "GRAFTON"},
    {"GUNN", "GUNNEDAH"},
    {"HOR", "HORSHAM"},
    {"MAND", "MANDURAH"},
    {"MEA", "MEADOWS", "THE-MEADOWS", "THE_MEADOWS", "THE MEADOWS"},
    {"MURR", "MURRAY-BRIDGE-STRAIGHT", "MURRAY_BRIDGE_STRAIGHT", "MURRAY BRIDGE STRAIGHT"},
    {"RICH", "RICHMOND"},
    {"ROCK", "ROCKHAMPTON"},
    {"SAL", "SALE"},
    {"TAREE", "TARE"},
    {"TEM", "TEMORA"},
    {"TWN", "TOWNSVILLE"},
    {"WRGL", "WARRAGUL"},
]


def _venue_key(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "-", str(value).strip().upper()).strip("-")


def _venue_slug_from_url(source_url: Any) -> str | None:
    match = re.search(r"/racing/([^/?#]+)/", str(source_url or ""), flags=re.IGNORECASE)
    if not match:
        return None
    return _venue_key(match.group(1))


def venue_exclusion_aliases(venue: Any, *, source_url: Any = None) -> set[str]:
    raw = str(venue or "").strip()
    slug = _venue_slug_from_url(source_url)
    if not raw and not slug:
        return set()
    aliases = set()
    if slug:
        aliases.update({slug, slug.replace("-", "_")})
    if not raw:
        return {item for item in aliases if item}
    aliases.update({raw.upper(), raw.upper().replace("_", "-"), raw.upper().replace("-", "_")})
    key = _venue_key(raw)
    aliases.add(key)
    for group in VENUE_EXCLUSION_ALIAS_GROUPS:
        keyed_group = {_venue_key(item) for item in group}
        if key in keyed_group:
            for item in group:
                upper = str(item).strip().upper()
                aliases.update({upper, upper.replace("_", "-"), upper.replace("-", "_"), _venue_key(upper)})
    return {item for item in aliases if item}


def _stable_race_id_with_venue(race_number: Any, venue: Any, race_date: Any) -> str | None:
    if race_number in (None, "") or venue in (None, "") or race_date in (None, ""):
        return None
    match = re.search(r"\d+", str(race_number))
    if not match:
        return None
    return f"Race {int(match.group(0))} - {str(venue).strip().upper()} - {str(race_date).strip()[:10]}"


def stable_race_id(race: Mapping[str, Any]) -> str | None:
    race_number = race.get("race_number")
    venue = race.get("venue") or race.get("venue_name")
    race_date = race.get("date") or race.get("race_date")
    return _stable_race_id_with_venue(race_number, venue, race_date)


def stable_race_id_variants(race: Mapping[str, Any]) -> set[str]:
    race_number = race.get("race_number")
    venue = race.get("venue") or race.get("venue_name")
    race_date = race.get("date") or race.get("race_date")
    source_url = race.get("url") or race.get("race_url")
    return {
        race_id
        for alias in venue_exclusion_aliases(venue, source_url=source_url)
        if (race_id := _stable_race_id_with_venue(race_number, alias, race_date))
    }


_RACE_ID_RE = re.compile(r"^Race\s+(\d+)\s+-\s+(.+)\s+-\s+(\d{4}-\d{2}-\d{2})$")


def expand_excluded_race_ids(race_ids: set[str]) -> set[str]:
    expanded = {race_id for race_id in race_ids if race_id}
    for race_id in list(expanded):
        match = _RACE_ID_RE.match(str(race_id).strip())
        if not match:
            continue
        race_number, venue, race_date = match.groups()
        for alias in venue_exclusion_aliases(venue):
            alias_id = _stable_race_id_with_venue(race_number, alias, race_date)
            if alias_id:
                expanded.add(alias_id)
    return expanded


def load_excluded_race_ids(*, values: list[str] | None = None, file_path: str | None = None) -> set[str]:
    excluded = {str(value).strip() for value in values or [] if str(value).strip()}
    if not file_path:
        return excluded
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"exclude_race_ids_file_not_found:{path}")
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, list):
        excluded.update(str(item).strip() for item in payload if str(item).strip())
    elif isinstance(payload, Mapping):
        for key in ("race_ids", "exclude_race_ids", "excluded_race_ids"):
            items = payload.get(key)
            if isinstance(items, list):
                excluded.update(str(item).strip() for item in items if str(item).strip())
    else:
        excluded.update(line.strip() for line in text.splitlines() if line.strip())
    return excluded


def _parse_race_jump_datetime(
    race: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> datetime | None:
    now = now or melbourne_now()
    date_text = str(race.get("date") or race.get("race_date") or "").strip()
    time_text = str(
        race.get("race_time")
        or race.get("jump_time")
        or race.get("start_time")
        or ""
    ).strip()
    if not date_text or not time_text:
        return None
    try:
        race_date = datetime.strptime(date_text[:10], "%Y-%m-%d").date()
    except Exception:
        return None

    cleaned = re.sub(r"\s+", " ", time_text.upper().replace(".", ":"))
    parsed_time = None
    for fmt in ("%I:%M %p", "%I:%M%p", "%H:%M", "%H%M"):
        try:
            parsed_time = datetime.strptime(cleaned, fmt).time()
            break
        except Exception:
            continue
    if parsed_time is None:
        return None
    return datetime(
        race_date.year,
        race_date.month,
        race_date.day,
        parsed_time.hour,
        parsed_time.minute,
        tzinfo=now.tzinfo,
    )


def race_window_record(
    race: Mapping[str, Any],
    *,
    now: datetime | None = None,
    min_minutes: float = 20.0,
    max_minutes: float = 160.0,
) -> dict[str, Any]:
    now = now or melbourne_now()
    jump_dt = _parse_race_jump_datetime(race, now=now)
    minutes_to_jump = None
    bucket = "missing_jump_time"
    if jump_dt is not None:
        minutes_to_jump = (jump_dt - now).total_seconds() / 60.0
        if minutes_to_jump < min_minutes:
            bucket = "past_or_too_close"
        elif minutes_to_jump <= max_minutes:
            bucket = "preferred_window"
        else:
            bucket = "future_outside_preferred_window"
    return {
        "race_id": stable_race_id(race),
        "race_id_aliases": sorted(stable_race_id_variants(race)),
        "race_url": race.get("url"),
        "race_number": race.get("race_number"),
        "venue": race.get("venue") or race.get("venue_name"),
        "date": race.get("date") or race.get("race_date"),
        "race_time": race.get("race_time") or race.get("jump_time"),
        "jump_datetime": jump_dt.isoformat() if jump_dt else None,
        "minutes_to_jump": minutes_to_jump,
        "bucket": bucket,
        "selected": bucket == "preferred_window",
    }


def select_prejump_races(
    races: list[Mapping[str, Any]],
    *,
    now: datetime | None = None,
    min_minutes: float = 20.0,
    max_minutes: float = 160.0,
    limit: int = 0,
    exclude_race_ids: set[str] | None = None,
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]]]:
    records = [
        race_window_record(
            race,
            now=now,
            min_minutes=min_minutes,
            max_minutes=max_minutes,
        )
        for race in races
    ]
    excluded = expand_excluded_race_ids(exclude_race_ids or set())
    for race, record in zip(races, records):
        race_ids = set(record.get("race_id_aliases") or stable_race_id_variants(race))
        if record.get("selected") is True and race_ids & excluded:
            record["selected"] = False
            record["excluded_reason"] = "excluded_race_id"
            record["bucket"] = "excluded_race_id"
    selected = [
        race
        for race, record in zip(races, records)
        if record["selected"] and race.get("url")
    ]
    if limit and limit > 0:
        selected = selected[:limit]
    return selected, records


def refresh_timing_summary(
    records: list[dict[str, Any]],
    *,
    min_minutes: float = 20.0,
    max_minutes: float = 160.0,
) -> dict[str, Any]:
    def _with_jump_datetime(record: dict[str, Any]) -> tuple[datetime, dict[str, Any]] | None:
        jump_text = record.get("jump_datetime")
        if not jump_text:
            return None
        try:
            return datetime.fromisoformat(str(jump_text)), record
        except ValueError:
            return None

    def _window_fields(jump_dt: datetime, record: dict[str, Any]) -> dict[str, Any]:
        window_opens = jump_dt - timedelta(minutes=max_minutes)
        window_closes = jump_dt - timedelta(minutes=min_minutes)
        minutes_to_jump = record.get("minutes_to_jump")
        minutes_until_open = None
        minutes_until_close = None
        if isinstance(minutes_to_jump, (int, float)):
            minutes_until_open = float(minutes_to_jump) - float(max_minutes)
            minutes_until_close = float(minutes_to_jump) - float(min_minutes)
        return {
            "next_race": record,
            "next_window_opens_at": window_opens.isoformat(),
            "next_window_closes_at": window_closes.isoformat(),
            "minutes_until_window_opens": minutes_until_open,
            "minutes_until_window_closes": minutes_until_close,
        }

    selected = sorted(
        (
            item
            for item in (_with_jump_datetime(record) for record in records)
            if item is not None and item[1].get("selected") is True
        ),
        key=lambda item: item[0],
    )
    if selected:
        jump_dt, record = selected[0]
        return {
            "status": "OPEN_NOW",
            "reason": "at_least_one_race_in_preferred_window",
            **_window_fields(jump_dt, record),
        }

    future = sorted(
        (
            item
            for item in (_with_jump_datetime(record) for record in records)
            if item is not None
            and item[1].get("bucket") == "future_outside_preferred_window"
        ),
        key=lambda item: item[0],
    )
    if future:
        jump_dt, record = future[0]
        return {
            "status": "WAITING_FOR_FUTURE_WINDOW",
            "reason": "next_race_not_yet_inside_preferred_window",
            "recommended_rerun_after_local": (
                jump_dt - timedelta(minutes=max_minutes)
            ).isoformat(),
            **_window_fields(jump_dt, record),
        }

    bucket_counts = Counter(record.get("bucket") for record in records)
    if bucket_counts.get("past_or_too_close"):
        status = "NO_FUTURE_RACES_IN_PREFERRED_WINDOW"
        reason = "remaining_races_are_past_or_too_close"
    elif bucket_counts.get("missing_jump_time"):
        status = "DATA_MISSING"
        reason = "race_jump_times_missing"
    else:
        status = "NO_RACES_FOUND"
        reason = "no_races_available"
    return {
        "status": status,
        "reason": reason,
        "next_race": None,
        "next_window_opens_at": None,
        "next_window_closes_at": None,
        "minutes_until_window_opens": None,
        "minutes_until_window_closes": None,
        "recommended_rerun_after_local": None,
    }


def _artifact_counts(upcoming_dir: Path) -> dict[str, int]:
    return {
        "accepted_csv_count": len(list(upcoming_dir.glob("*.csv"))),
        "sidecar_count": len(list(upcoming_dir.glob("*.csv.metadata.json"))),
        "raw_export_count": len(list((upcoming_dir / "raw_exports").glob("*.csv")))
        if (upcoming_dir / "raw_exports").exists()
        else 0,
        "quarantine_count": len(list((upcoming_dir / "quarantine").glob("*")))
        if (upcoming_dir / "quarantine").exists()
        else 0,
    }


def refresh_prejump_upcoming(args: argparse.Namespace) -> dict[str, Any]:
    upcoming_dir = Path(args.upcoming_dir)
    if not upcoming_dir.is_absolute():
        upcoming_dir = ROOT / upcoming_dir
    upcoming_dir.mkdir(parents=True, exist_ok=True)
    os.environ["UPCOMING_RACES_DIR"] = str(upcoming_dir)

    from upcoming_race_browser import UpcomingRaceBrowser

    now = melbourne_now()
    browser = UpcomingRaceBrowser()
    races = browser.get_upcoming_races(days_ahead=int(args.days_ahead))
    excluded_race_ids = load_excluded_race_ids(
        values=list(args.exclude_race_id or []),
        file_path=args.exclude_race_ids_file,
    )
    selected, records = select_prejump_races(
        races,
        now=now,
        min_minutes=float(args.min_minutes),
        max_minutes=float(args.max_minutes),
        limit=int(args.limit),
        exclude_race_ids=excluded_race_ids,
    )

    downloads: list[dict[str, Any]] = []
    if not args.dry_run:
        for race in selected:
            result = browser.download_race_csv(str(race["url"]))
            downloads.append(
                {
                    "race_url": race.get("url"),
                    "success": bool(result.get("success")) if isinstance(result, dict) else False,
                    "result": result,
                }
            )

    bucket_counts = Counter(record["bucket"] for record in records)
    artifact_counts = _artifact_counts(upcoming_dir)
    report = {
        "status": "SUCCESS",
        "dry_run": bool(args.dry_run),
        "generated_at": now.isoformat(),
        "upcoming_dir": str(upcoming_dir),
        "days_ahead": int(args.days_ahead),
        "window": {
            "min_minutes": float(args.min_minutes),
            "max_minutes": float(args.max_minutes),
        },
        "total_races_found": len(races),
        "selected_count": len(selected),
        "excluded_race_ids": sorted(excluded_race_ids),
        "excluded_count": sum(1 for record in records if record.get("excluded_reason")),
        "bucket_counts": dict(bucket_counts),
        "next_preferred_window": refresh_timing_summary(
            records,
            min_minutes=float(args.min_minutes),
            max_minutes=float(args.max_minutes),
        ),
        "selected_races": [
            record for record in records if record["selected"]
        ][: int(args.limit) if int(args.limit) > 0 else None],
        "downloads": downloads,
        **artifact_counts,
        "artifact_counts": artifact_counts,
        "no_snapshot_persist": True,
        "no_odds_capture": True,
        "no_result_ingest": True,
        "no_label_write": True,
        "no_retrain_or_promotion": True,
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upcoming-dir", default="upcoming_races")
    parser.add_argument("--days-ahead", type=int, default=0)
    parser.add_argument("--min-minutes", type=float, default=20.0)
    parser.add_argument("--max-minutes", type=float, default=160.0)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument(
        "--exclude-race-id",
        action="append",
        default=[],
        help="Stable race ID to skip, e.g. 'Race 1 - BEN - 2026-06-10'. May repeat.",
    )
    parser.add_argument(
        "--exclude-race-ids-file",
        help="Optional newline, JSON list, or JSON object file of race IDs to skip.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = refresh_prejump_upcoming(args)
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    print(text)
    if args.output:
        out = Path(args.output)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0 if report.get("status") == "SUCCESS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
