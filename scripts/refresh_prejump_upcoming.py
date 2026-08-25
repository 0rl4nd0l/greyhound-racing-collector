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
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from utils.csv_metadata import (  # noqa: E402
    canonical_thedogs_race_identity,
    load_safe_weather_track_metadata,
)
from utils.expert_form_metadata import safe_expert_form_metadata_from_payload  # noqa: E402
from utils.race_lifecycle import melbourne_now  # noqa: E402
from scripts.capture_thedogs_market_history import (  # noqa: E402
    validate_primary_native_identity_evidence,
)


def parse_current_time(value: str | None) -> datetime:
    if not value:
        return melbourne_now()
    text = str(value).strip()
    if len(text) >= 5 and text[-5] in {"+", "-"} and text[-4:].isdigit():
        text = f"{text[:-5]}{text[-5:-2]}:{text[-2:]}"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=melbourne_now().tzinfo)
    return parsed


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
    priority_race_id: str | None = None,
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
    selected_pairs = sorted(
        (
            (race, record)
            for race, record in zip(races, records)
            if record["selected"] and race.get("url")
        ),
        key=lambda pair: (
            0
            if priority_race_id
            and priority_race_id
            in set(pair[1].get("race_id_aliases") or [pair[1].get("race_id")])
            else 1,
            float(pair[1].get("minutes_to_jump"))
            if isinstance(pair[1].get("minutes_to_jump"), (int, float))
            else float("inf"),
            str(pair[1].get("venue") or ""),
            int(pair[1].get("race_number") or 0),
            str(pair[1].get("race_id") or ""),
        ),
    )
    if limit and limit > 0:
        selected_pairs = selected_pairs[:limit]
    for selection_order, (_, record) in enumerate(selected_pairs, start=1):
        record["selection_order"] = selection_order
    selected = [race for race, _ in selected_pairs]
    return selected, records


def selected_prejump_records(
    records: Sequence[Mapping[str, Any]],
    *,
    limit: int = 0,
) -> list[Mapping[str, Any]]:
    selected = sorted(
        (record for record in records if record.get("selected")),
        key=lambda record: (
            int(record.get("selection_order") or 999999),
            float(record.get("minutes_to_jump"))
            if isinstance(record.get("minutes_to_jump"), (int, float))
            else float("inf"),
            str(record.get("venue") or ""),
            int(record.get("race_number") or 0),
            str(record.get("race_id") or ""),
        ),
    )
    return selected[:limit] if limit and limit > 0 else selected


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


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _sidecar_path_for_csv(csv_path: Path) -> Path:
    return csv_path.with_name(csv_path.name + ".metadata.json")


def _sidecar_race_url(payload: Mapping[str, Any] | None) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    race_info = (
        payload.get("race_info")
        if isinstance(payload.get("race_info"), Mapping)
        else {}
    )
    value = (
        payload.get("race_url")
        or race_info.get("url")
        or payload.get("metadata_source_url")
    )
    text = str(value or "").strip()
    return text or None


def _metadata_record_for_csv(csv_path: Path) -> dict[str, Any]:
    sidecar_path = _sidecar_path_for_csv(csv_path)
    payload = _read_json_object(sidecar_path) if sidecar_path.exists() else None
    weather_track = load_safe_weather_track_metadata(csv_path)
    expert_form = safe_expert_form_metadata_from_payload(payload or {})
    shadow = (
        payload.get("prejump_shadow_metadata")
        if isinstance(payload, Mapping)
        and isinstance(payload.get("prejump_shadow_metadata"), Mapping)
        else {}
    )
    runner_rows = shadow.get("runner_box_name_list")
    native_runner_ids = (
        [
            row.get("source_native_runner_id")
            for row in runner_rows
            if isinstance(row, Mapping)
        ]
        if isinstance(runner_rows, list)
        else []
    )
    native_runner_boxes = (
        [
            (
                row.get("source_native_runner_id"),
                row.get("box_number") or row.get("box"),
            )
            for row in runner_rows
            if isinstance(row, Mapping)
        ]
        if isinstance(runner_rows, list)
        else []
    )
    native_race_id = shadow.get("source_native_race_id")
    native_identity_evidence_status = "not_required_direct_source_identity"
    native_identity_evidence_reason = None
    evidence = payload.get("native_identity_evidence") if isinstance(payload, Mapping) else None
    evidence_required = (
        shadow.get("source_native_identity_origin")
        == "thedogs_odds_api_exact_runner_set"
    )
    if evidence_required and evidence is None:
        native_identity_evidence_status = "rejected"
        native_identity_evidence_reason = "native_identity_evidence_missing"
        native_race_id = None
        native_runner_ids = []
    if evidence is not None:
        evidence_valid, native_identity_evidence_reason = (
            validate_primary_native_identity_evidence(
                evidence,
                expected_race_url=str(_sidecar_race_url(payload) or ""),
                expected_native_race_id=str(native_race_id or ""),
                expected_active_runner_boxes=native_runner_boxes,
                metadata_captured_at=str(
                    shadow.get("metadata_captured_at")
                    or payload.get("metadata_captured_at")
                    or ""
                ),
            )
        )
        native_identity_evidence_status = "verified" if evidence_valid else "rejected"
        if not evidence_valid:
            native_race_id = None
            native_runner_ids = []
    weather_present = bool(weather_track.get("weather"))
    track_present = bool(weather_track.get("track_condition"))
    expert_form_safe = bool(expert_form.get("metadata_is_leakage_safe"))
    return {
        "race_id": csv_path.stem,
        "race_url": _sidecar_race_url(payload),
        "csv_path": str(csv_path),
        "sidecar_path": str(sidecar_path) if sidecar_path.exists() else None,
        "sidecar_status": "present" if payload is not None else "missing_or_unreadable",
        "safe_weather_present": weather_present,
        "safe_track_condition_present": track_present,
        "safe_both_weather_track_present": weather_present and track_present,
        "safe_expert_form_present": expert_form_safe,
        "safe_all_weather_track_expert_form_present": (
            weather_present and track_present and expert_form_safe
        ),
        "runner_source_observed_at": shadow.get("metadata_captured_at"),
        "source_native_race_id": native_race_id,
        "source_native_runner_ids": native_runner_ids,
        "native_identity_evidence_status": native_identity_evidence_status,
        "native_identity_evidence_reason": native_identity_evidence_reason,
        "weather": weather_track.get("weather"),
        "track_condition": weather_track.get("track_condition"),
        "weather_track_metadata_source": weather_track.get("weather_track_metadata_source"),
        "weather_track_metadata_source_url": weather_track.get(
            "weather_track_metadata_source_url"
        )
        or weather_track.get("metadata_source_url"),
        "weather_track_rejected_reasons": list(
            weather_track.get("rejected_weather_track_metadata_sources") or []
        ),
        "expert_form_runner_count": int(expert_form.get("runner_count") or 0),
        "expert_form_rejected_reasons": list(expert_form.get("rejected_reasons") or []),
    }


def _missing_metadata_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "race_id": record.get("race_id"),
        "race_url": record.get("race_url"),
        "csv_path": None,
        "sidecar_path": None,
        "sidecar_status": "accepted_csv_missing",
        "safe_weather_present": False,
        "safe_track_condition_present": False,
        "safe_both_weather_track_present": False,
        "safe_expert_form_present": False,
        "safe_all_weather_track_expert_form_present": False,
        "runner_source_observed_at": None,
        "weather": None,
        "track_condition": None,
        "weather_track_metadata_source": None,
        "weather_track_metadata_source_url": None,
        "weather_track_rejected_reasons": ["accepted_csv_missing"],
        "expert_form_runner_count": 0,
        "expert_form_rejected_reasons": ["accepted_csv_missing"],
    }


def sidecar_metadata_coverage(
    upcoming_dir: Path,
    selected_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarise source-safe weather/track/expert-form sidecar coverage."""

    csv_records = [
        _metadata_record_for_csv(path) for path in sorted(upcoming_dir.glob("*.csv"))
    ]
    by_url = {
        str(record["race_url"]): record
        for record in csv_records
        if record.get("race_url")
    }
    by_race_id = {
        str(record["race_id"]): record
        for record in csv_records
        if record.get("race_id")
    }
    selected_rows: list[dict[str, Any]] = []
    for selected in selected_records:
        race_url = str(selected.get("race_url") or "").strip()
        race_ids = [
            str(value)
            for value in [
                selected.get("race_id"),
                *(selected.get("race_id_aliases") or []),
            ]
            if value
        ]
        record = by_url.get(race_url) if race_url else None
        if record is None:
            record = next(
                (by_race_id[race_id] for race_id in race_ids if race_id in by_race_id),
                None,
            )
        selected_rows.append(
            dict(record) if record is not None else _missing_metadata_record(selected)
        )

    selected_count = len(selected_rows)
    safe_weather = sum(1 for row in selected_rows if row["safe_weather_present"])
    safe_track = sum(1 for row in selected_rows if row["safe_track_condition_present"])
    safe_both = sum(1 for row in selected_rows if row["safe_both_weather_track_present"])
    safe_expert = sum(1 for row in selected_rows if row["safe_expert_form_present"])
    safe_all = sum(
        1 for row in selected_rows if row["safe_all_weather_track_expert_form_present"]
    )
    accepted_csvs = sum(1 for row in selected_rows if row.get("csv_path"))
    if selected_count == 0:
        status = "NOT_REQUESTED_NO_SELECTED_RACES"
        reason = "no_selected_races"
    elif safe_all == selected_count:
        status = "READY"
        reason = None
    elif accepted_csvs == 0:
        status = "DATA_MISSING"
        reason = "no_selected_race_csv_sidecars"
    else:
        status = "PARTIAL"
        missing_parts = []
        if safe_weather < selected_count:
            missing_parts.append("weather")
        if safe_track < selected_count:
            missing_parts.append("track_condition")
        if safe_expert < selected_count:
            missing_parts.append("expert_form")
        reason = "missing_safe_" + "_".join(missing_parts)

    return {
        "schema_version": "prejump_sidecar_metadata_coverage_v1",
        "status": status,
        "reason": reason,
        "selected_race_count": selected_count,
        "accepted_selected_csv_count": accepted_csvs,
        "safe_weather_race_count": safe_weather,
        "safe_track_condition_race_count": safe_track,
        "safe_both_weather_track_race_count": safe_both,
        "safe_expert_form_race_count": safe_expert,
        "safe_all_weather_track_expert_form_race_count": safe_all,
        "races": selected_rows,
    }


def current_index_metadata_selection(
    selected_records: Sequence[Mapping[str, Any]],
    metadata_coverage: Mapping[str, Any],
    *,
    source_generated_at: datetime | str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select only source-safe races without narrowing odds-capture candidates."""

    coverage_rows = metadata_coverage.get("races")
    if not isinstance(coverage_rows, list) or len(coverage_rows) != len(selected_records):
        coverage_rows = []
    eligible: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    try:
        generated = (
            source_generated_at
            if isinstance(source_generated_at, datetime)
            else datetime.fromisoformat(str(source_generated_at or ""))
        )
    except ValueError:
        generated = None
    for index, selected in enumerate(selected_records):
        row = coverage_rows[index] if index < len(coverage_rows) else None
        selected_url = str(selected.get("race_url") or "").strip()
        selected_ids = {
            str(value)
            for value in [
                selected.get("race_id"),
                *(selected.get("race_id_aliases") or []),
            ]
            if value
        }
        row_url = str(row.get("race_url") or "").strip() if isinstance(row, Mapping) else ""
        row_id = str(row.get("race_id") or "").strip() if isinstance(row, Mapping) else ""
        selected_identity = canonical_thedogs_race_identity(selected_url)
        row_identity = canonical_thedogs_race_identity(row_url)
        aligned = bool(
            isinstance(row, Mapping)
            and selected_identity is not None
            and row_identity is not None
            and selected_identity == row_identity
            and row_id in selected_ids
        )
        safe_components = bool(
            isinstance(row, Mapping)
            and row.get("csv_path")
            and row.get("safe_weather_present") is True
            and row.get("safe_track_condition_present") is True
            and row.get("safe_expert_form_present") is True
            and row.get("safe_all_weather_track_expert_form_present") is True
        )
        try:
            observed = datetime.fromisoformat(
                str(row.get("runner_source_observed_at") or "")
            )
            jump = datetime.fromisoformat(str(selected.get("jump_datetime") or ""))
        except (AttributeError, ValueError):
            observed = None
            jump = None
        safe_runner_timing = bool(
            generated is not None
            and generated.tzinfo is not None
            and generated.utcoffset() is not None
            and observed is not None
            and observed.tzinfo is not None
            and observed.utcoffset() is not None
            and jump is not None
            and jump.tzinfo is not None
            and jump.utcoffset() is not None
            and abs((generated - observed).total_seconds()) <= 1200
            and generated < jump
            and observed < jump
        )
        native_race_id = (
            row.get("source_native_race_id") if isinstance(row, Mapping) else None
        )
        native_runner_ids = (
            row.get("source_native_runner_ids") if isinstance(row, Mapping) else None
        )
        safe_native_identity = bool(
            isinstance(native_race_id, str)
            and native_race_id.isascii()
            and native_race_id.isdecimal()
            and isinstance(native_runner_ids, list)
            and len(native_runner_ids) >= 2
            and all(
                isinstance(value, str)
                and value.isascii()
                and value.isdecimal()
                for value in native_runner_ids
            )
            and len(native_runner_ids) == len(set(native_runner_ids))
        )
        if aligned and safe_components and safe_runner_timing and safe_native_identity:
            eligible.append(
                {**dict(selected), "source_native_race_id": native_race_id}
            )
            continue
        missing = []
        if not aligned:
            missing.append("metadata_alignment")
        elif isinstance(row, Mapping):
            if not row.get("csv_path"):
                missing.append("accepted_csv")
            if not row.get("safe_weather_present"):
                missing.append("weather")
            if not row.get("safe_track_condition_present"):
                missing.append("track_condition")
            if not row.get("safe_expert_form_present"):
                missing.append("expert_form")
            if not safe_runner_timing:
                missing.append("runner_source_timing")
            if not safe_native_identity:
                missing.append("native_source_identity")
        exclusions.append(
            {
                "race_id": selected.get("race_id"),
                "race_url": selected_url or None,
                "reason": "safe_metadata_incomplete",
                "missing_safe_metadata": missing,
            }
        )

    candidate_count = len(selected_records)
    eligible_count = len(eligible)
    if candidate_count == 0:
        status = "NOT_REQUESTED_NO_SELECTED_RACES"
    elif eligible_count == 0:
        status = "INCOMPLETE"
    elif exclusions:
        status = "READY_WITH_EXCLUSIONS"
    else:
        status = "READY"
    return eligible, {
        "schema_version": "current_index_metadata_selection_v1",
        "status": status,
        "candidate_race_count": candidate_count,
        "eligible_race_count": eligible_count,
        "excluded_race_count": len(exclusions),
        "exclusions": exclusions,
    }


def refresh_prejump_upcoming(args: argparse.Namespace) -> dict[str, Any]:
    upcoming_dir = Path(args.upcoming_dir)
    if not upcoming_dir.is_absolute():
        upcoming_dir = ROOT / upcoming_dir
    upcoming_dir.mkdir(parents=True, exist_ok=True)
    os.environ["UPCOMING_RACES_DIR"] = str(upcoming_dir)

    from upcoming_race_browser import UpcomingRaceBrowser

    now = parse_current_time(getattr(args, "current_time", None))
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
        priority_race_id=getattr(args, "priority_race_id", None),
    )

    downloads: list[dict[str, Any]] = []
    if not args.dry_run:
        selected_record_by_url = {
            str(record.get("race_url") or ""): record
            for record in records
            if record.get("selected") is True and record.get("race_url")
        }
        for race in selected:
            race_url = str(race["url"])
            selected_record = selected_record_by_url.get(race_url, {})
            result = browser.download_race_csv(
                race_url,
                race_info_hint={
                    **dict(race),
                    "race_discovery_key": selected_record.get("race_id"),
                    "jump_datetime": selected_record.get("jump_datetime"),
                },
            )
            downloads.append(
                {
                    "race_url": race.get("url"),
                    "success": bool(result.get("success")) if isinstance(result, dict) else False,
                    "result": result,
                }
            )

    bucket_counts = Counter(record["bucket"] for record in records)
    artifact_counts = _artifact_counts(upcoming_dir)
    selected_records = list(selected_prejump_records(records, limit=int(args.limit)))
    metadata_coverage = sidecar_metadata_coverage(upcoming_dir, selected_records)
    current_index_races, current_index_selection = current_index_metadata_selection(
        selected_records,
        metadata_coverage,
        source_generated_at=now,
    )
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
        "priority_race_id": getattr(args, "priority_race_id", None),
        "priority_race_selected": any(
            getattr(args, "priority_race_id", None)
            in set(record.get("race_id_aliases") or [record.get("race_id")])
            for record in selected_records
        )
        if getattr(args, "priority_race_id", None)
        else None,
        "bucket_counts": dict(bucket_counts),
        "next_preferred_window": refresh_timing_summary(
            records,
            min_minutes=float(args.min_minutes),
            max_minutes=float(args.max_minutes),
        ),
        "selected_races": list(
            selected_records
        ),
        "current_index_race_count": len(current_index_races),
        "current_index_races": current_index_races,
        "current_index_metadata_selection": current_index_selection,
        "downloads": downloads,
        **artifact_counts,
        "artifact_counts": artifact_counts,
        "sidecar_metadata_coverage": metadata_coverage,
        "metadata_collection_status": metadata_coverage.get("status"),
        "no_snapshot_persist": True,
        "no_odds_capture": True,
        "no_result_ingest": True,
        "no_label_write": True,
        "no_retrain_or_promotion": True,
    }
    if (
        bool(getattr(args, "require_safe_metadata", False))
        and current_index_selection.get("status") == "INCOMPLETE"
    ):
        report["status"] = "METADATA_COVERAGE_INCOMPLETE"
        report["reason"] = metadata_coverage.get("reason") or "safe_metadata_incomplete"
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upcoming-dir", default="upcoming_races")
    parser.add_argument("--days-ahead", type=int, default=0)
    parser.add_argument("--min-minutes", type=float, default=20.0)
    parser.add_argument("--max-minutes", type=float, default=160.0)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--priority-race-id")
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
    parser.add_argument("--current-time")
    parser.add_argument(
        "--require-safe-metadata",
        action="store_true",
        help=(
            "Exclude races lacking source-safe weather, track_condition, or "
            "expert-form sidecar metadata from current-index eligibility; return "
            "a non-success status when selected races exist but none is eligible."
        ),
    )
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
