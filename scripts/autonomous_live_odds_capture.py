#!/usr/bin/env python3
"""Validated autonomous live WIN odds capture for verified pre-jump races.

This lane plans fixed T-60/T-30/T-10/T-2 capture windows from refreshed
pre-jump CSV sidecars, fetches Sportsbet odds only when explicitly executed,
validates exact runner name/box identity before any append, and writes only
append-only live_odds rows after validation passes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from accuracy_program.odds_coverage import normalize_dog_name  # noqa: E402
from odds_auto_integrator import fetch_odds_for_target_race  # noqa: E402
from utils.runner_completeness import parse_runner_rows_from_csv  # noqa: E402


DEFAULT_DB = ROOT / "greyhound_racing_data.db"
DEFAULT_OUTPUT_PARENT = ROOT / "artifacts/full_evidence_orchestration_20260525"
OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/autonomous_live_odds_capture_"
CAPTURE_WINDOWS_MINUTES = (60, 30, 10, 2)
REQUIRED_CAPTURE_MARKETS = ("win", "place")
DEFAULT_PLACE_TOPN = 3
ACCEPTED_SPORTSBET_BOX_SOURCES = {"explicit_dom", "runner_text"}
RACE_FILE_RE = re.compile(r"^Race\s+(\d+)\s+-\s+(.+)\s+-\s+(\d{4}-\d{2}-\d{2})\.csv$")
POST_RACE_SOURCE_URL_MARKERS = ("result", "results", "dividend", "payout", "sp-only")
NO_UNSAFE_WRITE_GUARANTEES = {
    "registry_mutation": False,
    "production_pointer_update": False,
    "active_model_replacement": False,
    "label_write": False,
    "prediction_snapshot_write": False,
    "historical_manifest_rewrite": False,
    "race_metadata_write": False,
    "training": False,
    "tgr_enabled": False,
    "betting_action": False,
    "ev_action": False,
}


def now_id(now: datetime | None = None) -> str:
    return (now or datetime.now().astimezone()).strftime("%Y%m%dT%H%M%S%z")


def relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        logical = path if path.is_absolute() else ROOT / path
        return logical.absolute().relative_to(ROOT.absolute()).as_posix()
    except ValueError:
        return str(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def assert_output_dir_safe(output_dir: Path) -> Path:
    logical = output_dir if output_dir.is_absolute() else ROOT / output_dir
    try:
        relative = logical.absolute().relative_to(ROOT.absolute())
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    if ".." in relative.parts:
        raise ValueError("output_dir_must_not_contain_parent_traversal")
    if not relative.as_posix().startswith(OUTPUT_PREFIX):
        raise ValueError(
            f"output_dir_must_be_autonomous_live_odds_capture_artifact:{relative}"
        )
    return logical.absolute()


def unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    for index in range(1, 1000):
        candidate = Path(f"{base}_{index:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"output_dir_collision_exhausted:{base}")


def parse_current_time(value: str | None) -> datetime:
    if not value:
        return datetime.now().astimezone()
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    if len(text) >= 5 and text[-5] in {"+", "-"} and text[-4:].isdigit():
        text = f"{text[:-5]}{text[-5:-2]}:{text[-2:]}"
    parsed = datetime.fromisoformat(text)
    return parsed.astimezone() if parsed.tzinfo is None else parsed


def parse_timestamp(value: Any, *, current_time: datetime | None = None) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        parsed = None
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            for fmt in (
                "%Y-%m-%d %I:%M %p",
                "%Y-%m-%d %I:%M:%S %p",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%d %H:%M:%S",
            ):
                try:
                    parsed = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
        if parsed is None:
            return None
    if parsed.tzinfo is None and current_time is not None and current_time.tzinfo is not None:
        return parsed.replace(tzinfo=current_time.tzinfo)
    return parsed


def seconds_between(later: datetime | None, earlier: datetime | None) -> float | None:
    if later is None or earlier is None:
        return None
    left = later
    right = earlier
    if left.tzinfo is not None and right.tzinfo is None:
        right = right.replace(tzinfo=left.tzinfo)
    elif left.tzinfo is None and right.tzinfo is not None:
        left = left.replace(tzinfo=right.tzinfo)
    elif left.tzinfo is not None and right.tzinfo is not None:
        right = right.astimezone(left.tzinfo)
    return (left - right).total_seconds()


def safe_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(str(value).strip())
    except Exception:
        return None


def safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        parsed = float(value)
    except Exception:
        return None
    return parsed if math.isfinite(parsed) else None


def source_url_host_is(url: Any, expected_fragment: str) -> bool:
    try:
        host = urlparse(str(url or "")).netloc.lower()
    except Exception:
        return False
    return expected_fragment in host


def source_url_looks_post_race(url: Any) -> bool:
    text = str(url or "").lower()
    return any(marker in text for marker in POST_RACE_SOURCE_URL_MARKERS)


def race_identity_from_csv_path(path: Path) -> dict[str, Any]:
    match = RACE_FILE_RE.match(path.name)
    if not match:
        return {"race_number": None, "venue": None, "race_date": None}
    race_number, venue, race_date = match.groups()
    return {
        "race_number": int(race_number),
        "venue": venue.strip().upper(),
        "race_date": race_date,
    }


def stable_race_id(venue: Any, race_number: Any, race_date: Any) -> str | None:
    race_no = safe_int(race_number)
    venue_text = str(venue or "").strip().upper()
    race_date_text = str(race_date or "")[:10]
    if race_no is None or not venue_text or not race_date_text:
        return None
    return f"Race {race_no} - {venue_text} - {race_date_text}"


def sidecar_payload(csv_path: Path) -> dict[str, Any]:
    return read_json(csv_path.with_name(csv_path.name + ".metadata.json"))


def shadow_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    section = payload.get("prejump_shadow_metadata")
    return dict(section) if isinstance(section, Mapping) else {}


def canonical_alignment(payload: Mapping[str, Any], meta: Mapping[str, Any]) -> dict[str, Any]:
    section = payload.get("canonical_runner_alignment")
    if not isinstance(section, Mapping):
        section = meta.get("canonical_final_runner_alignment")
    return dict(section) if isinstance(section, Mapping) else {}


def normalize_runner(dog_name: Any, box_number: Any) -> dict[str, Any] | None:
    box = safe_int(box_number)
    name = str(dog_name or "").strip()
    identity = normalize_dog_name(name)
    if box is None or not identity:
        return None
    return {"box_number": box, "dog_name": name, "identity": identity}


def runner_rows_from_sidecar(meta: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in meta.get("runner_box_name_list") or []:
        if not isinstance(item, Mapping):
            continue
        runner = normalize_runner(item.get("dog_name"), item.get("box_number") or item.get("box"))
        if runner:
            rows.append(runner)
    return rows


def runner_rows_from_csv(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in parse_runner_rows_from_csv(csv_path):
        runner = normalize_runner(item.dog_name, item.box_number)
        if runner:
            rows.append(runner)
    return rows


def duplicate_values(values: Sequence[Any]) -> list[Any]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def runner_set_report(csv_path: Path, meta: Mapping[str, Any]) -> dict[str, Any]:
    sidecar_rows = runner_rows_from_sidecar(meta)
    csv_rows = runner_rows_from_csv(csv_path)
    expected = sidecar_rows or csv_rows
    sidecar_set = {(row["box_number"], row["identity"]) for row in sidecar_rows}
    csv_set = {(row["box_number"], row["identity"]) for row in csv_rows}
    expected_set = {(row["box_number"], row["identity"]) for row in expected}
    reasons: list[str] = []
    if not sidecar_rows:
        reasons.append("sidecar_runner_box_name_list_missing")
    if not csv_rows:
        reasons.append("csv_runner_rows_missing")
    if sidecar_rows and csv_rows and sidecar_set != csv_set:
        reasons.append("csv_sidecar_runner_set_mismatch")
    duplicate_boxes = duplicate_values([row["box_number"] for row in expected])
    duplicate_names = duplicate_values([row["identity"] for row in expected])
    if duplicate_boxes:
        reasons.append("duplicate_runner_boxes")
    if duplicate_names:
        reasons.append("duplicate_runner_names")
    return {
        "status": "PASS" if not reasons else "BLOCKED",
        "reasons": reasons,
        "runner_count": len(expected),
        "sidecar_runner_count": len(sidecar_rows),
        "csv_runner_count": len(csv_rows),
        "duplicate_boxes": duplicate_boxes,
        "duplicate_names": duplicate_names,
        "runners": expected,
    }


def jump_datetime_from_metadata(
    *,
    meta: Mapping[str, Any],
    race_date: Any,
    current_time: datetime,
) -> datetime | None:
    for key in (
        "jump_datetime",
        "jump_time_iso",
        "race_jump_datetime",
        "scheduled_jump_datetime",
        "start_datetime",
    ):
        parsed = parse_timestamp(meta.get(key), current_time=current_time)
        if parsed is not None:
            return parsed
    date_text = str(meta.get("race_date") or race_date or "")[:10]
    time_text = str(meta.get("jump_time") or meta.get("race_time") or "").strip()
    if not date_text or not time_text:
        return None
    return parse_timestamp(f"{date_text} {time_text}", current_time=current_time)


def due_capture_window(minutes_to_jump: float | None) -> tuple[int | None, str | None]:
    if minutes_to_jump is None:
        return None, "jump_time_missing"
    if minutes_to_jump <= 0:
        return None, "race_already_jumped"
    if minutes_to_jump > max(CAPTURE_WINDOWS_MINUTES):
        return None, "outside_capture_horizon"
    due = min(window for window in CAPTURE_WINDOWS_MINUTES if minutes_to_jump <= window)
    return due, None


def plan_item_from_csv(
    csv_path: Path,
    *,
    current_time: datetime,
) -> dict[str, Any]:
    payload = sidecar_payload(csv_path)
    meta = shadow_metadata(payload)
    alignment = canonical_alignment(payload, meta)
    file_identity = race_identity_from_csv_path(csv_path)
    race_date = meta.get("race_date") or file_identity.get("race_date")
    venue = meta.get("venue") or file_identity.get("venue")
    race_number = meta.get("race_number") or file_identity.get("race_number")
    canonical_id = stable_race_id(venue, race_number, race_date)
    jump_at = jump_datetime_from_metadata(
        meta=meta,
        race_date=race_date,
        current_time=current_time,
    )
    seconds_to_jump = seconds_between(jump_at, current_time)
    minutes_to_jump = seconds_to_jump / 60.0 if seconds_to_jump is not None else None
    capture_window, window_skip_reason = due_capture_window(minutes_to_jump)
    runners = runner_set_report(csv_path, meta)

    reasons: list[str] = []
    if not payload:
        reasons.append("sidecar_metadata_missing")
    if meta.get("status") != "PASS":
        reasons.append("prejump_shadow_metadata_not_pass")
    if meta.get("metadata_is_leakage_safe") is not True:
        reasons.append("metadata_not_leakage_safe")
    if alignment.get("status") != "aligned":
        reasons.append("canonical_runner_alignment_not_aligned")
    if alignment.get("canonical_runner_set_status") not in {"available", None}:
        reasons.append("canonical_runner_set_not_available")
    source_url = (
        meta.get("source_url")
        or alignment.get("canonical_source_url")
        or alignment.get("source_url")
        or payload.get("metadata_source_url")
    )
    if not source_url:
        reasons.append("thedogs_source_url_missing")
    elif not source_url_host_is(source_url, "thedogs.com.au"):
        reasons.append("thedogs_source_url_untrusted")
    elif source_url_looks_post_race(source_url):
        reasons.append("thedogs_source_url_post_race")
    if canonical_id is None:
        reasons.append("canonical_race_identity_missing")
    if jump_at is None:
        reasons.append("jump_datetime_missing")
    if window_skip_reason:
        reasons.append(window_skip_reason)
    if runners["status"] != "PASS":
        reasons.extend(str(reason) for reason in runners["reasons"])

    status = "READY_TO_CAPTURE" if not reasons else "SKIPPED"
    return {
        "schema_version": "autonomous_live_odds_capture_plan_item_v1",
        "status": status,
        "skip_reasons": sorted(set(reasons)),
        "csv_path": relpath(csv_path),
        "sidecar_path": relpath(csv_path.with_name(csv_path.name + ".metadata.json")),
        "canonical_race_identity": canonical_id,
        "venue": venue,
        "race_number": safe_int(race_number),
        "race_date": str(race_date or "")[:10] or None,
        "jump_datetime": jump_at.isoformat() if jump_at else None,
        "minutes_to_jump": minutes_to_jump,
        "capture_window_minutes": capture_window,
        "capture_mode": f"autonomous_prejump_t{capture_window}m"
        if capture_window is not None
        else None,
        "thedogs_source_url": source_url,
        "runner_set_validation": {
            key: value for key, value in runners.items() if key != "runners"
        },
        "expected_runners": runners["runners"],
    }


def build_capture_plan(
    *,
    input_dirs: Sequence[Path],
    current_time: datetime,
    limit: int | None = None,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for input_dir in input_dirs:
        for csv_path in sorted(input_dir.glob("*.csv")):
            if limit is not None and len(items) >= limit:
                break
            items.append(plan_item_from_csv(csv_path, current_time=current_time))
    ready_count = sum(1 for item in items if item.get("status") == "READY_TO_CAPTURE")
    return {
        "schema_version": "autonomous_live_odds_capture_plan_v1",
        "generated_at": current_time.isoformat(),
        "input_dirs": [relpath(path) for path in input_dirs],
        "capture_window_offsets_minutes": list(CAPTURE_WINDOWS_MINUTES),
        "candidate_race_count": len(items),
        "ready_to_capture_race_count": ready_count,
        "items": items,
    }


def expected_runner_set(plan_item: Mapping[str, Any]) -> set[tuple[int, str]]:
    expected = [
        normalize_runner(row.get("dog_name"), row.get("box_number"))
        for row in plan_item.get("expected_runners") or []
        if isinstance(row, Mapping)
    ]
    return {
        (row["box_number"], row["identity"])
        for row in expected
        if row and row.get("box_number") is not None and row.get("identity")
    }


def existing_capture_status(
    db_path: Path,
    *,
    race_id: str,
    capture_mode: str,
    plan_item: Mapping[str, Any],
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "schema_version": "autonomous_live_odds_existing_capture_status_v1",
        "status": "NONE",
        "race_id": race_id,
        "capture_mode": capture_mode,
        "required_markets": list(REQUIRED_CAPTURE_MARKETS),
        "observed_markets": [],
        "existing_row_count": 0,
        "expected_runner_count": 0,
        "complete_capture_rows": 0,
        "missing_required_markets": list(REQUIRED_CAPTURE_MARKETS),
        "missing_expected_runners_by_market": {},
        "reasons": [],
    }
    if not db_path.exists():
        return status

    expected_set = expected_runner_set(plan_item)
    status["expected_runner_count"] = len(expected_set)
    if not expected_set:
        status["status"] = "INVALID"
        status["reasons"] = ["expected_runner_set_missing"]
        return status

    try:
        conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
        conn.execute("PRAGMA query_only=ON")
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(live_odds)")}
        required_columns = {"race_id", "capture_mode", "box_number"}
        if not required_columns.issubset(columns) or not (
            {"dog_name", "dog_clean_name"} & columns
        ):
            status["status"] = "INVALID"
            status["reasons"] = [
                "existing_capture_identity_columns_missing:"
                + ",".join(
                    sorted(
                        (required_columns | {"dog_name", "dog_clean_name"}) - columns
                    )
                )
            ]
            return status

        select_columns = [
            column
            for column in (
                "capture_timestamp",
                "market_type",
                "box_number",
                "dog_name",
                "dog_clean_name",
            )
            if column in columns
        ]
        rows = conn.execute(
            f"SELECT {', '.join(select_columns)} "
            "FROM live_odds WHERE race_id = ? AND capture_mode = ?",
            (race_id, capture_mode),
        ).fetchall()
    except Exception:
        return status
    finally:
        try:
            conn.close()
        except Exception:
            pass

    row_dicts = [dict(zip(select_columns, row)) for row in rows]
    status["existing_row_count"] = len(row_dicts)
    if not row_dicts:
        return status

    observed_markets = sorted(
        {
            str(row.get("market_type") or "win").strip().lower()
            for row in row_dicts
            if str(row.get("market_type") or "win").strip().lower()
        }
    )
    status["observed_markets"] = observed_markets
    status["missing_required_markets"] = [
        market for market in REQUIRED_CAPTURE_MARKETS if market not in observed_markets
    ]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in row_dicts:
        grouped.setdefault(str(row.get("capture_timestamp") or ""), []).append(row)

    group_reports: list[dict[str, Any]] = []
    for capture_timestamp, group_rows in grouped.items():
        market_sets: dict[str, set[tuple[int, str]]] = {}
        for market in REQUIRED_CAPTURE_MARKETS:
            market_set: set[tuple[int, str]] = set()
            for row in group_rows:
                row_market = str(row.get("market_type") or "win").strip().lower()
                if row_market != market:
                    continue
                runner = normalize_runner(
                    row.get("dog_clean_name") or row.get("dog_name"),
                    row.get("box_number"),
                )
                if runner:
                    market_set.add((runner["box_number"], runner["identity"]))
            market_sets[market] = market_set

        missing_by_market = {
            market: sorted(expected_set - market_set)
            for market, market_set in market_sets.items()
        }
        missing_markets = [
            market for market, market_set in market_sets.items() if not market_set
        ]
        complete = all(not missing_by_market[market] for market in REQUIRED_CAPTURE_MARKETS)
        group_reports.append(
            {
                "capture_timestamp": capture_timestamp or None,
                "status": "COMPLETE" if complete and not missing_markets else "INCOMPLETE",
                "existing_row_count": len(group_rows),
                "observed_markets": [
                    market for market, market_set in market_sets.items() if market_set
                ],
                "missing_required_markets": missing_markets,
                "missing_expected_runners_by_market": {
                    market: [
                        {"box_number": box, "identity": identity}
                        for box, identity in missing
                    ]
                    for market, missing in missing_by_market.items()
                    if missing
                },
            }
        )

    complete_groups = [group for group in group_reports if group["status"] == "COMPLETE"]
    status["capture_groups"] = sorted(
        group_reports,
        key=lambda group: str(group.get("capture_timestamp") or ""),
    )
    if complete_groups:
        selected = sorted(
            complete_groups,
            key=lambda group: str(group.get("capture_timestamp") or ""),
        )[-1]
        status.update(
            {
                "status": "COMPLETE",
                "complete_capture_rows": int(selected["existing_row_count"]),
                "missing_required_markets": [],
                "missing_expected_runners_by_market": {},
                "selected_capture_timestamp": selected.get("capture_timestamp"),
                "reasons": [],
            }
        )
        return status

    selected = sorted(
        group_reports,
        key=lambda group: str(group.get("capture_timestamp") or ""),
    )[-1]
    reasons = []
    if status["missing_required_markets"]:
        reasons.append(
            "existing_capture_missing_required_markets:"
            + ",".join(status["missing_required_markets"])
        )
    status.update(
        {
            "status": "INCOMPLETE",
            "missing_required_markets": selected["missing_required_markets"],
            "missing_expected_runners_by_market": selected[
                "missing_expected_runners_by_market"
            ],
            "selected_capture_timestamp": selected.get("capture_timestamp"),
            "reasons": reasons,
        }
    )
    return status


def existing_capture_rows(db_path: Path, race_id: str, capture_mode: str) -> int:
    if not db_path.exists():
        return 0
    try:
        conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
        conn.execute("PRAGMA query_only=ON")
        row = conn.execute(
            "SELECT COUNT(*) FROM live_odds WHERE race_id = ? AND capture_mode = ?",
            (race_id, capture_mode),
        ).fetchone()
        return int(row[0] if row else 0)
    except Exception:
        return 0
    finally:
        try:
            conn.close()
        except Exception:
            pass


def fetched_source_url(fetch_result: Mapping[str, Any]) -> str | None:
    race_info = fetch_result.get("race_info")
    if not isinstance(race_info, Mapping):
        race_info = {}
    source_url = (
        race_info.get("venue_url")
        or race_info.get("sportsbet_url")
        or race_info.get("url")
    )
    return str(source_url) if source_url not in (None, "") else None


def normalize_fetched_runner(row: Mapping[str, Any]) -> dict[str, Any]:
    dog_name = row.get("dog_name") or row.get("dog_clean_name")
    return {
        "box_number": safe_int(row.get("box_number")),
        "dog_name": str(dog_name or "").strip(),
        "identity": normalize_dog_name(dog_name),
        "odds_decimal": safe_float(row.get("odds_decimal")),
        "sportsbet_box_source": str(row.get("sportsbet_box_source") or "").strip(),
        "sportsbet_list_position": row.get("sportsbet_list_position"),
        "sportsbet_raw_runner_text": row.get("sportsbet_raw_runner_text"),
        "raw": dict(row),
    }


def fetched_place_odds_rows(fetch_result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    race_info = fetch_result.get("race_info")
    if not isinstance(race_info, Mapping):
        race_info = {}
    rows = race_info.get("odds_data_place") or fetch_result.get("odds_data_place") or []
    return [row for row in rows if isinstance(row, Mapping)]


def classify_fetched_runner_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    invalid_price_reason: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    for raw in rows:
        row = normalize_fetched_runner(raw)
        row_reasons: list[str] = []
        if row["box_number"] is None:
            row_reasons.append("sportsbet_box_missing")
        if not row["identity"]:
            row_reasons.append("sportsbet_dog_name_missing")
        if row["odds_decimal"] is None or row["odds_decimal"] <= 1.0:
            row_reasons.append(invalid_price_reason)
        if row["sportsbet_box_source"] not in ACCEPTED_SPORTSBET_BOX_SOURCES:
            row_reasons.append("ambiguous_sportsbet_box_source")
        if row_reasons:
            rejected_rows.append({**row, "reasons": row_reasons})
        else:
            accepted_rows.append(row)
    return accepted_rows, rejected_rows


def validate_fetched_odds(
    *,
    plan_item: Mapping[str, Any],
    fetch_result: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    source_url = fetched_source_url(fetch_result)
    if fetch_result.get("success") is not True:
        reasons.append("sportsbet_fetch_not_successful")
    if not source_url:
        reasons.append("sportsbet_source_url_missing")
    elif not source_url_host_is(source_url, "sportsbet.com.au"):
        reasons.append("sportsbet_source_url_untrusted")
    elif source_url_looks_post_race(source_url):
        reasons.append("sportsbet_source_url_post_race")

    expected_set = expected_runner_set(plan_item)
    expected_rows = [
        {"box_number": box, "identity": identity}
        for box, identity in sorted(expected_set)
    ]
    if not expected_rows:
        reasons.append("expected_runner_set_missing")

    raw_win_rows = [
        row for row in fetch_result.get("odds_data") or [] if isinstance(row, Mapping)
    ]
    raw_place_rows = fetched_place_odds_rows(fetch_result)
    accepted_rows, rejected_rows = classify_fetched_runner_rows(
        raw_win_rows,
        invalid_price_reason="invalid_win_odds",
    )
    accepted_place_rows, rejected_place_rows = classify_fetched_runner_rows(
        raw_place_rows,
        invalid_price_reason="invalid_place_odds",
    )

    actual_set = {(row["box_number"], row["identity"]) for row in accepted_rows}
    place_set = {(row["box_number"], row["identity"]) for row in accepted_place_rows}
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    place_missing = sorted(expected_set - place_set)
    place_extra = sorted(place_set - expected_set)
    if missing:
        reasons.append("missing_expected_sportsbet_runners")
    if extra:
        reasons.append("extra_sportsbet_runners")
    if rejected_rows:
        reasons.append("rejected_sportsbet_runner_rows")
    if len(accepted_rows) != len(expected_rows):
        reasons.append("runner_count_mismatch")
    if not raw_place_rows and len(accepted_rows) == len(expected_rows) and expected_rows:
        reasons.append("sportsbet_place_market_missing")
    elif place_missing:
        reasons.append("missing_expected_sportsbet_place_runners")
    if place_extra:
        reasons.append("extra_sportsbet_place_runners")
    if rejected_place_rows:
        reasons.append("rejected_sportsbet_place_runner_rows")
    if len(accepted_place_rows) != len(expected_rows):
        reasons.append("place_runner_count_mismatch")

    place_market_complete = bool(expected_set) and expected_set.issubset(place_set)
    partial_win_market = bool(missing and actual_set and place_market_complete)
    validation_failure_root_cause = None
    if partial_win_market:
        validation_failure_root_cause = "sportsbet_win_market_partial_but_place_complete"
        reasons.append(validation_failure_root_cause)
    elif "sportsbet_place_market_missing" in reasons:
        validation_failure_root_cause = "sportsbet_place_market_missing"
    elif fetch_result.get("success") is not True:
        validation_failure_root_cause = "sportsbet_fetch_not_successful"
    elif missing:
        validation_failure_root_cause = "missing_expected_sportsbet_runners"
    elif extra:
        validation_failure_root_cause = "extra_sportsbet_runners"
    elif place_missing:
        validation_failure_root_cause = "missing_expected_sportsbet_place_runners"
    elif place_extra:
        validation_failure_root_cause = "extra_sportsbet_place_runners"
    elif rejected_rows:
        validation_failure_root_cause = "rejected_sportsbet_runner_rows"
    elif rejected_place_rows:
        validation_failure_root_cause = "rejected_sportsbet_place_runner_rows"
    elif len(accepted_rows) != len(expected_rows):
        validation_failure_root_cause = "runner_count_mismatch"
    elif len(accepted_place_rows) != len(expected_rows):
        validation_failure_root_cause = "place_runner_count_mismatch"

    fetch_win_count = safe_int(fetch_result.get("win_count"))
    if fetch_win_count is None:
        fetch_win_count = len(raw_win_rows)
    fetch_place_count = safe_int(fetch_result.get("place_count"))
    if fetch_place_count is None:
        fetch_place_count = len(raw_place_rows)

    return {
        "schema_version": "autonomous_live_odds_fetch_validation_v1",
        "status": "PASS" if not reasons else "BLOCKED",
        "reasons": sorted(set(reasons)),
        "validation_failure_root_cause": validation_failure_root_cause,
        "sportsbet_source_url": source_url,
        "sportsbet_source_race_identity": fetch_result.get("race_id"),
        "alias_race_id": fetch_result.get("alias_race_id"),
        "fetch_win_count": fetch_win_count,
        "fetch_place_count": fetch_place_count,
        "expected_runner_count": len(expected_rows),
        "accepted_runner_count": len(accepted_rows),
        "accepted_win_count": len(accepted_rows),
        "accepted_place_count": len(accepted_place_rows),
        "rejected_runner_count": len(rejected_rows),
        "rejected_place_runner_count": len(rejected_place_rows),
        "place_market_complete_for_expected_runners": place_market_complete,
        "partial_win_market": partial_win_market,
        "missing_expected_runners": [
            {"box_number": box, "identity": identity} for box, identity in missing
        ],
        "extra_sportsbet_runners": [
            {"box_number": box, "identity": identity} for box, identity in extra
        ],
        "missing_expected_place_runners": [
            {"box_number": box, "identity": identity} for box, identity in place_missing
        ],
        "extra_sportsbet_place_runners": [
            {"box_number": box, "identity": identity} for box, identity in place_extra
        ],
        "rejected_sportsbet_rows": [
            {
                "box_number": row["box_number"],
                "dog_name": row["dog_name"],
                "identity": row["identity"],
                "sportsbet_box_source": row["sportsbet_box_source"],
                "reasons": row["reasons"],
            }
            for row in rejected_rows
        ],
        "rejected_sportsbet_place_rows": [
            {
                "box_number": row["box_number"],
                "dog_name": row["dog_name"],
                "identity": row["identity"],
                "sportsbet_box_source": row["sportsbet_box_source"],
                "reasons": row["reasons"],
            }
            for row in rejected_place_rows
        ],
        "accepted_rows": [row["raw"] for row in accepted_rows],
        "accepted_place_rows": [row["raw"] for row in accepted_place_rows],
        "race_info": fetch_result.get("race_info"),
    }


def append_validated_capture(
    *,
    db_path: Path,
    plan_item: Mapping[str, Any],
    validation: Mapping[str, Any],
    capture_timestamp: datetime,
) -> dict[str, Any]:
    from sportsbet_odds_integrator import SportsbetOddsIntegrator

    race_info = dict(validation.get("race_info") or {})
    race_info.update(
        {
            "race_id": plan_item.get("canonical_race_identity"),
            "venue": plan_item.get("venue"),
            "race_number": plan_item.get("race_number"),
            "race_date": plan_item.get("race_date"),
            "venue_url": validation.get("sportsbet_source_url"),
            "sportsbet_url": validation.get("sportsbet_source_url"),
            "preserve_race_id": True,
        }
    )
    integrator = SportsbetOddsIntegrator(
        str(db_path),
        allow_auto_scrape_odds=True,
        setup_database=True,
    )
    capture_mode = str(plan_item.get("capture_mode") or "autonomous_prejump")
    capture_ts = capture_timestamp.isoformat()
    place_report = integrator.append_pre_jump_odds_snapshot(
        race_info,
        list(validation.get("accepted_place_rows") or []),
        market_type="place",
        topN=DEFAULT_PLACE_TOPN,
        capture_mode=capture_mode,
        capture_timestamp=capture_ts,
        write_race_metadata=False,
    )
    if (
        place_report.get("status") != "SUCCESS"
        or int(place_report.get("inserted_rows") or 0) <= 0
    ):
        return {
            "status": "FAILED",
            "race_id": plan_item.get("canonical_race_identity"),
            "source_url": validation.get("sportsbet_source_url"),
            "capture_mode": capture_mode,
            "capture_timestamp": capture_ts,
            "market_types": list(REQUIRED_CAPTURE_MARKETS),
            "inserted_rows": int(place_report.get("inserted_rows") or 0),
            "win_inserted_rows": 0,
            "place_inserted_rows": int(place_report.get("inserted_rows") or 0),
            "warnings": [
                f"place:{warning}" for warning in (place_report.get("warnings") or [])
            ],
            "append_only": True,
            "market_reports": {
                "win": {
                    "status": "SKIPPED",
                    "warnings": ["place_append_failed_before_win_append"],
                    "inserted_rows": 0,
                },
                "place": place_report,
            },
        }

    win_report = integrator.append_pre_jump_odds_snapshot(
        race_info,
        list(validation.get("accepted_rows") or []),
        market_type="win",
        capture_mode=capture_mode,
        capture_timestamp=capture_ts,
        write_race_metadata=False,
    )
    warnings = [
        f"{market}:{warning}"
        for market, report in (("place", place_report), ("win", win_report))
        for warning in (report.get("warnings") or [])
    ]
    place_inserted = int(place_report.get("inserted_rows") or 0)
    win_inserted = int(win_report.get("inserted_rows") or 0)
    status = (
        "SUCCESS"
        if place_report.get("status") == "SUCCESS"
        and win_report.get("status") == "SUCCESS"
        and place_inserted > 0
        and win_inserted > 0
        else "FAILED"
    )
    return {
        "status": status,
        "race_id": plan_item.get("canonical_race_identity"),
        "source_url": validation.get("sportsbet_source_url"),
        "capture_mode": capture_mode,
        "capture_timestamp": capture_ts,
        "market_types": list(REQUIRED_CAPTURE_MARKETS),
        "inserted_rows": place_inserted + win_inserted,
        "win_inserted_rows": win_inserted,
        "place_inserted_rows": place_inserted,
        "warnings": warnings,
        "append_only": True,
        "market_reports": {
            "win": win_report,
            "place": place_report,
        },
    }


def execute_capture_plan(
    *,
    plan: Mapping[str, Any],
    db_path: Path,
    current_time: datetime,
    execute: bool,
    allow_auto_scrape_odds: bool,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    inserted_rows = 0
    fetched_count = 0
    validation_pass_count = 0
    for item in plan.get("items") or []:
        if not isinstance(item, Mapping):
            continue
        result: dict[str, Any] = {
            "schema_version": "autonomous_live_odds_capture_attempt_v1",
            "canonical_race_identity": item.get("canonical_race_identity"),
            "venue": item.get("venue"),
            "race_number": item.get("race_number"),
            "race_date": item.get("race_date"),
            "jump_datetime": item.get("jump_datetime"),
            "capture_window_minutes": item.get("capture_window_minutes"),
            "capture_mode": item.get("capture_mode"),
            "status": "SKIPPED",
            "reasons": list(item.get("skip_reasons") or []),
            "db_write_performed": False,
            "inserted_rows": 0,
        }
        if item.get("status") != "READY_TO_CAPTURE":
            results.append(result)
            continue
        if not execute:
            result["status"] = "PLANNED_NOT_EXECUTED"
            result["reasons"] = ["execute_not_requested"]
            results.append(result)
            continue
        if not allow_auto_scrape_odds:
            result["status"] = "BLOCKED"
            result["reasons"] = ["allow_auto_scrape_odds_not_set"]
            results.append(result)
            continue
        existing_status = existing_capture_status(
            db_path,
            race_id=str(item.get("canonical_race_identity") or ""),
            capture_mode=str(item.get("capture_mode") or ""),
            plan_item=item,
        )
        if int(existing_status.get("existing_row_count") or 0) > 0:
            result["existing_capture"] = existing_status
            result["existing_capture_rows"] = int(
                existing_status.get("existing_row_count") or 0
            )
        if existing_status.get("status") == "COMPLETE":
            result["status"] = "SKIPPED_ALREADY_CAPTURED"
            result["reasons"] = ["capture_window_already_appended"]
            results.append(result)
            continue
        fetch_result = fetch_odds_for_target_race(
            str(db_path),
            item.get("venue"),
            safe_int(item.get("race_number")),
            item.get("race_date"),
            allow_auto_scrape_odds=True,
        )
        fetched_count += 1
        validation = validate_fetched_odds(plan_item=item, fetch_result=fetch_result)
        result["fetch"] = {
            "success": fetch_result.get("success") is True,
            "race_id": fetch_result.get("race_id"),
            "alias_race_id": fetch_result.get("alias_race_id"),
            "win_count": fetch_result.get("win_count"),
            "place_count": fetch_result.get("place_count"),
            "discovery_method": fetch_result.get("discovery_method"),
            "warnings": list(fetch_result.get("warnings") or []),
        }
        result["validation"] = {
            key: value
            for key, value in validation.items()
            if key not in {"accepted_rows", "accepted_place_rows", "race_info"}
        }
        if validation.get("status") != "PASS":
            result["status"] = "BLOCKED_VALIDATION_FAILED"
            result["reasons"] = list(validation.get("reasons") or [])
            results.append(result)
            continue
        validation_pass_count += 1
        append_report = append_validated_capture(
            db_path=db_path,
            plan_item=item,
            validation=validation,
            capture_timestamp=current_time,
        )
        inserted = int(append_report.get("inserted_rows") or 0)
        inserted_rows += inserted
        append_success = append_report.get("status") == "SUCCESS" and inserted > 0
        result["append_report"] = append_report
        result["status"] = "CAPTURED" if append_success else "BLOCKED_APPEND_FAILED"
        result["reasons"] = list(append_report.get("warnings") or [])
        result["db_write_performed"] = inserted > 0
        result["inserted_rows"] = inserted
        results.append(result)

    ready_count = int(plan.get("ready_to_capture_race_count") or 0)
    captured_count = sum(1 for row in results if row.get("status") == "CAPTURED")
    if captured_count:
        status = "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED"
    elif ready_count and execute:
        status = "AUTONOMOUS_LIVE_ODDS_CAPTURE_BLOCKED"
    elif ready_count:
        status = "AUTONOMOUS_LIVE_ODDS_CAPTURE_READY_NOT_EXECUTED"
    else:
        status = "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS"
    return {
        "schema_version": "autonomous_live_odds_capture_report_v1",
        "generated_at": current_time.isoformat(),
        "status": status,
        "execute_requested": execute,
        "allow_auto_scrape_odds": allow_auto_scrape_odds,
        "db_path": relpath(db_path),
        "write_scope": "append_only_live_odds_rows",
        "race_metadata_write": False,
        "fixed_capture_windows": list(CAPTURE_WINDOWS_MINUTES),
        "candidate_race_count": plan.get("candidate_race_count", 0),
        "ready_to_capture_race_count": ready_count,
        "fetch_attempt_count": fetched_count,
        "validation_pass_count": validation_pass_count,
        "captured_race_count": captured_count,
        "inserted_live_odds_rows": inserted_rows,
        "db_write_performed": inserted_rows > 0,
        "authorized_append_only_live_odds_write": inserted_rows > 0,
        "attempt_status_counts": dict(Counter(str(row.get("status")) for row in results)),
        "attempts": results,
        "no_unsafe_write_guarantees": dict(NO_UNSAFE_WRITE_GUARANTEES),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    current_time = parse_current_time(args.current_time)
    output_dir = assert_output_dir_safe(
        args.output_dir
        or DEFAULT_OUTPUT_PARENT / f"autonomous_live_odds_capture_{now_id(current_time)}"
    )
    output_dir = unique_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    plan = build_capture_plan(
        input_dirs=args.input_dir,
        current_time=current_time,
        limit=args.limit,
    )
    report = execute_capture_plan(
        plan=plan,
        db_path=args.db,
        current_time=current_time,
        execute=args.execute,
        allow_auto_scrape_odds=args.allow_auto_scrape_odds,
    )
    report["output_dir"] = relpath(output_dir)
    report["plan_path"] = relpath(output_dir / "autonomous_live_odds_capture_plan.json")
    report["attempts_path"] = relpath(output_dir / "autonomous_live_odds_capture_attempts.jsonl")
    write_json(output_dir / "autonomous_live_odds_capture_plan.json", plan)
    write_jsonl(output_dir / "autonomous_live_odds_capture_attempts.jsonl", report["attempts"])
    write_json(output_dir / "autonomous_live_odds_capture_report.json", report)
    write_text(output_dir / "final_status.txt", str(report["status"]) + "\n")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--current-time")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-auto-scrape-odds", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
