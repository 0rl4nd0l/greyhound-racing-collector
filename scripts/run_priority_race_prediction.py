#!/usr/bin/env python3
"""Plan or explicitly run one lock-aware pre-jump named-race prediction."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import re
import sqlite3
import sys
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from scripts.refresh_prejump_upcoming import (  # noqa: E402
    _parse_race_jump_datetime,
    parse_current_time,
    stable_race_id,
    stable_race_id_variants,
    venue_exclusion_aliases,
)


OUTPUT_SCHEMA = "manual_priority_race_prediction_v1"
FIXED_CAPTURE_WINDOWS_MINUTES = (60, 30, 10, 2)
TERMINAL_STATUSES = {
    "PLAN_ONLY",
    "WAITING_FOR_DAEMON_LOCK",
    "WAITING_FOR_CAPTURE_WINDOW",
    "BLOCKED_RACE_NOT_FOUND",
    "BLOCKED_RACE_AMBIGUOUS",
    "BLOCKED_RACE_ALREADY_JUMPED",
    "BLOCKED_EXACT_METADATA",
    "BLOCKED_RUNNER_IDENTITY",
    "BLOCKED_ODDS_CAPTURE",
    "BLOCKED_FEATURE_SEAL",
    "BLOCKED_MANUAL_PREDICTION",
    "PREDICTION_READY",
}
DEFAULT_DB = ROOT / "greyhound_racing_data.db"
DEFAULT_MODEL_DIR = ROOT / "artifacts/frozen_models/market_form_residual_v1"
DEFAULT_EVIDENCE_ROOT = ROOT / "artifacts/full_evidence_orchestration_20260525"
DEFAULT_CAPTURE_EVIDENCE_ROOTS = (
    DEFAULT_EVIDENCE_ROOT,
    ROOT.parent
    / "greyhound-autonomous-accuracy-odds-v1-20260610"
    / "artifacts/full_evidence_orchestration_20260525",
)
CAPTURE_PLAN_SCHEMA = "autonomous_live_odds_capture_plan_v1"
CAPTURE_PLAN_ITEM_SCHEMA = "autonomous_live_odds_capture_plan_item_v1"
CAPTURE_REPORT_SCHEMAS = {
    "autonomous_live_odds_capture_report_v1",
    "autonomous_live_odds_capture_t2_miss_cause_summary_v1",
}
CAPTURE_ATTEMPT_SCHEMA = "autonomous_live_odds_capture_attempt_v1"
CAPTURE_VALIDATION_SCHEMA = "autonomous_live_odds_capture_validation_v1"
CAPTURE_HANDOFF_SCAN_LIMIT = 256
REQUIRED_HANDOFF_DB_COLUMNS = {
    "race_id",
    "box_number",
    "dog_name",
    "dog_clean_name",
    "odds_decimal",
    "source_url",
    "capture_timestamp",
    "market_type",
    "source",
    "odds_level",
    "sportsbet_box_source",
    "capture_mode",
}


class CaptureHandoffError(RuntimeError):
    """A target-bearing autonomous receipt exists but is unsafe to consume."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )


def _token(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value or "").upper())


def _race_query_tokens(race: Mapping[str, Any]) -> set[str]:
    venue = race.get("venue") or race.get("venue_name")
    number = race.get("race_number")
    name = race.get("race_name") or race.get("name")
    values = {
        stable_race_id(race),
        name,
        f"{venue} race {number}",
        f"race {number} {venue}",
        f"{venue} r{number}",
    }
    values.update(stable_race_id_variants(race))
    for alias in venue_exclusion_aliases(
        venue, source_url=race.get("url") or race.get("race_url")
    ):
        values.update(
            {f"{alias} race {number}", f"race {number} {alias}", f"{alias} r{number}"}
        )
    return {_token(value) for value in values if value}


def resolve_target_race(
    races: Sequence[Mapping[str, Any]],
    *,
    race_id: str | None,
    race_query: str | None,
) -> tuple[str, Mapping[str, Any] | None, list[str]]:
    """Resolve one exact race without silently choosing among multiple matches."""

    if bool(race_id) == bool(race_query):
        raise ValueError("exactly_one_of_race_id_or_race_required")
    query = _token(race_id or race_query)
    matches = [race for race in races if query in _race_query_tokens(race)]
    identities = sorted(
        {value for race in matches if (value := stable_race_id(race)) is not None}
    )
    if not matches:
        return "BLOCKED_RACE_NOT_FOUND", None, []
    if len(matches) != 1:
        return "BLOCKED_RACE_AMBIGUOUS", None, identities
    return "RESOLVED", matches[0], identities


def next_fixed_window(
    *, jump_datetime: datetime, current_time: datetime
) -> dict[str, Any]:
    pending = []
    for offset in FIXED_CAPTURE_WINDOWS_MINUTES:
        target = jump_datetime - timedelta(minutes=offset)
        if target >= current_time:
            pending.append((target, offset))
    if pending:
        target, offset = sorted(pending)[0]
        return {
            "next_capture_window_minutes": offset,
            "next_capture_window_at": target.isoformat(),
            "seconds_until_next_window": max(
                0.0, (target - current_time).total_seconds()
            ),
        }
    return {
        "next_capture_window_minutes": None,
        "next_capture_window_at": None,
        "seconds_until_next_window": None,
    }


def base_output(
    *, status: str, current_time: datetime, target: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"unsupported_status:{status}")
    output: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA,
        "status": status,
        "generated_at": current_time.isoformat(),
        "fixed_capture_windows_minutes": list(FIXED_CAPTURE_WINDOWS_MINUTES),
        "activation": False,
        "persisted": False,
        "result_access": False,
        "model_mutation": False,
        "threshold_mutation": False,
        "betting": False,
    }
    if target is not None:
        output["race_id"] = stable_race_id(target)
        output["race_url"] = target.get("url") or target.get("race_url")
        output["venue"] = target.get("venue") or target.get("venue_name")
        output["race_number"] = target.get("race_number")
        output["race_date"] = target.get("date") or target.get("race_date")
        output["race_time"] = target.get("race_time") or target.get("jump_time")
    return output


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _odds_token(value: float) -> str:
    """Preserve the exact parsed IEEE-754 value for receipt/SQLite comparison."""

    return value.hex()


def _json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CaptureHandoffError(f"{label}_invalid_json") from exc
    if not isinstance(value, dict):
        raise CaptureHandoffError(f"{label}_not_object")
    return value


def _aware_datetime(value: Any, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value or ""))
    except ValueError as exc:
        raise CaptureHandoffError(f"{label}_invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CaptureHandoffError(f"{label}_timezone_missing")
    return parsed


def _path_inside(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise CaptureHandoffError(f"{label}_outside_evidence_root") from exc
    if not resolved.is_file():
        raise CaptureHandoffError(f"{label}_missing")
    return resolved


def _read_once(path: Path, root: Path, label: str) -> tuple[Path, bytes]:
    resolved = _path_inside(path, root, label)
    try:
        return resolved, resolved.read_bytes()
    except OSError as exc:
        raise CaptureHandoffError(f"{label}_unreadable") from exc


def _plan_source_path(value: Any, root: Path, label: str) -> Path:
    raw = Path(str(value or ""))
    if not str(value or "").strip():
        raise CaptureHandoffError(f"{label}_missing")
    return _path_inside(raw if raw.is_absolute() else ROOT / raw, root, label)


def _trusted_venue_tokens(venue: Any) -> set[str]:
    from config.venue_mapping import get_venue_full_name, normalize_venue

    raw = str(venue or "").strip()
    aliases = set(venue_exclusion_aliases(raw))
    if raw:
        canonical_code = normalize_venue(raw)
        aliases.update({canonical_code, get_venue_full_name(canonical_code)})
    tokens = {_token(alias) for alias in aliases if _token(alias)}
    for alias in tuple(tokens):
        for prefix in ("LADBROKES", "THE"):
            if alias.startswith(prefix) and len(alias) > len(prefix):
                tokens.add(alias.removeprefix(prefix))
    return tokens


def _fixed_window_source_matches(
    *,
    source_url: Any,
    thedogs_url: Any,
    venue: Any,
    race_date: Any,
    race_number: int,
) -> bool:
    from scripts.autonomous_live_odds_capture import (
        is_sportsbet_source_url,
        sportsbet_source_url_is_post_race,
    )

    try:
        sportsbet = urlparse(str(source_url or ""))
        thedogs = urlparse(str(thedogs_url or ""))
    except ValueError:
        return False
    thedogs_host = thedogs.netloc.lower().split("@")[-1].split(":")[0]
    if not (
        is_sportsbet_source_url(source_url)
        and not sportsbet_source_url_is_post_race(source_url)
        and thedogs.scheme in {"http", "https"}
        and (thedogs_host == "thedogs.com.au" or thedogs_host.endswith(".thedogs.com.au"))
        and not sportsbet_source_url_is_post_race(thedogs_url)
        and sportsbet.username is None
        and sportsbet.password is None
        and thedogs.username is None
        and thedogs.password is None
    ):
        return False
    sportsbet_parts = [part for part in sportsbet.path.split("/") if part]
    thedogs_parts = [part for part in thedogs.path.split("/") if part]
    race_index = next(
        (
            index
            for index, part in enumerate(sportsbet_parts)
            if re.fullmatch(rf"race-{race_number}(?:-|$).*", part.lower())
        ),
        None,
    )
    try:
        racing_index = next(
            index for index, part in enumerate(thedogs_parts) if part.lower() == "racing"
        )
        thedogs_venue = _token(thedogs_parts[racing_index + 1])
        thedogs_date = thedogs_parts[racing_index + 2]
        thedogs_race_number = int(thedogs_parts[racing_index + 3])
    except (StopIteration, IndexError, ValueError):
        return False
    if (
        race_index is None
        or race_index == 0
        or not thedogs_venue
        or thedogs_date != str(race_date or "")
        or thedogs_race_number != race_number
    ):
        return False
    sportsbet_venue = _token(sportsbet_parts[race_index - 1])
    aliases = _trusted_venue_tokens(venue)
    return bool(
        sportsbet_venue
        and thedogs_venue in aliases
        and sportsbet_venue in aliases
    )


def _provider_race_id_matches(
    value: Any, *, venue: Any, race_date: Any, race_number: int
) -> bool:
    match = re.fullmatch(
        r"(.+)_([0-9]{4}-[0-9]{2}-[0-9]{2})_([0-9]+)",
        str(value or "").strip(),
    )
    if not match:
        return False
    provider_venue, provider_date, provider_number = match.groups()
    aliases = _trusted_venue_tokens(venue)
    return (
        _token(provider_venue) in aliases
        and provider_date == str(race_date or "")
        and int(provider_number) == race_number
    )


def _canonical_receipt_rows(
    validation: Mapping[str, Any], *, source_url: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for market, key, count_key in (
        ("win", "accepted_rows", "accepted_row_count"),
        ("place", "accepted_place_rows", "accepted_place_row_count"),
    ):
        raw_rows = validation.get(key)
        if not isinstance(raw_rows, list) or not raw_rows:
            raise CaptureHandoffError(f"validation_{market}_rows_missing")
        if type(validation.get(count_key)) is not int or validation[count_key] != len(raw_rows):
            raise CaptureHandoffError(f"validation_{market}_count_mismatch")
        seen: set[tuple[int, str]] = set()
        seen_boxes: set[int] = set()
        seen_identities: set[str] = set()
        for raw_row in raw_rows:
            if not isinstance(raw_row, Mapping):
                raise CaptureHandoffError(f"validation_{market}_row_invalid")
            try:
                box = int(raw_row.get("box_number"))
                odds = float(raw_row.get("odds_decimal"))
            except (TypeError, ValueError) as exc:
                raise CaptureHandoffError(f"validation_{market}_row_invalid") from exc
            identity = _token(raw_row.get("identity") or raw_row.get("dog_name"))
            declared = _token(raw_row.get("identity"))
            name_identity = _token(raw_row.get("dog_name"))
            box_source = str(raw_row.get("sportsbet_box_source") or "")
            if (
                not 1 <= box <= 10
                or not identity
                or (declared and declared != name_identity)
                or not math.isfinite(odds)
                or odds <= 1.0
                or box_source not in {"explicit_dom", "runner_text"}
                or (box, identity) in seen
                or box in seen_boxes
                or identity in seen_identities
            ):
                raise CaptureHandoffError(f"validation_{market}_row_invalid")
            seen.add((box, identity))
            seen_boxes.add(box)
            seen_identities.add(identity)
            rows.append(
                {
                    "market_type": market,
                    "box_number": box,
                    "identity": identity,
                    "odds_decimal": _odds_token(odds),
                    "source_url": source_url,
                    "sportsbet_box_source": box_source,
                }
            )
    return sorted(rows, key=lambda row: (row["market_type"], row["box_number"], row["identity"]))


def _validate_append_report(
    attempt: Mapping[str, Any],
    *,
    race_id: str,
    source_url: str,
    capture_mode: str,
    append_time: datetime,
    receipt_rows: Sequence[Mapping[str, Any]],
) -> None:
    report = attempt.get("append_report")
    if not isinstance(report, Mapping):
        raise CaptureHandoffError("append_report_missing")
    counts = {
        market: sum(row.get("market_type") == market for row in receipt_rows)
        for market in ("win", "place")
    }
    expected_total = sum(counts.values())
    exact = {
        "status": "SUCCESS",
        "race_id": race_id,
        "source_url": source_url,
        "capture_mode": capture_mode,
        "capture_timestamp": append_time.isoformat(),
    }
    if any(report.get(key) != value for key, value in exact.items()):
        raise CaptureHandoffError("append_report_identity_mismatch")
    if (
        report.get("append_only") is not True
        or report.get("warnings") != []
        or set(report.get("market_types") or []) != {"win", "place"}
        or report.get("inserted_rows") != expected_total
        or report.get("win_inserted_rows") != counts["win"]
        or report.get("place_inserted_rows") != counts["place"]
        or attempt.get("inserted_rows") != expected_total
    ):
        raise CaptureHandoffError("append_report_count_or_integrity_mismatch")
    market_reports = report.get("market_reports")
    if not isinstance(market_reports, Mapping) or set(market_reports) != {"win", "place"}:
        raise CaptureHandoffError("append_report_market_reports_mismatch")
    for market in ("win", "place"):
        market_report = market_reports.get(market)
        if not isinstance(market_report, Mapping):
            raise CaptureHandoffError("append_report_market_reports_mismatch")
        expected = {
            "status": "SUCCESS",
            "race_id": race_id,
            "source_url": source_url,
            "capture_mode": capture_mode,
            "capture_timestamp": append_time.isoformat(),
            "market_type": market,
            "inserted_rows": counts[market],
            "skipped_rows": 0,
            "warnings": [],
            "append_only": True,
        }
        if any(market_report.get(key) != value for key, value in expected.items()):
            raise CaptureHandoffError("append_report_market_integrity_mismatch")


def _validate_plan_runner_binding(
    plan_item: Mapping[str, Any], validation: Mapping[str, Any]
) -> None:
    expected_rows = plan_item.get("expected_runners")
    if not isinstance(expected_rows, list) or not expected_rows:
        raise CaptureHandoffError("plan_expected_runners_missing")
    expected: set[tuple[int, str]] = set()
    expected_boxes: set[int] = set()
    expected_identities: set[str] = set()
    for row in expected_rows:
        if not isinstance(row, Mapping):
            raise CaptureHandoffError("plan_expected_runner_invalid")
        try:
            key = (int(row.get("box_number")), _token(row.get("dog_name") or row.get("identity")))
        except (TypeError, ValueError) as exc:
            raise CaptureHandoffError("plan_expected_runner_invalid") from exc
        if (
            not 1 <= key[0] <= 10
            or not key[1]
            or key in expected
            or key[0] in expected_boxes
            or key[1] in expected_identities
        ):
            raise CaptureHandoffError("plan_expected_runner_invalid")
        expected.add(key)
        expected_boxes.add(key[0])
        expected_identities.add(key[1])
    scratched_rows = validation.get("scratched_expected_runners")
    if not isinstance(scratched_rows, list):
        raise CaptureHandoffError("validation_scratched_runners_invalid")
    scratched: set[tuple[int, str]] = set()
    for row in scratched_rows:
        if not isinstance(row, Mapping):
            raise CaptureHandoffError("validation_scratched_runners_invalid")
        try:
            key = (int(row.get("box_number")), _token(row.get("dog_name") or row.get("identity")))
        except (TypeError, ValueError) as exc:
            raise CaptureHandoffError("validation_scratched_runners_invalid") from exc
        if key not in expected or key in scratched:
            raise CaptureHandoffError("validation_scratched_runners_mismatch")
        scratched.add(key)
    active = expected - scratched
    for key in (
        "scratched_expected_runners_with_odds",
        "missing_expected_runners",
        "extra_unexpected_runners",
        "place_missing_expected_runners",
        "place_extra_unexpected_runners",
        "rejected_rows",
        "rejected_place_rows",
        "reasons",
    ):
        if validation.get(key) != []:
            raise CaptureHandoffError(f"validation_{key}_not_empty")
    if (
        validation.get("expected_runner_count") != len(expected)
        or validation.get("active_expected_runner_count") != len(active)
        or validation.get("scratched_expected_runner_count") != len(scratched)
    ):
        raise CaptureHandoffError("validation_runner_count_mismatch")
    for rows_key in ("accepted_rows", "accepted_place_rows"):
        raw_rows = validation.get(rows_key)
        if not isinstance(raw_rows, list):
            raise CaptureHandoffError("validation_runner_set_mismatch")
        actual: set[tuple[int, str]] = set()
        for row in raw_rows:
            if not isinstance(row, Mapping):
                raise CaptureHandoffError("validation_runner_set_mismatch")
            try:
                key = (
                    int(row.get("box_number")),
                    _token(row.get("identity") or row.get("dog_name")),
                )
            except (TypeError, ValueError) as exc:
                raise CaptureHandoffError("validation_runner_set_mismatch") from exc
            actual.add(key)
        if actual != active:
            raise CaptureHandoffError("validation_runner_set_mismatch")


def _verify_receipt_db_rows(
    *,
    db_path: Path,
    race_id: str,
    capture_mode: str,
    append_time: datetime,
    receipt_rows: Sequence[Mapping[str, Any]],
) -> tuple[int, str]:
    database = db_path.resolve()
    if not database.is_file():
        raise CaptureHandoffError("db_missing")
    try:
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=2.0) as conn:
            conn.execute("PRAGMA query_only = ON")
            conn.execute("BEGIN")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(live_odds)")}
            missing = sorted(REQUIRED_HANDOFF_DB_COLUMNS - columns)
            if missing:
                raise CaptureHandoffError(
                    "db_provenance_columns_missing:" + ",".join(missing)
                )
            raw_rows = conn.execute(
                "SELECT box_number, dog_name, dog_clean_name, odds_decimal, "
                "source_url, market_type, source, odds_level, sportsbet_box_source "
                "FROM live_odds WHERE race_id = ? AND capture_mode = ? "
                "AND capture_timestamp = ?",
                (race_id, capture_mode, append_time.isoformat()),
            ).fetchall()
    except CaptureHandoffError:
        raise
    except sqlite3.Error as exc:
        raise CaptureHandoffError(f"db_read_failed:{type(exc).__name__}") from exc

    actual_rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        box, dog_name, clean_name, odds, source_url, market, source, odds_level, box_source = raw
        try:
            parsed_box = int(box)
            parsed_odds = float(odds)
        except (TypeError, ValueError) as exc:
            raise CaptureHandoffError("db_row_invalid") from exc
        dog_identity = _token(dog_name)
        clean_identity = _token(clean_name)
        identity = dog_identity or clean_identity
        if (
            market not in {"win", "place"}
            or str(source or "").strip().lower() != "sportsbet"
            or str(odds_level or "").strip().lower() not in {"dog", "runner"}
            or str(box_source or "") not in {"explicit_dom", "runner_text"}
            or not identity
            or (dog_identity and clean_identity and dog_identity != clean_identity)
            or not math.isfinite(parsed_odds)
            or parsed_odds <= 1.0
        ):
            raise CaptureHandoffError("db_row_invalid")
        actual_rows.append(
            {
                "market_type": str(market),
                "box_number": parsed_box,
                "identity": identity,
                "odds_decimal": _odds_token(parsed_odds),
                "source_url": str(source_url or ""),
                "sportsbet_box_source": str(box_source),
            }
        )
    actual_rows.sort(key=lambda row: (row["market_type"], row["box_number"], row["identity"]))
    expected_rows = [dict(row) for row in receipt_rows]
    if actual_rows != expected_rows:
        raise CaptureHandoffError("db_rows_mismatch")
    encoded = canonical_json(actual_rows).encode("utf-8")
    return len(actual_rows), _sha256_bytes(encoded)


def _finalized_target_report_time(
    *,
    plan_path: Path,
    root: Path,
    race_id: str,
    capture_window_minutes: int,
    fallback_time: datetime,
) -> datetime | None:
    """Identify a finalized target report when its paired plan is unreadable."""

    marker_path = plan_path.with_name("final_status.txt")
    report_path = plan_path.with_name("autonomous_live_odds_capture_report.json")
    if not marker_path.is_file() or not report_path.is_file():
        return None
    try:
        _, marker_raw = _read_once(marker_path, root, "capture_final_status")
        if marker_raw != b"AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED\n":
            return None
        _, report_raw = _read_once(report_path, root, "capture_report")
        report = _json_object(report_raw, "capture_report")
    except CaptureHandoffError:
        return None
    attempts = report.get("attempts")
    if not isinstance(attempts, list) or not any(
        isinstance(row, Mapping)
        and row.get("race_id") == race_id
        and row.get("capture_window_minutes") == capture_window_minutes
        for row in attempts
    ):
        return None
    try:
        return _aware_datetime(report.get("generated_at"), "report_generated_at")
    except CaptureHandoffError:
        return fallback_time


def _capture_contains_outcome_key(value: Any) -> bool:
    """Reject tokenized outcome keys while allowing the fetch-result wrapper."""

    from scripts.predict_market_form_residual import (
        INDEX_FALSE_OUTCOME_MARKERS,
        _index_key_is_outcome,
        _normalized_index_key,
    )

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = _normalized_index_key(key)
            if normalized in INDEX_FALSE_OUTCOME_MARKERS and item is False:
                continue
            if normalized != "fetch_result" and _index_key_is_outcome(key):
                return True
            if _capture_contains_outcome_key(item):
                return True
        return False
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_capture_contains_outcome_key(item) for item in value)
    return False


def discover_capture_handoff(
    *,
    evidence_roots: Sequence[Path],
    db_path: Path,
    race_id: str,
    jump_datetime: datetime,
    capture_window_minutes: int,
    current_time: datetime,
    scan_limit: int = CAPTURE_HANDOFF_SCAN_LIMIT,
) -> dict[str, Any] | None:
    """Find one finalized target receipt and bind it to one read-only DB snapshot."""

    from scripts.autonomous_live_odds_capture import (
        capture_timestamp_in_window,
        fetched_source_url,
    )

    if capture_window_minutes not in FIXED_CAPTURE_WINDOWS_MINUTES:
        raise CaptureHandoffError("capture_window_not_fixed")
    if current_time >= jump_datetime:
        raise CaptureHandoffError("race_already_jumped")
    if scan_limit <= 0 or scan_limit > 2048:
        raise CaptureHandoffError("capture_scan_limit_invalid")
    roots = sorted({Path(root).resolve() for root in evidence_roots if Path(root).is_dir()})
    if not roots:
        return None
    date_tokens = {
        jump_datetime.strftime("%Y%m%d"),
        (jump_datetime - timedelta(minutes=max(FIXED_CAPTURE_WINDOWS_MINUTES))).strftime(
            "%Y%m%d"
        ),
    }
    plan_paths: list[tuple[Path, Path]] = []
    for root in roots:
        for token in date_tokens:
            plan_paths.extend(
                (root, path)
                for path in root.glob(
                    f"autonomous_live_odds_capture_{token}T*/autonomous_live_odds_capture_plan.json"
                )
            )
    plan_paths = sorted(
        {(root, path.resolve()) for root, path in plan_paths},
        key=lambda item: str(item[1]),
        reverse=True,
    )[:scan_limit]
    valid: list[dict[str, Any]] = []
    invalid: list[tuple[datetime, str]] = []
    for root, plan_path in plan_paths:
        plan_time = current_time
        try:
            _, plan_raw = _read_once(plan_path, root, "capture_plan")
            plan = _json_object(plan_raw, "capture_plan")
        except CaptureHandoffError as exc:
            target_report_time = _finalized_target_report_time(
                plan_path=plan_path,
                root=root,
                race_id=race_id,
                capture_window_minutes=capture_window_minutes,
                fallback_time=current_time,
            )
            if target_report_time is not None:
                invalid.append(
                    (target_report_time, f"capture_plan_unreadable:{exc}")
                )
            continue
        plan_rows = [
            row
            for row in plan.get("races") or []
            if isinstance(row, Mapping)
            and row.get("race_id") == race_id
            and row.get("capture_window_minutes") == capture_window_minutes
        ]
        if not plan_rows:
            continue
        try:
            plan_time = _aware_datetime(plan.get("generated_at"), "plan_generated_at")
            if (
                plan.get("schema_version") != CAPTURE_PLAN_SCHEMA
                or len(plan_rows) != 1
                or _capture_contains_outcome_key(plan)
            ):
                raise CaptureHandoffError("capture_plan_contract_mismatch")
            plan_item = plan_rows[0]
            if (
                plan_item.get("schema_version") != CAPTURE_PLAN_ITEM_SCHEMA
                or plan_item.get("status") != "READY_TO_CAPTURE"
                or plan_item.get("race_id") != race_id
                or plan_item.get("capture_window_minutes") != capture_window_minutes
                or stable_race_id(plan_item) != race_id
                or _aware_datetime(plan_item.get("jump_datetime"), "plan_jump_datetime")
                != jump_datetime
            ):
                raise CaptureHandoffError("capture_plan_item_contract_mismatch")
            report_path = plan_path.with_name("autonomous_live_odds_capture_report.json")
            if not report_path.is_file():
                continue
            final_status_path = plan_path.with_name("final_status.txt")
            if not final_status_path.is_file():
                continue
            _, final_status_raw = _read_once(
                final_status_path, root, "capture_final_status"
            )
            if final_status_raw != b"AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED\n":
                continue
            _, report_raw = _read_once(report_path, root, "capture_report")
            report = _json_object(report_raw, "capture_report")
            if _capture_contains_outcome_key(report):
                raise CaptureHandoffError("capture_report_contains_outcome")
            expected_run_id = report_path.parent.name.removeprefix(
                "autonomous_live_odds_capture_"
            )
            if (
                report.get("schema_version") not in CAPTURE_REPORT_SCHEMAS
                or report.get("run_id") != expected_run_id
                or Path(str(report.get("output_dir") or "")).resolve()
                != report_path.parent.resolve()
                or report.get("generated_at") != plan.get("generated_at")
                or report.get("final_status")
                != "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED"
                or report.get("execute") is not True
                or report.get("allow_auto_scrape_odds") is not True
            ):
                raise CaptureHandoffError("capture_report_identity_mismatch")
            matching_attempts = [
                row
                for row in report.get("attempts") or []
                if isinstance(row, Mapping)
                and row.get("race_id") == race_id
                and row.get("capture_window_minutes") == capture_window_minutes
                and row.get("status") == "APPENDED"
            ]
            if not matching_attempts:
                continue
            if len(matching_attempts) != 1:
                raise CaptureHandoffError("accepted_capture_attempt_ambiguous")
            attempt = matching_attempts[0]
            validation = attempt.get("validation")
            if (
                attempt.get("schema_version") != CAPTURE_ATTEMPT_SCHEMA
                or attempt.get("reasons") != []
                or not isinstance(validation, Mapping)
                or validation.get("schema_version") != CAPTURE_VALIDATION_SCHEMA
                or validation.get("status") != "PASS"
            ):
                raise CaptureHandoffError("capture_attempt_contract_mismatch")
            fetch_time = _aware_datetime(attempt.get("fetch_time"), "capture_fetch_time")
            append_time = _aware_datetime(attempt.get("append_time"), "capture_append_time")
            if fetch_time > append_time or append_time > current_time:
                raise CaptureHandoffError("capture_timestamp_order_invalid")
            if plan_time > fetch_time:
                raise CaptureHandoffError("capture_plan_after_fetch")
            for timestamp in (fetch_time, append_time):
                in_window, window_reason = capture_timestamp_in_window(
                    timestamp.isoformat(),
                    jump_datetime=jump_datetime,
                    capture_window_minutes=capture_window_minutes,
                )
                if not in_window:
                    raise CaptureHandoffError(
                        str(window_reason or "capture_outside_fixed_window")
                    )
            source_url = str(validation.get("source_url") or "")
            try:
                race_number = int(plan_item.get("race_number"))
            except (TypeError, ValueError) as exc:
                raise CaptureHandoffError("plan_race_number_invalid") from exc
            if not _fixed_window_source_matches(
                source_url=source_url,
                thedogs_url=plan_item.get("thedogs_source_url"),
                venue=plan_item.get("venue"),
                race_date=plan_item.get("race_date"),
                race_number=race_number,
            ):
                raise CaptureHandoffError("capture_source_race_identity_mismatch")
            _validate_plan_runner_binding(plan_item, validation)
            if validation.get("failure_root_cause") not in (None, ""):
                raise CaptureHandoffError("capture_validation_failure_present")
            receipt_rows = _canonical_receipt_rows(validation, source_url=source_url)
            fetch_result = attempt.get("fetch_result")
            expected_market_count = len(receipt_rows) // 2
            if (
                not isinstance(fetch_result, Mapping)
                or fetch_result.get("success") is not True
                or fetch_result.get("write_performed") is not False
                or fetch_result.get("warnings") != []
                or fetch_result.get("alias_race_id") != race_id
                or fetch_result.get("opt_in_source")
                != "explicit argument allow_auto_scrape_odds"
                or fetch_result.get("discovery_method")
                not in {"sportsbet_landing", "sportsbet_meeting_exact_race"}
                or not _provider_race_id_matches(
                    fetch_result.get("race_id"),
                    venue=plan_item.get("venue"),
                    race_date=plan_item.get("race_date"),
                    race_number=race_number,
                )
                or fetch_result.get("win_count") != expected_market_count
                or fetch_result.get("place_count") != expected_market_count
                or fetched_source_url(fetch_result) not in (None, source_url)
            ):
                raise CaptureHandoffError("capture_fetch_result_mismatch")
            capture_mode = f"autonomous_prejump_t{capture_window_minutes}m"
            _validate_append_report(
                attempt,
                race_id=race_id,
                source_url=source_url,
                capture_mode=capture_mode,
                append_time=append_time,
                receipt_rows=receipt_rows,
            )
            valid.append(
                {
                    "root": root,
                    "plan_path": plan_path,
                    "plan_raw": plan_raw,
                    "plan_item": plan_item,
                    "report_path": report_path,
                    "report_raw": report_raw,
                    "run_id": expected_run_id,
                    "plan_time": plan_time,
                    "append_time": append_time,
                    "capture_mode": capture_mode,
                    "receipt_rows": receipt_rows,
                }
            )
        except CaptureHandoffError as exc:
            invalid.append((plan_time, str(exc)))

    if not valid:
        if invalid:
            _, reason = sorted(invalid, key=lambda item: item[0])[-1]
            raise CaptureHandoffError(reason)
        return None
    latest_append = max(row["append_time"] for row in valid)
    selected = [row for row in valid if row["append_time"] == latest_append]
    if len(selected) != 1:
        raise CaptureHandoffError("accepted_capture_attempt_ambiguous")
    chosen = selected[0]
    newer_invalid = [reason for when, reason in invalid if when >= chosen["plan_time"]]
    if newer_invalid:
        raise CaptureHandoffError("newer_capture_candidate_invalid:" + newer_invalid[-1])

    plan_item = chosen["plan_item"]
    root = chosen["root"]
    form_path = _plan_source_path(plan_item.get("csv_path"), root, "capture_form")
    sidecar_path = _plan_source_path(
        plan_item.get("sidecar_path"), root, "capture_sidecar"
    )
    if sidecar_path != form_path.with_name(form_path.name + ".metadata.json"):
        raise CaptureHandoffError("capture_sidecar_not_adjacent")
    _, form_raw = _read_once(form_path, root, "capture_form")
    _, sidecar_raw = _read_once(sidecar_path, root, "capture_sidecar")
    db_row_count, db_rows_sha = _verify_receipt_db_rows(
        db_path=db_path,
        race_id=race_id,
        capture_mode=chosen["capture_mode"],
        append_time=chosen["append_time"],
        receipt_rows=chosen["receipt_rows"],
    )
    return {
        "schema_version": "manual_priority_capture_handoff_v1",
        "race_id": race_id,
        "run_id": chosen["run_id"],
        "capture_window_minutes": capture_window_minutes,
        "capture_mode": chosen["capture_mode"],
        "append_timestamp": chosen["append_time"].isoformat(),
        "source_report_sha256": _sha256_bytes(chosen["report_raw"]),
        "source_plan_sha256": _sha256_bytes(chosen["plan_raw"]),
        "source_form_sha256": _sha256_bytes(form_raw),
        "source_sidecar_sha256": _sha256_bytes(sidecar_raw),
        "db_row_count": db_row_count,
        "db_rows_sha256": db_rows_sha,
        "consistency_claim": "HASH_SEALED_DB_BOUND_AT_USE_TIME",
        "historical_authentication": False,
        "_report_bytes": chosen["report_raw"],
        "_plan_bytes": chosen["plan_raw"],
        "_form_bytes": form_raw,
        "_sidecar_bytes": sidecar_raw,
        "_form_name": form_path.name,
    }


def wait_for_lock_or_handoff(
    *,
    acquire: Callable[[], Any] | None,
    handoff: Callable[[float], Mapping[str, Any] | None],
    busy_type: type[BaseException],
    max_wait_seconds: float,
    poll_seconds: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[Any | None, Mapping[str, Any] | None, float, Mapping[str, Any] | None]:
    started = monotonic()
    last_details: Mapping[str, Any] | None = None
    while True:
        elapsed = monotonic() - started
        receipt = handoff(elapsed)
        if receipt is not None:
            return None, receipt, elapsed, last_details
        if acquire is not None:
            try:
                return acquire(), None, monotonic() - started, last_details
            except busy_type as exc:
                details = getattr(exc, "payload", None)
                last_details = details if isinstance(details, Mapping) else None
        else:
            last_details = {"reason": "exact_autonomous_capture_handoff_missing"}
        elapsed = monotonic() - started
        remaining = max_wait_seconds - elapsed
        if remaining <= 0:
            return None, None, elapsed, last_details
        sleeper(min(max(poll_seconds, 0.01), remaining))


def _public_handoff(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in receipt.items()
        if not str(key).startswith("_")
    }


def _score_reused_handoff(
    receipt: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    target: Mapping[str, Any],
    jump: datetime,
    current_time: datetime,
    now_provider: Callable[[], datetime],
    feature_seal_fn: Callable[..., Mapping[str, Path]],
    score_fn: Callable[..., Mapping[str, Any]],
) -> dict[str, Any]:
    append_time = _aware_datetime(receipt.get("append_timestamp"), "handoff_append_time")
    if current_time < append_time:
        output = base_output(
            status="BLOCKED_ODDS_CAPTURE", current_time=current_time, target=target
        )
        output["reason"] = "capture_handoff_from_future"
        return output
    if current_time >= jump:
        output = base_output(
            status="BLOCKED_RACE_ALREADY_JUMPED", current_time=current_time, target=target
        )
        output["reason"] = "race_jumped_before_handoff_feature_seal"
        return output
    from scripts.autonomous_live_odds_capture import due_capture_window

    current_window, _ = due_capture_window(
        (jump - current_time).total_seconds() / 60.0
    )
    if current_window != receipt.get("capture_window_minutes"):
        output = base_output(
            status="BLOCKED_ODDS_CAPTURE", current_time=current_time, target=target
        )
        output["reason"] = "capture_window_changed_before_feature_seal"
        return output
    try:
        form_name = str(receipt["_form_name"])
        if Path(form_name).name != form_name:
            raise CaptureHandoffError("handoff_form_name_invalid")
        with tempfile.TemporaryDirectory(prefix="manual-priority-handoff-") as temp_name:
            work_dir = Path(temp_name)
            form_csv = work_dir / "source" / form_name
            sidecar = form_csv.with_name(form_csv.name + ".metadata.json")
            capture_path = work_dir / "capture.json"
            plan_path = work_dir / "capture.plan.json"
            form_csv.parent.mkdir(parents=True, exist_ok=True)
            form_csv.write_bytes(bytes(receipt["_form_bytes"]))
            sidecar.write_bytes(bytes(receipt["_sidecar_bytes"]))
            capture_path.write_bytes(bytes(receipt["_report_bytes"]))
            plan_path.write_bytes(bytes(receipt["_plan_bytes"]))
            staged_hashes = {
                "source_form_sha256": _sha256(form_csv),
                "source_sidecar_sha256": _sha256(sidecar),
                "source_report_sha256": _sha256(capture_path),
                "source_plan_sha256": _sha256(plan_path),
            }
            if any(receipt.get(key) != value for key, value in staged_hashes.items()):
                raise CaptureHandoffError("handoff_staged_hash_mismatch")
            feature_time = current_time
            feature_dir = work_dir / "features"
            try:
                sealed = feature_seal_fn(
                    form_csv=form_csv,
                    db_path=Path(args.db),
                    output_dir=feature_dir,
                    current_time=feature_time,
                )
            except Exception as exc:
                output = base_output(
                    status="BLOCKED_FEATURE_SEAL",
                    current_time=current_time,
                    target=target,
                )
                output["reason"] = f"{type(exc).__name__}:{exc}"
                return output
            try:
                score_time = now_provider()
                if score_time >= jump:
                    raise RuntimeError("race_jumped_before_manual_score")
                score_window, _ = due_capture_window(
                    (jump - score_time).total_seconds() / 60.0
                )
                if score_window != receipt.get("capture_window_minutes"):
                    raise RuntimeError("capture_window_changed_before_manual_score")
                prediction = score_fn(
                    race_id=str(receipt.get("race_id")),
                    form_csv_path=form_csv,
                    sidecar_path=sidecar,
                    feature_rows_path=Path(sealed["feature_rows"]),
                    feature_manifest_path=Path(sealed["feature_manifest"]),
                    implementation_manifest_path=Path(sealed["implementation_manifest"]),
                    capture_path=capture_path,
                    model_path=Path(args.model_dir) / "model.json",
                    manifest_path=Path(args.model_dir) / "manifest.json",
                    score_timestamp=score_time,
                )
            except Exception as exc:
                output = base_output(
                    status="BLOCKED_MANUAL_PREDICTION",
                    current_time=current_time,
                    target=target,
                )
                output["reason"] = f"{type(exc).__name__}:{exc}"
                return output
            output = base_output(
                status="PREDICTION_READY", current_time=current_time, target=target
            )
            output["prediction"] = prediction
            output["inserted_live_odds_rows"] = 0
            output["capture_reused"] = True
            output["capture_handoff"] = _public_handoff(receipt)
            output["feature_packet_ephemeral"] = True
            return output
    except (KeyError, TypeError, CaptureHandoffError) as exc:
        output = base_output(
            status="BLOCKED_ODDS_CAPTURE", current_time=current_time, target=target
        )
        output["reason"] = f"capture_handoff_invalid:{exc}"
        return output


def seal_live_features(
    *,
    form_csv: Path,
    db_path: Path,
    output_dir: Path,
    current_time: datetime,
) -> dict[str, Path]:
    """Build and hash-bind fresh feature rows without running another predictor."""

    from scripts.run_feature_recovery_execution_v1 import DEFAULT_SCHEMA, load_json
    from scripts.run_shadow_non_tgr_rf_evaluation import (
        IMPLEMENTATION_FILES,
        build_live_feature_rows,
        same_distance_same_grade_history_provenance_report,
        shadow_relpath,
        validate_schema_contract,
    )

    schema = load_json(DEFAULT_SCHEMA)
    audit = validate_schema_contract(schema)
    if audit.get("status") != "PASS":
        raise RuntimeError(f"schema_contract_failed:{audit.get('fail_reasons')}")
    rows = build_live_feature_rows(
        input_paths=[form_csv], schema=schema, db_path=db_path
    )
    if not rows:
        raise RuntimeError("feature_rows_missing")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "shadow_feature_rows.json"
    manifest_path = output_dir / "shadow_manifest.json"
    history_path = output_dir / "same_distance_same_grade_history_provenance.json"
    implementation_path = output_dir / "implementation_file_manifest.json"
    _write_json(rows_path, rows)
    _write_json(
        history_path,
        same_distance_same_grade_history_provenance_report(rows),
    )
    manifest = {
        "schema_version": "shadow_live_scoring_manifest_v1",
        "generated_at": current_time.isoformat(),
        "run_started_at": current_time.isoformat(),
        "feature_freeze_timestamp": current_time.isoformat(),
        "output_mode": "shadow_only",
        "input_files": [shadow_relpath(form_csv)],
        "prediction_rows": 0,
        "feature_rows": shadow_relpath(rows_path),
        "tgr_enabled": False,
        "registry_mutation": False,
        "production_prediction_write": False,
        "odds_used_for_shadow_scoring": False,
        "betting_output": False,
        "ev_output": False,
    }
    _write_json(manifest_path, manifest)
    artifacts = {
        shadow_relpath(path): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in (rows_path, manifest_path, history_path)
    }
    implementation = {
        "schema_version": "shadow_implementation_file_manifest_v1",
        "output_dir": shadow_relpath(output_dir),
        "git_head": "manual_priority_runtime",
        "git_branch": "manual_priority_runtime",
        "implementation_files": list(IMPLEMENTATION_FILES),
        "implementation_file_hashes": {
            relative: _sha256(ROOT / relative) for relative in IMPLEMENTATION_FILES
        },
        "artifact_files": artifacts,
    }
    _write_json(implementation_path, implementation)
    return {
        "feature_rows": rows_path,
        "feature_manifest": manifest_path,
        "implementation_manifest": implementation_path,
    }


def _target_plan(plan: Mapping[str, Any], race: Mapping[str, Any]) -> dict[str, Any]:
    target_ids = stable_race_id_variants(race)
    canonical = stable_race_id(race)
    if canonical:
        target_ids.add(canonical)
    rows = [
        dict(row)
        for row in plan.get("races") or []
        if isinstance(row, Mapping) and str(row.get("race_id") or "") in target_ids
    ]
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("status") or "UNKNOWN")
        counts[key] = counts.get(key, 0) + 1
    return {
        **dict(plan),
        "races": rows,
        "status_counts": counts,
        "ready_count": counts.get("READY_TO_CAPTURE", 0),
        "limit": 1,
    }


def acquire_with_bounded_wait(
    *,
    acquire: Callable[[], Any],
    busy_type: type[BaseException],
    max_wait_seconds: float,
    poll_seconds: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[Any | None, float, Mapping[str, Any] | None]:
    started = monotonic()
    last_details: Mapping[str, Any] | None = None
    while True:
        try:
            return acquire(), monotonic() - started, last_details
        except busy_type as exc:
            details = getattr(exc, "payload", None)
            last_details = details if isinstance(details, Mapping) else None
            elapsed = monotonic() - started
            remaining = max_wait_seconds - elapsed
            if remaining <= 0:
                return None, elapsed, last_details
            sleeper(min(max(poll_seconds, 0.01), remaining))


def run_command(
    args: argparse.Namespace,
    *,
    races: Sequence[Mapping[str, Any]],
    current_time: datetime,
    refresh_fn: Callable[..., Mapping[str, Any]] | None = None,
    capture_plan_fn: Callable[..., Mapping[str, Any]] | None = None,
    capture_execute_fn: Callable[..., Mapping[str, Any]] | None = None,
    feature_seal_fn: Callable[..., Mapping[str, Path]] = seal_live_features,
    score_fn: Callable[..., Mapping[str, Any]] | None = None,
    acquire_fn: Callable[..., Any] | None = None,
    release_fn: Callable[..., Mapping[str, Any]] | None = None,
    busy_type: type[BaseException] | None = None,
    capture_handoff_fn: Callable[..., Mapping[str, Any] | None] | None = None,
    now_provider: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    now_provider = now_provider or (lambda: datetime.now().astimezone())
    resolved, target, matches = resolve_target_race(
        races, race_id=args.race_id, race_query=args.race
    )
    if resolved != "RESOLVED" or target is None:
        output = base_output(status=resolved, current_time=current_time)
        output["matching_race_ids"] = matches
        return output
    jump = _parse_race_jump_datetime(target, now=current_time)
    if jump is None:
        output = base_output(
            status="BLOCKED_EXACT_METADATA", current_time=current_time, target=target
        )
        output["reason"] = "exact_jump_timestamp_missing"
        return output
    if jump <= current_time:
        output = base_output(
            status="BLOCKED_RACE_ALREADY_JUMPED", current_time=current_time, target=target
        )
        output["jump_timestamp"] = jump.isoformat()
        return output
    if not args.execute_collection:
        output = base_output(status="PLAN_ONLY", current_time=current_time, target=target)
        output["jump_timestamp"] = jump.isoformat()
        output.update(next_fixed_window(jump_datetime=jump, current_time=current_time))
        output["collection_execution_requested"] = False
        return output
    if not args.allow_auto_scrape_odds:
        output = base_output(
            status="BLOCKED_ODDS_CAPTURE", current_time=current_time, target=target
        )
        output["reason"] = "allow_auto_scrape_odds_flag_not_set"
        return output

    if refresh_fn is None:
        from scripts.refresh_prejump_upcoming import refresh_prejump_upcoming

        refresh_fn = refresh_prejump_upcoming
    if capture_plan_fn is None or capture_execute_fn is None:
        from scripts import autonomous_live_odds_capture as capture

        capture_plan_fn = capture_plan_fn or capture.build_capture_plan
        capture_execute_fn = capture_execute_fn or capture.execute_capture_plan
    else:
        from scripts import autonomous_live_odds_capture as capture
    capture_handoff_fn = capture_handoff_fn or discover_capture_handoff
    if score_fn is None:
        from scripts.predict_market_form_residual import score_from_artifacts

        score_fn = score_from_artifacts
    if acquire_fn is None or release_fn is None or busy_type is None:
        from scripts import shadow_autopilot_daemon as daemon

        acquire_fn = acquire_fn or daemon.acquire_lock
        release_fn = release_fn or daemon.release_lock
        busy_type = busy_type or daemon.LockBusy
    run_id = f"manual_priority_{uuid.uuid4().hex}"
    lock_path = Path(args.lock_path)

    def acquire() -> Any:
        return acquire_fn(
            lock_path=lock_path,
            run_id=run_id,
            stale_after_seconds=int(args.lock_stale_seconds),
            output_dir=Path(args.lock_output_dir),
        )

    raw_evidence_roots = getattr(args, "capture_evidence_root", None)
    evidence_roots = tuple(raw_evidence_roots or DEFAULT_CAPTURE_EVIDENCE_ROOTS)

    handoff_state: dict[str, int | None] = {"capture_window_minutes": None}

    def handoff(elapsed: float) -> Mapping[str, Any] | None:
        handoff_time = current_time + timedelta(seconds=max(0.0, elapsed))
        minutes_to_jump = (jump - handoff_time).total_seconds() / 60.0
        capture_window, _ = capture.due_capture_window(minutes_to_jump)
        handoff_state["capture_window_minutes"] = capture_window
        if capture_window is None:
            return None
        return capture_handoff_fn(
            evidence_roots=evidence_roots,
            db_path=Path(args.db),
            race_id=str(stable_race_id(target)),
            jump_datetime=jump,
            capture_window_minutes=capture_window,
            current_time=handoff_time,
        )

    try:
        lock, receipt, waited, details = wait_for_lock_or_handoff(
            acquire=(
                None
                if getattr(args, "require_autonomous_handoff", False)
                else acquire
            ),
            handoff=handoff,
            busy_type=busy_type,
            max_wait_seconds=float(args.max_wait_seconds),
            poll_seconds=float(args.poll_seconds),
        )
    except CaptureHandoffError as exc:
        reason = str(exc)
        output = base_output(
            status=(
                "BLOCKED_RUNNER_IDENTITY"
                if any(token in reason for token in ("runner", "identity", "scratch", "box"))
                else "BLOCKED_ODDS_CAPTURE"
            ),
            current_time=current_time,
            target=target,
        )
        output["reason"] = f"capture_handoff_invalid:{reason}"
        return output
    if receipt is not None:
        return _score_reused_handoff(
            receipt,
            args=args,
            target=target,
            jump=jump,
            current_time=now_provider(),
            now_provider=now_provider,
            feature_seal_fn=feature_seal_fn,
            score_fn=score_fn,
        )
    if lock is None:
        if getattr(args, "require_autonomous_handoff", False):
            due_window = handoff_state["capture_window_minutes"]
            output = base_output(
                status=(
                    "WAITING_FOR_CAPTURE_WINDOW"
                    if due_window is None
                    else "BLOCKED_ODDS_CAPTURE"
                ),
                current_time=current_time,
                target=target,
            )
            output["reason"] = (
                "no_fixed_capture_window_due"
                if due_window is None
                else "exact_autonomous_capture_handoff_missing"
            )
            output["waited_seconds"] = waited
            output["capture_window_minutes"] = due_window
            return output
        output = base_output(
            status="WAITING_FOR_DAEMON_LOCK", current_time=current_time, target=target
        )
        output["waited_seconds"] = waited
        output["lock_details"] = details
        return output

    try:
        current_time = now_provider()
        if current_time >= jump:
            output = base_output(
                status="BLOCKED_RACE_ALREADY_JUMPED",
                current_time=current_time,
                target=target,
            )
            output["jump_timestamp"] = jump.isoformat()
            output["reason"] = "race_jumped_while_waiting_for_daemon_lock"
            return output
        with tempfile.TemporaryDirectory(prefix="manual-priority-race-") as temp_name:
            work_dir = Path(temp_name)
            upcoming_dir = work_dir / "upcoming"
            refresh_args = argparse.Namespace(
                upcoming_dir=upcoming_dir,
                days_ahead=int(args.days_ahead),
                min_minutes=0.0,
                max_minutes=max(1.0, float(args.days_ahead + 1) * 1440.0),
                limit=1,
                exclude_race_id=[],
                exclude_race_ids_file=None,
                include_race_id=[stable_race_id(target)],
                dry_run=False,
                current_time=current_time.isoformat(),
                require_safe_metadata=True,
            )
            try:
                refresh_report = refresh_fn(refresh_args)
            except Exception as exc:
                output = base_output(
                    status="BLOCKED_EXACT_METADATA",
                    current_time=current_time,
                    target=target,
                )
                output["reason"] = f"exact_refresh_failed:{type(exc).__name__}:{exc}"
                return output
            coverage = refresh_report.get("sidecar_metadata_coverage") or {}
            coverage_rows = coverage.get("races") if isinstance(coverage, Mapping) else []
            if (
                refresh_report.get("selected_count") != 1
                or refresh_report.get("status") != "SUCCESS"
                or coverage.get("status") != "READY"
                or not isinstance(coverage_rows, list)
                or len(coverage_rows) != 1
            ):
                output = base_output(
                    status="BLOCKED_EXACT_METADATA",
                    current_time=current_time,
                    target=target,
                )
                output["reason"] = refresh_report.get("reason") or coverage.get("reason")
                return output
            form_csv = Path(str(coverage_rows[0].get("csv_path") or ""))
            sidecar = Path(str(coverage_rows[0].get("sidecar_path") or ""))
            if not form_csv.is_file() or not sidecar.is_file():
                output = base_output(
                    status="BLOCKED_EXACT_METADATA",
                    current_time=current_time,
                    target=target,
                )
                output["reason"] = "exact_form_or_sidecar_missing"
                return output

            try:
                plan = _target_plan(
                    capture_plan_fn(
                        [upcoming_dir], current_time=current_time, limit=1
                    ),
                    target,
                )
            except Exception as exc:
                output = base_output(
                    status="BLOCKED_ODDS_CAPTURE",
                    current_time=current_time,
                    target=target,
                )
                output["reason"] = f"capture_plan_failed:{type(exc).__name__}:{exc}"
                return output
            if plan.get("ready_count") != 1:
                output = base_output(
                    status="WAITING_FOR_CAPTURE_WINDOW",
                    current_time=current_time,
                    target=target,
                )
                output["jump_timestamp"] = jump.isoformat()
                output["capture_plan_status_counts"] = plan.get("status_counts") or {}
                output.update(next_fixed_window(jump_datetime=jump, current_time=current_time))
                return output
            try:
                capture_report = capture_execute_fn(
                    plan,
                    db_path=Path(args.db),
                    current_time=current_time,
                    execute=True,
                    allow_auto_scrape_odds=True,
                    fetch_timeout_seconds=float(args.fetch_timeout_seconds),
                )
            except Exception as exc:
                output = base_output(
                    status="BLOCKED_ODDS_CAPTURE",
                    current_time=current_time,
                    target=target,
                )
                output["reason"] = f"capture_execution_failed:{type(exc).__name__}:{exc}"
                return output
            attempts = [
                row
                for row in capture_report.get("attempts") or []
                if isinstance(row, Mapping)
            ]
            appended = [row for row in attempts if row.get("status") == "APPENDED"]
            if len(appended) != 1:
                reasons = [str(reason) for row in attempts for reason in row.get("reasons") or []]
                identity_failure = any(
                    token in reason.lower()
                    for reason in reasons
                    for token in ("runner", "box", "identity", "scratch")
                )
                output = base_output(
                    status=(
                        "BLOCKED_RUNNER_IDENTITY"
                        if identity_failure
                        else "BLOCKED_ODDS_CAPTURE"
                    ),
                    current_time=current_time,
                    target=target,
                )
                output["capture_status_counts"] = capture_report.get("status_counts") or {}
                output["reasons"] = reasons
                output["idempotent_existing_capture"] = bool(
                    any(row.get("status") == "SKIPPED_ALREADY_CAPTURED" for row in attempts)
                )
                return output
            capture_path = work_dir / "capture.json"
            _write_json(capture_path, capture_report)
            feature_dir = work_dir / "features"
            feature_time = now_provider()
            if feature_time >= jump:
                output = base_output(
                    status="BLOCKED_RACE_ALREADY_JUMPED",
                    current_time=feature_time,
                    target=target,
                )
                output["jump_timestamp"] = jump.isoformat()
                output["reason"] = "race_jumped_before_feature_seal"
                return output
            try:
                sealed = feature_seal_fn(
                    form_csv=form_csv,
                    db_path=Path(args.db),
                    output_dir=feature_dir,
                    current_time=feature_time,
                )
            except Exception as exc:
                output = base_output(
                    status="BLOCKED_FEATURE_SEAL",
                    current_time=current_time,
                    target=target,
                )
                output["reason"] = f"{type(exc).__name__}:{exc}"
                return output
            try:
                score_time = now_provider()
                if score_time >= jump:
                    raise RuntimeError("race_jumped_before_manual_score")
                prediction = score_fn(
                    race_id=str(appended[0].get("race_id")),
                    form_csv_path=form_csv,
                    sidecar_path=sidecar,
                    feature_rows_path=Path(sealed["feature_rows"]),
                    feature_manifest_path=Path(sealed["feature_manifest"]),
                    implementation_manifest_path=Path(sealed["implementation_manifest"]),
                    capture_path=capture_path,
                    model_path=Path(args.model_dir) / "model.json",
                    manifest_path=Path(args.model_dir) / "manifest.json",
                    score_timestamp=score_time,
                )
            except Exception as exc:
                output = base_output(
                    status="BLOCKED_MANUAL_PREDICTION",
                    current_time=current_time,
                    target=target,
                )
                output["reason"] = f"{type(exc).__name__}:{exc}"
                return output
            output = base_output(
                status="PREDICTION_READY", current_time=current_time, target=target
            )
            output["prediction"] = prediction
            output["inserted_live_odds_rows"] = int(
                capture_report.get("inserted_live_odds_rows") or 0
            )
            output["feature_packet_ephemeral"] = True
            return output
    finally:
        release_report = release_fn(lock_path, run_id)
        if release_report.get("released") is not True:
            raise RuntimeError(f"manual_lock_release_failed:{release_report}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--race-id", help="Exact stable race ID")
    target.add_argument("--race", help="Exact named-race query, e.g. 'Sandown race 7'")
    parser.add_argument("--execute-collection", action="store_true")
    parser.add_argument("--allow-auto-scrape-odds", action="store_true")
    parser.add_argument(
        "--require-autonomous-handoff",
        action="store_true",
        help=(
            "Reuse only an already-finalized autonomous capture; never acquire "
            "the writer lock or run the direct refresh/capture path."
        ),
    )
    parser.add_argument("--days-ahead", type=int, default=1)
    parser.add_argument("--current-time")
    parser.add_argument("--max-wait-seconds", type=float, default=0.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--fetch-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--capture-evidence-root",
        action="append",
        type=Path,
        help=(
            "Root containing finalized autonomous capture directories. "
            "May be repeated; defaults to the local and retained runtime evidence roots."
        ),
    )
    parser.add_argument(
        "--lock-path",
        type=Path,
        default=ROOT
        / "artifacts/full_evidence_orchestration_20260525/shadow_autopilot_daemon_runtime/shadow_autopilot.lock",
    )
    parser.add_argument(
        "--lock-output-dir",
        type=Path,
        default=ROOT / "artifacts/full_evidence_orchestration_20260525",
    )
    parser.add_argument("--lock-stale-seconds", type=int, default=3600)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.days_ahead < 0 or not 0 <= args.max_wait_seconds <= 600:
        raise SystemExit("invalid_bounded_wait_or_days_ahead")
    current_time = parse_current_time(args.current_time)
    from upcoming_race_browser import UpcomingRaceBrowser

    try:
        with contextlib.redirect_stdout(sys.stderr):
            races = UpcomingRaceBrowser(create_upcoming_dir=False).get_upcoming_races(
                days_ahead=args.days_ahead
            )
    except Exception as exc:
        output = base_output(
            status="BLOCKED_EXACT_METADATA", current_time=current_time
        )
        output["reason"] = f"schedule_discovery_failed:{type(exc).__name__}:{exc}"
    else:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                output = run_command(args, races=races, current_time=current_time)
        except Exception as exc:
            output = base_output(
                status="BLOCKED_MANUAL_PREDICTION", current_time=current_time
            )
            output["reason"] = f"command_failed:{type(exc).__name__}:{exc}"
    print(canonical_json(output))
    return 0 if output["status"] in {"PLAN_ONLY", "PREDICTION_READY"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
