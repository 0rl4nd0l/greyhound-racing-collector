#!/usr/bin/env python3
"""Audit immutable prospective TheDogs ``/odds`` snapshot receipts.

Each accepted snapshot is one temporal observation, irrespective of runner
count or the page's OPEN/LOW/HIGH extrema.  A race is trajectory-ready only
when all five prescribed windows contain complete, independently timed,
strictly pre-jump snapshots.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Mapping
from datetime import date, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.capture_thedogs_market_history import (
    NOMINAL_WINDOWS,
    RECEIPT_SCHEMA_VERSIONS,
    SERVER_DATE_TOLERANCE_SECONDS,
    SOURCE_CLASS,
    CaptureError,
    canonical_json_bytes,
    exact_odds_identity,
    normalize_api_snapshot,
    parse_jump_from_source,
    parse_source_runners,
    parse_timestamp,
    planned_time,
    sha256_bytes,
)

SCHEMA_VERSION = "thedogs_market_history_audit_v2"
READY = "THEDOGS_MARKET_HISTORY_CAPTURE_READY"
PARTIAL = "THEDOGS_MARKET_HISTORY_CAPTURE_PARTIAL"
BLOCKED = "BLOCKED_LIVE_CAPTURE_OR_PROVENANCE"


class AuditError(ValueError):
    """Raised when a snapshot or manifest cannot be verified safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_embedded_body(payload: Mapping[str, Any], *, field: str) -> bytes:
    encoded = payload.get("body_base64")
    if not isinstance(encoded, str) or not encoded:
        raise AuditError(f"{field}_raw_bytes_missing")
    try:
        body = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise AuditError(f"{field}_raw_bytes_invalid") from exc
    if sha256_bytes(body) != payload.get("body_sha256"):
        raise AuditError(f"{field}_raw_hash_mismatch")
    if len(body) != payload.get("body_bytes"):
        raise AuditError(f"{field}_raw_size_mismatch")
    return body


def _verify_http_receipt(
    payload: Any,
    *,
    field: str,
    expected_url: str | None = None,
    content_type: str,
    require_body: bool,
) -> bytes | None:
    if not isinstance(payload, Mapping):
        raise AuditError(f"{field}_receipt_missing")
    requested_url = str(payload.get("requested_url") or "")
    final_url = str(payload.get("final_url") or "")
    if not requested_url or requested_url != final_url:
        raise AuditError(f"{field}_redirect_or_url_mismatch")
    if expected_url is not None and requested_url != expected_url:
        raise AuditError(f"{field}_url_mismatch")
    if payload.get("status_code") != 200:
        raise AuditError(f"{field}_http_status_invalid")
    headers = payload.get("headers")
    if not isinstance(headers, Mapping):
        raise AuditError(f"{field}_http_headers_missing")
    observed_type = str(headers.get("content-type") or "").lower()
    if not observed_type.startswith(content_type):
        raise AuditError(f"{field}_content_type_invalid")
    observed_encoding = str(headers.get("content-encoding") or "").lower()
    if observed_encoding not in {"", "identity"}:
        raise AuditError(f"{field}_content_encoding_not_identity")
    start = parse_timestamp(payload.get("request_start_utc"), field=f"{field}_start")
    end = parse_timestamp(payload.get("request_end_utc"), field=f"{field}_end")
    if end < start:
        raise AuditError(f"{field}_receipt_time_order_invalid")
    server_date_text = str(headers.get("date") or "")
    if not server_date_text:
        raise AuditError(f"{field}_server_date_missing")
    try:
        server_date = parsedate_to_datetime(server_date_text).astimezone(timezone.utc)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{field}_server_date_invalid") from exc
    age_text = str(headers.get("age") or "0").strip()
    try:
        age_seconds = int(age_text)
    except ValueError as exc:
        raise AuditError(f"{field}_server_age_invalid") from exc
    if age_seconds < 0 or age_seconds > 300:
        raise AuditError(f"{field}_server_age_invalid")
    latest_server_date = server_date + timedelta(seconds=age_seconds)
    tolerance = timedelta(seconds=SERVER_DATE_TOLERANCE_SECONDS)
    if not (server_date <= end + tolerance and latest_server_date >= start - tolerance):
        raise AuditError(f"{field}_server_date_outside_request_interval")
    expected_hash = str(payload.get("body_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise AuditError(f"{field}_body_hash_invalid")
    return _decode_embedded_body(payload, field=field) if require_body else None


def _verify_api_url(url: str, expected_runner_ids: set[str]) -> None:
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "www.thedogs.com.au"
        or parsed.netloc != "www.thedogs.com.au"
        or parsed.path != "/api/runners/odds"
        or parsed.params
        or parsed.fragment
    ):
        raise AuditError("odds_api_url_invalid")
    query = parse_qs(parsed.query)
    observed_values = [str(value) for value in query.get("runner_ids[]", [])]
    observed_ids = set(observed_values)
    if (
        set(query) != {"runner_ids[]"}
        or len(observed_values) != len(observed_ids)
        or observed_ids != expected_runner_ids
    ):
        raise AuditError("odds_api_native_runner_query_mismatch")


def _finite_current_price(value: Any) -> bool:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(parsed) and parsed > 1.0


def _race_date_from_url(odds_url: str) -> date:
    identity = exact_odds_identity(odds_url)
    return date.fromisoformat(str(identity["race_date"]))


def _resolve_manifest_file(entry: Mapping[str, Any], key: str) -> Path:
    value = str(entry.get(key) or "").strip()
    if not value:
        raise AuditError(f"{key}_required")
    path = Path(value)
    if not path.is_file():
        raise AuditError(f"{key}_missing:{path}")
    return path.resolve()


def _audit_entry(
    entry: Mapping[str, Any], max_race_date: date | None
) -> dict[str, Any]:
    race_id = str(entry.get("race_id") or "").strip()
    if not race_id:
        raise AuditError("race_id_required")
    raw_path = _resolve_manifest_file(entry, "raw_html_path")
    receipt_path = _resolve_manifest_file(entry, "receipt_path")
    raw = raw_path.read_bytes()
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError("receipt_json_invalid") from exc
    if not isinstance(receipt, Mapping):
        raise AuditError("receipt_json_invalid")
    receipt_schema_version = receipt.get("schema_version")
    if receipt_schema_version not in RECEIPT_SCHEMA_VERSIONS:
        raise AuditError("receipt_schema_version_invalid")
    if receipt.get("source_class") != SOURCE_CLASS:
        raise AuditError("receipt_source_class_invalid")
    if receipt.get("race_id") != race_id:
        raise AuditError("receipt_race_id_mismatch")
    core_hash = str(receipt.get("receipt_core_sha256") or "")
    core = {key: value for key, value in receipt.items() if key != "receipt_core_sha256"}
    if core_hash != sha256_bytes(canonical_json_bytes(core)):
        raise AuditError("receipt_core_hash_mismatch")
    odds_url = str(entry.get("odds_url") or "").strip()
    if not odds_url:
        raise AuditError("manifest_exact_odds_url_required")
    try:
        identity = exact_odds_identity(odds_url)
    except CaptureError as exc:
        raise AuditError(str(exc)) from exc
    if receipt.get("race_identity") != identity:
        raise AuditError("receipt_race_identity_mismatch")
    if receipt.get("odds_url") != odds_url:
        raise AuditError("receipt_odds_url_mismatch")
    if receipt.get("race_url") != identity["race_url"]:
        raise AuditError("receipt_race_url_mismatch")
    if max_race_date is not None and _race_date_from_url(odds_url) > max_race_date:
        raise AuditError("race_outside_development_cutoff")
    raw_hash = sha256_bytes(raw)
    if raw_hash != receipt.get("raw_html_sha256"):
        raise AuditError("raw_html_hash_mismatch")
    if len(raw) != receipt.get("raw_html_bytes"):
        raise AuditError("raw_html_size_mismatch")
    odds_http = receipt.get("odds_page_http")
    _verify_http_receipt(
        odds_http,
        field="odds_page",
        expected_url=odds_url,
        content_type="text/html",
        require_body=False,
    )
    if not isinstance(odds_http, Mapping) or odds_http.get("body_sha256") != raw_hash:
        raise AuditError("odds_page_raw_hash_mismatch")
    request_start = parse_timestamp(
        receipt.get("request_start_utc"), field="request_start_utc"
    )
    request_end = parse_timestamp(
        receipt.get("request_end_utc"), field="request_end_utc"
    )
    capture_end = parse_timestamp(
        receipt.get("capture_end_utc"), field="capture_end_utc"
    )
    odds_request_start = parse_timestamp(
        odds_http.get("request_start_utc"), field="odds_page_start"
    )
    jump = parse_timestamp(receipt.get("jump_timestamp"), field="jump_timestamp")
    if request_end != capture_end:
        raise AuditError("request_end_capture_end_mismatch")
    if request_start != odds_request_start:
        raise AuditError("request_start_odds_page_start_mismatch")
    if request_end < request_start:
        raise AuditError("request_receipt_time_order_invalid")
    if capture_end >= jump:
        raise AuditError("capture_not_strictly_prejump")
    warm_meeting = receipt.get("warm_meeting_http")
    _verify_http_receipt(
        warm_meeting,
        field="warm_meeting",
        expected_url=str(identity["meeting_url"]),
        content_type="text/html",
        require_body=False,
    )
    if not isinstance(warm_meeting, Mapping):
        raise AuditError("warm_meeting_receipt_missing")
    capture_start = parse_timestamp(
        receipt.get("capture_start_utc"), field="capture_start_utc"
    )
    meeting_start = parse_timestamp(
        warm_meeting.get("request_start_utc"), field="warm_meeting_start"
    )
    if capture_start != meeting_start:
        raise AuditError("capture_start_warm_meeting_start_mismatch")
    jump_source = receipt.get("jump_source")
    jump_body = _verify_http_receipt(
        jump_source,
        field="jump_source",
        expected_url=str(identity["race_url"]),
        content_type="text/html",
        require_body=True,
    )
    assert jump_body is not None
    try:
        source_jump = parse_jump_from_source(jump_body)
    except CaptureError as exc:
        raise AuditError(str(exc)) from exc
    if source_jump != jump:
        raise AuditError("jump_source_timestamp_mismatch")
    try:
        source_runners = parse_source_runners(raw)
    except CaptureError as exc:
        raise AuditError(str(exc)) from exc
    source_all_ids = {runner.native_runner_id for runner in source_runners}
    source_active_ids = {
        runner.native_runner_id for runner in source_runners if runner.active
    }
    expected_values = entry.get("expected_active_runner_ids")
    if not isinstance(expected_values, list) or not expected_values:
        raise AuditError("expected_active_runner_ids_required")
    expected_ids = {str(value).strip() for value in expected_values}
    if "" in expected_ids or len(expected_ids) != len(expected_values):
        raise AuditError("expected_active_runner_ids_invalid")
    if expected_ids != source_active_ids:
        raise AuditError("expected_native_runner_set_mismatch")
    if set(receipt.get("active_native_runner_ids") or []) != source_active_ids:
        raise AuditError("receipt_active_native_runner_set_mismatch")
    if set(receipt.get("all_native_runner_ids") or []) != source_all_ids:
        raise AuditError("receipt_all_native_runner_set_mismatch")
    api_http = receipt.get("odds_api_http")
    api_body = _verify_http_receipt(
        api_http,
        field="odds_api",
        content_type="application/json",
        require_body=True,
    )
    assert api_body is not None
    if not isinstance(api_http, Mapping):
        raise AuditError("odds_api_receipt_missing")
    api_request_end = parse_timestamp(
        api_http.get("request_end_utc"), field="odds_api_end"
    )
    if request_end != api_request_end:
        raise AuditError("request_end_odds_api_end_mismatch")
    _verify_api_url(str(api_http.get("requested_url") or ""), source_all_ids)
    try:
        api_payload = json.loads(api_body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError("odds_api_json_invalid") from exc
    if not isinstance(api_payload, Mapping):
        raise AuditError("odds_api_json_invalid")
    try:
        normalized_rows, normalized_provider, native_race_id = normalize_api_snapshot(
            api_payload, source_runners
        )
    except CaptureError as exc:
        raise AuditError(str(exc)) from exc
    if receipt_schema_version == "thedogs_market_snapshot_receipt_v1":
        projected_rows = [
            {
                key: row[key]
                for key in (
                    "native_runner_id",
                    "runner_name",
                    "box",
                    "active",
                    "current_price",
                    "provider",
                )
            }
            for row in normalized_rows
        ]
    else:
        projected_rows = normalized_rows
    if receipt.get("runners") != projected_rows:
        raise AuditError("receipt_runner_projection_mismatch")
    if receipt.get("provider") != normalized_provider:
        raise AuditError("receipt_provider_projection_mismatch")
    if receipt.get("source_native_race_id") != native_race_id:
        raise AuditError("receipt_native_race_id_mismatch")
    if receipt.get("field_complete") is not True:
        raise AuditError("snapshot_not_marked_complete")
    active_rows = [row for row in normalized_rows if row["active"]]
    if len(active_rows) != len(source_active_ids):
        raise AuditError("active_field_incomplete")
    if any(not _finite_current_price(row.get("current_price")) for row in active_rows):
        raise AuditError("active_runner_current_price_invalid")
    nominal_window = str(receipt.get("nominal_window") or "").upper()
    if nominal_window not in NOMINAL_WINDOWS:
        raise AuditError("unsupported_nominal_window")
    expected_nominal = planned_time(jump, nominal_window)
    nominal_capture = parse_timestamp(
        receipt.get("nominal_capture_utc"), field="nominal_capture_utc"
    )
    if nominal_capture != expected_nominal:
        raise AuditError("nominal_capture_timestamp_mismatch")
    early_tolerance = receipt.get("early_tolerance_seconds")
    late_tolerance = receipt.get("late_tolerance_seconds")
    if (
        not isinstance(early_tolerance, int)
        or isinstance(early_tolerance, bool)
        or early_tolerance < 0
        or not isinstance(late_tolerance, int)
        or isinstance(late_tolerance, bool)
        or late_tolerance < 0
    ):
        raise AuditError("window_tolerance_invalid")
    if capture_end < nominal_capture - timedelta(seconds=early_tolerance):
        raise AuditError("capture_completed_before_nominal_window")
    if capture_end > nominal_capture + timedelta(seconds=late_tolerance):
        raise AuditError("capture_completed_after_nominal_window")
    if entry.get("nominal_window") not in (None, nominal_window):
        raise AuditError("manifest_nominal_window_mismatch")
    if receipt.get("open_low_high_are_temporal_observations") is not False:
        raise AuditError("open_low_high_temporal_semantics_invalid")
    provider = receipt["provider"]
    provider_classification = str(provider.get("classification") or "")
    if provider_classification not in {"provider_explicit", "provider_unknown"}:
        raise AuditError("provider_classification_invalid")
    return {
        "race_id": race_id,
        "race_identity": identity,
        "source_native_race_id": native_race_id,
        "odds_url": odds_url,
        "raw_html_path": str(raw_path),
        "receipt_path": str(receipt_path),
        "raw_html_sha256": raw_hash,
        "receipt_sha256": _sha256(receipt_path),
        "receipt_core_sha256": core_hash,
        "jump_source_sha256": sha256_bytes(jump_body),
        "odds_api_sha256": sha256_bytes(api_body),
        "nominal_window": nominal_window,
        "nominal_capture_utc": receipt.get("nominal_capture_utc"),
        "request_start_utc": receipt.get("request_start_utc"),
        "request_end_utc": receipt.get("request_end_utc"),
        "capture_end_utc": receipt.get("capture_end_utc"),
        "jump_timestamp": receipt.get("jump_timestamp"),
        "temporal_observation_count": 1,
        "snapshot_complete": True,
        "provider_classification": provider_classification,
        "provider": provider,
        "active_runner_count": len(active_rows),
        "all_runner_count": len(normalized_rows),
        "active_native_runner_ids": sorted(source_active_ids),
        "all_native_runner_ids": sorted(source_all_ids),
        "runners": projected_rows,
        "open_low_high_are_temporal_observations": False,
        "status": "ACCEPTED",
        "blockers": [],
    }


def _race_summary(race_id: str, snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in snapshots if row.get("status") == "ACCEPTED"]
    rejected = [row for row in snapshots if row.get("status") != "ACCEPTED"]
    by_window: dict[str, list[dict[str, Any]]] = {}
    for row in accepted:
        by_window.setdefault(str(row["nominal_window"]), []).append(row)
    conflicts = [
        window
        for window, rows in by_window.items()
        if len({row["receipt_sha256"] for row in rows}) > 1
    ]
    unique_by_window = {
        window: rows[0]
        for window, rows in by_window.items()
        if len({row["receipt_sha256"] for row in rows}) == 1
    }
    distinct_times = {
        str(row["request_end_utc"]) for row in unique_by_window.values()
    }
    source_native_race_ids = {
        str(row["source_native_race_id"]) for row in unique_by_window.values()
    }
    source_race_identities = {
        canonical_json_bytes(row["race_identity"]) for row in unique_by_window.values()
    }
    source_identity_conflict = (
        len(source_native_race_ids) > 1 or len(source_race_identities) > 1
    )
    runner_depth = Counter(
        runner_id
        for row in unique_by_window.values()
        for runner_id in row["active_native_runner_ids"]
    )
    accepted_windows = [
        window for window in NOMINAL_WINDOWS if window in unique_by_window
    ]
    missing_windows = [
        window for window in NOMINAL_WINDOWS if window not in unique_by_window
    ]
    trajectory_ready = (
        not conflicts
        and not rejected
        and not missing_windows
        and not source_identity_conflict
        and len(distinct_times) == len(NOMINAL_WINDOWS)
    )
    return {
        "race_id": race_id,
        "snapshot_entries": len(snapshots),
        "accepted_snapshot_entries": len(accepted),
        "rejected_snapshot_entries": len(rejected),
        "temporal_observation_count": len(distinct_times),
        "accepted_windows": accepted_windows,
        "missing_windows": missing_windows,
        "conflicting_windows": sorted(conflicts),
        "source_identity_conflict": source_identity_conflict,
        "source_native_race_ids": sorted(source_native_race_ids),
        "runner_temporal_depth": dict(sorted(runner_depth.items())),
        "trajectory_ready": trajectory_ready,
    }


def _write_report_idempotent(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise AuditError(f"conflicting_audit_report:{path}")
        return
    with path.open("xb") as handle:
        handle.write(payload)


def audit_manifest(
    manifest_path: Path,
    output_dir: Path | None = None,
    *,
    max_race_date: date | None = None,
) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise AuditError("manifest_must_be_nonempty_json_list")
    snapshots: list[dict[str, Any]] = []
    for entry in payload:
        if not isinstance(entry, Mapping):
            raise AuditError("manifest_entries_must_be_objects")
        try:
            snapshots.append(_audit_entry(entry, max_race_date))
        except (AuditError, CaptureError, OSError, UnicodeError, ValueError) as exc:
            snapshots.append(
                {
                    "race_id": str(entry.get("race_id") or ""),
                    "nominal_window": entry.get("nominal_window"),
                    "temporal_observation_count": 0,
                    "snapshot_complete": False,
                    "status": "REJECTED",
                    "blockers": [str(exc)],
                }
            )
    groups: dict[str, list[dict[str, Any]]] = {}
    for snapshot in snapshots:
        groups.setdefault(str(snapshot.get("race_id") or ""), []).append(snapshot)
    race_summary = [
        _race_summary(race_id, rows) for race_id, rows in sorted(groups.items())
    ]
    accepted_count = sum(row.get("status") == "ACCEPTED" for row in snapshots)
    rejected_count = len(snapshots) - accepted_count
    ready_count = sum(row["trajectory_ready"] for row in race_summary)
    if ready_count and ready_count == len(race_summary) and not rejected_count:
        final_status = READY
    elif accepted_count:
        final_status = PARTIAL
    else:
        final_status = BLOCKED
    report = {
        "schema_version": SCHEMA_VERSION,
        "source_class": SOURCE_CLASS,
        "manifest_path": str(manifest_path.resolve()),
        "max_race_date_inclusive": (
            max_race_date.isoformat() if max_race_date is not None else None
        ),
        "final_status": final_status,
        "snapshot_entry_count": len(snapshots),
        "accepted_snapshot_count": accepted_count,
        "rejected_snapshot_count": rejected_count,
        "unique_race_count": len(groups),
        "trajectory_ready_race_count": ready_count,
        "required_nominal_windows": list(NOMINAL_WINDOWS),
        "temporal_observation_semantics": (
            "one complete receipt-bound point-in-time snapshot at one fixed window"
        ),
        "open_low_high_are_temporal_observations": False,
        "race_summary": race_summary,
        "snapshots": snapshots,
        "no_write_guarantees": {
            "network_acquisition": False,
            "database_write": False,
            "model_write": False,
            "training": False,
            "scoring": False,
            "promotion": False,
            "betting": False,
        },
    }
    if output_dir is not None:
        encoded = (
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        ).encode("utf-8")
        _write_report_idempotent(
            output_dir / "thedogs_market_history_audit.json", encoded
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-race-date", type=date.fromisoformat)
    args = parser.parse_args()
    report = audit_manifest(
        args.manifest,
        args.output_dir,
        max_race_date=args.max_race_date,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["final_status"] != BLOCKED else 2


if __name__ == "__main__":
    raise SystemExit(main())
