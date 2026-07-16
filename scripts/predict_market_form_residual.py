#!/usr/bin/env python3
"""Score one race from exact sealed system features and strict pre-jump odds.

The command reads already-materialized artifacts and prints canonical JSON to
stdout. It has no database, network, feature-generation, output-file, service,
activation, deployment, promotion, EV, or betting path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse
from zoneinfo import ZoneInfo


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
ROOT_TEXT = str(ROOT)
if ROOT_TEXT not in sys.path:
    sys.path.insert(0, ROOT_TEXT)

from src.predictor.market_form_residual import (  # noqa: E402
    DEFAULT_ARTIFACT_DIR,
    FEATURES,
    ResidualContractError,
    _runner_set_sha256,
    load_frozen_model,
    score_race,
)


MELBOURNE = ZoneInfo("Australia/Melbourne")
ALLOWED_BOX_SOURCES = {"explicit_dom", "runner_text"}
OUTCOME_KEYS = {
    "actual_win",
    "finish_position",
    "official_result",
    "outcome",
    "placing",
    "result",
    "winner",
    "winner_name",
    "winner_odds",
}
POST_RACE_URL_TOKENS = {
    "result",
    "results",
    "dividend",
    "dividends",
    "payout",
    "payouts",
}
SIDECAR_SCHEMA = "form_guide_download_provenance_v1"
FEATURE_MANIFEST_SCHEMA = "shadow_live_scoring_manifest_v1"
IMPLEMENTATION_MANIFEST_SCHEMA = "shadow_implementation_file_manifest_v1"
FEATURE_GENERATOR_BRANCH = "codex/greyhound-resource-isolation-20260716"
FEATURE_GENERATOR_HEAD = "aa35fa70fc49"
FEATURE_GENERATOR_FILES = [
    "scripts/run_shadow_non_tgr_rf_evaluation.py",
    "tests/test_run_shadow_non_tgr_rf_evaluation.py",
]
CAPTURE_REPORT_SCHEMAS = {
    "autonomous_live_odds_capture_report_v1",
    # Current capture reports overlay this summary after the report header.
    "autonomous_live_odds_capture_t2_miss_cause_summary_v1",
}
CAPTURE_ATTEMPT_SCHEMA = "autonomous_live_odds_capture_attempt_v1"
CAPTURE_VALIDATION_SCHEMA = "autonomous_live_odds_capture_validation_v1"
OUTPUT_SCHEMA = "manual_market_form_residual_prediction_v2"


class ManualPredictionError(RuntimeError):
    """Raised when the exact manual scoring contract fails closed."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ManualPredictionError("input_not_canonical_json") from exc
    return (encoded + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_input(path: Path, label: str) -> tuple[bytes, str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ManualPredictionError(f"{label}_unreadable") from exc
    if not raw:
        raise ManualPredictionError(f"{label}_empty")
    return raw, _sha256_bytes(raw)


def _json_value(raw: bytes, label: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManualPredictionError(f"{label}_invalid_json") from exc


def _json_object(raw: bytes, label: str) -> dict[str, Any]:
    payload = _json_value(raw, label)
    if not isinstance(payload, dict):
        raise ManualPredictionError(f"{label}_not_object")
    return payload


def _runner_token(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _contains_outcome_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).strip().lower() in OUTCOME_KEYS or _contains_outcome_key(item)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_outcome_key(item) for item in value)
    return False


def _parse_timestamp(value: Any, label: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ManualPredictionError(f"{label}_missing")
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ManualPredictionError(f"{label}_invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ManualPredictionError(f"{label}_timezone_missing")
    return parsed


def _parse_date(value: Any, label: str) -> date:
    try:
        return date.fromisoformat(str(value or "").strip())
    except ValueError as exc:
        raise ManualPredictionError(f"{label}_invalid") from exc


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ManualPredictionError(f"{label}_invalid")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ManualPredictionError(f"{label}_invalid") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ManualPredictionError(f"{label}_invalid")
    if parsed < minimum:
        raise ManualPredictionError(f"{label}_invalid")
    return parsed


def _box(value: Any, label: str) -> int:
    parsed = _integer(value, label, minimum=1)
    if parsed > 8:
        raise ManualPredictionError(f"{label}_invalid")
    return parsed


def _nullable_float(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ManualPredictionError(f"{label}_invalid")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ManualPredictionError(f"{label}_invalid") from exc
    if not math.isfinite(parsed):
        raise ManualPredictionError(f"{label}_not_finite")
    return parsed


def _url_tokens(value: Any) -> tuple[Any, set[str]]:
    text = str(value or "").strip()
    try:
        parsed = urlparse(text)
    except ValueError:
        return None, set()
    tokens = {
        token
        for token in re.split(r"[^a-z0-9]+", f"{parsed.path} {parsed.query}".lower())
        if token
    }
    return parsed, tokens


def _trusted_thedogs_url(value: Any) -> bool:
    parsed, tokens = _url_tokens(value)
    if parsed is None:
        return False
    hostname = (parsed.hostname or "").lower()
    return bool(
        parsed.scheme == "https"
        and (hostname == "thedogs.com.au" or hostname.endswith(".thedogs.com.au"))
        and parsed.path.lower().startswith("/racing/")
        and not (tokens & POST_RACE_URL_TOKENS)
    )


def _trusted_sportsbet_url(value: Any, race_id: str) -> bool:
    parsed, tokens = _url_tokens(value)
    if parsed is None:
        return False
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not (
        hostname == "sportsbet.com.au" or hostname.endswith(".sportsbet.com.au")
    ):
        return False
    if tokens & POST_RACE_URL_TOKENS:
        return False
    path_match = re.fullmatch(
        r"/greyhound-racing/.+/race-(\d+)-\d+/?", parsed.path.lower()
    )
    race_match = re.fullmatch(
        r"Race (\d+) - [A-Z0-9_]+ - \d{4}-\d{2}-\d{2}", race_id
    )
    return bool(
        path_match
        and race_match
        and int(path_match.group(1)) == int(race_match.group(1))
    )


def _distance_metres(value: Any) -> float:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*[mM]?", str(value or "").strip())
    if not match:
        raise ManualPredictionError("target_distance_invalid")
    parsed = float(match.group(1))
    if not 1.0 < parsed < 10_000.0:
        raise ManualPredictionError("target_distance_invalid")
    return parsed


def _jump_timestamp(sidecar: Mapping[str, Any]) -> datetime:
    shadow = sidecar.get("prejump_shadow_metadata")
    race_info = sidecar.get("race_info")
    if not isinstance(shadow, Mapping):
        raise ManualPredictionError("prejump_shadow_metadata_missing")
    race_info = race_info if isinstance(race_info, Mapping) else {}
    explicit = shadow.get("jump_datetime") or sidecar.get("jump_datetime")
    if explicit:
        return _parse_timestamp(explicit, "jump_timestamp")
    race_date = _parse_date(
        shadow.get("race_date") or race_info.get("date"), "target_race_date"
    )
    raw_time = str(
        shadow.get("jump_time") or race_info.get("race_time") or ""
    ).strip()
    for fmt in ("%I:%M %p", "%I:%M%p", "%H:%M"):
        try:
            parsed_time = datetime.strptime(raw_time.upper(), fmt).time()
            return datetime.combine(race_date, parsed_time, tzinfo=MELBOURNE)
        except ValueError:
            continue
    raise ManualPredictionError("jump_time_invalid")


def _sidecar_context(sidecar: Mapping[str, Any]) -> dict[str, Any]:
    if sidecar.get("schema_version") != SIDECAR_SCHEMA:
        raise ManualPredictionError("sidecar_schema_mismatch")
    if _contains_outcome_key(sidecar):
        raise ManualPredictionError("sidecar_contains_outcome_field")
    shadow = sidecar.get("prejump_shadow_metadata")
    if not isinstance(shadow, Mapping):
        raise ManualPredictionError("prejump_shadow_metadata_missing")
    if shadow.get("status") != "PASS":
        raise ManualPredictionError("prejump_shadow_metadata_not_pass")
    if (
        sidecar.get("metadata_is_leakage_safe") is not True
        or shadow.get("metadata_is_leakage_safe") is not True
    ):
        raise ManualPredictionError("sidecar_metadata_not_leakage_safe")
    race_info = sidecar.get("race_info")
    race_info = race_info if isinstance(race_info, Mapping) else {}
    source_urls = [
        str(value).strip()
        for value in (
            shadow.get("source_url"),
            sidecar.get("race_url"),
            race_info.get("url"),
        )
        if str(value or "").strip()
    ]
    if not source_urls or len(set(source_urls)) != 1:
        raise ManualPredictionError("sidecar_source_url_alias_mismatch")
    source_url = source_urls[0]
    if not _trusted_thedogs_url(source_url):
        raise ManualPredictionError("sidecar_source_url_not_trusted_thedogs")
    alignment = shadow.get("canonical_final_runner_alignment")
    if not isinstance(alignment, Mapping):
        alignment = sidecar.get("canonical_runner_alignment")
    if not isinstance(alignment, Mapping) or str(alignment.get("status")).lower() not in {
        "aligned",
        "pass",
    }:
        raise ManualPredictionError("canonical_runner_alignment_not_verified")
    participants = shadow.get("runner_box_name_list")
    if not isinstance(participants, list) or len(participants) < 2:
        raise ManualPredictionError("sidecar_runner_set_missing")
    runners: dict[int, dict[str, Any]] = {}
    identities: set[str] = set()
    for row in participants:
        if not isinstance(row, Mapping):
            raise ManualPredictionError("sidecar_runner_invalid")
        box = _box(row.get("box_number"), "sidecar_runner_box")
        name = str(row.get("dog_name") or "").strip()
        identity = _runner_token(name)
        if not identity or box in runners or identity in identities:
            raise ManualPredictionError("sidecar_runner_invalid_or_duplicate")
        identities.add(identity)
        runners[box] = {"box_number": box, "dog_name": name, "identity": identity}
    completeness = sidecar.get("runner_completeness")
    if not isinstance(completeness, Mapping):
        raise ManualPredictionError("sidecar_runner_completeness_missing")
    if completeness.get("status") != "COMPLETE":
        raise ManualPredictionError("sidecar_runner_completeness_not_complete")
    if _integer(completeness.get("runner_count"), "sidecar_runner_count", minimum=2) != len(
        runners
    ):
        raise ManualPredictionError("sidecar_runner_count_mismatch")
    complete_rows = completeness.get("participants")
    if not isinstance(complete_rows, list):
        raise ManualPredictionError("sidecar_runner_completeness_participants_missing")
    complete_set = set()
    for row in complete_rows:
        if not isinstance(row, Mapping):
            raise ManualPredictionError("sidecar_runner_completeness_participant_invalid")
        complete_set.add(
            (
                _box(row.get("box_number"), "sidecar_complete_box"),
                _runner_token(row.get("dog_name")),
            )
        )
    runner_set = {(box, str(row["identity"])) for box, row in runners.items()}
    if complete_set != runner_set or len(complete_rows) != len(complete_set):
        raise ManualPredictionError("sidecar_runner_completeness_mismatch")
    target_date = _parse_date(
        shadow.get("race_date") or race_info.get("date"), "target_race_date"
    )
    race_number = _integer(
        shadow.get("race_number") or race_info.get("race_number"),
        "target_race_number",
        minimum=1,
    )
    venue = str(shadow.get("venue") or race_info.get("venue") or "").strip().upper()
    if not venue or not re.fullmatch(r"[A-Z0-9_]+", venue):
        raise ManualPredictionError("target_venue_invalid")
    target_distance = _distance_metres(shadow.get("distance") or race_info.get("distance"))
    target_grade = str(shadow.get("grade") or race_info.get("grade") or "").strip()
    if not target_grade:
        raise ManualPredictionError("target_grade_missing")
    jump = _jump_timestamp(sidecar)
    if jump.astimezone(MELBOURNE).date() != target_date:
        raise ManualPredictionError("jump_date_target_date_mismatch")
    return {
        "shadow": shadow,
        "runners": runners,
        "jump_timestamp": jump,
        "metadata_timestamp": _parse_timestamp(
            shadow.get("metadata_captured_at"), "metadata_capture_timestamp"
        ),
        "target_race_date": target_date,
        "target_race_number": race_number,
        "target_distance": target_distance,
        "target_venue": venue,
        "target_grade": target_grade,
        "source_url": str(source_url),
        "expected_race_id": f"Race {race_number} - {venue} - {target_date.isoformat()}",
    }


def _validate_form_binding(
    sidecar: Mapping[str, Any],
    *,
    form_csv_path: Path,
    form_raw: bytes,
    form_sha: str,
) -> None:
    if (
        sidecar.get("filename") != form_csv_path.name
        or _integer(sidecar.get("content_length"), "sidecar_content_length", minimum=1)
        != len(form_raw)
        or sidecar.get("content_sha256") != form_sha
    ):
        raise ManualPredictionError("form_csv_sidecar_hash_mismatch")


def _capture_attempts(raw: bytes, *, jsonl: bool) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    if jsonl:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ManualPredictionError("capture_jsonl_invalid_encoding") from exc
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManualPredictionError(f"capture_jsonl_invalid:{line_number}") from exc
            if not isinstance(row, dict):
                raise ManualPredictionError(f"capture_jsonl_row_not_object:{line_number}")
            attempts.append(row)
        payload_for_outcome_check: Any = attempts
    else:
        payload = _json_object(raw, "capture")
        payload_for_outcome_check = payload
        if isinstance(payload.get("attempts"), list):
            if payload.get("schema_version") not in CAPTURE_REPORT_SCHEMAS:
                raise ManualPredictionError("capture_report_schema_mismatch")
            for row in payload["attempts"]:
                if not isinstance(row, dict):
                    raise ManualPredictionError("capture_attempt_not_object")
                attempts.append(row)
        elif "race_id" in payload and "validation" in payload:
            attempts = [payload]
        else:
            raise ManualPredictionError("capture_attempts_missing")
    if _contains_outcome_key(payload_for_outcome_check):
        raise ManualPredictionError("capture_contains_outcome_field")
    return attempts


def _select_capture_attempt(raw: bytes, *, jsonl: bool, race_id: str) -> dict[str, Any]:
    matches = []
    for attempt in _capture_attempts(raw, jsonl=jsonl):
        validation = attempt.get("validation")
        if (
            attempt.get("race_id") == race_id
            and attempt.get("status") == "APPENDED"
            and isinstance(validation, Mapping)
            and validation.get("status") == "PASS"
        ):
            matches.append(attempt)
    if not matches:
        raise ManualPredictionError("accepted_capture_attempt_missing")
    if len(matches) != 1:
        raise ManualPredictionError("accepted_capture_attempt_ambiguous")
    attempt = matches[0]
    if attempt.get("schema_version") != CAPTURE_ATTEMPT_SCHEMA:
        raise ManualPredictionError("capture_attempt_schema_mismatch")
    if attempt.get("reasons") != []:
        raise ManualPredictionError("capture_attempt_reasons_not_empty")
    return attempt


def _active_capture_rows(
    attempt: Mapping[str, Any], sidecar_runners: Mapping[int, Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], datetime, datetime]:
    validation = attempt.get("validation")
    if not isinstance(validation, Mapping):
        raise ManualPredictionError("capture_validation_missing")
    if validation.get("schema_version") != CAPTURE_VALIDATION_SCHEMA:
        raise ManualPredictionError("capture_validation_schema_mismatch")
    required = {
        "source_url",
        "accepted_rows",
        "accepted_row_count",
        "rejected_rows",
        "expected_runner_count",
        "active_expected_runner_count",
        "scratched_expected_runner_count",
        "scratched_expected_runners",
        "scratched_expected_runners_with_odds",
        "missing_expected_runners",
        "extra_unexpected_runners",
        "failure_root_cause",
        "reasons",
    }
    missing = required - set(validation)
    if missing:
        raise ManualPredictionError(f"capture_validation_fields_missing:{sorted(missing)}")
    race_id = str(attempt.get("race_id") or "")
    if not _trusted_sportsbet_url(validation.get("source_url"), race_id):
        raise ManualPredictionError("capture_source_url_not_trusted_sportsbet")
    if validation.get("reasons") != []:
        raise ManualPredictionError("capture_validation_reasons_not_empty")
    if validation.get("failure_root_cause") not in (None, ""):
        raise ManualPredictionError("capture_validation_failure_root_cause_present")
    for key in (
        "rejected_rows",
        "scratched_expected_runners_with_odds",
        "missing_expected_runners",
        "extra_unexpected_runners",
    ):
        if validation.get(key) != []:
            raise ManualPredictionError(f"capture_{key}_not_empty")
    scratched_rows = validation.get("scratched_expected_runners")
    if not isinstance(scratched_rows, list):
        raise ManualPredictionError("capture_scratched_runners_invalid")
    scratched: set[tuple[int, str]] = set()
    for row in scratched_rows:
        if not isinstance(row, Mapping):
            raise ManualPredictionError("capture_scratched_runner_invalid")
        key = (
            _box(row.get("box_number"), "capture_scratched_box"),
            _runner_token(row.get("dog_name") or row.get("identity")),
        )
        if (
            not key[1]
            or key in scratched
            or key[0] not in sidecar_runners
            or sidecar_runners[key[0]]["identity"] != key[1]
        ):
            raise ManualPredictionError("capture_scratched_runner_mismatch_or_duplicate")
        scratched.add(key)
    expected = {
        (box, str(row["identity"])): row
        for box, row in sidecar_runners.items()
        if (box, str(row["identity"])) not in scratched
    }
    if _integer(
        validation.get("expected_runner_count"),
        "capture_expected_runner_count",
        minimum=2,
    ) != len(sidecar_runners):
        raise ManualPredictionError("capture_expected_runner_count_mismatch")
    if _integer(
        validation.get("scratched_expected_runner_count"),
        "capture_scratched_runner_count",
    ) != len(scratched):
        raise ManualPredictionError("capture_scratched_runner_count_mismatch")
    if _integer(
        validation.get("active_expected_runner_count"),
        "capture_active_runner_count",
        minimum=2,
    ) != len(expected):
        raise ManualPredictionError("capture_active_runner_count_mismatch")
    rows = validation.get("accepted_rows")
    if not isinstance(rows, list):
        raise ManualPredictionError("capture_accepted_rows_invalid")
    if _integer(
        validation.get("accepted_row_count"),
        "capture_accepted_row_count",
        minimum=2,
    ) != len(rows) or len(rows) != len(expected):
        raise ManualPredictionError("capture_runner_count_mismatch")
    output = []
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ManualPredictionError("capture_runner_invalid")
        box = _box(row.get("box_number"), "capture_runner_box")
        name = str(row.get("dog_name") or "").strip()
        identity = _runner_token(name)
        declared_identity = _runner_token(row.get("identity"))
        key = (box, identity)
        odds = _nullable_float(row.get("odds_decimal"), "capture_runner_odds")
        if (
            not identity
            or (declared_identity and declared_identity != identity)
            or key in seen
            or key not in expected
        ):
            raise ManualPredictionError("capture_runner_set_mismatch_or_duplicate")
        if row.get("sportsbet_box_source") not in ALLOWED_BOX_SOURCES:
            raise ManualPredictionError("capture_runner_box_source_invalid")
        if odds is None or odds <= 1.0:
            raise ManualPredictionError("capture_runner_odds_invalid")
        seen.add(key)
        output.append(
            {"box_number": box, "dog_name": name, "identity": identity, "odds": odds}
        )
    if seen != set(expected):
        raise ManualPredictionError("capture_runner_set_mismatch")
    fetch_time = _parse_timestamp(attempt.get("fetch_time"), "capture_fetch_timestamp")
    append_time = _parse_timestamp(attempt.get("append_time"), "capture_append_timestamp")
    if fetch_time > append_time:
        raise ManualPredictionError("capture_append_before_fetch")
    return output, fetch_time, append_time


def _manifest_entry(
    implementation: Mapping[str, Any], path: Path, label: str
) -> Mapping[str, Any]:
    artifact_files = implementation.get("artifact_files")
    if not isinstance(artifact_files, Mapping):
        raise ManualPredictionError("implementation_manifest_artifact_files_missing")
    matches = []
    resolved = path.resolve()
    for raw_path, entry in artifact_files.items():
        if Path(str(raw_path)).resolve() == resolved:
            matches.append(entry)
    if len(matches) != 1 or not isinstance(matches[0], Mapping):
        raise ManualPredictionError(f"implementation_manifest_{label}_entry_mismatch")
    return matches[0]


def _feature_packet(
    *,
    feature_rows_path: Path,
    feature_rows_raw: bytes,
    feature_rows_sha: str,
    feature_manifest_path: Path,
    feature_manifest_raw: bytes,
    feature_manifest_sha: str,
    implementation_manifest_path: Path,
    implementation_manifest: Mapping[str, Any],
    form_csv_path: Path,
    context: Mapping[str, Any],
    race_id: str,
) -> tuple[dict[int, dict[str, Any]], datetime, datetime]:
    parent = feature_rows_path.parent.resolve()
    if feature_rows_path.name != "shadow_feature_rows.json":
        raise ManualPredictionError("feature_rows_filename_invalid")
    if feature_manifest_path.resolve() != (parent / "shadow_manifest.json"):
        raise ManualPredictionError("feature_manifest_not_adjacent")
    if implementation_manifest_path.resolve() != (
        parent / "implementation_file_manifest.json"
    ):
        raise ManualPredictionError("implementation_manifest_not_adjacent")
    if implementation_manifest.get("schema_version") != IMPLEMENTATION_MANIFEST_SCHEMA:
        raise ManualPredictionError("implementation_manifest_schema_mismatch")
    if _contains_outcome_key(implementation_manifest):
        raise ManualPredictionError("implementation_manifest_contains_outcome_field")
    if implementation_manifest.get("git_branch") != FEATURE_GENERATOR_BRANCH:
        raise ManualPredictionError("feature_generator_branch_mismatch")
    if implementation_manifest.get("git_head") != FEATURE_GENERATOR_HEAD:
        raise ManualPredictionError("feature_generator_head_mismatch")
    if Path(str(implementation_manifest.get("output_dir") or "")).resolve() != parent:
        raise ManualPredictionError("feature_generator_output_dir_mismatch")
    implementation_files = implementation_manifest.get("implementation_files")
    if implementation_files != FEATURE_GENERATOR_FILES:
        raise ManualPredictionError("feature_generator_identity_missing")
    for path, raw, sha, label in (
        (feature_rows_path, feature_rows_raw, feature_rows_sha, "feature_rows"),
        (
            feature_manifest_path,
            feature_manifest_raw,
            feature_manifest_sha,
            "feature_manifest",
        ),
    ):
        entry = _manifest_entry(implementation_manifest, path, label)
        if entry.get("sha256") != sha or _integer(
            entry.get("bytes"), f"{label}_declared_bytes", minimum=1
        ) != len(raw):
            raise ManualPredictionError(f"{label}_manifest_hash_mismatch")
    feature_manifest = _json_object(feature_manifest_raw, "feature_manifest")
    if _contains_outcome_key(feature_manifest):
        raise ManualPredictionError("feature_manifest_contains_outcome_field")
    if feature_manifest.get("schema_version") != FEATURE_MANIFEST_SCHEMA:
        raise ManualPredictionError("feature_manifest_schema_mismatch")
    exact_flags = {
        "output_mode": "shadow_only",
        "betting_output": False,
        "ev_output": False,
        "odds_used_for_shadow_scoring": False,
        "production_prediction_write": False,
        "registry_mutation": False,
        "tgr_enabled": False,
    }
    for key, expected in exact_flags.items():
        if feature_manifest.get(key) != expected:
            raise ManualPredictionError(f"feature_manifest_{key}_mismatch")
    if Path(str(feature_manifest.get("feature_rows") or "")).resolve() != (
        feature_rows_path.resolve()
    ):
        raise ManualPredictionError("feature_manifest_rows_path_mismatch")
    input_files = feature_manifest.get("input_files")
    if not isinstance(input_files, list) or sum(
        Path(str(item)).resolve() == form_csv_path.resolve() for item in input_files
    ) != 1:
        raise ManualPredictionError("feature_manifest_source_csv_mismatch")
    feature_time = _parse_timestamp(
        feature_manifest.get("feature_freeze_timestamp"), "feature_freeze_timestamp"
    )
    generated_time = _parse_timestamp(
        feature_manifest.get("generated_at"), "feature_manifest_generated_at"
    )
    if feature_time > generated_time:
        raise ManualPredictionError("feature_manifest_generated_before_freeze")
    rows = _json_value(feature_rows_raw, "feature_rows")
    if not isinstance(rows, list) or _contains_outcome_key(rows):
        raise ManualPredictionError("feature_rows_invalid_or_contains_outcome")
    selected = [row for row in rows if isinstance(row, Mapping) and row.get("race_id") == race_id]
    if len(selected) < 2:
        raise ManualPredictionError("feature_rows_for_race_missing")
    by_box: dict[int, dict[str, Any]] = {}
    for row in selected:
        if "features" in row:
            raise ManualPredictionError("feature_values_must_be_top_level_and_exact")
        box = _box(row.get("box_number"), "feature_runner_box")
        name = str(row.get("dog_name") or "").strip()
        identity = _runner_token(name)
        if (
            box in by_box
            or box not in context["runners"]
            or context["runners"][box]["identity"] != identity
        ):
            raise ManualPredictionError("feature_runner_set_mismatch_or_duplicate")
        if Path(str(row.get("source_csv") or "")).resolve() != form_csv_path.resolve():
            raise ManualPredictionError("feature_row_source_csv_mismatch")
        if row.get("metadata_is_leakage_safe") is not True:
            raise ManualPredictionError("feature_row_metadata_not_safe")
        if row.get("target_metadata_from_sidecar") is not True:
            raise ManualPredictionError("feature_row_target_metadata_not_from_sidecar")
        if row.get("target_metadata_rejected_sources") != []:
            raise ManualPredictionError("feature_row_target_metadata_rejected_sources")
        if row.get("target_distance_source_is_safe") != 1:
            raise ManualPredictionError("feature_row_target_distance_not_safe")
        if row.get("target_grade_provenance_safe") != 1:
            raise ManualPredictionError("feature_row_target_grade_not_safe")
        if row.get("target_distance_missing") != 0 or row.get("target_grade_missing") != 0:
            raise ManualPredictionError("feature_row_target_metadata_missing")
        if not _trusted_thedogs_url(row.get("target_metadata_source_url")):
            raise ManualPredictionError("feature_row_target_url_not_trusted_thedogs")
        if str(row.get("target_metadata_source_url")) != context["source_url"]:
            raise ManualPredictionError("feature_row_target_url_sidecar_mismatch")
        if _parse_date(row.get("race_date"), "feature_row_race_date") != context[
            "target_race_date"
        ]:
            raise ManualPredictionError("feature_row_race_date_mismatch")
        if str(row.get("venue") or "").strip().upper() != context["target_venue"]:
            raise ManualPredictionError("feature_row_venue_mismatch")
        if _integer(row.get("race_number"), "feature_row_race_number", minimum=1) != context[
            "target_race_number"
        ]:
            raise ManualPredictionError("feature_row_race_number_mismatch")
        distance = _nullable_float(row.get("target_distance_safe"), "feature_target_distance")
        if distance is None or not math.isclose(
            distance, float(context["target_distance"]), rel_tol=0.0, abs_tol=1e-9
        ):
            raise ManualPredictionError("feature_row_target_distance_mismatch")
        if _runner_token(row.get("target_grade_safe")) != _runner_token(
            context["target_grade"]
        ):
            raise ManualPredictionError("feature_row_target_grade_mismatch")
        if row.get("same_distance_same_grade_history_cutoff") != (
            "strictly_before_target_race"
        ) or row.get("same_distance_same_grade_history_cutoff_basis") != (
            "race_date_less_than_target_race_date"
        ):
            raise ManualPredictionError("feature_history_cutoff_mismatch")
        if row.get("same_distance_same_grade_target_race_rows_used") != 0:
            raise ManualPredictionError("feature_target_race_rows_used")
        if row.get("same_distance_same_grade_post_outcome_rows_used") != 0:
            raise ManualPredictionError("feature_post_outcome_rows_used")
        if row.get("same_distance_same_grade_post_outcome_fields_used") != []:
            raise ManualPredictionError("feature_post_outcome_fields_used")
        features: dict[str, float | None] = {}
        for feature in FEATURES:
            if feature not in row:
                raise ManualPredictionError(f"feature_value_missing:{feature}")
            features[feature] = _nullable_float(row[feature], f"feature_value:{feature}")
        by_box[box] = {
            "box_number": box,
            "dog_name": name,
            "identity": identity,
            "features": features,
        }
    if set(by_box) != set(context["runners"]):
        raise ManualPredictionError("feature_race_incomplete_or_runner_set_mismatch")
    return by_box, feature_time, generated_time


def score_from_artifacts(
    *,
    race_id: str,
    form_csv_path: Path,
    sidecar_path: Path,
    feature_rows_path: Path,
    feature_manifest_path: Path,
    implementation_manifest_path: Path,
    capture_path: Path,
    model_path: Path,
    manifest_path: Path,
    score_timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Validate immutable sealed inputs and return one outcome-free ranking."""

    if not race_id or any(token in race_id for token in ("/", "\\", "..")):
        raise ManualPredictionError("race_id_invalid")
    expected_sidecar = form_csv_path.with_name(form_csv_path.name + ".metadata.json")
    if sidecar_path.resolve() != expected_sidecar.resolve():
        raise ManualPredictionError("sidecar_not_adjacent_to_form_csv")

    # Every mutable source artifact is read exactly once. Hashes below describe
    # the same immutable bytes that are parsed and scored.
    form_raw, form_sha = _read_input(form_csv_path, "form_csv")
    sidecar_raw, sidecar_sha = _read_input(sidecar_path, "sidecar")
    feature_rows_raw, feature_rows_sha = _read_input(feature_rows_path, "feature_rows")
    feature_manifest_raw, feature_manifest_sha = _read_input(
        feature_manifest_path, "feature_manifest"
    )
    implementation_raw, implementation_sha = _read_input(
        implementation_manifest_path, "implementation_manifest"
    )
    capture_raw, capture_sha = _read_input(capture_path, "capture")

    sidecar = _json_object(sidecar_raw, "sidecar")
    _validate_form_binding(
        sidecar,
        form_csv_path=form_csv_path,
        form_raw=form_raw,
        form_sha=form_sha,
    )
    context = _sidecar_context(sidecar)
    if race_id != context["expected_race_id"]:
        raise ManualPredictionError("race_id_sidecar_mismatch")
    implementation = _json_object(implementation_raw, "implementation_manifest")
    feature_by_box, feature_time, feature_generated_time = _feature_packet(
        feature_rows_path=feature_rows_path,
        feature_rows_raw=feature_rows_raw,
        feature_rows_sha=feature_rows_sha,
        feature_manifest_path=feature_manifest_path,
        feature_manifest_raw=feature_manifest_raw,
        feature_manifest_sha=feature_manifest_sha,
        implementation_manifest_path=implementation_manifest_path,
        implementation_manifest=implementation,
        form_csv_path=form_csv_path,
        context=context,
        race_id=race_id,
    )
    attempt = _select_capture_attempt(
        capture_raw, jsonl=capture_path.suffix.lower() == ".jsonl", race_id=race_id
    )
    capture_rows, fetch_time, append_time = _active_capture_rows(
        attempt, context["runners"]
    )
    jump = context["jump_timestamp"]
    score_time = score_timestamp or datetime.now().astimezone()
    if score_time.tzinfo is None or score_time.utcoffset() is None:
        raise ManualPredictionError("score_timestamp_timezone_missing")
    timeline = (
        context["metadata_timestamp"],
        feature_time,
        feature_generated_time,
        fetch_time,
        append_time,
        score_time,
        jump,
    )
    if any(left > right for left, right in zip(timeline, timeline[1:])):
        raise ManualPredictionError("source_timestamp_order_invalid")
    if not score_time < jump:
        raise ManualPredictionError("manual_score_not_prejump")

    selected_attempt_sha = _sha256_bytes(_canonical_bytes(attempt))
    feature_source_sha = _sha256_bytes(
        _canonical_bytes(
            {
                "feature_manifest_sha256": feature_manifest_sha,
                "feature_rows_sha256": feature_rows_sha,
                "form_csv_sha256": form_sha,
                "implementation_manifest_sha256": implementation_sha,
                "sidecar_sha256": sidecar_sha,
            }
        )
    )
    odds_source_sha = _sha256_bytes(
        _canonical_bytes(
            {
                "capture_artifact_sha256": capture_sha,
                "selected_attempt_sha256": selected_attempt_sha,
            }
        )
    )
    active_by_box = {
        int(row["box_number"]): row for row in capture_rows
    }
    runners = []
    runner_ids = []
    for box in sorted(active_by_box):
        capture_row = active_by_box[box]
        feature_row = feature_by_box[box]
        runner_id = f"{race_id}|box:{box}|dog:{feature_row['identity']}"
        runner_ids.append(runner_id)
        runners.append(
            {
                "race_id": race_id,
                "runner_id": runner_id,
                "box_number": box,
                "dog_name": feature_row["dog_name"],
                "strict_win_odds": float(capture_row["odds"]),
                "features": feature_row["features"],
                "feature_source_sha256": feature_source_sha,
                "odds_source_sha256": odds_source_sha,
                "feature_freeze_timestamp": feature_time.isoformat(),
                "odds_capture_timestamp": fetch_time.isoformat(),
            }
        )
    frozen = load_frozen_model(model_path, manifest_path)
    expected_ids = sorted(runner_ids)
    provenance = {
        "race_id": race_id,
        "expected_runner_ids": expected_ids,
        "runner_set_sha256": _runner_set_sha256(expected_ids),
        "jump_timestamp": jump.isoformat(),
        "score_timestamp": score_time.isoformat(),
    }
    record = score_race(frozen, runners, provenance)
    ranking = sorted(
        record["predictions"],
        key=lambda row: (-float(row["full_probability"]), int(row["box_number"])),
    )
    output = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "MANUAL_PREJUMP_FROZEN_RESIDUAL_PREDICTION",
        "race_id": race_id,
        "jump_timestamp": jump.isoformat(),
        "score_timestamp": score_time.isoformat(),
        "metadata_capture_timestamp": context["metadata_timestamp"].isoformat(),
        "feature_freeze_timestamp": feature_time.isoformat(),
        "feature_manifest_generated_at": feature_generated_time.isoformat(),
        "odds_capture_timestamp": fetch_time.isoformat(),
        "odds_append_timestamp": append_time.isoformat(),
        "model_sha256": record["model_sha256"],
        "manifest_sha256": record["manifest_sha256"],
        "runner_set_sha256": record["runner_set_sha256"],
        "record_key": record["record_key"],
        "variants": {"full_strength": 1.0, "half_strength": 0.5},
        "source_contract": {
            "feature_source": "exact_hash_bound_system_shadow_feature_rows",
            "feature_reconstruction_performed": False,
            "database_access": False,
            "network_access": False,
        },
        "input_hashes": {
            "form_csv_sha256": form_sha,
            "sidecar_sha256": sidecar_sha,
            "feature_rows_sha256": feature_rows_sha,
            "feature_manifest_sha256": feature_manifest_sha,
            "implementation_manifest_sha256": implementation_sha,
            "capture_artifact_sha256": capture_sha,
            "selected_attempt_sha256": selected_attempt_sha,
            "feature_source_sha256": feature_source_sha,
            "odds_source_sha256": odds_source_sha,
        },
        "predictions": [
            {
                "rank": rank,
                "box": int(row["box_number"]),
                "dog": row["dog_name"],
                "win_odds": float(row["strict_win_odds"]),
                "market_probability": float(row["market_probability"]),
                "half_probability": float(row["half_probability"]),
                "full_probability": float(row["full_probability"]),
                "full_minus_market": float(row["full_probability"])
                - float(row["market_probability"]),
            }
            for rank, row in enumerate(ranking, start=1)
        ],
        "probability_sums": {
            key: sum(float(row[f"{key}_probability"]) for row in ranking)
            for key in ("market", "half", "full")
        },
        "activation": False,
        "persisted": False,
        "outcomes_present": False,
    }
    _canonical_bytes(output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--race-id", required=True)
    parser.add_argument("--form-csv", required=True, type=Path)
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--feature-rows", required=True, type=Path)
    parser.add_argument("--feature-manifest", type=Path)
    parser.add_argument("--implementation-manifest", type=Path)
    parser.add_argument("--capture", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact_dir = ROOT / DEFAULT_ARTIFACT_DIR
    sidecar = args.sidecar or args.form_csv.with_name(args.form_csv.name + ".metadata.json")
    feature_manifest = args.feature_manifest or args.feature_rows.with_name(
        "shadow_manifest.json"
    )
    implementation_manifest = args.implementation_manifest or args.feature_rows.with_name(
        "implementation_file_manifest.json"
    )
    try:
        output = score_from_artifacts(
            race_id=args.race_id,
            form_csv_path=args.form_csv,
            sidecar_path=sidecar,
            feature_rows_path=args.feature_rows,
            feature_manifest_path=feature_manifest,
            implementation_manifest_path=implementation_manifest,
            capture_path=args.capture,
            model_path=artifact_dir / "model.json",
            manifest_path=artifact_dir / "manifest.json",
        )
    except (ManualPredictionError, ResidualContractError) as exc:
        sys.stderr.buffer.write(
            _canonical_bytes(
                {"status": "BLOCKED_MANUAL_PREDICTION", "reason": str(exc)}
            )
        )
        return 2
    sys.stdout.buffer.write(_canonical_bytes(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
