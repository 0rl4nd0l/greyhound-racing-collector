#!/usr/bin/env python3
"""Score one race from exact sealed system features and strict pre-jump odds.

The command reads already-materialized artifacts and prints canonical JSON to
stdout. An explicit ``--append-shadow-output`` path may also persist the same
outcome-free frozen record to one append-only JSONL. It has no database,
network, feature-generation, service, activation, deployment, promotion, EV,
or betting path.
"""

from __future__ import annotations

import argparse
import difflib
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
    SHADOW_RECORD_SCHEMA,
    FrozenResidualModel,
    ResidualContractError,
    _runner_set_sha256,
    append_shadow_record,
    load_frozen_model,
    score_race,
)


MELBOURNE = ZoneInfo("Australia/Melbourne")
ALLOWED_BOX_SOURCES = {"explicit_dom", "runner_text"}
OUTCOME_KEYS = {
    "actual_win",
    "actual_winner",
    "db_finish_position",
    "db_result_position",
    "db_scraped_finish_position",
    "dividend",
    "finish",
    "finish_position",
    "finishing_position",
    "future_position",
    "future_result",
    "future_time",
    "individual_time",
    "is_placer",
    "is_winner",
    "margin",
    "mgn",
    "official_finish_position",
    "official_position",
    "official_result",
    "official_result_status",
    "official_winner",
    "outcome",
    "payout",
    "place",
    "placing",
    "plc",
    "position",
    "race_time_result",
    "result",
    "result_position",
    "result_status",
    "results",
    "results_status",
    "scraped_finish_position",
    "scraped_raw_result",
    "starting_price",
    "target_finish_position",
    "time",
    "win_time",
    "winner",
    "winner_margin",
    "winner_name",
    "winner_odds",
    "winning_time",
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
LEGACY_FEATURE_ROWS_SHA256 = (
    "0ebecaf980665545aa8c19d1a4b1ef976bd069049d42f7f6ebde0f3b29a36b62"
)
LEGACY_IMPLEMENTATION_MANIFEST_SHA256 = (
    "9822a77a4d69a72c8b7b2e7d234538b6207b99530b3b717fb9cb31f64929a651"
)
LEGACY_FEATURE_GENERATOR_FILES = [
    "scripts/run_shadow_non_tgr_rf_evaluation.py",
    "tests/test_run_shadow_non_tgr_rf_evaluation.py",
]
FEATURE_GENERATOR_FILES = [
    "scripts/run_shadow_non_tgr_rf_evaluation.py",
    "scripts/run_feature_recovery_execution_v1.py",
    "utils/expert_form_metadata.py",
    "utils/prejump_weather.py",
    "utils/http_client.py",
    "utils/csv_metadata.py",
    "utils/runner_completeness.py",
    "utils/race_lifecycle.py",
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
DEFAULT_EVIDENCE_ROOT = ROOT / "artifacts/full_evidence_orchestration_20260525"
VENUE_URL_ALIASES = {
    "MAND": {"mand", "mandurah"},
    "SAN": {"san", "sandown", "sandown-park"},
    "SHEP": {"shep", "shepparton"},
    "WPK": {"wpk", "wentworth-park"},
}


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


def _existing_replay_score_timestamp(
    path: Path, identity: Mapping[str, str]
) -> datetime | None:
    """Return one prior score time as a hint for writer-validated exact replay."""

    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ManualPredictionError("shadow_output_unreadable") from exc
    if not raw:
        return None
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ManualPredictionError("shadow_output_invalid_utf8") from exc
    matches = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ManualPredictionError(f"shadow_output_blank_line:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ManualPredictionError(
                f"shadow_output_invalid_json:{line_number}"
            ) from exc
        if not isinstance(row, Mapping):
            raise ManualPredictionError(f"shadow_output_row_not_object:{line_number}")
        if row.get("schema_version") == SHADOW_RECORD_SCHEMA and all(
            row.get(key) == value for key, value in identity.items()
        ):
            matches.append(row)
    if len(matches) > 1:
        raise ManualPredictionError("shadow_output_replay_identity_ambiguous")
    if not matches:
        return None
    return _parse_timestamp(
        matches[0].get("score_timestamp"), "shadow_replay_score_timestamp"
    )


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


def _race_id_parts(race_id: str) -> tuple[int, str, date]:
    match = re.fullmatch(
        r"Race\s+([0-9]{1,2})\s+-\s+(.+?)\s+-\s+([0-9]{4}-[0-9]{2}-[0-9]{2})",
        str(race_id or "").strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        raise ManualPredictionError("discovery_race_id_invalid")
    return (
        int(match.group(1)),
        match.group(2).strip().upper(),
        _parse_date(match.group(3), "discovery_race_date"),
    )


def _race_query_parts(query: str) -> tuple[int, str]:
    text = str(query or "").strip()
    match = re.search(r"\b(?:R|RACE)\s*([0-9]{1,2})\b", text, flags=re.IGNORECASE)
    if not match:
        raise ManualPredictionError("race_query_number_missing")
    race_number = int(match.group(1))
    venue = _runner_token(f"{text[: match.start()]} {text[match.end() :]}")
    if not venue:
        raise ManualPredictionError("race_query_venue_missing")
    return race_number, venue


def _venue_query_match_rank(
    query: str,
    *,
    canonical_venue: str,
    full_aliases: Sequence[str],
) -> int | None:
    canonical = _runner_token(canonical_venue)
    aliases = {_runner_token(value) for value in full_aliases if _runner_token(value)}
    if query == canonical:
        return 0
    if query in aliases:
        return 1
    if any(
        (len(query) >= 4 and query in alias) or (len(alias) >= 4 and alias in query)
        for alias in aliases
    ):
        return 2
    if any(
        min(len(query), len(alias)) >= 5
        and difflib.SequenceMatcher(a=query, b=alias).ratio() >= 0.8
        for alias in aliases
    ):
        return 3
    return None


def _path_in_roots(path: Path, roots: Sequence[Path]) -> bool:
    resolved = path.resolve()
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            continue
        return True
    return False


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
    try:
        port = parsed.port
    except ValueError:
        return False
    return bool(
        parsed.scheme == "https"
        and hostname == "www.thedogs.com.au"
        and parsed.username is None
        and parsed.password is None
        and port in (None, 443)
        and re.fullmatch(
            r"/racing/[a-z0-9]+(?:-[a-z0-9]+)*/\d{4}-\d{2}-\d{2}/\d{1,2}/[a-z0-9]+(?:-[a-z0-9]+)*/?",
            parsed.path.lower(),
        )
        and not (tokens & POST_RACE_URL_TOKENS)
    )


def _venue_url_aliases(venue: str) -> set[str]:
    normalized = str(venue or "").strip().lower().replace("_", "-")
    aliases = {normalized}
    aliases.update(VENUE_URL_ALIASES.get(str(venue or "").strip().upper(), set()))
    return aliases


def _trusted_sportsbet_url(value: Any, race_id: str, expected_venue: str) -> bool:
    parsed, tokens = _url_tokens(value)
    if parsed is None:
        return False
    hostname = (parsed.hostname or "").lower()
    try:
        port = parsed.port
    except ValueError:
        return False
    if (
        parsed.scheme != "https"
        or hostname != "www.sportsbet.com.au"
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
    ):
        return False
    if tokens & POST_RACE_URL_TOKENS:
        return False
    path_match = re.fullmatch(
        r"/(?:betting/)?greyhound-racing/australia-nz/([a-z0-9]+(?:-[a-z0-9]+)*)/race-(\d+)(?:-\d+)?/?",
        parsed.path.lower(),
    )
    race_match = re.fullmatch(
        r"Race (\d+) - [A-Z0-9_]+(?:-[A-Z0-9_]+)* - \d{4}-\d{2}-\d{2}", race_id
    )
    return bool(
        path_match
        and race_match
        and path_match.group(1) in _venue_url_aliases(expected_venue)
        and int(path_match.group(2)) == int(race_match.group(1))
    )


def _canonical_target_grade(value: Any) -> str | None:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    grade_match = re.fullmatch(r"Grade\s+([1-7])", text, flags=re.IGNORECASE)
    if grade_match is None:
        grade_match = re.fullmatch(
            r"([1-7])(?:st|nd|rd|th)(?:\s+Grade)?",
            text,
            flags=re.IGNORECASE,
        )
    if grade_match:
        return f"GRADE{grade_match.group(1)}"
    exact = {
        "MAIDEN": "MAIDEN",
        "NOVICE": "NOVICE",
        "OPEN": "OPEN",
        "MIXED": "MIXED",
        "RESTRICTED": "RESTRICTED",
        "FREE FOR ALL": "FFA",
        "FFA": "FFA",
        "NO GRADE": "NO_GRADE",
        "NON GRADED": "NO_GRADE",
    }
    return exact.get(text.upper())


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
    raw_time = str(shadow.get("jump_time") or race_info.get("race_time") or "").strip()
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
    if not isinstance(alignment, Mapping) or str(
        alignment.get("status")
    ).lower() not in {
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
    if _integer(
        completeness.get("runner_count"), "sidecar_runner_count", minimum=2
    ) != len(runners):
        raise ManualPredictionError("sidecar_runner_count_mismatch")
    complete_rows = completeness.get("participants")
    if not isinstance(complete_rows, list):
        raise ManualPredictionError("sidecar_runner_completeness_participants_missing")
    complete_set = set()
    for row in complete_rows:
        if not isinstance(row, Mapping):
            raise ManualPredictionError(
                "sidecar_runner_completeness_participant_invalid"
            )
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
    if not venue or not re.fullmatch(r"[A-Z0-9_]+(?:-[A-Z0-9_]+)*", venue):
        raise ManualPredictionError("target_venue_invalid")
    target_distance = _distance_metres(
        shadow.get("distance") or race_info.get("distance")
    )
    target_grade = str(shadow.get("grade") or race_info.get("grade") or "").strip()
    canonical_grade = _canonical_target_grade(target_grade)
    if canonical_grade is None:
        raise ManualPredictionError("target_grade_not_exact_supported_alias")
    path_parts = [part for part in urlparse(source_url).path.split("/") if part]
    if (
        len(path_parts) != 5
        or path_parts[0].lower() != "racing"
        or path_parts[1].lower() not in _venue_url_aliases(venue)
        or path_parts[2] != target_date.isoformat()
        or not path_parts[3].isdigit()
        or int(path_parts[3]) != race_number
    ):
        raise ManualPredictionError("sidecar_source_url_race_binding_mismatch")
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
        "target_grade": canonical_grade,
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
                raise ManualPredictionError(
                    f"capture_jsonl_invalid:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ManualPredictionError(
                    f"capture_jsonl_row_not_object:{line_number}"
                )
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
    attempt: Mapping[str, Any],
    sidecar_runners: Mapping[int, Mapping[str, Any]],
    context: Mapping[str, Any],
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
        raise ManualPredictionError(
            f"capture_validation_fields_missing:{sorted(missing)}"
        )
    race_id = str(attempt.get("race_id") or "")
    if not _trusted_sportsbet_url(
        validation.get("source_url"), race_id, str(context["target_venue"])
    ):
        raise ManualPredictionError("capture_source_url_not_trusted_sportsbet")
    if (
        validation.get("source_race_number") != context["target_race_number"]
        or validation.get("source_race_date") != context["target_race_date"].isoformat()
        or validation.get("source_venue") != context["target_venue"]
    ):
        raise ManualPredictionError("capture_source_race_binding_mismatch")
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
            raise ManualPredictionError(
                "capture_scratched_runner_mismatch_or_duplicate"
            )
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
    append_time = _parse_timestamp(
        attempt.get("append_time"), "capture_append_timestamp"
    )
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


def _validate_feature_generator_identity(
    implementation_manifest: Mapping[str, Any],
    *,
    implementation_manifest_sha: str,
    feature_rows_sha: str,
) -> None:
    legacy_branch = (
        implementation_manifest.get("git_branch") == FEATURE_GENERATOR_BRANCH
    )
    legacy_head = implementation_manifest.get("git_head") == FEATURE_GENERATOR_HEAD
    legacy_packet = (
        implementation_manifest_sha == LEGACY_IMPLEMENTATION_MANIFEST_SHA256
        and feature_rows_sha == LEGACY_FEATURE_ROWS_SHA256
    )
    if legacy_branch and legacy_head and legacy_packet:
        if implementation_manifest.get("implementation_files") != (
            LEGACY_FEATURE_GENERATOR_FILES
        ):
            raise ManualPredictionError("feature_generator_identity_missing")
        return

    declared_hashes = implementation_manifest.get("implementation_file_hashes")
    if not isinstance(declared_hashes, Mapping):
        if legacy_branch and legacy_head:
            raise ManualPredictionError("feature_generator_legacy_packet_hash_mismatch")
        if not legacy_branch:
            raise ManualPredictionError("feature_generator_branch_mismatch")
        raise ManualPredictionError("feature_generator_head_mismatch")
    implementation_files = implementation_manifest.get("implementation_files")
    if implementation_files != FEATURE_GENERATOR_FILES:
        raise ManualPredictionError("feature_generator_identity_missing")
    if set(declared_hashes) != set(FEATURE_GENERATOR_FILES):
        raise ManualPredictionError(
            "feature_generator_implementation_hash_set_mismatch"
        )
    for relative in FEATURE_GENERATOR_FILES:
        declared = str(declared_hashes.get(relative) or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", declared):
            raise ManualPredictionError("feature_generator_implementation_hash_invalid")
        path = (ROOT / relative).resolve()
        try:
            path.relative_to(ROOT.resolve())
            actual = _sha256_bytes(path.read_bytes())
        except (OSError, ValueError) as exc:
            raise ManualPredictionError(
                "feature_generator_implementation_file_unreadable"
            ) from exc
        if actual != declared:
            raise ManualPredictionError(
                "feature_generator_implementation_hash_mismatch"
            )


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
    implementation_manifest_sha: str,
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
    if Path(str(implementation_manifest.get("output_dir") or "")).resolve() != parent:
        raise ManualPredictionError("feature_generator_output_dir_mismatch")
    _validate_feature_generator_identity(
        implementation_manifest,
        implementation_manifest_sha=implementation_manifest_sha,
        feature_rows_sha=feature_rows_sha,
    )
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
    if (
        not isinstance(input_files, list)
        or sum(
            Path(str(item)).resolve() == form_csv_path.resolve() for item in input_files
        )
        != 1
    ):
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
    selected = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("race_id") == race_id
    ]
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
        if (
            row.get("target_distance_missing") != 0
            or row.get("target_grade_missing") != 0
        ):
            raise ManualPredictionError("feature_row_target_metadata_missing")
        if not _trusted_thedogs_url(row.get("target_metadata_source_url")):
            raise ManualPredictionError("feature_row_target_url_not_trusted_thedogs")
        if str(row.get("target_metadata_source_url")) != context["source_url"]:
            raise ManualPredictionError("feature_row_target_url_sidecar_mismatch")
        if (
            _parse_date(row.get("race_date"), "feature_row_race_date")
            != context["target_race_date"]
        ):
            raise ManualPredictionError("feature_row_race_date_mismatch")
        if str(row.get("venue") or "").strip().upper() != context["target_venue"]:
            raise ManualPredictionError("feature_row_venue_mismatch")
        if (
            _integer(row.get("race_number"), "feature_row_race_number", minimum=1)
            != context["target_race_number"]
        ):
            raise ManualPredictionError("feature_row_race_number_mismatch")
        distance = _nullable_float(
            row.get("target_distance_safe"), "feature_target_distance"
        )
        if distance is None or not math.isclose(
            distance, float(context["target_distance"]), rel_tol=0.0, abs_tol=1e-9
        ):
            raise ManualPredictionError("feature_row_target_distance_mismatch")
        if (
            _canonical_target_grade(row.get("target_grade_safe"))
            != context["target_grade"]
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
            features[feature] = _nullable_float(
                row[feature], f"feature_value:{feature}"
            )
        by_box[box] = {
            "box_number": box,
            "dog_name": name,
            "identity": identity,
            "features": features,
        }
    if set(by_box) != set(context["runners"]):
        raise ManualPredictionError("feature_race_incomplete_or_runner_set_mismatch")
    return by_box, feature_time, generated_time


def discover_race_artifacts(
    *,
    race_query: str,
    evidence_roots: Sequence[Path],
    score_timestamp: datetime,
) -> dict[str, Any]:
    """Resolve one race to one sealed feature packet and one capture report."""

    if score_timestamp.tzinfo is None or score_timestamp.utcoffset() is None:
        raise ManualPredictionError("score_timestamp_timezone_missing")
    roots = sorted({Path(root).resolve() for root in evidence_roots})
    if not roots or any(not root.is_dir() for root in roots):
        raise ManualPredictionError("evidence_root_missing_or_not_directory")
    race_number, venue_query = _race_query_parts(race_query)

    feature_candidates: list[dict[str, Any]] = []
    seen_feature_packets: set[Path] = set()
    for root in roots:
        for feature_rows_path in sorted(root.rglob("shadow_feature_rows.json")):
            feature_rows_path = feature_rows_path.resolve()
            if feature_rows_path in seen_feature_packets:
                continue
            seen_feature_packets.add(feature_rows_path)
            feature_manifest_path = feature_rows_path.with_name("shadow_manifest.json")
            implementation_manifest_path = feature_rows_path.with_name(
                "implementation_file_manifest.json"
            )
            if not all(
                _path_in_roots(path, roots)
                for path in (
                    feature_rows_path,
                    feature_manifest_path,
                    implementation_manifest_path,
                )
            ):
                raise ManualPredictionError("discovery_path_outside_evidence_root")
            if (
                not feature_manifest_path.is_file()
                or not implementation_manifest_path.is_file()
            ):
                continue
            feature_rows_raw, _ = _read_input(
                feature_rows_path, "discovery_feature_rows"
            )
            rows = _json_value(feature_rows_raw, "discovery_feature_rows")
            if not isinstance(rows, list):
                continue
            if _contains_outcome_key(rows):
                raise ManualPredictionError("discovery_feature_packet_contains_outcome")
            race_ids = sorted(
                {
                    str(row.get("race_id"))
                    for row in rows
                    if isinstance(row, Mapping) and row.get("race_id") not in (None, "")
                }
            )
            for race_id in race_ids:
                try:
                    candidate_number, candidate_venue, candidate_date = _race_id_parts(
                        race_id
                    )
                except ManualPredictionError:
                    continue
                if candidate_number != race_number:
                    continue
                selected_rows = [
                    row
                    for row in rows
                    if isinstance(row, Mapping) and row.get("race_id") == race_id
                ]
                aliases: list[str] = []
                for row in selected_rows:
                    row_venue = str(row.get("venue") or "")
                    if _runner_token(row_venue) != _runner_token(candidate_venue):
                        aliases.append(row_venue)
                    source_url = str(row.get("target_metadata_source_url") or "")
                    try:
                        path_parts = [
                            part
                            for part in urlparse(source_url).path.split("/")
                            if part
                        ]
                    except ValueError:
                        path_parts = []
                    for index, part in enumerate(path_parts[:-1]):
                        if part.lower() == "racing":
                            aliases.append(path_parts[index + 1])
                            break
                venue_match_rank = _venue_query_match_rank(
                    venue_query,
                    canonical_venue=candidate_venue,
                    full_aliases=aliases,
                )
                if venue_match_rank is None:
                    continue
                source_csv_values = {
                    str(row.get("source_csv") or "").strip() for row in selected_rows
                }
                source_csv_values.discard("")
                if len(source_csv_values) != 1:
                    continue
                form_csv_path = Path(next(iter(source_csv_values)))
                if not form_csv_path.is_absolute():
                    form_csv_path = ROOT / form_csv_path
                form_csv_path = form_csv_path.resolve()
                sidecar_path = form_csv_path.with_name(
                    form_csv_path.name + ".metadata.json"
                )
                if (
                    not form_csv_path.is_file()
                    or not sidecar_path.is_file()
                    or not _path_in_roots(form_csv_path, roots)
                    or not _path_in_roots(sidecar_path, roots)
                ):
                    continue
                feature_manifest_raw, _ = _read_input(
                    feature_manifest_path, "discovery_feature_manifest"
                )
                feature_manifest = _json_object(
                    feature_manifest_raw, "discovery_feature_manifest"
                )
                if _contains_outcome_key(feature_manifest):
                    raise ManualPredictionError(
                        "discovery_feature_manifest_contains_outcome"
                    )
                generated_at = _parse_timestamp(
                    feature_manifest.get("generated_at"),
                    "discovery_feature_manifest_generated_at",
                )
                if generated_at > score_timestamp:
                    continue
                feature_candidates.append(
                    {
                        "race_id": race_id,
                        "race_date": candidate_date,
                        "venue_code": candidate_venue,
                        "venue_match_rank": venue_match_rank,
                        "generated_at": generated_at,
                        "form_csv_path": form_csv_path,
                        "sidecar_path": sidecar_path,
                        "feature_rows_path": feature_rows_path,
                        "feature_manifest_path": feature_manifest_path,
                        "implementation_manifest_path": implementation_manifest_path,
                    }
                )

    if not feature_candidates:
        raise ManualPredictionError("race_feature_packet_not_found")
    best_venue_rank = min(row["venue_match_rank"] for row in feature_candidates)
    feature_candidates = [
        row for row in feature_candidates if row["venue_match_rank"] == best_venue_rank
    ]
    if len({row["venue_code"] for row in feature_candidates}) != 1:
        raise ManualPredictionError("race_query_ambiguous")
    current_date = score_timestamp.astimezone(MELBOURNE).date()
    available_dates = sorted(
        {
            row["race_date"]
            for row in feature_candidates
            if row["race_date"] >= current_date
        }
    )
    target_date = (
        available_dates[0]
        if available_dates
        else max(row["race_date"] for row in feature_candidates)
    )
    dated = [row for row in feature_candidates if row["race_date"] == target_date]
    race_ids = {row["race_id"] for row in dated}
    if len(race_ids) != 1:
        raise ManualPredictionError("race_query_ambiguous")
    latest_feature_time = max(row["generated_at"] for row in dated)
    selected_features = [
        row for row in dated if row["generated_at"] == latest_feature_time
    ]
    if len(selected_features) != 1:
        raise ManualPredictionError("race_feature_packet_ambiguous")
    selected_feature = selected_features[0]
    race_id = str(selected_feature["race_id"])

    capture_candidates: list[dict[str, Any]] = []
    seen_capture_reports: set[Path] = set()
    for root in roots:
        for capture_path in sorted(
            root.rglob("autonomous_live_odds_capture_report.json")
        ):
            capture_path = capture_path.resolve()
            if capture_path in seen_capture_reports:
                continue
            seen_capture_reports.add(capture_path)
            if not _path_in_roots(capture_path, roots):
                raise ManualPredictionError("discovery_path_outside_evidence_root")
            capture_raw, _ = _read_input(capture_path, "discovery_capture")
            try:
                capture_payload = _json_object(capture_raw, "discovery_capture")
            except ManualPredictionError:
                continue
            if _contains_outcome_key(capture_payload):
                raise ManualPredictionError("discovery_capture_contains_outcome")
            raw_attempts = capture_payload.get("attempts")
            if not isinstance(raw_attempts, list) or not any(
                isinstance(attempt, Mapping) and attempt.get("race_id") == race_id
                for attempt in raw_attempts
            ):
                continue
            try:
                attempts = _capture_attempts(capture_raw, jsonl=False)
            except ManualPredictionError as exc:
                raise ManualPredictionError("race_capture_report_invalid") from exc
            matches = [
                attempt
                for attempt in attempts
                if attempt.get("race_id") == race_id
                and attempt.get("status") == "APPENDED"
                and isinstance(attempt.get("validation"), Mapping)
                and attempt["validation"].get("status") == "PASS"
            ]
            if not matches:
                continue
            if len(matches) != 1:
                raise ManualPredictionError("accepted_capture_attempt_ambiguous")
            append_time = _parse_timestamp(
                matches[0].get("append_time"), "discovery_capture_append_timestamp"
            )
            if append_time <= score_timestamp:
                capture_candidates.append(
                    {"capture_path": capture_path, "append_time": append_time}
                )
    if not capture_candidates:
        raise ManualPredictionError("race_capture_report_not_found")
    latest_append = max(row["append_time"] for row in capture_candidates)
    selected_captures = [
        row for row in capture_candidates if row["append_time"] == latest_append
    ]
    if len(selected_captures) != 1:
        raise ManualPredictionError("race_capture_report_ambiguous")

    return {
        **selected_feature,
        "capture_path": selected_captures[0]["capture_path"],
    }


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
    shadow_output_path: Path | None = None,
    frozen_model: FrozenResidualModel | None = None,
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
        implementation_manifest_sha=implementation_sha,
        form_csv_path=form_csv_path,
        context=context,
        race_id=race_id,
    )
    attempt = _select_capture_attempt(
        capture_raw, jsonl=capture_path.suffix.lower() == ".jsonl", race_id=race_id
    )
    capture_rows, fetch_time, append_time = _active_capture_rows(
        attempt, context["runners"], context
    )
    jump = context["jump_timestamp"]
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
    active_by_box = {int(row["box_number"]): row for row in capture_rows}
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
    frozen = frozen_model or load_frozen_model(model_path, manifest_path)
    expected_ids = sorted(runner_ids)
    runner_set_sha = _runner_set_sha256(expected_ids)
    score_time = score_timestamp
    if score_time is None and shadow_output_path is not None:
        score_time = _existing_replay_score_timestamp(
            shadow_output_path,
            {
                "race_id": race_id,
                "runner_set_sha256": runner_set_sha,
                "model_sha256": frozen.model_sha256,
                "manifest_sha256": frozen.manifest_sha256,
                "effective_state_sha256": frozen.effective_state_sha256,
            },
        )
    score_time = score_time or datetime.now().astimezone()
    if score_time.tzinfo is None or score_time.utcoffset() is None:
        raise ManualPredictionError("score_timestamp_timezone_missing")
    feature_timeline = (
        context["metadata_timestamp"],
        feature_time,
        feature_generated_time,
        score_time,
        jump,
    )
    odds_timeline = (
        context["metadata_timestamp"],
        fetch_time,
        append_time,
        score_time,
        jump,
    )
    if any(
        left > right
        for timeline in (feature_timeline, odds_timeline)
        for left, right in zip(timeline, timeline[1:])
    ):
        raise ManualPredictionError("source_timestamp_order_invalid")
    if not score_time < jump:
        raise ManualPredictionError("manual_score_not_prejump")
    provenance = {
        "race_id": race_id,
        "expected_runner_ids": expected_ids,
        "runner_set_sha256": runner_set_sha,
        "jump_timestamp": jump.isoformat(),
        "score_timestamp": score_time.isoformat(),
    }
    record = score_race(frozen, runners, provenance)
    ranking = sorted(
        record["predictions"],
        key=lambda row: (-float(row["full_probability"]), int(row["box_number"])),
    )
    persistence_status = (
        append_shadow_record(
            shadow_output_path,
            record,
            frozen=frozen,
            runners=runners,
            provenance=provenance,
        )
        if shadow_output_path is not None
        else None
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
        "persisted": persistence_status is not None,
        "persistence_status": persistence_status,
        "shadow_output_path": (
            str(shadow_output_path.resolve())
            if shadow_output_path is not None
            else None
        ),
        "outcomes_present": False,
    }
    _canonical_bytes(output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--race-id")
    parser.add_argument(
        "--race",
        help='Race-first query such as "sandown r6" over the evidence root.',
    )
    parser.add_argument(
        "--evidence-root",
        action="append",
        type=Path,
        help=(
            "Outcome-free evidence root to search in race-first mode. May be repeated; "
            "defaults to the repository evidence root."
        ),
    )
    parser.add_argument("--form-csv", type=Path)
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--feature-rows", type=Path)
    parser.add_argument("--feature-manifest", type=Path)
    parser.add_argument("--implementation-manifest", type=Path)
    parser.add_argument("--capture", type=Path)
    parser.add_argument(
        "--append-shadow-output",
        type=Path,
        help=(
            "Explicit append-only .jsonl path for the canonical outcome-free "
            "frozen shadow record. The parent directory must already exist; "
            "an identical prior stable identity is writer-validated and returned "
            "as EXACT_REPLAY."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact_dir = ROOT / DEFAULT_ARTIFACT_DIR
    try:
        score_time = datetime.now().astimezone()
        if args.race:
            if any(
                value is not None
                for value in (
                    args.race_id,
                    args.form_csv,
                    args.sidecar,
                    args.feature_rows,
                    args.feature_manifest,
                    args.implementation_manifest,
                    args.capture,
                )
            ):
                raise ManualPredictionError(
                    "race_first_mode_cannot_mix_explicit_artifact_arguments"
                )
            discovered = discover_race_artifacts(
                race_query=args.race,
                evidence_roots=args.evidence_root or [DEFAULT_EVIDENCE_ROOT],
                score_timestamp=score_time,
            )
            race_id = str(discovered["race_id"])
            form_csv = Path(discovered["form_csv_path"])
            sidecar = Path(discovered["sidecar_path"])
            feature_rows = Path(discovered["feature_rows_path"])
            feature_manifest = Path(discovered["feature_manifest_path"])
            implementation_manifest = Path(discovered["implementation_manifest_path"])
            capture = Path(discovered["capture_path"])
        else:
            if args.evidence_root:
                raise ManualPredictionError("evidence_root_requires_race_first_mode")
            if (
                not args.race_id
                or args.form_csv is None
                or args.feature_rows is None
                or args.capture is None
            ):
                raise ManualPredictionError("explicit_artifact_arguments_incomplete")
            race_id = args.race_id
            form_csv = args.form_csv
            sidecar = args.sidecar or form_csv.with_name(
                form_csv.name + ".metadata.json"
            )
            feature_rows = args.feature_rows
            feature_manifest = args.feature_manifest or feature_rows.with_name(
                "shadow_manifest.json"
            )
            implementation_manifest = (
                args.implementation_manifest
                or feature_rows.with_name("implementation_file_manifest.json")
            )
            capture = args.capture
        output = score_from_artifacts(
            race_id=race_id,
            form_csv_path=form_csv,
            sidecar_path=sidecar,
            feature_rows_path=feature_rows,
            feature_manifest_path=feature_manifest,
            implementation_manifest_path=implementation_manifest,
            capture_path=capture,
            model_path=artifact_dir / "model.json",
            manifest_path=artifact_dir / "manifest.json",
            score_timestamp=(
                None
                if args.append_shadow_output is not None
                else score_time
                if args.race
                else None
            ),
            shadow_output_path=args.append_shadow_output,
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
