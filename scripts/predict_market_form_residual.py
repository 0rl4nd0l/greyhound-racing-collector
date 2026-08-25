#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy==1.26.4"]
# ///
"""Score one race from exact sealed system features and strict pre-jump odds.

The command reads already-materialized artifacts and prints canonical JSON to
stdout. It has no database, network, feature-generation, output-file, service,
activation, deployment, promotion, EV, or betting path.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
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
    EFFECTIVE_STATE_SCHEMA,
    FEATURES,
    NUMERICAL_CANONICALIZATION_CONTRACT,
    ResidualContractError,
    SHADOW_RECORD_SCHEMA,
    _runner_set_sha256,
    load_frozen_model,
    score_race,
)
from src.predictor.scoring_parity import (  # noqa: E402
    SCORING_CONFIG_SHA256,
    build_core_output,
    build_scoring_input,
    parity_binding,
)
from config.venue_mapping import VENUE_MAPPING, normalize_venue  # noqa: E402
from utils.csv_metadata import (  # noqa: E402
    THEDOGS_EXACT_RACE_PAGE_GRADE_SOURCE,
    THEDOGS_MEETING_CARD_GRADE_SOURCE,
    canonical_thedogs_meeting_card_url,
    canonical_thedogs_race_identity,
    canonical_thedogs_venue_identity,
    normalize_exact_target_grade,
    target_grade_equivalence_key,
)


MELBOURNE = ZoneInfo("Australia/Melbourne")
ALLOWED_BOX_SOURCES = {"explicit_dom", "runner_text"}
OUTCOME_KEYS = frozenset(
    {
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
)
INDEX_OUTCOME_KEYS = OUTCOME_KEYS | frozenset(
    {
        "actual_wins",
        "finish_positions",
        "official_results",
        "outcomes",
        "placings",
        "results",
        "winners",
        "winner_names",
        "winner_odds_values",
    }
)
INDEX_OUTCOME_TOKENS = frozenset(
    {
        "finish",
        "finishes",
        "outcome",
        "outcomes",
        "placing",
        "placings",
        "result",
        "results",
        "winner",
        "winners",
    }
)
INDEX_FALSE_OUTCOME_MARKERS = frozenset({"outcomes_present", "outcomes_read"})
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
    # Scheduled collector receipts wrap one hash-bound canonical capture attempt.
    "collector_exact_capture_source_v1",
}
CAPTURE_ATTEMPT_SCHEMA = "autonomous_live_odds_capture_attempt_v1"
CAPTURE_VALIDATION_SCHEMA = "autonomous_live_odds_capture_validation_v1"
OUTPUT_SCHEMA = "manual_market_form_residual_prediction_v3"
INDEX_PREDICTION_SCHEMA = "manual_market_form_residual_prediction_v2"
DEFAULT_EVIDENCE_ROOT = ROOT / "artifacts/full_evidence_orchestration_20260525"
DEFAULT_RETAINED_EVIDENCE_ROOTS = (
    ROOT.parent
    / "greyhound-autonomous-accuracy-odds-v1-20260610"
    / "artifacts/full_evidence_orchestration_20260525",
)
DEFAULT_INDEX_MAX_AGE = timedelta(hours=36)
EARLY_RESIDUAL_STATUS_GLOB = (
    "shadow_autopilot_daemonization_v1_*/early_residual_shadow_status.json"
)
REFRESH_REPORT_FILENAME = "odds_capture_refresh_report.json"
MAX_REFRESH_REPORT_RUN_LAG = timedelta(minutes=5)
DIAGNOSTIC_VENUE_CODE_OVERRIDES = MappingProxyType(
    {
        "ALBION": "ALBION",
        "ALBIONPARK": "ALBION",
        "ANGLEPARK": "APK",
        "AP": "ALBION",
        "APK": "APK",
        "GOSF": "GOSF",
        "GOSFORD": "GOSF",
        "MOUNT": "MOUNT",
        "MOUNTGAMBIER": "MOUNT",
        "MTG": "MOUNT",
        "TARE": "TAREE",
        "TAREE": "TAREE",
    }
)
REFRESH_QUARANTINE_REASONS = MappingProxyType(
    {
        "target_metadata_not_verified:missing_target_grade": "missing_target_grade",
    }
)
EARLY_RESIDUAL_STATUS_SCHEMA = "early_residual_shadow_prediction_status_v1"
EARLY_RESIDUAL_PLAN_SCHEMA = "early_residual_shadow_prediction_plan_v1"
EARLY_RESIDUAL_STATUS_KEYS = frozenset(
    {
        "activation",
        "appended_count",
        "blocked_count",
        "exact_replay_count",
        "lock_release_preceded_stage_completion",
        "outcomes_read",
        "plan",
        "race_count",
        "races",
        "schema_version",
        "status",
    }
)
EARLY_RESIDUAL_PLAN_KEYS = frozenset(
    {
        "activation",
        "autopilot_output_dir",
        "blockers",
        "outcomes_read",
        "production_db_access",
        "race_count",
        "races",
        "run_id",
        "schema_version",
        "shadow_output_path",
        "status",
    }
)
EARLY_RESIDUAL_PLAN_RACE_KEYS = frozenset(
    {
        "capture_path",
        "feature_command",
        "feature_model_path",
        "feature_output_dir",
        "form_csv_path",
        "race_id",
        "score_command",
        "sidecar_path",
    }
)
EARLY_RESIDUAL_STATUS_RACE_KEYS = frozenset(
    {"blocker", "feature_step", "prediction", "race_id", "score_step", "status"}
)
EARLY_RESIDUAL_STEP_KEYS = frozenset(
    {
        "command",
        "cwd",
        "duration_seconds",
        "finished_at",
        "name",
        "returncode",
        "started_at",
        "status",
        "stderr_path",
        "stdout_path",
        "timed_out",
        "timeout_deadline_at",
        "timeout_seconds",
    }
)
EARLY_RESIDUAL_PREDICTION_KEYS = frozenset(
    {
        "activation",
        "feature_freeze_timestamp",
        "feature_manifest_generated_at",
        "input_hashes",
        "jump_timestamp",
        "manifest_sha256",
        "metadata_capture_timestamp",
        "model_sha256",
        "odds_append_timestamp",
        "odds_capture_timestamp",
        "outcomes_present",
        "persisted",
        "persistence_status",
        "predictions",
        "probability_sums",
        "race_id",
        "record_key",
        "runner_set_sha256",
        "schema_version",
        "score_timestamp",
        "scoring_parity",
        "shadow_output_path",
        "source_contract",
        "status",
        "variants",
    }
)
EARLY_RESIDUAL_INPUT_HASH_KEYS = frozenset(
    {
        "capture_artifact_sha256",
        "feature_manifest_sha256",
        "feature_rows_sha256",
        "feature_source_sha256",
        "form_csv_sha256",
        "implementation_manifest_sha256",
        "odds_source_sha256",
        "selected_attempt_sha256",
        "sidecar_sha256",
    }
)
EARLY_RESIDUAL_RUNNER_PREDICTION_KEYS = frozenset(
    {
        "box",
        "dog",
        "full_minus_market",
        "full_probability",
        "half_probability",
        "market_probability",
        "rank",
        "win_odds",
    }
)
EARLY_RESIDUAL_PROBABILITY_SUM_KEYS = frozenset({"full", "half", "market"})
EARLY_RESIDUAL_SOURCE_CONTRACT_KEYS = frozenset(
    {
        "database_access",
        "feature_reconstruction_performed",
        "feature_source",
        "network_access",
    }
)
EARLY_RESIDUAL_VARIANT_KEYS = frozenset({"full_strength", "half_strength"})
EARLY_RESIDUAL_PARITY_KEYS = frozenset(
    {
        "input_schema_version",
        "input_sha256",
        "core_output_schema_version",
        "core_output_sha256",
        "config_sha256",
        "numeric_canonicalization_sha256",
    }
)
VENUE_CODE_PATTERN = r"[A-Z0-9_]+(?:-[A-Z0-9_]+)*"
NON_RACING_VENUE_IDENTITIES = frozenset({"UNKNOWN", "TEST_VEN", "RACE"})
FORBIDDEN_EVIDENCE_PATH_MARKERS = frozenset({"form_only_v1", "pr51"})
# Finite union of the exact GRADE_MAP and GRADE_VOCAB_MAP contracts, their
# canonical values, and source-observed exact labels covered by the scorer
# tests. Bare ``M`` is intentionally absent because the source contracts
# disagree on whether it means Maiden or Mixed.
GRADE_ALIASES = MappingProxyType(
    {
        "1": "GRADE 1",
        "1ST GRADE": "GRADE 1",
        "2": "GRADE 2",
        "2/3": "MIXED 2/3",
        "2/3/4": "MIXED 2/3/4",
        "2ND GRADE": "GRADE 2",
        "3": "GRADE 3",
        "3/4": "MIXED 3/4",
        "3/4/5": "MIXED 3/4/5",
        "3RD GRADE": "GRADE 3",
        "3RD/4TH GRADE": "3RD/4TH GRADE",
        "4": "GRADE 4",
        "4/5": "MIXED 4/5",
        "4TH GRADE": "GRADE 4",
        "4TH/5TH GRADE": "4TH/5TH GRADE",
        "5": "GRADE 5",
        "5/6": "MIXED 5/6",
        "5/M": "5/M",
        "5TH GRADE": "GRADE 5",
        "5TH/6TH GRADE": "5TH/6TH GRADE",
        "6": "GRADE 6",
        "6TH GRADE": "GRADE 6",
        "7": "GRADE 7",
        "7TH GRADE": "GRADE 7",
        "8": "GRADE 8",
        "BEST 8": "BEST 8",
        "BT8": "BT8",
        "FFA": "FREE FOR ALL",
        "FREE FOR ALL": "FREE FOR ALL",
        "GRADE 1": "GRADE 1",
        "GRADE 2": "GRADE 2",
        "GRADE 3": "GRADE 3",
        "GRADE 4": "GRADE 4",
        "GRADE 5": "GRADE 5",
        "GRADE 6": "GRADE 6",
        "GRADE 7": "GRADE 7",
        "GRADE 8": "GRADE 8",
        "GROUP 1": "GROUP 1",
        "GROUP 2": "GROUP 2",
        "GROUP 3": "GROUP 3",
        "I": "I",
        "INV": "INVITATION",
        "INVITATION": "INVITATION",
        "INVITATIONAL": "INVITATION",
        "J/M": "J/M",
        "M1/M2/M3": "M1/M2/M3",
        "M2/M3": "M2/M3",
        "M3": "M3",
        "M4/M5": "M4/M5",
        "M5": "M5",
        "M6": "M6",
        "MAIDEN": "MAIDEN",
        "MASTERS": "MASTERS",
        "MDN": "MAIDEN",
        "MI4/5MA": "MI4/5MA",
        "MIXED": "MIXED",
        "MIXED 2/3": "MIXED 2/3",
        "MIXED 2/3/4": "MIXED 2/3/4",
        "MIXED 3/4": "MIXED 3/4",
        "MIXED 3/4/5": "MIXED 3/4/5",
        "MIXED 4/5": "MIXED 4/5",
        "MIXED 5/6": "MIXED 5/6",
        "MIXED 6/7": "MIXED 6/7",
        "MX": "MIXED",
        "N/P": "N/P",
        "NG": "NON GRADED",
        "NG1-4": "NG1-4",
        "NON GRADED": "NON GRADED",
        "NOV": "NOVICE",
        "NOVICE": "NOVICE",
        "NP": "N/P",
        "OPEN": "OPEN",
        "OTHER": "OTHER",
        "P5": "P5",
        "PM": "PM",
        "R/W": "R/W",
        "RESTRICTED": "RESTRICTED WIN",
        "RESTRICTED WIN": "RESTRICTED WIN",
        "RESTRICTED WIN FINAL": "RESTRICTED WIN",
        "RESTRICTED WIN HEAT": "RESTRICTED WIN",
        "RW": "R/W",
        "S/E": "SPECIAL EVENT",
        "SE": "SPECIAL EVENT",
        "SPECIAL EVENT": "SPECIAL EVENT",
        "TG1-4W": "TG1-4W",
        "TG1-6W": "TG1-6W",
        "TG5+W": "TG5+W",
        "TIER 3 - GRADE 5": "GRADE 5",
        "TIER 3 - GRADE 6": "GRADE 6",
        "TIER 3 - GRADE 7": "GRADE 7",
        "TIER 3 - MAIDEN": "MAIDEN",
        "TIER 3 - RESTRICTED WIN": "RESTRICTED WIN",
    }
)


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


def _canonical_target_grade(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return GRADE_ALIASES.get(value.strip().upper())


def _contains_outcome_key(
    value: Any,
    forbidden_keys: frozenset[str] = OUTCOME_KEYS,
) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).strip().lower() in forbidden_keys
            or _contains_outcome_key(item, forbidden_keys)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_outcome_key(item, forbidden_keys) for item in value)
    return False


def _normalized_index_key(value: Any) -> str:
    raw_key = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(value).strip())
    return re.sub(r"[^a-z0-9]+", "_", raw_key.lower()).strip("_")


def _index_key_is_outcome(value: Any) -> bool:
    key = _normalized_index_key(value)
    tokens = frozenset(token for token in key.split("_") if token)
    return key in INDEX_OUTCOME_KEYS or bool(tokens & INDEX_OUTCOME_TOKENS)


def _contains_index_outcome_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = _normalized_index_key(key)
            if normalized_key in INDEX_FALSE_OUTCOME_MARKERS and item is False:
                continue
            if _index_key_is_outcome(key) or _contains_index_outcome_key(item):
                return True
        return False
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_index_outcome_key(item) for item in value)
    return False


def _require_index_keys(
    value: Any,
    allowed_keys: frozenset[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManualPredictionError("early_residual_status_index_shape_invalid")
    if set(value) - allowed_keys:
        raise ManualPredictionError("early_residual_status_index_unknown_field")
    return value


def _require_index_scalar_values(
    value: Mapping[str, Any],
    *,
    excluded_keys: frozenset[str] = frozenset(),
) -> None:
    if any(
        isinstance(item, (Mapping, list))
        for key, item in value.items()
        if key not in excluded_keys
    ):
        raise ManualPredictionError("early_residual_status_index_shape_invalid")


def _require_index_string_list(value: Any) -> None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ManualPredictionError("early_residual_status_index_shape_invalid")


def _validate_index_step(value: Any) -> None:
    if value is None:
        return
    step = _require_index_keys(value, EARLY_RESIDUAL_STEP_KEYS)
    if "command" in step:
        _require_index_string_list(step["command"])
    _require_index_scalar_values(step, excluded_keys=frozenset({"command"}))


def _validate_index_prediction(value: Any, *, expected_race_id: Any) -> None:
    if value is None:
        return
    prediction = _require_index_keys(value, EARLY_RESIDUAL_PREDICTION_KEYS)
    nested_mappings = {
        "input_hashes": EARLY_RESIDUAL_INPUT_HASH_KEYS,
        "probability_sums": EARLY_RESIDUAL_PROBABILITY_SUM_KEYS,
        "source_contract": EARLY_RESIDUAL_SOURCE_CONTRACT_KEYS,
        "variants": EARLY_RESIDUAL_VARIANT_KEYS,
        "scoring_parity": EARLY_RESIDUAL_PARITY_KEYS,
    }
    for key, allowed_keys in nested_mappings.items():
        if key not in prediction:
            continue
        nested = _require_index_keys(prediction[key], allowed_keys)
        _require_index_scalar_values(nested)
    if "predictions" in prediction:
        rows = prediction["predictions"]
        if not isinstance(rows, list):
            raise ManualPredictionError("early_residual_status_index_shape_invalid")
        for row in rows:
            runner = _require_index_keys(row, EARLY_RESIDUAL_RUNNER_PREDICTION_KEYS)
            _require_index_scalar_values(runner)
    _require_index_scalar_values(
        prediction,
        excluded_keys=frozenset({*nested_mappings, "predictions"}),
    )
    source_contract = prediction.get("source_contract")
    variants = prediction.get("variants")
    if (
        prediction.get("activation") is not False
        or prediction.get("outcomes_present") is not False
        or prediction.get("persisted") is not True
        or prediction.get("persistence_status") not in {"APPENDED", "EXACT_REPLAY"}
        or prediction.get("schema_version") != INDEX_PREDICTION_SCHEMA
        or prediction.get("status") != "MANUAL_PREJUMP_FROZEN_RESIDUAL_PREDICTION"
        or not isinstance(expected_race_id, str)
        or prediction.get("race_id") != expected_race_id
        or not isinstance(source_contract, Mapping)
        or source_contract.get("database_access") is not False
        or source_contract.get("network_access") is not False
        or source_contract.get("feature_reconstruction_performed") is not False
        or source_contract.get("feature_source")
        != "exact_hash_bound_system_shadow_feature_rows"
        or not isinstance(variants, Mapping)
        or type(variants.get("full_strength")) not in {int, float}
        or type(variants.get("half_strength")) not in {int, float}
        or float(variants["full_strength"]) != 1.0
        or float(variants["half_strength"]) != 0.5
    ):
        raise ManualPredictionError("early_residual_status_index_unsafe")


def _validate_index_authority_shape(status: Mapping[str, Any]) -> None:
    _require_index_keys(status, EARLY_RESIDUAL_STATUS_KEYS)
    plan = _require_index_keys(status.get("plan"), EARLY_RESIDUAL_PLAN_KEYS)
    if "blockers" in plan:
        _require_index_string_list(plan["blockers"])
    plan_races = plan.get("races")
    if not isinstance(plan_races, list):
        raise ManualPredictionError("early_residual_status_index_shape_invalid")
    for value in plan_races:
        race = _require_index_keys(value, EARLY_RESIDUAL_PLAN_RACE_KEYS)
        for key in ("feature_command", "score_command"):
            if key in race:
                _require_index_string_list(race[key])
        _require_index_scalar_values(
            race,
            excluded_keys=frozenset({"feature_command", "score_command"}),
        )
    status_races = status.get("races")
    if not isinstance(status_races, list):
        raise ManualPredictionError("early_residual_status_index_shape_invalid")
    for value in status_races:
        race = _require_index_keys(value, EARLY_RESIDUAL_STATUS_RACE_KEYS)
        _validate_index_step(race.get("feature_step"))
        _validate_index_step(race.get("score_step"))
        _validate_index_prediction(
            race.get("prediction"), expected_race_id=race.get("race_id")
        )
        _require_index_scalar_values(
            race,
            excluded_keys=frozenset({"feature_step", "score_step", "prediction"}),
        )
    _require_index_scalar_values(
        plan,
        excluded_keys=frozenset({"blockers", "races"}),
    )
    _require_index_scalar_values(
        status,
        excluded_keys=frozenset({"plan", "races"}),
    )


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


def _configured_venue_identity(value: Any) -> str | None:
    """Resolve only venue spellings proved by the checked-in canonical map."""

    if not isinstance(value, str):
        return None
    raw = value.strip().upper()
    if not raw or re.fullmatch(VENUE_CODE_PATTERN, raw) is None:
        return None
    candidates = {
        raw,
        raw.replace("-", " "),
        raw.replace("-", "_"),
        raw.replace("_", " "),
        raw.replace("_", "-"),
    }
    normalized = {
        normalize_venue(candidate)
        for candidate in candidates
        if candidate in VENUE_MAPPING
    }
    if len(normalized) != 1:
        return None
    configured = next(iter(normalized))
    canonical = canonical_thedogs_venue_identity(configured)
    if (
        configured in NON_RACING_VENUE_IDENTITIES
        or canonical in NON_RACING_VENUE_IDENTITIES
    ):
        return None
    return canonical


def _race_identity_equivalent(
    caller_race_id: Any,
    evidence_race_id: Any,
    *,
    source_url: Any,
) -> bool:
    """Bind caller and sealed-source aliases by exact structured identity."""

    try:
        caller_number, caller_venue, caller_date = _race_id_parts(caller_race_id)
        evidence_number, evidence_venue, evidence_date = _race_id_parts(
            evidence_race_id
        )
    except ManualPredictionError:
        return False
    source_identity = canonical_thedogs_race_identity(source_url)
    if source_identity is None:
        return False
    venues = (
        _configured_venue_identity(caller_venue),
        _configured_venue_identity(evidence_venue),
        _configured_venue_identity(source_identity["venue_slug"]),
    )
    return bool(
        all(venue is not None for venue in venues)
        and len(set(venues)) == 1
        and caller_date == evidence_date
        and caller_date.isoformat() == source_identity["race_date"]
        and caller_number == evidence_number
        and caller_number == source_identity["race_number"]
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


def _path_in_forbidden_pr51_domain(path: Path) -> bool:
    """Keep PR 51 FORM_ONLY_V1 acquisition and sealed domains isolated."""

    return any(
        any(marker in part.lower() for marker in FORBIDDEN_EVIDENCE_PATH_MARKERS)
        for part in path.resolve().parts
    )


def _require_non_pr51_artifact_paths(paths: Sequence[Path]) -> None:
    if any(_path_in_forbidden_pr51_domain(path) for path in paths):
        raise ManualPredictionError("pr51_form_only_v1_evidence_forbidden")


def _indexed_evidence_roots(
    evidence_root: Path,
    *,
    score_timestamp: datetime,
) -> list[Path]:
    """Resolve current sealed packet directories from outcome-free status indexes."""

    if score_timestamp.tzinfo is None or score_timestamp.utcoffset() is None:
        raise ManualPredictionError("score_timestamp_timezone_missing")
    root = evidence_root.resolve()
    if not root.is_dir():
        return []
    score_time = score_timestamp.astimezone(MELBOURNE)
    oldest_allowed = score_time - DEFAULT_INDEX_MAX_AGE
    indexed_roots: set[Path] = set()
    for status_path in sorted(root.glob(EARLY_RESIDUAL_STATUS_GLOB)):
        timestamp_match = re.fullmatch(
            r"shadow_autopilot_daemonization_v1_"
            r"(\d{8}T\d{6}[+-]\d{4})_odds_capture",
            status_path.parent.name,
        )
        if timestamp_match is None:
            continue
        try:
            status_time = datetime.strptime(
                timestamp_match.group(1), "%Y%m%dT%H%M%S%z"
            ).astimezone(MELBOURNE)
        except ValueError:
            continue
        if status_time < oldest_allowed or status_time > score_time:
            continue
        try:
            status_relative = status_path.relative_to(root)
        except ValueError:
            raise ManualPredictionError(
                "early_residual_status_index_path_escape"
            ) from None
        status_cursor = root
        for part in status_relative.parts:
            status_cursor = status_cursor / part
            if status_cursor.is_symlink():
                raise ManualPredictionError("early_residual_status_index_path_escape")
        try:
            resolved_status_path = status_path.resolve(strict=True)
        except OSError:
            raise ManualPredictionError(
                "early_residual_status_index_unreadable"
            ) from None
        if not _path_in_roots(resolved_status_path, [root]):
            raise ManualPredictionError("early_residual_status_index_path_escape")
        status_raw, _ = _read_input(resolved_status_path, "early_residual_status_index")
        status = _json_object(status_raw, "early_residual_status_index")
        if _contains_index_outcome_key(status):
            raise ManualPredictionError("early_residual_status_index_contains_outcome")
        plan = status.get("plan")
        if (
            not isinstance(plan, Mapping)
            or status.get("status") not in {"PASS", "BLOCKED"}
            or plan.get("status") != "READY"
        ):
            continue
        _validate_index_authority_shape(status)
        if (
            status.get("activation") is not False
            or status.get("outcomes_read") is not False
            or status.get("lock_release_preceded_stage_completion") is not False
            or plan.get("activation") is not False
            or plan.get("outcomes_read") is not False
            or plan.get("blockers") != []
            or plan.get("production_db_access") != "sqlite_mode_ro_feature_history_only"
        ):
            raise ManualPredictionError("early_residual_status_index_unsafe")
        if (
            status.get("schema_version") != EARLY_RESIDUAL_STATUS_SCHEMA
            or plan.get("schema_version") != EARLY_RESIDUAL_PLAN_SCHEMA
        ):
            raise ManualPredictionError("early_residual_status_index_schema_mismatch")
        expected_run_id = f"{timestamp_match.group(1)}_odds_capture"
        if plan.get("run_id") != expected_run_id:
            raise ManualPredictionError("early_residual_status_index_run_id_mismatch")
        races = plan.get("races")
        if not isinstance(races, list):
            raise ManualPredictionError("early_residual_status_index_races_invalid")
        if (
            type(status.get("race_count")) is not int
            or type(plan.get("race_count")) is not int
            or status.get("race_count") != len(races)
            or plan.get("race_count") != len(races)
        ):
            raise ManualPredictionError("early_residual_status_index_races_invalid")
        status_races = status.get("races")
        if not isinstance(status_races, list) or len(status_races) != len(races):
            raise ManualPredictionError("early_residual_status_index_races_invalid")
        if not all(
            isinstance(race, Mapping) and isinstance(race.get("race_id"), str)
            for race in races
        ) or not all(
            isinstance(race, Mapping)
            and isinstance(race.get("race_id"), str)
            and race.get("status") in {"APPENDED", "BLOCKED", "EXACT_REPLAY"}
            for race in status_races
        ):
            raise ManualPredictionError("early_residual_status_index_races_invalid")
        plan_race_ids = [str(race["race_id"]).strip() for race in races]
        status_race_ids = [str(race["race_id"]).strip() for race in status_races]
        if (
            any(not race_id for race_id in plan_race_ids)
            or status_race_ids != plan_race_ids
            or len(set(plan_race_ids)) != len(plan_race_ids)
        ):
            raise ManualPredictionError("early_residual_status_index_races_invalid")
        observed_counts = {
            "APPENDED": sum(race["status"] == "APPENDED" for race in status_races),
            "BLOCKED": sum(race["status"] == "BLOCKED" for race in status_races),
            "EXACT_REPLAY": sum(
                race["status"] == "EXACT_REPLAY" for race in status_races
            ),
        }
        declared_counts = (
            status.get("appended_count"),
            status.get("blocked_count"),
            status.get("exact_replay_count"),
        )
        if (
            any(type(value) is not int or value < 0 for value in declared_counts)
            or status.get("appended_count") != observed_counts["APPENDED"]
            or status.get("blocked_count") != observed_counts["BLOCKED"]
            or status.get("exact_replay_count") != observed_counts["EXACT_REPLAY"]
            or (status.get("status") == "PASS" and observed_counts["BLOCKED"] != 0)
            or (status.get("status") == "BLOCKED" and observed_counts["BLOCKED"] == 0)
        ):
            raise ManualPredictionError("early_residual_status_index_unsafe")
        for race in races:
            if not isinstance(race, Mapping):
                raise ManualPredictionError("early_residual_status_index_race_invalid")
            try:
                form_csv_path = Path(race["form_csv_path"]).resolve()
                sidecar_path = Path(race["sidecar_path"]).resolve()
                feature_output_dir = Path(race["feature_output_dir"]).resolve()
                capture_path = Path(race["capture_path"]).resolve()
            except (KeyError, TypeError):
                raise ManualPredictionError(
                    "early_residual_status_index_paths_invalid"
                ) from None
            indexed_paths = (
                form_csv_path,
                sidecar_path,
                feature_output_dir,
                capture_path,
            )
            if not all(_path_in_roots(path, [root]) for path in indexed_paths):
                raise ManualPredictionError("early_residual_status_index_path_escape")
            if sidecar_path != form_csv_path.with_name(
                form_csv_path.name + ".metadata.json"
            ):
                raise ManualPredictionError(
                    "early_residual_status_index_sidecar_mismatch"
                )
            feature_files = (
                feature_output_dir / "shadow_feature_rows.json",
                feature_output_dir / "shadow_manifest.json",
                feature_output_dir / "implementation_file_manifest.json",
            )
            if not (
                form_csv_path.is_file()
                and sidecar_path.is_file()
                and capture_path.is_file()
                and all(path.is_file() for path in feature_files)
            ):
                continue
            indexed_roots.update(
                {form_csv_path.parent, feature_output_dir, capture_path.parent}
            )
    return sorted(indexed_roots)


def _default_race_first_evidence_roots(score_timestamp: datetime) -> list[Path]:
    roots: set[Path] = set()
    if DEFAULT_EVIDENCE_ROOT.is_dir():
        roots.add(DEFAULT_EVIDENCE_ROOT.resolve())
    for retained_root in DEFAULT_RETAINED_EVIDENCE_ROOTS:
        roots.update(
            _indexed_evidence_roots(
                retained_root,
                score_timestamp=score_timestamp,
            )
        )
    return sorted(roots)


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
        rf"Race (\d+) - {VENUE_CODE_PATTERN} - \d{{4}}-\d{{2}}-\d{{2}}",
        race_id,
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


def _agreed_race_date(shadow: Mapping[str, Any], race_info: Mapping[str, Any]) -> date:
    if "race_date" not in shadow or "date" not in race_info:
        raise ManualPredictionError("target_race_date_missing")
    shadow_date = _parse_date(shadow["race_date"], "target_race_date")
    race_info_date = _parse_date(race_info["date"], "target_race_date")
    if shadow_date != race_info_date:
        raise ManualPredictionError("target_race_date_mismatch")
    return shadow_date


def _agreed_race_number(shadow: Mapping[str, Any], race_info: Mapping[str, Any]) -> int:
    if "race_number" not in shadow or "race_number" not in race_info:
        raise ManualPredictionError("target_race_number_missing")
    shadow_number = _integer(shadow["race_number"], "target_race_number", minimum=1)
    race_info_number = _integer(
        race_info["race_number"], "target_race_number", minimum=1
    )
    if shadow_number != race_info_number:
        raise ManualPredictionError("target_race_number_mismatch")
    return shadow_number


def _agreed_distance(shadow: Mapping[str, Any], race_info: Mapping[str, Any]) -> float:
    if "distance" not in shadow or "distance" not in race_info:
        raise ManualPredictionError("target_distance_missing")
    shadow_distance = _distance_metres(shadow["distance"])
    race_info_distance = _distance_metres(race_info["distance"])
    if shadow_distance != race_info_distance:
        raise ManualPredictionError("target_distance_mismatch")
    return shadow_distance


def _jump_timestamp(sidecar: Mapping[str, Any], race_date: date) -> datetime:
    shadow = sidecar.get("prejump_shadow_metadata")
    race_info = sidecar.get("race_info")
    if not isinstance(shadow, Mapping):
        raise ManualPredictionError("prejump_shadow_metadata_missing")
    race_info = race_info if isinstance(race_info, Mapping) else {}
    supplied_datetime_timestamps = []
    for mapping in (shadow, sidecar):
        if "jump_datetime" in mapping:
            value = mapping["jump_datetime"]
            if (
                value is None
                or isinstance(value, bool)
                or not isinstance(value, str)
                or not value.strip()
            ):
                raise ManualPredictionError("jump_timestamp_invalid")
            supplied_datetime_timestamps.append(
                _parse_timestamp(value, "jump_timestamp").timestamp()
            )

    supplied_time_timestamps = []
    for mapping, key in ((shadow, "jump_time"), (race_info, "race_time")):
        if key not in mapping:
            continue
        value = mapping[key]
        if (
            value is None
            or isinstance(value, bool)
            or not isinstance(value, str)
            or not value.strip()
        ):
            raise ManualPredictionError("jump_time_invalid")
        for fmt in ("%I:%M %p", "%I:%M%p", "%H:%M", "%H:%M:%S"):
            try:
                parsed_time = datetime.strptime(value.strip().upper(), fmt).time()
                supplied_time_timestamps.append(
                    datetime.combine(
                        race_date, parsed_time, tzinfo=MELBOURNE
                    ).timestamp()
                )
                break
            except ValueError:
                continue
        else:
            raise ManualPredictionError("jump_time_invalid")

    supplied = supplied_datetime_timestamps + supplied_time_timestamps
    if not supplied:
        raise ManualPredictionError("jump_time_invalid")
    if any(value != supplied[0] for value in supplied[1:]):
        raise ManualPredictionError("jump_timestamp_mismatch")
    timezone = ZoneInfo("UTC") if supplied_datetime_timestamps else MELBOURNE
    return datetime.fromtimestamp(supplied[0], tz=timezone)


def _agreed_sidecar_value(sidecar: Mapping[str, Any], key: str) -> Any:
    race_info = sidecar.get("race_info")
    race_info = race_info if isinstance(race_info, Mapping) else {}
    values = []
    for mapping in (sidecar, race_info):
        if key not in mapping:
            continue
        value = mapping[key]
        if (
            value is None
            or isinstance(value, bool)
            or (isinstance(value, str) and not value.strip())
        ):
            raise ManualPredictionError(f"sidecar_{key}_invalid")
        if (key.endswith("url") or key.endswith("source_url")) and not isinstance(
            value, str
        ):
            raise ManualPredictionError(f"sidecar_{key}_invalid")
        values.append(value)
    if not values:
        raise ManualPredictionError(f"sidecar_{key}_missing")
    if key.endswith("url") or key.endswith("source_url"):
        canonical = [
            canonical_thedogs_race_identity(value)["canonical_url"]
            if canonical_thedogs_race_identity(value)
            else str(value).strip()
            for value in values
        ]
        if any(value != canonical[0] for value in canonical[1:]):
            raise ManualPredictionError(f"sidecar_{key}_mismatch")
    elif any(value != values[0] for value in values[1:]):
        raise ManualPredictionError(f"sidecar_{key}_mismatch")
    return values[0]


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
    source_urls = []
    for mapping, key in (
        (shadow, "source_url"),
        (sidecar, "race_url"),
        (race_info, "url"),
    ):
        if key not in mapping:
            continue
        value = mapping[key]
        if (
            value is None
            or isinstance(value, bool)
            or not isinstance(value, str)
            or not value.strip()
        ):
            raise ManualPredictionError("sidecar_source_url_alias_invalid")
        source_urls.append(value.strip())
    if any(not _trusted_thedogs_url(value) for value in source_urls):
        if len(set(source_urls)) == 1:
            raise ManualPredictionError("sidecar_source_url_not_trusted_thedogs")
        raise ManualPredictionError("sidecar_source_url_alias_mismatch")
    source_identities = [
        canonical_thedogs_race_identity(value) for value in source_urls
    ]
    if not source_urls or any(identity is None for identity in source_identities):
        raise ManualPredictionError("sidecar_source_url_alias_mismatch")
    if len({identity["canonical_url"] for identity in source_identities}) != 1:
        raise ManualPredictionError("sidecar_source_url_alias_mismatch")
    source_url = source_urls[0]
    source_identity = source_identities[0]
    if source_identity is None:
        raise ManualPredictionError("sidecar_source_url_not_canonical_thedogs_race")
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
    target_date = _agreed_race_date(shadow, race_info)
    race_number = _agreed_race_number(shadow, race_info)
    venue_values = []
    if "venue" in shadow:
        venue_values.append(shadow.get("venue"))
    if "venue" in race_info:
        venue_values.append(race_info.get("venue"))
    if not venue_values or any(not isinstance(value, str) for value in venue_values):
        raise ManualPredictionError("target_venue_invalid")
    normalized_venues = [value.strip().upper() for value in venue_values]
    if any(
        not venue or not re.fullmatch(VENUE_CODE_PATTERN, venue)
        for venue in normalized_venues
    ):
        raise ManualPredictionError("target_venue_invalid")
    if len(set(normalized_venues)) != 1:
        raise ManualPredictionError("target_venue_alias_mismatch")
    venue = normalized_venues[0]
    target_distance = _agreed_distance(shadow, race_info)
    target_grade_values = []
    if "grade" in shadow:
        target_grade_values.append(shadow.get("grade"))
    if "grade" in race_info:
        target_grade_values.append(race_info.get("grade"))
    if not target_grade_values or any(
        value is None or value == "" for value in target_grade_values
    ):
        raise ManualPredictionError("target_grade_missing")
    target_grade_canonicals = [
        _canonical_target_grade(value) for value in target_grade_values
    ]
    if any(value is None for value in target_grade_canonicals):
        raise ManualPredictionError("target_grade_invalid")
    if len(set(target_grade_canonicals)) != 1:
        raise ManualPredictionError("target_grade_alias_mismatch")
    target_grade_value = target_grade_values[0]
    target_grade_canonical = target_grade_canonicals[0]
    grade_source = _agreed_sidecar_value(sidecar, "target_grade_source")
    grade_schema = _agreed_sidecar_value(sidecar, "target_grade_context_schema")
    grade_exact_value = _agreed_sidecar_value(sidecar, "target_grade_exact_value")
    grade_proof_key = _agreed_sidecar_value(sidecar, "target_grade_equivalence_key")
    grade_race_url = _agreed_sidecar_value(sidecar, "target_grade_race_url")
    grade_source_url = _agreed_sidecar_value(sidecar, "target_grade_source_url")
    grade_source_sha256 = (
        str(_agreed_sidecar_value(sidecar, "target_grade_source_sha256"))
        .strip()
        .lower()
    )
    grade_race_date = str(
        _agreed_sidecar_value(sidecar, "target_grade_race_date")
    ).strip()
    grade_race_number = _integer(
        _agreed_sidecar_value(sidecar, "target_grade_race_number"),
        "target_grade_race_number",
        minimum=1,
    )
    grade_venue = _agreed_sidecar_value(sidecar, "target_grade_venue")
    grade_identity = canonical_thedogs_race_identity(grade_race_url)
    grade_source_identity = canonical_thedogs_race_identity(grade_source_url)
    normalized_exact_grade = normalize_exact_target_grade(grade_exact_value)
    expected_grade_schema = {
        THEDOGS_MEETING_CARD_GRADE_SOURCE: "thedogs_meeting_card_exact_race_v1",
        THEDOGS_EXACT_RACE_PAGE_GRADE_SOURCE: "thedogs_exact_race_page_v1",
    }.get(grade_source)
    grade_source_url_valid = bool(
        (
            grade_source == THEDOGS_MEETING_CARD_GRADE_SOURCE
            and canonical_thedogs_meeting_card_url(
                grade_source_url,
                race_date=source_identity["race_date"],
            )
            is not None
        )
        or (
            grade_source == THEDOGS_EXACT_RACE_PAGE_GRADE_SOURCE
            and grade_source_identity is not None
            and grade_source_identity["canonical_url"]
            == source_identity["canonical_url"]
        )
    )
    if (
        expected_grade_schema is None
        or grade_schema != expected_grade_schema
        or grade_identity is None
        or grade_identity["canonical_url"] != source_identity["canonical_url"]
        or target_date.isoformat() != source_identity["race_date"]
        or race_number != source_identity["race_number"]
        or canonical_thedogs_venue_identity(venue)
        != canonical_thedogs_venue_identity(source_identity["venue_slug"])
        or grade_race_date != source_identity["race_date"]
        or grade_race_number != source_identity["race_number"]
        or canonical_thedogs_venue_identity(grade_venue)
        != canonical_thedogs_venue_identity(source_identity["venue_slug"])
        or normalized_exact_grade is None
        or target_grade_equivalence_key(grade_exact_value) != grade_proof_key
        or target_grade_equivalence_key(target_grade_value) != grade_proof_key
        or not grade_source_url_valid
        or not re.fullmatch(r"[0-9a-f]{64}", grade_source_sha256)
    ):
        raise ManualPredictionError("target_grade_proof_mismatch")
    jump = _jump_timestamp(sidecar, target_date)
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
        "target_grade": target_grade_value,
        "target_grade_canonical": target_grade_canonical,
        "target_grade_proof_key": grade_proof_key,
        "target_grade_source_sha256": grade_source_sha256,
        "target_grade_source_url": str(grade_source_url),
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
        raise ManualPredictionError(
            f"capture_validation_fields_missing:{sorted(missing)}"
        )
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
) -> None:
    declared_hashes = implementation_manifest.get("implementation_file_hashes")
    if not isinstance(declared_hashes, Mapping):
        raise ManualPredictionError("feature_generator_implementation_hashes_missing")
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
    _validate_feature_generator_identity(implementation_manifest)
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
        if isinstance(row, Mapping)
        and _race_identity_equivalent(
            race_id,
            row.get("race_id"),
            source_url=context["source_url"],
        )
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
        if (
            _configured_venue_identity(row.get("venue"))
            != _configured_venue_identity(context["target_venue"])
        ):
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
        feature_grade_canonical = _canonical_target_grade(row.get("target_grade_safe"))
        if feature_grade_canonical is None:
            raise ManualPredictionError("feature_row_target_grade_invalid")
        if feature_grade_canonical != context["target_grade_canonical"]:
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
    exact_race_id: str | None = None,
    evidence_roots: Sequence[Path],
    score_timestamp: datetime,
) -> dict[str, Any]:
    """Resolve one race to one sealed feature packet and one capture report."""

    if score_timestamp.tzinfo is None or score_timestamp.utcoffset() is None:
        raise ManualPredictionError("score_timestamp_timezone_missing")
    roots = sorted({Path(root).resolve() for root in evidence_roots})
    if not roots or any(not root.is_dir() for root in roots):
        raise ManualPredictionError("evidence_root_missing_or_not_directory")
    if any(_path_in_forbidden_pr51_domain(root) for root in roots):
        raise ManualPredictionError("pr51_form_only_v1_evidence_forbidden")
    race_number, venue_query = _race_query_parts(race_query)
    exact_identity = None
    exact_venue = None
    if exact_race_id is not None:
        exact_identity = str(exact_race_id).strip()
        _, exact_venue, _ = _race_id_parts(exact_identity)

    feature_candidates: list[dict[str, Any]] = []
    seen_feature_packets: set[Path] = set()
    for root in roots:
        for feature_rows_path in sorted(root.rglob("shadow_feature_rows.json")):
            feature_rows_path = feature_rows_path.resolve()
            if _path_in_forbidden_pr51_domain(feature_rows_path):
                continue
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
                source_urls = {
                    str(row.get("target_metadata_source_url") or "").strip()
                    for row in selected_rows
                }
                source_urls.discard("")
                if exact_identity is not None and (
                    len(source_urls) != 1
                    or not _race_identity_equivalent(
                        exact_identity,
                        race_id,
                        source_url=next(iter(source_urls), None),
                    )
                ):
                    continue
                aliases: list[str] = []
                if exact_venue is not None:
                    aliases.append(exact_venue)
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
                if any(
                    _path_in_forbidden_pr51_domain(path)
                    for path in (form_csv_path, sidecar_path)
                ):
                    continue
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
    race_id = exact_identity or str(selected_feature["race_id"])

    capture_candidates: list[dict[str, Any]] = []
    seen_capture_reports: set[Path] = set()
    for root in roots:
        for capture_path in sorted(
            root.rglob("autonomous_live_odds_capture_report.json")
        ):
            capture_path = capture_path.resolve()
            if _path_in_forbidden_pr51_domain(capture_path):
                continue
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
        "race_id": race_id,
        "capture_path": selected_captures[0]["capture_path"],
    }


def _normalized_thedogs_race_url(value: Any) -> str | None:
    identity = canonical_thedogs_race_identity(value)
    return str(identity["canonical_url"]) if identity is not None else None


def _thedogs_race_url_parts(value: Any) -> tuple[str, date, int] | None:
    identity = canonical_thedogs_race_identity(value)
    if identity is None:
        return None
    return (
        str(identity["venue_slug"]),
        date.fromisoformat(str(identity["race_date"])),
        int(identity["race_number"]),
    )


def _diagnostic_venue_identity_matches(selected_venue: str, venue_slug: str) -> bool:
    """Require the report venue and URL slug to belong to one declared identity."""

    def configured(value: str) -> str:
        raw = str(value or "").strip()
        token = _runner_token(raw)
        if token in DIAGNOSTIC_VENUE_CODE_OVERRIDES:
            return DIAGNOSTIC_VENUE_CODE_OVERRIDES[token]
        candidates = (
            raw,
            raw.replace("-", " "),
            raw.replace("-", "_"),
            raw.replace("_", " "),
            raw.replace("_", "-"),
        )
        for candidate in candidates:
            normalized = normalize_venue(candidate)
            if normalized != candidate.upper():
                normalized_token = _runner_token(normalized)
                return DIAGNOSTIC_VENUE_CODE_OVERRIDES.get(
                    normalized_token, normalized_token
                )
        return DIAGNOSTIC_VENUE_CODE_OVERRIDES.get(token, token)

    return configured(selected_venue) == configured(venue_slug)


def _refresh_report_contains_outcome_key(
    value: Any,
    *,
    path: tuple[str, ...] = (),
) -> bool:
    """Reject outcome fields while allowing the downloader's operation result wrapper."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = _normalized_index_key(key)
            if normalized_key == "no_result_ingest" and item is True:
                continue
            if normalized_key in INDEX_FALSE_OUTCOME_MARKERS and item is False:
                continue
            if normalized_key == "result" and path == ("downloads", "*"):
                if _refresh_report_contains_outcome_key(
                    item,
                    path=(*path, normalized_key),
                ):
                    return True
                continue
            if _index_key_is_outcome(key):
                return True
            if _refresh_report_contains_outcome_key(
                item,
                path=(*path, normalized_key),
            ):
                return True
        return False
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(
            _refresh_report_contains_outcome_key(item, path=(*path, "*"))
            for item in value
        )
    return False


def _resolve_diagnostic_path(
    root: Path,
    raw_path: Any,
    *,
    label: str,
    directory: bool,
) -> Path:
    path = Path(str(raw_path or ""))
    if not path.is_absolute() or ".." in path.parts:
        raise ManualPredictionError(f"{label}_path_invalid")
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise ManualPredictionError(f"{label}_path_escape") from None
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ManualPredictionError(f"{label}_path_escape")
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        raise ManualPredictionError(f"{label}_unreadable") from None
    if not _path_in_roots(resolved, [root]):
        raise ManualPredictionError(f"{label}_path_escape")
    if directory and not resolved.is_dir():
        raise ManualPredictionError(f"{label}_unreadable")
    if not directory and not resolved.is_file():
        raise ManualPredictionError(f"{label}_unreadable")
    return resolved


def _quarantined_race_reason(
    *,
    race_query: str,
    diagnostic_roots: Sequence[Path],
    score_timestamp: datetime,
) -> str | None:
    """Return one closed outcome-free refresh quarantine reason for an exact race."""

    if score_timestamp.tzinfo is None or score_timestamp.utcoffset() is None:
        raise ManualPredictionError("score_timestamp_timezone_missing")
    race_number, venue_query = _race_query_parts(race_query)
    score_time = score_timestamp.astimezone(MELBOURNE)
    oldest_allowed = score_time - DEFAULT_INDEX_MAX_AGE
    candidates: list[dict[str, Any]] = []

    for raw_root in sorted({Path(root).resolve() for root in diagnostic_roots}):
        if not raw_root.is_dir():
            continue
        root = raw_root.resolve()
        for status_path in sorted(root.glob(EARLY_RESIDUAL_STATUS_GLOB)):
            timestamp_match = re.fullmatch(
                r"shadow_autopilot_daemonization_v1_"
                r"(\d{8}T\d{6}[+-]\d{4})_odds_capture",
                status_path.parent.name,
            )
            if timestamp_match is None:
                continue
            try:
                status_time = datetime.strptime(
                    timestamp_match.group(1), "%Y%m%dT%H%M%S%z"
                ).astimezone(MELBOURNE)
            except ValueError:
                continue
            if status_time < oldest_allowed or status_time > score_time:
                continue
            status_path = _resolve_diagnostic_path(
                root,
                status_path,
                label="refresh_quarantine_status_index",
                directory=False,
            )
            status_raw, _ = _read_input(
                status_path,
                "refresh_quarantine_status_index",
            )
            status = _json_object(status_raw, "refresh_quarantine_status_index")
            if _contains_index_outcome_key(status):
                raise ManualPredictionError(
                    "refresh_quarantine_status_index_contains_outcome"
                )
            plan = status.get("plan")
            if not isinstance(plan, Mapping):
                continue
            status_state = status.get("status")
            plan_state = plan.get("status")
            if (status_state, plan_state) == (
                "SKIPPED_NO_NEW_CAPTURE",
                "SKIPPED_NO_NEW_CAPTURE",
            ):
                if plan.get("blockers") != []:
                    continue
                expected_blockers: list[str] = []
            elif (status_state, plan_state) == ("BLOCKED", "BLOCKED"):
                if plan.get("blockers") != ["residual_feature_handoff_not_pass"]:
                    continue
                expected_blockers = ["residual_feature_handoff_not_pass"]
            else:
                continue
            _validate_index_authority_shape(status)
            expected_run_id = f"{timestamp_match.group(1)}_odds_capture"
            if (
                status.get("schema_version") != EARLY_RESIDUAL_STATUS_SCHEMA
                or plan.get("schema_version") != EARLY_RESIDUAL_PLAN_SCHEMA
                or plan.get("run_id") != expected_run_id
            ):
                raise ManualPredictionError(
                    "refresh_quarantine_status_index_identity_mismatch"
                )
            if (
                status.get("activation") is not False
                or status.get("outcomes_read") is not False
                or status.get("lock_release_preceded_stage_completion") is not False
                or plan.get("activation") is not False
                or plan.get("outcomes_read") is not False
                or plan.get("blockers") != expected_blockers
                or plan.get("production_db_access")
                != "sqlite_mode_ro_feature_history_only"
                or status.get("race_count") != 0
                or plan.get("race_count") != 0
                or status.get("races") != []
                or plan.get("races") != []
                or status.get("appended_count") != 0
                or status.get("blocked_count") != 0
                or status.get("exact_replay_count") != 0
            ):
                raise ManualPredictionError("refresh_quarantine_status_index_unsafe")

            expected_output_name = f"shadow_autopilot_v1_{expected_run_id}_autopilot"
            output_dir = _resolve_diagnostic_path(
                root,
                plan.get("autopilot_output_dir"),
                label="refresh_quarantine_output_dir",
                directory=True,
            )
            if output_dir.name != expected_output_name:
                raise ManualPredictionError(
                    "refresh_quarantine_output_dir_identity_mismatch"
                )
            shadow_output_path = plan.get("shadow_output_path")
            if shadow_output_path and not _path_in_roots(
                Path(str(shadow_output_path)).resolve(),
                [root],
            ):
                raise ManualPredictionError(
                    "refresh_quarantine_status_index_path_escape"
                )

            report_path = _resolve_diagnostic_path(
                root,
                output_dir / REFRESH_REPORT_FILENAME,
                label="refresh_quarantine_report",
                directory=False,
            )
            report_raw, _ = _read_input(report_path, "refresh_quarantine_report")
            report = _json_object(report_raw, "refresh_quarantine_report")
            if _refresh_report_contains_outcome_key(report):
                raise ManualPredictionError(
                    "refresh_quarantine_report_contains_outcome"
                )
            report_time = _parse_timestamp(
                report.get("generated_at"),
                "refresh_quarantine_report_generated_at",
            ).astimezone(MELBOURNE)
            if report_time < oldest_allowed or report_time > score_time:
                continue
            if not (
                status_time <= report_time <= status_time + MAX_REFRESH_REPORT_RUN_LAG
            ):
                raise ManualPredictionError(
                    "refresh_quarantine_report_run_time_mismatch"
                )
            selected_races = report.get("selected_races")
            downloads = report.get("downloads")
            if (
                report.get("no_snapshot_persist") is not True
                or report.get("no_odds_capture") is not True
                or report.get("no_result_ingest") is not True
                or report.get("no_label_write") is not True
                or report.get("no_retrain_or_promotion") is not True
                or type(report.get("selected_count")) is not int
                or not isinstance(selected_races, list)
                or report.get("selected_count") != len(selected_races)
                or not isinstance(downloads, list)
                or type(report.get("quarantine_count")) is not int
            ):
                raise ManualPredictionError("refresh_quarantine_report_unsafe")
            if (
                report.get("status") != "METADATA_COVERAGE_INCOMPLETE"
                or report.get("quarantine_count") < 1
            ):
                continue

            for selected in selected_races:
                if not isinstance(selected, Mapping):
                    raise ManualPredictionError(
                        "refresh_quarantine_report_shape_invalid"
                    )
                source_url = selected.get("race_url")
                normalized_url = _normalized_thedogs_race_url(source_url)
                url_parts = _thedogs_race_url_parts(source_url)
                if normalized_url is None or url_parts is None:
                    raise ManualPredictionError("refresh_quarantine_race_url_invalid")
                venue_slug, candidate_date, candidate_number = url_parts
                try:
                    selected_number = _integer(
                        selected.get("race_number"),
                        "refresh_quarantine_race_number",
                        minimum=1,
                    )
                    selected_date = _parse_date(
                        selected.get("date"),
                        "refresh_quarantine_race_date",
                    )
                except ManualPredictionError as exc:
                    raise ManualPredictionError(
                        "refresh_quarantine_report_shape_invalid"
                    ) from exc
                if (
                    selected_number != candidate_number
                    or selected_date != candidate_date
                    or candidate_number != race_number
                ):
                    continue

                canonical_venue = str(selected.get("venue") or "").strip().upper()
                raw_aliases = selected.get("race_id_aliases")
                try:
                    primary_number, primary_venue, primary_date = _race_id_parts(
                        str(selected.get("race_id") or "")
                    )
                except ManualPredictionError as exc:
                    raise ManualPredictionError(
                        "refresh_quarantine_race_identity_mismatch"
                    ) from exc
                if (
                    not canonical_venue
                    or not _diagnostic_venue_identity_matches(
                        canonical_venue, venue_slug
                    )
                    or primary_number != candidate_number
                    or primary_date != candidate_date
                    or _runner_token(primary_venue) != _runner_token(canonical_venue)
                    or not isinstance(raw_aliases, list)
                    or not all(isinstance(value, str) for value in raw_aliases)
                    or len(set(raw_aliases)) != len(raw_aliases)
                    or selected.get("race_id") not in raw_aliases
                ):
                    raise ManualPredictionError(
                        "refresh_quarantine_race_identity_mismatch"
                    )
                aliases = []
                for raw_race_id in raw_aliases:
                    try:
                        alias_number, alias_venue, alias_date = _race_id_parts(
                            raw_race_id
                        )
                    except ManualPredictionError as exc:
                        raise ManualPredictionError(
                            "refresh_quarantine_race_identity_mismatch"
                        ) from exc
                    if (
                        alias_number != candidate_number
                        or alias_date != candidate_date
                        or not _diagnostic_venue_identity_matches(
                            canonical_venue, alias_venue
                        )
                    ):
                        raise ManualPredictionError(
                            "refresh_quarantine_race_identity_mismatch"
                        )
                    aliases.append(alias_venue)
                if _runner_token(venue_slug) not in {
                    _runner_token(alias) for alias in aliases
                }:
                    raise ManualPredictionError(
                        "refresh_quarantine_race_identity_mismatch"
                    )
                venue_match_rank = _venue_query_match_rank(
                    venue_query,
                    canonical_venue=canonical_venue,
                    full_aliases=aliases,
                )
                if venue_match_rank is None:
                    continue

                matching_downloads = [
                    download
                    for download in downloads
                    if isinstance(download, Mapping)
                    and _normalized_thedogs_race_url(download.get("race_url"))
                    == normalized_url
                ]
                if len(matching_downloads) != 1:
                    raise ManualPredictionError("refresh_quarantine_download_ambiguous")
                download = matching_downloads[0]
                result = download.get("result")
                normalization = (
                    result.get("normalization")
                    if isinstance(result, Mapping)
                    and isinstance(result.get("normalization"), Mapping)
                    else None
                )
                if (
                    download.get("success") is not False
                    or not isinstance(result, Mapping)
                    or result.get("success") is not False
                    or not isinstance(normalization, Mapping)
                    or normalization.get("normalization_status") != "rejected"
                ):
                    continue
                reason = REFRESH_QUARANTINE_REASONS.get(
                    normalization.get("normalization_failure_reason")
                )
                if reason is None:
                    continue
                candidates.append(
                    {
                        "race_date": candidate_date,
                        "report_time": report_time,
                        "reason": reason,
                        "venue": canonical_venue,
                        "venue_match_rank": venue_match_rank,
                    }
                )

    if not candidates:
        return None
    best_venue_rank = min(candidate["venue_match_rank"] for candidate in candidates)
    candidates = [
        candidate
        for candidate in candidates
        if candidate["venue_match_rank"] == best_venue_rank
    ]
    if len({candidate["venue"] for candidate in candidates}) != 1:
        raise ManualPredictionError("race_quarantine_report_ambiguous")
    current_date = score_time.date()
    available_dates = sorted(
        {
            candidate["race_date"]
            for candidate in candidates
            if candidate["race_date"] >= current_date
        }
    )
    target_date = (
        available_dates[0]
        if available_dates
        else max(candidate["race_date"] for candidate in candidates)
    )
    candidates = [
        candidate for candidate in candidates if candidate["race_date"] == target_date
    ]
    latest_report_time = max(candidate["report_time"] for candidate in candidates)
    candidates = [
        candidate
        for candidate in candidates
        if candidate["report_time"] == latest_report_time
    ]
    if len(candidates) != 1:
        raise ManualPredictionError("race_quarantine_report_ambiguous")
    return str(candidates[0]["reason"])


def discover_race_artifacts_with_diagnostics(
    *,
    race_query: str,
    evidence_roots: Sequence[Path],
    diagnostic_roots: Sequence[Path],
    score_timestamp: datetime,
) -> dict[str, Any]:
    """Discover a scoreable packet or expose one closed upstream quarantine reason."""

    try:
        return discover_race_artifacts(
            race_query=race_query,
            evidence_roots=evidence_roots,
            score_timestamp=score_timestamp,
        )
    except ManualPredictionError as exc:
        if str(exc) not in {
            "race_feature_packet_not_found",
            "evidence_root_missing_or_not_directory",
        }:
            raise
        reason = _quarantined_race_reason(
            race_query=race_query,
            diagnostic_roots=diagnostic_roots,
            score_timestamp=score_timestamp,
        )
        if reason is not None:
            raise ManualPredictionError(
                f"race_feature_packet_quarantined:{reason}"
            ) from None
        raise


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
    _require_non_pr51_artifact_paths(
        (
            form_csv_path,
            sidecar_path,
            feature_rows_path,
            feature_manifest_path,
            implementation_manifest_path,
            capture_path,
        )
    )
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
    if not _race_identity_equivalent(
        race_id,
        context["expected_race_id"],
        source_url=context["source_url"],
    ):
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
        attempt, context["runners"]
    )
    jump = context["jump_timestamp"]
    score_time = score_timestamp or datetime.now().astimezone()
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
    frozen = load_frozen_model(model_path, manifest_path)
    expected_ids = sorted(runner_ids)
    scoring_input = build_scoring_input(
        race_id=race_id,
        runner_set_sha256=_runner_set_sha256(expected_ids),
        runners=[
            {key: value for key, value in runner.items() if key != "race_id"}
            for runner in runners
        ],
        cutoff_timestamp=feature_time.isoformat(),
        capture_timestamp=fetch_time.isoformat(),
        score_timestamp=score_time.isoformat(),
        jump_timestamp=jump.isoformat(),
        model_sha256=frozen.model_sha256,
        manifest_sha256=frozen.manifest_sha256,
        effective_state_sha256=frozen.effective_state_sha256,
        config_sha256=SCORING_CONFIG_SHA256,
        scoring_parameters={
            "full_strength": frozen.full_strength,
            "half_strength": frozen.half_strength,
            "residual_cap": frozen.residual_cap,
            "within_race_centering": frozen.within_race_centering,
            "market_offset": frozen.market_offset,
            "normalization": frozen.normalization,
        },
    )
    record = score_race(frozen, scoring_input.scorer_runners, scoring_input.provenance)
    if record.get("schema_version") != SHADOW_RECORD_SCHEMA:
        raise ManualPredictionError("scorer_record_schema_mismatch")
    core_output = build_core_output(scoring_input, record)
    scoring_parity = parity_binding(scoring_input, core_output)
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
        "record_schema_version": record["schema_version"],
        "record_checksum_sha256": record["record_checksum_sha256"],
        "scoring_parity": scoring_parity,
        "effective_state_schema_version": EFFECTIVE_STATE_SCHEMA,
        "effective_state_sha256": record["effective_state_sha256"],
        "numerical_canonicalization_contract": dict(
            NUMERICAL_CANONICALIZATION_CONTRACT
        ),
        "canonical_runner_order": [
            {
                "box": int(row["box_number"]),
                "dog": row["dog_name"],
                "runner_id": row["runner_id"],
            }
            for row in runners
        ],
        "target_grade": context["target_grade"],
        "target_grade_proof_key": context["target_grade_proof_key"],
        "target_grade_source_url": context["target_grade_source_url"],
        "target_grade_source_sha256": context["target_grade_source_sha256"],
        "variants": {"full_strength": 1.0, "half_strength": 0.5},
        "source_contract": {
            "feature_source": "exact_hash_bound_system_shadow_feature_rows",
            "feature_reconstruction_performed": False,
            "database_access": False,
            "network_access": False,
            "manual_scoring_read_only": True,
            "persistence_interface_crossed": False,
            "persistence_status": "NOT_REQUESTED_READ_ONLY",
            "writer_status_contract": [
                "APPENDED",
                "EXACT_REPLAY",
                "COMMIT_STATE_UNKNOWN",
            ],
            "history_migration_performed": False,
        },
        "verified_artifacts": {
            "model_sha256": record["model_sha256"],
            "manifest_sha256": record["manifest_sha256"],
            "feature_source_sha256": feature_source_sha,
            "odds_source_sha256": odds_source_sha,
            "meeting_card_source_sha256": context["target_grade_source_sha256"],
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
            "meeting_card_source_sha256": context["target_grade_source_sha256"],
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
            key: math.fsum(float(row[f"{key}_probability"]) for row in ranking)
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
            "defaults to the repository root plus current sealed system packet indexes."
        ),
    )
    parser.add_argument("--form-csv", type=Path)
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--feature-rows", type=Path)
    parser.add_argument("--feature-manifest", type=Path)
    parser.add_argument("--implementation-manifest", type=Path)
    parser.add_argument("--capture", type=Path)
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
            evidence_roots = args.evidence_root or _default_race_first_evidence_roots(
                score_time
            )
            diagnostic_roots = args.evidence_root or [
                DEFAULT_EVIDENCE_ROOT,
                *DEFAULT_RETAINED_EVIDENCE_ROOTS,
            ]
            discovered = discover_race_artifacts_with_diagnostics(
                race_query=args.race,
                evidence_roots=evidence_roots,
                diagnostic_roots=diagnostic_roots,
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
            score_timestamp=score_time if args.race else None,
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
