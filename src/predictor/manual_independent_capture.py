"""Fail-closed contract for isolated manual research capture artifacts.

This module defines data and authority contracts only.  It does not acquire a
lock, launch a browser, read a source, write an artifact, or score a race.
"""

from __future__ import annotations

import json
import math
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any

from scripts.refresh_prejump_upcoming import stable_race_id, stable_race_id_variants
from src.predictor.on_demand import (
    PredictionBlocked,
    canonical_bytes,
    canonical_runner_set,
    sealed_runner_set_sha256,
    sha256_bytes,
)
from utils.csv_metadata import (
    canonical_thedogs_race_identity,
    canonical_thedogs_venue_identity,
)

CONTRACT_VERSION = "manual-independent-capture-v1"
CONFIG_SCHEMA_VERSION = "manual_independent_capture_config_v1"
TERMINAL_ARTIFACT_SCHEMA_VERSION = "manual_independent_capture_terminal_v1"
AUTHORITY_PROFILE = "manual_independent_capture_research_only_v1"
PHASE7_EXCLUSION_REASON = "manual_research_only_noncanonical"
DOWNSTREAM_ADMISSIBILITY = "research_only_noncanonical_phase7_excluded"

SAFETY_FIELDS = MappingProxyType(
    {
        "research_only": True,
        "canonical": False,
        "phase7_excluded": True,
        "phase7_eligible": False,
        "phase7_exclusion_reason": PHASE7_EXCLUSION_REASON,
    }
)

PROTECTED_PATH_KEYS = frozenset(
    {
        "autonomous_shared_lock",
        "canonical_database",
        "canonical_history_root",
        "live_odds_root",
        "forward_corpus_root",
        "collector_requests_root",
        "collector_state_root",
        "result_evidence_root",
        "services_root",
        "timers_root",
    }
)

AUTHORITY_MATRIX = MappingProxyType(
    {
        "schema_version": "manual_independent_capture_authority_v1",
        "allowed_reads": (
            "exact_selected_thedogs_race",
            "declared_prejump_source_bytes",
            "explicit_research_model_bytes",
            "manual_config_bytes",
        ),
        "allowed_writes": (
            "manual_runs_root_only",
            "manual_browser_profile_only",
            "manual_capture_lock_only",
        ),
        "forbidden_reads": tuple(sorted(PROTECTED_PATH_KEYS)),
        "forbidden_writes": tuple(sorted(PROTECTED_PATH_KEYS)),
        "lock_authority": "manual_capture_lock_only",
        "browser_authority": "manual_browser_profile_only",
        "downstream_admissibility": DOWNSTREAM_ADMISSIBILITY,
    }
)

TERMINAL_STATUS_BY_FAILURE_CODE = MappingProxyType(
    {
        "MANUAL_BUSY": "BLOCKED",
        "EXACT_RACE_INVALID": "BLOCKED",
        "INSUFFICIENT_PREJUMP_MARGIN": "BLOCKED",
        "SOURCE_TIMEOUT": "FAILED",
        "SOURCE_MALFORMED": "FAILED",
        "IDENTITY_MISMATCH": "FAILED",
        "RUNNER_SET_MISMATCH": "FAILED",
        "ODDS_INVALID": "FAILED",
        "CANCELLED": "CANCELLED",
        "TIMED_OUT": "TIMED_OUT",
        "PROCESS_REAP_UNCONFIRMED": "FAILED",
        "FEATURE_BLOCKED": "BLOCKED",
        "SCORING_BLOCKED": "BLOCKED",
    }
)
TERMINAL_STATUSES = frozenset(
    {"CAPTURE_READY", *TERMINAL_STATUS_BY_FAILURE_CODE.values()}
)
CAPTURE_EVIDENCE_FAILURE_CODES = frozenset({"FEATURE_BLOCKED", "SCORING_BLOCKED"})
ZERO_SOURCE_ATTEMPT_CODES = frozenset(
    {"MANUAL_BUSY", "EXACT_RACE_INVALID", "INSUFFICIENT_PREJUMP_MARGIN"}
)
ONE_SOURCE_ATTEMPT_CODES = frozenset(
    {
        "SOURCE_TIMEOUT",
        "SOURCE_MALFORMED",
        "IDENTITY_MISMATCH",
        "RUNNER_SET_MISMATCH",
        "ODDS_INVALID",
        "FEATURE_BLOCKED",
        "SCORING_BLOCKED",
    }
)
VARIABLE_SOURCE_ATTEMPT_CODES = frozenset(
    {"CANCELLED", "TIMED_OUT", "PROCESS_REAP_UNCONFIRMED"}
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_OBJECT_RE = re.compile(r"[0-9a-f]{40}")
_FORBIDDEN_MEMBER_PARTS = frozenset(
    {
        "autonomous-shared-lock",
        "canonical",
        "canonical-db",
        "canonical-history",
        "collector-requests",
        "collector-state",
        "forward-corpus",
        "live-odds",
        "phase-7",
        "phase7",
        "result",
        "result-evidence",
        "results",
        "services",
        "timers",
    }
)
_MAX_CANONICAL_JSON_BYTES = 2 * 1024 * 1024


class ManualIndependentCaptureRejected(RuntimeError):
    """One stable fail-closed contract rejection."""

    def __init__(self, code: str, **details: Any) -> None:
        super().__init__(code)
        self.code = code
        self.details = details


def _reject(code: str, **details: Any) -> ManualIndependentCaptureRejected:
    return ManualIndependentCaptureRejected(code, **details)


def authority_matrix() -> dict[str, Any]:
    """Return a copy so callers cannot weaken the process-wide vocabulary."""

    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in AUTHORITY_MATRIX.items()
    }


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate:{key}")
        value[key] = item
    return value


def _reject_nonfinite(value: Any, field: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise _reject("NONFINITE_VALUE", field=field)
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, f"{field}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{field}[{index}]")


def parse_canonical_json(
    raw: bytes, *, max_bytes: int = _MAX_CANONICAL_JSON_BYTES
) -> Any:
    """Parse exact canonical JSON with duplicate and non-finite rejection."""

    if not isinstance(raw, bytes) or not raw or len(raw) > max_bytes:
        raise _reject("CANONICAL_JSON_INVALID", reason="size")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise _reject("CANONICAL_JSON_INVALID", reason="json") from exc
    _reject_nonfinite(value, "document")
    if canonical_bytes(value) != raw:
        raise _reject("CANONICAL_JSON_INVALID", reason="noncanonical")
    return value


def _exact(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise _reject("CONTRACT_FIELDS_INVALID", field=label)
    return value


def _timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise _reject("TIMESTAMP_INVALID", field=label)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise _reject("TIMESTAMP_INVALID", field=label) from exc
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() is None
        or parsed.isoformat() != value
    ):
        raise _reject("TIMESTAMP_INVALID", field=label)
    return parsed


def _optional_timestamp(value: Any, label: str) -> datetime | None:
    return None if value is None else _timestamp(value, label)


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise _reject("HASH_INVALID", field=label)
    return value


def _git_object(value: Any, label: str) -> str:
    if not isinstance(value, str) or _GIT_OBJECT_RE.fullmatch(value) is None:
        raise _reject("HASH_INVALID", field=label)
    return value


def _canonical_uuid(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise _reject("IDENTIFIER_INVALID", field=label)
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as exc:
        raise _reject("IDENTIFIER_INVALID", field=label) from exc
    if parsed.version != 4 or str(parsed) != value:
        raise _reject("IDENTIFIER_INVALID", field=label)
    return value


def _integer(value: Any, label: str, *, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise _reject("INTEGER_INVALID", field=label)
    return value


def _safety(value: Any, label: str) -> dict[str, Any]:
    row = _exact(value, set(SAFETY_FIELDS), label)
    if dict(row) != SAFETY_FIELDS:
        raise _reject("SAFETY_CLAIM_INVALID", field=label)
    return dict(row)


def _absolute_path(value: Any, label: str) -> tuple[Path, Path]:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise _reject("UNSAFE_PATH", field=label)
    path = Path(value)
    if (
        not path.is_absolute()
        or path == Path("/")
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise _reject("UNSAFE_PATH", field=label)
    return path, Path(os.path.realpath(path))


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _validate_path_contract(
    value: Any, forbidden_paths: Mapping[str, str]
) -> dict[str, str]:
    paths = _exact(
        value,
        {
            "operations_root",
            "manual_root",
            "browser_profile",
            "runs_root",
            "manual_lock",
        },
        "config.paths",
    )
    if (
        not isinstance(forbidden_paths, Mapping)
        or set(forbidden_paths) != PROTECTED_PATH_KEYS
    ):
        raise _reject("FORBIDDEN_PATH_SET_INVALID")

    checked = {
        name: _absolute_path(paths[name], f"config.paths.{name}") for name in paths
    }
    operations_root = checked["operations_root"][0]
    manual_root = operations_root / CONTRACT_VERSION
    expected = {
        "manual_root": manual_root,
        "browser_profile": manual_root / "browser-profile",
        "runs_root": manual_root / "runs",
        "manual_lock": manual_root / "manual-capture.lock",
    }
    if any(checked[name][0] != path for name, path in expected.items()):
        raise _reject("MANUAL_PATH_LAYOUT_INVALID")

    protected = {
        name: _absolute_path(path, f"forbidden_paths.{name}")
        for name, path in forbidden_paths.items()
    }
    for manual_name, (manual_lexical, manual_resolved) in checked.items():
        for protected_name, (
            protected_lexical,
            protected_resolved,
        ) in protected.items():
            if _paths_overlap(manual_lexical, protected_lexical) or _paths_overlap(
                manual_resolved, protected_resolved
            ):
                raise _reject(
                    "PATH_AUTHORITY_CONFLICT",
                    manual=manual_name,
                    protected=protected_name,
                )
    return {name: str(path) for name, (path, _) in checked.items()}


def validate_config(
    value: Any, *, forbidden_paths: Mapping[str, str]
) -> dict[str, Any]:
    """Validate configuration plus external protected-root authority."""

    config = _exact(
        value,
        {
            "schema_version",
            "contract_version",
            "safety",
            "authority_profile",
            "paths",
            "timing",
            "attempt_policy",
        },
        "config",
    )
    if config["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise _reject("SCHEMA_VERSION_UNSUPPORTED", field="config.schema_version")
    if config["contract_version"] != CONTRACT_VERSION:
        raise _reject("CONTRACT_VERSION_UNSUPPORTED")
    _safety(config["safety"], "config.safety")
    if config["authority_profile"] != AUTHORITY_PROFILE:
        raise _reject("AUTHORITY_PROFILE_INVALID")
    _validate_path_contract(config["paths"], forbidden_paths)

    timing = _exact(
        config["timing"],
        {
            "minimum_prejump_margin_seconds",
            "hard_timeout_seconds",
            "cancellation_grace_seconds",
        },
        "config.timing",
    )
    _integer(
        timing["minimum_prejump_margin_seconds"],
        "config.timing.minimum_prejump_margin_seconds",
        minimum=1,
        maximum=7200,
    )
    _integer(
        timing["hard_timeout_seconds"],
        "config.timing.hard_timeout_seconds",
        minimum=1,
        maximum=900,
    )
    _integer(
        timing["cancellation_grace_seconds"],
        "config.timing.cancellation_grace_seconds",
        minimum=1,
        maximum=120,
    )

    policy = _exact(
        config["attempt_policy"],
        {
            "max_concurrent_manual_runs",
            "max_capture_attempts",
            "retries_allowed",
            "replay_allowed",
        },
        "config.attempt_policy",
    )
    if dict(policy) != {
        "max_concurrent_manual_runs": 1,
        "max_capture_attempts": 1,
        "retries_allowed": False,
        "replay_allowed": False,
    }:
        raise _reject("ATTEMPT_AUTHORITY_INVALID")
    return deepcopy(dict(config))


def _race(value: Any, label: str) -> dict[str, Any]:
    race = _exact(
        value,
        {
            "url",
            "race_id",
            "race_date",
            "venue",
            "venue_slug",
            "race_number",
            "scheduled_start",
        },
        label,
    )
    if not all(
        isinstance(race[name], str) and race[name] and race[name] == race[name].strip()
        for name in ("url", "race_id", "race_date", "venue", "venue_slug")
    ):
        raise _reject("RACE_IDENTITY_INVALID")
    identity = canonical_thedogs_race_identity(race["url"])
    if identity is None or identity["canonical_url"] != race["url"]:
        raise _reject("EXACT_RACE_INVALID")
    if isinstance(race["race_number"], bool) or not isinstance(
        race["race_number"], int
    ):
        raise _reject("RACE_IDENTITY_INVALID")
    venue = canonical_thedogs_venue_identity(race["venue"])
    url_venue = canonical_thedogs_venue_identity(identity["venue_slug"])
    projection = {
        "race_number": race["race_number"],
        "venue": race["venue"],
        "race_date": race["race_date"],
        "url": race["url"],
    }
    if (
        identity["race_date"] != race["race_date"]
        or identity["race_number"] != race["race_number"]
        or identity["venue_slug"] != race["venue_slug"]
        or venue is None
        or venue != url_venue
        or race["venue"] != url_venue
        or race["race_id"] != stable_race_id(projection)
        or race["race_id"] not in stable_race_id_variants(projection)
    ):
        raise _reject("RACE_IDENTITY_DISAGREEMENT")
    scheduled_start = _timestamp(race["scheduled_start"], f"{label}.scheduled_start")
    if scheduled_start.date().isoformat() != race["race_date"]:
        raise _reject("RACE_IDENTITY_DISAGREEMENT", field=f"{label}.scheduled_start")
    return dict(race)


def _request(value: Any, *, exact_race_invalid: bool) -> dict[str, Any]:
    request = _exact(
        value,
        {
            "request_id",
            "requested_at",
            "requested_race_url",
            "selected_race",
            "minimum_prejump_margin_seconds",
            "attempt_authority",
            "manual_concurrency",
            "safety",
        },
        "artifact.request",
    )
    _canonical_uuid(request["request_id"], "artifact.request.request_id")
    _timestamp(request["requested_at"], "artifact.request.requested_at")
    if (
        not isinstance(request["requested_race_url"], str)
        or not request["requested_race_url"]
        or len(request["requested_race_url"]) > 2048
        or request["requested_race_url"] != request["requested_race_url"].strip()
        or any(
            ord(character) < 32 or ord(character) == 127
            for character in request["requested_race_url"]
        )
    ):
        raise _reject("EXACT_RACE_INVALID")
    if exact_race_invalid:
        if request["selected_race"] is not None:
            raise _reject("RACE_IDENTITY_DISAGREEMENT")
    else:
        race = _race(request["selected_race"], "artifact.request.selected_race")
        if request["requested_race_url"] != race["url"]:
            raise _reject("RACE_IDENTITY_DISAGREEMENT")
    _integer(
        request["minimum_prejump_margin_seconds"],
        "artifact.request.minimum_prejump_margin_seconds",
        minimum=1,
        maximum=7200,
    )
    if request["attempt_authority"] != "one_attempt":
        raise _reject("ATTEMPT_AUTHORITY_INVALID")
    if request["manual_concurrency"] != "one_manual_run":
        raise _reject("ATTEMPT_AUTHORITY_INVALID")
    _safety(request["safety"], "artifact.request.safety")
    return deepcopy(dict(request))


def _relative_member_path(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 512
        or value.startswith("/")
        or "\\" in value
        or "\x00" in value
    ):
        raise _reject("UNSAFE_ARTIFACT_PATH", field=label)
    path = Path(value)
    if path.as_posix() != value or any(part in {"", ".", ".."} for part in path.parts):
        raise _reject("UNSAFE_ARTIFACT_PATH", field=label)
    normalized_parts = {part.lower().replace("_", "-") for part in path.parts}
    if normalized_parts & _FORBIDDEN_MEMBER_PARTS:
        raise _reject("FORBIDDEN_ARTIFACT_LOCATOR", field=label)
    return value


def _source_files(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > 32:
        raise _reject(
            "CONTRACT_FIELDS_INVALID", field="artifact.provenance.source_files"
        )
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        row = _exact(
            item,
            {
                "path",
                "content_class",
                "outcome_scope",
                "source_timestamp",
                "bytes",
                "sha256",
            },
            f"artifact.provenance.source_files[{index}]",
        )
        if row["content_class"] not in {
            "prejump_race_source",
            "prejump_form",
            "prejump_sidecar",
        }:
            raise _reject("SOURCE_CLASS_INVALID", field=f"source_files[{index}]")
        if row["outcome_scope"] != "target_same_future_outcomes_excluded":
            raise _reject("OUTCOME_SCOPE_INVALID", field=f"source_files[{index}]")
        path = _relative_member_path(row["path"], f"source_files[{index}].path")
        _timestamp(row["source_timestamp"], f"source_files[{index}].source_timestamp")
        _integer(
            row["bytes"], f"source_files[{index}].bytes", minimum=1, maximum=2**31 - 1
        )
        _sha256(row["sha256"], f"source_files[{index}].sha256")
        rows.append({**row, "path": path})
    if [row["path"] for row in rows] != sorted(row["path"] for row in rows):
        raise _reject("ARTIFACT_MEMBERS_INVALID", reason="source_order")
    if len({row["path"] for row in rows}) != len(rows):
        raise _reject("ARTIFACT_MEMBERS_INVALID", reason="duplicate_source")
    return rows


def _artifact_hashes(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise _reject(
            "CONTRACT_FIELDS_INVALID", field="artifact.provenance.artifact_hashes"
        )
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        row = _exact(
            item,
            {"role", "path", "bytes", "sha256"},
            f"artifact.provenance.artifact_hashes[{index}]",
        )
        if row["role"] not in {"config", "model", "capture"}:
            raise _reject(
                "ARTIFACT_ROLE_INVALID", field=f"artifact_hashes[{index}].role"
            )
        path = _relative_member_path(row["path"], f"artifact_hashes[{index}].path")
        _integer(
            row["bytes"],
            f"artifact_hashes[{index}].bytes",
            minimum=1,
            maximum=2**31 - 1,
        )
        _sha256(row["sha256"], f"artifact_hashes[{index}].sha256")
        rows.append({**row, "path": path})
    if [row["path"] for row in rows] != sorted(row["path"] for row in rows):
        raise _reject("ARTIFACT_MEMBERS_INVALID", reason="artifact_order")
    if len({row["path"] for row in rows}) != len(rows) or len(
        {row["role"] for row in rows}
    ) != len(rows):
        raise _reject("ARTIFACT_MEMBERS_INVALID", reason="duplicate_artifact")
    return rows


def _capture(value: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    capture = _exact(value, {"runner_set"}, "artifact.capture")
    raw_runners = capture["runner_set"]
    if not isinstance(raw_runners, list) or len(raw_runners) > 16:
        raise _reject("RUNNER_SET_INVALID")
    if not raw_runners:
        return {"runner_set": []}, []
    runners: list[dict[str, Any]] = []
    canonical: list[dict[str, Any]] = []
    for index, item in enumerate(raw_runners):
        row = _exact(
            item,
            {
                "box_number",
                "display_name",
                "identity",
                "source_native_runner_id",
                "decimal_odds",
            },
            f"artifact.capture.runner_set[{index}]",
        )
        odds = row["decimal_odds"]
        if (
            isinstance(odds, bool)
            or not isinstance(odds, (int, float))
            or not math.isfinite(float(odds))
            or odds <= 0
        ):
            raise _reject("ODDS_INVALID", field=f"runner_set[{index}].decimal_odds")
        identity_row = {key: row[key] for key in row if key != "decimal_odds"}
        canonical.append(identity_row)
        runners.append(dict(row))
    try:
        canonical_runner_set(canonical, "artifact.capture.runner_set")
    except PredictionBlocked as exc:
        raise _reject("RUNNER_SET_INVALID", source_code=exc.code) from exc
    return {"runner_set": runners}, canonical


def _validate_terminal_pair(status: Any, failure_code: Any) -> str | None:
    if status not in TERMINAL_STATUSES:
        raise _reject("TERMINAL_STATUS_INVALID")
    if status == "CAPTURE_READY":
        if failure_code is not None:
            raise _reject("TERMINAL_FAILURE_CONFLICT")
        return None
    if (
        not isinstance(failure_code, str)
        or TERMINAL_STATUS_BY_FAILURE_CODE.get(failure_code) != status
    ):
        raise _reject("TERMINAL_FAILURE_CONFLICT")
    return failure_code


def _validate_members(
    *,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    source_files: Sequence[Mapping[str, Any]],
    artifact_hashes: Sequence[Mapping[str, Any]],
    member_bytes: Mapping[str, bytes],
    capture_evidence: bool,
    capture: Mapping[str, Any],
) -> None:
    if not isinstance(member_bytes, Mapping) or any(
        not isinstance(name, str) or not isinstance(raw, bytes)
        for name, raw in member_bytes.items()
    ):
        raise _reject("ARTIFACT_MEMBERS_INVALID", reason="member_bytes")
    rows = [*source_files, *artifact_hashes]
    paths = [row["path"] for row in rows]
    if len(paths) != len(set(paths)) or set(member_bytes) != set(paths):
        raise _reject("ARTIFACT_MEMBERS_INVALID", reason="membership")
    for row in rows:
        raw = member_bytes[row["path"]]
        if len(raw) != row["bytes"] or sha256_bytes(raw) != row["sha256"]:
            raise _reject("HASH_DRIFT", path=row["path"])

    by_role = {row["role"]: row for row in artifact_hashes}
    expected_roles = {"config", "model"}
    if capture_evidence:
        expected_roles.add("capture")
    if set(by_role) != expected_roles:
        raise _reject("ARTIFACT_ROLE_INVALID", reason="required_roles")
    config_raw = member_bytes[by_role["config"]["path"]]
    if (
        config_raw != canonical_bytes(config)
        or by_role["config"]["sha256"] != provenance["config_sha256"]
    ):
        raise _reject("CONFIG_HASH_DRIFT")
    if by_role["model"]["sha256"] != provenance["model_sha256"]:
        raise _reject("MODEL_HASH_DRIFT")
    if capture_evidence:
        capture_raw = member_bytes[by_role["capture"]["path"]]
        if capture_raw != canonical_bytes(capture):
            raise _reject("CAPTURE_HASH_DRIFT")


def validate_terminal_artifact(
    value: Any,
    *,
    config: Mapping[str, Any],
    forbidden_paths: Mapping[str, str],
    member_bytes: Mapping[str, bytes],
    expected_source_commit: str,
    expected_source_tree: str,
    expected_run_id: str | None = None,
    expected_request_id: str | None = None,
    expected_request_sha256: str | None = None,
    seen_run_ids: AbstractSet[str] = frozenset(),
    seen_request_ids: AbstractSet[str] = frozenset(),
    seen_request_sha256s: AbstractSet[str] = frozenset(),
) -> dict[str, Any]:
    """Validate one terminal record and all supplied byte/hash bindings."""

    validated_config = validate_config(config, forbidden_paths=forbidden_paths)
    artifact = _exact(
        value,
        {
            "schema_version",
            "contract_version",
            "run_id",
            "safety",
            "authority_profile",
            "request",
            "timing",
            "attempt",
            "terminal",
            "provenance",
            "capture",
            "closure",
        },
        "artifact",
    )
    if artifact["schema_version"] != TERMINAL_ARTIFACT_SCHEMA_VERSION:
        raise _reject("SCHEMA_VERSION_UNSUPPORTED", field="artifact.schema_version")
    if artifact["contract_version"] != CONTRACT_VERSION:
        raise _reject("CONTRACT_VERSION_UNSUPPORTED")
    run_id = _canonical_uuid(artifact["run_id"], "artifact.run_id")
    _safety(artifact["safety"], "artifact.safety")
    if artifact["authority_profile"] != AUTHORITY_PROFILE:
        raise _reject("AUTHORITY_PROFILE_INVALID")

    terminal = _exact(
        artifact["terminal"], {"status", "failure_code"}, "artifact.terminal"
    )
    failure_code = _validate_terminal_pair(terminal["status"], terminal["failure_code"])
    request = _request(
        artifact["request"], exact_race_invalid=failure_code == "EXACT_RACE_INVALID"
    )
    request_id = request["request_id"]
    request_sha256 = canonical_sha256(request)

    if expected_run_id is not None and run_id != expected_run_id:
        raise _reject("ARTIFACT_CONFLICT", field="run_id")
    if expected_request_id is not None and request_id != expected_request_id:
        raise _reject("ARTIFACT_CONFLICT", field="request_id")
    if (
        expected_request_sha256 is not None
        and request_sha256 != expected_request_sha256
    ):
        raise _reject("ARTIFACT_CONFLICT", field="request_sha256")
    if (
        run_id in seen_run_ids
        or request_id in seen_request_ids
        or request_sha256 in seen_request_sha256s
    ):
        raise _reject("REPLAYED_ARTIFACT")

    configured_margin = validated_config["timing"]["minimum_prejump_margin_seconds"]
    if request["minimum_prejump_margin_seconds"] != configured_margin:
        raise _reject(
            "CONFIG_REQUEST_DISAGREEMENT", field="minimum_prejump_margin_seconds"
        )

    timing = _exact(
        artifact["timing"],
        {
            "submitted_at",
            "readiness_checked_at",
            "deadline_at",
            "cleanup_deadline_at",
            "capture_timestamp",
            "readiness_prejump_margin_seconds",
            "capture_prejump_margin_seconds",
            "cancel_requested_at",
            "terminal_at",
        },
        "artifact.timing",
    )
    submitted = _timestamp(timing["submitted_at"], "artifact.timing.submitted_at")
    readiness = _timestamp(
        timing["readiness_checked_at"], "artifact.timing.readiness_checked_at"
    )
    deadline = _timestamp(timing["deadline_at"], "artifact.timing.deadline_at")
    cleanup_deadline = _optional_timestamp(
        timing["cleanup_deadline_at"], "artifact.timing.cleanup_deadline_at"
    )
    captured = _optional_timestamp(
        timing["capture_timestamp"], "artifact.timing.capture_timestamp"
    )
    cancelled = _optional_timestamp(
        timing["cancel_requested_at"], "artifact.timing.cancel_requested_at"
    )
    terminal_at = _timestamp(timing["terminal_at"], "artifact.timing.terminal_at")
    if (
        submitted.isoformat() != request["requested_at"]
        or not submitted <= readiness <= terminal_at
    ):
        raise _reject("TIMING_INVALID", reason="ordering")
    expected_deadline = readiness + timedelta(
        seconds=validated_config["timing"]["hard_timeout_seconds"]
    )
    if deadline != expected_deadline:
        raise _reject("TIMING_INVALID", reason="deadline")
    if failure_code == "TIMED_OUT":
        if terminal_at != deadline:
            raise _reject("TIMING_INVALID", reason="timeout_deadline")
    elif failure_code != "PROCESS_REAP_UNCONFIRMED" and terminal_at > deadline:
        raise _reject("LATE_ARTIFACT")
    if cancelled is not None and not readiness <= cancelled <= terminal_at:
        raise _reject("CANCELLATION_INVALID")
    if failure_code == "CANCELLED" and cancelled is None:
        raise _reject("CANCELLATION_INVALID")
    if cancelled is not None and failure_code not in {
        "CANCELLED",
        "TIMED_OUT",
        "PROCESS_REAP_UNCONFIRMED",
    }:
        raise _reject("CANCELLATION_INVALID")
    cancellation_grace = validated_config["timing"]["cancellation_grace_seconds"]
    if cancelled is not None:
        expected_cleanup_deadline = min(
            deadline, cancelled + timedelta(seconds=cancellation_grace)
        )
        if cleanup_deadline != expected_cleanup_deadline:
            raise _reject("CANCELLATION_INVALID", reason="cleanup_deadline")
        if (
            failure_code != "PROCESS_REAP_UNCONFIRMED"
            and terminal_at > cleanup_deadline
        ):
            raise _reject("CANCELLATION_INVALID", reason="late_cancel_terminal")
    elif failure_code in {"TIMED_OUT", "PROCESS_REAP_UNCONFIRMED"}:
        if cleanup_deadline != deadline:
            raise _reject("CANCELLATION_INVALID", reason="cleanup_deadline")
    elif cleanup_deadline is not None:
        raise _reject("CANCELLATION_INVALID", reason="unexpected_cleanup_deadline")
    if failure_code == "PROCESS_REAP_UNCONFIRMED" and (
        cleanup_deadline is None or terminal_at < cleanup_deadline
    ):
        raise _reject("CANCELLATION_INVALID", reason="early_reap_failure")

    selected_race = request["selected_race"]
    readiness_margin = timing["readiness_prejump_margin_seconds"]
    capture_margin = timing["capture_prejump_margin_seconds"]
    if selected_race is None:
        if (
            readiness_margin is not None
            or capture_margin is not None
            or captured is not None
        ):
            raise _reject("TIMING_INVALID", reason="invalid_race_timing")
    else:
        scheduled = _timestamp(
            selected_race["scheduled_start"], "selected_race.scheduled_start"
        )
        calculated_readiness_margin = (scheduled - readiness).total_seconds()
        if (
            isinstance(readiness_margin, bool)
            or not isinstance(readiness_margin, int)
            or calculated_readiness_margin != readiness_margin
        ):
            raise _reject("TIMING_INVALID", reason="readiness_margin")
        if failure_code == "INSUFFICIENT_PREJUMP_MARGIN":
            if readiness_margin >= configured_margin:
                raise _reject("TIMING_INVALID", reason="margin_not_insufficient")
        elif readiness_margin < configured_margin:
            raise _reject("INSUFFICIENT_PREJUMP_MARGIN")
        if captured is None:
            if capture_margin is not None:
                raise _reject("TIMING_INVALID", reason="capture_margin_without_capture")
        else:
            calculated_capture_margin = (scheduled - captured).total_seconds()
            if (
                isinstance(capture_margin, bool)
                or not isinstance(capture_margin, int)
                or calculated_capture_margin != capture_margin
                or capture_margin < configured_margin
                or not readiness <= captured <= terminal_at
            ):
                raise _reject("TIMING_INVALID", reason="capture_margin")

    attempt = _exact(
        artifact["attempt"],
        {"attempt_count", "source_attempt_count"},
        "artifact.attempt",
    )
    if attempt["attempt_count"] != 1 or isinstance(attempt["attempt_count"], bool):
        raise _reject("ATTEMPT_AUTHORITY_INVALID")
    source_attempts = attempt["source_attempt_count"]
    if source_attempts not in {0, 1} or isinstance(source_attempts, bool):
        raise _reject("ATTEMPT_AUTHORITY_INVALID")
    if failure_code in ZERO_SOURCE_ATTEMPT_CODES and source_attempts != 0:
        raise _reject("ATTEMPT_AUTHORITY_INVALID")
    if failure_code in ONE_SOURCE_ATTEMPT_CODES and source_attempts != 1:
        raise _reject("ATTEMPT_AUTHORITY_INVALID")
    if failure_code is None and source_attempts != 1:
        raise _reject("ATTEMPT_AUTHORITY_INVALID")
    if failure_code not in {
        None,
        *ZERO_SOURCE_ATTEMPT_CODES,
        *ONE_SOURCE_ATTEMPT_CODES,
        *VARIABLE_SOURCE_ATTEMPT_CODES,
    }:
        raise _reject("TERMINAL_FAILURE_CONFLICT")

    provenance = _exact(
        artifact["provenance"],
        {
            "source_commit",
            "source_tree",
            "config_sha256",
            "model_sha256",
            "request_sha256",
            "race_identity_sha256",
            "runner_set_sha256",
            "odds_sha256",
            "source_files",
            "artifact_hashes",
        },
        "artifact.provenance",
    )
    source_commit = _git_object(
        provenance["source_commit"], "artifact.provenance.source_commit"
    )
    source_tree = _git_object(
        provenance["source_tree"], "artifact.provenance.source_tree"
    )
    trusted_commit = _git_object(expected_source_commit, "expected_source_commit")
    trusted_tree = _git_object(expected_source_tree, "expected_source_tree")
    if source_commit != trusted_commit or source_tree != trusted_tree:
        raise _reject("SOURCE_PROVENANCE_MISMATCH")
    for name in ("config_sha256", "model_sha256", "request_sha256"):
        _sha256(provenance[name], f"artifact.provenance.{name}")
    if provenance["config_sha256"] != canonical_sha256(validated_config):
        raise _reject("CONFIG_HASH_DRIFT")
    if provenance["request_sha256"] != request_sha256:
        raise _reject("REQUEST_HASH_DRIFT")

    source_files = _source_files(provenance["source_files"])
    artifact_hashes = _artifact_hashes(provenance["artifact_hashes"])
    capture, canonical_runners = _capture(artifact["capture"])
    capture_evidence = (
        failure_code is None or failure_code in CAPTURE_EVIDENCE_FAILURE_CODES
    )
    if capture_evidence:
        if (
            selected_race is None
            or captured is None
            or not source_files
            or not canonical_runners
        ):
            raise _reject("CAPTURE_EVIDENCE_INCOMPLETE")
        expected_race_sha = canonical_sha256(selected_race)
        expected_runner_sha = sealed_runner_set_sha256(selected_race, canonical_runners)
        expected_odds_sha = canonical_sha256(
            [
                {"box_number": row["box_number"], "decimal_odds": row["decimal_odds"]}
                for row in capture["runner_set"]
            ]
        )
        for name, expected in (
            ("race_identity_sha256", expected_race_sha),
            ("runner_set_sha256", expected_runner_sha),
            ("odds_sha256", expected_odds_sha),
        ):
            _sha256(provenance[name], f"artifact.provenance.{name}")
            if provenance[name] != expected:
                raise _reject("IDENTITY_HASH_DRIFT", field=name)
        for row in source_files:
            source_time = _timestamp(
                row["source_timestamp"], "source_files.source_timestamp"
            )
            if source_time > captured or source_time >= _timestamp(
                selected_race["scheduled_start"], "selected_race.scheduled_start"
            ):
                raise _reject("SOURCE_TIMING_INVALID", path=row["path"])
    else:
        if (
            captured is not None
            or canonical_runners
            or any(
                provenance[name] is not None
                for name in ("runner_set_sha256", "odds_sha256")
            )
        ):
            raise _reject("CAPTURE_EVIDENCE_FORBIDDEN")
        if selected_race is None:
            if (
                failure_code != "EXACT_RACE_INVALID"
                or provenance["race_identity_sha256"] is not None
            ):
                raise _reject("CAPTURE_EVIDENCE_FORBIDDEN")
        else:
            _sha256(
                provenance["race_identity_sha256"],
                "artifact.provenance.race_identity_sha256",
            )
            if provenance["race_identity_sha256"] != canonical_sha256(selected_race):
                raise _reject("IDENTITY_HASH_DRIFT", field="race_identity_sha256")
            scheduled = _timestamp(
                selected_race["scheduled_start"], "selected_race.scheduled_start"
            )
            for row in source_files:
                source_time = _timestamp(
                    row["source_timestamp"], "source_files.source_timestamp"
                )
                if source_time > terminal_at or source_time >= scheduled:
                    raise _reject("SOURCE_TIMING_INVALID", path=row["path"])
        if source_attempts == 0 and source_files:
            raise _reject("CAPTURE_EVIDENCE_FORBIDDEN", reason="source_without_attempt")

    _validate_members(
        config=validated_config,
        provenance=provenance,
        source_files=source_files,
        artifact_hashes=artifact_hashes,
        member_bytes=member_bytes,
        capture_evidence=capture_evidence,
        capture=capture,
    )

    closure = _exact(
        artifact["closure"],
        {
            "bundle_closed",
            "closed_at",
            "phase7_accessed",
            "outcome_accessed",
            "canonical_write_claimed",
            "downstream_admissibility",
        },
        "artifact.closure",
    )
    if (
        closure["bundle_closed"] is not True
        or _timestamp(closure["closed_at"], "artifact.closure.closed_at") != terminal_at
        or closure["phase7_accessed"] is not False
        or closure["outcome_accessed"] is not False
        or closure["canonical_write_claimed"] is not False
        or closure["downstream_admissibility"] != DOWNSTREAM_ADMISSIBILITY
    ):
        raise _reject("DOWNSTREAM_AUTHORITY_INVALID")
    return deepcopy(dict(artifact))


__all__ = [
    "AUTHORITY_MATRIX",
    "AUTHORITY_PROFILE",
    "CONFIG_SCHEMA_VERSION",
    "CONTRACT_VERSION",
    "DOWNSTREAM_ADMISSIBILITY",
    "PHASE7_EXCLUSION_REASON",
    "PROTECTED_PATH_KEYS",
    "SAFETY_FIELDS",
    "TERMINAL_ARTIFACT_SCHEMA_VERSION",
    "TERMINAL_STATUS_BY_FAILURE_CODE",
    "ManualIndependentCaptureRejected",
    "authority_matrix",
    "canonical_bytes",
    "canonical_sha256",
    "parse_canonical_json",
    "validate_config",
    "validate_terminal_artifact",
]
