"""Strict, versioned Level-1 read API for the Operator UI.

Providers are server-owned adapters registered explicitly after application
construction.  They return immutable foundation evidence, never status
mappings.  This module validates the complete envelope and the finite
resource-specific read model before the security layer audits the exact bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any

from flask import Flask, request

from .foundation import EvidenceEnvelope, EvidenceStatus, POLICIES
from .security import PreparedDisclosure


API_VERSION = "v1"
API_SCHEMA = "operator_ui_level_1_api_v1"
API_PREFIX = "/operator-ui/api/v1"
_REGISTRY_KEY = "operator_ui_level_1_api_providers"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_ZONE = re.compile(r"(?:UTC(?:[+-][0-9]{2}:[0-9]{2})?|[A-Za-z]+(?:[/_-][A-Za-z0-9+_-]+)+)\Z")
_URL = re.compile(r"https://(?:www\.)?thedogs\.com\.au/[^\s]{1,480}\Z")
_MAX_ITEMS = 100
_MAX_TEXT_BYTES = 512
_CLOCK_SKEW_SECONDS = 1e-6
_NON_HEALTHY_DISCLOSURE_RESOURCES = frozenset({"collector", "system"})
_NON_HEALTHY_DISCLOSURE_STATUSES = frozenset(
    {
        EvidenceStatus.STALE,
        EvidenceStatus.UNAVAILABLE_DATA_MISSING,
        EvidenceStatus.DIVERGENT,
    }
)
_FINITE_EMPTY_INVALID_RESOURCES = frozenset({"upcoming_races", "race_detail", "collector"})

_RESOURCE_POLICIES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "overview": frozenset({"P-OPS-5"}),
        "upcoming_races": frozenset({"P-UPCOMING-300-PREJUMP"}),
        "race_detail": frozenset({"P-UPCOMING-300-PREJUMP"}),
        "recent_predictions": frozenset({"P-BUNDLE-LIST-60"}),
        "prediction_detail": frozenset(
            {"P-IMMUTABLE-HISTORICAL", "P-JOB-5-DEADLINE"}
        ),
        "collector": frozenset(
            {
                "P-COLLECTOR-FULL-DYNAMIC",
                "P-COLLECTOR-ODDS-DYNAMIC",
                "P-COLLECTOR-AGGREGATE",
            }
        ),
        "corpus": frozenset({"P-REPORT-24H"}),
        "models": frozenset(
            {"P-CATALOG-60", "P-REPORT-24H", "P-IMMUTABLE-HISTORICAL"}
        ),
        "system": frozenset({"P-DEPLOY-60"}),
        "audit": frozenset({"P-OPS-5"}),
    }
)

_RESOURCE_DEFAULT_POLICY: Mapping[str, str] = MappingProxyType(
    {
        "overview": "P-OPS-5",
        "upcoming_races": "P-UPCOMING-300-PREJUMP",
        "race_detail": "P-UPCOMING-300-PREJUMP",
        "recent_predictions": "P-BUNDLE-LIST-60",
        "prediction_detail": "P-IMMUTABLE-HISTORICAL",
        "collector": "P-COLLECTOR-AGGREGATE",
        "corpus": "P-REPORT-24H",
        "models": "P-CATALOG-60",
        "system": "P-DEPLOY-60",
        "audit": "P-OPS-5",
    }
)

_OVERVIEW_STATUSES = frozenset(
    {
        "AVAILABLE/FRESH",
        "STALE",
        "UNAVAILABLE/DATA_MISSING",
        "INVALID/INTEGRITY_FAILED",
        "DIVERGENT",
    }
)
_RACE_STATUSES = frozenset({"SCHEDULED", "SCRATCHED", "ACTIVE"})
_PREDICTION_LIFECYCLES = frozenset(
    {
        "SUBMITTED",
        "VALIDATED",
        "WAITING_FOR_CLAIM",
        "CLAIMED",
        "ATTEMPT_STARTED",
        "RESPONSE_RECORDED",
        "RECEIPT_VERIFIED",
        "CONSUMED",
        "SCORING",
        "PREDICTION_READY",
        "FAILED",
        "REJECTED",
        "EXPIRED",
        "TIMED_OUT",
    }
)
_COLLECTOR_LANE_STATUSES = frozenset(
    {
        "ACTIVE",
        "RECEIPT_READY",
        "REQUEST_EXPIRED",
        "RACE_NOT_FOUND",
        "CAPTURE_WINDOW_CLOSED",
        "IDENTITY_MISMATCH",
        "CAPTURE_FAILED",
        "STALE",
        "DATA_MISSING",
        "INTEGRITY_FAILED",
        "DIVERGENT",
    }
)
_COLLECTOR_TERMINAL_STATUSES = frozenset(
    {
        "RECEIPT_READY",
        "REQUEST_EXPIRED",
        "RACE_NOT_FOUND",
        "CAPTURE_WINDOW_CLOSED",
        "IDENTITY_MISMATCH",
        "CAPTURE_FAILED",
    }
)
_CORPUS_STATUSES = frozenset(
    {"ADMISSIBLE", "NOT_ADMISSIBLE", "STALE", "DATA_MISSING", "DIVERGENT"}
)
_MODEL_ROLES = frozenset({"BASELINE", "LATEST_RESEARCH", "CHALLENGER"})
_EVALUATION_STATUSES = frozenset(
    {"REPORTED", "UNAVAILABLE", "STALE", "INVALID", "DIVERGENT"}
)
_COMPONENT_STATUSES = frozenset(
    {"HEALTHY", "DEGRADED", "STALE", "DATA_MISSING", "INVALID", "DIVERGENT"}
)
_AUDIT_CLASSIFICATIONS = frozenset(
    {
        "AVAILABLE/FRESH",
        "STALE",
        "UNAVAILABLE/DATA_MISSING",
        "INVALID/INTEGRITY_FAILED",
        "DIVERGENT",
        "NON_OPERATIONAL/AUTHENTICATION_REQUIRED",
        "NON_OPERATIONAL/AUTHORIZATION_DENIED",
        "NON_OPERATIONAL/PROVIDER_ERROR",
        "NON_OPERATIONAL/AUDIT_UNAVAILABLE",
    }
)


@dataclass(frozen=True)
class APIObservation:
    """One provider result: foundation evidence plus a finite read model."""

    evidence: EvidenceEnvelope
    data: Mapping[str, Any]


def _text(value: Any, *, identifier: bool = False) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > _MAX_TEXT_BYTES
        or any(character < " " or character == "\x7f" for character in value)
    ):
        raise ValueError("value must be a bounded printable string")
    if identifier and _IDENTIFIER.fullmatch(value) is None:
        raise ValueError("value must be a bounded lexical identifier")
    return value


def _id(value: Any) -> str:
    return _text(value, identifier=True)


def _hash(value: Any) -> str:
    value = _text(value)
    if _HASH.fullmatch(value) is None:
        raise ValueError("value must be a lowercase SHA-256")
    return value


def _git_oid(value: Any) -> str:
    value = _text(value)
    if _GIT_OID.fullmatch(value) is None:
        raise ValueError("value must be a lowercase Git object identity")
    return value


def _utc(value: Any, *, field: str = "timestamp") -> tuple[str, datetime]:
    text = _text(value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return text, parsed.astimezone(timezone.utc)


def _finite_number(value: Any, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("value must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError("value must be finite and in range")
    return result


def _exact(value: Any, fields: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("object has unknown or missing fields")
    return value


def _items(value: Any) -> list[Any]:
    if not isinstance(value, list) or len(value) > _MAX_ITEMS:
        raise ValueError("collection must be a bounded list")
    return value


def _require_unique(values: list[dict[str, Any]], field: str, label: str) -> None:
    identities = [item[field] for item in values]
    if len(identities) != len(set(identities)):
        raise ValueError(f"{label} must be unique")


def _named_hashes(value: Any, *, required: bool = True) -> dict[str, str]:
    if (
        not isinstance(value, dict)
        or len(value) > _MAX_ITEMS
        or (required and not value)
    ):
        raise ValueError("named hashes must be a finite object")
    return {_id(name): _hash(digest) for name, digest in value.items()}


def _identity(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or not value or len(value) > _MAX_ITEMS:
        raise ValueError("evidence identity must be a finite non-empty object")
    return {_id(name): _text(item) for name, item in value.items()}


def _runner(value: Any, *, sealed: bool = False) -> dict[str, Any]:
    raw = _exact(
        value,
        frozenset(
            {"runner_id", "source_runner_id", "box", "name", "scratch_state"}
        ),
    )
    box = raw["box"]
    if isinstance(box, bool) or not isinstance(box, int) or not 1 <= box <= 20:
        raise ValueError("runner box is invalid")
    scratch = _text(raw["scratch_state"])
    if scratch not in _RACE_STATUSES:
        raise ValueError("unknown scratch state")
    return {
        "box": box,
        "name": _text(raw["name"]),
        "runner_id": _id(raw["runner_id"]),
        "scratch_state": scratch,
        "source_runner_id": (
            None if sealed and raw["source_runner_id"] is None
            else _id(raw["source_runner_id"])
        ),
    }


def _race(value: Any, *, route_id: str | None) -> dict[str, Any]:
    legacy_fields = frozenset({
                "race_id",
                "source_race_id",
                "source_url",
                "racing_date",
                "venue",
                "meeting_slug",
                "race_number",
                "jump_utc",
                "source_zone",
                "distance_metres",
                "grade",
                "runners",
                "runner_set_sha256",
    })
    sealed_fields = legacy_fields | {"route_id"}
    if not isinstance(value, dict) or set(value) not in {legacy_fields, sealed_fields}:
        raise ValueError("object has unknown or missing fields")
    raw = value
    sealed = "route_id" in raw
    race_id = _text(raw["race_id"]) if sealed else _id(raw["race_id"])
    resource_id = _id(raw["route_id"]) if sealed else race_id
    if route_id is not None and resource_id != route_id:
        raise ValueError("race identity does not match route")
    url = _text(raw["source_url"])
    if _URL.fullmatch(url) is None:
        raise ValueError("race URL must be a canonical TheDogs URL")
    date = _text(raw["racing_date"])
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("racing date is invalid") from exc
    race_number = raw["race_number"]
    if isinstance(race_number, bool) or not isinstance(race_number, int) or not 1 <= race_number <= 99:
        raise ValueError("race number is invalid")
    _, jump = _utc(raw["jump_utc"], field="jump_utc")
    zone = _text(raw["source_zone"])
    if _ZONE.fullmatch(zone) is None:
        raise ValueError("source zone is invalid")
    distance = raw["distance_metres"]
    if distance is not None and (
        isinstance(distance, bool) or not isinstance(distance, int) or distance <= 0
    ):
        raise ValueError("distance must be positive or null")
    grade = raw["grade"]
    if grade is not None:
        grade = _text(grade)
    runners = [_runner(item, sealed=sealed) for item in _items(raw["runners"])]
    if not runners or len({item["runner_id"] for item in runners}) != len(runners):
        raise ValueError("runner identities must be non-empty and unique")
    native_ids = [item["source_runner_id"] for item in runners if item["source_runner_id"] is not None]
    if len(native_ids) != len(set(native_ids)):
        raise ValueError("source runner identities must be unique")
    if len({item["box"] for item in runners}) != len(runners):
        raise ValueError("runner boxes must be unique")
    return {
        "distance_metres": distance,
        "grade": grade,
        "jump_utc": raw["jump_utc"],
        "meeting_slug": None if sealed and raw["meeting_slug"] is None else _id(raw["meeting_slug"]),
        "race_id": race_id,
        **({"route_id": resource_id} if sealed else {}),
        "race_number": race_number,
        "racing_date": date,
        "runner_set_sha256": _hash(raw["runner_set_sha256"]),
        "runners": runners,
        "source_race_id": _text(raw["source_race_id"]) if sealed else _id(raw["source_race_id"]),
        "source_url": url,
        "source_zone": zone,
        "venue": _text(raw["venue"]),
        "_jump": jump,
    }


def _prediction(value: Any, *, route_id: str | None) -> dict[str, Any]:
    legacy_fields = frozenset({
                "prediction_id",
                "job_id",
                "race_id",
                "model_id",
                "model_sha256",
                "config_id",
                "config_sha256",
                "lifecycle_status",
                "probabilities",
                "bundle_sha256",
                "evidence_identities",
    })
    sealed_fields = (legacy_fields - {"lifecycle_status"}) | {
        "terminal_status", "blocker_stage", "blocker_code", "evidence_names"
    }
    if not isinstance(value, dict) or set(value) not in {legacy_fields, sealed_fields}:
        raise ValueError("object has unknown or missing fields")
    raw = value
    sealed = "terminal_status" in raw
    prediction_id = _id(raw["prediction_id"])
    if route_id is not None and prediction_id != route_id:
        raise ValueError("prediction identity does not match route")
    lifecycle = _text(raw["terminal_status"] if sealed else raw["lifecycle_status"])
    if lifecycle not in ({"PREDICTION_READY", "PREDICTION_BLOCKED"} if sealed else _PREDICTION_LIFECYCLES):
        raise ValueError("unknown prediction lifecycle")
    blocker_stage = raw.get("blocker_stage")
    blocker_code = raw.get("blocker_code")
    if sealed and ((lifecycle == "PREDICTION_READY" and (blocker_stage is not None or blocker_code is not None)) or (lifecycle == "PREDICTION_BLOCKED" and (blocker_stage not in {"PROTOCOL", "VALIDATION", "SCORING"} or blocker_code is None))):
        raise ValueError("terminal blocker identity is invalid")
    probabilities: list[dict[str, Any]] | None = None
    if raw["probabilities"] is not None:
        if lifecycle != "PREDICTION_READY":
            raise ValueError("probabilities require PREDICTION_READY")
        probabilities = []
        total = 0.0
        for item in _items(raw["probabilities"]):
            part = _exact(item, frozenset({"runner_id", "probability"}))
            probability = _finite_number(part["probability"], minimum=0)
            if probability > 1:
                raise ValueError("probability exceeds one")
            total += probability
            probabilities.append(
                {"probability": probability, "runner_id": _id(part["runner_id"])}
            )
        if not probabilities or not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError("probabilities must be non-empty and sum to one")
        if len({item["runner_id"] for item in probabilities}) != len(probabilities):
            raise ValueError("probability runner identities must be unique")
    evidence_names = None
    if sealed:
        evidence_names = []
        for item in _items(raw["evidence_names"]):
            name = _text(item)
            if name.startswith("/") or "\\" in name or any(part in {"", ".", ".."} for part in name.split("/")):
                raise ValueError("evidence name must be fixed-root-relative")
            evidence_names.append(name)
        if not evidence_names or len(evidence_names) != len(set(evidence_names)):
            raise ValueError("evidence names must be non-empty and unique")
    return {
        "bundle_sha256": _hash(raw["bundle_sha256"]),
        "config_id": _id(raw["config_id"]),
        "config_sha256": _hash(raw["config_sha256"]),
        "evidence_identities": _identity(raw["evidence_identities"]),
        "job_id": None if sealed and raw["job_id"] is None else _id(raw["job_id"]),
        **({"terminal_status": lifecycle, "blocker_stage": blocker_stage,
            "blocker_code": blocker_code, "evidence_names": evidence_names}
           if sealed else {"lifecycle_status": lifecycle}),
        "model_id": _id(raw["model_id"]),
        "model_sha256": None if sealed and raw["model_sha256"] is None else _hash(raw["model_sha256"]),
        "prediction_id": prediction_id,
        "probabilities": probabilities,
        "race_id": _text(raw["race_id"]) if sealed else _id(raw["race_id"]),
    }


def _collector_lane(value: Any, *, request_now: datetime) -> dict[str, Any]:
    raw = _exact(
        value,
        frozenset(
            {
                "lane",
                "status",
                "run_id",
                "phase",
                "cycle_state",
                "deadline_utc",
                "state_age_seconds",
                "component_identity",
                "reference_hashes",
                "operational_context",
            }
        ),
    )
    lane = _text(raw["lane"])
    if lane not in {"FULL_DAEMON", "ODDS_ONLY"}:
        raise ValueError("unknown collector lane")
    status = _text(raw["status"])
    if status not in _COLLECTOR_LANE_STATUSES:
        raise ValueError("unknown collector lane status")
    context_raw = _exact(
        raw["operational_context"],
        frozenset({
            "final_status", "final_verdict", "status",
            "next_meaningful_action", "next_meaningful_action_at",
            "lock_owner", "recent_capture",
        }),
    )
    optional_text = lambda item: None if item is None else _text(item)
    optional_time = lambda item: None if item is None else _utc(item, field="operational timestamp")[0]
    owner = None
    if context_raw["lock_owner"] is not None:
        owner_raw = _exact(
            context_raw["lock_owner"], frozenset({"kind", "run_id", "started_at"})
        )
        owner = {
            "kind": _text(owner_raw["kind"]),
            "run_id": _text(owner_raw["run_id"]),
            "started_at": _utc(owner_raw["started_at"], field="lock owner started_at")[0],
        }
    capture_raw = _exact(
        context_raw["recent_capture"],
        frozenset({
            "inserted_live_odds_rows", "ready_count", "status_counts",
            "blocked_attempt_count",
        }),
    )
    def optional_count(item: Any) -> int | None:
        if item is None:
            return None
        if type(item) is not int or item < 0 or item > 1_000_000_000:
            raise ValueError("collector count is invalid or unbounded")
        return item
    status_counts = None
    if capture_raw["status_counts"] is not None:
        if not isinstance(capture_raw["status_counts"], Mapping) or len(capture_raw["status_counts"]) > 64:
            raise ValueError("collector status counts are invalid or unbounded")
        status_counts = {
            _text(name): optional_count(count)
            for name, count in capture_raw["status_counts"].items()
        }
    context = {
        "final_status": optional_text(context_raw["final_status"]),
        "final_verdict": optional_text(context_raw["final_verdict"]),
        "status": optional_text(context_raw["status"]),
        "next_meaningful_action": optional_text(context_raw["next_meaningful_action"]),
        "next_meaningful_action_at": optional_time(context_raw["next_meaningful_action_at"]),
        "lock_owner": owner,
        "recent_capture": {
            "inserted_live_odds_rows": optional_count(capture_raw["inserted_live_odds_rows"]),
            "ready_count": optional_count(capture_raw["ready_count"]),
            "status_counts": status_counts,
            "blocked_attempt_count": optional_count(capture_raw["blocked_attempt_count"]),
        },
    }
    deadline_text, deadline_at = (
        (None, None)
        if raw["deadline_utc"] is None
        else _utc(raw["deadline_utc"], field="deadline_utc")
    )
    state_age = (
        None
        if raw["state_age_seconds"] is None
        else _finite_number(raw["state_age_seconds"], minimum=0)
    )
    deadline_statuses = _COLLECTOR_TERMINAL_STATUSES | {"ACTIVE"}
    if status in deadline_statuses and (
        deadline_at is None or state_age is None
    ):
        raise ValueError("current collector lane requires deadline and age")
    if (
        status in deadline_statuses
        and request_now.astimezone(timezone.utc) > deadline_at
    ):
        raise ValueError("current collector lane has passed its deadline")
    return {
        "component_identity": _identity(raw["component_identity"]),
        "deadline_utc": deadline_text,
        "lane": lane,
        "cycle_state": _text(raw["cycle_state"]),
        "operational_context": context,
        "phase": _text(raw["phase"]),
        "reference_hashes": _named_hashes(
            raw["reference_hashes"],
            required=status not in {"DATA_MISSING", "INTEGRITY_FAILED", "DIVERGENT"},
        ),
        "run_id": _text(raw["run_id"]),
        "state_age_seconds": state_age,
        "status": status,
    }


def _corpus_report(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("corpus report is invalid")
    status = _text(value.get("status"))
    if status == "UNAVAILABLE":
        raw = _exact(
            value,
            frozenset(
                {
                    "report_id",
                    "chain_hashes",
                    "generated_at",
                    "status",
                    "admission_gap",
                }
            ),
        )
        return {
            "chain_hashes": _named_hashes(raw["chain_hashes"]),
            "generated_at": _utc(raw["generated_at"], field="generated_at")[0],
            "report_id": _id(raw["report_id"]),
            "status": status,
            "admission_gap": _text(raw["admission_gap"]),
        }
    raw = _exact(
        value,
        frozenset(
            {
                "report_id",
                "population_id",
                "population_count",
                "funnel_counts",
                "exclusions",
                "chain_hashes",
                "generated_at",
                "status",
                "admission_gap",
            }
        ),
    )
    counts = raw["funnel_counts"]
    if not isinstance(counts, dict) or not counts:
        raise ValueError("funnel counts are required")
    normalized_counts: dict[str, int] = {}
    for name, count in counts.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("funnel count is invalid")
        normalized_counts[_id(name)] = count
    exclusions = []
    for item in _items(raw["exclusions"]):
        part = _exact(item, frozenset({"reason", "count"}))
        if isinstance(part["count"], bool) or not isinstance(part["count"], int) or part["count"] < 0:
            raise ValueError("exclusion count is invalid")
        exclusions.append({"count": part["count"], "reason": _id(part["reason"])})
    population_count = raw["population_count"]
    if isinstance(population_count, bool) or not isinstance(population_count, int) or population_count < 0:
        raise ValueError("population count is invalid")
    status = _text(raw["status"])
    if status not in _CORPUS_STATUSES:
        raise ValueError("unknown corpus status")
    return {
        "chain_hashes": _named_hashes(raw["chain_hashes"]),
        "exclusions": exclusions,
        "funnel_counts": normalized_counts,
        "generated_at": _utc(raw["generated_at"], field="generated_at")[0],
        "population_count": population_count,
        "population_id": _id(raw["population_id"]),
        "report_id": _id(raw["report_id"]),
        "status": status,
        "admission_gap": _text(raw["admission_gap"]),
    }


def _model(value: Any) -> dict[str, Any]:
    raw = _exact(
        value,
        frozenset(
            {
                "model_id",
                "model_sha256",
                "config_id",
                "config_sha256",
                "manifest_sha256",
                "role",
                "evaluation_status",
                "evaluation_claim",
                "slice_id",
                "evaluation_hashes",
            }
        ),
    )
    role = _text(raw["role"])
    status = _text(raw["evaluation_status"])
    if role not in _MODEL_ROLES or status not in _EVALUATION_STATUSES:
        raise ValueError("unknown model role or evaluation status")
    slice_id = raw["slice_id"]
    claim = raw["evaluation_claim"]
    if (slice_id is None) != (claim is None):
        raise ValueError("evaluation slice and claim must be jointly available")
    model_hash = raw["model_sha256"]
    manifest_hash = raw["manifest_sha256"]
    if (model_hash is None) != (manifest_hash is None):
        raise ValueError("model and manifest identities must be jointly available")
    return {
        "config_id": _id(raw["config_id"]),
        "config_sha256": _hash(raw["config_sha256"]),
        "evaluation_claim": None if claim is None else _text(claim),
        "evaluation_hashes": _named_hashes(
            raw["evaluation_hashes"], required=claim is not None
        ),
        "evaluation_status": status,
        "manifest_sha256": None if manifest_hash is None else _hash(manifest_hash),
        "model_id": _id(raw["model_id"]),
        "model_sha256": None if model_hash is None else _hash(model_hash),
        "role": role,
        "slice_id": None if slice_id is None else _id(slice_id),
    }


def _component(value: Any, *, request_now: datetime) -> dict[str, Any]:
    raw = _exact(
        value,
        frozenset(
            {
                "component",
                "status",
                "source_commit",
                "source_tree",
                "deployed_commit",
                "deployed_tree",
                "version",
                "observed_at",
                "age_seconds",
                "reference_hashes",
                "service_status",
            }
        ),
    )
    status = _text(raw["status"])
    if status not in _COMPONENT_STATUSES:
        raise ValueError("unknown component status")
    age = _finite_number(raw["age_seconds"], minimum=0)
    observed_text, observed = _utc(raw["observed_at"], field="observed_at")
    computed_age = (
        request_now.astimezone(timezone.utc) - observed
    ).total_seconds()
    if computed_age < 0 or not math.isclose(
        age, computed_age, abs_tol=_CLOCK_SKEW_SECONDS
    ):
        raise ValueError("component age is future or inconsistent")
    bounded_nonhealthy = status in {"DEGRADED", "STALE", "DIVERGENT"}
    unavailable = bounded_nonhealthy and raw["reference_hashes"] is None

    def component_oid(value: Any) -> str | None:
        if unavailable and value is None:
            return None
        return _git_oid(value)

    source_commit = component_oid(raw["source_commit"])
    source_tree = component_oid(raw["source_tree"])
    deployed_commit = component_oid(raw["deployed_commit"])
    deployed_tree = component_oid(raw["deployed_tree"])
    identities_match = (
        source_commit == deployed_commit and source_tree == deployed_tree
    )
    if status in {"HEALTHY", "DEGRADED"} and age > 60:
        raise ValueError("healthy or degraded component is stale")
    if status == "STALE" and age <= 60:
        raise ValueError("stale component has fresh evidence")
    if not unavailable and not identities_match and status in {"HEALTHY", "DEGRADED", "STALE"}:
        raise ValueError("component identity mismatch must be divergent")
    service_status = _exact(raw["service_status"], frozenset({"full", "odds"}))
    services: dict[str, dict[str, Any]] = {}
    for lane, value in service_status.items():
        service = _exact(
            value, frozenset({"active_state", "sub_state", "exec_main_pid"})
        )
        pid = service["exec_main_pid"]
        if bounded_nonhealthy:
            active_state = (
                None if service["active_state"] is None else _text(service["active_state"])
            )
            sub_state = (
                None if service["sub_state"] is None else _text(service["sub_state"])
            )
            if pid is not None and (type(pid) is not int or pid < 0):
                raise ValueError("service process identity is invalid")
        else:
            active_state = _text(service["active_state"])
            sub_state = _text(service["sub_state"])
            if type(pid) is not int or pid < 0:
                raise ValueError("service process identity is invalid")
        services[lane] = {
            "active_state": active_state,
            "sub_state": sub_state,
            "exec_main_pid": pid,
        }
    return {
        "age_seconds": age,
        "component": _id(raw["component"]),
        "deployed_commit": deployed_commit,
        "deployed_tree": deployed_tree,
        "observed_at": observed_text,
        "reference_hashes": (
            None if unavailable else _named_hashes(raw["reference_hashes"])
        ),
        "source_commit": source_commit,
        "source_tree": source_tree,
        "service_status": services,
        "status": status,
        "version": _text(raw["version"]),
    }


def _audit_event(value: Any) -> dict[str, Any]:
    raw = _exact(
        value,
        frozenset(
            {
                "event_id",
                "event_time_utc",
                "classification",
                "event_hash",
                "previous_event_hash",
                "segment_id",
                "reference_hashes",
            }
        ),
    )
    classification = _text(raw["classification"])
    if classification not in _AUDIT_CLASSIFICATIONS:
        raise ValueError("unknown audit classification")
    previous = raw["previous_event_hash"]
    return {
        "classification": classification,
        "event_hash": _hash(raw["event_hash"]),
        "event_id": _id(raw["event_id"]),
        "event_time_utc": _utc(raw["event_time_utc"], field="event_time_utc")[0],
        "previous_event_hash": None if previous is None else _hash(previous),
        "reference_hashes": _named_hashes(raw["reference_hashes"]),
        "segment_id": _id(raw["segment_id"]),
    }


def _validate_data(
    resource: str,
    value: Any,
    route_id: str | None,
    request_now: datetime,
) -> dict[str, Any]:
    if resource == "overview":
        raw = _exact(value, frozenset({"sections"}))
        sections = []
        for item in _items(raw["sections"]):
            part = _exact(
                item, frozenset({"resource", "classification", "evidence_identity"})
            )
            classification = _text(part["classification"])
            if classification not in _OVERVIEW_STATUSES:
                raise ValueError("unknown overview classification")
            sections.append(
                {
                    "classification": classification,
                    "evidence_identity": _identity(part["evidence_identity"]),
                    "resource": _id(part["resource"]),
                }
            )
        _require_unique(sections, "resource", "overview resources")
        return {"sections": sections}
    if resource in {"upcoming_races", "race_detail"}:
        key = "races" if resource == "upcoming_races" else "race"
        raw = _exact(value, frozenset({key}))
        if resource == "race_detail":
            return {key: _race(raw[key], route_id=route_id)}
        races = [_race(item, route_id=None) for item in _items(raw[key])]
        _require_unique(races, "race_id", "upcoming race identities")
        _require_unique(races, "source_race_id", "upcoming source race identities")
        _require_unique(races, "source_url", "upcoming source race URLs")
        return {key: races}
    if resource in {"recent_predictions", "prediction_detail"}:
        key = "predictions" if resource == "recent_predictions" else "prediction"
        raw = _exact(value, frozenset({key}))
        if resource == "prediction_detail":
            return {key: _prediction(raw[key], route_id=route_id)}
        predictions = [
            _prediction(item, route_id=None) for item in _items(raw[key])
        ]
        _require_unique(
            predictions, "prediction_id", "recent prediction identities"
        )
        return {key: predictions}
    key, validator = {
        "collector": (
            "lanes",
            lambda item: _collector_lane(item, request_now=request_now),
        ),
        "corpus": ("reports", _corpus_report),
        "models": ("models", _model),
        "system": (
            "components",
            lambda item: _component(item, request_now=request_now),
        ),
        "audit": ("events", _audit_event),
    }[resource]
    raw = _exact(value, frozenset({key}))
    values = [validator(item) for item in _items(raw[key])]
    if resource == "collector":
        lanes = [item["lane"] for item in values]
        if len(lanes) != 2 or set(lanes) != {"FULL_DAEMON", "ODDS_ONLY"}:
            raise ValueError("collector requires exactly one lane of each kind")
    elif resource == "corpus":
        _require_unique(values, "report_id", "corpus report identities")
    elif resource == "models":
        _require_unique(values, "model_id", "model identities")
    elif resource == "system":
        _require_unique(values, "component", "system components")
    else:
        _require_unique(values, "event_id", "audit event identities")
    return {key: values}


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _validate_envelope(
    resource: str, envelope: Any, request_now: datetime
) -> tuple[dict[str, Any], EvidenceStatus, tuple[str, ...], tuple[str, ...]]:
    if type(envelope) is not EvidenceEnvelope:
        raise ValueError("provider must return a server-minted EvidenceEnvelope")
    serialized = envelope.to_dict()
    if set(serialized) != {
        "source_kind",
        "source_identity",
        "content_sha256",
        "source_locator",
        "source_at",
        "generated_at",
        "observed_at",
        "server_observed_at",
        "age_seconds",
        "freshness_policy",
        "availability",
        "schema_integrity",
        "reference_hashes",
        "evidence_identity",
        "status",
        "supported_claim",
    }:
        raise ValueError("evidence envelope serialization is not exact")
    # Re-serialization detects partially initialized or exotic mutated values.
    if json.loads(envelope.to_json()) != serialized:
        raise ValueError("evidence envelope serialization is inconsistent")
    _text(envelope.source_kind, identifier=True)
    _text(envelope.source_identity)
    _text(envelope.source_locator, identifier=True)
    _text(envelope.supported_claim)
    policy = POLICIES.get(envelope.freshness_policy)
    if policy is None or envelope.freshness_policy not in _RESOURCE_POLICIES[resource]:
        raise ValueError("policy is unknown or invalid for resource")
    try:
        status = EvidenceStatus(envelope.status)
    except (TypeError, ValueError) as exc:
        raise ValueError("unknown evidence status") from exc
    if envelope.availability not in {"present", "missing", "unreadable", "error"}:
        raise ValueError("unknown evidence availability")
    if envelope.schema_integrity not in {"valid", "failed", "unknown"}:
        raise ValueError("unknown schema integrity")
    references = _named_hashes(serialized["reference_hashes"], required=False)
    identity = serialized["evidence_identity"]
    if identity is not None:
        identity = _identity(identity)
    content: tuple[str, ...] = ()
    if envelope.content_sha256 is not None:
        content = (_hash(envelope.content_sha256),)
    expected_axis = {
        EvidenceStatus.AVAILABLE_FRESH: ("present", "valid"),
        EvidenceStatus.STALE: ("present", "valid"),
        EvidenceStatus.UNAVAILABLE_DATA_MISSING: (None, None),
        EvidenceStatus.INVALID_INTEGRITY_FAILED: (None, "failed"),
        EvidenceStatus.DIVERGENT: ("present", "valid"),
    }[status]
    operational_degraded = (
        resource in _NON_HEALTHY_DISCLOSURE_RESOURCES
        and status is EvidenceStatus.UNAVAILABLE_DATA_MISSING
        and envelope.availability == "present"
        and envelope.schema_integrity == "valid"
    )
    if (
        not operational_degraded
        and expected_axis[0] is not None
        and envelope.availability != expected_axis[0]
    ) or (
        not operational_degraded
        and expected_axis[1] is not None
        and envelope.schema_integrity != expected_axis[1]
    ):
        raise ValueError("evidence status axes are inconsistent")
    if status is EvidenceStatus.UNAVAILABLE_DATA_MISSING and not operational_degraded and (
        envelope.availability == "present" or envelope.schema_integrity == "valid"
    ):
        raise ValueError("missing evidence axes are inconsistent")
    bounded_incomplete_system = (
        resource == "system"
        and status in {EvidenceStatus.STALE, EvidenceStatus.DIVERGENT}
        and bool(content)
        and identity is None
    )
    if status in {
        EvidenceStatus.AVAILABLE_FRESH,
        EvidenceStatus.STALE,
        EvidenceStatus.DIVERGENT,
    } and (not content or (identity is None and not bounded_incomplete_system)):
        raise ValueError("classified evidence requires content and exact identity")

    _, observed = _utc(envelope.server_observed_at, field="server_observed_at")
    request_utc = request_now.astimezone(timezone.utc)
    if abs((request_utc - observed).total_seconds()) > _CLOCK_SKEW_SECONDS:
        raise ValueError("provider used a different server observation identity")
    times = [
        _utc(value, field=name)[1]
        for name, value in (
            ("source_at", envelope.source_at),
            ("generated_at", envelope.generated_at),
            ("observed_at", envelope.observed_at),
        )
        if value is not None
    ]
    age = envelope.age_seconds
    if status in {
        EvidenceStatus.UNAVAILABLE_DATA_MISSING,
        EvidenceStatus.INVALID_INTEGRITY_FAILED,
    } and not operational_degraded:
        if times or age is not None:
            raise ValueError("unavailable or invalid evidence cannot claim age")
    elif status in {EvidenceStatus.AVAILABLE_FRESH, EvidenceStatus.STALE}:
        if len(times) != 1 or age is None:
            raise ValueError("evidence requires one source time and finite age")
        age = _finite_number(age, minimum=0)
        computed_age = (request_utc - times[0]).total_seconds()
        if computed_age < 0 or not math.isclose(
            age, computed_age, abs_tol=_CLOCK_SKEW_SECONDS
        ):
            raise ValueError("evidence age is future or inconsistent")
    else:
        if bool(times) != (age is not None) or len(times) > 1:
            raise ValueError("divergent evidence time and age must be paired")
        if age is not None:
            age = _finite_number(age, minimum=0)
            computed_age = (request_utc - times[0]).total_seconds()
            if computed_age < 0 or not math.isclose(
                age, computed_age, abs_tol=_CLOCK_SKEW_SECONDS
            ):
                raise ValueError("divergent evidence age is future or inconsistent")

    if policy.mode == "fixed":
        assert policy.maximum_age_seconds is not None
        derived = (
            EvidenceStatus.AVAILABLE_FRESH
            if age is not None and age <= policy.maximum_age_seconds
            else EvidenceStatus.STALE
        )
        if status not in {
            EvidenceStatus.INVALID_INTEGRITY_FAILED,
            EvidenceStatus.DIVERGENT,
            EvidenceStatus.UNAVAILABLE_DATA_MISSING,
        } and status is not derived:
            raise ValueError("fixed-policy status disagrees with request clock")
        status = (
            status
            if status
            in {
                EvidenceStatus.INVALID_INTEGRITY_FAILED,
                EvidenceStatus.DIVERGENT,
                EvidenceStatus.UNAVAILABLE_DATA_MISSING,
            }
            else derived
        )
    # Adapter and historical policies have no invented API threshold.  Their
    # status remains the validated server observation's finite status.
    normalized = {
        **serialized,
        "reference_hashes": references,
        "evidence_identity": identity,
        "status": status.value,
    }
    return normalized, status, content, tuple(references.values())


def register_level_1_provider(
    app: Flask, resource: str, provider: Callable[..., APIObservation]
) -> None:
    """Register one server-owned adapter after API/application construction."""
    if resource not in _RESOURCE_POLICIES:
        raise ValueError("unknown API resource")
    if not callable(provider):
        raise TypeError("API provider must be callable")
    registry = app.extensions.get(_REGISTRY_KEY)
    if not isinstance(registry, dict):
        raise RuntimeError("Operator UI Level-1 API is not installed")
    if resource in registry:
        raise ValueError("API provider replacement is forbidden")
    registry[resource] = provider


def install_level_1_api(app: Flask) -> bool:
    """Install the sole authenticated, audited, GET-only API namespace."""
    operational_get = app.extensions.get("operator_ui_operational_get")
    if operational_get is None:
        return False
    if app.extensions.get("operator_ui_level_1_api_installed"):
        raise RuntimeError("Operator UI Level-1 API is already installed")
    clock = app.config.get("OPERATOR_UI_CLOCK", lambda: datetime.now(timezone.utc))
    if not callable(clock):
        raise ValueError("API clock must be callable")
    app.extensions[_REGISTRY_KEY] = {}

    def register_route(resource: str, rule: str, *, detail: bool = False) -> None:
        endpoint = f"operator_ui_api_v1_{resource}"

        @app.get(rule, endpoint=endpoint)
        @operational_get(policy=f"LEVEL_1_API_V1_{resource.upper()}")
        def view(identifier: str | None = None) -> PreparedDisclosure:
            # GET means no input beyond the bounded route identity.  Reject
            # even an empty-valued query key or any encoded/body/form bytes.
            if request.query_string or request.get_data(cache=True):
                raise ValueError("operational GET accepts no query or body")
            route_id = _id(identifier) if detail else None
            request_now = clock()
            if not isinstance(request_now, datetime) or request_now.tzinfo is None:
                raise ValueError("API clock must return an aware datetime")
            registry = app.extensions[_REGISTRY_KEY]
            provider = registry.get(resource)
            if provider is None:
                observed_text = (
                    request_now.astimezone(timezone.utc)
                    .isoformat(timespec="microseconds")
                    .replace("+00:00", "Z")
                )
                source_identity = f"operator_ui.adapter.{resource}.unregistered"
                envelope = {
                    "source_kind": "server_adapter_registry",
                    "source_identity": source_identity,
                    "content_sha256": None,
                    "source_locator": f"operator_ui.adapter_registry.{resource}",
                    "source_at": None,
                    "generated_at": None,
                    "observed_at": None,
                    "server_observed_at": observed_text,
                    "age_seconds": None,
                    "freshness_policy": _RESOURCE_DEFAULT_POLICY[resource],
                    "availability": "missing",
                    "schema_integrity": "unknown",
                    "reference_hashes": {},
                    "evidence_identity": None,
                    "status": EvidenceStatus.UNAVAILABLE_DATA_MISSING.value,
                    "supported_claim": (
                        f"No server-owned {resource} adapter was registered "
                        "for this request."
                    ),
                }
                body = _canonical(
                    {
                        "api_version": API_VERSION,
                        "classification": EvidenceStatus.UNAVAILABLE_DATA_MISSING.value,
                        "evidence": envelope,
                        "reason": "ADAPTER_NOT_REGISTERED",
                        "resource": resource,
                        "schema": API_SCHEMA,
                        "server_observed_at": observed_text,
                        "stale": False,
                    }
                )
                response_hash = hashlib.sha256(body).hexdigest()
                return PreparedDisclosure(
                    body=body,
                    classification=EvidenceStatus.UNAVAILABLE_DATA_MISSING,
                    evidence_source_identifiers=(source_identity,),
                    content_hashes=(),
                    reference_hashes=(response_hash,),
                )
            raw = (
                provider(route_id, request_now)
                if detail
                else provider(request_now)
            )
            if type(raw) is not APIObservation:
                raise ValueError("provider must return an APIObservation")
            envelope, classification, content, references = _validate_envelope(
                resource, raw.evidence, request_now
            )
            response: dict[str, Any] = {
                "api_version": API_VERSION,
                "classification": classification.value,
                "evidence": envelope,
                "resource": resource,
                "schema": API_SCHEMA,
                "server_observed_at": envelope["server_observed_at"],
                "stale": classification is EvidenceStatus.STALE,
            }
            disclose_data = classification is EvidenceStatus.AVAILABLE_FRESH or (
                resource in _NON_HEALTHY_DISCLOSURE_RESOURCES
                and classification in _NON_HEALTHY_DISCLOSURE_STATUSES
                and bool(raw.data)
            )
            if disclose_data:
                data = _validate_data(resource, raw.data, route_id, request_now)
                if resource in {"upcoming_races", "race_detail"}:
                    races: Sequence[dict[str, Any]] = (
                        data["races"]
                        if resource == "upcoming_races"
                        else (data["race"],)
                    )
                    now_utc = request_now.astimezone(timezone.utc)
                    if any(race.pop("_jump") <= now_utc for race in races):
                        raise ValueError("available race is not strictly pre-jump")
                    if envelope["freshness_policy"] != "P-UPCOMING-300-PREJUMP":
                        raise ValueError("available race has wrong policy")
                    if envelope["age_seconds"] > 300:
                        raise ValueError("available race evidence exceeds 300 seconds")
                response["data"] = data
            elif (
                (
                    resource in _FINITE_EMPTY_INVALID_RESOURCES
                    or resource == "system" and not raw.data
                )
                and classification is EvidenceStatus.INVALID_INTEGRITY_FAILED
            ):
                response["data"] = {}
            elif raw.data:
                raise ValueError("non-available evidence must disclose no data")
            body = _canonical(response)
            response_hash = hashlib.sha256(body).hexdigest()
            return PreparedDisclosure(
                body=body,
                classification=classification,
                evidence_source_identifiers=(envelope["source_identity"],),
                content_hashes=content,
                reference_hashes=(*references, response_hash),
            )

    register_route("overview", f"{API_PREFIX}/overview")
    register_route("upcoming_races", f"{API_PREFIX}/races/upcoming")
    register_route("race_detail", f"{API_PREFIX}/races/<identifier>", detail=True)
    register_route("recent_predictions", f"{API_PREFIX}/predictions/recent")
    register_route(
        "prediction_detail", f"{API_PREFIX}/predictions/<identifier>", detail=True
    )
    register_route("collector", f"{API_PREFIX}/collector")
    register_route("corpus", f"{API_PREFIX}/corpus")
    register_route("models", f"{API_PREFIX}/models")
    register_route("system", f"{API_PREFIX}/system")
    register_route("audit", f"{API_PREFIX}/audit")
    app.extensions["operator_ui_level_1_api_installed"] = True
    return True


__all__ = [
    "API_PREFIX",
    "API_SCHEMA",
    "API_VERSION",
    "APIObservation",
    "install_level_1_api",
    "register_level_1_provider",
]
