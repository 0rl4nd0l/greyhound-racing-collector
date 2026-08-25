"""Safety core for isolated, on-demand pre-jump research predictions."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import stat
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from scripts.refresh_prejump_upcoming import stable_race_id, stable_race_id_variants
from utils.csv_metadata import (
    canonical_thedogs_race_identity,
    canonical_thedogs_venue_identity,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_ALIASES = {
    "latest-research": "market_form_residual_v1",
    "market-form-residual-v1": "market_form_residual_v1",
    "market_form_residual_v1": "market_form_residual_v1",
    "market-only": "market_only_v1",
    "market-only-implied": "market_only_v1",
    "market_only_implied": "market_only_v1",
    "market_only_v1": "market_only_v1",
}
MODEL_FILES = {
    "market_only_v1": None,
    "market_form_residual_v1": ROOT / "artifacts/frozen_models/market_form_residual_v1",
}
SCHEMA_FILES = {
    "market_only_v1": ROOT / "configs/prediction/schemas/market_only_v1.schema.json",
    "market_form_residual_v1": ROOT
    / "configs/prediction/schemas/market_form_residual_v1.schema.json",
}
OUTCOME_KEYS = {
    "actual_win",
    "finish_position",
    "official_result",
    "outcome",
    "placing",
    "result",
    "winner",
    "winner_name",
}

PREDICTION_BUNDLE_INDEX_NAME = "prediction_bundle_index_v1.json"
PREDICTION_BUNDLE_LOCK_NAME = "prediction_bundle_index_v1.lock"
PREDICTION_BUNDLE_INDEX_SCHEMA = "on_demand_prediction_bundle_index_v1"
PREDICTION_RESULT_SCHEMA_V2 = "on_demand_race_prediction_v2"
PREDICTION_MANIFEST_SCHEMA_V2 = "on_demand_prediction_bundle_manifest_v2"
PREDICTION_BUNDLE_DIRECTORY_RE = __import__("re").compile(
    r"prediction_[0-9]{8}T[0-9]{12}[+-][0-9]{4}_[0-9a-f]{12}"
)
_SHA256_RE = __import__("re").compile(r"[0-9a-f]{64}")
_UUID_RE = __import__("re").compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)
_JOB_ID_RE = re.compile(r"job_[0-9a-f]{32}")

# The current-index reader propagates these codes dynamically.  Keep this
# finite map explicit so producer coverage does not depend on literal-call ASTs.
CURRENT_INDEX_BLOCKER_STAGE_BY_CODE = {
    code: "PROTOCOL"
    for code in {
        "CURRENT_INDEX_INVALID",
        "CURRENT_INDEX_PATH_UNSAFE",
        "CURRENT_INDEX_PUBLICATION_INVALID",
        "CURRENT_INDEX_PUBLICATION_MISSING",
        "CURRENT_INDEX_REPORT_INVALID",
        "CURRENT_INDEX_REPORT_MISSING",
        "CURRENT_INDEX_SIZE_INVALID",
        "CURRENT_INDEX_SOURCE_CHANGED",
        "CURRENT_INDEX_SOURCE_INVALID",
        "CURRENT_INDEX_SOURCE_MISSING",
        "CURRENT_INDEX_STALE",
        "CURRENT_INDEX_UNAVAILABLE",
        "CURRENT_INDEX_UNBOUNDED",
    }
}

# This is the exhaustive producer terminal vocabulary.  The producer imports
# this mapping and the verifier checks the exact pair; neither side guesses a
# stage for an unknown code.
BLOCKER_STAGE_BY_CODE = {
    **CURRENT_INDEX_BLOCKER_STAGE_BY_CODE,
    **{code: "PROTOCOL" for code in {
        "CAPTURE_FAILED", "CAPTURE_WINDOW_CLOSED", "COLLECTOR_CAPTURE_ONE_UNAVAILABLE",
        "COLLECTOR_PROCESS_INVALID", "COLLECTOR_PROTOCOL_INVALID", "DISCOVERY_BUDGET_INVALID",
        "DISCOVERY_TIMEOUT", "DUPLICATE_ATTEMPT", "DUPLICATE_CLAIM", "DUPLICATE_EXACT_RECEIPT",
        "DUPLICATE_RECEIPT", "DUPLICATE_RESPONSE", "EXACT_METADATA_UNAVAILABLE", "IDENTITY_MISMATCH",
        "INSUFFICIENT_PREJUMP_MARGIN", "LOCK_PATH_UNSAFE", "LOCK_RELEASE_FAILED",
        "ODDS_SOURCE_UNSUPPORTED", "RECEIPT_AMBIGUOUS", "RECEIPT_CONTAINS_OUTCOME",
        "RECEIPT_INVALID", "RECEIPT_STALE", "RECEIPT_TAMPERED", "RECEIPT_UNAVAILABLE",
        "REPLAYED_REQUEST", "RESPONSE_ALREADY_CONSUMED", "SOURCE_FILE_UNSAFE",
        "PREDICTION_BUNDLE_INDEX_LOCK_INVALID", "PREDICTION_BUNDLE_INDEX_LOCK_RELEASE_FAILED",
        "PREDICTION_BUNDLE_INDEX_LOCK_REPLACED", "PREDICTION_BUNDLE_INDEX_LOCK_UNAVAILABLE",
        "PREDICTION_BUNDLE_INDEX_WRITE_FAILED", "PREDICTION_BUNDLE_REPLACED",
    }},
    **{code: "VALIDATION" for code in {
        "AMBIGUOUS_RACE", "BUNDLE_SOURCE_UNSAFE", "CONFIG_INVALID_JSON", "CONFIG_NOT_CANONICAL",
        "CONFIG_SCHEMA_MISMATCH", "CURRENT_TIME_TIMEZONE_MISSING", "EXACT_RACE_IDENTITY_UNAVAILABLE",
        "HISTORY_CUTOFF_AMBIGUOUS", "HISTORY_DATABASE_BUSY", "HISTORY_DATABASE_CHANGED", "HISTORY_DATABASE_INTEGRITY_FAILED", "HISTORY_DATABASE_UNAVAILABLE", "HISTORY_IDENTITY_AMBIGUOUS", "HISTORY_SEAL_WRITE_FAILED",
        "HISTORY_SCHEMA_AMBIGUOUS", "HISTORY_SCHEMA_MISSING", "MODEL_ARTIFACT_MISSING",
        "MODEL_CONFIG_MISMATCH", "MODEL_SCHEMA_INVALID", "MODEL_SCHEMA_MISSING", "MODEL_UNSUPPORTED", "NO_MATCH",
        "ODDS_TIMESTAMP_AMBIGUOUS", "OUTPUT_ROOT_UNSAFE", "OUTPUT_ROOT_WRITABLE_BY_OTHERS",
        "POST_JUMP", "RUNNER_SET_AMBIGUOUS", "TARGET_EXCLUSION_WEAK", "WRITE_TARGET_EXISTS",
        "PREDICTION_BUNDLE_CHANGED", "PREDICTION_BUNDLE_DEADLINE_EXCEEDED",
        "PREDICTION_BUNDLE_ENUMERATION_FAILED", "PREDICTION_BUNDLE_IDENTITY_MISMATCH",
        "PREDICTION_BUNDLE_INVALID", "PREDICTION_BUNDLE_MEMBERSHIP_MISMATCH",
        "PREDICTION_BUNDLE_OPEN_FAILED", "PREDICTION_BUNDLE_UNSAFE_TYPE",
    }},
    **{code: "SCORING" for code in {
        "FEATURE_SEAL_FAILED", "FROZEN_MODEL_DRIFT", "MARKET_UNAVAILABLE",
        "PREDICTION_INTERNAL_ERROR", "RESIDUAL_SCORER_FAILED",
    }},
}
INDEX_MAX_BYTES = 512 * 1024
INDEX_MAX_ENTRIES = 256
BUNDLE_MAX_ENTRIES = 32
BUNDLE_CONTROL_MAX_BYTES = 1024 * 1024
BUNDLE_FILE_MAX_BYTES = 64 * 1024 * 1024
BUNDLE_AGGREGATE_MAX_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class VerifiedPredictionBundle:
    directory: str
    index_entry: dict[str, Any]
    result: dict[str, Any]
    manifest: dict[str, Any]
    request: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class VerifiedPredictionBundleIndex:
    """Exact descriptor-retained producer index snapshot for read adapters."""

    schema_version: str
    published_at: str | None
    entries: tuple[Mapping[str, Any], ...]
    canonical_bytes: bytes
    sha256: str


def _blocked(code: str, **details: Any) -> PredictionBlocked:
    return PredictionBlocked(code, **details)


def _deadline(start: float, seconds: float, monotonic: Callable[[], float]) -> None:
    if monotonic() - start > seconds:
        raise _blocked("PREDICTION_BUNDLE_DEADLINE_EXCEEDED")


def _canonical_json(raw: bytes, *, max_bytes: int, label: str) -> Any:
    if not raw or len(raw) > max_bytes:
        raise _blocked("PREDICTION_BUNDLE_INVALID", field=label, reason="size")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise _blocked("PREDICTION_BUNDLE_INVALID", field=label, reason="json") from exc
    if canonical_bytes(value) != raw:
        raise _blocked("PREDICTION_BUNDLE_INVALID", field=label, reason="noncanonical")
    _reject_nonfinite(value, label)
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate:{key}")
        value[key] = item
    return value


def _reject_nonfinite(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise _blocked("PREDICTION_BUNDLE_INVALID", field=label, reason="nonfinite")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, f"{label}.{key}")
    elif isinstance(value, list):
        for offset, item in enumerate(value):
            _reject_nonfinite(item, f"{label}[{offset}]")


def _exact_fields(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise _blocked("PREDICTION_BUNDLE_INVALID", field=label, reason="fields")
    return value


OPERATIONAL_INDEX_PROVENANCE_FIELDS = {
    "schema", "index_schema_version", "run_id", "packet_sha256",
    "source_refresh_sha256", "publication_sha256", "state_sha256", "report_sha256",
}


def validate_operational_index_provenance(value: Any) -> dict[str, str]:
    provenance = _exact_fields(value, OPERATIONAL_INDEX_PROVENANCE_FIELDS, "operational_index_provenance")
    if provenance.get("schema") != "operator_ui_operational_index_admission_v1":
        raise _blocked("PREDICTION_BUNDLE_IDENTITY_MISMATCH", field="operational_index_provenance.schema")
    for name in ("index_schema_version", "run_id"):
        item=provenance.get(name)
        if not isinstance(item,str) or not item or len(item.encode())>256 or any(ord(char)<32 or ord(char)==127 for char in item):
            raise _blocked("PREDICTION_BUNDLE_IDENTITY_MISMATCH", field=f"operational_index_provenance.{name}")
    for name in OPERATIONAL_INDEX_PROVENANCE_FIELDS-{"schema","index_schema_version","run_id"}:
        if not isinstance(provenance.get(name),str) or re.fullmatch(r"[0-9a-f]{64}",provenance[name]) is None:
            raise _blocked("PREDICTION_BUNDLE_IDENTITY_MISMATCH", field=f"operational_index_provenance.{name}")
    return dict(provenance)


def _timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise _blocked("PREDICTION_BUNDLE_INVALID", field=label)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise _blocked("PREDICTION_BUNDLE_INVALID", field=label) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.isoformat() != value:
        raise _blocked("PREDICTION_BUNDLE_INVALID", field=label)
    return parsed


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise _blocked("PREDICTION_BUNDLE_INVALID", field=label)
    return value


def _prediction_id(value: Any) -> str:
    if not isinstance(value, str) or _UUID_RE.fullmatch(value) is None:
        raise _blocked("PREDICTION_BUNDLE_INVALID", field="prediction_id")
    try:
        if str(uuid.UUID(value)) != value:
            raise ValueError(value)
    except ValueError as exc:
        raise _blocked("PREDICTION_BUNDLE_INVALID", field="prediction_id") from exc
    return value


def _job_id(value: Any, label: str = "job_id") -> str | None:
    if value is not None and (
        not isinstance(value, str) or _JOB_ID_RE.fullmatch(value) is None
    ):
        raise _blocked("PREDICTION_BUNDLE_INVALID", field=label)
    return value


def _canonical_runner(value: Any, label: str) -> dict[str, Any]:
    row = _exact_fields(
        value,
        {"box_number", "display_name", "identity", "source_native_runner_id"},
        label,
    )
    box = row["box_number"]
    native_id = row["source_native_runner_id"]
    if (
        isinstance(box, bool) or not isinstance(box, int) or box <= 0
        or not isinstance(row["display_name"], str) or not row["display_name"]
        or row["display_name"] != row["display_name"].strip()
        or not isinstance(row["identity"], str) or not row["identity"]
        or row["identity"] != row["identity"].strip().upper()
        or (native_id is not None and (
            not isinstance(native_id, str) or not native_id
            or native_id != native_id.strip()
        ))
    ):
        raise _blocked("PREDICTION_BUNDLE_INVALID", field=label)
    return dict(row)


def canonical_runner_set(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) < 2:
        raise _blocked("PREDICTION_BUNDLE_INVALID", field=label)
    rows = [_canonical_runner(row, f"{label}[{offset}]") for offset, row in enumerate(value)]
    if rows != sorted(rows, key=lambda row: (row["box_number"], row["identity"])):
        raise _blocked("PREDICTION_BUNDLE_INVALID", field=label, reason="order")
    if len({row["box_number"] for row in rows}) != len(rows) or len({row["identity"] for row in rows}) != len(rows):
        raise _blocked("PREDICTION_BUNDLE_INVALID", field=label, reason="duplicate")
    native = [row["source_native_runner_id"] for row in rows if row["source_native_runner_id"] is not None]
    if len(native) != len(set(native)):
        raise _blocked("PREDICTION_BUNDLE_INVALID", field=label, reason="duplicate_native_id")
    return rows


def sealed_runner_set_sha256(race: Mapping[str, Any], runners: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(canonical_bytes({"race": dict(race), "runners": list(runners)}))


def _relative_name(value: Any) -> str:
    if not isinstance(value, str) or "\\" in value or value.startswith("/"):
        raise _blocked("PREDICTION_BUNDLE_INVALID", field="files.name")
    parts = value.split("/")
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise _blocked("PREDICTION_BUNDLE_INVALID", field="files.name")
    if len(parts) > BUNDLE_MAX_ENTRIES:
        raise _blocked("PREDICTION_BUNDLE_INVALID", field="files.name", reason="depth")
    return value


def logical_bundle_sha256(manifest: Mapping[str, Any]) -> str:
    payload = {
        "schema_version": manifest.get("schema_version"),
        "prediction_id": manifest.get("prediction_id"),
        "job_id": manifest.get("job_id"),
        "files": manifest.get("files"),
    }
    return sha256_bytes(canonical_bytes(payload))


def validate_prediction_bundle_manifest_v2(value: Any) -> dict[str, Any]:
    manifest = _exact_fields(
        value, {"schema_version", "prediction_id", "job_id", "files"}, "manifest"
    )
    if manifest["schema_version"] != PREDICTION_MANIFEST_SCHEMA_V2:
        raise _blocked("PREDICTION_BUNDLE_INVALID", field="manifest.schema_version")
    _prediction_id(manifest["prediction_id"])
    _job_id(manifest["job_id"], "manifest.job_id")
    files = manifest["files"]
    if not isinstance(files, Mapping) or not files or len(files) > BUNDLE_MAX_ENTRIES:
        raise _blocked("PREDICTION_BUNDLE_INVALID", field="manifest.files")
    if list(files) != sorted(files) or "bundle_manifest.json" in files:
        raise _blocked("PREDICTION_BUNDLE_INVALID", field="manifest.files")
    total = 0
    for name, evidence in files.items():
        _relative_name(name)
        item = _exact_fields(evidence, {"bytes", "sha256"}, f"files.{name}")
        size = item["bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or not 0 <= size <= BUNDLE_FILE_MAX_BYTES:
            raise _blocked("PREDICTION_BUNDLE_INVALID", field=f"files.{name}.bytes")
        _sha(item["sha256"], f"files.{name}.sha256")
        total += size
    if total > BUNDLE_AGGREGATE_MAX_BYTES:
        raise _blocked("PREDICTION_BUNDLE_INVALID", field="manifest.files", reason="aggregate")
    return dict(manifest)


def validate_prediction_result_v2(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version", "prediction_id", "job_id", "generated_at", "status",
        "blocker_stage", "blocker", "research_only", "production_persisted",
        "betting_output", "race", "model", "config", "evidence", "prediction",
    }
    result = _exact_fields(value, fields, "result")
    if result["schema_version"] != PREDICTION_RESULT_SCHEMA_V2:
        raise _blocked("PREDICTION_BUNDLE_INVALID", field="result.schema_version")
    _prediction_id(result["prediction_id"])
    _job_id(result["job_id"], "result.job_id")
    _timestamp(result["generated_at"], "result.generated_at")
    if (result["research_only"], result["production_persisted"], result["betting_output"]) != (True, False, False):
        raise _blocked("PREDICTION_BUNDLE_INVALID", field="result.safety")
    race = _exact_fields(result["race"], {"race_id", "url", "race_date", "venue", "venue_slug", "race_number", "jump_timestamp"}, "result.race")
    if not all(isinstance(race[key], str) and race[key] for key in ("race_id", "url", "race_date", "venue", "venue_slug")):
        raise _blocked("PREDICTION_BUNDLE_INVALID", field="result.race")
    try:
        if date.fromisoformat(race["race_date"]).isoformat() != race["race_date"]:
            raise ValueError(race["race_date"])
    except ValueError as exc:
        raise _blocked("PREDICTION_BUNDLE_INVALID", field="result.race.race_date") from exc
    identity = canonical_thedogs_race_identity(race["url"])
    if identity is None or identity["canonical_url"] != race["url"]:
        raise _blocked("PREDICTION_BUNDLE_INVALID", field="result.race.url")
    if not isinstance(race["race_number"], int) or isinstance(race["race_number"], bool) or race["race_number"] <= 0:
        raise _blocked("PREDICTION_BUNDLE_INVALID", field="result.race.race_number")
    url_venue = canonical_thedogs_venue_identity(identity["venue_slug"])
    sealed_venue = canonical_thedogs_venue_identity(race["venue"])
    race_projection = {"race_number": race["race_number"], "venue": race["venue"], "race_date": race["race_date"], "url": race["url"]}
    if (
        identity["race_date"] != race["race_date"]
        or identity["race_number"] != race["race_number"]
        or identity["venue_slug"] != race["venue_slug"]
        or url_venue is None
        or sealed_venue != url_venue
        or race["venue"] != url_venue
        or race["race_id"] != stable_race_id(race_projection)
        or race["race_id"] not in stable_race_id_variants(race_projection)
    ):
        raise _blocked("PREDICTION_BUNDLE_INVALID", field="result.race.identity")
    _timestamp(race["jump_timestamp"], "result.race.jump_timestamp")
    model = _exact_fields(result["model"], {"requested", "resolved", "alias_resolved", "schema_sha256", "artifact_identity", "artifact_sha256", "artifact_manifest_identity", "artifact_manifest_sha256"}, "result.model")
    normalized_requested = (
        model["requested"].strip().lower()
        if isinstance(model["requested"], str)
        else None
    )
    if (
        normalized_requested not in MODEL_ALIASES
        or model["resolved"] != MODEL_ALIASES[normalized_requested]
        or not isinstance(model["alias_resolved"], bool)
        or model["alias_resolved"] != (normalized_requested != model["resolved"])
    ):
        raise _blocked("PREDICTION_BUNDLE_INVALID", field="result.model.identity")
    if model["resolved"] == "market_only_v1":
        expected = ("UNAVAILABLE_NOT_APPLICABLE", None, "UNAVAILABLE_NOT_APPLICABLE", None)
        actual = (model["artifact_identity"], model["artifact_sha256"], model["artifact_manifest_identity"], model["artifact_manifest_sha256"])
        if actual != expected:
            raise _blocked("PREDICTION_BUNDLE_INVALID", field="result.model.artifacts")
    else:
        if model["artifact_identity"] != "AVAILABLE" or model["artifact_manifest_identity"] != "AVAILABLE":
            raise _blocked("PREDICTION_BUNDLE_INVALID", field="result.model.artifacts")
        _sha(model["artifact_sha256"], "result.model.artifact_sha256")
        _sha(model["artifact_manifest_sha256"], "result.model.artifact_manifest_sha256")
    _sha(model["schema_sha256"], "result.model.schema_sha256")
    config = _exact_fields(result["config"], {"sha256"}, "result.config")
    _sha(config["sha256"], "result.config.sha256")
    evidence = _exact_fields(result["evidence"], {"request", "config", "model_schema", "model_artifact", "model_manifest", "runner_set_sha256", "prediction_output_sha256", "protocol_chain", "authenticated_cutoff"}, "result.evidence")
    for name in ("request", "config", "model_schema"):
        _relative_name(evidence[name])
    for name in ("model_artifact", "model_manifest"):
        if evidence[name] is not None:
            _relative_name(evidence[name])
    expected_locators = (
        (None, None)
        if model["resolved"] == "market_only_v1"
        else ("model/model.json", "model/manifest.json")
    )
    if (evidence["model_artifact"], evidence["model_manifest"]) != expected_locators:
        raise _blocked("PREDICTION_BUNDLE_INVALID", field="result.evidence.model")
    status, stage, blocker, prediction = result["status"], result["blocker_stage"], result["blocker"], result["prediction"]
    if status == "PREDICTION_READY":
        chain = evidence["protocol_chain"]
        manual_chain_keys = {
            "request_id", "request_sha256", "claim_sha256", "attempt_sha256",
            "response_sha256", "receipt_sha256", "consume_sha256",
            "authenticated_receipt_sha256",
        }
        collector_chain_keys = {
            "protocol_kind", "collector_run_id", "capture_attempt_sha256",
            "collector_exact_receipt_sha256",
        }
        if isinstance(chain, Mapping) and set(chain) == manual_chain_keys:
            chain = _exact_fields(
                chain, manual_chain_keys, "result.evidence.protocol_chain"
            )
            if not isinstance(chain["request_id"], str) or not chain["request_id"]:
                raise _blocked(
                    "PREDICTION_BUNDLE_INVALID",
                    field="result.evidence.protocol_chain.request_id",
                )
            for name in set(chain) - {"request_id"}:
                _sha(chain[name], f"result.evidence.protocol_chain.{name}")
        elif isinstance(chain, Mapping) and set(chain) == collector_chain_keys:
            chain = _exact_fields(
                chain, collector_chain_keys, "result.evidence.protocol_chain"
            )
            if (
                chain["protocol_kind"] != "collector_exact_capture_v1"
                or not isinstance(chain["collector_run_id"], str)
                or not chain["collector_run_id"]
            ):
                raise _blocked(
                    "PREDICTION_BUNDLE_INVALID",
                    field="result.evidence.protocol_chain",
                )
            for name in (
                "capture_attempt_sha256",
                "collector_exact_receipt_sha256",
            ):
                _sha(chain[name], f"result.evidence.protocol_chain.{name}")
        else:
            raise _blocked(
                "PREDICTION_BUNDLE_INVALID",
                field="result.evidence.protocol_chain",
            )
        cutoff=_exact_fields(evidence["authenticated_cutoff"],{"history_seal_sha256","cutoff_timestamp","source_sha256","sealed_sha256"},"result.evidence.authenticated_cutoff")
        _timestamp(cutoff["cutoff_timestamp"],"result.evidence.authenticated_cutoff.cutoff_timestamp")
        for name in ("history_seal_sha256","source_sha256","sealed_sha256"):_sha(cutoff[name],f"result.evidence.authenticated_cutoff.{name}")
        if stage is not None or blocker is not None or not isinstance(prediction, Mapping):
            raise _blocked("PREDICTION_BUNDLE_INVALID", field="result.terminal")
        rows = prediction.get("predictions")
        if set(prediction) != {"predictions"} or not isinstance(rows, list) or len(rows) < 2:
            raise _blocked("PREDICTION_BUNDLE_INVALID", field="result.prediction")
        seen_box: set[int] = set(); seen_runner: set[str] = set(); total = 0.0
        previous_probability = math.inf
        for rank, row in enumerate(rows, 1):
            item = _exact_fields(row, {"rank", "box_number", "dog_name", "identity", "source_native_runner_id", "probability"}, "result.prediction.row")
            probability = item["probability"]
            if item["rank"] != rank or isinstance(item["box_number"], bool) or not isinstance(item["box_number"], int) or not isinstance(item["dog_name"], str) or not item["dog_name"] or isinstance(probability, bool) or not isinstance(probability, (int, float)) or not math.isfinite(probability) or not 0 <= probability <= 1 or probability > previous_probability:
                raise _blocked("PREDICTION_BUNDLE_INVALID", field="result.prediction.row")
            if item["box_number"] in seen_box or item["dog_name"] in seen_runner:
                raise _blocked("PREDICTION_BUNDLE_INVALID", field="result.prediction.row", reason="duplicate")
            seen_box.add(item["box_number"]); seen_runner.add(item["dog_name"]); total += probability
            previous_probability = probability
        if abs(total - 1.0) > 1e-12:
            raise _blocked("PREDICTION_BUNDLE_INVALID", field="result.prediction", reason="sum")
        if evidence["prediction_output_sha256"] != sha256_bytes(canonical_bytes(rows)):
            raise _blocked("PREDICTION_BUNDLE_INVALID", field="result.evidence.prediction_output_sha256")
    elif status == "PREDICTION_BLOCKED":
        if prediction is not None or evidence["prediction_output_sha256"] is not None:
            raise _blocked("PREDICTION_BUNDLE_INVALID", field="result.terminal")
        block = _exact_fields(blocker, {"code"}, "result.blocker")
        if BLOCKER_STAGE_BY_CODE.get(block["code"]) != stage:
            raise _blocked("PREDICTION_BUNDLE_INVALID", field="result.blocker.code")
    else:
        raise _blocked("PREDICTION_BUNDLE_INVALID", field="result.status")
    return dict(result)


def _validate_request_binding(raw: bytes, result: Mapping[str, Any]) -> dict[str, Any]:
    value=_canonical_json(raw, max_bytes=BUNDLE_CONTROL_MAX_BYTES, label="request")
    schema=value.get("schema_version") if isinstance(value,Mapping) else None
    fields={
            "schema_version", "prediction_id", "job_id", "race_query", "race_id", "jump_timestamp",
            "request_timestamp", "odds_source", "model", "config_sha256",
            "research_only", "runners", "runner_set_sha256",
        }
    if schema=="on_demand_prediction_request_v2":fields.add("operational_index_provenance")
    request = _exact_fields(value,fields,"request")
    if schema not in {"on_demand_prediction_request_v1","on_demand_prediction_request_v2"}:
        raise _blocked("PREDICTION_BUNDLE_IDENTITY_MISMATCH", field="request.schema")
    if schema=="on_demand_prediction_request_v2":validate_operational_index_provenance(request["operational_index_provenance"])
    _timestamp(request["jump_timestamp"], "request.jump_timestamp")
    _timestamp(request["request_timestamp"], "request.request_timestamp")
    _prediction_id(request["prediction_id"])
    _job_id(request["job_id"], "request.job_id")
    runners = canonical_runner_set(request["runners"], "request.runners")
    if request["runner_set_sha256"] != sealed_runner_set_sha256(result["race"], runners) or request["runner_set_sha256"] != result["evidence"]["runner_set_sha256"]:
        raise _blocked("PREDICTION_BUNDLE_IDENTITY_MISMATCH", field="request.runners")
    if result["status"] == "PREDICTION_READY":
        predicted = result["prediction"]["predictions"]
        projection = sorted((row["box_number"], row["dog_name"], row["identity"], row["source_native_runner_id"]) for row in predicted)
        expected_projection = [(row["box_number"], row["display_name"], row["identity"], row["source_native_runner_id"]) for row in runners]
        if projection != expected_projection:
            raise _blocked("PREDICTION_BUNDLE_IDENTITY_MISMATCH", field="request.runners")
    public_model = _exact_fields(
        request["model"],
        {
            "requested", "resolved", "alias_resolved", "model_sha256",
            "manifest_sha256", "schema_sha256",
        },
        "request.model",
    )
    sealed_model = result["model"]
    expected = {
        "requested": sealed_model["requested"],
        "resolved": sealed_model["resolved"],
        "alias_resolved": sealed_model["alias_resolved"],
        "model_sha256": sealed_model["artifact_sha256"],
        "manifest_sha256": sealed_model["artifact_manifest_sha256"],
        "schema_sha256": sealed_model["schema_sha256"],
    }
    if (
        dict(public_model) != expected
        or request["prediction_id"] != result["prediction_id"]
        or request["job_id"] != result["job_id"]
        or request["race_id"] != result["race"]["race_id"]
        or request["jump_timestamp"] != result["race"]["jump_timestamp"]
        or request["config_sha256"] != result["config"]["sha256"]
        or request["research_only"] is not True
    ):
        raise _blocked("PREDICTION_BUNDLE_IDENTITY_MISMATCH", field="request")
    return dict(request)


def _validate_authenticated_cutoff(
    contents: Mapping[str, bytes], result: Mapping[str, Any]
) -> None:
    required = {"features/history_seal.json", "features/sealed_history.db"}
    if not required.issubset(contents):
        raise _blocked("PREDICTION_BUNDLE_INVALID", reason="sealed_protocol_required")
    cutoff = result["evidence"]["authenticated_cutoff"]
    history_raw = contents["features/history_seal.json"]
    history = _canonical_json(
        history_raw, max_bytes=BUNDLE_CONTROL_MAX_BYTES, label="history_seal"
    )
    history_keys={"schema_version","cutoff_timestamp","source_sha256","sealed_sha256","target_race_id","cutoff_basis","safe_race_count","safe_dog_row_count","excluded_target_metadata_rows","excluded_at_or_after_cutoff_metadata_rows","excluded_ambiguous_date_metadata_rows","target_rows_materialized","at_or_after_cutoff_rows_materialized"}
    count_keys=("safe_race_count","safe_dog_row_count","excluded_target_metadata_rows","excluded_at_or_after_cutoff_metadata_rows","excluded_ambiguous_date_metadata_rows","target_rows_materialized","at_or_after_cutoff_rows_materialized")
    digests=(cutoff.get("history_seal_sha256"),cutoff.get("sealed_sha256"),cutoff.get("source_sha256"),history.get("sealed_sha256"),history.get("source_sha256"))
    jump=_timestamp(result["race"]["jump_timestamp"],"race.jump_timestamp")
    if set(history) != history_keys or history.get("schema_version") != "sealed_prediction_history_v1" or history.get("cutoff_basis") != "race_date_strictly_before_target_jump_date" or any(not isinstance(history.get(key),int) or isinstance(history.get(key),bool) or history[key] < 0 for key in count_keys) or any(not isinstance(value,str) or _SHA256_RE.fullmatch(value) is None for value in digests) or sha256_bytes(history_raw) != cutoff["history_seal_sha256"] or sha256_bytes(contents["features/sealed_history.db"]) != history.get("sealed_sha256") or history.get("sealed_sha256") != cutoff["sealed_sha256"] or history.get("source_sha256") != cutoff["source_sha256"] or _timestamp(history.get("cutoff_timestamp"),"history.cutoff_timestamp") != jump or _timestamp(cutoff.get("cutoff_timestamp"),"cutoff.cutoff_timestamp") != jump or history.get("target_rows_materialized") != 0 or history.get("at_or_after_cutoff_rows_materialized") != 0 or history.get("target_race_id") != result["race"]["race_id"]: raise _blocked("PREDICTION_BUNDLE_IDENTITY_MISMATCH", field="authenticated_cutoff")


def _validate_collector_exact_protocol(
    contents: Mapping[str, bytes], result: Mapping[str, Any]
) -> None:
    member_name = "protocol/collector_exact_receipt.json"
    protocol_members = {name for name in contents if name.startswith("protocol/")}
    if protocol_members != {member_name}:
        raise _blocked("PREDICTION_BUNDLE_INVALID", reason="sealed_protocol_required")
    chain = result["evidence"]["protocol_chain"]
    raw = contents[member_name]
    receipt = _canonical_json(
        raw,
        max_bytes=BUNDLE_CONTROL_MAX_BYTES,
        label="protocol.collector_exact_receipt",
    )
    receipt_keys = {
        "schema_version", "race_id", "collector_run_id", "captured_at",
        "emitted_at", "sealed_handoff", "artifacts", "form_name",
    }
    handoff_keys = {
        "schema_version", "race_id", "race", "append_timestamp",
        "runner_set_sha256", "source_report_sha256", "source_form_sha256",
        "source_sidecar_sha256", "capture_attempt_sha256",
        "append_report_sha256",
    }
    if (
        set(receipt) != receipt_keys
        or receipt.get("schema_version") != "collector-exact-capture-receipt-v1"
        or receipt.get("race_id") != result["race"]["race_id"]
        or receipt.get("collector_run_id") != chain["collector_run_id"]
        or sha256_bytes(raw) != chain["collector_exact_receipt_sha256"]
    ):
        raise _blocked(
            "PREDICTION_BUNDLE_IDENTITY_MISMATCH",
            field="protocol.collector_exact_receipt",
        )
    handoff = receipt.get("sealed_handoff")
    result_race = {
        key: result["race"][key]
        for key in (
            "race_id", "url", "venue", "race_number", "race_date",
            "jump_timestamp",
        )
    }
    if (
        not isinstance(handoff, Mapping)
        or set(handoff) != handoff_keys
        or handoff.get("schema_version") != "on_demand_verified_collector_capture_v2"
        or handoff.get("race_id") != result["race"]["race_id"]
        or handoff.get("race") != result_race
        or handoff.get("runner_set_sha256")
        != runner_set_sha256(result["prediction"]["predictions"])
        or handoff.get("capture_attempt_sha256")
        != chain["capture_attempt_sha256"]
    ):
        raise _blocked(
            "PREDICTION_BUNDLE_IDENTITY_MISMATCH",
            field="protocol.collector_exact_receipt.handoff",
        )
    captured = _timestamp(receipt.get("captured_at"), "protocol.captured_at")
    emitted = _timestamp(receipt.get("emitted_at"), "protocol.emitted_at")
    jump = _timestamp(result["race"]["jump_timestamp"], "race.jump_timestamp")
    if (
        receipt.get("captured_at") != handoff.get("append_timestamp")
        or not captured <= emitted
        or not captured < jump
    ):
        raise _blocked(
            "PREDICTION_BUNDLE_IDENTITY_MISMATCH",
            field="protocol.collector_exact_receipt.time",
        )
    artifacts = receipt.get("artifacts")
    labels = ("report", "form", "sidecar")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(labels):
        raise _blocked(
            "PREDICTION_BUNDLE_IDENTITY_MISMATCH",
            field="protocol.collector_exact_receipt.artifacts",
        )
    external_paths: dict[str, str] = {}
    for label in labels:
        artifact = artifacts[label]
        if not isinstance(artifact, Mapping) or set(artifact) != {"path", "sha256"}:
            raise _blocked(
                "PREDICTION_BUNDLE_IDENTITY_MISMATCH",
                field="protocol.collector_exact_receipt.artifacts",
            )
        external_paths[label] = _relative_name(artifact.get("path"))
        if artifact.get("sha256") != handoff.get(f"source_{label}_sha256"):
            raise _blocked(
                "PREDICTION_BUNDLE_IDENTITY_MISMATCH",
                field="protocol.collector_exact_receipt.artifacts",
            )
    form_name = _relative_name(receipt.get("form_name"))
    if (
        Path(form_name).name != form_name
        or form_name != Path(external_paths["form"]).name
        or len(set(external_paths.values())) != len(external_paths)
    ):
        raise _blocked(
            "PREDICTION_BUNDLE_IDENTITY_MISMATCH",
            field="protocol.collector_exact_receipt.artifacts",
        )
    bundle_sources = {
        "report": "source/capture.json",
        "form": f"source/{form_name}",
        "sidecar": f"source/{form_name}.metadata.json",
    }
    if (
        len(set(bundle_sources.values())) != len(bundle_sources)
        or not set(bundle_sources.values()).issubset(contents)
    ):
        raise _blocked("PREDICTION_BUNDLE_INVALID", reason="sealed_protocol_required")
    if any(
        sha256_bytes(contents[bundle_sources[label]])
        != artifacts[label]["sha256"]
        for label in labels
    ):
        raise _blocked(
            "PREDICTION_BUNDLE_IDENTITY_MISMATCH",
            field="protocol.collector_exact_receipt.source",
        )
    source_report = _canonical_json(
        contents[bundle_sources["report"]],
        max_bytes=BUNDLE_CONTROL_MAX_BYTES,
        label="source.capture",
    )
    source_plan = (
        source_report.get("source_plan_item")
        if isinstance(source_report, Mapping)
        else None
    )
    source_attempt = (
        source_report.get("source_attempt")
        if isinstance(source_report, Mapping)
        else None
    )
    source_race_id = (
        source_report.get("source_race_id")
        if isinstance(source_report, Mapping)
        else None
    )
    adapted_attempt = (
        {**source_attempt, "race_id": result["race"]["race_id"]}
        if isinstance(source_attempt, Mapping)
        else None
    )
    if (
        not isinstance(source_report, Mapping)
        or set(source_report) != {
            "schema_version", "collector_run_id", "generated_at", "race_id",
            "source_race_id", "source_plan_item", "source_attempt", "attempts",
        }
        or source_report.get("schema_version")
        != "collector_exact_capture_source_v1"
        or source_report.get("race_id") != result["race"]["race_id"]
        or source_report.get("collector_run_id") != chain["collector_run_id"]
        or source_report.get("generated_at") != receipt.get("emitted_at")
        or not isinstance(source_race_id, str)
        or not source_race_id
        or not isinstance(source_plan, Mapping)
        or source_plan.get("schema_version")
        != "autonomous_live_odds_capture_plan_item_v1"
        or source_plan.get("status") != "READY_TO_CAPTURE"
        or source_plan.get("race_id") != source_race_id
        or not isinstance(source_attempt, Mapping)
        or source_attempt.get("schema_version")
        != "autonomous_live_odds_capture_attempt_v1"
        or source_attempt.get("race_id") != source_race_id
        or source_attempt.get("status") != "APPENDED"
        or source_report.get("attempts") != [adapted_attempt]
        or sha256_bytes(canonical_bytes(adapted_attempt))
        != chain["capture_attempt_sha256"]
        or not isinstance(source_attempt.get("append_report"), Mapping)
        or sha256_bytes(canonical_bytes(source_attempt["append_report"]))
        != handoff.get("append_report_sha256")
    ):
        raise _blocked(
            "PREDICTION_BUNDLE_IDENTITY_MISMATCH",
            field="protocol.collector_exact_receipt.source_report",
        )


def _validate_sealed_protocol(contents: Mapping[str, bytes], result: Mapping[str, Any]) -> None:
    chain = result["evidence"]["protocol_chain"]
    if chain.get("protocol_kind") == "collector_exact_capture_v1":
        _validate_collector_exact_protocol(contents, result)
        _validate_authenticated_cutoff(contents, result)
        return
    names = ("request", "claim", "attempt", "response", "receipt", "consume", "authenticated_receipt")
    protocol_members = {f"protocol/{name}.json" for name in names}
    required = protocol_members | {"features/history_seal.json", "features/sealed_history.db"}
    if (
        {name for name in contents if name.startswith("protocol/")}
        != protocol_members
        or not required.issubset(contents)
    ):
        raise _blocked("PREDICTION_BUNDLE_INVALID", reason="sealed_protocol_required")
    chain = result["evidence"]["protocol_chain"]
    values = {name: _canonical_json(contents[f"protocol/{name}.json"], max_bytes=BUNDLE_CONTROL_MAX_BYTES, label=f"protocol.{name}") for name in names}
    expected_keys = {
        "request": {"schema_version","request_id","created_at","expires_at","race","expected_runners","expected_runner_set_sha256","requested_output","research_only","attempt_authority"},
        "claim": {"schema_version","request_id","request_sha256","collector_run_id","claimed_at","safe_boundary"},
        "attempt": {"schema_version","request_id","request_sha256","claim_sha256","collector_run_id","attempt_number","started_at"},
        "response": {"schema_version","request_id","request_sha256","claim_sha256","attempt_sha256","race","status","reason","responded_at","receipt"},
        "receipt": {"schema_version","request_id","request_sha256","race","runners","runner_set_sha256","captured_at","emitted_at","source_evidence","sealed_handoff"},
        "consume": {"schema_version","request_id","response_sha256","status","consumed_at","consume_once"},
        "authenticated_receipt": {"schema_version","race_id","request_id","form_name","receipt","artifacts"},
    }
    schemas = {
        "request": "manual-prediction-collector-request-v1",
        "claim": "manual-prediction-collector-claim-v1",
        "attempt": "manual-prediction-collector-attempt-v1",
        "response": "manual-prediction-collector-response-v1",
        "receipt": "manual-prediction-collector-receipt-v1",
        "consume": "manual-prediction-collector-consume-v1",
        "authenticated_receipt": "manual-prediction-exact-receipt-index-v1",
    }
    for name in names:
        hash_name = "authenticated_receipt_sha256" if name == "authenticated_receipt" else f"{name}_sha256"
        if set(values[name]) != expected_keys[name] or values[name].get("schema_version") != schemas[name] or sha256_bytes(contents[f"protocol/{name}.json"]) != chain[hash_name] or values[name].get("request_id") != chain["request_id"]: raise _blocked("PREDICTION_BUNDLE_IDENTITY_MISMATCH", field=f"protocol.{name}")
    request, claim, attempt, response, receipt, consume, exact = (values[name] for name in names)
    result_race={key:result["race"][key] for key in ("race_id","url","venue","race_number","race_date","jump_timestamp")}
    if request.get("race") != result_race or request.get("expected_runner_set_sha256") != receipt.get("runner_set_sha256") or request.get("expected_runners") != receipt.get("runners") or request.get("attempt_authority") != "one_attempt" or request.get("research_only") is not True: raise _blocked("PREDICTION_BUNDLE_IDENTITY_MISMATCH", field="protocol.request")
    created=_timestamp(request.get("created_at"),"protocol.request.created_at"); expires=_timestamp(request.get("expires_at"),"protocol.request.expires_at"); claimed=_timestamp(claim.get("claimed_at"),"protocol.claim.claimed_at"); started=_timestamp(attempt.get("started_at"),"protocol.attempt.started_at")
    if claim.get("request_sha256") != chain["request_sha256"] or attempt.get("request_sha256") != chain["request_sha256"] or attempt.get("claim_sha256") != chain["claim_sha256"] or response.get("request_sha256") != chain["request_sha256"] or response.get("claim_sha256") != chain["claim_sha256"] or response.get("attempt_sha256") != chain["attempt_sha256"] or claim.get("safe_boundary") is not True or attempt.get("collector_run_id") != claim.get("collector_run_id") or attempt.get("attempt_number") != 1 or not created <= claimed <= started < expires: raise _blocked("PREDICTION_BUNDLE_IDENTITY_MISMATCH", field="protocol.hash_chain")
    reference = response.get("receipt")
    sealed=receipt.get("sealed_handoff"); source=receipt.get("source_evidence")
    responded=_timestamp(response.get("responded_at"),"protocol.response.responded_at"); captured=_timestamp(receipt.get("captured_at"),"protocol.receipt.captured_at"); emitted=_timestamp(receipt.get("emitted_at"),"protocol.receipt.emitted_at"); consumed=_timestamp(consume.get("consumed_at"),"protocol.consume.consumed_at")
    if not isinstance(reference,Mapping) or reference.get("sha256") != chain["receipt_sha256"] or receipt.get("request_sha256") != chain["request_sha256"] or receipt.get("race") != request.get("race") or consume.get("response_sha256") != chain["response_sha256"] or consume.get("consume_once") is not True or consume.get("status") != response.get("status") or response.get("race") != request.get("race") or response.get("status") != "RECEIPT_READY" or response.get("reason") is not None or reference != {"schema_version":"manual-prediction-collector-receipt-v1","path":f"receipts/{chain['request_id']}.json","sha256":chain["receipt_sha256"]} or not started <= captured <= emitted == responded <= consumed < expires: raise _blocked("PREDICTION_BUNDLE_IDENTITY_MISMATCH", field="protocol.receipt_chain")
    if not isinstance(sealed,Mapping) or exact.get("race_id") != result["race"]["race_id"] or exact.get("receipt",{}).get("sha256") != chain["receipt_sha256"] or not isinstance(source,Mapping) or sealed.get("race") != request.get("race") or sealed.get("race_id") != result["race"]["race_id"] or sealed.get("runner_set_sha256") != receipt.get("runner_set_sha256") or any(source.get(key)!=sealed.get(key) for key in ("source_report_sha256","source_form_sha256","source_sidecar_sha256","capture_attempt_sha256","append_report_sha256")): raise _blocked("PREDICTION_BUNDLE_IDENTITY_MISMATCH", field="protocol.sealed_handoff")
    artifacts=exact.get("artifacts")
    artifact_labels=("report","form","sidecar")
    if not isinstance(artifacts,Mapping) or set(artifacts) != set(artifact_labels): raise _blocked("PREDICTION_BUNDLE_IDENTITY_MISMATCH", field="protocol.authenticated_receipt")
    artifact_paths=[]
    for key in artifact_labels:
        member=artifacts[key]
        if not isinstance(member,Mapping) or set(member)!={"path","sha256"}: raise _blocked("PREDICTION_BUNDLE_IDENTITY_MISMATCH", field="protocol.authenticated_receipt")
        path=_relative_name(member.get("path"))
        if len(path)>512 or any(ord(character)<32 or ord(character)==127 for character in path): raise _blocked("PREDICTION_BUNDLE_IDENTITY_MISMATCH", field="protocol.authenticated_receipt.path")
        artifact_paths.append(path)
    if exact.get("receipt") != reference or len(set(artifact_paths)) != len(artifact_paths) or any(artifacts[key].get("sha256") != sealed.get(f"source_{key}_sha256") for key in artifact_labels) or exact.get("form_name") != Path(artifacts["form"]["path"]).name: raise _blocked("PREDICTION_BUNDLE_IDENTITY_MISMATCH", field="protocol.authenticated_receipt")
    _validate_authenticated_cutoff(contents, result)


class PredictionBlocked(RuntimeError):
    """Fail-closed operator result with one stable blocker code."""

    def __init__(self, code: str, **details: Any) -> None:
        super().__init__(code)
        self.code = code
        self.details = details


@dataclass(frozen=True)
class ModelIdentity:
    requested: str
    resolved: str
    alias: bool
    model_path: Path | None
    manifest_path: Path | None
    model_sha256: str | None
    manifest_sha256: str | None
    schema_path: Path
    schema_sha256: str


@dataclass
class Dependencies:
    schedule: Callable[
        [datetime, float, Path, Path, int], Sequence[Mapping[str, Any]]
    ]
    seal_features: Callable[..., Mapping[str, Path]]
    score_residual: Callable[..., Mapping[str, Any]]
    now: Callable[[], datetime]
    capture_one: Callable[..., Mapping[str, Any]] | None = None
    monotonic: Callable[[], float] = time.monotonic


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _contains_outcome(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in OUTCOME_KEYS or _contains_outcome(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_outcome(item) for item in value)
    return False


def _write_canonical(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise PredictionBlocked("WRITE_TARGET_EXISTS", path=str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        try:
            handle = os.fdopen(descriptor, "wb")
        except Exception:
            os.close(descriptor)
            raise
        with handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _copy_exact(source: Path, target: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise PredictionBlocked("SOURCE_FILE_UNSAFE", path=str(source))
    if target.exists() or target.is_symlink():
        raise PredictionBlocked("WRITE_TARGET_EXISTS", path=str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(target, flags, 0o600)
    try:
        try:
            writer = os.fdopen(descriptor, "wb")
        except Exception:
            os.close(descriptor)
            raise
        with source.open("rb") as reader, writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
    except Exception:
        target.unlink(missing_ok=True)
        raise


def write_exact_bytes(target: Path, value: bytes) -> None:
    if target.exists() or target.is_symlink():
        raise PredictionBlocked("WRITE_TARGET_EXISTS", path=str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(target, flags, 0o600)
    try:
        try:
            handle = os.fdopen(descriptor, "wb")
        except Exception:
            os.close(descriptor)
            raise
        with handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        target.unlink(missing_ok=True)
        raise


def create_bundle(output_root: Path, now: datetime) -> Path:
    root = output_root.resolve()
    if output_root.is_symlink() or (root.exists() and not root.is_dir()):
        raise PredictionBlocked("OUTPUT_ROOT_UNSAFE", path=str(output_root))
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    mode = stat.S_IMODE(root.stat().st_mode)
    if mode & 0o022:
        raise PredictionBlocked("OUTPUT_ROOT_WRITABLE_BY_OTHERS", path=str(root))
    bundle = (
        root / f"prediction_{now.strftime('%Y%m%dT%H%M%S%f%z')}_{uuid.uuid4().hex[:12]}"
    )
    bundle.mkdir(mode=0o700)
    return bundle


def resolve_model(requested: str) -> ModelIdentity:
    normalized = requested.strip().lower()
    resolved = MODEL_ALIASES.get(normalized)
    if resolved is None:
        raise PredictionBlocked("MODEL_UNSUPPORTED", requested=requested)
    schema_path = SCHEMA_FILES[resolved]
    if not schema_path.is_file():
        raise PredictionBlocked("MODEL_SCHEMA_MISSING", path=str(schema_path))
    artifact_dir = MODEL_FILES[resolved]
    model_path = artifact_dir / "model.json" if artifact_dir else None
    manifest_path = artifact_dir / "manifest.json" if artifact_dir else None
    if artifact_dir and (not model_path.is_file() or not manifest_path.is_file()):
        raise PredictionBlocked("MODEL_ARTIFACT_MISSING", model=resolved)
    return ModelIdentity(
        requested=requested,
        resolved=resolved,
        alias=normalized != resolved,
        model_path=model_path,
        manifest_path=manifest_path,
        model_sha256=sha256_file(model_path) if model_path else None,
        manifest_sha256=sha256_file(manifest_path) if manifest_path else None,
        schema_path=schema_path,
        schema_sha256=sha256_file(schema_path),
    )


def _validate_simple_schema(
    value: Any, schema: Mapping[str, Any], label: str = "config"
) -> None:
    expected_type = schema.get("type")
    type_map = {
        "object": dict,
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
    }
    if expected_type in type_map and (
        not isinstance(value, type_map[expected_type])
        or isinstance(value, bool)
        and expected_type in {"integer", "number"}
    ):
        raise PredictionBlocked(
            "CONFIG_SCHEMA_MISMATCH", field=label, reason=f"type:{expected_type}"
        )
    if isinstance(value, Mapping):
        required = schema.get("required") or []
        missing = sorted(set(required) - set(value))
        if missing:
            raise PredictionBlocked(
                "CONFIG_SCHEMA_MISMATCH",
                field=label,
                reason=f"missing:{','.join(missing)}",
            )
        properties = schema.get("properties") or {}
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise PredictionBlocked(
                    "CONFIG_SCHEMA_MISMATCH",
                    field=label,
                    reason=f"extra:{','.join(extra)}",
                )
        for key, child in properties.items():
            if key in value:
                _validate_simple_schema(value[key], child, f"{label}.{key}")
    if "const" in schema and value != schema["const"]:
        raise PredictionBlocked("CONFIG_SCHEMA_MISMATCH", field=label, reason="const")
    if "enum" in schema and value not in schema["enum"]:
        raise PredictionBlocked("CONFIG_SCHEMA_MISMATCH", field=label, reason="enum")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise PredictionBlocked(
                "CONFIG_SCHEMA_MISMATCH", field=label, reason="minimum"
            )
        if "maximum" in schema and value > schema["maximum"]:
            raise PredictionBlocked(
                "CONFIG_SCHEMA_MISMATCH", field=label, reason="maximum"
            )


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON numeric constant: {value}")


def load_config(path: Path, model: ModelIdentity) -> tuple[dict[str, Any], str, bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw, parse_constant=_reject_nonfinite_json_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PredictionBlocked("CONFIG_INVALID_JSON", path=str(path)) from exc
    if not isinstance(value, dict) or canonical_bytes(value) != raw:
        raise PredictionBlocked("CONFIG_NOT_CANONICAL", path=str(path))
    if value.get("model") != model.resolved:
        raise PredictionBlocked(
            "MODEL_CONFIG_MISMATCH",
            requested=model.resolved,
            configured=value.get("model"),
        )
    try:
        schema = json.loads(model.schema_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise PredictionBlocked(
            "MODEL_SCHEMA_INVALID", path=str(model.schema_path)
        ) from exc
    _validate_simple_schema(value, schema)
    return value, sha256_bytes(raw), raw


def runner_set_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    keys = sorted(
        f"{int(row['box_number'])}:{str(row.get('identity') or row.get('dog_name') or '').strip().upper()}"
        for row in rows
    )
    if len(keys) < 2 or len(keys) != len(set(keys)):
        raise PredictionBlocked("RUNNER_SET_AMBIGUOUS")
    return sha256_bytes(canonical_bytes(keys))


def normalize_validation_receipt(
    *,
    race_id: str,
    captured_at: datetime,
    validation: Mapping[str, Any],
    source_kind: str,
) -> dict[str, Any]:
    if validation.get("status") != "PASS" or validation.get("reasons") not in (
        None,
        [],
    ):
        raise PredictionBlocked(
            "MARKET_UNAVAILABLE", reasons=validation.get("reasons") or []
        )
    markets: dict[str, list[dict[str, Any]]] = {}
    for market, key in (("win", "accepted_rows"), ("place", "accepted_place_rows")):
        raw_rows = validation.get(key)
        if not isinstance(raw_rows, list) or not raw_rows:
            raise PredictionBlocked("MARKET_UNAVAILABLE", market=market)
        rows: list[dict[str, Any]] = []
        for raw in raw_rows:
            if not isinstance(raw, Mapping):
                raise PredictionBlocked("RUNNER_SET_AMBIGUOUS", market=market)
            try:
                box = int(raw.get("box_number"))
                odds = float(raw.get("odds_decimal"))
            except (TypeError, ValueError) as exc:
                raise PredictionBlocked("RUNNER_SET_AMBIGUOUS", market=market) from exc
            identity = (
                str(raw.get("identity") or raw.get("dog_name") or "").strip().upper()
            )
            dog_name = str(
                raw.get("dog_name") or raw.get("dog_clean_name") or ""
            ).strip()
            if (
                not 1 <= box <= 10
                or not identity
                or not dog_name
                or not math.isfinite(odds)
                or odds <= 1
            ):
                raise PredictionBlocked("RUNNER_SET_AMBIGUOUS", market=market)
            rows.append(
                {
                    "box_number": box,
                    "dog_name": dog_name,
                    "identity": identity,
                    "odds_decimal": odds,
                }
            )
        rows.sort(key=lambda row: (row["box_number"], row["identity"]))
        if len({(row["box_number"], row["identity"]) for row in rows}) != len(rows):
            raise PredictionBlocked("RUNNER_SET_AMBIGUOUS", market=market)
        markets[market] = rows
    win_keys = {(row["box_number"], row["identity"]) for row in markets["win"]}
    place_keys = {(row["box_number"], row["identity"]) for row in markets["place"]}
    if win_keys != place_keys:
        raise PredictionBlocked(
            "RUNNER_SET_AMBIGUOUS", reason="win_place_runner_mismatch"
        )
    return {
        "schema_version": "on_demand_odds_receipt_v1",
        "race_id": race_id,
        "captured_at": captured_at.isoformat(),
        "source_kind": source_kind,
        "source_url": validation.get("source_url"),
        "markets": markets,
        "runner_set_sha256": runner_set_sha256(markets["win"]),
    }


def receipt_from_handoff(
    receipt: Mapping[str, Any], *, current_time: datetime, max_age_seconds: int
) -> tuple[dict[str, Any], bytes, bytes, bytes]:
    try:
        captured_at = datetime.fromisoformat(str(receipt["append_timestamp"]))
        report_raw = bytes(receipt["_report_bytes"])
        form_raw = bytes(receipt["_form_bytes"])
        sidecar_raw = bytes(receipt["_sidecar_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PredictionBlocked("RECEIPT_INVALID") from exc
    if captured_at.tzinfo is None or captured_at.utcoffset() is None:
        raise PredictionBlocked("RECEIPT_INVALID", reason="timestamp_timezone_missing")
    age = (current_time - captured_at).total_seconds()
    if age < 0 or age > max_age_seconds:
        raise PredictionBlocked("RECEIPT_STALE", age_seconds=age)
    try:
        report = json.loads(report_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PredictionBlocked("RECEIPT_INVALID") from exc
    if _contains_outcome(report):
        raise PredictionBlocked("RECEIPT_CONTAINS_OUTCOME")
    attempts = report.get("attempts") if isinstance(report, Mapping) else None
    matches = [
        row
        for row in attempts or []
        if isinstance(row, Mapping)
        and row.get("race_id") == receipt.get("race_id")
        and row.get("status") == "APPENDED"
        and isinstance(row.get("validation"), Mapping)
        and row["validation"].get("status") == "PASS"
    ]
    if len(matches) != 1:
        raise PredictionBlocked("RECEIPT_AMBIGUOUS")
    normalized = normalize_validation_receipt(
        race_id=str(receipt.get("race_id")),
        captured_at=captured_at,
        validation=matches[0]["validation"],
        source_kind="verified_autonomous_receipt",
    )
    expected_hashes = {
        "source_report_sha256": sha256_bytes(report_raw),
        "source_form_sha256": sha256_bytes(form_raw),
        "source_sidecar_sha256": sha256_bytes(sidecar_raw),
    }
    if any(receipt.get(key) != value for key, value in expected_hashes.items()):
        raise PredictionBlocked("RECEIPT_TAMPERED")
    normalized["source_hashes"] = expected_hashes
    normalized["handoff"] = {
        key: value for key, value in receipt.items() if not str(key).startswith("_")
    }
    return normalized, report_raw, form_raw, sidecar_raw


def _table_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')]


def _history_sidecars_clear(source: Path) -> bool:
    for suffix in ("-wal", "-journal"):
        path = Path(f"{source}{suffix}")
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            return False
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink() or metadata.st_size:
            return False
    return True


def _remove_history_work_file(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        return type(exc).__name__
    return None


def _verified_history_snapshot(source: Path, directory: Path) -> tuple[Path, str]:
    """Copy one checkpointed SQLite image without writing beside the source."""

    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PredictionBlocked("HISTORY_SEAL_WRITE_FAILED") from exc
    if not _history_sidecars_clear(source):
        raise PredictionBlocked("HISTORY_DATABASE_BUSY")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        source_fd = os.open(source, flags)
    except OSError as exc:
        raise PredictionBlocked("HISTORY_DATABASE_UNAVAILABLE") from exc
    snapshot_fd = -1
    snapshot_path: Path | None = None
    try:
        try:
            before = os.fstat(source_fd)
        except OSError as exc:
            raise PredictionBlocked("HISTORY_DATABASE_UNAVAILABLE") from exc
        if not stat.S_ISREG(before.st_mode):
            raise PredictionBlocked("HISTORY_DATABASE_UNAVAILABLE")
        try:
            snapshot_fd, raw_path = tempfile.mkstemp(
                prefix=".history-source-", suffix=".db", dir=directory
            )
            snapshot_path = Path(raw_path)
            os.fchmod(snapshot_fd, 0o600)
        except OSError as exc:
            raise PredictionBlocked("HISTORY_SEAL_WRITE_FAILED") from exc
        copied = hashlib.sha256()
        while True:
            try:
                chunk = os.read(source_fd, 1024 * 1024)
            except OSError as exc:
                raise PredictionBlocked("HISTORY_DATABASE_UNAVAILABLE") from exc
            if not chunk:
                break
            copied.update(chunk)
            view = memoryview(chunk)
            while view:
                try:
                    written = os.write(snapshot_fd, view)
                except OSError as exc:
                    raise PredictionBlocked("HISTORY_SEAL_WRITE_FAILED") from exc
                if written <= 0:
                    raise PredictionBlocked("HISTORY_SEAL_WRITE_FAILED")
                view = view[written:]
        try:
            os.fsync(snapshot_fd)
        except OSError as exc:
            raise PredictionBlocked("HISTORY_SEAL_WRITE_FAILED") from exc
        try:
            os.lseek(source_fd, 0, os.SEEK_SET)
        except OSError as exc:
            raise PredictionBlocked("HISTORY_DATABASE_UNAVAILABLE") from exc
        current = hashlib.sha256()
        while True:
            try:
                chunk = os.read(source_fd, 1024 * 1024)
            except OSError as exc:
                raise PredictionBlocked("HISTORY_DATABASE_UNAVAILABLE") from exc
            if not chunk:
                break
            current.update(chunk)
        try:
            after = os.fstat(source_fd)
        except OSError as exc:
            raise PredictionBlocked("HISTORY_DATABASE_CHANGED") from exc
        stable_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        sidecars_clear = _history_sidecars_clear(source)
        try:
            named = source.lstat()
        except OSError as exc:
            raise PredictionBlocked("HISTORY_DATABASE_CHANGED") from exc
        if (
            not stable_identity
            or copied.digest() != current.digest()
            or not sidecars_clear
            or not stat.S_ISREG(named.st_mode)
            or (named.st_dev, named.st_ino) != (after.st_dev, after.st_ino)
        ):
            raise PredictionBlocked("HISTORY_DATABASE_CHANGED")
        return snapshot_path, copied.hexdigest()
    except PredictionBlocked as exc:
        cleanup_error = _remove_history_work_file(snapshot_path)
        if cleanup_error is not None:
            exc.details["snapshot_cleanup_error"] = cleanup_error
        raise
    finally:
        active_error = sys.exc_info()[1]
        close_errors: dict[str, str] = {}
        if snapshot_fd >= 0:
            try:
                os.close(snapshot_fd)
            except OSError as exc:
                close_errors["snapshot_descriptor_close_error"] = type(exc).__name__
        try:
            os.close(source_fd)
        except OSError as exc:
            close_errors["source_descriptor_close_error"] = type(exc).__name__
        if close_errors:
            if isinstance(active_error, PredictionBlocked):
                active_error.details.update(close_errors)
            elif active_error is None:
                cleanup_error = _remove_history_work_file(snapshot_path)
                if cleanup_error is not None:
                    close_errors["snapshot_cleanup_error"] = cleanup_error
                code = (
                    "HISTORY_SEAL_WRITE_FAILED"
                    if "snapshot_descriptor_close_error" in close_errors
                    else "HISTORY_DATABASE_UNAVAILABLE"
                )
                raise PredictionBlocked(code, **close_errors)


def seal_history_database(
    *,
    source: Path,
    target: Path,
    target_race_id: str,
    cutoff: datetime,
    runner_names: Sequence[str],
) -> dict[str, Any]:
    if not source.is_file() or source.is_symlink() or target.exists():
        raise PredictionBlocked("HISTORY_DATABASE_UNAVAILABLE")
    snapshot, source_sha256 = _verified_history_snapshot(source, target.parent)
    source_db: sqlite3.Connection | None = None
    target_db: sqlite3.Connection | None = None
    sqlite_phase = "source"
    completed = False
    failure: Exception | None = None
    try:
        source_uri = f"file:{snapshot.resolve()}?mode=ro&immutable=1"
        source_db = sqlite3.connect(source_uri, uri=True)
        source_db.row_factory = sqlite3.Row
        source_db.execute("PRAGMA query_only=ON")
        source_db.execute("BEGIN")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PredictionBlocked("HISTORY_SEAL_WRITE_FAILED") from exc
        sqlite_phase = "target"
        target_db = sqlite3.connect(target)
        sqlite_phase = "source"
        if source_db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise PredictionBlocked("HISTORY_DATABASE_INTEGRITY_FAILED")
        schemas: dict[str, str] = {}
        for table in ("race_metadata", "dog_race_data"):
            row = source_db.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if row is None or not row[0]:
                raise PredictionBlocked("HISTORY_SCHEMA_MISSING", table=table)
            schemas[table] = str(row[0])
            sqlite_phase = "target"
            target_db.execute(schemas[table])
            sqlite_phase = "source"
        race_columns = _table_columns(source_db, "race_metadata")
        dog_columns = _table_columns(source_db, "dog_race_data")
        if (
            not {"race_id", "race_date"}.issubset(race_columns)
            or "race_id" not in dog_columns
        ):
            raise PredictionBlocked("HISTORY_SCHEMA_AMBIGUOUS")
        duplicate_ids = source_db.execute(
            "SELECT race_id FROM race_metadata WHERE race_id IS NOT NULL GROUP BY race_id HAVING COUNT(*) != 1 LIMIT 1"
        ).fetchone()
        if duplicate_ids:
            raise PredictionBlocked(
                "HISTORY_IDENTITY_AMBIGUOUS", race_id=duplicate_ids[0]
            )
        cutoff_date = cutoff.date().isoformat()
        metadata_rows = [
            dict(row)
            for row in source_db.execute(
                'SELECT "race_id", "race_date" FROM race_metadata'
            )
        ]
        safe_ids: set[str] = set()
        excluded_target = excluded_at_or_after = ambiguous_dates = 0
        for row in metadata_rows:
            race_id = str(row.get("race_id") or "")
            raw_date = str(row.get("race_date") or "")
            if race_id == target_race_id:
                excluded_target += 1
                continue
            try:
                parsed = date.fromisoformat(raw_date[:10])
            except ValueError:
                ambiguous_dates += 1
                continue
            if parsed.isoformat() >= cutoff_date:
                excluded_at_or_after += 1
                continue
            if not race_id:
                ambiguous_dates += 1
                continue
            safe_ids.add(race_id)

        def rows_for_safe_ids(table: str) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            ordered = sorted(safe_ids)
            for offset in range(0, len(ordered), 500):
                chunk = ordered[offset : offset + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows.extend(
                    dict(row)
                    for row in source_db.execute(
                        f'SELECT * FROM "{table}" WHERE "race_id" IN ({placeholders})',
                        tuple(chunk),
                    )
                )
            return rows

        safe_metadata = rows_for_safe_ids("race_metadata")
        normalized_names = {
            name.strip().upper() for name in runner_names if name.strip()
        }
        name_column = (
            "dog_clean_name"
            if "dog_clean_name" in dog_columns
            else "dog_name"
            if "dog_name" in dog_columns
            else None
        )
        if name_column is None:
            raise PredictionBlocked(
                "HISTORY_SCHEMA_AMBIGUOUS", reason="dog_name_missing"
            )
        relevant_ambiguous: list[str] = []
        if normalized_names:
            relevant_history = source_db.execute(
                f'''SELECT DISTINCT dr.race_id, rm.race_date
                    FROM dog_race_data dr
                    LEFT JOIN race_metadata rm ON rm.race_id = dr.race_id
                    WHERE UPPER(TRIM(COALESCE(dr."{name_column}", ''))) IN ({",".join("?" for _ in normalized_names)})''',
                tuple(sorted(normalized_names)),
            )
            for history_row in relevant_history:
                history_race_id = str(history_row[0] or "")
                raw_history_date = str(history_row[1] or "")
                try:
                    date.fromisoformat(raw_history_date[:10])
                except ValueError:
                    relevant_ambiguous.append(history_race_id or "<missing>")
        if relevant_ambiguous:
            raise PredictionBlocked(
                "HISTORY_CUTOFF_AMBIGUOUS",
                race_ids=sorted(relevant_ambiguous),
                row_count=len(relevant_ambiguous),
            )

        def insert_rows(
            table: str, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]
        ) -> None:
            if not rows:
                return
            column_sql = ",".join(f'"{column}"' for column in columns)
            placeholders = ",".join("?" for _ in columns)
            target_db.executemany(
                f'INSERT INTO "{table}" ({column_sql}) VALUES ({placeholders})',
                [tuple(row.get(column) for column in columns) for row in rows],
            )

        dog_rows = rows_for_safe_ids("dog_race_data")
        sqlite_phase = "target"
        insert_rows("race_metadata", race_columns, safe_metadata)
        insert_rows("dog_race_data", dog_columns, dog_rows)
        target_db.commit()
        target_db.execute("PRAGMA optimize")
    except sqlite3.Error as exc:
        code = (
            "HISTORY_SEAL_WRITE_FAILED"
            if sqlite_phase == "target"
            else "HISTORY_DATABASE_INTEGRITY_FAILED"
        )
        failure = PredictionBlocked(code, error=type(exc).__name__)
        failure.__cause__ = exc
    except Exception as exc:
        failure = exc
    finally:
        if target_db is not None:
            try:
                target_db.close()
            except sqlite3.Error as exc:
                detail = {"target_connection_close_error": type(exc).__name__}
                if failure is None:
                    failure = PredictionBlocked(
                        "HISTORY_SEAL_WRITE_FAILED", **detail
                    )
                elif isinstance(failure, PredictionBlocked):
                    failure.details.update(detail)
        if source_db is not None:
            try:
                source_db.close()
            except sqlite3.Error as exc:
                detail = {"source_connection_close_error": type(exc).__name__}
                if failure is None:
                    failure = PredictionBlocked(
                        "HISTORY_DATABASE_INTEGRITY_FAILED",
                        **detail,
                    )
                elif isinstance(failure, PredictionBlocked):
                    failure.details.update(detail)
        snapshot_cleanup_error = _remove_history_work_file(snapshot)
        if snapshot_cleanup_error is not None and failure is None:
            failure = PredictionBlocked(
                "HISTORY_SEAL_WRITE_FAILED",
                snapshot_cleanup_error=snapshot_cleanup_error,
            )
        completed = failure is None
        cleanup_errors = {
            "snapshot_cleanup_error": snapshot_cleanup_error,
            "target_cleanup_error": _remove_history_work_file(target)
            if not completed
            else None,
        }
        cleanup_errors = {
            key: value for key, value in cleanup_errors.items() if value is not None
        }
        if cleanup_errors:
            if isinstance(failure, PredictionBlocked):
                failure.details.update(cleanup_errors)
            elif failure is None:
                failure = PredictionBlocked(
                    "HISTORY_SEAL_WRITE_FAILED", **cleanup_errors
                )
    if failure is not None:
        raise failure
    return {
        "schema_version": "sealed_prediction_history_v1",
        "source_sha256": source_sha256,
        "sealed_sha256": sha256_file(target),
        "target_race_id": target_race_id,
        "cutoff_timestamp": cutoff.isoformat(),
        "cutoff_basis": "race_date_strictly_before_target_jump_date",
        "safe_race_count": len(safe_ids),
        "safe_dog_row_count": len(dog_rows),
        "excluded_target_metadata_rows": excluded_target,
        "excluded_at_or_after_cutoff_metadata_rows": excluded_at_or_after,
        "excluded_ambiguous_date_metadata_rows": ambiguous_dates,
        "target_rows_materialized": 0,
        "at_or_after_cutoff_rows_materialized": 0,
    }


def market_only_prediction(receipt: Mapping[str, Any]) -> dict[str, Any]:
    win_rows = list((receipt.get("markets") or {}).get("win") or [])
    inverse = [1.0 / float(row["odds_decimal"]) for row in win_rows]
    total = sum(inverse)
    if len(win_rows) < 2 or not math.isfinite(total) or total <= 0:
        raise PredictionBlocked("MARKET_UNAVAILABLE")
    predictions = [
        {
            "box_number": int(row["box_number"]),
            "dog_name": str(row["dog_name"]),
            "probability": value / total,
            "win_odds": float(row["odds_decimal"]),
        }
        for row, value in zip(win_rows, inverse)
    ]
    predictions.sort(key=lambda row: (-row["probability"], row["box_number"]))
    for rank, row in enumerate(predictions, start=1):
        row["rank"] = rank
    return {
        "adapter": "market_only_v1",
        "variant": "market_only_implied",
        "probability_sum": sum(row["probability"] for row in predictions),
        "predictions": predictions,
    }


def bundle_manifest(
    bundle: Path, *, exclude: Sequence[str] = ("bundle_manifest.json",)
) -> dict[str, Any]:
    excluded = set(exclude)
    files = {
        path.relative_to(bundle).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(bundle.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and path.relative_to(bundle).as_posix() not in excluded
    }
    return {"schema_version": "on_demand_prediction_bundle_manifest_v1", "files": files}


def verify_bundle(bundle: Path) -> dict[str, Any]:
    if any(path.is_symlink() for path in bundle.rglob("*")):
        raise PredictionBlocked("REPLAY_TAMPERED")
    manifest_path = bundle / "bundle_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise PredictionBlocked("REPLAY_MANIFEST_INVALID") from exc
    expected = manifest.get("files") if isinstance(manifest, Mapping) else None
    if not isinstance(expected, Mapping):
        raise PredictionBlocked("REPLAY_MANIFEST_INVALID")
    actual = bundle_manifest(bundle)["files"]
    if actual != expected:
        raise PredictionBlocked("REPLAY_TAMPERED")
    return dict(manifest)


def _open_flags(*, directory: bool = False) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if directory:
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
    elif hasattr(os, "O_NONBLOCK"):
        # Opening a FIFO read-only blocks before its type can be rejected.
        # Nonblocking has no effect on ordinary file reads and lets fstat make
        # the descriptor-safe regular-file decision immediately.
        flags |= os.O_NONBLOCK
    return flags


def _identity(descriptor: int, *, directory: bool) -> tuple[int, int, int, int]:
    value = os.fstat(descriptor)
    valid = stat.S_ISDIR(value.st_mode) if directory else stat.S_ISREG(value.st_mode)
    if not valid:
        raise _blocked("PREDICTION_BUNDLE_UNSAFE_TYPE")
    return value.st_dev, value.st_ino, value.st_mode, value.st_size


def _same_identity(descriptor: int, expected: tuple[int, int, int, int], *, directory: bool) -> None:
    observed = _identity(descriptor, directory=directory)
    if observed[:2] != expected[:2] or stat.S_IFMT(observed[2]) != stat.S_IFMT(expected[2]):
        raise _blocked("PREDICTION_BUNDLE_REPLACED")


def _same_named_identity(
    parent_fd: int,
    name: str,
    descriptor: int,
    expected: tuple[int, int, int, int],
    *,
    directory: bool,
) -> None:
    _same_identity(descriptor, expected, directory=directory)
    try:
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise _blocked("PREDICTION_BUNDLE_REPLACED", name=name) from exc
    if (named.st_dev, named.st_ino) != expected[:2]:
        raise _blocked("PREDICTION_BUNDLE_REPLACED", name=name)


def _same_root_identity(
    root: Path, descriptor: int, expected: tuple[int, int, int, int]
) -> None:
    try:
        check_fd = os.open(root, _open_flags(directory=True))
    except OSError as exc:
        raise _blocked("PREDICTION_BUNDLE_REPLACED", name="root") from exc
    try:
        if _identity(check_fd, directory=True)[:2] != expected[:2]:
            raise _blocked("PREDICTION_BUNDLE_REPLACED", name="root")
        _same_identity(descriptor, expected, directory=True)
    finally:
        os.close(check_fd)


def _openat(
    parent_fd: int,
    name: str,
    *,
    directory: bool,
    missing_code: str = "PREDICTION_BUNDLE_OPEN_FAILED",
) -> tuple[int, tuple[int, int, int, int]]:
    try:
        descriptor = os.open(name, _open_flags(directory=directory), dir_fd=parent_fd)
    except FileNotFoundError as exc:
        raise _blocked(missing_code, name=name) from exc
    except OSError as exc:
        raise _blocked("PREDICTION_BUNDLE_OPEN_FAILED", name=name) from exc
    try:
        identity = _identity(descriptor, directory=directory)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (named.st_dev, named.st_ino) != identity[:2]:
            raise _blocked("PREDICTION_BUNDLE_REPLACED", name=name)
        return descriptor, identity
    except Exception:
        os.close(descriptor)
        raise


def _read_fd(
    descriptor: int,
    identity: tuple[int, int, int, int],
    *,
    max_bytes: int,
    start: float,
    seconds: float,
    monotonic: Callable[[], float],
) -> bytes:
    before = os.fstat(descriptor)
    if identity[3] > max_bytes:
        raise _blocked("PREDICTION_BUNDLE_INVALID", reason="size")
    chunks: list[bytes] = []
    remaining = max_bytes + 1
    os.lseek(descriptor, 0, os.SEEK_SET)
    while remaining:
        _deadline(start, seconds, monotonic)
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk); remaining -= len(chunk)
    raw = b"".join(chunks)
    if len(raw) > max_bytes:
        raise _blocked("PREDICTION_BUNDLE_INVALID", reason="size")
    after = os.fstat(descriptor)
    if (after.st_dev, after.st_ino, after.st_mode, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (
        before.st_dev, before.st_ino, before.st_mode, before.st_size, before.st_mtime_ns, before.st_ctime_ns
    ) or after.st_size != len(raw):
        raise _blocked("PREDICTION_BUNDLE_CHANGED")
    _same_identity(descriptor, identity, directory=False)
    return raw


def validate_prediction_bundle_index_v1(
    value: Any, *, require_publication_time: bool = False
) -> dict[str, Any]:
    fields = set(value) if isinstance(value, Mapping) else set()
    allowed = {"schema_version", "entries"}
    sealed = allowed | {"published_at"}
    accepted = (sealed,) if require_publication_time else (allowed, sealed)
    if not any(fields == candidate for candidate in accepted):
        raise _blocked("PREDICTION_BUNDLE_INVALID", field="index", reason="fields")
    index = _exact_fields(value, fields, "index")
    if index["schema_version"] != PREDICTION_BUNDLE_INDEX_SCHEMA:
        raise _blocked("PREDICTION_BUNDLE_INVALID", field="index.schema_version")
    if "published_at" in index:
        _timestamp(index["published_at"], "index.published_at")
    entries = index["entries"]
    if not isinstance(entries, list) or len(entries) > INDEX_MAX_ENTRIES:
        raise _blocked("PREDICTION_BUNDLE_INVALID", field="index.entries")
    fields = {"directory", "prediction_id", "job_id", "generated_at", "status", "blocker_stage", "manifest_sha256", "logical_bundle_sha256"}
    seen_ids: set[str] = set(); seen_dirs: set[str] = set(); order: list[tuple[float, str]] = []
    for entry in entries:
        row = _exact_fields(entry, fields, "index.entry")
        directory = row["directory"]
        if not isinstance(directory, str) or PREDICTION_BUNDLE_DIRECTORY_RE.fullmatch(directory) is None or directory in seen_dirs:
            raise _blocked("PREDICTION_BUNDLE_INVALID", field="index.entry.directory")
        prediction_id = _prediction_id(row["prediction_id"])
        if prediction_id in seen_ids:
            raise _blocked("PREDICTION_BUNDLE_INVALID", field="index.entry.prediction_id")
        _job_id(row["job_id"], "index.entry.job_id")
        generated = _timestamp(row["generated_at"], "index.entry.generated_at")
        if row["status"] not in {"PREDICTION_READY", "PREDICTION_BLOCKED"}:
            raise _blocked("PREDICTION_BUNDLE_INVALID", field="index.entry.status")
        if (
            row["status"] == "PREDICTION_READY"
            and row["blocker_stage"] is not None
        ) or (
            row["status"] == "PREDICTION_BLOCKED"
            and row["blocker_stage"] not in {"PROTOCOL", "VALIDATION", "SCORING"}
        ):
            raise _blocked("PREDICTION_BUNDLE_INVALID", field="index.entry.blocker_stage")
        _sha(row["manifest_sha256"], "index.entry.manifest_sha256"); _sha(row["logical_bundle_sha256"], "index.entry.logical_bundle_sha256")
        seen_dirs.add(directory); seen_ids.add(prediction_id)
        order.append((-generated.timestamp(), prediction_id))
    if order != sorted(order):
        raise _blocked("PREDICTION_BUNDLE_INVALID", field="index.entries", reason="order")
    return dict(index)


def verify_prediction_bundle_index(
    root: Path, *, monotonic: Callable[[], float] = time.monotonic,
    return_verified_view: bool = False,
) -> dict[str, Any] | VerifiedPredictionBundleIndex:
    start = monotonic()
    root_fd = os.open(root, _open_flags(directory=True))
    try:
        root_identity = _identity(root_fd, directory=True)
        index_fd, index_identity = _openat(
            root_fd,
            PREDICTION_BUNDLE_INDEX_NAME,
            directory=False,
            missing_code="PREDICTION_BUNDLE_INDEX_UNAVAILABLE",
        )
        try:
            raw = _read_fd(index_fd, index_identity, max_bytes=INDEX_MAX_BYTES, start=start, seconds=1.0, monotonic=monotonic)
            value = validate_prediction_bundle_index_v1(
                _canonical_json(raw, max_bytes=INDEX_MAX_BYTES, label="index"),
                require_publication_time=False,
            )
            named = os.stat(PREDICTION_BUNDLE_INDEX_NAME, dir_fd=root_fd, follow_symlinks=False)
            if (named.st_dev, named.st_ino) != index_identity[:2]:
                raise _blocked("PREDICTION_BUNDLE_REPLACED")
            _same_root_identity(root, root_fd, root_identity)
            _deadline(start, 1.0, monotonic)
            if return_verified_view:
                return VerifiedPredictionBundleIndex(
                    schema_version=value["schema_version"],
                    published_at=value.get("published_at"),
                    entries=tuple(value["entries"]), canonical_bytes=raw,
                    sha256=sha256_bytes(raw),
                )
            return value
        finally:
            os.close(index_fd)
    finally:
        os.close(root_fd)


def _directory_children(descriptor: int) -> tuple[str, ...]:
    try:
        children = tuple(os.listdir(descriptor))
    except OSError as exc:
        raise _blocked("PREDICTION_BUNDLE_ENUMERATION_FAILED") from exc
    if len(children) > BUNDLE_MAX_ENTRIES * 2 + 1 or len(children) != len(set(children)):
        raise _blocked("PREDICTION_BUNDLE_INVALID", reason="enumeration_bound")
    return children


def verify_indexed_prediction_bundle(
    root: Path,
    entry: Mapping[str, Any],
    *,
    monotonic: Callable[[], float] = time.monotonic,
) -> VerifiedPredictionBundle:
    # A selected entry is validated independently of index listing freshness.
    validate_prediction_bundle_index_v1({"schema_version": PREDICTION_BUNDLE_INDEX_SCHEMA, "entries": [dict(entry)]})
    start = monotonic(); root_fd = os.open(root, _open_flags(directory=True)); descriptors: list[int] = []
    try:
        root_identity = _identity(root_fd, directory=True)
        bundle_fd, bundle_identity = _openat(root_fd, str(entry["directory"]), directory=True); descriptors.append(bundle_fd)
        manifest_fd, manifest_identity = _openat(bundle_fd, "bundle_manifest.json", directory=False); descriptors.append(manifest_fd)
        manifest_raw = _read_fd(manifest_fd, manifest_identity, max_bytes=BUNDLE_CONTROL_MAX_BYTES, start=start, seconds=5.0, monotonic=monotonic)
        if sha256_bytes(manifest_raw) != entry["manifest_sha256"]:
            raise _blocked("PREDICTION_BUNDLE_CHANGED", field="manifest")
        manifest = validate_prediction_bundle_manifest_v2(_canonical_json(manifest_raw, max_bytes=BUNDLE_CONTROL_MAX_BYTES, label="manifest"))
        if logical_bundle_sha256(manifest) != entry["logical_bundle_sha256"]:
            raise _blocked("PREDICTION_BUNDLE_CHANGED", field="logical_bundle_sha256")
        parent_children: dict[str, set[str]] = {"": {"bundle_manifest.json"}}
        directory_fds: dict[str, int] = {"": bundle_fd}
        directory_ids: dict[str, tuple[int, int, int, int]] = {"": bundle_identity}
        directory_names: dict[str, tuple[int, str]] = {}
        for name in manifest["files"]:
            parts = name.split("/"); parent = ""
            for component in parts[:-1]:
                parent_children.setdefault(parent, set()).add(component)
                child = f"{parent}/{component}".strip("/")
                if child not in directory_fds:
                    child_fd, child_id = _openat(directory_fds[parent], component, directory=True)
                    descriptors.append(child_fd); directory_fds[child] = child_fd; directory_ids[child] = child_id
                    directory_names[child] = (directory_fds[parent], component)
                parent_children.setdefault(child, set()); parent = child
            parent_children.setdefault(parent, set()).add(parts[-1])
        if sum(len(items) for items in parent_children.values()) > BUNDLE_MAX_ENTRIES * 2 + 1:
            raise _blocked("PREDICTION_BUNDLE_INVALID", reason="enumeration_bound")
        for parent, expected in parent_children.items():
            _deadline(start, 5.0, monotonic)
            if set(_directory_children(directory_fds[parent])) != expected:
                raise _blocked("PREDICTION_BUNDLE_MEMBERSHIP_MISMATCH", directory=parent)
        contents: dict[str, bytes] = {}
        file_handles: list[tuple[int, str, int, tuple[int, int, int, int]]] = []
        retained_file_identities = {manifest_identity[:2]: "bundle_manifest.json"}
        aggregate_bytes = 0
        for name, expected in manifest["files"].items():
            parent, leaf = name.rsplit("/", 1) if "/" in name else ("", name)
            file_fd, file_identity = _openat(directory_fds[parent], leaf, directory=False); descriptors.append(file_fd)
            prior_name = retained_file_identities.get(file_identity[:2])
            if prior_name is not None:
                raise _blocked(
                    "PREDICTION_BUNDLE_INVALID",
                    reason="duplicate_file_identity",
                    field=name,
                    aliases=prior_name,
                )
            retained_file_identities[file_identity[:2]] = name
            raw = _read_fd(file_fd, file_identity, max_bytes=BUNDLE_FILE_MAX_BYTES, start=start, seconds=5.0, monotonic=monotonic)
            if len(raw) != expected["bytes"] or sha256_bytes(raw) != expected["sha256"]:
                raise _blocked("PREDICTION_BUNDLE_CHANGED", field=name)
            aggregate_bytes += len(raw)
            if aggregate_bytes > BUNDLE_AGGREGATE_MAX_BYTES:
                raise _blocked("PREDICTION_BUNDLE_INVALID", reason="aggregate")
            contents[name] = raw
            file_handles.append((directory_fds[parent], leaf, file_fd, file_identity))
        if not {"result.json", "request.json", "config.json", "model/config.schema.json"}.issubset(contents):
            raise _blocked("PREDICTION_BUNDLE_INVALID", reason="required_files")
        result = validate_prediction_result_v2(_canonical_json(contents["result.json"], max_bytes=BUNDLE_CONTROL_MAX_BYTES, label="result"))
        if result["prediction_id"] != entry["prediction_id"] or result["job_id"] != entry["job_id"] or result["generated_at"] != entry["generated_at"] or result["status"] != entry["status"] or result["blocker_stage"] != entry["blocker_stage"] or manifest["prediction_id"] != entry["prediction_id"] or manifest["job_id"] != entry["job_id"]:
            raise _blocked("PREDICTION_BUNDLE_IDENTITY_MISMATCH")
        if any(
            result["evidence"][name] is not None
            and result["evidence"][name] not in contents
            for name in ("request", "config", "model_schema", "model_artifact", "model_manifest")
        ):
            raise _blocked("PREDICTION_BUNDLE_IDENTITY_MISMATCH", field="evidence")
        if result["evidence"]["request"] != "request.json" or result["evidence"]["config"] != "config.json" or result["evidence"]["model_schema"] != "model/config.schema.json":
            raise _blocked("PREDICTION_BUNDLE_IDENTITY_MISMATCH", field="evidence")
        if sha256_bytes(contents["config.json"]) != result["config"]["sha256"]:
            raise _blocked("PREDICTION_BUNDLE_IDENTITY_MISMATCH", field="config")
        request = _validate_request_binding(contents["request.json"], result)
        if result["status"] == "PREDICTION_READY":
            _validate_sealed_protocol(contents, result)
        if sha256_bytes(contents[result["evidence"]["model_schema"]]) != result["model"]["schema_sha256"]:
            raise _blocked("PREDICTION_BUNDLE_IDENTITY_MISMATCH", field="model_schema")
        for key, hash_key in (("model_artifact", "artifact_sha256"), ("model_manifest", "artifact_manifest_sha256")):
            locator = result["evidence"][key]
            if locator is None:
                if result["model"][hash_key] is not None:
                    raise _blocked("PREDICTION_BUNDLE_IDENTITY_MISMATCH", field=key)
            elif sha256_bytes(contents[locator]) != result["model"][hash_key]:
                raise _blocked("PREDICTION_BUNDLE_IDENTITY_MISMATCH", field=key)
        for parent_fd, leaf, file_fd, file_identity in file_handles:
            _same_named_identity(
                parent_fd, leaf, file_fd, file_identity, directory=False
            )
        _same_named_identity(
            bundle_fd,
            "bundle_manifest.json",
            manifest_fd,
            manifest_identity,
            directory=False,
        )
        for name in directory_fds:
            _same_identity(directory_fds[name], directory_ids[name], directory=True)
        for name, (parent_fd, component) in directory_names.items():
            _same_named_identity(
                parent_fd,
                component,
                directory_fds[name],
                directory_ids[name],
                directory=True,
            )
        _same_named_identity(
            root_fd,
            str(entry["directory"]),
            bundle_fd,
            bundle_identity,
            directory=True,
        )
        _same_root_identity(root, root_fd, root_identity)
        _deadline(start, 5.0, monotonic)
        return VerifiedPredictionBundle(str(entry["directory"]), dict(entry), result, manifest, request)
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        os.close(root_fd)


def build_prediction_bundle_manifest_v2(
    bundle: Path, *, prediction_id: str, job_id: str | None
) -> dict[str, Any]:
    _prediction_id(prediction_id)
    _job_id(job_id)
    start = time.monotonic()
    parent = bundle.parent
    root_fd = os.open(parent, _open_flags(directory=True))
    descriptors: list[int] = []
    try:
        root_identity = _identity(root_fd, directory=True)
        bundle_fd, bundle_identity = _openat(root_fd, bundle.name, directory=True)
        descriptors.append(bundle_fd)
        directories: dict[str, int] = {"": bundle_fd}
        directory_ids: dict[str, tuple[int, int, int, int]] = {"": bundle_identity}
        directory_names: dict[str, tuple[int, str]] = {}
        files: dict[str, dict[str, Any]] = {}
        file_handles: list[tuple[int, str, int, tuple[int, int, int, int]]] = []
        retained_file_identities: dict[tuple[int, int], str] = {}
        pending = [""]
        observed_entries = 0
        aggregate_bytes = 0
        while pending:
            prefix = pending.pop()
            directory_fd = directories[prefix]
            names = sorted(_directory_children(directory_fd))
            observed_entries += len(names)
            if observed_entries > BUNDLE_MAX_ENTRIES * 2 + 1:
                raise _blocked("PREDICTION_BUNDLE_INVALID", reason="enumeration_bound")
            for name in names:
                relative = f"{prefix}/{name}".strip("/")
                _relative_name(relative)
                if relative == "bundle_manifest.json":
                    manifest_fd, manifest_identity = _openat(
                        directory_fd, name, directory=False
                    )
                    descriptors.append(manifest_fd)
                    prior_name = retained_file_identities.get(
                        manifest_identity[:2]
                    )
                    if prior_name is not None:
                        raise _blocked(
                            "PREDICTION_BUNDLE_INVALID",
                            reason="duplicate_file_identity",
                            field=relative,
                            aliases=prior_name,
                        )
                    retained_file_identities[manifest_identity[:2]] = relative
                    file_handles.append(
                        (directory_fd, name, manifest_fd, manifest_identity)
                    )
                    continue
                try:
                    named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                except OSError as exc:
                    raise _blocked("PREDICTION_BUNDLE_CHANGED", field=relative) from exc
                if stat.S_ISDIR(named.st_mode):
                    child_fd, child_identity = _openat(directory_fd, name, directory=True)
                    descriptors.append(child_fd)
                    if (named.st_dev, named.st_ino) != child_identity[:2]:
                        raise _blocked("PREDICTION_BUNDLE_REPLACED", field=relative)
                    directories[relative] = child_fd
                    directory_ids[relative] = child_identity
                    directory_names[relative] = (directory_fd, name)
                    pending.append(relative)
                    continue
                if not stat.S_ISREG(named.st_mode):
                    raise _blocked("PREDICTION_BUNDLE_UNSAFE_TYPE", name=relative)
                if len(files) >= BUNDLE_MAX_ENTRIES:
                    raise _blocked("PREDICTION_BUNDLE_INVALID", reason="file_count")
                file_fd, file_identity = _openat(directory_fd, name, directory=False)
                descriptors.append(file_fd)
                if (named.st_dev, named.st_ino) != file_identity[:2]:
                    raise _blocked("PREDICTION_BUNDLE_CHANGED", field=relative)
                prior_name = retained_file_identities.get(file_identity[:2])
                if prior_name is not None:
                    raise _blocked(
                        "PREDICTION_BUNDLE_INVALID",
                        reason="duplicate_file_identity",
                        field=relative,
                        aliases=prior_name,
                    )
                retained_file_identities[file_identity[:2]] = relative
                raw = _read_fd(file_fd, file_identity, max_bytes=BUNDLE_FILE_MAX_BYTES,
                               start=start, seconds=5.0, monotonic=time.monotonic)
                aggregate_bytes += len(raw)
                if aggregate_bytes > BUNDLE_AGGREGATE_MAX_BYTES:
                    raise _blocked("PREDICTION_BUNDLE_INVALID", reason="aggregate")
                files[relative] = {"bytes": len(raw), "sha256": sha256_bytes(raw)}
                file_handles.append((directory_fd, name, file_fd, file_identity))
        for parent_fd, name, file_fd, identity in file_handles:
            _same_named_identity(parent_fd, name, file_fd, identity, directory=False)
        for name, descriptor in directories.items():
            _same_identity(descriptor, directory_ids[name], directory=True)
        for name, (parent_fd, component) in directory_names.items():
            _same_named_identity(parent_fd, component, directories[name], directory_ids[name], directory=True)
        _same_named_identity(root_fd, bundle.name, bundle_fd, bundle_identity, directory=True)
        _same_root_identity(parent, root_fd, root_identity)
        _deadline(start, 5.0, time.monotonic)
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        os.close(root_fd)
    manifest = {
        "schema_version": PREDICTION_MANIFEST_SCHEMA_V2,
        "prediction_id": prediction_id,
        "job_id": job_id,
        "files": dict(sorted(files.items())),
    }
    return validate_prediction_bundle_manifest_v2(manifest)


def prediction_bundle_index_entry(
    *, bundle: Path, result: Mapping[str, Any], manifest_raw: bytes
) -> dict[str, Any]:
    validated = validate_prediction_result_v2(result)
    manifest = validate_prediction_bundle_manifest_v2(
        _canonical_json(manifest_raw, max_bytes=BUNDLE_CONTROL_MAX_BYTES, label="manifest")
    )
    if validated["prediction_id"] != manifest["prediction_id"] or validated["job_id"] != manifest["job_id"]:
        raise _blocked("PREDICTION_BUNDLE_IDENTITY_MISMATCH")
    return {
        "directory": bundle.name,
        "prediction_id": validated["prediction_id"],
        "job_id": validated["job_id"],
        "generated_at": validated["generated_at"],
        "status": validated["status"],
        "blocker_stage": validated["blocker_stage"],
        "manifest_sha256": sha256_bytes(manifest_raw),
        "logical_bundle_sha256": logical_bundle_sha256(manifest),
    }


def _acquire_index_lock(
    root_fd: int,
    *,
    start: float,
    monotonic: Callable[[], float],
) -> tuple[int, tuple[int, int, int, int], bytes]:
    token = uuid.uuid4().hex
    payload = canonical_bytes({"pid": os.getpid(), "token": token})
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(PREDICTION_BUNDLE_LOCK_NAME, flags, 0o600, dir_fd=root_fd)
    except OSError as exc:
        raise _blocked("PREDICTION_BUNDLE_INDEX_LOCK_UNAVAILABLE") from exc
    try:
        identity = _identity(descriptor, directory=False)
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise _blocked("PREDICTION_BUNDLE_INDEX_LOCK_INVALID")
        os.fsync(descriptor)
        named = os.stat(PREDICTION_BUNDLE_LOCK_NAME, dir_fd=root_fd, follow_symlinks=False)
        if (named.st_dev, named.st_ino) != identity[:2]:
            raise _blocked("PREDICTION_BUNDLE_INDEX_LOCK_REPLACED")
        _deadline(start, 1.0, monotonic)
        return descriptor, identity, payload
    except Exception:
        try:
            named = os.stat(
                PREDICTION_BUNDLE_LOCK_NAME,
                dir_fd=root_fd,
                follow_symlinks=False,
            )
            opened = os.fstat(descriptor)
            if (named.st_dev, named.st_ino) == (opened.st_dev, opened.st_ino):
                os.unlink(PREDICTION_BUNDLE_LOCK_NAME, dir_fd=root_fd)
                os.fsync(root_fd)
        except OSError:
            pass
        finally:
            os.close(descriptor)
        raise


def _release_index_lock(root_fd: int, descriptor: int, identity: tuple[int, int, int, int], payload: bytes) -> None:
    try:
        _same_identity(descriptor, identity, directory=False)
        named = os.stat(PREDICTION_BUNDLE_LOCK_NAME, dir_fd=root_fd, follow_symlinks=False)
        if (named.st_dev, named.st_ino) != identity[:2]:
            raise _blocked("PREDICTION_BUNDLE_INDEX_LOCK_RELEASE_FAILED")
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.read(descriptor, len(payload) + 1) != payload:
            raise _blocked("PREDICTION_BUNDLE_INDEX_LOCK_RELEASE_FAILED")
        os.unlink(PREDICTION_BUNDLE_LOCK_NAME, dir_fd=root_fd)
        os.fsync(root_fd)
    except (OSError, PredictionBlocked) as exc:
        if isinstance(exc, PredictionBlocked):
            raise
        raise _blocked("PREDICTION_BUNDLE_INDEX_LOCK_RELEASE_FAILED") from exc


def publish_prediction_bundle_index_entry(
    root: Path,
    entry: Mapping[str, Any] | None,
    *,
    monotonic: Callable[[], float] = time.monotonic,
    _clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    single = None
    if entry is not None:
        single = validate_prediction_bundle_index_v1({"schema_version": PREDICTION_BUNDLE_INDEX_SCHEMA, "entries": [dict(entry)]})["entries"][0]
    start = monotonic(); root_fd = os.open(root, _open_flags(directory=True)); lock_fd: int | None = None
    try:
        root_identity = _identity(root_fd, directory=True)
        lock_fd, lock_identity, lock_payload = _acquire_index_lock(
            root_fd, start=start, monotonic=monotonic
        )
        _deadline(start, 1.0, monotonic)
        try:
            _same_root_identity(root, root_fd, root_identity)
            try:
                index_fd, index_identity = _openat(
                    root_fd,
                    PREDICTION_BUNDLE_INDEX_NAME,
                    directory=False,
                    missing_code="PREDICTION_BUNDLE_INDEX_UNAVAILABLE",
                )
            except PredictionBlocked as exc:
                try:
                    os.stat(PREDICTION_BUNDLE_INDEX_NAME, dir_fd=root_fd, follow_symlinks=False)
                except FileNotFoundError:
                    current = {"schema_version": PREDICTION_BUNDLE_INDEX_SCHEMA, "entries": []}
                else:
                    raise exc
            else:
                try:
                    raw = _read_fd(index_fd, index_identity, max_bytes=INDEX_MAX_BYTES, start=start, seconds=1.0, monotonic=monotonic)
                    current = validate_prediction_bundle_index_v1(_canonical_json(raw, max_bytes=INDEX_MAX_BYTES, label="index"))
                finally:
                    os.close(index_fd)
            if single is None:
                entries = [dict(row) for row in current["entries"]]
            else:
                matching = [row for row in current["entries"] if row["prediction_id"] == single["prediction_id"]]
                if matching and matching[0] != single:
                    raise _blocked("PREDICTION_BUNDLE_IDENTITY_MISMATCH", field="prediction_id")
                entries = [dict(row) for row in current["entries"] if row["prediction_id"] != single["prediction_id"]]
                entries.append(dict(single))
            entries.sort(key=lambda row: (-_timestamp(row["generated_at"], "generated_at").timestamp(), row["prediction_id"]))
            entries = entries[:INDEX_MAX_ENTRIES]
            if entries == current["entries"] and "published_at" in current:
                return current
            publication_time = _clock()
            if not isinstance(publication_time, datetime) or publication_time.tzinfo is None or publication_time.utcoffset() is None:
                raise _blocked("PREDICTION_BUNDLE_INVALID", field="index.published_at")
            publication_time = publication_time.astimezone(timezone.utc)
            if "published_at" in current and publication_time < _timestamp(current["published_at"], "index.published_at"):
                raise _blocked("PREDICTION_BUNDLE_PUBLICATION_TIME_REGRESSION")
            publication_text = publication_time.isoformat()
            updated = validate_prediction_bundle_index_v1({"schema_version": PREDICTION_BUNDLE_INDEX_SCHEMA, "published_at": publication_text, "entries": entries})
            raw = canonical_bytes(updated)
            if len(raw) > INDEX_MAX_BYTES:
                raise _blocked("PREDICTION_BUNDLE_INVALID", reason="index_size")
            temporary = f".{PREDICTION_BUNDLE_INDEX_NAME}.{uuid.uuid4().hex}.tmp"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_CLOEXEC"): flags |= os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"): flags |= os.O_NOFOLLOW
            temp_fd = os.open(temporary, flags, 0o600, dir_fd=root_fd)
            temp_identity: tuple[int, int, int, int] | None = None
            try:
                if os.write(temp_fd, raw) != len(raw):
                    raise _blocked("PREDICTION_BUNDLE_INDEX_WRITE_FAILED")
                os.fsync(temp_fd)
                temp_identity = _identity(temp_fd, directory=False)
                _deadline(start, 1.0, monotonic)
            except Exception:
                try:
                    os.unlink(temporary, dir_fd=root_fd)
                except OSError:
                    pass
                raise
            finally:
                os.close(temp_fd)
            try:
                _same_root_identity(root, root_fd, root_identity)
                _same_named_identity(
                    root_fd,
                    PREDICTION_BUNDLE_LOCK_NAME,
                    lock_fd,
                    lock_identity,
                    directory=False,
                )
                os.lseek(lock_fd, 0, os.SEEK_SET)
                if os.read(lock_fd, len(lock_payload) + 1) != lock_payload:
                    raise _blocked("PREDICTION_BUNDLE_INDEX_LOCK_REPLACED")
                _deadline(start, 1.0, monotonic)
                os.replace(temporary, PREDICTION_BUNDLE_INDEX_NAME, src_dir_fd=root_fd, dst_dir_fd=root_fd)
                os.fsync(root_fd)
                named = os.stat(
                    PREDICTION_BUNDLE_INDEX_NAME,
                    dir_fd=root_fd,
                    follow_symlinks=False,
                )
                if temp_identity is None or (named.st_dev, named.st_ino) != temp_identity[:2]:
                    raise _blocked("PREDICTION_BUNDLE_REPLACED", name=PREDICTION_BUNDLE_INDEX_NAME)
            except Exception:
                try: os.unlink(temporary, dir_fd=root_fd)
                except OSError: pass
                raise
            _same_root_identity(root, root_fd, root_identity)
            return updated
        finally:
            if lock_fd is not None:
                try:
                    _release_index_lock(root_fd, lock_fd, lock_identity, lock_payload)
                finally:
                    os.close(lock_fd); lock_fd = None
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        os.close(root_fd)
