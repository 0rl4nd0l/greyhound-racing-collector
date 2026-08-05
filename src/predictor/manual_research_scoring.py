"""Sealed-evidence-only form features and research scoring.

This module deliberately accepts bytes and an already loaded frozen model.  It
has no path to SQLite, canonical history, live odds, results, Phase 7, or
prediction persistence.  The only sealed-evidence read is delegated to the
GHU-052 verifier; the verifier returns the validated normalized odds used here.
"""

from __future__ import annotations

import fcntl
import hashlib
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from src.predictor.manual_independent_capture import (
    canonical_bytes,
    parse_canonical_json,
)
from src.predictor.manual_independent_capture_sealer import (
    SealedManualEvidence,
    SealExpectations,
    SealingIdentity,
    verify_manual_evidence_bundle,
)
from src.predictor.market_form_residual import (
    EFFECTIVE_STATE_SCHEMA,
    FEATURES,
    SHADOW_RECORD_SCHEMA,
    FrozenResidualModel,
    ResidualContractError,
    score_race,
)

EMBEDDED_FORM_SCHEMA = "manual_research_embedded_form_v1"
PREDICTION_SCHEMA = "manual_research_prediction_v1"
PREDICTION_MANIFEST_SCHEMA = "manual_research_prediction_manifest_v1"
FEATURE_ADAPTER_VERSION = "manual_research_embedded_form_features_v1"
SCORER_VERSION = "manual_research_market_form_residual_v1"
PUBLISH_PROTOCOL = "same_filesystem_fsync_atomic_directory_rename_v1"
FEATURE_SCHEMA_RELATIVE = Path(
    "configs/prediction/manual-independent-capture-v1/embedded-form.schema.json"
)
PREDICTION_SCHEMA_RELATIVE = Path(
    "configs/prediction/manual-independent-capture-v1/research-prediction.schema.json"
)
PREDICTION_MANIFEST_SCHEMA_RELATIVE = Path(
    "configs/prediction/manual-independent-capture-v1/research-prediction-manifest.schema.json"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}$")
_FEATURE_FIELDS = frozenset(FEATURES)
_HISTORY_FIELDS = frozenset(
    {
        "prior_race_id",
        "event_timestamp",
        "race_date",
        "venue",
        "distance_m",
        "grade",
        "prior_finish",
        "prior_margin",
    }
)
_OUTCOME_WORDS = frozenset(
    {"result", "results", "outcome", "outcomes", "winner", "winners", "placing", "placings"}
)


class ManualResearchScoringRejected(RuntimeError):
    """A fail-closed research scoring rejection."""

    def __init__(self, code: str, **details: Any) -> None:
        super().__init__(code)
        self.code = code
        self.details = details


@dataclass(frozen=True)
class ResearchScoringIdentity:
    source_commit: str
    source_tree: str
    feature_adapter_sha256: str
    scorer_sha256: str
    embedded_form_schema_sha256: str
    prediction_schema_sha256: str
    prediction_manifest_schema_sha256: str


@dataclass(frozen=True)
class ResearchPredictionExpectations:
    evidence_bundle_id: str
    evidence_manifest_sha256: str
    race_identity_sha256: str
    form_sha256: str
    config_sha256: str
    model_sha256: str
    model_manifest_sha256: str
    runner_set_sha256: str
    odds_sha256: str
    cutoff_timestamp: str
    scheduled_start: str
    effective_state_sha256: str
    implementation: ResearchScoringIdentity
    feature_sha256: str | None = None


@dataclass(frozen=True)
class VerifiedResearchPrediction:
    bundle_dir: Path
    prediction: Mapping[str, Any]
    manifest: Mapping[str, Any]
    manifest_sha256: str
    replayed: bool


def build_research_scoring_identity(
    *, repo_root: Path, source_commit: str, source_tree: str
) -> ResearchScoringIdentity:
    """Hash only the reviewed GHU-053 implementation and schema files."""

    root = Path(repo_root).resolve()

    def file_hash(relative: Path) -> str:
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise _reject("IMPLEMENTATION_PATH_INVALID", path=str(relative)) from exc
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise _reject("IMPLEMENTATION_UNAVAILABLE", path=str(relative)) from exc
        return hashlib.sha256(raw).hexdigest()

    return ResearchScoringIdentity(
        source_commit=source_commit,
        source_tree=source_tree,
        feature_adapter_sha256=file_hash(Path("src/predictor/manual_research_scoring.py")),
        scorer_sha256=file_hash(Path("src/predictor/market_form_residual.py")),
        embedded_form_schema_sha256=file_hash(FEATURE_SCHEMA_RELATIVE),
        prediction_schema_sha256=file_hash(PREDICTION_SCHEMA_RELATIVE),
        prediction_manifest_schema_sha256=file_hash(PREDICTION_MANIFEST_SCHEMA_RELATIVE),
    )


def _reject(code: str, **details: Any) -> ManualResearchScoringRejected:
    return ManualResearchScoringRejected(code, **details)


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise _reject("HASH_INVALID", field=field)
    return value


def _git_object(value: Any, field: str) -> str:
    if not isinstance(value, str) or _GIT_OBJECT_RE.fullmatch(value) is None:
        raise _reject("IMPLEMENTATION_IDENTITY_INVALID", field=field)
    return value


def _finite(value: Any, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise _reject("NONFINITE_INPUT", field=field)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise _reject("NONFINITE_INPUT", field=field) from exc
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise _reject("NONFINITE_INPUT", field=field)
    return number


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise _reject("TIMESTAMP_INVALID", field=field)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise _reject("TIMESTAMP_INVALID", field=field) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.isoformat() != value:
        raise _reject("TIMESTAMP_INVALID", field=field)
    return parsed


def _exact(value: Any, fields: set[str], field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise _reject("SCHEMA_FIELDS_INVALID", field=field)
    return value


def _canonical_form(raw: bytes, expected_sha256: str) -> Mapping[str, Any]:
    if not isinstance(raw, bytes) or not raw:
        raise _reject("FORM_MISSING")
    if hashlib.sha256(raw).hexdigest() != _sha(expected_sha256, "form_sha256"):
        raise _reject("FORM_HASH_MISMATCH")
    try:
        value = parse_canonical_json(raw, max_bytes=4 * 1024 * 1024)
    except Exception as exc:
        raise _reject("FORM_NOT_CANONICAL") from exc
    return _exact(
        value,
        {"schema_version", "safety", "source", "cutoff_timestamp", "target", "runners"},
        "form",
    )


def _validate_identity(identity: ResearchScoringIdentity) -> dict[str, str]:
    if not isinstance(identity, ResearchScoringIdentity):
        raise _reject("IMPLEMENTATION_IDENTITY_INVALID")
    result = {
        "source_commit": _git_object(identity.source_commit, "source_commit"),
        "source_tree": _git_object(identity.source_tree, "source_tree"),
        "feature_adapter_sha256": _sha(identity.feature_adapter_sha256, "feature_adapter_sha256"),
        "scorer_sha256": _sha(identity.scorer_sha256, "scorer_sha256"),
        "embedded_form_schema_sha256": _sha(
            identity.embedded_form_schema_sha256, "embedded_form_schema_sha256"
        ),
        "prediction_schema_sha256": _sha(identity.prediction_schema_sha256, "prediction_schema_sha256"),
        "prediction_manifest_schema_sha256": _sha(
            identity.prediction_manifest_schema_sha256, "prediction_manifest_schema_sha256"
        ),
    }
    return result


def _validate_config(raw: bytes, expected_sha256: str, model: FrozenResidualModel) -> tuple[Mapping[str, Any], str]:
    if not isinstance(raw, bytes) or not raw:
        raise _reject("CONFIG_MISSING")
    config_sha256 = hashlib.sha256(raw).hexdigest()
    if config_sha256 != _sha(expected_sha256, "config_sha256"):
        raise _reject("CONFIG_HASH_MISMATCH")
    try:
        config = parse_canonical_json(raw, max_bytes=64 * 1024)
    except Exception as exc:
        raise _reject("CONFIG_NOT_CANONICAL") from exc
    row = _exact(
        config,
        {"schema_version", "safety", "scorer_id", "model_id", "model_sha256", "model_manifest_sha256", "feature_adapter_version", "ranking", "persistence"},
        "config",
    )
    if (
        row["schema_version"] != "manual_research_scoring_config_v1"
        or row["safety"]
        != {
            "research_only": True,
            "canonical": False,
            "phase7_excluded": True,
            "phase7_eligible": False,
            "phase7_exclusion_reason": "manual_research_only_noncanonical",
        }
        or row["scorer_id"] != SCORER_VERSION
        or row["model_id"] != "market_form_residual_v1"
        or row["model_sha256"] != model.model_sha256
        or row["model_manifest_sha256"] != model.manifest_sha256
        or row["feature_adapter_version"] != FEATURE_ADAPTER_VERSION
        or row["ranking"] != {"primary_probability": "full_probability", "tie_break": "box_ascending"}
        or row["persistence"] != "none"
    ):
        raise _reject("CONFIG_CONTRACT_MISMATCH")
    return row, config_sha256


def _grade(value: Any, field: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise _reject("FORM_FIELD_INVALID", field=field)
    normalized = " ".join(value.upper().split())
    return normalized


def _target_and_form_rows(form: Mapping[str, Any], evidence: Mapping[str, Any]) -> tuple[datetime, Mapping[str, Any], dict[int, list[dict[str, Any]]]]:
    if form["schema_version"] != EMBEDDED_FORM_SCHEMA:
        raise _reject("FORM_SCHEMA_MISMATCH")
    if form["safety"] != {
        "research_only": True,
        "canonical": False,
        "phase7_excluded": True,
        "phase7_eligible": False,
        "phase7_exclusion_reason": "manual_research_only_noncanonical",
    }:
        raise _reject("FORM_SAFETY_INVALID")
    source = _exact(form["source"], {"source_class", "source_timestamp"}, "form.source")
    if source["source_class"] != "embedded_research_form_v1":
        raise _reject("FORM_SOURCE_INVALID")
    source_time = _timestamp(source["source_timestamp"], "form.source_timestamp")
    cutoff = _timestamp(form["cutoff_timestamp"], "form.cutoff_timestamp")
    if not source_time < cutoff:
        raise _reject("FORM_SOURCE_NOT_PRE_CUTOFF")
    target = _exact(
        form["target"],
        {"race_identity_sha256", "race_id", "race_date", "venue", "distance_m", "grade"},
        "form.target",
    )
    race = evidence["race"]
    if (
        target["race_identity_sha256"] != evidence["race_identity_sha256"]
        or target["race_id"] != race["race_id"]
        or target["race_date"] != race["race_date"]
        or target["venue"] != race["venue"]
    ):
        raise _reject("FORM_TARGET_BINDING_MISMATCH")
    try:
        target_date = date.fromisoformat(target["race_date"])
    except (TypeError, ValueError) as exc:
        raise _reject("FORM_FIELD_INVALID", field="target.race_date") from exc
    _finite(target["distance_m"], "target.distance_m", minimum=1.0)
    target_grade = _grade(target["grade"], "target.grade")
    assert target_grade is not None
    target_context = {**dict(target), "distance_m": float(target["distance_m"]), "grade": target_grade}
    rows_by_box: dict[int, list[dict[str, Any]]] = {}
    if not isinstance(form["runners"], list) or len(form["runners"]) < 2:
        raise _reject("FORM_RUNNER_SET_INVALID")
    last_box = 0
    for runner_index, runner_value in enumerate(form["runners"]):
        runner = _exact(runner_value, {"box_number", "display_name", "history"}, f"form.runners[{runner_index}]")
        box = runner["box_number"]
        if isinstance(box, bool) or not isinstance(box, int) or not 1 <= box <= 10 or box <= last_box:
            raise _reject("FORM_RUNNER_ORDER_INVALID")
        last_box = box
        if not isinstance(runner["display_name"], str) or not runner["display_name"].strip() or runner["display_name"] != runner["display_name"].strip():
            raise _reject("FORM_RUNNER_SET_INVALID")
        history = runner["history"]
        if not isinstance(history, list):
            raise _reject("FORM_HISTORY_INVALID", box=box)
        normalized_history: list[dict[str, Any]] = []
        prior_time: datetime | None = None
        for history_index, history_value in enumerate(history):
            row = _exact(history_value, set(_HISTORY_FIELDS), f"form.runners[{runner_index}].history[{history_index}]")
            if not isinstance(row["prior_race_id"], str) or not row["prior_race_id"].strip() or row["prior_race_id"] == race["race_id"]:
                raise _reject("FORM_HISTORY_INVALID", box=box)
            event_time = _timestamp(row["event_timestamp"], "history.event_timestamp")
            if prior_time is not None and event_time <= prior_time:
                raise _reject("FORM_HISTORY_ORDER_INVALID", box=box)
            if not event_time < cutoff:
                raise _reject("FORM_HISTORY_AFTER_CUTOFF", box=box)
            prior_time = event_time
            try:
                history_date = date.fromisoformat(row["race_date"])
            except (TypeError, ValueError) as exc:
                raise _reject("FORM_FIELD_INVALID", field="history.race_date") from exc
            if history_date >= target_date:
                raise _reject("FORM_HISTORY_NOT_PRIOR", box=box)
            venue = _grade(row["venue"], "history.venue")
            grade = _grade(row["grade"], "history.grade", nullable=True)
            distance = None if row["distance_m"] is None else _finite(row["distance_m"], "history.distance_m", minimum=1.0)
            finish = row["prior_finish"]
            if isinstance(finish, bool) or not isinstance(finish, int) or not 1 <= finish <= 10:
                raise _reject("FORM_FIELD_INVALID", field="history.prior_finish")
            margin = None if row["prior_margin"] is None else _finite(row["prior_margin"], "history.prior_margin")
            normalized_history.append(
                {
                    "prior_race_id": row["prior_race_id"],
                    "event_timestamp": event_time,
                    "race_date": history_date,
                    "venue": venue,
                    "distance_m": distance,
                    "grade": grade,
                    "prior_finish": finish,
                    "prior_margin": margin,
                }
            )
        rows_by_box[box] = normalized_history
    return cutoff, target_context, rows_by_box


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else math.fsum(values) / len(values)


def _rate(rows: Sequence[Mapping[str, Any]], predicate: Any) -> float:
    return 0.0 if not rows else sum(1 for row in rows if predicate(row)) / len(rows)


def _features(history: Sequence[Mapping[str, Any]], target: Mapping[str, Any], target_date: date) -> dict[str, float | None]:
    recent = list(history[-5:])
    recent3 = list(history[-3:])
    finishes = [float(row["prior_finish"]) for row in history]
    recent_finishes = [float(row["prior_finish"]) for row in recent]
    recent3_finishes = [float(row["prior_finish"]) for row in recent3]
    margins = [float(row["prior_margin"]) for row in recent if row["prior_margin"] is not None]
    same_venue = [row for row in history if row["venue"] == target["venue"].upper()]
    same_distance = [
        row for row in history
        if row["distance_m"] is not None and abs(row["distance_m"] - target["distance_m"]) <= 50.0
    ]
    same_grade = [row for row in history if row["grade"] == target["grade"]]
    last_date = history[-1]["race_date"] if history else None
    values: dict[str, float | None] = {
        "prior_start_count": float(len(history)),
        "days_since_last_start": None if last_date is None else float((target_date - last_date).days),
        "recent_finish_mean_3": _mean(recent3_finishes),
        "recent_finish_best_5": min(recent_finishes) if recent_finishes else None,
        "recent_win_rate_5": _rate(recent, lambda row: row["prior_finish"] == 1),
        "recent_place_rate_5": _rate(recent, lambda row: row["prior_finish"] <= 3),
        "recent_avg_margin_5": _mean(margins),
        "career_win_rate": _rate(history, lambda row: row["prior_finish"] == 1),
        "career_place_rate": _rate(history, lambda row: row["prior_finish"] <= 3),
        "career_avg_finish": _mean(finishes),
        "starts_same_venue": float(len(same_venue)),
        "win_rate_same_venue": _rate(same_venue, lambda row: row["prior_finish"] == 1),
        "starts_same_distance": float(len(same_distance)),
        "win_rate_same_distance": _rate(same_distance, lambda row: row["prior_finish"] == 1),
        "same_grade_start_count": float(len(same_grade)),
        "same_grade_win_rate": _rate(same_grade, lambda row: row["prior_finish"] == 1),
    }
    assert set(values) == _FEATURE_FIELDS
    return values


def _runner_id(race_id: str, box: int, name: str) -> str:
    token = re.sub(r"[^A-Z0-9]", "", name.upper())
    if not token:
        raise _reject("RUNNER_ID_INVALID", box=box)
    return f"{race_id}|box:{box}|dog:{token}"


def _scorer_runner_set_sha256(runner_ids: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(runner_ids)) + "\n").encode("utf-8")).hexdigest()


def _feature_payload(
    *, form_sha256: str, cutoff: datetime, target: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], identity: Mapping[str, str]
) -> dict[str, Any]:
    return {
        "schema_version": "manual_research_feature_rows_v1",
        "adapter_version": FEATURE_ADAPTER_VERSION,
        "adapter_sha256": identity["feature_adapter_sha256"],
        "form_sha256": form_sha256,
        "cutoff_timestamp": cutoff.isoformat(),
        "target": dict(target),
        "runners": [dict(row) for row in rows],
    }


def _prediction_payload(
    *, evidence: SealedManualEvidence, form_sha256: str, config_sha256: str, model: FrozenResidualModel,
    identity: Mapping[str, str], feature_payload: Mapping[str, Any], feature_sha256: str, record: Mapping[str, Any],
    config: Mapping[str, Any], cutoff: datetime,
) -> dict[str, Any]:
    bundle = evidence.bundle
    normalized = evidence.normalized_odds
    race = bundle["race"]
    by_box = {int(row["box_number"]): row for row in feature_payload["runners"]}
    predictions = []
    for row in record["predictions"]:
        feature_row = by_box[int(row["box_number"])]
        predictions.append(
            {
                "runner_id": row["runner_id"],
                "box_number": int(row["box_number"]),
                "display_name": row["dog_name"],
                "strict_win_odds": float(row["strict_win_odds"]),
                "market_probability": float(row["market_probability"]),
                "half_probability": float(row["half_probability"]),
                "full_probability": float(row["full_probability"]),
                "features": dict(feature_row["values"]),
            }
        )
    ranked = sorted(predictions, key=lambda row: (-row["full_probability"], row["box_number"]))
    ranked = [{**row, "rank": index} for index, row in enumerate(ranked, start=1)]
    seed = {
        "schema_version": PREDICTION_SCHEMA,
        "evidence_bundle_id": bundle["bundle_id"],
        "race_identity_sha256": bundle["race_identity_sha256"],
        "evidence_manifest_sha256": evidence.manifest_sha256,
        "form_sha256": form_sha256,
        "config_sha256": config_sha256,
        "model_sha256": model.model_sha256,
        "model_manifest_sha256": model.manifest_sha256,
        "feature_sha256": feature_sha256,
        "effective_state_sha256": record["effective_state_sha256"],
        "implementation": identity,
        "runner_set_sha256": bundle["runner_set_sha256"],
        "cutoff_timestamp": cutoff.isoformat(),
    }
    bundle_id = hashlib.sha256(canonical_bytes(seed)).hexdigest()
    return {
        "schema_version": PREDICTION_SCHEMA,
        "bundle_id": bundle_id,
        "safety": {
            "research_only": True,
            "canonical": False,
            "phase7_excluded": True,
            "phase7_eligible": False,
            "phase7_exclusion_reason": "manual_research_only_noncanonical",
        },
        "scorer_id": SCORER_VERSION,
        "race": dict(race),
        "race_identity_sha256": bundle["race_identity_sha256"],
        "runner_set_sha256": bundle["runner_set_sha256"],
        "timing": {
            "sealed_cutoff_timestamp": cutoff.isoformat(),
            "odds_capture_timestamp": bundle["timing"]["capture_timestamp"],
            "source_timestamp": bundle["timing"]["source_timestamp"],
            "scheduled_start": race["scheduled_start"],
        },
        "evidence": {
            "bundle_id": bundle["bundle_id"],
            "manifest_sha256": evidence.manifest_sha256,
            "odds_sha256": normalized["odds_sha256"],
        },
        "form": {
            "schema_version": EMBEDDED_FORM_SCHEMA,
            "sha256": form_sha256,
            "feature_sha256": feature_sha256,
        },
        "features": {
            "schema_version": feature_payload["schema_version"],
            "adapter_version": FEATURE_ADAPTER_VERSION,
            "adapter_sha256": identity["feature_adapter_sha256"],
            "form_sha256": form_sha256,
            "cutoff_timestamp": cutoff.isoformat(),
            "target": dict(feature_payload["target"]),
            "sha256": feature_sha256,
            "runners": list(feature_payload["runners"]),
        },
        "model": {
            "model_id": "market_form_residual_v1",
            "model_sha256": model.model_sha256,
            "manifest_sha256": model.manifest_sha256,
            "effective_state_schema": EFFECTIVE_STATE_SCHEMA,
            "effective_state_sha256": record["effective_state_sha256"],
        },
        "config": {
            "schema_version": config["schema_version"],
            "sha256": config_sha256,
        },
        "implementation": dict(identity),
        "ranking": {"primary_probability": "full_probability", "tie_break": "box_ascending"},
        "predictions": ranked,
    }


def _plain_directory(path: Path, parent: Path | None = None) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        return False
    if parent is not None:
        try:
            path.resolve(strict=True).relative_to(parent.resolve(strict=True))
        except (OSError, ValueError):
            return False
    return True


def _write_member(root: Path, relative: str, raw: bytes) -> None:
    path = root / relative
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directories + filenames:
            if (current_path / name).is_symlink():
                raise _reject("UNSAFE_OUTPUT_PATH")
        for name in filenames:
            path = current_path / name
            os.chmod(path, 0o400)
        os.chmod(current_path, 0o500)
        descriptor = os.open(current_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _safe_remove_stage(stage: Path, root: Path) -> None:
    if not stage.exists() and not stage.is_symlink():
        return
    if stage.is_symlink() or not _plain_directory(stage, parent=root):
        raise _reject("UNSAFE_OUTPUT_PATH")
    for current, directories, filenames in os.walk(stage, topdown=False, followlinks=False):
        for name in filenames:
            (Path(current) / name).unlink()
        for name in directories:
            child = Path(current) / name
            if child.is_symlink():
                raise _reject("UNSAFE_OUTPUT_PATH")
            child.rmdir()
    stage.rmdir()


def _publish_prediction(
    prediction: Mapping[str, Any], *, output_root: Path, stage_hook: Any = None
) -> tuple[Path, bool]:
    if not _plain_directory(output_root):
        raise _reject("OUTPUT_ROOT_UNSAFE")
    bundle_id = prediction["bundle_id"]
    destination = output_root / bundle_id
    stage = output_root / f".tmp-{bundle_id}"
    lock_path = output_root / ".research-prediction.lock"
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if destination.exists() or destination.is_symlink():
            return destination, True
        _safe_remove_stage(stage, output_root)
        stage.mkdir(mode=0o700)
        if stage_hook:
            stage_hook("stage_created", stage)
        prediction_raw = canonical_bytes(prediction)
        manifest = {
            "schema_version": PREDICTION_MANIFEST_SCHEMA,
            "safety": prediction["safety"],
            "bundle_id": bundle_id,
            "publication_protocol": PUBLISH_PROTOCOL,
            "members": [
                {"path": "prediction.json", "bytes": len(prediction_raw), "sha256": hashlib.sha256(prediction_raw).hexdigest()}
            ],
        }
        _write_member(stage, "prediction.json", prediction_raw)
        if stage_hook:
            stage_hook("prediction_written", stage)
        manifest_raw = canonical_bytes(manifest)
        _write_member(stage, "manifest.json", manifest_raw)
        if stage_hook:
            stage_hook("manifest_written", stage)
        _fsync_tree(stage)
        os.rename(stage, destination)
        descriptor = os.open(output_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if stage_hook:
            stage_hook("renamed", destination)
        return destination, False
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _validate_prediction_document(prediction: Mapping[str, Any], expected: ResearchPredictionExpectations) -> dict[str, Any]:
    if set(prediction) != {
        "schema_version", "bundle_id", "safety", "scorer_id", "race", "race_identity_sha256", "runner_set_sha256", "timing", "evidence", "form", "features", "model", "config", "implementation", "ranking", "predictions"
    }:
        raise _reject("PREDICTION_SCHEMA_INVALID")
    if prediction["schema_version"] != PREDICTION_SCHEMA or prediction["safety"] != {
        "research_only": True, "canonical": False, "phase7_excluded": True, "phase7_eligible": False, "phase7_exclusion_reason": "manual_research_only_noncanonical"
    }:
        raise _reject("PREDICTION_SAFETY_INVALID")
    _sha(prediction["bundle_id"], "bundle_id")
    for value, field in (
        (expected.evidence_bundle_id, "evidence_bundle_id"),
        (expected.evidence_manifest_sha256, "evidence_manifest_sha256"),
        (expected.form_sha256, "form_sha256"),
        (expected.config_sha256, "config_sha256"),
        (expected.model_sha256, "model_sha256"),
        (expected.model_manifest_sha256, "model_manifest_sha256"),
    ):
        _sha(value, field)
    if (
        set(prediction["evidence"]) != {"bundle_id", "manifest_sha256", "odds_sha256"}
        or prediction["evidence"]["bundle_id"] != expected.evidence_bundle_id
        or prediction["evidence"]["manifest_sha256"] != expected.evidence_manifest_sha256
            or prediction["form"]["sha256"] != expected.form_sha256
            or prediction["config"]["sha256"] != expected.config_sha256
            or prediction["model"]["model_sha256"] != expected.model_sha256
            or prediction["model"]["manifest_sha256"] != expected.model_manifest_sha256
            or prediction["race_identity_sha256"] != expected.race_identity_sha256
            or prediction["runner_set_sha256"] != expected.runner_set_sha256
            or prediction["evidence"]["odds_sha256"] != expected.odds_sha256
            or prediction["model"]["effective_state_sha256"] != expected.effective_state_sha256
        or prediction["implementation"] != _validate_identity(expected.implementation)
    ):
        raise _reject("PREDICTION_IDENTITY_MISMATCH")
    _sha(prediction["race_identity_sha256"], "race_identity_sha256")
    _sha(prediction["runner_set_sha256"], "runner_set_sha256")
    _sha(prediction["evidence"]["odds_sha256"], "odds_sha256")
    if hashlib.sha256(canonical_bytes(prediction["race"])).hexdigest() != prediction["race_identity_sha256"]:
        raise _reject("RACE_IDENTITY_MISMATCH")
    if set(prediction["race"]) != {"url", "race_id", "race_date", "venue", "venue_slug", "race_number", "scheduled_start"}:
        raise _reject("PREDICTION_RACE_INVALID")
    timing = _exact(prediction["timing"], {"sealed_cutoff_timestamp", "odds_capture_timestamp", "source_timestamp", "scheduled_start"}, "timing")
    cutoff = _timestamp(timing["sealed_cutoff_timestamp"], "sealed_cutoff_timestamp")
    if (
        timing["odds_capture_timestamp"] != timing["sealed_cutoff_timestamp"]
        or timing["scheduled_start"] != prediction["race"]["scheduled_start"]
        or timing["sealed_cutoff_timestamp"] != expected.cutoff_timestamp
        or timing["scheduled_start"] != expected.scheduled_start
        or not _timestamp(timing["source_timestamp"], "source_timestamp") <= cutoff
        or not cutoff < _timestamp(timing["scheduled_start"], "scheduled_start")
    ):
        raise _reject("PREDICTION_TIMING_INVALID")
    if set(prediction["form"]) != {"schema_version", "sha256", "feature_sha256"} or prediction["form"]["schema_version"] != EMBEDDED_FORM_SCHEMA:
        raise _reject("FORM_BINDING_INVALID")
    if set(prediction["model"]) != {"model_id", "model_sha256", "manifest_sha256", "effective_state_schema", "effective_state_sha256"} or prediction["model"]["model_id"] != "market_form_residual_v1" or prediction["model"]["effective_state_schema"] != EFFECTIVE_STATE_SCHEMA:
        raise _reject("MODEL_BINDING_INVALID")
    if set(prediction["config"]) != {"schema_version", "sha256"}:
        raise _reject("CONFIG_BINDING_INVALID")
    if prediction["scorer_id"] != SCORER_VERSION or prediction["ranking"] != {"primary_probability": "full_probability", "tie_break": "box_ascending"}:
        raise _reject("PREDICTION_CONTRACT_INVALID")
    if not isinstance(prediction["predictions"], list) or len(prediction["predictions"]) < 2:
        raise _reject("PREDICTION_RUNNER_SET_INVALID")
    seen_boxes: set[int] = set()
    previous_rank = 0
    previous_probability: float | None = None
    previous_box: int | None = None
    for index, row_value in enumerate(prediction["predictions"]):
        row = _exact(row_value, {"runner_id", "box_number", "display_name", "strict_win_odds", "market_probability", "half_probability", "full_probability", "features", "rank"}, f"predictions[{index}]")
        box = row["box_number"]
        if isinstance(box, bool) or not isinstance(box, int) or not 1 <= box <= 10 or box in seen_boxes:
            raise _reject("PREDICTION_RUNNER_SET_INVALID")
        seen_boxes.add(box)
        if not isinstance(row["runner_id"], str) or not row["runner_id"] or not isinstance(row["display_name"], str) or not row["display_name"].strip() or row["display_name"] != row["display_name"].strip():
            raise _reject("PREDICTION_RUNNER_SET_INVALID")
        _finite(row["strict_win_odds"], "strict_win_odds", minimum=1.000000000000001)
        if row["rank"] != index + 1 or row["rank"] <= previous_rank:
            raise _reject("RANKING_INVALID")
        full = _finite(row["full_probability"], "full_probability")
        for field in ("market_probability", "half_probability"):
            probability = _finite(row[field], field)
            if not 0.0 <= probability <= 1.0:
                raise _reject("PROBABILITY_INVALID", field=field)
        if not 0.0 <= full <= 1.0:
            raise _reject("PROBABILITY_INVALID", field="full_probability")
        if previous_probability is not None and (full > previous_probability or (full == previous_probability and previous_box is not None and box < previous_box)):
            raise _reject("RANKING_INVALID")
        previous_probability, previous_box, previous_rank = full, box, row["rank"]
        features = row["features"]
        if not isinstance(features, Mapping) or set(features) != _FEATURE_FIELDS:
            raise _reject("FEATURES_INVALID")
        for field, value in features.items():
            if value is not None:
                _finite(value, f"features.{field}")
    feature_payload = prediction["features"]
    if set(feature_payload) != {"schema_version", "adapter_version", "adapter_sha256", "form_sha256", "cutoff_timestamp", "target", "sha256", "runners"}:
        raise _reject("FEATURES_INVALID")
    if not isinstance(feature_payload["runners"], list):
        raise _reject("FEATURES_INVALID")
    feature_by_runner = {item.get("runner_id"): item for item in feature_payload["runners"] if isinstance(item, Mapping)}
    if len(feature_by_runner) != len(prediction["predictions"]):
        raise _reject("FEATURES_INVALID")
    for row in prediction["predictions"]:
        feature_row = feature_by_runner.get(row["runner_id"])
        if not isinstance(feature_row, Mapping) or set(feature_row) != {"runner_id", "box_number", "display_name", "values"} or feature_row["box_number"] != row["box_number"] or feature_row["display_name"] != row["display_name"] or feature_row["values"] != row["features"]:
            raise _reject("FEATURE_PREDICTION_MISMATCH")
    if any(not math.isclose(math.fsum(row[field] for row in prediction["predictions"]), 1.0, rel_tol=0.0, abs_tol=1e-12) for field in ("market_probability", "half_probability", "full_probability")):
        raise _reject("PROBABILITY_NOT_NORMALIZED")
    if feature_payload["sha256"] != prediction["form"]["feature_sha256"]:
        raise _reject("FEATURE_HASH_MISMATCH")
    feature_copy = dict(feature_payload)
    feature_sha = feature_copy.pop("sha256")
    expected_feature_content = {
        "schema_version": "manual_research_feature_rows_v1",
        "adapter_version": FEATURE_ADAPTER_VERSION,
        "adapter_sha256": prediction["implementation"]["feature_adapter_sha256"],
        "form_sha256": expected.form_sha256,
        "cutoff_timestamp": prediction["timing"]["sealed_cutoff_timestamp"],
        "target": feature_payload["target"],
        "runners": feature_payload["runners"],
    }
    if feature_copy != expected_feature_content or hashlib.sha256(canonical_bytes(feature_copy)).hexdigest() != feature_sha:
        raise _reject("FEATURE_HASH_MISMATCH")
    _sha(feature_sha, "feature_sha256")
    if expected.feature_sha256 is not None and feature_sha != expected.feature_sha256:
        raise _reject("FEATURE_HASH_MISMATCH")
    if _contains_forbidden_prediction_key(prediction):
        raise _reject("FORBIDDEN_PREDICTION_FIELD")
    return dict(prediction)


def _contains_forbidden_prediction_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(str(key).strip().lower() in _OUTCOME_WORDS or _contains_forbidden_prediction_key(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_forbidden_prediction_key(item) for item in value)
    return False


def _read_regular_member(path: Path, *, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise _reject("UNSAFE_OUTPUT_PATH", path=str(path)) from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
            raise _reject("PARTIAL_OR_EXTRA_OUTPUT", path=str(path))
        chunks: list[bytes] = []
        remaining = max_bytes
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        if remaining == 0 and os.read(descriptor, 1):
            raise _reject("PARTIAL_OR_EXTRA_OUTPUT", path=str(path))
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def verify_research_prediction_bundle(bundle_dir: Path, *, output_root: Path, expected: ResearchPredictionExpectations) -> VerifiedResearchPrediction:
    if bundle_dir.parent != output_root or not _plain_directory(output_root) or not _plain_directory(bundle_dir, parent=output_root):
        raise _reject("UNSAFE_OUTPUT_PATH")
    for current, directories, filenames in os.walk(bundle_dir, followlinks=False):
        if directories:
            raise _reject("PARTIAL_OR_EXTRA_OUTPUT")
        for name in directories + filenames:
            if (Path(current) / name).is_symlink():
                raise _reject("UNSAFE_OUTPUT_PATH")
    files = {path.relative_to(bundle_dir).as_posix() for path in bundle_dir.rglob("*") if path.is_file()}
    if files != {"manifest.json", "prediction.json"}:
        raise _reject("PARTIAL_OR_EXTRA_OUTPUT")
    manifest_raw = _read_regular_member(bundle_dir / "manifest.json", max_bytes=256 * 1024)
    prediction_raw = _read_regular_member(bundle_dir / "prediction.json", max_bytes=4 * 1024 * 1024)
    try:
        manifest = parse_canonical_json(manifest_raw, max_bytes=256 * 1024)
        prediction = parse_canonical_json(prediction_raw, max_bytes=4 * 1024 * 1024)
    except Exception as exc:
        raise _reject("PREDICTION_NOT_CANONICAL") from exc
    manifest = _exact(manifest, {"schema_version", "safety", "bundle_id", "publication_protocol", "members"}, "manifest")
    if manifest["schema_version"] != PREDICTION_MANIFEST_SCHEMA or manifest["publication_protocol"] != PUBLISH_PROTOCOL or manifest["bundle_id"] != bundle_dir.name or manifest["safety"] != prediction["safety"]:
        raise _reject("MANIFEST_INVALID")
    if manifest["members"] != [{"path": "prediction.json", "bytes": len(prediction_raw), "sha256": hashlib.sha256(prediction_raw).hexdigest()}]:
        raise _reject("MANIFEST_INVALID")
    validated = _validate_prediction_document(prediction, expected)
    seed = {
        "schema_version": PREDICTION_SCHEMA,
        "evidence_bundle_id": validated["evidence"]["bundle_id"],
        "race_identity_sha256": validated["race_identity_sha256"],
        "evidence_manifest_sha256": validated["evidence"]["manifest_sha256"],
        "form_sha256": validated["form"]["sha256"],
        "config_sha256": validated["config"]["sha256"],
        "model_sha256": validated["model"]["model_sha256"],
        "model_manifest_sha256": validated["model"]["manifest_sha256"],
        "feature_sha256": validated["features"]["sha256"],
        "effective_state_sha256": validated["model"]["effective_state_sha256"],
        "implementation": validated["implementation"],
        "runner_set_sha256": validated["runner_set_sha256"],
        "cutoff_timestamp": validated["timing"]["sealed_cutoff_timestamp"],
    }
    if hashlib.sha256(canonical_bytes(seed)).hexdigest() != validated["bundle_id"]:
        raise _reject("PREDICTION_BUNDLE_ID_MISMATCH")
    return VerifiedResearchPrediction(bundle_dir, validated, dict(manifest), hashlib.sha256(manifest_raw).hexdigest(), False)


def score_verified_manual_evidence(
    *,
    sealed_bundle_dir: Path,
    run_dir: Path,
    evidence_expected: SealExpectations,
    evidence_identity: SealingIdentity,
    embedded_form_bytes: bytes,
    form_sha256: str,
    config_bytes: bytes,
    config_sha256: str,
    frozen_model: FrozenResidualModel,
    expected_model_sha256: str,
    expected_model_manifest_sha256: str,
    scoring_identity: ResearchScoringIdentity,
    output_root: Path,
    stage_hook: Any = None,
) -> VerifiedResearchPrediction:
    """Verify GHU-052, score one bounded race, and publish atomically."""

    evidence = verify_manual_evidence_bundle(
        sealed_bundle_dir,
        run_dir=run_dir,
        expected=evidence_expected,
        expected_identity=evidence_identity,
    )
    identity = _validate_identity(scoring_identity)
    if not isinstance(frozen_model, FrozenResidualModel):
        raise _reject("MODEL_INVALID")
    if frozen_model.model_sha256 != _sha(expected_model_sha256, "expected_model_sha256") or frozen_model.manifest_sha256 != _sha(expected_model_manifest_sha256, "expected_model_manifest_sha256"):
        raise _reject("MODEL_HASH_DRIFT")
    if evidence.bundle["producer"]["model_sha256"] != frozen_model.model_sha256:
        raise _reject("MODEL_HASH_DRIFT")
    config, actual_config_sha256 = _validate_config(config_bytes, config_sha256, frozen_model)
    form = _canonical_form(embedded_form_bytes, form_sha256)
    cutoff, target, rows_by_box = _target_and_form_rows(form, evidence.bundle)
    if cutoff.isoformat() != evidence.bundle["timing"]["capture_timestamp"]:
        raise _reject("FORM_CUTOFF_MISMATCH")
    evidence_runners = evidence.bundle["runner_set"]
    odds_runners = evidence.normalized_odds["runners"]
    if [dict(row) for row in evidence_runners] != [
        {key: value for key, value in row.items() if key != "decimal_odds"} for row in odds_runners
    ]:
        raise _reject("RUNNER_ORDER_MISMATCH")
    form_runners = {int(row["box_number"]): row for row in form["runners"]}
    if set(form_runners) != {int(row["box_number"]) for row in evidence_runners}:
        raise _reject("RUNNER_SET_MISMATCH")
    feature_rows = []
    scorer_runners = []
    runner_ids: list[str] = []
    for evidence_runner, odds_runner in zip(evidence_runners, odds_runners, strict=True):
        box = int(evidence_runner["box_number"])
        form_runner = form_runners[box]
        if form_runner["display_name"] != evidence_runner["display_name"]:
            raise _reject("RUNNER_NAME_MISMATCH", box=box)
        runner_id = _runner_id(evidence.bundle["race"]["race_id"], box, form_runner["display_name"])
        runner_ids.append(runner_id)
        values = _features(rows_by_box[box], target, date.fromisoformat(target["race_date"]))
        feature_rows.append({"runner_id": runner_id, "box_number": box, "display_name": form_runner["display_name"], "values": values})
    feature_payload = _feature_payload(form_sha256=form_sha256, cutoff=cutoff, target=target, rows=feature_rows, identity=identity)
    feature_sha256 = hashlib.sha256(canonical_bytes(feature_payload)).hexdigest()
    for feature_row, odds_runner in zip(feature_rows, odds_runners, strict=True):
        scorer_runners.append(
            {
                "race_id": evidence.bundle["race"]["race_id"],
                "runner_id": feature_row["runner_id"],
                "box_number": feature_row["box_number"],
                "dog_name": feature_row["display_name"],
                "strict_win_odds": float(odds_runner["decimal_odds"]),
                "features": feature_row["values"],
                "feature_source_sha256": feature_sha256,
                "odds_source_sha256": evidence.bundle["normalized_odds"]["odds_sha256"],
                "feature_freeze_timestamp": form["source"]["source_timestamp"],
                "odds_capture_timestamp": evidence.bundle["timing"]["capture_timestamp"],
            }
        )
    provenance = {
        "race_id": evidence.bundle["race"]["race_id"],
        "expected_runner_ids": sorted(runner_ids),
        "runner_set_sha256": _scorer_runner_set_sha256(runner_ids),
        "jump_timestamp": evidence.bundle["race"]["scheduled_start"],
        "score_timestamp": evidence.bundle["timing"]["capture_timestamp"],
    }
    try:
        record = score_race(frozen_model, scorer_runners, provenance)
    except ResidualContractError as exc:
        raise _reject("SCORING_BLOCKED", reason=str(exc)) from exc
    if record["schema_version"] != SHADOW_RECORD_SCHEMA:
        raise _reject("SCORING_SCHEMA_MISMATCH")
    prediction = _prediction_payload(
        evidence=evidence,
        form_sha256=form_sha256,
        config_sha256=actual_config_sha256,
        model=frozen_model,
        identity=identity,
        feature_payload=feature_payload,
        feature_sha256=feature_sha256,
        record=record,
        config=config,
        cutoff=cutoff,
    )
    expected = ResearchPredictionExpectations(
        evidence_bundle_id=evidence.bundle["bundle_id"],
        evidence_manifest_sha256=evidence.manifest_sha256,
        race_identity_sha256=evidence.bundle["race_identity_sha256"],
        form_sha256=form_sha256,
        config_sha256=actual_config_sha256,
        model_sha256=frozen_model.model_sha256,
        model_manifest_sha256=frozen_model.manifest_sha256,
        runner_set_sha256=evidence.bundle["runner_set_sha256"],
        odds_sha256=evidence.bundle["normalized_odds"]["odds_sha256"],
        cutoff_timestamp=evidence.bundle["timing"]["capture_timestamp"],
        scheduled_start=evidence.bundle["race"]["scheduled_start"],
        effective_state_sha256=record["effective_state_sha256"],
        implementation=scoring_identity,
        feature_sha256=feature_sha256,
    )
    destination, replayed = _publish_prediction(prediction, output_root=output_root, stage_hook=stage_hook)
    verified = verify_research_prediction_bundle(destination, output_root=output_root, expected=expected)
    return VerifiedResearchPrediction(destination, verified.prediction, verified.manifest, verified.manifest_sha256, replayed)


__all__ = [
    "EMBEDDED_FORM_SCHEMA",
    "FEATURE_ADAPTER_VERSION",
    "PREDICTION_MANIFEST_SCHEMA",
    "PREDICTION_SCHEMA",
    "ManualResearchScoringRejected",
    "ResearchPredictionExpectations",
    "ResearchScoringIdentity",
    "VerifiedResearchPrediction",
    "build_research_scoring_identity",
    "score_verified_manual_evidence",
    "verify_research_prediction_bundle",
]
