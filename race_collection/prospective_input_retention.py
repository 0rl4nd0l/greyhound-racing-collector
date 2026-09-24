"""Input-only retention primitive used by the existing collector callback.

The caller must supply an approved, authenticated pre-race input inventory.
This function proves retention, not semantic feature completeness/admissibility.
It is deliberately separate from prediction and research-population admission.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from src.predictor.on_demand import PredictionBlocked, seal_history_database

REQUIRED_ROLES = frozenset({
    "normalized_form", "form_metadata", "raw_form", "primary_page",
    "primary_page_receipt", "exact_odds_receipt", "odds_report",
    "model", "model_manifest", "configuration", "feature_schema",
    "generator_source_archive", "environment_lock",
    "feature_replay_worker",
})


class RetentionRejected(ValueError):
    """A finite outcome-free reason; never include source rows in errors."""


def failure_code(error: Exception) -> str:
    if isinstance(error, RetentionRejected):
        return str(error)
    if isinstance(error, PredictionBlocked):
        return error.code
    return "RETENTION_PROCESSING_FAILED"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise RetentionRejected("TIMESTAMP_INVALID")
    return value


def retain_inputs(
    *,
    destination: Path,
    race_id: str,
    runner_names: Sequence[str],
    observed_at: datetime,
    prediction_cutoff: datetime,
    jump_at: datetime,
    history_source: Path,
    files: Mapping[str, tuple[Path, str]],
    clock: Callable[[], datetime] = _now,
    generate_features: bool = False,
    max_bundle_bytes: int | None = None,
    authorization_config_sha256: str | None = None,
) -> dict:
    """Archive exact supplied inputs plus an as-held pre-cutoff history snapshot.

    All required files are hash checked. A complete manifest is published only
    when acquisition finishes before the planned prediction cutoff and jump.
    The unchanged history sealer excludes target/same-date/future DB records.
    Protected earlier dates still require caller authority before this call.
    No retrospective timestamp is accepted as a substitute for the real clock.
    """
    start = _aware(clock())
    if not _aware(observed_at) <= start < _aware(prediction_cutoff) < _aware(jump_at):
        raise RetentionRejected("NOT_PROSPECTIVE")
    if not race_id or not runner_names or any(not n.strip() for n in runner_names):
        raise RetentionRejected("IDENTITY_INCOMPLETE")
    if set(files) != REQUIRED_ROLES:
        raise RetentionRejected("INPUT_INVENTORY_INCOMPLETE")
    if destination.exists() or destination.is_symlink():
        raise RetentionRejected("DESTINATION_EXISTS")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive reservation prevents a second caller replacing a prior attempt.
    destination.mkdir(mode=0o700)
    stage = None
    published = False
    try:
        stage = Path(tempfile.mkdtemp(prefix=".input-stage-", dir=destination.parent))
        manifest_files = {}
        for role, (source, expected) in sorted(files.items()):
            if not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise RetentionRejected("SOURCE_HASH_INVALID")
            if source.is_symlink() or not source.is_file():
                raise RetentionRejected("SOURCE_UNAVAILABLE")
            raw = source.read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise RetentionRejected("SOURCE_HASH_MISMATCH")
            relative = Path("inputs") / ("form" if role in {"normalized_form", "form_metadata"} else role) / source.name
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            manifest_files[role] = {
                "path": relative.as_posix(), "sha256": expected,
                "original_path": str(source.absolute()), "bytes": len(raw),
            }
        # Reuse the existing verified-copy/immutable-SQLite cutoff implementation.
        history = seal_history_database(
            source=history_source, target=stage / "history.db",
            target_race_id=race_id, cutoff=jump_at, runner_names=runner_names,
            runner_scope_only=True,
        )
        history_raw = (json.dumps(history, sort_keys=True, separators=(",", ":")) + "\n").encode()
        with (stage / "history_seal.json").open("xb") as stream:
            stream.write(history_raw)
            stream.flush()
            os.fsync(stream.fileno())
        if generate_features:
            from race_collection.retained_feature_replay import generate_retained_features

            generated = generate_retained_features(stage, manifest_files)
            with (stage / "feature_values.json").open("xb") as stream:
                stream.write(generated)
                stream.flush()
                os.fsync(stream.fileno())
        bundle_bytes = sum(path.stat().st_size for path in stage.rglob("*") if path.is_file())
        if max_bundle_bytes is not None and bundle_bytes > max_bundle_bytes:
            raise RetentionRejected("BUNDLE_STORAGE_BUDGET_EXCEEDED")
        completed = _aware(clock())
        if not start <= completed < prediction_cutoff:
            raise RetentionRejected("CUTOFF_PASSED_DURING_RETENTION")
        manifest = {
            "schema_version": "prospective_input_retention_v1",
            "authorization_config_sha256": authorization_config_sha256,
            "status": "INPUTS_PENDING_COMPLETION",
            "race_id": race_id,
            "capture_started_at": start.isoformat(),
            "capture_completed_at": completed.isoformat(),
            "source_observed_at": observed_at.isoformat(),
            "prediction_cutoff": prediction_cutoff.isoformat(),
            "jump_at": jump_at.isoformat(),
            "files": manifest_files,
            "history": {
                "path": "history.db", "sha256": history["sealed_sha256"],
                "source_sha256": history["source_sha256"],
                "cutoff_basis": history["cutoff_basis"],
                "scope": "active_runner_names_using_feature_loader_normalization",
            },
            "history_seal": {"path": "history_seal.json", "sha256": hashlib.sha256(history_raw).hexdigest()},
            "feature_values_sha256": hashlib.sha256(generated).hexdigest() if generate_features else None,
            "bundle_bytes": bundle_bytes,
            "predictions_generated": False,
            "eligibility_assessed": False,
        }
        for child in stage.iterdir():
            shutil.move(str(child), destination / child.name)
        # The manifest alone is provisional. Only the completion receipt binds
        # a checked post-write timestamp to its digest; incomplete attempts fail.
        if not completed <= _aware(clock()) < prediction_cutoff:
            raise RetentionRejected("CUTOFF_PASSED_DURING_RETENTION")
        raw_manifest = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
        with (destination / "manifest.json").open("xb") as stream:
            stream.write(raw_manifest)
            stream.flush()
            os.fsync(stream.fileno())
        sealed_at = _aware(clock())
        if not completed <= sealed_at < prediction_cutoff:
            raise RetentionRejected("CUTOFF_PASSED_DURING_RETENTION")
        completion = {
            "status": "INPUTS_RETAINED_NOT_QUALIFIED",
            "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
            "inputs_sealed_at": sealed_at.isoformat(),
        }
        with (destination / "completion.json").open("xb") as stream:
            stream.write((json.dumps(completion, sort_keys=True) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        published = True
        return manifest
    finally:
        if stage is not None:
            shutil.rmtree(stage)
        if not published:
            shutil.rmtree(destination)
