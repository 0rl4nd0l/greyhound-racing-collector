#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "beautifulsoup4==4.13.4",
#   "numpy==1.26.4",
#   "requests==2.32.4",
#   "selenium==4.34.2",
#   "webdriver-manager==4.0.2",
# ]
# ///
"""Build one isolated, research-only prediction for an exact pre-jump race."""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from race_collection.manual_prediction_collector_request import (  # noqa: E402
    PROTOCOL_DIRECTORY,
    RECEIPT_READY,
    ManualPredictionCollectorProtocol,
    ProtocolRejected,
)
from scripts.predict_market_form_residual import (  # noqa: E402
    DEFAULT_EVIDENCE_ROOT,
    DEFAULT_RETAINED_EVIDENCE_ROOTS,
    FEATURE_GENERATOR_FILES,
    discover_race_artifacts,
    score_from_artifacts,
)
from scripts.predict_market_form_residual import (
    ManualPredictionError as CaptureHandoffError,
)
from scripts.refresh_prejump_upcoming import (  # noqa: E402
    _parse_race_jump_datetime,
    parse_current_time,
    stable_race_id,
    stable_race_id_variants,
    venue_exclusion_aliases,
)
from src.predictor.on_demand import (  # noqa: E402
    BLOCKER_STAGE_BY_CODE,
    Dependencies,
    PredictionBlocked,
    _job_id,
    _copy_exact,
    _write_canonical,
    build_prediction_bundle_manifest_v2,
    bundle_manifest,
    canonical_bytes,
    create_bundle,
    load_config,
    market_only_prediction,
    prediction_bundle_index_entry,
    publish_prediction_bundle_index_entry,
    receipt_from_handoff,
    resolve_model,
    seal_history_database,
    sealed_runner_set_sha256,
    sha256_bytes,
    sha256_file,
    verify_bundle,
    write_exact_bytes,
)
from utils.csv_metadata import canonical_thedogs_race_identity  # noqa: E402
from utils.csv_metadata import canonical_thedogs_venue_identity  # noqa: E402

DEFAULT_DB = ROOT / "greyhound_racing_data.db"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/on_demand_prediction_runs"
DEFAULT_CAPTURE_EVIDENCE_ROOTS = (
    DEFAULT_EVIDENCE_ROOT,
    *DEFAULT_RETAINED_EVIDENCE_ROOTS,
)
DEFAULT_COLLECTOR_REQUEST_ROOT = DEFAULT_EVIDENCE_ROOT / PROTOCOL_DIRECTORY
DEFAULT_LOCK = (
    ROOT
    / "artifacts/full_evidence_orchestration_20260525/shadow_autopilot_daemon_runtime/shadow_autopilot.lock"
)
def _token(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value or "").upper())


def _authoritative_meeting_aliases(race: Mapping[str, Any]) -> set[str] | None:
    """Keep same-code Murray Bridge meetings distinct by their TheDogs slug."""

    identity = canonical_thedogs_race_identity(race.get("url") or race.get("race_url"))
    if identity is None or identity["venue_slug"] not in {
        "murray-bridge",
        "murray-bridge-straight",
    }:
        return None
    return {
        value
        for value in (
            race.get("venue"),
            race.get("venue_name"),
            identity["venue_slug"],
        )
        if value
    }


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
    authoritative_aliases = _authoritative_meeting_aliases(race)
    aliases = authoritative_aliases or venue_exclusion_aliases(
        venue, source_url=race.get("url") or race.get("race_url")
    )
    if authoritative_aliases is None:
        values.update(stable_race_id_variants(race))
    for alias in aliases:
        values.update(
            {
                f"{alias} race {number}",
                f"race {number} {alias}",
                f"{alias} r{number}",
                f"Race {number} - {alias} - {race.get('date') or race.get('race_date')}",
            }
        )
    return {_token(value) for value in values if value}


def resolve_target_race(
    races: Sequence[Mapping[str, Any]],
    *,
    race_id: str | None,
    race_query: str | None,
    race_url: str | None = None,
) -> tuple[str, Mapping[str, Any] | None, list[str]]:
    """Resolve exactly one current master race identity without guessing."""

    if sum(bool(value) for value in (race_id, race_query, race_url)) != 1:
        raise ValueError("exactly_one_race_selector_required")
    if race_url:
        identity = canonical_thedogs_race_identity(race_url)
        if identity is None:
            return "BLOCKED_RACE_NOT_FOUND", None, []
        matches = [
            race
            for race in races
            if canonical_thedogs_race_identity(
                race.get("url") or race.get("race_url")
            )
            == identity
        ]
    else:
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


def _default_schedule(
    current_time: datetime,
    timeout_seconds: float,
    index_path: Path,
    evidence_root: Path,
    max_age_seconds: int,
) -> Sequence[Mapping[str, Any]]:
    from race_collection.synchronous_manual_capture import (
        bounded_current_race_index,
    )

    return bounded_current_race_index(
        current_time=current_time,
        timeout_seconds=timeout_seconds,
        index_path=index_path,
        evidence_root=evidence_root,
        max_age_seconds=max_age_seconds,
    )


def discover_capture_handoff(
    *,
    evidence_roots: Sequence[Path],
    db_path: Path,
    race_id: str,
    jump_datetime: datetime,
    capture_window_minutes: int,
    current_time: datetime,
) -> dict[str, Any] | None:
    """Adapt one master-verified sealed packet to the reuse-only receipt seam."""

    del db_path, capture_window_minutes
    match = re.fullmatch(
        r"Race\s+([0-9]{1,2})\s+-\s+(.+?)\s+-\s+[0-9]{4}-[0-9]{2}-[0-9]{2}",
        race_id,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise CaptureHandoffError("capture_packet_race_id_invalid")
    race_query = f"{match.group(2)} r{match.group(1)}"
    available_roots = [Path(root) for root in evidence_roots if Path(root).is_dir()]
    if not available_roots:
        return None
    try:
        packet = discover_race_artifacts(
            race_query=race_query,
            exact_race_id=race_id,
            evidence_roots=available_roots,
            score_timestamp=current_time,
        )
    except CaptureHandoffError as exc:
        if str(exc) in {
            "race_feature_packet_not_found",
            "race_capture_report_not_found",
        }:
            return None
        raise
    if str(packet.get("race_id")) != race_id:
        raise CaptureHandoffError("capture_packet_identity_mismatch")
    artifact_prediction = score_from_artifacts(
        race_id=race_id,
        form_csv_path=Path(packet["form_csv_path"]),
        sidecar_path=Path(packet["sidecar_path"]),
        feature_rows_path=Path(packet["feature_rows_path"]),
        feature_manifest_path=Path(packet["feature_manifest_path"]),
        implementation_manifest_path=Path(packet["implementation_manifest_path"]),
        capture_path=Path(packet["capture_path"]),
        model_path=ROOT / "artifacts/frozen_models/market_form_residual_v1/model.json",
        manifest_path=ROOT
        / "artifacts/frozen_models/market_form_residual_v1/manifest.json",
        score_timestamp=current_time,
    )
    try:
        packet_jump = datetime.fromisoformat(str(artifact_prediction["jump_timestamp"]))
        append_timestamp = str(artifact_prediction["odds_append_timestamp"])
        input_hashes = artifact_prediction["input_hashes"]
    except (KeyError, TypeError, ValueError) as exc:
        raise CaptureHandoffError("capture_packet_prediction_invalid") from exc
    if packet_jump != jump_datetime:
        raise CaptureHandoffError("capture_packet_jump_mismatch")

    raw_by_label: dict[str, bytes] = {}
    for label, path_key, hash_key in (
        ("report", "capture_path", "capture_artifact_sha256"),
        ("form", "form_csv_path", "form_csv_sha256"),
        ("sidecar", "sidecar_path", "sidecar_sha256"),
    ):
        path = Path(packet[path_key])
        if path.is_symlink() or not path.is_file():
            raise CaptureHandoffError(f"capture_packet_{label}_unsafe")
        raw = path.read_bytes()
        if sha256_bytes(raw) != input_hashes.get(hash_key):
            raise CaptureHandoffError(f"capture_packet_{label}_changed_after_score")
        raw_by_label[label] = raw

    form_name = Path(packet["form_csv_path"]).name
    return {
        "schema_version": "on_demand_verified_master_packet_v1",
        "race_id": race_id,
        "append_timestamp": append_timestamp,
        "source_report_sha256": sha256_bytes(raw_by_label["report"]),
        "source_form_sha256": sha256_bytes(raw_by_label["form"]),
        "source_sidecar_sha256": sha256_bytes(raw_by_label["sidecar"]),
        "packet_record_schema_version": artifact_prediction["record_schema_version"],
        "packet_record_checksum_sha256": artifact_prediction["record_checksum_sha256"],
        "packet_effective_state_schema_version": artifact_prediction[
            "effective_state_schema_version"
        ],
        "packet_effective_state_sha256": artifact_prediction["effective_state_sha256"],
        "_report_bytes": raw_by_label["report"],
        "_form_bytes": raw_by_label["form"],
        "_sidecar_bytes": raw_by_label["sidecar"],
        "_form_name": form_name,
    }


def seal_live_features(
    *,
    form_csv: Path,
    db_path: Path,
    output_dir: Path,
    current_time: datetime,
) -> dict[str, Path]:
    """Build a current-master feature packet inside the isolated run bundle."""

    from scripts.run_feature_recovery_execution_v1 import DEFAULT_SCHEMA, load_json
    from scripts.run_shadow_non_tgr_rf_evaluation import (
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
    _write_canonical(rows_path, rows)
    _write_canonical(
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
    _write_canonical(manifest_path, manifest)
    artifacts = {
        shadow_relpath(path): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in (rows_path, manifest_path, history_path)
    }
    implementation = {
        "schema_version": "shadow_implementation_file_manifest_v1",
        "output_dir": shadow_relpath(output_dir),
        "git_head": "on_demand_isolated_runtime",
        "git_branch": "on_demand_isolated_runtime",
        "implementation_files": list(FEATURE_GENERATOR_FILES),
        "implementation_file_hashes": {
            relative: sha256_file(ROOT / relative)
            for relative in FEATURE_GENERATOR_FILES
        },
        "artifact_files": artifacts,
    }
    _write_canonical(implementation_path, implementation)
    return {
        "feature_rows": rows_path,
        "feature_manifest": manifest_path,
        "implementation_manifest": implementation_path,
    }


def default_dependencies(args: argparse.Namespace) -> Dependencies:
    from race_collection.synchronous_manual_capture import invoke_capture_one
    from scripts.predict_market_form_residual import score_from_artifacts

    def capture_one(**values: Any) -> Mapping[str, Any]:
        command = [
            sys.executable,
            str(ROOT / "scripts/shadow_autopilot_daemon.py"),
            "capture-one",
            "--evidence-root",
            str(values["evidence_root"]),
            "--protocol-root",
            str(values["protocol_root"]),
            "--request-id",
            str(values["request_id"]),
            "--db",
            str(args.db),
            "--lock-path",
            str(args.lock_path),
            "--output-dir",
            str(values["output_dir"]),
            "--minimum-margin-seconds",
            str(values["minimum_margin_seconds"]),
            "--minimum-post-lock-margin-seconds",
            str(values["minimum_post_lock_margin_seconds"]),
            "--minimum-fetch-margin-seconds",
            str(values["minimum_fetch_margin_seconds"]),
            "--fetch-timeout-seconds",
            str(args.fetch_timeout_seconds),
        ]
        return invoke_capture_one(
            command=command,
            timeout_seconds=float(values["timeout_seconds"]),
        )

    return Dependencies(
        schedule=_default_schedule,
        seal_features=seal_live_features,
        score_residual=score_from_artifacts,
        now=lambda: datetime.now().astimezone(),
        capture_one=capture_one,
    )


def _public_model(model: Any) -> dict[str, Any]:
    return {
        "requested": model.requested,
        "resolved": model.resolved,
        "alias_resolved": model.alias,
        "model_sha256": model.model_sha256,
        "manifest_sha256": model.manifest_sha256,
        "schema_sha256": model.schema_sha256,
    }


def _sealed_model(model: Any) -> dict[str, Any]:
    available = model.model_sha256 is not None
    return {
        "requested": model.requested,
        "resolved": model.resolved,
        "alias_resolved": model.alias,
        "schema_sha256": model.schema_sha256,
        "artifact_identity": "AVAILABLE" if available else "UNAVAILABLE_NOT_APPLICABLE",
        "artifact_sha256": model.model_sha256,
        "artifact_manifest_identity": "AVAILABLE" if available else "UNAVAILABLE_NOT_APPLICABLE",
        "artifact_manifest_sha256": model.manifest_sha256,
    }


def _blocker_stage(code: str) -> str:
    try:
        return BLOCKER_STAGE_BY_CODE[code]
    except KeyError as exc:
        raise PredictionBlocked("UNSEALED_BLOCKER_CODE", blocker_code=code) from exc


def _sealed_result(
    *, state: Mapping[str, Any], generated_at: datetime, blocker: PredictionBlocked | None,
    prediction: Mapping[str, Any] | None,
) -> dict[str, Any]:
    model = state["model"]
    race_identity = state["race"]
    ready = blocker is None
    rows = None
    if ready:
        runners_by_box = {row["box_number"]: row for row in state["runners"]}
        rows = [
            {"rank": int(row["rank"]), "box_number": int(row["box_number"]), "dog_name": str(row["dog_name"]), "identity": runners_by_box[int(row["box_number"])]["identity"], "source_native_runner_id": runners_by_box[int(row["box_number"])]["source_native_runner_id"], "probability": float(row["probability"])}
            for row in (prediction or {}).get("predictions", [])
        ]
    return {
        "schema_version": "on_demand_race_prediction_v2",
        "prediction_id": state["prediction_id"],
        "job_id": state["job_id"],
        "generated_at": generated_at.isoformat(),
        "status": "PREDICTION_READY" if ready else "PREDICTION_BLOCKED",
        "blocker_stage": None if ready else _blocker_stage(blocker.code),
        "blocker": None if ready else {"code": blocker.code},
        "research_only": True,
        "production_persisted": False,
        "betting_output": False,
        "race": race_identity,
        "model": _sealed_model(model),
        "config": {"sha256": state["config_sha"]},
        "evidence": {
            "request": "request.json", "config": "config.json",
            "model_schema": "model/config.schema.json",
            "model_artifact": "model/model.json" if model.model_path else None,
            "model_manifest": "model/manifest.json" if model.manifest_path else None,
            "runner_set_sha256": state["runner_set_sha256"],
            "prediction_output_sha256": sha256_bytes(canonical_bytes(rows)) if ready else None,
            "protocol_chain": state.get("protocol_chain"),
            "authenticated_cutoff": state.get("authenticated_cutoff"),
        },
        "prediction": None if not ready else {"predictions": rows},
    }


def _selected_protocol_chain(protocol: ManualPredictionCollectorProtocol, handoff: Mapping[str, Any]) -> tuple[dict[str, str],dict[str,bytes]]:
    """Bind the already-validated exact handoff to its immutable protocol chain."""
    public={str(key):value for key,value in handoff.items() if not str(key).startswith("_")}
    try:
        return protocol.snapshot_authenticated_handoff(public)
    except ProtocolRejected as exc:
        raise PredictionBlocked("COLLECTOR_PROTOCOL_INVALID",reason=exc.code) from exc


def _seal_and_publish_v2(state: dict[str, Any], result: Mapping[str, Any]) -> None:
    bundle = Path(state["bundle"])
    _write_canonical(bundle / "result.json", result)
    manifest = build_prediction_bundle_manifest_v2(
        bundle, prediction_id=str(state["prediction_id"]), job_id=state["job_id"]
    )
    manifest_raw = canonical_bytes(manifest)
    _write_canonical(bundle / "bundle_manifest.json", manifest)
    state["terminal_sealed"] = True
    entry = prediction_bundle_index_entry(bundle=bundle, result=result, manifest_raw=manifest_raw)
    publish_prediction_bundle_index_entry(bundle.parent, entry)
    state["catalog_published"] = True


def _selected_variant(prediction: Mapping[str, Any], variant: str) -> dict[str, Any]:
    probability_key = {
        "full_strength": "full_probability",
        "half_strength": "half_probability",
    }[variant]
    rows = [
        {
            "box_number": int(row["box_number"] if "box_number" in row else row["box"]),
            "dog_name": str(row["dog_name"] if "dog_name" in row else row["dog"]),
            "probability": float(row[probability_key]),
            "market_probability": float(row["market_probability"]),
            "win_odds": float(row["win_odds"]),
        }
        for row in prediction.get("predictions") or []
    ]
    rows.sort(key=lambda row: (-row["probability"], row["box_number"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return {
        "adapter": "market_form_residual_v1",
        "variant": variant,
        "probability_sum": sum(row["probability"] for row in rows),
        "predictions": rows,
        "artifact_prediction": dict(prediction),
    }


def _bundle_file(bundle: Path, path: Path, *, label: str) -> tuple[Path, str]:
    """Return one regular, non-symlink bundle file and its stable relative path."""

    bundle_root = bundle.resolve()
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise PredictionBlocked(
            "BUNDLE_SOURCE_UNSAFE", source=label, path=str(candidate)
        )
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(bundle_root)
    except (OSError, ValueError) as exc:
        raise PredictionBlocked(
            "BUNDLE_SOURCE_UNSAFE", source=label, path=str(candidate)
        ) from exc
    return resolved, relative.as_posix()


def _validated_refreshed_sources(
    bundle: Path, form_csv: Path, sidecar: Path
) -> tuple[Path, Path]:
    form, _ = _bundle_file(bundle, form_csv, label="refreshed_form_csv")
    metadata, _ = _bundle_file(bundle, sidecar, label="refreshed_sidecar")
    if metadata != form.with_name(form.name + ".metadata.json"):
        raise PredictionBlocked(
            "BUNDLE_SOURCE_UNSAFE",
            source="refreshed_sidecar",
            reason="sidecar_not_exactly_adjacent",
        )
    return form, metadata


def _persist_blocked_bundle(
    state: Mapping[str, Any], exc: PredictionBlocked, generated_at: datetime
) -> None:
    """Seal the smallest post-bundle blocker as a research-only result."""

    bundle = Path(state["bundle"])
    result = _sealed_result(
        state=state, generated_at=generated_at, blocker=exc, prediction=None
    )
    _seal_and_publish_v2(state, result)
    exc.details["prediction_id"] = state["prediction_id"]


def _request_race(
    target: Mapping[str, Any],
    *,
    race_id: str,
    jump: datetime,
) -> dict[str, Any]:
    try:
        raw_number = target.get("race_number")
        if isinstance(raw_number, bool) or not isinstance(raw_number, int) or raw_number <= 0:
            raise ValueError(raw_number)
        race_number = raw_number
    except (TypeError, ValueError) as exc:
        raise PredictionBlocked("EXACT_RACE_IDENTITY_UNAVAILABLE") from exc
    race_date = str(target.get("date") or target.get("race_date") or "").strip()
    venue = str(target.get("venue") or target.get("venue_name") or "").strip().upper()
    url = str(target.get("url") or target.get("race_url") or "").strip()
    identity = canonical_thedogs_race_identity(url)
    url_venue = (
        canonical_thedogs_venue_identity(identity["venue_slug"])
        if identity is not None
        else None
    )
    projection = {"race_number": race_number, "venue": venue, "race_date": race_date[:10], "url": url}
    if (
        not race_date or not venue or identity is None
        or identity["canonical_url"] != url
        or identity["race_date"] != race_date[:10]
        or identity["race_number"] != race_number
        or url_venue is None
        or canonical_thedogs_venue_identity(venue) != url_venue
        or venue != url_venue
        or stable_race_id(projection) != race_id
        or race_id not in stable_race_id_variants(projection)
    ):
        raise PredictionBlocked("EXACT_RACE_IDENTITY_UNAVAILABLE")
    return {
        "race_id": race_id,
        "url": url,
        "venue": venue,
        "venue_slug": identity["venue_slug"],
        "race_number": race_number,
        "race_date": race_date[:10],
        "jump_timestamp": jump.isoformat(),
    }


def _request_expected_runners(
    target: Mapping[str, Any],
) -> list[dict[str, Any]]:
    values = target.get("participants") or target.get("runners") or []
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    rows: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, Mapping):
            return []
        try:
            box = int(value.get("box_number") or value.get("box"))
        except (TypeError, ValueError):
            return []
        dog_name = str(value.get("display_name") or value.get("dog_name") or value.get("name") or "").strip()
        identity = str(value.get("identity") or "").strip()
        if not dog_name or not identity:
            return []
        native_id = value.get("source_native_runner_id")
        if native_id is not None and (not isinstance(native_id, str) or not native_id or native_id != native_id.strip()):
            return []
        rows.append(
            {
                "box_number": box,
                "display_name": dog_name,
                "identity": identity,
                "source_native_runner_id": native_id,
            }
        )
    if rows != sorted(rows, key=lambda row: (row["box_number"], row["identity"])) or len(rows) < 2:
        return []
    return rows


def _terminalize_incomplete_capture_request(
    protocol: ManualPredictionCollectorProtocol,
    *,
    request_id: str,
    now: datetime,
    reason: str,
) -> None:
    """Close a published request after its collector child has been reaped."""

    if protocol.read_response(request_id) is not None:
        return
    try:
        context = protocol.claimed_request(request_id)
    except ProtocolRejected as exc:
        if exc.code != "CLAIM_NOT_FOUND":
            raise
        context = protocol.claim_request(
            request_id,
            now=now,
            collector_run_id="manual_capture_one_launch_cleanup",
        )
    protocol.publish_terminal(
        context,
        status="CAPTURE_FAILED",
        now=now,
        reason=reason,
    )


def _acquire_or_reuse(
    dependencies: Dependencies,
    *,
    protocol: ManualPredictionCollectorProtocol,
    target: Mapping[str, Any],
    odds_source: str,
    evidence_roots: Sequence[Path],
    db_path: Path,
    race_id: str,
    jump: datetime,
    current_time: datetime,
    latency_budget: Any,
    receipt_max_age_seconds: int,
    fetch_timeout_seconds: float,
) -> tuple[
    Mapping[str, Any] | None,
    list[dict[str, Any]],
    datetime,
]:
    del db_path
    rejected_receipts: list[dict[str, Any]] = []
    if odds_source not in {"auto", "capture", "receipt"}:
        raise PredictionBlocked("ODDS_SOURCE_UNSUPPORTED", odds_source=odds_source)
    try:
        handoff = protocol.discover_exact_handoff(
            race_id=race_id,
            current_time=current_time,
            max_age_seconds=receipt_max_age_seconds,
        )
        if handoff is not None:
            if (jump - current_time).total_seconds() <= latency_budget.reuse_margin_seconds:
                raise PredictionBlocked(
                    "INSUFFICIENT_PREJUMP_MARGIN",
                    phase="reuse_validation_and_scoring",
                    remaining_seconds=(jump - current_time).total_seconds(),
                    required_seconds=latency_budget.reuse_margin_seconds,
                )
            return handoff, rejected_receipts, current_time
        if odds_source == "receipt":
            raise PredictionBlocked("RECEIPT_UNAVAILABLE")
        remaining = (jump - current_time).total_seconds()
        if remaining <= latency_budget.capture_margin_seconds:
            raise PredictionBlocked(
                "INSUFFICIENT_PREJUMP_MARGIN",
                phase="capture_validation_and_scoring",
                remaining_seconds=remaining,
                required_seconds=latency_budget.capture_margin_seconds,
            )
        protocol_race = _request_race(target, race_id=race_id, jump=jump)
        protocol_race.pop("venue_slug")
        published = protocol.publish_request(
            race=protocol_race,
            expected_runners=[
                {
                    "box_number": row["box_number"],
                    "dog_name": row["display_name"],
                    "identity": row["identity"],
                }
                for row in _request_expected_runners(target)
            ],
            created_at=current_time,
            expires_at=jump,
        )
        if dependencies.capture_one is None:
            raise PredictionBlocked("COLLECTOR_CAPTURE_ONE_UNAVAILABLE")
        evidence_root = Path(evidence_roots[0]).resolve()
        request_id = str(published["request_id"])
        try:
            capture_result = dependencies.capture_one(
                protocol_root=protocol.root,
                evidence_root=evidence_root,
                request_id=request_id,
                output_dir=(
                    evidence_root / f"synchronous_manual_capture_{request_id}"
                ),
                minimum_margin_seconds=latency_budget.capture_margin_seconds,
                minimum_post_lock_margin_seconds=(
                    latency_budget.post_lock_margin_seconds
                ),
                minimum_fetch_margin_seconds=(
                    latency_budget.pre_fetch_margin_seconds(
                        fetch_timeout_seconds
                    )
                ),
                timeout_seconds=(
                    latency_budget.lock_seconds
                    + latency_budget.capture_seconds
                    + latency_budget.validation_seconds
                    + latency_budget.safety_seconds
                ),
            )
        except BaseException:
            _terminalize_incomplete_capture_request(
                protocol,
                request_id=request_id,
                now=max(current_time, dependencies.now()),
                reason="collector_capture_one_launch_or_cancellation_failed",
            )
            raise
        response = protocol.read_response(request_id)
        if response is None:
            _terminalize_incomplete_capture_request(
                protocol,
                request_id=request_id,
                now=max(current_time, dependencies.now()),
                reason="collector_capture_one_returned_without_response",
            )
            raise PredictionBlocked(
                "COLLECTOR_PROTOCOL_INVALID",
                request_id=request_id,
                reason="synchronous_capture_returned_without_terminal_response",
            )
        response_observed_time = max(
            dependencies.now(),
            datetime.fromisoformat(str(response["responded_at"])),
        )
        consumed = protocol.consume_response(
            request_id,
            now=response_observed_time,
        )
        if response["status"] != RECEIPT_READY:
            terminal_status = str(
                capture_result.get("status") or response["status"]
            )
            raise PredictionBlocked(
                terminal_status,
                request_id=request_id,
                reason=response.get("reason"),
                **(
                    {"busy": capture_result.get("busy")}
                    if capture_result.get("busy") is not None
                    else {}
                ),
            )
        handoff = protocol.discover_exact_handoff(
            race_id=race_id,
            current_time=response_observed_time,
            max_age_seconds=receipt_max_age_seconds,
        )
        if handoff is None:
            raise PredictionBlocked(
                "RECEIPT_INVALID",
                reason="sealed_response_receipt_unavailable",
            )
        normalized, _, _, _ = receipt_from_handoff(
            handoff,
            current_time=response_observed_time,
            max_age_seconds=receipt_max_age_seconds,
        )
        protocol.verify_ready_handoff(
            consumed["receipt"],
            handoff=handoff,
            normalized_receipt=normalized,
        )
        remaining = (jump - response_observed_time).total_seconds()
        if remaining <= latency_budget.reuse_margin_seconds:
            raise PredictionBlocked(
                "INSUFFICIENT_PREJUMP_MARGIN",
                phase="post_capture_validation_and_scoring",
                remaining_seconds=remaining,
                required_seconds=latency_budget.reuse_margin_seconds,
            )
    except PredictionBlocked:
        raise
    except ProtocolRejected as exc:
        raise PredictionBlocked(
            "COLLECTOR_PROTOCOL_INVALID",
            reason=exc.code,
            **exc.details,
        ) from exc
    return handoff, rejected_receipts, response_observed_time


def _run_prediction(
    args: argparse.Namespace, dependencies: Dependencies, state: dict[str, Any]
) -> dict[str, Any]:
    current_time = (
        parse_current_time(args.current_time)
        if args.current_time
        else dependencies.now()
    )
    if current_time.tzinfo is None or current_time.utcoffset() is None:
        raise PredictionBlocked("CURRENT_TIME_TIMEZONE_MISSING")
    model = resolve_model(args.model)
    config, config_sha, config_raw = load_config(Path(args.config), model)
    from race_collection.synchronous_manual_capture import (
        CaptureOneRejected,
        LatencyBudget,
    )

    latency_budget = LatencyBudget.from_config(config["bundle"]["latency_budget"])
    evidence_roots = tuple(
        Path(path)
        for path in (
            args.capture_evidence_root or DEFAULT_CAPTURE_EVIDENCE_ROOTS
        )
    )
    primary_evidence_root = evidence_roots[0]
    current_index_path = (
        DEFAULT_CAPTURE_EVIDENCE_ROOTS[0]
        / "shadow_autopilot_daemon_runtime"
        / "manual_prediction_current_race_index.json"
    )
    discovery_started = dependencies.monotonic()
    try:
        races = dependencies.schedule(
            current_time,
            latency_budget.discovery_seconds,
            Path(current_index_path),
            primary_evidence_root,
            int(config["bundle"]["current_index_max_age_seconds"]),
        )
    except CaptureOneRejected as exc:
        raise PredictionBlocked(exc.code, **exc.details) from exc
    discovery_elapsed = max(
        0.0, dependencies.monotonic() - discovery_started
    )
    if discovery_elapsed > latency_budget.discovery_seconds:
        raise PredictionBlocked(
            "DISCOVERY_TIMEOUT",
            elapsed_seconds=discovery_elapsed,
            budget_seconds=latency_budget.discovery_seconds,
        )
    readiness_time = max(
        dependencies.now(),
        current_time + timedelta(seconds=discovery_elapsed),
    )
    resolved, target, matches = resolve_target_race(
        races,
        race_id=getattr(args, "race_id", None),
        race_query=getattr(args, "race", None),
        race_url=getattr(args, "race_url", None),
    )
    if resolved != "RESOLVED" or target is None:
        raise PredictionBlocked(resolved, matching_race_ids=matches)
    race_id = str(stable_race_id(target) or "")
    jump = _parse_race_jump_datetime(target, now=current_time)
    if not race_id or jump is None:
        raise PredictionBlocked("EXACT_RACE_IDENTITY_UNAVAILABLE")
    if jump <= readiness_time:
        raise PredictionBlocked(
            "POST_JUMP", race_id=race_id, jump_timestamp=jump.isoformat()
        )

    race = _request_race(target, race_id=race_id, jump=jump)
    runners = _request_expected_runners(target)
    if not runners:
        raise PredictionBlocked("RUNNER_SET_AMBIGUOUS")
    runner_hash = sealed_runner_set_sha256(race, runners)

    # A v2 bundle is a promise that exact race and runner identity exists.
    # Do not expose bundle state until every schema-required identity is known.
    bundle = create_bundle(Path(args.output_root), current_time)
    state.update(
        bundle=bundle,
        prediction_id=str(uuid.uuid4()),
        job_id=_job_id(getattr(args, "job_id", None)),
        model=model,
        config_sha=config_sha,
        race=race,
        runners=runners,
        runner_set_sha256=runner_hash,
    )
    race_selector = (
        getattr(args, "race", None)
        or getattr(args, "race_id", None)
        or getattr(args, "race_url", None)
    )
    request = {
        "schema_version": "on_demand_prediction_request_v1",
        "prediction_id": state["prediction_id"],
        "job_id": state["job_id"],
        "race_query": race_selector,
        "race_id": race_id,
        "jump_timestamp": jump.isoformat(),
        "request_timestamp": current_time.isoformat(),
        "odds_source": args.odds_source,
        "model": _public_model(model),
        "config_sha256": config_sha,
        "research_only": True,
        "runners": state["runners"],
        "runner_set_sha256": state["runner_set_sha256"],
    }
    _write_canonical(bundle / "request.json", request)
    write_exact_bytes(bundle / "config.json", config_raw)
    _copy_exact(model.schema_path, bundle / "model" / "config.schema.json")
    if model.model_path and model.manifest_path:
        _copy_exact(model.model_path, bundle / "model" / "model.json")
        _copy_exact(model.manifest_path, bundle / "model" / "manifest.json")

    protocol=ManualPredictionCollectorProtocol(
            Path(getattr(args,"collector_request_root",DEFAULT_COLLECTOR_REQUEST_ROOT))
        )
    (
        handoff,
        rejected_receipts,
        receipt_validation_time,
    ) = _acquire_or_reuse(
        dependencies,
        protocol=protocol,
        target=target,
        odds_source=args.odds_source,
        evidence_roots=evidence_roots,
        db_path=Path(args.db),
        race_id=race_id,
        jump=jump,
        current_time=readiness_time,
        latency_budget=latency_budget,
            receipt_max_age_seconds=int(
                config["bundle"]["receipt_max_age_seconds"]
            ),
            fetch_timeout_seconds=float(args.fetch_timeout_seconds),
        )
    if handoff is not None:
        state["protocol_chain"],protocol_members=_selected_protocol_chain(protocol,handoff)
        for member,raw in protocol_members.items():write_exact_bytes(bundle/"protocol"/(member+".json"),raw)
        receipt, capture_raw, form_raw, sidecar_raw = receipt_from_handoff(
            handoff,
            current_time=receipt_validation_time,
            max_age_seconds=int(config["bundle"]["receipt_max_age_seconds"]),
        )
        form_name = str(handoff.get("_form_name") or "form.csv")
        if Path(form_name).name != form_name:
            raise PredictionBlocked("RECEIPT_INVALID")
        form_csv = bundle / "source" / form_name
        sidecar = form_csv.with_name(form_csv.name + ".metadata.json")
        capture_path = bundle / "source" / "capture.json"
        write_exact_bytes(form_csv, form_raw)
        write_exact_bytes(sidecar, sidecar_raw)
        write_exact_bytes(capture_path, capture_raw)
    else:
        raise PredictionBlocked("RECEIPT_UNAVAILABLE")

    try:
        odds_captured_at = datetime.fromisoformat(str(receipt["captured_at"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise PredictionBlocked("ODDS_TIMESTAMP_AMBIGUOUS") from exc
    if odds_captured_at.tzinfo is None or odds_captured_at.utcoffset() is None:
        raise PredictionBlocked("ODDS_TIMESTAMP_AMBIGUOUS")
    if odds_captured_at >= jump:
        raise PredictionBlocked(
            "POST_JUMP", race_id=race_id, odds_captured_at=odds_captured_at.isoformat()
        )

    _write_canonical(bundle / "odds_receipt.json", receipt)
    runner_names = [str(row["dog_name"]) for row in receipt["markets"]["win"]]
    sealed_db = bundle / "features" / "sealed_history.db"
    history_path = bundle / "features" / "history_seal.json"
    if not sealed_db.exists():
        history = seal_history_database(
            source=Path(args.db),
            target=sealed_db,
            target_race_id=race_id,
            cutoff=jump,
            runner_names=runner_names,
        )
        _write_canonical(history_path, history)
    else:
        history = json.loads(history_path.read_bytes())
    if (
        history.get("target_rows_materialized") != 0
        or history.get("at_or_after_cutoff_rows_materialized") != 0
    ):
        raise PredictionBlocked("TARGET_EXCLUSION_WEAK")
    state["authenticated_cutoff"]={"history_seal_sha256":sha256_file(history_path),"cutoff_timestamp":str(history["cutoff_timestamp"]),"source_sha256":str(history["source_sha256"]),"sealed_sha256":str(history["sealed_sha256"])}

    score_time = dependencies.now()
    if score_time >= jump:
        raise PredictionBlocked("POST_JUMP", race_id=race_id)
    if model.resolved == "market_only_v1":
        prediction = market_only_prediction(receipt)
        feature_identity: dict[str, Any] = {"required": False}
    else:
        feature_dir = bundle / "features" / "sealed"
        try:
            sealed = dependencies.seal_features(
                form_csv=form_csv,
                db_path=sealed_db,
                output_dir=feature_dir,
                current_time=score_time,
            )
        except PredictionBlocked:
            raise
        except Exception as exc:
            raise PredictionBlocked(
                "FEATURE_SEAL_FAILED", error=type(exc).__name__
            ) from exc
        feature_rows = json.loads(Path(sealed["feature_rows"]).read_bytes())
        unsafe_rows = [
            row
            for row in feature_rows
            if row.get("same_distance_same_grade_target_race_rows_used")
            not in (0, None)
            or row.get("same_distance_same_grade_post_outcome_rows_used")
            not in (0, None)
        ]
        if unsafe_rows:
            raise PredictionBlocked("TARGET_EXCLUSION_WEAK")
        try:
            artifact_prediction = dependencies.score_residual(
                race_id=race_id,
                form_csv_path=form_csv,
                sidecar_path=sidecar,
                feature_rows_path=Path(sealed["feature_rows"]),
                feature_manifest_path=Path(sealed["feature_manifest"]),
                implementation_manifest_path=Path(sealed["implementation_manifest"]),
                capture_path=capture_path,
                model_path=bundle / "model" / "model.json",
                manifest_path=bundle / "model" / "manifest.json",
                score_timestamp=score_time,
            )
        except PredictionBlocked:
            raise
        except Exception as exc:
            raise PredictionBlocked(
                "RESIDUAL_SCORER_FAILED", error=type(exc).__name__
            ) from exc
        if (
            artifact_prediction.get("model_sha256") != model.model_sha256
            or artifact_prediction.get("manifest_sha256") != model.manifest_sha256
        ):
            raise PredictionBlocked("FROZEN_MODEL_DRIFT")
        prediction = _selected_variant(artifact_prediction, str(config["variant"]))
        replay_paths = {
            "form_csv": _bundle_file(bundle, form_csv, label="form_csv")[1],
            "sidecar": _bundle_file(bundle, sidecar, label="sidecar")[1],
            "feature_rows": _bundle_file(
                bundle, Path(sealed["feature_rows"]), label="feature_rows"
            )[1],
            "feature_manifest": _bundle_file(
                bundle, Path(sealed["feature_manifest"]), label="feature_manifest"
            )[1],
            "implementation_manifest": _bundle_file(
                bundle,
                Path(sealed["implementation_manifest"]),
                label="implementation_manifest",
            )[1],
            "capture": _bundle_file(bundle, capture_path, label="capture")[1],
            "model": _bundle_file(
                bundle, bundle / "model" / "model.json", label="model"
            )[1],
            "manifest": _bundle_file(
                bundle, bundle / "model" / "manifest.json", label="manifest"
            )[1],
        }
        feature_identity = {
            "required": True,
            "feature_rows_sha256": sha256_file(Path(sealed["feature_rows"])),
            "feature_manifest_sha256": sha256_file(Path(sealed["feature_manifest"])),
            "implementation_manifest_sha256": sha256_file(
                Path(sealed["implementation_manifest"])
            ),
            "replay_paths": replay_paths,
        }

    completed_time = dependencies.now()
    if completed_time.tzinfo is None or completed_time.utcoffset() is None:
        raise PredictionBlocked("CURRENT_TIME_TIMEZONE_MISSING")
    if completed_time >= jump:
        raise PredictionBlocked(
            "POST_JUMP", race_id=race_id, completed_at=completed_time.isoformat()
        )

    legacy_result = {
        "schema_version": "on_demand_race_prediction_v1",
        "status": "PREDICTION_READY",
        "research_only": True,
        "production_persisted": False,
        "betting_output": False,
        "race": {
            "query": race_selector,
            "race_id": race_id,
            "jump_timestamp": jump.isoformat(),
        },
        "score_timestamp": score_time.isoformat(),
        "odds_source": receipt["source_kind"],
        "rejected_receipts": rejected_receipts,
        "runner_set_sha256": receipt["runner_set_sha256"],
        "model": _public_model(model),
        "config": {"sha256": config_sha, "value": config},
        "history_seal": history,
        "source_identity": {
            "form_csv_sha256": sha256_file(form_csv),
            "sidecar_sha256": sha256_file(sidecar),
            "capture_sha256": sha256_file(capture_path),
            "sealed_history_sha256": sha256_file(sealed_db),
            "odds_captured_at": receipt["captured_at"],
        },
        "feature_identity": feature_identity,
        "prediction": prediction,
        "blockers": [],
        "bundle": str(bundle.resolve()),
    }
    result = _sealed_result(
        state=state,
        generated_at=completed_time,
        blocker=None,
        prediction=legacy_result["prediction"],
    )
    _seal_and_publish_v2(state, result)
    return result


def run_prediction(
    args: argparse.Namespace, dependencies: Dependencies
) -> dict[str, Any]:
    state: dict[str, Any] = {}
    try:
        return _run_prediction(args, dependencies, state)
    except PredictionBlocked as exc:
        bundle = state.get("bundle")
        if bundle is not None and not state.get("terminal_sealed"):
            try:
                _persist_blocked_bundle(state, exc, dependencies.now())
            except (OSError, TypeError, ValueError, PredictionBlocked) as persist_exc:
                # Never replace the smallest operational blocker with a reporting failure.
                exc.details["bundle"] = str(bundle.resolve())
                exc.details["bundle_persistence_error"] = type(persist_exc).__name__
        raise
    except Exception as exc:
        blocked = PredictionBlocked(
            "PREDICTION_INTERNAL_ERROR", error=type(exc).__name__
        )
        bundle = state.get("bundle")
        if bundle is not None and not state.get("terminal_sealed"):
            try:
                _persist_blocked_bundle(state, blocked, dependencies.now())
            except (OSError, TypeError, ValueError, PredictionBlocked) as persist_exc:
                blocked.details["bundle"] = str(bundle.resolve())
                blocked.details["bundle_persistence_error"] = type(persist_exc).__name__
        raise blocked from exc


def replay_bundle(
    bundle: Path,
    score_residual: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    verify_bundle(bundle)
    try:
        result = json.loads((bundle / "result.json").read_bytes())
        receipt = json.loads((bundle / "odds_receipt.json").read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise PredictionBlocked("REPLAY_INPUT_INVALID") from exc
    adapter = result.get("prediction", {}).get("adapter")
    if adapter == "market_only_v1":
        replayed = market_only_prediction(receipt)
        if canonical_bytes(replayed) != canonical_bytes(result["prediction"]):
            raise PredictionBlocked("REPLAY_NONDETERMINISTIC")
    else:
        feature_identity = result.get("feature_identity") or {}
        replay_paths = feature_identity.get("replay_paths")
        if not isinstance(replay_paths, Mapping):
            raise PredictionBlocked(
                "REPLAY_INPUT_INVALID", reason="replay_paths_missing"
            )

        def replay_path(label: str) -> Path:
            raw = replay_paths.get(label)
            if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
                raise PredictionBlocked("REPLAY_INPUT_INVALID", source=label)
            path, relative = _bundle_file(bundle, bundle / raw, label=label)
            if relative != raw:
                raise PredictionBlocked("REPLAY_INPUT_INVALID", source=label)
            return path

        inputs = {label: replay_path(label) for label in replay_paths}
        required = {
            "form_csv",
            "sidecar",
            "feature_rows",
            "feature_manifest",
            "implementation_manifest",
            "capture",
            "model",
            "manifest",
        }
        if set(inputs) != required:
            raise PredictionBlocked("REPLAY_INPUT_INVALID", reason="replay_path_set")
        expected = {
            "feature_rows_sha256": sha256_file(inputs["feature_rows"]),
            "feature_manifest_sha256": sha256_file(inputs["feature_manifest"]),
            "implementation_manifest_sha256": sha256_file(
                inputs["implementation_manifest"]
            ),
        }
        if any(feature_identity.get(key) != value for key, value in expected.items()):
            raise PredictionBlocked("REPLAY_TAMPERED")
        try:
            score_time = datetime.fromisoformat(str(result["score_timestamp"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise PredictionBlocked(
                "REPLAY_INPUT_INVALID", reason="score_timestamp"
            ) from exc
        if score_time.tzinfo is None or score_time.utcoffset() is None:
            raise PredictionBlocked(
                "REPLAY_INPUT_INVALID", reason="score_timestamp_timezone"
            )
        if score_residual is None:
            from scripts.predict_market_form_residual import score_from_artifacts

            score_residual = score_from_artifacts
        try:
            artifact_prediction = score_residual(
                race_id=str(result["race"]["race_id"]),
                form_csv_path=inputs["form_csv"],
                sidecar_path=inputs["sidecar"],
                feature_rows_path=inputs["feature_rows"],
                feature_manifest_path=inputs["feature_manifest"],
                implementation_manifest_path=inputs["implementation_manifest"],
                capture_path=inputs["capture"],
                model_path=inputs["model"],
                manifest_path=inputs["manifest"],
                score_timestamp=score_time,
            )
            replayed = _selected_variant(
                artifact_prediction, str(result["prediction"]["variant"])
            )
        except PredictionBlocked:
            raise
        except Exception as exc:
            raise PredictionBlocked(
                "REPLAY_SCORER_FAILED", error=type(exc).__name__
            ) from exc
        if canonical_bytes(replayed) != canonical_bytes(result["prediction"]):
            raise PredictionBlocked("REPLAY_NONDETERMINISTIC")
    return result


def list_configs() -> dict[str, Any]:
    """Return the finite checked-in config catalog with resolved immutable IDs."""

    catalog = []
    for name, selector, relative in (
        (
            "market-form-residual-v1",
            "latest-research",
            Path("configs/prediction/manual-default.json"),
        ),
        (
            "market-only",
            "market-only",
            Path("configs/prediction/market-only.json"),
        ),
    ):
        model = resolve_model(selector)
        config, config_sha, _ = load_config(ROOT / relative, model)
        catalog.append(
            {
                "name": name,
                "selector": selector,
                "config": relative.as_posix(),
                "config_sha256": config_sha,
                "resolved_config": config,
                "model": _public_model(model),
            }
        )
    return {
        "schema_version": "on_demand_prediction_config_catalog_v1",
        "status": "CONFIGS_AVAILABLE",
        "configs": catalog,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--race", help="Exact named upcoming race, e.g. 'gunnedah r5'")
    target.add_argument("--race-id", help="Exact canonical race ID")
    target.add_argument("--race-url", help="Exact canonical TheDogs race URL")
    target.add_argument("--replay-bundle", type=Path)
    target.add_argument("--list-configs", action="store_true")
    parser.add_argument("--model", default="latest-research")
    parser.add_argument("--job-id")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/prediction/manual-default.json"
    )
    parser.add_argument(
        "--odds-source", choices=("auto", "receipt", "capture"), default="auto"
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--current-time")
    parser.add_argument("--fetch-timeout-seconds", type=float, default=45.0)
    parser.add_argument(
        "--capture-evidence-root",
        action="append",
        type=Path,
        default=None,
    )
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK)
    parser.add_argument(
        "--collector-request-root",
        type=Path,
        default=DEFAULT_COLLECTOR_REQUEST_ROOT,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.list_configs:
            output = list_configs()
        elif args.replay_bundle:
            output = replay_bundle(args.replay_bundle.resolve())
        else:
            output = run_prediction(args, default_dependencies(args))
    except PredictionBlocked as exc:
        output = {
            "schema_version": "on_demand_race_prediction_v1",
            "status": exc.code,
            "research_only": True,
            "production_persisted": False,
            "blockers": [{"code": exc.code, **exc.details}],
        }
        if "bundle" in exc.details:
            output["bundle"] = exc.details["bundle"]
    print(canonical_bytes(output).decode(), end="")
    return 0 if output.get("status") in {"PREDICTION_READY", "CONFIGS_AVAILABLE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
