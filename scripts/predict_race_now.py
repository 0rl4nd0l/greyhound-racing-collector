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
import contextlib
import json
import os
import socket
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.refresh_prejump_upcoming import (  # noqa: E402
    _parse_race_jump_datetime,
    parse_current_time,
    stable_race_id,
)
from scripts.run_priority_race_prediction import (  # noqa: E402
    CaptureHandoffError,
    DEFAULT_CAPTURE_EVIDENCE_ROOTS,
    discover_capture_handoff,
    resolve_target_race,
    seal_live_features,
)
from src.predictor.on_demand import (  # noqa: E402
    Dependencies,
    PredictionBlocked,
    _copy_exact,
    _write_canonical,
    bundle_manifest,
    canonical_bytes,
    create_bundle,
    load_config,
    market_only_prediction,
    normalize_validation_receipt,
    receipt_from_handoff,
    resolve_model,
    seal_history_database,
    sha256_file,
    verify_bundle,
    write_exact_bytes,
)


DEFAULT_DB = ROOT / "greyhound_racing_data.db"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/on_demand_prediction_runs"
DEFAULT_LOCK = (
    ROOT
    / "artifacts/full_evidence_orchestration_20260525/shadow_autopilot_daemon_runtime/shadow_autopilot.lock"
)


class CollectorLockBusy(RuntimeError):
    """An existing collector lock is always busy; it is never reclaimed here."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        super().__init__("collector_lock_busy_no_steal")
        self.payload = dict(payload)


@dataclass(frozen=True)
class OwnedCollectorLock:
    path: Path
    run_id: str
    device: int
    inode: int


def _read_lock_for_diagnostics(lock_path: Path) -> Mapping[str, Any] | None:
    if lock_path.is_symlink():
        return None
    try:
        value = json.loads(lock_path.read_bytes())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _unlink_owned_lock_inode(lock: OwnedCollectorLock) -> bool:
    try:
        current = lock.path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return False
    if lock.path.is_symlink() or (current.st_dev, current.st_ino) != (
        lock.device,
        lock.inode,
    ):
        return False
    lock.path.unlink()
    return True


def _acquire_collector_lock_no_steal(
    lock_path: Path, *, run_id: str, output_dir: Path
) -> OwnedCollectorLock:
    if lock_path.is_symlink() or not lock_path.parent.is_dir():
        raise PredictionBlocked("LOCK_PATH_UNSAFE", path=str(lock_path))
    payload = {
        "schema_version": "shadow_autopilot_daemon_lock_v1",
        "run_id": run_id,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_at": datetime.now().astimezone().isoformat(),
        "output_dir": str(output_dir.resolve()),
        "acquisition_policy": "on_demand_no_steal_v1",
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError as exc:
        raise CollectorLockBusy(
            {
                "lock_path": str(lock_path),
                "existing_lock": _read_lock_for_diagnostics(lock_path),
                "reason": "existing_lock_present_no_steal",
            }
        ) from exc
    except OSError as exc:
        raise PredictionBlocked(
            "LOCK_ACQUIRE_FAILED", error=type(exc).__name__
        ) from exc
    try:
        opened = os.fstat(descriptor)
    except OSError as exc:
        try:
            opened = os.stat(descriptor)
        except OSError:
            os.close(descriptor)
            raise PredictionBlocked(
                "LOCK_ACQUIRE_FAILED",
                reason="descriptor_identity_unavailable",
                cleanup="not_safe",
            ) from exc
        os.close(descriptor)
        failed_lock = OwnedCollectorLock(
            path=lock_path,
            run_id=run_id,
            device=opened.st_dev,
            inode=opened.st_ino,
        )
        _unlink_owned_lock_inode(failed_lock)
        raise PredictionBlocked(
            "LOCK_ACQUIRE_FAILED", reason="descriptor_stat_failed"
        ) from exc
    lock = OwnedCollectorLock(
        path=lock_path,
        run_id=run_id,
        device=opened.st_dev,
        inode=opened.st_ino,
    )
    try:
        try:
            handle = os.fdopen(descriptor, "wb")
        except Exception:
            os.close(descriptor)
            raise
        with handle:
            handle.write(canonical_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
    except Exception as exc:
        _unlink_owned_lock_inode(lock)
        raise PredictionBlocked(
            "LOCK_ACQUIRE_FAILED", error=type(exc).__name__
        ) from exc
    return lock


def _release_owned_collector_lock(lock: OwnedCollectorLock) -> None:
    payload = _read_lock_for_diagnostics(lock.path)
    if payload is None or payload.get("run_id") != lock.run_id:
        raise PredictionBlocked("LOCK_RELEASE_FAILED", reason="ownership_unverified")
    if not _unlink_owned_lock_inode(lock):
        raise PredictionBlocked("LOCK_RELEASE_FAILED", reason="inode_changed")


def _default_schedule(days_ahead: int) -> Sequence[Mapping[str, Any]]:
    from upcoming_race_browser import UpcomingRaceBrowser

    with contextlib.redirect_stdout(sys.stderr):
        return UpcomingRaceBrowser(create_upcoming_dir=False).get_upcoming_races(
            days_ahead=days_ahead
        )


def _default_refresh(
    target: Mapping[str, Any], bundle: Path, current_time: datetime, days_ahead: int
) -> tuple[Path, Path]:
    from scripts.refresh_prejump_upcoming import refresh_prejump_upcoming

    upcoming_dir = bundle / "source" / "upcoming"
    args = argparse.Namespace(
        upcoming_dir=upcoming_dir,
        days_ahead=days_ahead,
        min_minutes=0.0,
        max_minutes=max(1.0, float(days_ahead + 1) * 1440.0),
        limit=1,
        exclude_race_id=[],
        exclude_race_ids_file=None,
        include_race_id=[stable_race_id(target)],
        dry_run=False,
        current_time=current_time.isoformat(),
        require_safe_metadata=True,
    )
    with contextlib.redirect_stdout(sys.stderr):
        report = refresh_prejump_upcoming(args)
    coverage = report.get("sidecar_metadata_coverage") or {}
    rows = coverage.get("races") if isinstance(coverage, Mapping) else None
    if (
        report.get("status") != "SUCCESS"
        or report.get("selected_count") != 1
        or coverage.get("status") != "READY"
        or not isinstance(rows, list)
        or len(rows) != 1
    ):
        raise PredictionBlocked(
            "EXACT_METADATA_UNAVAILABLE",
            reason=report.get("reason") or coverage.get("reason"),
        )
    form = Path(str(rows[0].get("csv_path") or ""))
    sidecar = Path(str(rows[0].get("sidecar_path") or ""))
    if (
        not form.is_file()
        or not sidecar.is_file()
        or sidecar != form.with_name(form.name + ".metadata.json")
    ):
        raise PredictionBlocked("EXACT_METADATA_UNAVAILABLE")
    return form, sidecar


def _default_fetch(
    context: Mapping[str, Any], isolated_db: Path, timeout: float
) -> Mapping[str, Any]:
    from scripts import autonomous_live_odds_capture as capture

    form_csv = Path(str(context["form_csv"]))
    current_time = datetime.now().astimezone()
    jump = context["jump_timestamp"]
    if current_time >= jump:
        raise PredictionBlocked("POST_JUMP")
    item = capture.build_plan_item(form_csv, current_time)
    target_race_id = str(context["race_id"])
    non_window_blockers = [
        reason
        for reason in item.get("blockers") or []
        if reason
        not in {
            "outside_capture_windows",
            "capture_window_passed",
            "race_too_far_away",
        }
    ]
    if item.get("race_id") != target_race_id or non_window_blockers:
        raise PredictionBlocked(
            "EXACT_METADATA_UNAVAILABLE", reasons=non_window_blockers
        )
    with contextlib.redirect_stdout(sys.stderr):
        fetched = capture.fetch_odds_for_target_race_with_timeout(
            str(isolated_db),
            item.get("venue"),
            item.get("race_number"),
            item.get("race_date"),
            allow_auto_scrape_odds=True,
            timeout_seconds=timeout,
        )
    validation = capture.validate_fetched_odds(item, fetched)
    captured_at = datetime.now().astimezone()
    if captured_at >= jump:
        raise PredictionBlocked("POST_JUMP")
    return {
        "captured_at": captured_at.isoformat(),
        "plan_item": item,
        "validation": validation,
        "fetch_result": {
            key: value
            for key, value in fetched.items()
            if key not in {"odds_data", "odds_data_place", "race_info"}
        },
    }


def default_dependencies(args: argparse.Namespace) -> Dependencies:
    from scripts.predict_market_form_residual import score_from_artifacts

    run_id = f"on_demand_{uuid.uuid4().hex}"

    def acquire_lock() -> Any:
        return _acquire_collector_lock_no_steal(
            Path(args.lock_path),
            run_id=run_id,
            output_dir=Path(args.lock_output_dir),
        )

    def release_lock(handle: Any) -> None:
        if not isinstance(handle, OwnedCollectorLock):
            raise PredictionBlocked("LOCK_RELEASE_FAILED", reason="handle_invalid")
        _release_owned_collector_lock(handle)

    return Dependencies(
        schedule=_default_schedule,
        refresh=_default_refresh,
        discover_receipt=discover_capture_handoff,
        fetch_odds=_default_fetch,
        acquire_lock=acquire_lock,
        release_lock=release_lock,
        lock_busy_type=CollectorLockBusy,
        seal_features=seal_live_features,
        score_residual=score_from_artifacts,
        now=lambda: datetime.now().astimezone(),
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


def _capture_attempt(
    *, race_id: str, captured_at: datetime, validation: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": "autonomous_live_odds_capture_attempt_v1",
        "race_id": race_id,
        "status": "APPENDED",
        "reasons": [],
        "fetch_time": captured_at.isoformat(),
        "append_time": captured_at.isoformat(),
        "persistence_scope": "isolated_research_bundle_only",
        "validation": dict(validation),
    }


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


def _persist_blocked_bundle(bundle: Path, exc: PredictionBlocked) -> None:
    """Seal the smallest post-bundle blocker as a research-only result."""

    bundle_path = str(bundle.resolve())
    blocker = {"code": exc.code, **exc.details}
    result = {
        "schema_version": "on_demand_race_prediction_v1",
        "status": exc.code,
        "research_only": True,
        "production_persisted": False,
        "betting_output": False,
        "blockers": [blocker],
        "bundle": bundle_path,
    }
    _write_canonical(bundle / "result.json", result)
    _write_canonical(bundle / "bundle_manifest.json", bundle_manifest(bundle))
    exc.details["bundle"] = bundle_path


def _discover(
    dependencies: Dependencies,
    *,
    evidence_roots: Sequence[Path],
    db_path: Path,
    race_id: str,
    jump: datetime,
    current_time: datetime,
) -> Mapping[str, Any] | None:
    from scripts.autonomous_live_odds_capture import due_capture_window

    capture_window, _ = due_capture_window((jump - current_time).total_seconds() / 60.0)
    if capture_window is None:
        return None
    try:
        return dependencies.discover_receipt(
            evidence_roots=evidence_roots,
            db_path=db_path,
            race_id=race_id,
            jump_datetime=jump,
            capture_window_minutes=capture_window,
            current_time=current_time,
        )
    except CaptureHandoffError as exc:
        reason = str(exc)
        code = "RECEIPT_AMBIGUOUS" if "ambiguous" in reason else "RECEIPT_INVALID"
        raise PredictionBlocked(code, reason=reason) from exc


def _acquire_or_reuse(
    dependencies: Dependencies,
    *,
    odds_source: str,
    evidence_roots: Sequence[Path],
    db_path: Path,
    race_id: str,
    jump: datetime,
    current_time: datetime,
    wait_seconds: float,
    poll_seconds: float,
) -> tuple[Any | None, Mapping[str, Any] | None]:
    if odds_source in {"auto", "receipt"}:
        receipt = _discover(
            dependencies,
            evidence_roots=evidence_roots,
            db_path=db_path,
            race_id=race_id,
            jump=jump,
            current_time=current_time,
        )
        if receipt is not None:
            return None, receipt
        if odds_source == "receipt":
            raise PredictionBlocked("RECEIPT_UNAVAILABLE")
    started = dependencies.monotonic()
    while True:
        try:
            return dependencies.acquire_lock(), None
        except dependencies.lock_busy_type as exc:
            elapsed = dependencies.monotonic() - started
            if odds_source == "auto":
                check_time = current_time + timedelta(seconds=max(0.0, elapsed))
                receipt = _discover(
                    dependencies,
                    evidence_roots=evidence_roots,
                    db_path=db_path,
                    race_id=race_id,
                    jump=jump,
                    current_time=check_time,
                )
                if receipt is not None:
                    return None, receipt
            remaining = wait_seconds - elapsed
            if remaining <= 0:
                details = getattr(exc, "payload", None)
                raise PredictionBlocked("BUSY", lock_details=details) from exc
            dependencies.sleep(min(max(poll_seconds, 0.01), remaining))


def _run_prediction(
    args: argparse.Namespace, dependencies: Dependencies, state: dict[str, Path]
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
    races = dependencies.schedule(int(args.days_ahead))
    resolved, target, matches = resolve_target_race(
        races, race_id=None, race_query=args.race
    )
    if resolved != "RESOLVED" or target is None:
        raise PredictionBlocked(resolved, matching_race_ids=matches)
    race_id = str(stable_race_id(target) or "")
    jump = _parse_race_jump_datetime(target, now=current_time)
    if not race_id or jump is None:
        raise PredictionBlocked("EXACT_RACE_IDENTITY_UNAVAILABLE")
    if jump <= current_time:
        raise PredictionBlocked(
            "POST_JUMP", race_id=race_id, jump_timestamp=jump.isoformat()
        )

    bundle = create_bundle(Path(args.output_root), current_time)
    state["bundle"] = bundle
    request = {
        "schema_version": "on_demand_prediction_request_v1",
        "race_query": args.race,
        "race_id": race_id,
        "jump_timestamp": jump.isoformat(),
        "request_timestamp": current_time.isoformat(),
        "odds_source": args.odds_source,
        "model": _public_model(model),
        "config_sha256": config_sha,
        "research_only": True,
    }
    _write_canonical(bundle / "request.json", request)
    write_exact_bytes(bundle / "config.json", config_raw)
    _copy_exact(model.schema_path, bundle / "model" / "config.schema.json")
    if model.model_path and model.manifest_path:
        _copy_exact(model.model_path, bundle / "model" / "model.json")
        _copy_exact(model.manifest_path, bundle / "model" / "manifest.json")

    lock_handle: Any | None = None
    try:
        lock_handle, handoff = _acquire_or_reuse(
            dependencies,
            odds_source=args.odds_source,
            evidence_roots=tuple(Path(path) for path in args.capture_evidence_root),
            db_path=Path(args.db),
            race_id=race_id,
            jump=jump,
            current_time=current_time,
            wait_seconds=float(config["bundle"]["lock_wait_seconds"]),
            poll_seconds=float(config["bundle"]["poll_seconds"]),
        )
        if handoff is not None:
            receipt, capture_raw, form_raw, sidecar_raw = receipt_from_handoff(
                handoff,
                current_time=current_time,
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
            form_csv, sidecar = dependencies.refresh(
                target, bundle, current_time, int(args.days_ahead)
            )
            form_csv, sidecar = _validated_refreshed_sources(bundle, form_csv, sidecar)
            plan_context = {
                "race_id": race_id,
                "form_csv": str(form_csv),
                "current_time": dependencies.now(),
                "jump_timestamp": jump,
            }
            provisional_names: list[str] = []
            try:
                sidecar_value = json.loads(sidecar.read_bytes())
                participants = (
                    sidecar_value.get("participants")
                    or sidecar_value.get("runners")
                    or []
                )
                provisional_names = [
                    str(row.get("dog_name") or row.get("name") or "")
                    for row in participants
                    if isinstance(row, Mapping)
                ]
            except (OSError, json.JSONDecodeError):
                pass
            sealed_db = bundle / "features" / "sealed_history.db"
            history = seal_history_database(
                source=Path(args.db),
                target=sealed_db,
                target_race_id=race_id,
                cutoff=jump,
                runner_names=provisional_names,
            )
            _write_canonical(bundle / "features" / "history_seal.json", history)
            try:
                fetched = dependencies.fetch_odds(
                    plan_context, sealed_db, float(args.fetch_timeout_seconds)
                )
            except PredictionBlocked:
                raise
            except Exception as exc:
                raise PredictionBlocked(
                    "MARKET_UNAVAILABLE", error=type(exc).__name__
                ) from exc
            try:
                captured_at = datetime.fromisoformat(str(fetched["captured_at"]))
                validation = fetched["validation"]
            except (KeyError, TypeError, ValueError) as exc:
                raise PredictionBlocked("MARKET_UNAVAILABLE") from exc
            if not isinstance(validation, Mapping):
                raise PredictionBlocked("MARKET_UNAVAILABLE")
            receipt = normalize_validation_receipt(
                race_id=race_id,
                captured_at=captured_at,
                validation=validation,
                source_kind="isolated_immediate_capture",
            )
            attempt = _capture_attempt(
                race_id=race_id, captured_at=captured_at, validation=validation
            )
            capture_path = bundle / "source" / "capture.json"
            _write_canonical(capture_path, attempt)
            _write_canonical(bundle / "source" / "capture_provenance.json", fetched)
    finally:
        if lock_handle is not None:
            dependencies.release_lock(lock_handle)

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

    result = {
        "schema_version": "on_demand_race_prediction_v1",
        "status": "PREDICTION_READY",
        "research_only": True,
        "production_persisted": False,
        "betting_output": False,
        "race": {
            "query": args.race,
            "race_id": race_id,
            "jump_timestamp": jump.isoformat(),
        },
        "score_timestamp": score_time.isoformat(),
        "odds_source": receipt["source_kind"],
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
    _write_canonical(bundle / "result.json", result)
    manifest = bundle_manifest(bundle)
    _write_canonical(bundle / "bundle_manifest.json", manifest)
    return result


def run_prediction(
    args: argparse.Namespace, dependencies: Dependencies
) -> dict[str, Any]:
    state: dict[str, Path] = {}
    try:
        return _run_prediction(args, dependencies, state)
    except PredictionBlocked as exc:
        bundle = state.get("bundle")
        if bundle is not None:
            try:
                _persist_blocked_bundle(bundle, exc)
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
        if bundle is not None:
            try:
                _persist_blocked_bundle(bundle, blocked)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--race", help="Exact named upcoming race, e.g. 'gunnedah r5'")
    target.add_argument("--replay-bundle", type=Path)
    parser.add_argument("--model", default="latest-research")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/prediction/manual-default.json"
    )
    parser.add_argument(
        "--odds-source", choices=("auto", "receipt", "capture"), default="auto"
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--days-ahead", type=int, default=1)
    parser.add_argument("--current-time")
    parser.add_argument("--fetch-timeout-seconds", type=float, default=45.0)
    parser.add_argument(
        "--capture-evidence-root",
        action="append",
        type=Path,
        default=list(DEFAULT_CAPTURE_EVIDENCE_ROOTS),
    )
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--lock-output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.replay_bundle:
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
    return 0 if output.get("status") == "PREDICTION_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
