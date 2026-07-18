#!/usr/bin/env python3
"""Plan or explicitly run one lock-aware pre-jump named-race prediction."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import sys
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from scripts.refresh_prejump_upcoming import (  # noqa: E402
    _parse_race_jump_datetime,
    parse_current_time,
    stable_race_id,
    stable_race_id_variants,
    venue_exclusion_aliases,
)


OUTPUT_SCHEMA = "manual_priority_race_prediction_v1"
FIXED_CAPTURE_WINDOWS_MINUTES = (60, 30, 10, 2)
TERMINAL_STATUSES = {
    "PLAN_ONLY",
    "WAITING_FOR_DAEMON_LOCK",
    "WAITING_FOR_CAPTURE_WINDOW",
    "BLOCKED_RACE_NOT_FOUND",
    "BLOCKED_RACE_AMBIGUOUS",
    "BLOCKED_RACE_ALREADY_JUMPED",
    "BLOCKED_EXACT_METADATA",
    "BLOCKED_RUNNER_IDENTITY",
    "BLOCKED_ODDS_CAPTURE",
    "BLOCKED_FEATURE_SEAL",
    "BLOCKED_MANUAL_PREDICTION",
    "PREDICTION_READY",
}
DEFAULT_DB = ROOT / "greyhound_racing_data.db"
DEFAULT_MODEL_DIR = ROOT / "artifacts/frozen_models/market_form_residual_v1"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )


def _token(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value or "").upper())


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
    values.update(stable_race_id_variants(race))
    for alias in venue_exclusion_aliases(
        venue, source_url=race.get("url") or race.get("race_url")
    ):
        values.update(
            {f"{alias} race {number}", f"race {number} {alias}", f"{alias} r{number}"}
        )
    return {_token(value) for value in values if value}


def resolve_target_race(
    races: Sequence[Mapping[str, Any]],
    *,
    race_id: str | None,
    race_query: str | None,
) -> tuple[str, Mapping[str, Any] | None, list[str]]:
    """Resolve one exact race without silently choosing among multiple matches."""

    if bool(race_id) == bool(race_query):
        raise ValueError("exactly_one_of_race_id_or_race_required")
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


def next_fixed_window(
    *, jump_datetime: datetime, current_time: datetime
) -> dict[str, Any]:
    pending = []
    for offset in FIXED_CAPTURE_WINDOWS_MINUTES:
        target = jump_datetime - timedelta(minutes=offset)
        if target >= current_time:
            pending.append((target, offset))
    if pending:
        target, offset = sorted(pending)[0]
        return {
            "next_capture_window_minutes": offset,
            "next_capture_window_at": target.isoformat(),
            "seconds_until_next_window": max(
                0.0, (target - current_time).total_seconds()
            ),
        }
    return {
        "next_capture_window_minutes": None,
        "next_capture_window_at": None,
        "seconds_until_next_window": None,
    }


def base_output(
    *, status: str, current_time: datetime, target: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"unsupported_status:{status}")
    output: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA,
        "status": status,
        "generated_at": current_time.isoformat(),
        "fixed_capture_windows_minutes": list(FIXED_CAPTURE_WINDOWS_MINUTES),
        "activation": False,
        "persisted": False,
        "result_access": False,
        "model_mutation": False,
        "threshold_mutation": False,
        "betting": False,
    }
    if target is not None:
        output["race_id"] = stable_race_id(target)
        output["race_url"] = target.get("url") or target.get("race_url")
        output["venue"] = target.get("venue") or target.get("venue_name")
        output["race_number"] = target.get("race_number")
        output["race_date"] = target.get("date") or target.get("race_date")
        output["race_time"] = target.get("race_time") or target.get("jump_time")
    return output


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal_live_features(
    *,
    form_csv: Path,
    db_path: Path,
    output_dir: Path,
    current_time: datetime,
) -> dict[str, Path]:
    """Build and hash-bind fresh feature rows without running another predictor."""

    from scripts.run_feature_recovery_execution_v1 import DEFAULT_SCHEMA, load_json
    from scripts.run_shadow_non_tgr_rf_evaluation import (
        IMPLEMENTATION_FILES,
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
    _write_json(rows_path, rows)
    _write_json(
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
    _write_json(manifest_path, manifest)
    artifacts = {
        shadow_relpath(path): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in (rows_path, manifest_path, history_path)
    }
    implementation = {
        "schema_version": "shadow_implementation_file_manifest_v1",
        "output_dir": shadow_relpath(output_dir),
        "git_head": "manual_priority_runtime",
        "git_branch": "manual_priority_runtime",
        "implementation_files": list(IMPLEMENTATION_FILES),
        "implementation_file_hashes": {
            relative: _sha256(ROOT / relative) for relative in IMPLEMENTATION_FILES
        },
        "artifact_files": artifacts,
    }
    _write_json(implementation_path, implementation)
    return {
        "feature_rows": rows_path,
        "feature_manifest": manifest_path,
        "implementation_manifest": implementation_path,
    }


def _target_plan(plan: Mapping[str, Any], race: Mapping[str, Any]) -> dict[str, Any]:
    target_ids = stable_race_id_variants(race)
    canonical = stable_race_id(race)
    if canonical:
        target_ids.add(canonical)
    rows = [
        dict(row)
        for row in plan.get("races") or []
        if isinstance(row, Mapping) and str(row.get("race_id") or "") in target_ids
    ]
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("status") or "UNKNOWN")
        counts[key] = counts.get(key, 0) + 1
    return {
        **dict(plan),
        "races": rows,
        "status_counts": counts,
        "ready_count": counts.get("READY_TO_CAPTURE", 0),
        "limit": 1,
    }


def acquire_with_bounded_wait(
    *,
    acquire: Callable[[], Any],
    busy_type: type[BaseException],
    max_wait_seconds: float,
    poll_seconds: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[Any | None, float, Mapping[str, Any] | None]:
    started = monotonic()
    last_details: Mapping[str, Any] | None = None
    while True:
        try:
            return acquire(), monotonic() - started, last_details
        except busy_type as exc:
            details = getattr(exc, "payload", None)
            last_details = details if isinstance(details, Mapping) else None
            elapsed = monotonic() - started
            remaining = max_wait_seconds - elapsed
            if remaining <= 0:
                return None, elapsed, last_details
            sleeper(min(max(poll_seconds, 0.01), remaining))


def run_command(
    args: argparse.Namespace,
    *,
    races: Sequence[Mapping[str, Any]],
    current_time: datetime,
    refresh_fn: Callable[..., Mapping[str, Any]] | None = None,
    capture_plan_fn: Callable[..., Mapping[str, Any]] | None = None,
    capture_execute_fn: Callable[..., Mapping[str, Any]] | None = None,
    feature_seal_fn: Callable[..., Mapping[str, Path]] = seal_live_features,
    score_fn: Callable[..., Mapping[str, Any]] | None = None,
    acquire_fn: Callable[..., Any] | None = None,
    release_fn: Callable[..., Mapping[str, Any]] | None = None,
    busy_type: type[BaseException] | None = None,
    now_provider: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    now_provider = now_provider or (lambda: datetime.now().astimezone())
    resolved, target, matches = resolve_target_race(
        races, race_id=args.race_id, race_query=args.race
    )
    if resolved != "RESOLVED" or target is None:
        output = base_output(status=resolved, current_time=current_time)
        output["matching_race_ids"] = matches
        return output
    jump = _parse_race_jump_datetime(target, now=current_time)
    if jump is None:
        output = base_output(
            status="BLOCKED_EXACT_METADATA", current_time=current_time, target=target
        )
        output["reason"] = "exact_jump_timestamp_missing"
        return output
    if jump <= current_time:
        output = base_output(
            status="BLOCKED_RACE_ALREADY_JUMPED", current_time=current_time, target=target
        )
        output["jump_timestamp"] = jump.isoformat()
        return output
    if not args.execute_collection:
        output = base_output(status="PLAN_ONLY", current_time=current_time, target=target)
        output["jump_timestamp"] = jump.isoformat()
        output.update(next_fixed_window(jump_datetime=jump, current_time=current_time))
        output["collection_execution_requested"] = False
        return output
    if not args.allow_auto_scrape_odds:
        output = base_output(
            status="BLOCKED_ODDS_CAPTURE", current_time=current_time, target=target
        )
        output["reason"] = "allow_auto_scrape_odds_flag_not_set"
        return output

    if refresh_fn is None:
        from scripts.refresh_prejump_upcoming import refresh_prejump_upcoming

        refresh_fn = refresh_prejump_upcoming
    if capture_plan_fn is None or capture_execute_fn is None:
        from scripts import autonomous_live_odds_capture as capture

        capture_plan_fn = capture_plan_fn or capture.build_capture_plan
        capture_execute_fn = capture_execute_fn or capture.execute_capture_plan
    if score_fn is None:
        from scripts.predict_market_form_residual import score_from_artifacts

        score_fn = score_from_artifacts
    if acquire_fn is None or release_fn is None or busy_type is None:
        from scripts import shadow_autopilot_daemon as daemon

        acquire_fn = acquire_fn or daemon.acquire_lock
        release_fn = release_fn or daemon.release_lock
        busy_type = busy_type or daemon.LockBusy
    run_id = f"manual_priority_{uuid.uuid4().hex}"
    lock_path = Path(args.lock_path)

    def acquire() -> Any:
        return acquire_fn(
            lock_path=lock_path,
            run_id=run_id,
            stale_after_seconds=int(args.lock_stale_seconds),
            output_dir=Path(args.lock_output_dir),
        )

    lock, waited, details = acquire_with_bounded_wait(
        acquire=acquire,
        busy_type=busy_type,
        max_wait_seconds=float(args.max_wait_seconds),
        poll_seconds=float(args.poll_seconds),
    )
    if lock is None:
        output = base_output(
            status="WAITING_FOR_DAEMON_LOCK", current_time=current_time, target=target
        )
        output["waited_seconds"] = waited
        output["lock_details"] = details
        return output

    try:
        current_time = now_provider()
        if current_time >= jump:
            output = base_output(
                status="BLOCKED_RACE_ALREADY_JUMPED",
                current_time=current_time,
                target=target,
            )
            output["jump_timestamp"] = jump.isoformat()
            output["reason"] = "race_jumped_while_waiting_for_daemon_lock"
            return output
        with tempfile.TemporaryDirectory(prefix="manual-priority-race-") as temp_name:
            work_dir = Path(temp_name)
            upcoming_dir = work_dir / "upcoming"
            refresh_args = argparse.Namespace(
                upcoming_dir=upcoming_dir,
                days_ahead=int(args.days_ahead),
                min_minutes=0.0,
                max_minutes=max(1.0, float(args.days_ahead + 1) * 1440.0),
                limit=1,
                exclude_race_id=[],
                exclude_race_ids_file=None,
                include_race_id=[stable_race_id(target)],
                dry_run=False,
                current_time=current_time.isoformat(),
                require_safe_metadata=True,
            )
            try:
                refresh_report = refresh_fn(refresh_args)
            except Exception as exc:
                output = base_output(
                    status="BLOCKED_EXACT_METADATA",
                    current_time=current_time,
                    target=target,
                )
                output["reason"] = f"exact_refresh_failed:{type(exc).__name__}:{exc}"
                return output
            coverage = refresh_report.get("sidecar_metadata_coverage") or {}
            coverage_rows = coverage.get("races") if isinstance(coverage, Mapping) else []
            if (
                refresh_report.get("selected_count") != 1
                or refresh_report.get("status") != "SUCCESS"
                or coverage.get("status") != "READY"
                or not isinstance(coverage_rows, list)
                or len(coverage_rows) != 1
            ):
                output = base_output(
                    status="BLOCKED_EXACT_METADATA",
                    current_time=current_time,
                    target=target,
                )
                output["reason"] = refresh_report.get("reason") or coverage.get("reason")
                return output
            form_csv = Path(str(coverage_rows[0].get("csv_path") or ""))
            sidecar = Path(str(coverage_rows[0].get("sidecar_path") or ""))
            if not form_csv.is_file() or not sidecar.is_file():
                output = base_output(
                    status="BLOCKED_EXACT_METADATA",
                    current_time=current_time,
                    target=target,
                )
                output["reason"] = "exact_form_or_sidecar_missing"
                return output

            try:
                plan = _target_plan(
                    capture_plan_fn(
                        [upcoming_dir], current_time=current_time, limit=1
                    ),
                    target,
                )
            except Exception as exc:
                output = base_output(
                    status="BLOCKED_ODDS_CAPTURE",
                    current_time=current_time,
                    target=target,
                )
                output["reason"] = f"capture_plan_failed:{type(exc).__name__}:{exc}"
                return output
            if plan.get("ready_count") != 1:
                output = base_output(
                    status="WAITING_FOR_CAPTURE_WINDOW",
                    current_time=current_time,
                    target=target,
                )
                output["jump_timestamp"] = jump.isoformat()
                output["capture_plan_status_counts"] = plan.get("status_counts") or {}
                output.update(next_fixed_window(jump_datetime=jump, current_time=current_time))
                return output
            try:
                capture_report = capture_execute_fn(
                    plan,
                    db_path=Path(args.db),
                    current_time=current_time,
                    execute=True,
                    allow_auto_scrape_odds=True,
                    fetch_timeout_seconds=float(args.fetch_timeout_seconds),
                )
            except Exception as exc:
                output = base_output(
                    status="BLOCKED_ODDS_CAPTURE",
                    current_time=current_time,
                    target=target,
                )
                output["reason"] = f"capture_execution_failed:{type(exc).__name__}:{exc}"
                return output
            attempts = [
                row
                for row in capture_report.get("attempts") or []
                if isinstance(row, Mapping)
            ]
            appended = [row for row in attempts if row.get("status") == "APPENDED"]
            if len(appended) != 1:
                reasons = [str(reason) for row in attempts for reason in row.get("reasons") or []]
                identity_failure = any(
                    token in reason.lower()
                    for reason in reasons
                    for token in ("runner", "box", "identity", "scratch")
                )
                output = base_output(
                    status=(
                        "BLOCKED_RUNNER_IDENTITY"
                        if identity_failure
                        else "BLOCKED_ODDS_CAPTURE"
                    ),
                    current_time=current_time,
                    target=target,
                )
                output["capture_status_counts"] = capture_report.get("status_counts") or {}
                output["reasons"] = reasons
                output["idempotent_existing_capture"] = bool(
                    any(row.get("status") == "SKIPPED_ALREADY_CAPTURED" for row in attempts)
                )
                return output
            capture_path = work_dir / "capture.json"
            _write_json(capture_path, capture_report)
            feature_dir = work_dir / "features"
            feature_time = now_provider()
            if feature_time >= jump:
                output = base_output(
                    status="BLOCKED_RACE_ALREADY_JUMPED",
                    current_time=feature_time,
                    target=target,
                )
                output["jump_timestamp"] = jump.isoformat()
                output["reason"] = "race_jumped_before_feature_seal"
                return output
            try:
                sealed = feature_seal_fn(
                    form_csv=form_csv,
                    db_path=Path(args.db),
                    output_dir=feature_dir,
                    current_time=feature_time,
                )
            except Exception as exc:
                output = base_output(
                    status="BLOCKED_FEATURE_SEAL",
                    current_time=current_time,
                    target=target,
                )
                output["reason"] = f"{type(exc).__name__}:{exc}"
                return output
            try:
                score_time = now_provider()
                if score_time >= jump:
                    raise RuntimeError("race_jumped_before_manual_score")
                prediction = score_fn(
                    race_id=str(appended[0].get("race_id")),
                    form_csv_path=form_csv,
                    sidecar_path=sidecar,
                    feature_rows_path=Path(sealed["feature_rows"]),
                    feature_manifest_path=Path(sealed["feature_manifest"]),
                    implementation_manifest_path=Path(sealed["implementation_manifest"]),
                    capture_path=capture_path,
                    model_path=Path(args.model_dir) / "model.json",
                    manifest_path=Path(args.model_dir) / "manifest.json",
                    score_timestamp=score_time,
                )
            except Exception as exc:
                output = base_output(
                    status="BLOCKED_MANUAL_PREDICTION",
                    current_time=current_time,
                    target=target,
                )
                output["reason"] = f"{type(exc).__name__}:{exc}"
                return output
            output = base_output(
                status="PREDICTION_READY", current_time=current_time, target=target
            )
            output["prediction"] = prediction
            output["inserted_live_odds_rows"] = int(
                capture_report.get("inserted_live_odds_rows") or 0
            )
            output["feature_packet_ephemeral"] = True
            return output
    finally:
        release_report = release_fn(lock_path, run_id)
        if release_report.get("released") is not True:
            raise RuntimeError(f"manual_lock_release_failed:{release_report}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--race-id", help="Exact stable race ID")
    target.add_argument("--race", help="Exact named-race query, e.g. 'Sandown race 7'")
    parser.add_argument("--execute-collection", action="store_true")
    parser.add_argument("--allow-auto-scrape-odds", action="store_true")
    parser.add_argument("--days-ahead", type=int, default=1)
    parser.add_argument("--current-time")
    parser.add_argument("--max-wait-seconds", type=float, default=0.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--fetch-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--lock-path",
        type=Path,
        default=ROOT
        / "artifacts/full_evidence_orchestration_20260525/shadow_autopilot_daemon_runtime/shadow_autopilot.lock",
    )
    parser.add_argument(
        "--lock-output-dir",
        type=Path,
        default=ROOT / "artifacts/full_evidence_orchestration_20260525",
    )
    parser.add_argument("--lock-stale-seconds", type=int, default=3600)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.days_ahead < 0 or not 0 <= args.max_wait_seconds <= 600:
        raise SystemExit("invalid_bounded_wait_or_days_ahead")
    current_time = parse_current_time(args.current_time)
    from upcoming_race_browser import UpcomingRaceBrowser

    try:
        with contextlib.redirect_stdout(sys.stderr):
            races = UpcomingRaceBrowser(create_upcoming_dir=False).get_upcoming_races(
                days_ahead=args.days_ahead
            )
    except Exception as exc:
        output = base_output(
            status="BLOCKED_EXACT_METADATA", current_time=current_time
        )
        output["reason"] = f"schedule_discovery_failed:{type(exc).__name__}:{exc}"
    else:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                output = run_command(args, races=races, current_time=current_time)
        except Exception as exc:
            output = base_output(
                status="BLOCKED_MANUAL_PREDICTION", current_time=current_time
            )
            output["reason"] = f"command_failed:{type(exc).__name__}:{exc}"
    print(canonical_json(output))
    return 0 if output["status"] in {"PLAN_ONLY", "PREDICTION_READY"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
