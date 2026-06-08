#!/usr/bin/env python3
"""Capture durable, result-free pre-jump prediction snapshots.

Default mode is dry-run. Use --persist to write JSON snapshots under the local
snapshot directory. The script never scrapes by default and refuses to persist
non-live lifecycle states.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accuracy_program.snapshots import (  # noqa: E402
    assert_no_result_fields,
    build_prediction_snapshot,
    persist_prediction_snapshot,
)
from utils.runner_completeness import (  # noqa: E402
    analyze_csv_runner_completeness,
    canonical_race_url_from_sidecar,
    fetch_canonical_runner_set,
    verify_final_runner_set,
)
from utils.csv_metadata import verify_canonical_sidecar_target_metadata  # noqa: E402
from utils.race_lifecycle import (  # noqa: E402
    STALE_FORM_GUIDE,
    UPCOMING_NOT_JUMPED,
    classify_race_file,
)


SAFE_ENV_DEFAULTS = {
    "WATCH_DOWNLOADS": "0",
    "WATCH_UPCOMING": "0",
    "PREDICTION_IMPORT_MODE": "prediction_only",
    "ENABLE_LIVE_SCRAPING": "0",
    "ENABLE_RESULTS_SCRAPERS": "0",
    "ENABLE_AUTO_SCRAPE_ODDS": "0",
    "SPORTSBET_DOM_FALLBACK_ODDS": "0",
    "TGR_ENABLED": "0",
    "DISABLE_SPORTSBET_INTEGRATOR": "1",
    "INGEST_EMBEDDED_HISTORY_ON_PREDICT": "0",
    "MEM_LOGGER_DISABLED": "1",
    "MEM_WATCHDOG_DISABLED": "1",
}


def _safe_db_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def _configure_safe_runtime(db_path: Path) -> None:
    for key, value in SAFE_ENV_DEFAULTS.items():
        os.environ[key] = value
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    os.environ["DATABASE_PATH"] = str(db_path)
    os.environ["GREYHOUND_DB_PATH"] = str(db_path)
    os.environ["STAGING_DB_PATH"] = str(db_path)
    os.environ["ANALYTICS_DB_PATH"] = str(db_path)
    os.environ["SINGLE_DB_MODE"] = "1"


def _candidate_files(race_files: list[str], upcoming_dir: str) -> list[Path]:
    if race_files:
        out: list[Path] = []
        for raw in race_files:
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = (ROOT / path).resolve()
            if path.exists():
                out.append(path)
                continue
            fallback = ROOT / upcoming_dir / raw
            if fallback.exists():
                out.append(fallback.resolve())
        return sorted(out)
    directory = Path(upcoming_dir)
    if not directory.is_absolute():
        directory = ROOT / directory
    return sorted(path.resolve() for path in directory.glob("*.csv"))


def _probability_sum(snapshot: dict[str, Any]) -> dict[str, Any]:
    probs = []
    for row in snapshot.get("predictions") or []:
        try:
            probs.append(float(row.get("win_prob_norm")))
        except Exception:
            continue
    return {
        "runner_count": len(snapshot.get("predictions") or []),
        "probability_sum": sum(probs) if probs else None,
        "abs_error": abs(sum(probs) - 1.0) if probs else None,
    }


def _set_readiness_requirement(
    snapshot: dict[str, Any],
    key: str,
    value: bool,
) -> None:
    readiness = snapshot.get("snapshot_readiness")
    if not isinstance(readiness, dict):
        return
    requirements = readiness.get("requirements")
    if not isinstance(requirements, dict):
        return
    requirements[key] = bool(value)
    readiness["status"] = "READY" if all(requirements.values()) else "NOT_READY"


def _apply_target_metadata_to_snapshot(
    snapshot: dict[str, Any],
    target_metadata: dict[str, Any],
) -> None:
    verified = target_metadata.get("target_metadata_status") == "verified"
    snapshot["target_metadata_status"] = target_metadata.get("target_metadata_status")
    snapshot["target_metadata_failure_reason"] = target_metadata.get(
        "target_metadata_failure_reason"
    )
    snapshot["target_distance"] = target_metadata.get("target_distance") if verified else None
    snapshot["target_grade"] = target_metadata.get("target_grade") if verified else None
    snapshot["target_distance_source"] = (
        target_metadata.get("target_distance_source") if verified else None
    )
    snapshot["target_grade_source"] = (
        target_metadata.get("target_grade_source") if verified else None
    )
    snapshot["metadata_is_leakage_safe"] = bool(verified)
    snapshot["metadata_source_detail"] = (
        target_metadata.get("metadata_source_detail") if verified else None
    )
    snapshot["canonical_race_url"] = target_metadata.get("canonical_race_url")
    snapshot["race_time_mapping_status"] = target_metadata.get("race_time_mapping_status")
    snapshot["race_time_source"] = target_metadata.get("race_time_source")
    snapshot["target_metadata_verification"] = target_metadata
    _set_readiness_requirement(snapshot, "target_metadata_verified", verified)


def _readiness_failure_categories(snapshot: dict[str, Any]) -> list[str]:
    readiness = snapshot.get("snapshot_readiness")
    requirements = (
        readiness.get("requirements")
        if isinstance(readiness, dict) and isinstance(readiness.get("requirements"), dict)
        else {}
    )
    categories: set[str] = set()
    for key, value in requirements.items():
        if value is True:
            continue
        if key in {"pre_jump_lifecycle"}:
            categories.add("lifecycle")
        elif key in {
            "source_runner_set_complete",
            "predictions_match_source_runner_set",
            "final_runner_set_verified",
        }:
            categories.add("runner_set")
        elif key == "target_metadata_verified":
            categories.add("metadata")
        elif "odds" in key:
            categories.add("odds_provenance")
        else:
            categories.add("data_integrity")
    return sorted(categories)


def _persistence_skip_category(
    *,
    live_lifecycle: bool,
    runner_set_complete: bool,
    final_runner_verified: bool,
    target_metadata_verified: bool,
    allow_unverified_runner_set: bool,
    mechanics_only: bool,
) -> str | None:
    if mechanics_only:
        return "lifecycle"
    if not live_lifecycle:
        return "lifecycle"
    if not runner_set_complete:
        return "runner_set"
    if not final_runner_verified:
        return "runner_set"
    if not target_metadata_verified:
        return "metadata"
    return None


def _should_write_snapshot(
    *,
    persist: bool,
    live_lifecycle: bool,
    runner_set_complete: bool,
    final_runner_verified: bool,
    target_metadata_verified: bool,
    allow_unverified_runner_set: bool,
    mechanics_only: bool,
) -> bool:
    return bool(
        persist
        and live_lifecycle
        and runner_set_complete
        and final_runner_verified
        and target_metadata_verified
        and not mechanics_only
    )


def _capture_live_odds_for_lifecycle(
    *,
    db_path: Path,
    lifecycle: Any,
) -> dict[str, Any]:
    from odds_auto_integrator import ensure_odds_for_target_race

    venue = getattr(lifecycle, "venue", None)
    race_number = getattr(lifecycle, "race_number", None)
    race_date = getattr(lifecycle, "race_date", None)
    if not venue or not race_number or not race_date:
        return {
            "status": "DATA_MISSING",
            "success": False,
            "reason": "missing_lifecycle_target_identity",
            "venue": venue,
            "race_number": race_number,
            "race_date": race_date,
            "append_only": True,
        }
    return ensure_odds_for_target_race(
        str(db_path),
        venue,
        race_number,
        race_date,
        allow_auto_scrape_odds=True,
        append_only=True,
    )


def _capture_one(
    *,
    race_file: Path,
    lifecycle: Any,
    db_path: Path,
    snapshot_dir: Path,
    persist: bool,
    mechanics_only: bool,
    capture_live_odds: bool,
    allow_unverified_runner_set: bool,
) -> dict[str, Any]:
    from app import enhance_prediction_with_csv_meta, run_prediction_for_race_file

    prediction_timestamp = datetime.now().isoformat(timespec="seconds")
    source_runner_completeness = analyze_csv_runner_completeness(race_file).as_dict()
    canonical_url = canonical_race_url_from_sidecar(race_file)
    target_metadata = verify_canonical_sidecar_target_metadata(
        race_file,
        race_number=getattr(lifecycle, "race_number", None),
        canonical_url=canonical_url,
    )
    target_metadata_verified = (
        target_metadata.get("target_metadata_status") == "verified"
    )
    final_runner_set_verification: dict[str, Any] | None = None
    final_runner_verified = False
    if not mechanics_only and getattr(lifecycle, "status", None) == UPCOMING_NOT_JUMPED:
        canonical_runner_set = fetch_canonical_runner_set(canonical_url)
        final_runner_set_verification = verify_final_runner_set(
            source_runner_completeness,
            canonical_runner_set,
        )
        final_runner_verified = (
            final_runner_set_verification.get("final_runner_set_status") == "verified"
        )

    odds_capture: dict[str, Any] | None = None
    if (
        capture_live_odds
        and not mechanics_only
        and getattr(lifecycle, "status", None) == UPCOMING_NOT_JUMPED
        and final_runner_verified
        and target_metadata_verified
    ):
        odds_capture = _capture_live_odds_for_lifecycle(
            db_path=db_path,
            lifecycle=lifecycle,
        )

    result = run_prediction_for_race_file(str(race_file))
    if not isinstance(result, dict) or not result.get("success"):
        return {
            "status": "FAILED",
            "race_file": str(race_file),
            "lifecycle_status": getattr(lifecycle, "status", None),
            "odds_capture_requested": bool(capture_live_odds),
            "odds_capture": odds_capture,
            "final_runner_set_verification": final_runner_set_verification,
            "target_metadata_status": target_metadata.get("target_metadata_status"),
            "target_metadata_failure_reason": target_metadata.get(
                "target_metadata_failure_reason"
            ),
            "target_metadata": target_metadata,
            "error": (
                (result or {}).get("error") if isinstance(result, dict) else "prediction_failed"
            ),
        }
    try:
        result = enhance_prediction_with_csv_meta(result, str(race_file))
    except Exception:
        pass
    snapshot = build_prediction_snapshot(
        result,
        source_file_path=str(race_file),
        lifecycle=lifecycle,
        prediction_timestamp=prediction_timestamp,
        feature_freeze_timestamp=prediction_timestamp,
        source_runner_completeness=source_runner_completeness,
        final_runner_set_verification=final_runner_set_verification,
    )
    _apply_target_metadata_to_snapshot(snapshot, target_metadata)
    assert_no_result_fields(snapshot)
    live_lifecycle = snapshot.get("lifecycle_status") == UPCOMING_NOT_JUMPED
    runner_set_complete = snapshot.get("runner_set_complete") is True
    final_runner_verified = snapshot.get("final_runner_set_status") == "verified"
    target_metadata_verified = snapshot.get("target_metadata_status") == "verified"
    write_snapshot = _should_write_snapshot(
        persist=persist,
        live_lifecycle=live_lifecycle,
        runner_set_complete=runner_set_complete,
        final_runner_verified=final_runner_verified,
        target_metadata_verified=target_metadata_verified,
        allow_unverified_runner_set=allow_unverified_runner_set,
        mechanics_only=mechanics_only,
    )
    persistence = persist_prediction_snapshot(
        snapshot,
        snapshot_dir,
        dry_run=not write_snapshot,
        require_final_runner_verification=True,
    )
    if persist and not write_snapshot:
        if not live_lifecycle:
            persistence["status"] = "skipped_non_live_lifecycle"
        elif not runner_set_complete:
            persistence["status"] = "skipped_incomplete_runner_set"
        elif not final_runner_verified:
            persistence["status"] = "skipped_pre_jump_runner_set_unverified"
            persistence["reason"] = "pre_jump_runner_set_unverified"
            persistence["final_runner_set_status"] = snapshot.get("final_runner_set_status")
            persistence["final_runner_set_mismatch_reason"] = snapshot.get(
                "final_runner_set_mismatch_reason"
            )
        elif not target_metadata_verified:
            persistence["status"] = "skipped_target_metadata_not_verified"
            persistence["reason"] = "target_metadata_not_verified"
            persistence["target_metadata_status"] = snapshot.get("target_metadata_status")
            persistence["target_metadata_failure_reason"] = snapshot.get(
                "target_metadata_failure_reason"
            )
        else:
            persistence["status"] = "skipped_not_persistable"
        persistence["skip_category"] = _persistence_skip_category(
            live_lifecycle=live_lifecycle,
            runner_set_complete=runner_set_complete,
            final_runner_verified=final_runner_verified,
            target_metadata_verified=target_metadata_verified,
            allow_unverified_runner_set=allow_unverified_runner_set,
            mechanics_only=mechanics_only,
        )

    priced_rows = [
        row
        for row in snapshot.get("predictions") or []
        if row.get("ev_win") is not None and row.get("odds") is not None
    ]
    readiness_failure_categories = _readiness_failure_categories(snapshot)
    return {
        "status": "SUCCESS",
        "race_file": str(race_file),
        "mechanics_only_not_live": mechanics_only,
        "race_id": snapshot.get("race_id"),
        "stable_race_key": snapshot.get("stable_race_key"),
        "lifecycle_status": snapshot.get("lifecycle_status"),
        "snapshot_state": snapshot.get("snapshot_state"),
        "prediction_timestamp": snapshot.get("prediction_timestamp"),
        "feature_freeze_timestamp": snapshot.get("feature_freeze_timestamp"),
        "model_version": snapshot.get("model_version"),
        "runner_count": len(snapshot.get("predictions") or []),
        "runner_set_complete": runner_set_complete,
        "source_runner_completeness": source_runner_completeness,
        "final_runner_set_verified": final_runner_verified,
        "final_runner_set_status": snapshot.get("final_runner_set_status"),
        "final_runner_set_mismatch_reason": snapshot.get(
            "final_runner_set_mismatch_reason"
        ),
        "final_runner_set_verification": final_runner_set_verification,
        "target_metadata_status": snapshot.get("target_metadata_status"),
        "target_distance": snapshot.get("target_distance"),
        "target_grade": snapshot.get("target_grade"),
        "target_distance_source": snapshot.get("target_distance_source"),
        "target_grade_source": snapshot.get("target_grade_source"),
        "target_metadata_failure_reason": snapshot.get(
            "target_metadata_failure_reason"
        ),
        "target_metadata": target_metadata,
        "odds_capture_requested": bool(capture_live_odds),
        "odds_capture": odds_capture,
        "priced_ev_runner_count": len(priced_rows),
        "snapshot_readiness": snapshot.get("snapshot_readiness"),
        "snapshot_readiness_failure_categories": readiness_failure_categories,
        "persistence_skip_category": persistence.get("skip_category")
        or ("dry_run" if not persist else None),
        "probability_sum_check": _probability_sum(snapshot),
        "persistence": persistence,
        "leakage_check": "passed_result_free_snapshot",
    }


def _endpoint_health_checks() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for name, url in {
        "api_health": "http://127.0.0.1:5002/api/health",
        "model_health": "http://127.0.0.1:5002/api/model_health",
    }.items():
        try:
            with urlopen(url, timeout=0.75) as response:
                checks[name] = {
                    "status": "reachable" if response.status == 200 else "degraded",
                    "http_status": response.status,
                }
        except URLError as exc:
            checks[name] = {"status": "not_running_or_unreachable", "error": str(exc)}
        except Exception as exc:
            checks[name] = {"status": "error", "error": f"{type(exc).__name__}:{exc}"}
    return checks


def _sqlite_quick_check(db_path: Path) -> dict[str, Any]:
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            row = conn.execute("PRAGMA quick_check").fetchone()
        value = row[0] if row else None
        return {"status": "ok" if value == "ok" else "degraded", "quick_check": value}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}:{exc}"}


def _regular_checks(db_path: Path, captures: list[dict[str, Any]]) -> dict[str, Any]:
    model_versions = sorted(
        {
            str(capture.get("model_version"))
            for capture in captures
            if capture.get("model_version") not in (None, "", "unknown")
        }
    )
    unknown_model_version_count = sum(
        1 for capture in captures if capture.get("model_version") in (None, "", "unknown")
    )
    leakage_passed = all(
        capture.get("leakage_check") == "passed_result_free_snapshot"
        for capture in captures
        if capture.get("status") == "SUCCESS"
    )
    return {
        "endpoint_health": _endpoint_health_checks(),
        "model_version": {
            "status": "ok" if model_versions and unknown_model_version_count == 0 else "warning",
            "versions": model_versions,
            "unknown_count": unknown_model_version_count,
        },
        "calibration_drift": {
            "status": "not_evaluated_no_result_ingestion",
            "reason": "capture fix is label-free and does not ingest results",
        },
        "data_integrity": _sqlite_quick_check(db_path),
        "temporal_leakage": {
            "status": "passed" if leakage_passed else "not_run_or_failed",
            "guard": "assert_no_result_fields",
        },
    }


def capture_snapshots(args: argparse.Namespace) -> dict[str, Any]:
    db_path = _safe_db_path(args.db)
    if not db_path.exists():
        return {
            "status": "DATA_MISSING",
            "reason": "db_path_not_found",
            "db_path": str(db_path),
            "capture_count": 0,
            "captures": [],
            "data_missing": ["db_path_not_found"],
        }
    _configure_safe_runtime(db_path)
    files = _candidate_files(args.race_file or [], args.upcoming_dir)
    lifecycles = [
        (path, classify_race_file(path, db_path=str(db_path), source_context="csv_file"))
        for path in files
    ]
    counts = Counter(lifecycle.status for _, lifecycle in lifecycles)
    live_targets = [
        (path, lifecycle)
        for path, lifecycle in lifecycles
        if lifecycle.status == UPCOMING_NOT_JUMPED
    ]

    mechanics_only = False
    targets = live_targets
    data_missing: list[str] = []
    if not targets:
        data_missing.append("no_genuinely_upcoming_not_jumped_local_races")
        if args.mechanics_on_stale:
            mechanics_only = True
            targets = [
                (path, lifecycle)
                for path, lifecycle in lifecycles
                if lifecycle.status == STALE_FORM_GUIDE
            ][:1]

    if args.limit and args.limit > 0:
        targets = targets[: args.limit]

    captures = [
        _capture_one(
            race_file=path,
            lifecycle=lifecycle,
            db_path=db_path,
            snapshot_dir=Path(args.snapshot_dir),
            persist=bool(args.persist),
            mechanics_only=mechanics_only,
            capture_live_odds=bool(args.capture_live_odds),
            allow_unverified_runner_set=bool(args.allow_unverified_runner_set),
        )
        for path, lifecycle in targets
    ]
    final_runner_counts = Counter(
        str(capture.get("final_runner_set_status") or "not_checked")
        for capture in captures
    )
    target_metadata_counts = Counter(
        str(capture.get("target_metadata_status") or "not_checked")
        for capture in captures
    )
    persisted_with_top_level_metadata_count = sum(
        1
        for capture in captures
        if (capture.get("persistence") or {}).get("status") == "persisted"
        and capture.get("target_distance") not in (None, "")
        and capture.get("target_grade") not in (None, "")
    )

    if captures and mechanics_only:
        status = "MECHANICS_ONLY_NOT_LIVE"
    elif captures:
        status = "SUCCESS"
    else:
        status = "DATA_MISSING"

    return {
        "status": status,
        "dry_run": not bool(args.persist),
        "persist_requested": bool(args.persist),
        "odds_capture_requested": bool(args.capture_live_odds),
        "allow_unverified_runner_set": bool(args.allow_unverified_runner_set),
        "db_path": str(db_path),
        "snapshot_dir": str(Path(args.snapshot_dir)),
        "candidate_files": len(files),
        "lifecycle_counts": dict(counts),
        "final_runner_set_counts": dict(final_runner_counts),
        "target_metadata_counts": dict(target_metadata_counts),
        "metadata_verified_count": target_metadata_counts.get("verified", 0),
        "metadata_missing_count": target_metadata_counts.get("missing", 0),
        "metadata_unsafe_count": target_metadata_counts.get("unsafe", 0),
        "metadata_mismatch_count": target_metadata_counts.get("mismatch", 0),
        "persisted_with_top_level_metadata_count": persisted_with_top_level_metadata_count,
        "capture_count": len(captures),
        "captures": captures,
        "data_missing": data_missing,
        "regular_checks": _regular_checks(db_path, captures),
        "safe_runtime_env": {key: os.environ.get(key) for key in sorted(SAFE_ENV_DEFAULTS)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="greyhound_racing_data_writable.db")
    parser.add_argument("--upcoming-dir", default="upcoming_races")
    parser.add_argument("--snapshot-dir", default="artifacts/prediction_snapshots")
    parser.add_argument("--race-file", action="append", help="Specific local race CSV")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--persist", action="store_true", help="Write result-free JSON snapshots")
    parser.add_argument(
        "--capture-live-odds",
        action="store_true",
        help="Explicitly capture append-only Sportsbet dog-level win odds before prediction snapshots",
    )
    parser.add_argument(
        "--mechanics-on-stale",
        action="store_true",
        help="If no live races exist, run one stale-form-guide mechanics test without persisting",
    )
    parser.add_argument(
        "--allow-unverified-runner-set",
        action="store_true",
        help=(
            "Deprecated diagnostic flag; persistence still requires verified "
            "canonical pre-race runner-set verification"
        ),
    )
    parser.add_argument("--output", help="Optional report JSON path")
    args = parser.parse_args()

    report = capture_snapshots(args)
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    print(text)
    if args.output:
        out = Path(args.output)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0 if report.get("status") in {"SUCCESS", "MECHANICS_ONLY_NOT_LIVE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
