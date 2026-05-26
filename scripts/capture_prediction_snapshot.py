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
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accuracy_program.snapshots import (  # noqa: E402
    assert_no_result_fields,
    build_prediction_snapshot,
    persist_prediction_snapshot,
)
from utils.runner_completeness import analyze_csv_runner_completeness  # noqa: E402
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
) -> dict[str, Any]:
    from app import enhance_prediction_with_csv_meta, run_prediction_for_race_file

    odds_capture: dict[str, Any] | None = None
    if (
        capture_live_odds
        and not mechanics_only
        and getattr(lifecycle, "status", None) == UPCOMING_NOT_JUMPED
    ):
        odds_capture = _capture_live_odds_for_lifecycle(
            db_path=db_path,
            lifecycle=lifecycle,
        )

    prediction_timestamp = datetime.now().isoformat(timespec="seconds")
    source_runner_completeness = analyze_csv_runner_completeness(race_file).as_dict()
    result = run_prediction_for_race_file(str(race_file))
    if not isinstance(result, dict) or not result.get("success"):
        return {
            "status": "FAILED",
            "race_file": str(race_file),
            "lifecycle_status": getattr(lifecycle, "status", None),
            "odds_capture_requested": bool(capture_live_odds),
            "odds_capture": odds_capture,
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
    )
    assert_no_result_fields(snapshot)
    live_lifecycle = snapshot.get("lifecycle_status") == UPCOMING_NOT_JUMPED
    runner_set_complete = snapshot.get("runner_set_complete") is True
    write_snapshot = bool(persist and live_lifecycle and runner_set_complete and not mechanics_only)
    persistence = persist_prediction_snapshot(
        snapshot,
        snapshot_dir,
        dry_run=not write_snapshot,
    )
    if persist and not write_snapshot:
        if not live_lifecycle:
            persistence["status"] = "skipped_non_live_lifecycle"
        elif not runner_set_complete:
            persistence["status"] = "skipped_incomplete_runner_set"
        else:
            persistence["status"] = "skipped_not_persistable"

    priced_rows = [
        row
        for row in snapshot.get("predictions") or []
        if row.get("ev_win") is not None and row.get("odds") is not None
    ]
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
        "odds_capture_requested": bool(capture_live_odds),
        "odds_capture": odds_capture,
        "priced_ev_runner_count": len(priced_rows),
        "snapshot_readiness": snapshot.get("snapshot_readiness"),
        "probability_sum_check": _probability_sum(snapshot),
        "persistence": persistence,
        "leakage_check": "passed_result_free_snapshot",
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
        )
        for path, lifecycle in targets
    ]

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
        "db_path": str(db_path),
        "snapshot_dir": str(Path(args.snapshot_dir)),
        "candidate_files": len(files),
        "lifecycle_counts": dict(counts),
        "capture_count": len(captures),
        "captures": captures,
        "data_missing": data_missing,
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
