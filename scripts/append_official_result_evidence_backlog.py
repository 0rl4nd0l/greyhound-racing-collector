#!/usr/bin/env python3
"""Append already-collected official-result artifacts into evidence tables.

This runner is intentionally narrow: it only consumes source-backed
``official_result_*.jsonl`` artifacts produced by
``autonomous_official_result_capture.py`` and appends validated rows into the
append-only official-result evidence tables. It never writes canonical labels,
rewrites snapshots, trains models, or changes production pointers.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from scripts import autonomous_official_result_capture as capture


DEFAULT_OUTPUT_PREFIX = (
    "artifacts/full_evidence_orchestration_20260525/"
    "official_result_evidence_append_backlog_"
)
NO_WRITE_GUARANTEES = {
    **capture.NO_WRITE_GUARANTEES,
    "db_write": False,
    "label_write": False,
}


def now_id(now: datetime | None = None) -> str:
    return (now or datetime.now().astimezone()).strftime("%Y%m%dT%H%M%S%z")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        logical = path if path.is_absolute() else ROOT / path
        return logical.absolute().relative_to(ROOT.absolute()).as_posix()
    except ValueError:
        return str(path)


def assert_output_dir_safe(output_dir: Path) -> Path:
    logical = output_dir if output_dir.is_absolute() else ROOT / output_dir
    try:
        relative = logical.absolute().relative_to(ROOT.absolute())
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    if ".." in relative.parts:
        raise ValueError("output_dir_must_not_contain_parent_traversal")
    if not relative.as_posix().startswith(DEFAULT_OUTPUT_PREFIX):
        raise ValueError(
            "output_dir_must_be_official_result_evidence_append_backlog_artifact:"
            f"{relative}"
        )
    return logical.absolute()


def unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    for index in range(1, 1000):
        candidate = Path(f"{base}_{index:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"output_dir_collision_exhausted:{base}")


def artifact_paths(artifact_dir: Path) -> dict[str, Path]:
    return {
        "race_rows": artifact_dir / "official_result_races.jsonl",
        "runner_rows": artifact_dir / "official_result_runners.jsonl",
        "quarantine_rows": artifact_dir / "official_result_quarantine.jsonl",
    }


def missing_artifact_status(
    *,
    artifact_dir: Path,
    paths: Mapping[str, Path],
    execute: bool,
) -> dict[str, Any]:
    missing = [
        key
        for key in ("race_rows", "runner_rows")
        if not paths[key].exists()
    ]
    return {
        **capture.evidence_db_ingest_not_executed(),
        "execute": execute,
        "status": "BLOCKED_ARTIFACT_FILES_MISSING",
        "artifact_dir": relpath(artifact_dir),
        "missing_files": [relpath(paths[key]) for key in missing],
        "blocker_reason_counts": {"artifact_files_missing": len(missing)},
        "db_write_performed": False,
        "label_write_performed": False,
    }


def process_artifact_dir(
    *,
    artifact_dir: Path,
    db_path: Path,
    execute: bool,
    lock_status: Mapping[str, Any] | None,
    require_lock_free: bool,
) -> dict[str, Any]:
    logical_artifact_dir = artifact_dir if artifact_dir.is_absolute() else ROOT / artifact_dir
    paths = artifact_paths(logical_artifact_dir)
    item: dict[str, Any] = {
        "artifact_dir": relpath(logical_artifact_dir),
        "race_rows_path": relpath(paths["race_rows"]),
        "runner_rows_path": relpath(paths["runner_rows"]),
        "quarantine_rows_path": relpath(paths["quarantine_rows"]),
        "execute": execute,
    }
    if not paths["race_rows"].exists() or not paths["runner_rows"].exists():
        status = missing_artifact_status(
            artifact_dir=logical_artifact_dir,
            paths=paths,
            execute=execute,
        )
    else:
        rows = capture.load_official_result_artifact_rows(
            race_rows_path=paths["race_rows"],
            runner_rows_path=paths["runner_rows"],
            quarantine_rows_path=(
                paths["quarantine_rows"] if paths["quarantine_rows"].exists() else None
            ),
        )
        item["race_ids"] = sorted(
            {
                str(row.get("race_id") or "")
                for row in rows.get("race_rows") or []
                if row.get("race_id")
            }
        )
        if (
            execute
            and require_lock_free
            and lock_status is not None
            and not bool(lock_status.get("write_allowed"))
        ):
            status = capture.official_result_evidence_ingest_blocked_by_lock(
                db_path=db_path,
                artifact_rows=rows,
                shared_lock=lock_status,
            )
        else:
            status = capture.append_official_result_evidence_to_db(
                db_path=db_path,
                artifact_rows=rows,
                output_dir=logical_artifact_dir,
                execute=execute,
            )
    item["official_result_evidence_db_ingest"] = status
    item["status"] = status.get("status")
    item["valid_race_rows"] = status.get("valid_race_rows", 0)
    item["valid_runner_rows"] = status.get("valid_runner_rows", 0)
    item["inserted_race_rows"] = status.get("inserted_race_rows", 0)
    item["inserted_runner_rows"] = status.get("inserted_runner_rows", 0)
    item["db_write_performed"] = bool(status.get("db_write_performed"))
    item["label_write_performed"] = bool(status.get("label_write_performed"))
    return item


def build_report(
    *,
    generated_at: datetime,
    db_path: Path,
    artifact_dirs: Sequence[Path],
    output_dir: Path,
    execute: bool,
    lock_path: Path | None,
    require_lock_free: bool,
    items: Sequence[Mapping[str, Any]],
    lock_status: Mapping[str, Any] | None,
) -> dict[str, Any]:
    status_counts = Counter(str(item.get("status") or "UNKNOWN") for item in items)
    inserted_race_rows = sum(int(item.get("inserted_race_rows") or 0) for item in items)
    inserted_runner_rows = sum(int(item.get("inserted_runner_rows") or 0) for item in items)
    db_write_performed = any(bool(item.get("db_write_performed")) for item in items)
    if not items:
        final_status = "NO_ARTIFACTS"
    elif db_write_performed:
        final_status = "APPENDED_OFFICIAL_RESULT_EVIDENCE_BACKLOG"
    elif any(str(item.get("status")) == "BLOCKED_SHARED_LOCK_HELD" for item in items):
        final_status = "BLOCKED_SHARED_LOCK_HELD"
    elif all(str(item.get("status") or "").startswith("NOOP_") for item in items):
        final_status = "NOOP_ALREADY_PRESENT"
    elif execute:
        final_status = "NO_DB_WRITE_PERFORMED"
    else:
        final_status = "READY_NOT_EXECUTED"

    return {
        "schema_version": "official_result_evidence_append_backlog_report_v1",
        "generated_at": generated_at.isoformat(),
        "output_dir": relpath(output_dir),
        "db_path": str(db_path),
        "execute": execute,
        "require_lock_free": require_lock_free,
        "lock_path": str(lock_path) if lock_path else None,
        "shared_lock_status": dict(lock_status) if lock_status is not None else None,
        "artifact_count": len(artifact_dirs),
        "artifact_dirs": [relpath(path) for path in artifact_dirs],
        "processed_count": len(items),
        "final_status": final_status,
        "status_counts": dict(sorted(status_counts.items())),
        "inserted_race_rows": inserted_race_rows,
        "inserted_runner_rows": inserted_runner_rows,
        "db_write_performed": db_write_performed,
        "label_write_performed": any(bool(item.get("label_write_performed")) for item in items),
        "items": list(items),
        "no_write_guarantees": {
            **NO_WRITE_GUARANTEES,
            "db_write": db_write_performed,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, action="append", required=True)
    parser.add_argument("--db", type=Path, default=ROOT / "greyhound_racing_data.db")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--execute-db-ingest",
        action="store_true",
        help="Append into official-result evidence tables. Canonical labels are never written.",
    )
    parser.add_argument("--lock-path", type=Path)
    parser.add_argument(
        "--require-lock-free",
        action="store_true",
        help="Fail closed without DB writes if the shared daemon lock is live.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    generated_at = datetime.now().astimezone()
    output_dir = assert_output_dir_safe(
        args.output_dir
        or capture.DEFAULT_EVIDENCE_ROOT
        / f"official_result_evidence_append_backlog_{now_id(generated_at)}"
    )
    output_dir = unique_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    lock_status = (
        capture.shared_lock_status(args.lock_path)
        if args.require_lock_free and args.execute_db_ingest
        else None
    )
    items = [
        process_artifact_dir(
            artifact_dir=artifact_dir,
            db_path=args.db,
            execute=args.execute_db_ingest,
            lock_status=lock_status,
            require_lock_free=args.require_lock_free,
        )
        for artifact_dir in args.artifact_dir
    ]
    report = build_report(
        generated_at=generated_at,
        db_path=args.db,
        artifact_dirs=args.artifact_dir,
        output_dir=output_dir,
        execute=args.execute_db_ingest,
        lock_path=args.lock_path,
        require_lock_free=args.require_lock_free,
        items=items,
        lock_status=lock_status,
    )
    write_json(output_dir / "official_result_evidence_append_backlog_report.json", report)
    write_text(output_dir / "final_status.txt", str(report["final_status"]) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
