#!/usr/bin/env python3
"""Build a report-only diagnostic for live-odds races missing shadow candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/autonomous_accuracy_odds_status_"
NO_WRITE_GUARANTEES = {
    "db_write": False,
    "label_write": False,
    "odds_write": False,
    "join_acceptance_changed": False,
    "registry_mutation": False,
    "production_pointer_update": False,
    "production_promotion": False,
    "training": False,
    "betting_or_ev_action": False,
    "tgr_enabled": False,
}

RACE_ID_RE = re.compile(r"^Race\s+(?P<race_number>\d+)\s+-\s+(?P<venue>.+)\s+-\s+(?P<race_date>\d{4}-\d{2}-\d{2})$")


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
    if not relative.as_posix().startswith(OUTPUT_PREFIX):
        raise ValueError(
            f"output_dir_must_be_autonomous_accuracy_odds_status_artifact:{relative}"
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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{path}")
    return payload


def load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"jsonl_row_must_be_object:{path}:{line_number}")
            rows.append(payload)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_race_id(race_id: str | None) -> dict[str, Any]:
    if not race_id:
        return {}
    match = RACE_ID_RE.match(race_id)
    if not match:
        return {}
    return {
        "race_number": int(match.group("race_number")),
        "venue": match.group("venue"),
        "race_date": match.group("race_date"),
    }


def race_ids_from_jsonl(path: Path | None) -> set[str]:
    return {
        str(row["race_id"])
        for row in load_jsonl(path)
        if row.get("race_id") not in (None, "")
    }


def race_ids_from_upcoming_dir(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    race_ids: set[str] = set()
    for csv_path in path.glob("Race * - *.csv"):
        race_ids.add(csv_path.stem)
    return race_ids


def extract_recovery_items(queue: Mapping[str, Any]) -> list[dict[str, Any]]:
    flat_items = queue.get("items")
    if isinstance(flat_items, list):
        return [dict(item) for item in flat_items if isinstance(item, dict)]

    items: list[dict[str, Any]] = []
    queues = queue.get("queues")
    if isinstance(queues, Mapping):
        for queue_name, queue_items in queues.items():
            if not isinstance(queue_items, list):
                continue
            for item in queue_items:
                if isinstance(item, dict):
                    row = dict(item)
                    row.setdefault("queue", queue_name)
                    items.append(row)
    return items


def nearby_races(race_id: str, available_race_ids: set[str]) -> list[str]:
    identity = parse_race_id(race_id)
    if not identity:
        return []
    candidates: list[tuple[int, int, str]] = []
    for candidate in available_race_ids:
        candidate_identity = parse_race_id(candidate)
        if (
            candidate_identity.get("venue") != identity["venue"]
            or candidate_identity.get("race_date") != identity["race_date"]
        ):
            continue
        candidate_number = candidate_identity.get("race_number")
        if not isinstance(candidate_number, int):
            continue
        distance = abs(candidate_number - identity["race_number"])
        candidates.append((distance, candidate_number, candidate))
    return [candidate for _, _, candidate in sorted(candidates)[:5]]


def coverage_cause(
    *,
    in_latest_shadow: bool,
    in_latest_stage2: bool,
    in_refreshed_upcoming: bool,
    in_candidate_source: bool,
) -> str:
    if in_latest_shadow and in_latest_stage2:
        return "covered_by_latest_shadow_predictions"
    if in_latest_shadow and not in_latest_stage2:
        return "latest_baseline_shadow_only_stage2_missing"
    if in_refreshed_upcoming and not in_latest_shadow:
        return "refreshed_upcoming_without_prediction_rows"
    if in_candidate_source and not in_latest_shadow:
        return "historical_candidate_available_but_not_latest_shadow_artifact"
    return "absent_from_shadow_candidate_sources"


def build_diagnostic(
    *,
    recovery_queue: Mapping[str, Any],
    shadow_prediction_race_ids: set[str],
    stage2_prediction_race_ids: set[str],
    refreshed_upcoming_race_ids: set[str],
    candidate_source_race_ids: set[str],
    generated_at: datetime,
    source_paths: Mapping[str, Path | None],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in extract_recovery_items(recovery_queue):
        if item.get("recovery_action") != "inspect_shadow_run_candidate_coverage":
            continue
        race_id = str(item.get("race_id") or "")
        in_latest_shadow = race_id in shadow_prediction_race_ids
        in_latest_stage2 = race_id in stage2_prediction_race_ids
        in_refreshed_upcoming = race_id in refreshed_upcoming_race_ids
        in_candidate_source = race_id in candidate_source_race_ids
        rows.append(
            {
                "race_id": race_id,
                "venue": item.get("venue"),
                "race_number": item.get("race_number"),
                "canonical_live_odds_race_id": item.get("canonical_live_odds_race_id"),
                "latest_live_odds_capture": item.get("latest_capture"),
                "live_odds_row_count": item.get("live_odds_row_count"),
                "live_odds_box_count": item.get("live_odds_box_count"),
                "queue_reason": item.get("reason"),
                "authorized_action": item.get("authorized_action"),
                "in_latest_shadow_predictions": in_latest_shadow,
                "in_latest_stage2_predictions": in_latest_stage2,
                "in_refreshed_upcoming": in_refreshed_upcoming,
                "in_shadow_candidate_source_report": in_candidate_source,
                "nearby_latest_shadow_races": nearby_races(
                    race_id, shadow_prediction_race_ids | stage2_prediction_race_ids
                ),
                "nearby_candidate_source_races": nearby_races(
                    race_id, candidate_source_race_ids
                ),
                "coverage_cause": coverage_cause(
                    in_latest_shadow=in_latest_shadow,
                    in_latest_stage2=in_latest_stage2,
                    in_refreshed_upcoming=in_refreshed_upcoming,
                    in_candidate_source=in_candidate_source,
                ),
                "next_authorized_action": (
                    "diagnostic_review_only_no_join_or_backfill; inspect why the "
                    "race was outside the exact shadow candidate source before any "
                    "manual recovery"
                ),
                "db_write_performed": False,
                "join_acceptance_changed": False,
            }
        )

    cause_counts = Counter(row["coverage_cause"] for row in rows)
    return {
        "schema_version": "shadow_candidate_coverage_diagnostic_v1",
        "generated_at": generated_at.isoformat(),
        "diagnostic_only": True,
        "no_write_guarantees": NO_WRITE_GUARANTEES,
        "source_paths": {key: relpath(value) for key, value in source_paths.items()},
        "input_counts": {
            "recovery_queue_items": len(extract_recovery_items(recovery_queue)),
            "shadow_prediction_race_count": len(shadow_prediction_race_ids),
            "stage2_prediction_race_count": len(stage2_prediction_race_ids),
            "refreshed_upcoming_race_count": len(refreshed_upcoming_race_ids),
            "candidate_source_race_count": len(candidate_source_race_ids),
        },
        "summary": {
            "diagnostic_race_count": len(rows),
            "coverage_cause_counts": dict(sorted(cause_counts.items())),
            "all_diagnostic_races_absent_from_latest_shadow_predictions": all(
                not row["in_latest_shadow_predictions"] for row in rows
            ),
            "all_diagnostic_races_absent_from_latest_stage2_predictions": all(
                not row["in_latest_stage2_predictions"] for row in rows
            ),
            "db_write_performed": False,
            "join_acceptance_changed": False,
        },
        "items": rows,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Shadow Candidate Coverage Diagnostic",
        "",
        f"Generated at: `{report['generated_at']}`",
        "",
        "This is a report-only diagnostic. It did not write labels, odds, DB rows, joins, registry state, model pointers, betting output, or EV output.",
        "",
        "## Summary",
        "",
        f"- Diagnostic races: `{summary['diagnostic_race_count']}`",
        f"- All absent from latest baseline shadow predictions: `{summary['all_diagnostic_races_absent_from_latest_shadow_predictions']}`",
        f"- All absent from latest Stage 2 predictions: `{summary['all_diagnostic_races_absent_from_latest_stage2_predictions']}`",
        f"- DB write performed: `{summary['db_write_performed']}`",
        f"- Join acceptance changed: `{summary['join_acceptance_changed']}`",
        "",
        "Coverage cause counts:",
    ]
    for cause, count in summary["coverage_cause_counts"].items():
        lines.append(f"- `{cause}`: `{count}`")

    lines.extend(
        [
            "",
            "## Diagnostic Items",
            "",
            "| Race | Odds rows | Boxes | Latest odds capture | Cause | Nearby latest shadow races |",
            "| --- | ---: | ---: | --- | --- | --- |",
        ]
    )
    for row in report["items"]:
        nearby = ", ".join(row["nearby_latest_shadow_races"]) or "none"
        lines.append(
            "| {race_id} | {rows} | {boxes} | {capture} | `{cause}` | {nearby} |".format(
                race_id=row["race_id"],
                rows=row.get("live_odds_row_count"),
                boxes=row.get("live_odds_box_count"),
                capture=row.get("latest_live_odds_capture"),
                cause=row["coverage_cause"],
                nearby=nearby,
            )
        )

    lines.extend(
        [
            "",
            "## Operator Decision",
            "",
            "The unresolved odds races are not join-ready from this artifact set because the exact race IDs are absent from the latest baseline and Stage 2 shadow prediction outputs. The safe next action is diagnostic review of shadow candidate coverage/source-refresh timing, not manual joining or backfill.",
            "",
        ]
    )
    return "\n".join(lines)


def output_manifest(output_dir: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        files[relpath(path) or str(path)] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "schema_version": "shadow_candidate_coverage_diagnostic_manifest_v1",
        "output_dir": relpath(output_dir),
        "files": files,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-queue", required=True, type=Path)
    parser.add_argument("--shadow-predictions-jsonl", required=True, type=Path)
    parser.add_argument("--stage2-shadow-predictions-jsonl", required=True, type=Path)
    parser.add_argument("--refreshed-upcoming-dir", required=True, type=Path)
    parser.add_argument("--shadow-candidate-source-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = unique_dir(assert_output_dir_safe(args.output_dir))
    candidate_source_report = load_json(args.shadow_candidate_source_report)
    report = build_diagnostic(
        recovery_queue=load_json(args.recovery_queue),
        shadow_prediction_race_ids=race_ids_from_jsonl(args.shadow_predictions_jsonl),
        stage2_prediction_race_ids=race_ids_from_jsonl(
            args.stage2_shadow_predictions_jsonl
        ),
        refreshed_upcoming_race_ids=race_ids_from_upcoming_dir(
            args.refreshed_upcoming_dir
        ),
        candidate_source_race_ids=set(
            str(race_id)
            for race_id in candidate_source_report.get("candidate_race_ids", [])
        ),
        generated_at=datetime.now().astimezone(),
        source_paths={
            "recovery_queue": args.recovery_queue,
            "shadow_predictions_jsonl": args.shadow_predictions_jsonl,
            "stage2_shadow_predictions_jsonl": args.stage2_shadow_predictions_jsonl,
            "refreshed_upcoming_dir": args.refreshed_upcoming_dir,
            "shadow_candidate_source_report": args.shadow_candidate_source_report,
        },
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "shadow_candidate_coverage_diagnostic.json", report)
    write_text(output_dir / "SUMMARY.md", render_markdown(report))
    write_json(output_dir / "manifest.json", output_manifest(output_dir))
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
