#!/usr/bin/env python3
"""Audit manifest-backed prediction snapshots for box bias and label readiness.

This script intentionally reads frozen snapshot artifacts and the manifest only.
It does not fetch results, write labels, rewrite snapshots, or mutate the model
registry. It is designed to replace stale top-level `predictions/*.json` as the
canonical box-bias evidence surface.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

DEFAULT_MANIFEST = Path("artifacts/prediction_snapshots/manifest.jsonl")
DEFAULT_OUTPUT = Path("artifacts/prediction_snapshot_audit/readiness_box_bias_audit.json")
READY_STATUS = "READY"
SNAPSHOT_SCHEMA = "prediction_snapshot_v1"
PRE_JUMP_LIFECYCLE = "upcoming_not_jumped"


def _parse_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except ValueError:
        return None


def _parse_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    raw = str(value).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_manifest_rows(manifest_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if not manifest_path.exists():
        return rows, [{"reason": "manifest_missing", "path": str(manifest_path)}]
    with manifest_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(
                    {
                        "reason": "manifest_line_unreadable",
                        "line_number": line_number,
                        "error": str(exc),
                    }
                )
                continue
            if isinstance(row, dict):
                row["_manifest_line_number"] = line_number
                rows.append(row)
            else:
                errors.append(
                    {
                        "reason": "manifest_line_not_object",
                        "line_number": line_number,
                    }
                )
    return rows, errors


def _resolve_snapshot_path(raw_path: Any, *, manifest_path: Path, repo_root: Path) -> Path:
    path = Path(str(raw_path or ""))
    if path.is_absolute():
        return path
    candidate = repo_root / path
    if candidate.exists():
        return candidate
    # Fallback for manifests stored inside a non-root artifact directory.
    return manifest_path.parent / path


def _load_snapshot(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as exc:  # noqa: BLE001 - diagnostic script records exact failure class.
        return None, f"snapshot_unreadable:{type(exc).__name__}"
    if not isinstance(data, dict):
        return None, "snapshot_not_object"
    return data, None


def _prediction_rows(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in ("predictions", "enhanced_predictions"):
        rows = snapshot.get(key)
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, Mapping)]
    nested = snapshot.get("prediction")
    if isinstance(nested, Mapping):
        for key in ("predictions", "enhanced_predictions"):
            rows = nested.get(key)
            if isinstance(rows, list):
                return [dict(row) for row in rows if isinstance(row, Mapping)]
    return []


def _probability(row: Mapping[str, Any]) -> float | None:
    for key in ("win_prob_norm", "win_prob", "win_probability", "final_score", "prediction_score"):
        parsed = _safe_float(row.get(key))
        if parsed is None:
            continue
        if parsed > 1.5:
            parsed /= 100.0
        return max(0.0, min(1.0, parsed))
    return None


def _top_pick(predictions: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for index, row in enumerate(predictions):
        probability = _probability(row)
        if probability is None:
            continue
        scored.append((probability, -index, row))
    if not scored:
        return None
    return max(scored, key=lambda item: (item[0], item[1]))[2]


def _snapshot_sort_key(record: Mapping[str, Any]) -> str:
    return str(
        record.get("feature_freeze_timestamp")
        or record.get("prediction_timestamp")
        or record.get("created_at")
        or ""
    )


def _race_key(snapshot: Mapping[str, Any], manifest_row: Mapping[str, Any]) -> str:
    return str(
        snapshot.get("race_id")
        or manifest_row.get("race_id")
        or snapshot.get("stable_race_key")
        or manifest_row.get("stable_race_key")
        or snapshot.get("source_file_path")
        or manifest_row.get("snapshot_path")
        or "UNKNOWN_RACE"
    )


def _skip_reasons(
    snapshot: Mapping[str, Any] | None,
    manifest_row: Mapping[str, Any],
    predictions: list[dict[str, Any]],
    *,
    min_runners: int,
) -> list[str]:
    if snapshot is None:
        return ["snapshot_missing_or_unreadable"]
    reasons: list[str] = []
    if snapshot.get("schema_version") != SNAPSHOT_SCHEMA:
        reasons.append("schema_not_prediction_snapshot_v1")
    readiness = snapshot.get("snapshot_readiness")
    readiness_status = readiness.get("status") if isinstance(readiness, Mapping) else None
    if readiness_status != READY_STATUS:
        reasons.append("snapshot_readiness_not_ready")
    if snapshot.get("lifecycle_status") != PRE_JUMP_LIFECYCLE:
        reasons.append("not_pre_jump_lifecycle")
    if snapshot.get("is_pre_jump_snapshot") is False:
        reasons.append("is_pre_jump_snapshot_false")
    if len(predictions) < min_runners:
        reasons.append("runner_count_below_minimum")
    if not predictions:
        reasons.append("runner_rows_missing")
    elif _top_pick(predictions) is None:
        reasons.append("top_pick_probability_missing")
    manifest_lifecycle = manifest_row.get("lifecycle_status")
    if manifest_lifecycle and manifest_lifecycle != PRE_JUMP_LIFECYCLE:
        reasons.append("manifest_not_pre_jump_lifecycle")
    return reasons


def _source_csv_available(snapshot: Mapping[str, Any]) -> bool:
    source = snapshot.get("source_file_path")
    return bool(source and Path(str(source)).exists())


def _summarize_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    top_box_counts: Counter[str] = Counter()
    runner_counts: Counter[str] = Counter()
    by_date: Counter[str] = Counter()
    by_venue: Counter[str] = Counter()
    probability_missing = 0
    total = 0
    for record in records:
        total += 1
        runner_counts[str(record.get("runner_count"))] += 1
        if record.get("race_date"):
            by_date[str(record["race_date"])] += 1
        if record.get("venue"):
            by_venue[str(record["venue"])] += 1
        top_box = record.get("top_pick_box")
        if top_box is None:
            probability_missing += 1
        else:
            top_box_counts[str(top_box)] += 1
    box1 = top_box_counts.get("1", 0)
    return {
        "record_count": total,
        "top_pick_box_distribution": dict(sorted(top_box_counts.items(), key=lambda item: item[0])),
        "box1_share": (box1 / total if total else None),
        "runner_count_distribution": dict(sorted(runner_counts.items(), key=lambda item: item[0])),
        "date_distribution": dict(sorted(by_date.items())),
        "venue_distribution": dict(sorted(by_venue.items())),
        "top_pick_probability_missing_count": probability_missing,
    }


def build_audit(
    *,
    manifest_path: Path,
    repo_root: Path,
    date_from: date | None = None,
    date_to: date | None = None,
    min_runners: int = 2,
    box1_max_share: float = 0.50,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    manifest_rows, manifest_errors = _load_manifest_rows(manifest_path)
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = list(manifest_errors)

    for manifest_row in manifest_rows:
        snapshot_path = _resolve_snapshot_path(
            manifest_row.get("snapshot_path"),
            manifest_path=manifest_path,
            repo_root=repo_root,
        )
        snapshot, load_error = _load_snapshot(snapshot_path)
        if load_error:
            skipped.append(
                {
                    "reason": load_error,
                    "snapshot_path": str(snapshot_path),
                    "manifest_line_number": manifest_row.get("_manifest_line_number"),
                }
            )
            continue
        assert snapshot is not None
        race_date = _parse_date(snapshot.get("race_date") or manifest_row.get("race_date"))
        if date_from and (race_date is None or race_date < date_from):
            continue
        if date_to and (race_date is None or race_date > date_to):
            continue

        predictions = _prediction_rows(snapshot)
        reasons = _skip_reasons(snapshot, manifest_row, predictions, min_runners=min_runners)
        top = _top_pick(predictions)
        top_box = _safe_int(top.get("box_number") or top.get("box")) if top else None
        top_probability = _probability(top) if top else None
        readiness = snapshot.get("snapshot_readiness") if isinstance(snapshot.get("snapshot_readiness"), Mapping) else {}
        source_csv_available = _source_csv_available(snapshot)
        jump_dt = _parse_datetime(snapshot.get("jump_datetime"))
        jumped_by_now = bool(jump_dt and jump_dt <= now)
        if not jump_dt and race_date:
            jumped_by_now = race_date <= now.date()
        ready = not reasons
        label_candidate_like = bool(ready and source_csv_available and jumped_by_now)
        records.append(
            {
                "race_id": _race_key(snapshot, manifest_row),
                "stable_race_key": snapshot.get("stable_race_key") or manifest_row.get("stable_race_key"),
                "snapshot_path": str(snapshot_path),
                "manifest_line_number": manifest_row.get("_manifest_line_number"),
                "race_date": race_date.isoformat() if race_date else None,
                "venue": snapshot.get("venue") or manifest_row.get("venue"),
                "race_number": snapshot.get("race_number") or manifest_row.get("race_number"),
                "schema_version": snapshot.get("schema_version"),
                "snapshot_readiness_status": readiness.get("status"),
                "lifecycle_status": snapshot.get("lifecycle_status"),
                "feature_freeze_timestamp": snapshot.get("feature_freeze_timestamp"),
                "prediction_timestamp": snapshot.get("prediction_timestamp"),
                "runner_count": len(predictions),
                "top_pick_box": top_box,
                "top_pick_probability": top_probability,
                "top_pick_dog_name": (top or {}).get("dog_name") or (top or {}).get("name"),
                "ready_for_box_bias_gate": ready,
                "skip_reasons": reasons,
                "source_csv_available": source_csv_available,
                "jumped_by_audit_time": jumped_by_now,
                "result_label_candidate_like": label_candidate_like,
            }
        )

    latest_by_race: dict[str, dict[str, Any]] = {}
    for record in records:
        key = str(record.get("race_id") or record.get("stable_race_key") or record.get("snapshot_path"))
        existing = latest_by_race.get(key)
        if existing is None or _snapshot_sort_key(record) >= _snapshot_sort_key(existing):
            latest_by_race[key] = record

    ready_records = [record for record in records if record.get("ready_for_box_bias_gate")]
    latest_records = list(latest_by_race.values())
    latest_ready_records = [record for record in latest_records if record.get("ready_for_box_bias_gate")]
    label_candidate_like_records = [record for record in latest_ready_records if record.get("result_label_candidate_like")]
    skip_counter: Counter[str] = Counter()
    for record in records:
        for reason in record.get("skip_reasons") or []:
            skip_counter[str(reason)] += 1

    latest_summary = _summarize_records(latest_ready_records)
    box1_share = latest_summary.get("box1_share")
    gate_status = "PASS" if box1_share is not None and box1_share <= box1_max_share else "FAIL"
    if not latest_ready_records:
        gate_status = "DATA_MISSING"

    return {
        "schema_version": "prediction_snapshot_readiness_box_bias_audit_v1",
        "generated_at": now.isoformat(),
        "manifest_path": str(manifest_path),
        "date_filter": {
            "date_from": date_from.isoformat() if date_from else None,
            "date_to": date_to.isoformat() if date_to else None,
        },
        "criteria": {
            "snapshot_schema": SNAPSHOT_SCHEMA,
            "snapshot_readiness_status": READY_STATUS,
            "lifecycle_status": PRE_JUMP_LIFECYCLE,
            "min_runners": min_runners,
            "box1_max_share": box1_max_share,
            "latest_snapshot_per_race": True,
        },
        "counts": {
            "manifest_rows": len(manifest_rows),
            "snapshot_records_loaded": len(records),
            "ready_snapshot_instances": len(ready_records),
            "latest_races": len(latest_records),
            "latest_ready_races": len(latest_ready_records),
            "latest_ready_result_label_candidate_like": len(label_candidate_like_records),
            "skipped_manifest_or_load_errors": len(skipped),
        },
        "gate": {
            "status": gate_status,
            "reason": (
                "box1_share_under_threshold"
                if gate_status == "PASS"
                else "box1_share_over_threshold"
                if gate_status == "FAIL"
                else "no_latest_ready_races"
            ),
            "box1_share": box1_share,
            "box1_max_share": box1_max_share,
            "evaluated_latest_ready_races": len(latest_ready_records),
        },
        "all_ready_snapshot_instances_summary": _summarize_records(ready_records),
        "latest_ready_races_summary": latest_summary,
        "latest_ready_result_label_candidate_like_summary": _summarize_records(label_candidate_like_records),
        "skip_reason_counts": dict(sorted(skip_counter.items())),
        "load_errors": skipped,
        "latest_ready_records": sorted(
            latest_ready_records,
            key=lambda row: (str(row.get("race_date") or ""), str(row.get("venue") or ""), str(row.get("race_number") or ""), str(row.get("race_id") or "")),
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit manifest-backed ready pre-jump prediction snapshots for box bias and label-readiness."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to prediction snapshot manifest JSONL")
    parser.add_argument("--repo-root", default=".", help="Repository root used to resolve relative snapshot paths")
    parser.add_argument("--date-from", help="Inclusive race-date lower bound YYYY-MM-DD")
    parser.add_argument("--date-to", help="Inclusive race-date upper bound YYYY-MM-DD")
    parser.add_argument("--min-runners", type=int, default=2, help="Minimum runner rows required for box-bias gate")
    parser.add_argument("--box1-max-share", type=float, default=0.50, help="Maximum allowed box-1 favourite share")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSON report path")
    parser.add_argument("--stdout", action="store_true", help="Also print the JSON report to stdout")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest_path = Path(args.manifest)
    repo_root = Path(args.repo_root)
    date_from = _parse_date(args.date_from)
    date_to = _parse_date(args.date_to)
    if args.date_from and date_from is None:
        raise SystemExit("invalid --date-from; expected YYYY-MM-DD")
    if args.date_to and date_to is None:
        raise SystemExit("invalid --date-to; expected YYYY-MM-DD")
    if args.min_runners < 1:
        raise SystemExit("--min-runners must be >= 1")
    if not 0 <= args.box1_max_share <= 1:
        raise SystemExit("--box1-max-share must be between 0 and 1")

    report = build_audit(
        manifest_path=manifest_path,
        repo_root=repo_root,
        date_from=date_from,
        date_to=date_to,
        min_runners=args.min_runners,
        box1_max_share=args.box1_max_share,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.stdout:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        gate = report["gate"]
        print(
            " ".join(
                [
                    f"status={gate['status']}",
                    f"box1_share={gate['box1_share']}",
                    f"latest_ready_races={gate['evaluated_latest_ready_races']}",
                    f"output={output_path}",
                ]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
