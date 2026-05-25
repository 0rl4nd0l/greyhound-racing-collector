#!/usr/bin/env python3
"""Evaluate frozen prediction snapshots against post-race labels.

This script is intentionally read-only for SQLite. It does not train, scrape,
or change model/ranking behavior. It scores only predictions that were already
frozen in snapshot JSON files.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accuracy_program.evaluation import (
    blend_probabilities,
    market_implied_probabilities,
    score_predictions,
)
from accuracy_program.snapshots import assert_no_result_fields
from utils.runner_completeness import (
    MIN_COMPLETE_RUNNERS,
    RunnerRow,
    analyze_prediction_runner_match,
    analyze_runner_rows,
)

DURABLE_SNAPSHOT_REQUIREMENTS = {
    "result_free",
    "pre_jump_lifecycle",
    "pre_jump_snapshot_state",
    "prediction_timestamp_present",
    "feature_freeze_timestamp_present",
    "source_file_path_present",
    "model_version_present",
    "runner_rows_present",
    "runner_rows_have_identity",
    "runner_rows_have_probabilities",
    "source_runner_set_complete",
    "predictions_match_source_runner_set",
    "priced_runners_have_odds_timestamps",
    "priced_runners_captured_before_prediction",
    "priced_runners_captured_before_jump",
    "missing_live_odds_explicit",
}


def _open_readonly(db_path: str | Path) -> sqlite3.Connection:
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _snapshot_files(paths: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob("**/*.json")))
        elif path.is_file():
            files.append(path)
    return files


def _norm_name(value: Any) -> str:
    import re

    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}
    except Exception:
        return set()


def _select_existing(conn: sqlite3.Connection, table: str, columns: list[str]) -> list[str]:
    present = _table_columns(conn, table)
    return [column for column in columns if column in present]


def _snapshot_readiness(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    predictions = [
        row
        for row in snapshot.get("predictions") or []
        if isinstance(row, Mapping)
    ]
    priced_rows = [
        row
        for row in predictions
        if isinstance(row.get("odds_snapshot"), Mapping)
        and row["odds_snapshot"].get("market_odds_win") is not None
    ]
    missing_odds_rows = [
        row
        for row in predictions
        if not (
            isinstance(row.get("odds_snapshot"), Mapping)
            and row["odds_snapshot"].get("market_odds_win") is not None
        )
    ]
    missing_live_odds_explicit = all(
        "missing_live_odds" in (row.get("data_quality_flags") or [])
        for row in missing_odds_rows
    )
    if predictions and not missing_odds_rows:
        missing_live_odds_explicit = True

    source_report = _snapshot_runner_report(snapshot)
    source_status = str(source_report.get("status") or "UNVERIFIED")
    runner_match = analyze_prediction_runner_match(predictions, source_report)

    requirements = {
        "result_free": True,
        "pre_jump_lifecycle": snapshot.get("lifecycle_status") == "upcoming_not_jumped",
        "pre_jump_snapshot_state": snapshot.get("snapshot_state")
        == "pre_jump_feature_freeze",
        "prediction_timestamp_present": bool(snapshot.get("prediction_timestamp")),
        "feature_freeze_timestamp_present": bool(snapshot.get("feature_freeze_timestamp")),
        "source_file_path_present": bool(snapshot.get("source_file_path")),
        "model_version_present": bool(snapshot.get("model_version")),
        "runner_rows_present": bool(predictions),
        "runner_rows_have_identity": all(
            row.get("dog_name") and row.get("box_number") is not None
            for row in predictions
        )
        if predictions
        else False,
        "runner_rows_have_probabilities": all(
            row.get("win_prob_norm") is not None for row in predictions
        )
        if predictions
        else False,
        "source_runner_set_complete": source_status == "COMPLETE",
        "predictions_match_source_runner_set": runner_match.get("status") == "MATCHED",
        "priced_runners_have_odds_timestamps": all(
            row["odds_snapshot"].get("odds_timestamp") for row in priced_rows
        ),
        "priced_runners_captured_before_prediction": all(
            row["odds_snapshot"].get("odds_captured_before_prediction") is True
            for row in priced_rows
        ),
        "priced_runners_captured_before_jump": all(
            row["odds_snapshot"].get("odds_captured_before_jump") is True
            for row in priced_rows
        ),
        "missing_live_odds_explicit": missing_live_odds_explicit,
    }
    failed = [
        requirement
        for requirement, passed in requirements.items()
        if not passed
    ]
    return {
        "status": "READY" if not failed else "NOT_READY",
        "requirements": requirements,
        "failed_requirements": failed,
        "counts": {
            "runner_count": len(predictions),
            "priced_runner_count": len(priced_rows),
            "missing_live_odds_count": len(missing_odds_rows),
        },
        "source_runner_completeness": source_report,
        "prediction_runner_match": runner_match,
    }


def _snapshot_runner_report(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    source_report = snapshot.get("source_runner_completeness")
    if isinstance(source_report, Mapping) and source_report:
        return dict(source_report)

    participants = []
    for participant in snapshot.get("frozen_participants") or []:
        if not isinstance(participant, Mapping):
            continue
        try:
            box_number = int(participant.get("box_number"))
        except (TypeError, ValueError):
            continue
        dog_name = str(participant.get("dog_name") or "").strip()
        if dog_name:
            participants.append(RunnerRow(box_number=box_number, dog_name=dog_name))

    if not participants:
        for row in snapshot.get("predictions") or []:
            if not isinstance(row, Mapping):
                continue
            try:
                box_number = int(row.get("box_number"))
            except (TypeError, ValueError):
                continue
            dog_name = str(row.get("dog_name") or row.get("dog_clean_name") or "").strip()
            if dog_name:
                participants.append(RunnerRow(box_number=box_number, dog_name=dog_name))

    return analyze_runner_rows(
        participants,
        source=f"snapshot:{snapshot.get('race_id')}",
        min_complete_runners=MIN_COMPLETE_RUNNERS,
    ).as_dict()


def _corpus_readiness_report(
    *,
    files_found: int,
    rejected_snapshots: list[dict[str, str]],
    readiness_status_counts: Counter[str],
    readiness_failures: list[dict[str, Any]],
) -> dict[str, Any]:
    if files_found == 0:
        return {
            "status": "DATA_MISSING",
            "reason": "no_frozen_pre_jump_snapshot_files_found",
            "durable_pre_jump_snapshot_requirements": sorted(
                DURABLE_SNAPSHOT_REQUIREMENTS
            ),
        }
    if rejected_snapshots or readiness_status_counts.get("NOT_READY", 0):
        return {
            "status": "NOT_READY",
            "snapshot_files": files_found,
            "readiness_status_counts": dict(readiness_status_counts),
            "rejected_snapshots": rejected_snapshots,
            "readiness_failures": readiness_failures[:25],
            "durable_pre_jump_snapshot_requirements": sorted(
                DURABLE_SNAPSHOT_REQUIREMENTS
            ),
        }
    return {
        "status": "READY",
        "snapshot_files": files_found,
        "readiness_status_counts": dict(readiness_status_counts),
        "durable_pre_jump_snapshot_requirements": sorted(
            DURABLE_SNAPSHOT_REQUIREMENTS
        ),
    }


def _race_metadata(conn: sqlite3.Connection, snapshot: Mapping[str, Any]) -> dict[str, Any]:
    columns = _select_existing(
        conn,
        "race_metadata",
        [
            "race_id",
            "venue",
            "race_number",
            "race_date",
            "distance",
            "results_status",
            "winner_source",
            "data_quality_note",
            "winner_name",
        ],
    )
    if not columns or "race_id" not in columns:
        return {}

    race_id = str(snapshot.get("race_id") or "")
    select_clause = ", ".join(columns)
    if race_id:
        row = conn.execute(
            f"SELECT {select_clause} FROM race_metadata WHERE race_id = ? LIMIT 1",
            (race_id,),
        ).fetchone()
        if row:
            return dict(row)

    race_date = snapshot.get("race_date")
    race_number = snapshot.get("race_number")
    if not race_date or race_number is None or "race_date" not in columns or "race_number" not in columns:
        return {}
    venue = snapshot.get("venue")
    if not venue or "venue" not in columns:
        return {}
    snapshot_race_id = str(snapshot.get("race_id") or "").strip()

    query = (
        f"SELECT {select_clause} FROM race_metadata "
        "WHERE race_date = ? AND CAST(race_number AS INTEGER) = ? "
        "AND (upper(replace(replace(venue, ' ', ''), '_', '')) = ? "
        "OR upper(replace(replace(race_id, ' ', ''), '_', '')) LIKE ?) "
        "ORDER BY CASE "
        "WHEN race_id = ? THEN 0 "
        "WHEN upper(replace(replace(venue, ' ', ''), '_', '')) = ? THEN 1 "
        "WHEN upper(replace(replace(race_id, ' ', ''), '_', '')) LIKE ? THEN 2 "
        "ELSE 3 END"
    )
    venue_norm = _norm_name(venue)
    params: list[Any] = [
        race_date,
        int(race_number),
        venue_norm,
        f"%{venue_norm}%",
        snapshot_race_id,
        venue_norm,
        f"%{venue_norm}%",
    ]
    query += " LIMIT 1"
    row = conn.execute(query, params).fetchone()
    return dict(row) if row else {}


def _labels_by_runner(conn: sqlite3.Connection, race_id: str) -> dict[str, dict[str, Any]]:
    columns = _select_existing(
        conn,
        "dog_race_data",
        [
            "dog_clean_name",
            "dog_name",
            "box_number",
            "finish_position",
            "placing",
            "scraped_finish_position",
            "data_source",
        ],
    )
    if not columns:
        return {}
    select_clause = ", ".join(columns)
    rows = conn.execute(
        f"SELECT {select_clause} FROM dog_race_data WHERE race_id = ?",
        (race_id,),
    ).fetchall()
    labels: dict[str, dict[str, Any]] = {}
    for row in rows:
        data = dict(row)
        pos = data.get("finish_position") or data.get("placing") or data.get("scraped_finish_position")
        try:
            finish_position = int(str(pos).strip())
        except Exception:
            continue
        if finish_position <= 0:
            continue
        actual_win = finish_position == 1
        label = {
            "actual_win": int(actual_win),
            "finish_position": finish_position,
            "label_source": data.get("data_source"),
            "result_detail_quality": "finish_position",
        }
        for key in (
            _norm_name(data.get("dog_clean_name") or data.get("dog_name")),
            f"box:{data.get('box_number')}",
        ):
            if key and key != "box:None":
                labels[key] = label
    return labels


def _winner_only_labels(
    metadata: Mapping[str, Any],
    predictions: list[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Build win/loss labels from race_metadata.winner_name only.

    This is intentionally narrower than full result labeling. It supports
    win-label metrics when fallback sources provide a reliable winner but not
    every runner's exact finish position.
    """

    winner_name = str(metadata.get("winner_name") or "").strip()
    winner_key = _norm_name(winner_name)
    if not winner_key:
        return {}, {"reason": "missing_winner_name"}

    matches = [
        runner
        for runner in predictions
        if _norm_name(runner.get("dog_name") or runner.get("dog_clean_name"))
        == winner_key
    ]
    if len(matches) != 1:
        return (
            {},
            {
                "reason": "winner_name_match_count_not_one",
                "winner_name": winner_name,
                "match_count": len(matches),
            },
        )

    source = (
        metadata.get("winner_source")
        or metadata.get("results_status")
        or "race_metadata_winner_name"
    )
    labels: dict[str, dict[str, Any]] = {}
    for runner in predictions:
        name = runner.get("dog_name") or runner.get("dog_clean_name")
        box = runner.get("box_number")
        actual_win = _norm_name(name) == winner_key
        label = {
            "actual_win": int(actual_win),
            "finish_position": 1 if actual_win else None,
            "label_source": source,
            "result_detail_quality": "winner_only",
        }
        for key in (_norm_name(name), f"box:{box}"):
            if key and key != "box:None":
                labels[key] = label

    return labels, {
        "winner_name": winner_name,
        "label_source": source,
        "result_detail_quality": "winner_only",
    }


def _runner_rows(snapshot: Mapping[str, Any], conn: sqlite3.Connection) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    snapshot_race_id = str(snapshot.get("race_id") or "")
    metadata = _race_metadata(conn, snapshot)
    if not metadata:
        return [], {
            "race_id": snapshot_race_id,
            "label_race_id": snapshot_race_id,
            "label_quality": "missing_race_metadata",
            "missing_reason": "missing_race_metadata",
        }

    runner_report = _snapshot_runner_report(snapshot)
    if runner_report.get("status") != "COMPLETE":
        return [], {
            "race_id": snapshot_race_id,
            "label_race_id": metadata.get("race_id") or snapshot_race_id,
            "label_quality": "incomplete_runner_set",
            "missing_reason": "incomplete_runner_set",
            "runner_completeness": runner_report,
        }

    label_race_id = str(metadata.get("race_id") or snapshot_race_id)
    labels = _labels_by_runner(conn, label_race_id)
    label_status = str(metadata.get("results_status") or "").strip().lower()
    source_note = str(metadata.get("winner_source") or metadata.get("data_quality_note") or "")
    if label_status == "partial_sportsbet_results" or "partial_sportsbet_results" in source_note:
        label_quality = "partial_sportsbet_results"
    elif metadata.get("winner_name") or label_status in {"complete", "completed", "resulted"}:
        label_quality = "official_or_complete_result"
    else:
        label_quality = "missing_result_label"

    predictions = [
        runner
        for runner in snapshot.get("predictions") or []
        if isinstance(runner, Mapping)
    ]

    missing_labels: list[dict[str, Any]] = []
    for runner in predictions:
        name = runner.get("dog_name") or runner.get("dog_clean_name")
        box = runner.get("box_number")
        label = labels.get(_norm_name(name)) or labels.get(f"box:{box}") or {}
        if "actual_win" not in label:
            missing_labels.append({"dog_name": str(name or ""), "box_number": box})

    winner_only_summary: dict[str, Any] | None = None
    if missing_labels and metadata.get("winner_name"):
        winner_only_labels, winner_only_summary = _winner_only_labels(metadata, predictions)
        if winner_only_labels:
            labels = winner_only_labels
            missing_labels = []
            if label_quality == "partial_sportsbet_results":
                label_quality = "partial_sportsbet_winner_only"
            else:
                label_quality = "winner_name_only_result"

    rows: list[dict[str, Any]] = []
    for runner in predictions:
        name = runner.get("dog_name") or runner.get("dog_clean_name")
        box = runner.get("box_number")
        label = labels.get(_norm_name(name)) or labels.get(f"box:{box}") or {}
        if "actual_win" not in label:
            continue
        odds_snapshot = runner.get("odds_snapshot") if isinstance(runner.get("odds_snapshot"), Mapping) else {}
        odds_win = _valid_pre_jump_odds(runner, odds_snapshot)
        rows.append(
            {
                "race_id": snapshot_race_id or label_race_id,
                "label_race_id": label_race_id,
                "dog_name": str(name or ""),
                "box_number": box,
                "race_date": metadata.get("race_date") or snapshot.get("race_date"),
                "venue": metadata.get("venue"),
                "distance": metadata.get("distance"),
                "win_prob_norm": _safe_float(runner.get("win_prob_norm")),
                "actual_win": int(label["actual_win"]),
                "finish_position": label.get("finish_position"),
                "odds_win": odds_win,
                "ev_win": (
                    _safe_float(runner.get("win_prob_norm")) * odds_win - 1.0
                    if odds_win is not None
                    and _safe_float(runner.get("win_prob_norm")) is not None
                    else None
                ),
                "label_quality": label_quality,
                "result_detail_quality": label.get("result_detail_quality"),
            }
        )
    if missing_labels:
        return [], {
            "race_id": snapshot_race_id,
            "label_race_id": label_race_id,
            "label_quality": "missing_dog_result_labels",
            "missing_reason": "missing_dog_result_labels",
            "missing_label_count": len(missing_labels),
            "missing_labels": missing_labels[:25],
            "winner_only_labeling": winner_only_summary,
        }
    if len(rows) != len(predictions):
        return [], {
            "race_id": snapshot_race_id,
            "label_race_id": label_race_id,
            "label_quality": "runner_label_count_mismatch",
            "missing_reason": f"runner_label_count_mismatch:{len(rows)}!={len(predictions)}",
        }
    winner_count = sum(int(row.get("actual_win") or 0) for row in rows)
    if winner_count != 1:
        return [], {
            "race_id": snapshot_race_id,
            "label_race_id": label_race_id,
            "label_quality": "invalid_winner_count",
            "missing_reason": f"invalid_winner_count:{winner_count}",
        }
    summary = {
        "race_id": snapshot_race_id,
        "label_race_id": label_race_id,
        "label_quality": label_quality,
    }
    if winner_only_summary:
        summary["winner_only_labeling"] = winner_only_summary
    return rows, summary


def _valid_pre_jump_odds(
    runner: Mapping[str, Any], odds_snapshot: Mapping[str, Any]
) -> float | None:
    odds = _safe_float(
        odds_snapshot.get("market_odds_win")
        or runner.get("odds")
        or runner.get("odds_win")
        or runner.get("market_odds_win")
    )
    if odds is None or odds <= 1.0:
        return None
    odds_timestamp = odds_snapshot.get("odds_timestamp") or runner.get("odds_timestamp")
    if not odds_timestamp:
        return None
    before_prediction = odds_snapshot.get("odds_captured_before_prediction")
    if before_prediction is not True:
        return None
    before_jump = odds_snapshot.get("odds_captured_before_jump")
    if before_jump is not True:
        return None
    provenance = odds_snapshot.get("odds_provenance")
    if not isinstance(provenance, Mapping) or not provenance.get("source"):
        return None
    return odds


def _ev_roi_coverage(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_race: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_race.setdefault(str(row.get("race_id")), []).append(row)

    complete_odds_races = []
    valid_rows: list[Mapping[str, Any]] = []
    partial_odds_races = []
    for race_id, race_rows in by_race.items():
        race_valid = [
            row
            for row in race_rows
            if row.get("odds_win") is not None and row.get("ev_win") is not None
        ]
        if len(race_valid) == len(race_rows) and race_rows:
            complete_odds_races.append(race_id)
            valid_rows.extend(race_valid)
        elif race_valid:
            partial_odds_races.append(race_id)

    if not complete_odds_races:
        return {
            "status": "DATA_MISSING",
            "reason": (
                "partial_pre_jump_dog_level_odds"
                if partial_odds_races
                else "no_valid_pre_jump_dog_level_odds"
            ),
            "valid_pre_jump_dog_odds_rows": 0,
            "missing_or_invalid_odds_rows": len(rows),
            "races_with_valid_pre_jump_odds": 0,
            "partial_odds_races": partial_odds_races,
        }
    return {
        "status": "SUCCESS",
        "valid_pre_jump_dog_odds_rows": len(valid_rows),
        "missing_or_invalid_odds_rows": len(rows) - len(valid_rows),
        "races_with_valid_pre_jump_odds": len(complete_odds_races),
        "partial_odds_races": partial_odds_races,
    }


def _arm_rows_from_market(rows: list[Mapping[str, Any]], arm: str) -> list[dict[str, Any]]:
    by_race: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_race.setdefault(str(row.get("race_id")), []).append(row)

    out: list[dict[str, Any]] = []
    for race_rows in by_race.values():
        odds = {
            str(row.get("dog_name")): float(row.get("odds_win"))
            for row in race_rows
            if row.get("odds_win") is not None
        }
        market = market_implied_probabilities(odds)
        if not market:
            continue
        if arm == "market_implied":
            probs = market
        else:
            model = {
                str(row.get("dog_name")): float(row.get("win_prob_norm"))
                for row in race_rows
                if row.get("win_prob_norm") is not None
            }
            probs = blend_probabilities(model, market, model_weight=0.5)
        for row in race_rows:
            name = str(row.get("dog_name"))
            if name not in probs:
                continue
            cloned = dict(row)
            cloned["win_prob_norm"] = probs[name]
            out.append(cloned)
    return out


def evaluate_snapshots(db_path: str, snapshot_paths: list[str]) -> dict[str, Any]:
    files = _snapshot_files(snapshot_paths)
    if not files:
        return {
            "status": "DATA_MISSING",
            "reason": "no_snapshot_files_found",
            "snapshot_corpus_readiness": _corpus_readiness_report(
                files_found=0,
                rejected_snapshots=[],
                readiness_status_counts=Counter(),
                readiness_failures=[],
            ),
            "metrics_by_arm": {},
        }

    rows: list[dict[str, Any]] = []
    lifecycle_counts: Counter[str] = Counter()
    label_quality_counts: Counter[str] = Counter()
    readiness_status_counts: Counter[str] = Counter()
    rejected_snapshots: list[dict[str, str]] = []
    readiness_failures: list[dict[str, Any]] = []

    with _open_readonly(db_path) as conn:
        for path in files:
            try:
                snapshot = json.loads(path.read_text(encoding="utf-8"))
                assert_no_result_fields(snapshot)
            except Exception as exc:
                rejected_snapshots.append({"path": str(path), "reason": str(exc)})
                continue
            readiness = _snapshot_readiness(snapshot)
            readiness_status_counts[str(readiness.get("status") or "unknown")] += 1
            if readiness.get("status") != "READY":
                readiness_failures.append(
                    {
                        "path": str(path),
                        "failed_requirements": readiness.get("failed_requirements", []),
                        "counts": readiness.get("counts", {}),
                    }
                )
                label_quality_counts["snapshot_not_ready"] += 1
                continue
            lifecycle_counts[str(snapshot.get("lifecycle_status") or "unknown")] += 1
            race_rows, summary = _runner_rows(snapshot, conn)
            rows.extend(race_rows)
            label_quality_counts[str(summary.get("label_quality") or "unknown")] += 1

    corpus_readiness = _corpus_readiness_report(
        files_found=len(files),
        rejected_snapshots=rejected_snapshots,
        readiness_status_counts=readiness_status_counts,
        readiness_failures=readiness_failures,
    )

    scorable_rows = [row for row in rows if row.get("win_prob_norm") is not None]
    if not scorable_rows:
        return {
            "status": "DATA_MISSING",
            "reason": "no_scorable_snapshot_rows_with_labels",
            "snapshot_files": len(files),
            "rejected_snapshots": rejected_snapshots,
            "snapshot_corpus_readiness": corpus_readiness,
            "lifecycle_counts": dict(lifecycle_counts),
            "label_quality_counts": dict(label_quality_counts),
            "metrics_by_arm": {},
        }

    metrics_by_arm = {"model_only": score_predictions(scorable_rows)}
    market_rows = _arm_rows_from_market(scorable_rows, "market_implied")
    if market_rows:
        metrics_by_arm["market_implied"] = score_predictions(market_rows)
    blend_rows = _arm_rows_from_market(scorable_rows, "simple_blend_50")
    if blend_rows:
        metrics_by_arm["simple_blend_50"] = score_predictions(blend_rows)

    return {
        "status": "SUCCESS",
        "snapshot_files": len(files),
        "snapshots_rejected": len(rejected_snapshots),
        "rejected_snapshots": rejected_snapshots,
        "snapshot_corpus_readiness": corpus_readiness,
        "runner_rows_scored": len(scorable_rows),
        "lifecycle_counts": dict(lifecycle_counts),
        "label_quality_counts": dict(label_quality_counts),
        "ev_roi_coverage": _ev_roi_coverage(scorable_rows),
        "metrics_by_arm": metrics_by_arm,
        "calibration_blending_decision": (
            "report_only: no calibration/blending deployment is made by this script"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="greyhound_racing_data.db")
    parser.add_argument(
        "--snapshots",
        nargs="+",
        required=True,
        help="Snapshot JSON files or directories",
    )
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    report = evaluate_snapshots(args.db, args.snapshots)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0 if report.get("status") == "SUCCESS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
