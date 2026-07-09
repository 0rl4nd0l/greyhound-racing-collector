#!/usr/bin/env python3
"""Build a no-write official-vs-DB gap review for one prioritized race.

This report-only helper is intended for the first manual-review race from the
rolling no-box failure queue. It compares existing official lookup evidence,
current DB rows, rolling predictions, and winner-only rows so a future approved
repair can be scoped exactly. It never fetches official pages, writes labels or
DB rows, regenerates datasets, trains models, updates registries, enables TGR,
or produces betting/EV actions.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/"
SCHEMA_VERSION = "single_race_official_gap_review_packet_v1"
STATUS_OK = "REPORT_ONLY_SINGLE_RACE_OFFICIAL_GAP_REVIEW"
STATUS_FAILURES = "REPORT_ONLY_SINGLE_RACE_OFFICIAL_GAP_REVIEW_WITH_FAILURES"
WRITES_PERFORMED = {
    "db_write": False,
    "label_write": False,
    "metadata_write": False,
    "official_fetch": False,
    "snapshot_mutation": False,
    "manifest_mutation": False,
    "dataset_regeneration": False,
    "model_training": False,
    "model_persistence": False,
    "registry_mutation": False,
    "promotion": False,
    "tgr_enablement": False,
    "betting_decision": False,
    "ev_action": False,
}
FORBIDDEN_WITHOUT_EXPLICIT_APPROVAL = [
    "write_official_safe_labels",
    "mutate_db",
    "metadata_write",
    "dog_row_insert",
    "dataset_regeneration",
    "model_training_or_promotion",
    "registry_update",
    "enable_tgr",
    "betting_or_ev_action",
]
RUNNER_REVIEW_FIELDS = [
    "race_id",
    "official_finish_position",
    "official_box_number",
    "official_dog_name",
    "name_key",
    "db_matched",
    "db_box_number",
    "db_finish_position",
    "db_placing",
    "db_scraped_finish_position",
    "db_data_source",
    "db_box_matches_official",
    "db_finish_matches_official",
    "prediction_present",
    "predicted_rank",
    "prediction_score",
    "actual_win",
    "winner_only_present",
    "gap_flags",
]
PREDICTION_FORBIDDEN_FIELDS = {
    "box_number",
    "official_box_number",
    "db_box_number",
    "finish_position",
    "official_finish_position",
    "db_finish_position",
    "placing",
    "scraped_finish_position",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_not_object:{path}")
    return payload


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"jsonl_row_not_object:{path}:{line_number}")
            rows.append(row)
    return rows


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _name_key(value: Any) -> str:
    text = re.sub(r"^\s*\d{1,2}\s*[\.\):-]\s*", "", str(value or "").strip())
    text = text.replace('"', "").replace("'", "").replace("`", "")
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).strip().lower()
    return re.sub(r"\s+", " ", text)


def _repo_output_path(path: Path, root: Path | None = None) -> tuple[Path, str]:
    root_path = (root or ROOT).expanduser().resolve(strict=False)
    logical = path.expanduser()
    if not logical.is_absolute():
        logical = root_path / logical
    resolved = logical.resolve(strict=False)
    try:
        relative = resolved.relative_to(root_path).as_posix()
    except ValueError as exc:
        raise ValueError(f"output_dir_must_be_inside_repo:{logical}") from exc
    return resolved, relative


def _repo_relative_text(path: Path, root: Path | None = None) -> str:
    return _repo_output_path(path, root)[1]


def _assert_output_dir_safe(output_dir: Path, root: Path | None = None) -> Path:
    resolved, relative = _repo_output_path(output_dir, root)
    if not relative.startswith(ALLOWED_OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_under_artifacts:{relative}")
    return resolved


def _connect_read_only(db_path: Path) -> sqlite3.Connection:
    resolved = db_path.expanduser().resolve()
    conn = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _validate_lookup_packet(
    *,
    path: Path,
    packet: Mapping[str, Any],
    failures: list[str],
) -> None:
    if packet.get("status") != "REPORT_ONLY":
        failures.append(f"lookup_packet_status_not_report_only:{path}")
    writes = _mapping(packet.get("writes_performed"))
    for key, value in writes.items():
        if key == "official_fetch":
            continue
        if value is not False:
            failures.append(f"lookup_packet_write_flag_true:{path}:{key}")


def _lookup_result_for_race(
    *,
    lookup_packet_paths: Sequence[Path],
    race_id: str,
    failures: list[str],
) -> dict[str, Any] | None:
    result = None
    for path in lookup_packet_paths:
        resolved = path.expanduser().resolve()
        packet = _load_json(resolved)
        _validate_lookup_packet(path=resolved, packet=packet, failures=failures)
        for row in _list(packet.get("results")):
            row_map = dict(_mapping(row))
            if row_map.get("legacy_race_id") == race_id:
                result = row_map
                result["_lookup_packet"] = str(resolved)
                break
        if result is not None:
            break
    return result


def _failure_review_row(
    *,
    failure_review_csv_path: Path | None,
    race_id: str,
) -> dict[str, Any] | None:
    if not failure_review_csv_path:
        return None
    for row in _load_csv_rows(failure_review_csv_path.expanduser().resolve()):
        if row.get("race_id") == race_id:
            return row
    return None


def _fetch_db_race(
    *,
    db_path: Path,
    race_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    with _connect_read_only(db_path) as conn:
        quick_check = conn.execute("PRAGMA quick_check").fetchone()
        metadata_rows = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM race_metadata WHERE race_id = ?",
                (race_id,),
            )
        ]
        dog_rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT dog_name, box_number, finish_position, placing,
                       scraped_finish_position, data_source
                FROM dog_race_data
                WHERE race_id = ?
                ORDER BY box_number, dog_name
                """,
                (race_id,),
            )
        ]
    db_state = {
        "db_path": str(db_path.expanduser().resolve()),
        "quick_check": quick_check[0] if quick_check else None,
        "read_only": True,
        "query_only": True,
        "race_metadata_row_count": len(metadata_rows),
        "dog_race_data_row_count": len(dog_rows),
    }
    return (metadata_rows[0] if metadata_rows else {}), dog_rows, db_state


def _rows_for_race(path: Path, race_id: str) -> list[dict[str, Any]]:
    return [row for row in _load_jsonl(path.expanduser().resolve()) if row.get("race_id") == race_id]


def _validate_no_write_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    failures: list[str],
) -> None:
    for index, row in enumerate(rows, start=1):
        for flag in (
            "box_features_allowed",
            "finish_order_labels_allowed",
            "top3_labels_allowed",
            "official_safe_label_candidate",
            "label_write_approved",
        ):
            if flag in row and row.get(flag) is not False:
                failures.append(f"row_flag_not_false:{path}:{index}:{flag}")


def _validate_prediction_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    failures: list[str],
) -> None:
    _validate_no_write_rows(rows=rows, path=path, failures=failures)
    for index, row in enumerate(rows, start=1):
        forbidden = sorted(set(row).intersection(PREDICTION_FORBIDDEN_FIELDS))
        if forbidden:
            failures.append(
                f"prediction_row_forbidden_fields:{path}:{index}:{','.join(forbidden)}"
            )


def _row_by_name(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {}
    for row in rows:
        key = _name_key(row.get("dog_name") or row.get("dog_name_key"))
        if key and key not in result:
            result[key] = row
    return result


def _official_rows(lookup_result: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not lookup_result:
        return []
    rows = _list(lookup_result.get("official_runner_rows")) or _list(lookup_result.get("positions"))
    return [
        {
            "box_number": _safe_int(_mapping(row).get("box_number")),
            "dog_name": _mapping(row).get("dog_name"),
            "finish_position": _safe_int(_mapping(row).get("finish_position")),
            "name_key": _name_key(_mapping(row).get("dog_name")),
        }
        for row in rows
    ]


def _runner_review_rows(
    *,
    race_id: str,
    official_rows: Sequence[Mapping[str, Any]],
    db_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    winner_only_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    db_by_name = _row_by_name(db_rows)
    prediction_by_name = _row_by_name(prediction_rows)
    winner_only_by_name = _row_by_name(winner_only_rows)
    rows = []
    for official in sorted(
        official_rows,
        key=lambda row: (_safe_int(row.get("finish_position")) or 999, _safe_int(row.get("box_number")) or 999),
    ):
        key = str(official.get("name_key") or "")
        db_row = _mapping(db_by_name.get(key))
        prediction = _mapping(prediction_by_name.get(key))
        winner_only = _mapping(winner_only_by_name.get(key))
        db_box = _safe_int(db_row.get("box_number"))
        db_finish = _safe_int(db_row.get("finish_position"))
        official_box = _safe_int(official.get("box_number"))
        official_finish = _safe_int(official.get("finish_position"))
        gap_flags = []
        if not db_row:
            gap_flags.append("missing_db_runner")
        elif db_box != official_box:
            gap_flags.append("db_box_differs_from_official")
        if db_row and db_finish != official_finish:
            gap_flags.append("db_finish_differs_from_official")
        if not prediction:
            gap_flags.append("missing_prediction_row")
        if not winner_only:
            gap_flags.append("missing_winner_only_row")
        rows.append(
            {
                "race_id": race_id,
                "official_finish_position": official_finish,
                "official_box_number": official_box,
                "official_dog_name": official.get("dog_name"),
                "name_key": key,
                "db_matched": bool(db_row),
                "db_box_number": db_box,
                "db_finish_position": db_finish,
                "db_placing": db_row.get("placing"),
                "db_scraped_finish_position": db_row.get("scraped_finish_position"),
                "db_data_source": db_row.get("data_source"),
                "db_box_matches_official": bool(db_row) and db_box == official_box,
                "db_finish_matches_official": bool(db_row) and db_finish == official_finish,
                "prediction_present": bool(prediction),
                "predicted_rank": _safe_int(prediction.get("predicted_rank")),
                "prediction_score": _safe_float(prediction.get("score")),
                "actual_win": _safe_int(prediction.get("actual_win")),
                "winner_only_present": bool(winner_only),
                "gap_flags": "|".join(gap_flags),
            }
        )
    return rows


def _feature_summary(prediction_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not prediction_rows:
        return {
            "prediction_rows": 0,
            "feature_join_status_counts": {},
            "history_feature_join_status_counts": {},
            "min_history_values_filled": None,
            "winner_history_values_filled": None,
        }
    winner = next((row for row in prediction_rows if _safe_int(row.get("actual_win")) == 1), {})
    return {
        "prediction_rows": len(prediction_rows),
        "feature_join_status_counts": dict(
            sorted(Counter(str(row.get("feature_join_status") or "DATA_MISSING") for row in prediction_rows).items())
        ),
        "history_feature_join_status_counts": dict(
            sorted(Counter(str(row.get("history_feature_join_status") or "DATA_MISSING") for row in prediction_rows).items())
        ),
        "min_history_values_filled": min(
            (_safe_int(row.get("history_feature_values_filled")) or 0)
            for row in prediction_rows
        ),
        "winner_history_values_filled": _safe_int(winner.get("history_feature_values_filled")),
        "winner_predicted_rank": _safe_int(winner.get("predicted_rank")),
        "winner_score": _safe_float(winner.get("score")),
    }


def build_gap_review_packet(
    *,
    race_id: str,
    lookup_packet_paths: Sequence[Path],
    db_path: Path,
    prediction_rows_path: Path,
    winner_only_rows_path: Path,
    failure_review_csv_path: Path | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    lookup_result = _lookup_result_for_race(
        lookup_packet_paths=lookup_packet_paths,
        race_id=race_id,
        failures=failures,
    )
    if lookup_result is None:
        failures.append(f"lookup_result_missing:{race_id}")
    db_metadata, db_rows, db_state = _fetch_db_race(db_path=db_path, race_id=race_id)
    if db_state.get("quick_check") != "ok":
        failures.append("db_quick_check_failed")
    prediction_rows = _rows_for_race(prediction_rows_path, race_id)
    winner_only_rows = _rows_for_race(winner_only_rows_path, race_id)
    _validate_prediction_rows(
        rows=prediction_rows,
        path=prediction_rows_path.expanduser().resolve(),
        failures=failures,
    )
    _validate_no_write_rows(
        rows=winner_only_rows,
        path=winner_only_rows_path.expanduser().resolve(),
        failures=failures,
    )
    official = _official_rows(lookup_result)
    runner_review_rows = _runner_review_rows(
        race_id=race_id,
        official_rows=official,
        db_rows=db_rows,
        prediction_rows=prediction_rows,
        winner_only_rows=winner_only_rows,
    )
    missing_db = [row for row in runner_review_rows if row["db_matched"] is not True]
    missing_prediction = [row for row in runner_review_rows if row["prediction_present"] is not True]
    missing_winner_only = [row for row in runner_review_rows if row["winner_only_present"] is not True]
    box_drift = [
        row
        for row in runner_review_rows
        if row["db_matched"] is True and row["db_box_matches_official"] is not True
    ]
    finish_drift = [
        row
        for row in runner_review_rows
        if row["db_matched"] is True and row["db_finish_matches_official"] is not True
    ]
    summary = {
        "race_id": race_id,
        "status_target": STATUS_OK,
        "lookup_status": lookup_result.get("lookup_status") if lookup_result else None,
        "result_parse_ready": lookup_result.get("result_parse_ready") if lookup_result else None,
        "label_write_ready": lookup_result.get("label_write_ready") if lookup_result else None,
        "lookup_skip_reasons": lookup_result.get("skip_reasons") if lookup_result else [],
        "official_runner_count": len(official),
        "db_runner_count": len(db_rows),
        "prediction_runner_count": len(prediction_rows),
        "winner_only_runner_count": len(winner_only_rows),
        "missing_db_runner_count": len(missing_db),
        "missing_prediction_runner_count": len(missing_prediction),
        "missing_winner_only_runner_count": len(missing_winner_only),
        "db_box_drift_count": len(box_drift),
        "db_finish_drift_count": len(finish_drift),
        "db_results_status": db_metadata.get("results_status"),
        "db_winner_name": db_metadata.get("winner_name"),
        "db_winner_source": db_metadata.get("winner_source"),
        "field_size": db_metadata.get("field_size"),
        "actual_field_size": db_metadata.get("actual_field_size"),
        "distance": db_metadata.get("distance"),
        "grade": db_metadata.get("grade"),
        "can_direct_label_write": False,
        "can_expand_training_without_approval": False,
        "recommended_next_action": (
            "manual_review_official_full_finish_order_and_prepare_exact_no_write_repair_plan"
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": STATUS_OK if not failures else STATUS_FAILURES,
        "failures": failures,
        "report_only": True,
        "write_ready": False,
        "label_write_approved": False,
        "label_writes_performed": False,
        "approval_required_before_label_write": True,
        "approval_required_before_db_write": True,
        "approval_required_before_dataset_regeneration": True,
        "model_training_performed": False,
        "model_promotion_allowed": False,
        "writes_performed": dict(WRITES_PERFORMED),
        "forbidden_without_explicit_approval": list(FORBIDDEN_WITHOUT_EXPLICIT_APPROVAL),
        "source_evidence": {
            "lookup_packets": [str(path.expanduser().resolve()) for path in lookup_packet_paths],
            "lookup_packet_matched": lookup_result.get("_lookup_packet") if lookup_result else None,
            "db": str(db_path.expanduser().resolve()),
            "prediction_rows": str(prediction_rows_path.expanduser().resolve()),
            "winner_only_rows": str(winner_only_rows_path.expanduser().resolve()),
            "failure_review_csv": (
                str(failure_review_csv_path.expanduser().resolve())
                if failure_review_csv_path
                else None
            ),
        },
        "failure_review_row": _failure_review_row(
            failure_review_csv_path=failure_review_csv_path,
            race_id=race_id,
        ),
        "db_state": db_state,
        "summary": summary,
        "feature_summary": _feature_summary(prediction_rows),
        "db_metadata": db_metadata,
        "official_rows": official,
        "db_rows": db_rows,
        "runner_review_rows": runner_review_rows,
    }


def write_outputs(
    output_dir: Path,
    packet: Mapping[str, Any],
    *,
    root: Path | None = None,
) -> None:
    output_dir = _assert_output_dir_safe(output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_payload = {key: value for key, value in packet.items() if key != "runner_review_rows"}
    (output_dir / "single_race_official_gap_review_packet.json").write_text(
        json.dumps(json_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "single_race_official_gap_runner_review.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=RUNNER_REVIEW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(packet.get("runner_review_rows") or [])
    summary = _mapping(packet.get("summary"))
    feature = _mapping(packet.get("feature_summary"))
    lines = [
        "# Single Race Official Gap Review",
        "",
        f"Status: `{packet.get('status')}`.",
        "",
        "No DB rows, labels, snapshots, manifests, datasets, models, registries, TGR settings, betting decisions, EV actions, or official fetches were changed or performed.",
        "",
        "## Summary",
        "",
        f"- Race: `{summary.get('race_id')}`",
        f"- Official runners: `{summary.get('official_runner_count')}`",
        f"- DB runners: `{summary.get('db_runner_count')}`",
        f"- Prediction runners: `{summary.get('prediction_runner_count')}`",
        f"- Missing DB runners: `{summary.get('missing_db_runner_count')}`",
        f"- Missing prediction runners: `{summary.get('missing_prediction_runner_count')}`",
        f"- DB box drift rows: `{summary.get('db_box_drift_count')}`",
        f"- DB finish drift rows: `{summary.get('db_finish_drift_count')}`",
        f"- DB winner source: `{summary.get('db_winner_source')}`",
        f"- Winner predicted rank: `{feature.get('winner_predicted_rank')}`",
        "",
        "## Next Safe Action",
        "",
        "Prepare an exact no-write repair plan for this race. Any DB/label write still requires explicit approval, an exact row allowlist, and a pre-op backup.",
        "",
    ]
    (output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--race-id", required=True)
    parser.add_argument("--lookup-packet", action="append", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--predictions-jsonl", required=True)
    parser.add_argument("--winner-only-rows-jsonl", required=True)
    parser.add_argument("--failure-review-csv")
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Iterable[str] | None = None, *, root: Path | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    packet = build_gap_review_packet(
        race_id=args.race_id,
        lookup_packet_paths=[Path(path) for path in args.lookup_packet],
        db_path=Path(args.db),
        prediction_rows_path=Path(args.predictions_jsonl),
        winner_only_rows_path=Path(args.winner_only_rows_jsonl),
        failure_review_csv_path=Path(args.failure_review_csv) if args.failure_review_csv else None,
    )
    write_outputs(Path(args.output_dir), packet, root=root)
    print(json.dumps({"status": packet["status"], "summary": packet["summary"]}, indent=2, sort_keys=True))
    return 1 if packet["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
