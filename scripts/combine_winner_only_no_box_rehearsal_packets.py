#!/usr/bin/env python3
"""Combine report-only no-box actual-win rehearsal packets.

This helper keeps the window-expansion workflow mechanical: it merges already
materialized winner-only/no-box rehearsal rows and refuses unsafe source packets.
It does not fetch official data, write labels, mutate databases, train models,
promote models, enable TGR, or create betting/EV actions.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_winner_only_no_box_rehearsal_packet import (
    FORBIDDEN_ROW_FIELDS,
    FORBIDDEN_WITHOUT_EXPLICIT_APPROVAL,
    ROWS_SCHEMA_VERSION,
    SCHEMA_VERSION,
    WRITES_PERFORMED,
    _sample_size_gate,
    write_rehearsal_outputs,
)


PACKET_FILE = "winner_only_no_box_actual_win_rehearsal_packet.json"
ROWS_FILE = "winner_only_no_box_actual_win_rows.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_not_object:{path}")
    return payload


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


def _packet_path(source: Path) -> Path:
    if source.is_dir():
        return source / PACKET_FILE
    return source


def _rows_path(packet_path: Path) -> Path:
    return packet_path.parent / ROWS_FILE


def _load_source(source: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    packet_path = _packet_path(source).expanduser().resolve()
    rows_path = _rows_path(packet_path)
    if not packet_path.exists():
        raise FileNotFoundError(packet_path)
    if not rows_path.exists():
        raise FileNotFoundError(rows_path)
    return packet_path, _load_json(packet_path), _load_jsonl(rows_path)


def _writes_all_false(packet: Mapping[str, Any]) -> bool:
    writes = _mapping(packet.get("writes_performed"))
    return bool(writes) and all(value is False for value in writes.values())


def _validate_source_packet(
    *,
    packet_path: Path,
    packet: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    failures: list[str] = []
    if packet.get("schema_version") != SCHEMA_VERSION:
        failures.append(f"schema_version_mismatch:{packet_path}")
    if packet.get("report_only") is not True:
        failures.append(f"source_not_report_only:{packet_path}")
    if packet.get("write_ready") is not False:
        failures.append(f"source_write_ready_not_false:{packet_path}")
    if packet.get("label_write_approved") is not False:
        failures.append(f"source_label_write_approved_not_false:{packet_path}")
    if packet.get("model_training_performed") is not False:
        failures.append(f"source_model_training_performed_not_false:{packet_path}")
    if not _writes_all_false(packet):
        failures.append(f"source_writes_not_all_false:{packet_path}")
    if _list(packet.get("failures")):
        failures.append(f"source_has_failures:{packet_path}")

    row_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        race_id = str(row.get("race_id") or "").strip()
        if not race_id:
            failures.append(f"row_missing_race_id:{packet_path}:{index}")
            continue
        row_groups[race_id].append(row)
        forbidden = sorted(set(row).intersection(FORBIDDEN_ROW_FIELDS))
        if forbidden:
            failures.append(f"row_forbidden_fields:{packet_path}:{index}:{','.join(forbidden)}")
        for flag in (
            "box_features_allowed",
            "finish_order_labels_allowed",
            "top3_labels_allowed",
            "official_safe_label_candidate",
            "label_write_approved",
        ):
            if row.get(flag) is not False:
                failures.append(f"row_flag_not_false:{packet_path}:{index}:{flag}")

    for race_id, race_rows in row_groups.items():
        positive_count = sum(1 for row in race_rows if int(row.get("actual_win") or 0) == 1)
        if positive_count != 1:
            failures.append(f"race_positive_count_not_one:{packet_path}:{race_id}:{positive_count}")
    return failures


def _race_kind_counts(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    by_race: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        race_id = str(row.get("race_id") or "")
        by_race.setdefault(race_id, row)
    counts: Counter[str] = Counter()
    for row in by_race.values():
        if row.get("field_complete_for_ranking") is True:
            counts["complete_field"] += 1
        else:
            counts["partial_field"] += 1
    return counts


def combine_rehearsal_packets(
    sources: Sequence[Path],
    *,
    expected_races: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    failures: list[str] = []
    all_rows: list[dict[str, Any]] = []
    source_packets = []
    race_summaries = []
    blocked_records = []
    recovery_records = []
    blocked_or_non_candidate_records_seen = 0
    p1_recovery_candidates = 0

    seen_races: set[str] = set()
    for source in sources:
        packet_path, packet, rows = _load_source(source)
        source_packets.append(str(packet_path))
        failures.extend(_validate_source_packet(packet_path=packet_path, packet=packet, rows=rows))

        source_races = {str(row.get("race_id") or "").strip() for row in rows if row.get("race_id")}
        duplicate_races = sorted(seen_races.intersection(source_races))
        for race_id in duplicate_races:
            failures.append(f"duplicate_race_id_across_sources:{race_id}")
        seen_races.update(source_races)

        all_rows.extend(dict(row) for row in rows)
        race_summaries.extend(_list(packet.get("race_summaries")))
        blocked_records.extend(_list(packet.get("blocked_or_non_candidate_records")))
        summary = _mapping(packet.get("summary"))
        blocked_or_non_candidate_records_seen += int(
            summary.get("blocked_or_non_candidate_records_seen") or 0
        )
        p1_recovery_candidates += int(summary.get("p1_recovery_candidates") or 0)
        recovery_queue = _mapping(packet.get("recovery_queue"))
        recovery_records.extend(_list(recovery_queue.get("records")))

    if expected_races is not None and len(seen_races) != expected_races:
        failures.append(f"race_count_mismatch:expected_{expected_races}:actual_{len(seen_races)}")

    race_kind_counts = _race_kind_counts(all_rows)
    actual_win_candidate_count = len(seen_races)
    ranking_ready_candidate_count = race_kind_counts.get("complete_field", 0)
    sample_gate = _sample_size_gate(actual_win_candidate_count, ranking_ready_candidate_count)
    row_status_counts = Counter(
        str(summary.get("status") or "DATA_MISSING") for summary in race_summaries
    )
    actual_win_positive_rows = sum(1 for row in all_rows if int(row.get("actual_win") or 0) == 1)
    actual_win_negative_rows = sum(1 for row in all_rows if int(row.get("actual_win") or 0) == 0)
    recovery_lane_counts = Counter(
        str(record.get("recovery_lane") or "DATA_MISSING")
        for record in recovery_records
        if isinstance(record, Mapping)
    )
    priority_counts = Counter(
        str(record.get("priority") or "DATA_MISSING")
        for record in recovery_records
        if isinstance(record, Mapping)
    )

    if failures:
        status = "REPORT_ONLY_COMBINED_WITH_FAILURES"
    elif sample_gate["minimum_smoke_actual_win_eval"]["status"] == "PASS":
        status = "REPORT_ONLY_READY_FOR_NO_BOX_ACTUAL_WIN_EVALUATION"
    else:
        status = "REPORT_ONLY_COMBINED_REHEARSAL_INSUFFICIENT_CONFIRMED_RACES"

    if sample_gate["minimum_rolling_temporal_eval"]["status"] != "PASS":
        next_gate = "collect_more_metadata_confirmed_winner_only_candidates_before_rolling_eval"
    elif sample_gate["minimum_ranking_model_comparison"]["status"] != "PASS":
        next_gate = "collect_more_complete_field_candidates_before_ranking_comparison"
    else:
        next_gate = "run_report_only_no_box_actual_win_eval_and_ranking_comparison"

    recovery_queue = {
        "schema_version": "winner_only_no_box_recovery_queue_v1",
        "generated_at": utc_now(),
        "report_only": True,
        "write_ready": False,
        "label_write_approved": False,
        "model_training_performed": False,
        "writes_performed": dict(WRITES_PERFORMED),
        "summary": {
            "records_reviewed": len(recovery_records),
            "recovery_queue_count": len(recovery_records),
            "recovery_lane_counts": dict(sorted(recovery_lane_counts.items())),
            "priority_counts": dict(sorted(priority_counts.items())),
            "p1_name_set_and_parser_repair_candidates": p1_recovery_candidates,
            "next_recommended_gate": next_gate,
        },
        "forbidden_without_explicit_approval": list(FORBIDDEN_WITHOUT_EXPLICIT_APPROVAL),
        "records": recovery_records,
    }

    packet = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": status,
        "failures": failures,
        "report_only": True,
        "write_ready": False,
        "label_write_approved": False,
        "model_training_performed": False,
        "model_promotion_allowed": False,
        "writes_performed": dict(WRITES_PERFORMED),
        "source_packets": {
            "combined_packet_count": len(source_packets),
            "rehearsal_packets": source_packets,
        },
        "scope": {
            "allowed_use": "combined_no_box_actual_win_data_contract_rehearsal",
            "not_allowed_use": [
                "official_safe_label_write",
                "strict_full_finish_training",
                "top3_or_finish_order_training",
                "box_feature_training",
                "model_promotion",
                "betting_or_ev_action",
            ],
            "row_schema_version": ROWS_SCHEMA_VERSION,
            "forbidden_row_fields": sorted(FORBIDDEN_ROW_FIELDS),
        },
        "summary": {
            "confirmed_winner_only_candidate_count": actual_win_candidate_count,
            "complete_field_confirmed_winner_only_candidate_count": race_kind_counts.get(
                "complete_field",
                0,
            ),
            "partial_field_confirmed_winner_only_candidate_count": race_kind_counts.get(
                "partial_field",
                0,
            ),
            "race_grouped_ranking_ready_candidate_count": ranking_ready_candidate_count,
            "blocked_or_non_candidate_records_seen": blocked_or_non_candidate_records_seen,
            "materialized_races": actual_win_candidate_count,
            "materialized_rows": len(all_rows),
            "actual_win_positive_rows": actual_win_positive_rows,
            "actual_win_negative_rows": actual_win_negative_rows,
            "race_status_counts": dict(sorted(row_status_counts.items())),
            "no_box_row_policy_pass": not any(
                set(row).intersection(FORBIDDEN_ROW_FIELDS) for row in all_rows
            ),
            "strict_full_finish_label_candidate_count": 0,
            "official_safe_label_candidate_count": 0,
            "can_evaluate_model": sample_gate["minimum_smoke_actual_win_eval"]["status"] == "PASS",
            "sample_size_gate": sample_gate,
            "recovery_queue_count": len(recovery_records),
            "p1_recovery_candidates": p1_recovery_candidates,
            "next_recommended_gate": next_gate,
            "can_evaluate_race_grouped_model": (
                sample_gate["minimum_ranking_model_comparison"]["status"] == "PASS"
            ),
        },
        "forbidden_without_explicit_approval": list(FORBIDDEN_WITHOUT_EXPLICIT_APPROVAL),
        "race_summaries": race_summaries,
        "blocked_or_non_candidate_records": blocked_records[:25],
        "recovery_queue": recovery_queue,
    }
    return packet, all_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", action="append", required=True, help="Packet file or packet directory.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-races", type=int)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    packet, rows = combine_rehearsal_packets(
        [Path(value) for value in args.packet],
        expected_races=args.expected_races,
    )
    write_rehearsal_outputs(Path(args.output_dir), packet, rows)
    print(json.dumps(packet["summary"], indent=2, sort_keys=True))
    return 1 if packet["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
