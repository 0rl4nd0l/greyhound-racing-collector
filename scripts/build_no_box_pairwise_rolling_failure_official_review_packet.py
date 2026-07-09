#!/usr/bin/env python3
"""Prioritize official review for no-box rolling pairwise failures.

This report-only helper connects rolling Top1/Top3 misses to existing official
lookup dry-run packets and winner-only no-box rehearsal rows. It does not fetch
official pages, write labels, mutate DB rows, regenerate datasets, train or
persist models, update registries, enable TGR, or produce betting/EV actions.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/"
SCHEMA_VERSION = "no_box_pairwise_rolling_failure_official_review_packet_v1"
STATUS_OK = "REPORT_ONLY_ROLLING_FAILURE_OFFICIAL_REVIEW_PACKET"
STATUS_FAILURES = "REPORT_ONLY_ROLLING_FAILURE_OFFICIAL_REVIEW_PACKET_WITH_FAILURES"
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
    "regenerate_canonical_dataset",
    "train_or_promote_model",
    "update_registry",
    "enable_tgr",
    "betting_or_ev_action",
]
CSV_FIELDS = [
    "priority",
    "review_lane",
    "race_id",
    "identity_key",
    "race_date",
    "venue",
    "race_number",
    "winner_rank",
    "top1_hit",
    "top3_hit",
    "field_size",
    "field_scope",
    "distance_bucket",
    "winner_box_bucket",
    "source_gap_status",
    "best_queue_policy_key",
    "best_queue_key",
    "lookup_status",
    "result_parse_ready",
    "label_write_ready",
    "skip_reasons",
    "positions_count",
    "terminal_status_count",
    "source_url",
    "winner_only_materialized",
    "winner_only_row_count",
    "recommended_next_action",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_not_object:{path}")
    return payload


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y"}


def _safe_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


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


def _identity_key(row: Mapping[str, Any]) -> str:
    return str(row.get("identity_key") or "").strip()


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


def _lookup_index(
    lookup_packet_paths: Sequence[Path],
    *,
    failures: list[str],
) -> tuple[dict[str, Mapping[str, Any]], Counter[str]]:
    index: dict[str, Mapping[str, Any]] = {}
    status_counts: Counter[str] = Counter()
    for path in lookup_packet_paths:
        resolved = path.expanduser().resolve()
        packet = _load_json(resolved)
        _validate_lookup_packet(path=resolved, packet=packet, failures=failures)
        for result in _list(packet.get("results")):
            result_map = _mapping(result)
            race_id = str(result_map.get("legacy_race_id") or "")
            if not race_id:
                continue
            status_counts[str(result_map.get("lookup_status") or "DATA_MISSING")] += 1
            index.setdefault(race_id, result_map)
    return index, status_counts


def _validate_winner_only_rows(
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
                failures.append(f"winner_only_row_flag_not_false:{path}:{index}:{flag}")


def _winner_only_index(
    paths: Sequence[Path],
    *,
    failures: list[str],
) -> dict[str, list[Mapping[str, Any]]]:
    index: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for path in paths:
        resolved = path.expanduser().resolve()
        rows = _load_jsonl(resolved)
        _validate_winner_only_rows(rows=rows, path=resolved, failures=failures)
        for row in rows:
            race_id = str(row.get("race_id") or row.get("legacy_race_id") or "")
            if race_id:
                index[race_id].append(row)
    return index


def _skip_reasons(result: Mapping[str, Any]) -> list[str]:
    return [str(item) for item in _list(result.get("skip_reasons")) if str(item)]


def _review_lane(
    *,
    crosswalk: Mapping[str, Any],
    lookup: Mapping[str, Any] | None,
    winner_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, int, str]:
    top1_hit = _as_bool(crosswalk.get("top1_hit"))
    top3_hit = _as_bool(crosswalk.get("top3_hit"))
    parsed = bool(lookup and lookup.get("result_parse_ready") is True)
    terminal_count = len(_list(lookup.get("terminal_statuses") if lookup else []))
    winner_only = bool(winner_rows)
    if not top3_hit and parsed:
        return (
            "P0_TOP3_MISS_PARSED_OFFICIAL_REVIEW",
            0,
            "manual_review_full_finish_order_then_expand_winner_only_or_repair_source_metadata",
        )
    if not top1_hit and parsed and terminal_count == 0 and winner_only:
        return (
            "P1_TOP1_MISS_PARSED_WINNER_ONLY_READY_REVIEW",
            1,
            "review_existing_winner_only_no_box_rows_and_source_metadata_before_any_label_write",
        )
    if not top1_hit and parsed:
        return (
            "P2_TOP1_MISS_PARSED_OFFICIAL_REVIEW",
            2,
            "manual_review_official_positions_and_field_scope_before_any_label_write",
        )
    if not top1_hit:
        return (
            "P3_TOP1_MISS_LOOKUP_BLOCKED_REVIEW",
            3,
            "resolve_lookup_slug_or_parser_blocker_before_label_expansion",
        )
    if parsed and winner_only:
        return (
            "P4_HIT_PARSED_WINNER_ONLY_BACKFILL_REVIEW",
            4,
            "lower_priority_source_metadata_review_after_failure_rows",
        )
    if parsed:
        return (
            "P5_HIT_PARSED_SOURCE_BACKFILL_REVIEW",
            5,
            "lower_priority_source_bucket_repair_review_after_failure_rows",
        )
    return (
        "P6_HIT_LOOKUP_BLOCKED_BACKLOG",
        6,
        "defer_until_failure_rows_and_parsed_source_gaps_are_resolved",
    )


def _json_cell(value: Any) -> str:
    if value in (None, ""):
        return ""
    return json.dumps(value, sort_keys=True)


def _review_rows(
    *,
    crosswalk_rows: Sequence[Mapping[str, Any]],
    lookup_by_race: Mapping[str, Mapping[str, Any]],
    winner_rows_by_race: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for crosswalk in crosswalk_rows:
        race_id = str(crosswalk.get("race_id") or "")
        lookup = lookup_by_race.get(race_id)
        winner_rows = list(winner_rows_by_race.get(race_id, []))
        lane, priority, action = _review_lane(
            crosswalk=crosswalk,
            lookup=lookup,
            winner_rows=winner_rows,
        )
        skip_reasons = _skip_reasons(lookup or {})
        rows.append(
            {
                "priority": priority,
                "review_lane": lane,
                "race_id": race_id,
                "identity_key": _identity_key(crosswalk),
                "race_date": crosswalk.get("race_date"),
                "venue": crosswalk.get("venue"),
                "race_number": crosswalk.get("race_number"),
                "winner_rank": crosswalk.get("winner_rank"),
                "top1_hit": _as_bool(crosswalk.get("top1_hit")),
                "top3_hit": _as_bool(crosswalk.get("top3_hit")),
                "field_size": crosswalk.get("field_size"),
                "field_scope": crosswalk.get("field_scope"),
                "distance_bucket": crosswalk.get("distance_bucket"),
                "winner_box_bucket": crosswalk.get("winner_box_bucket"),
                "source_gap_status": crosswalk.get("source_gap_status"),
                "best_queue_policy_key": crosswalk.get("best_queue_policy_key"),
                "best_queue_key": crosswalk.get("best_queue_key"),
                "lookup_status": lookup.get("lookup_status") if lookup else "DATA_MISSING",
                "result_parse_ready": bool(lookup and lookup.get("result_parse_ready") is True),
                "label_write_ready": bool(lookup and lookup.get("label_write_ready") is True),
                "skip_reasons": "|".join(skip_reasons),
                "positions_count": len(_list(lookup.get("positions") if lookup else [])),
                "terminal_status_count": len(_list(lookup.get("terminal_statuses") if lookup else [])),
                "source_url": lookup.get("source_url") if lookup else "",
                "winner_only_materialized": bool(winner_rows),
                "winner_only_row_count": len(winner_rows),
                "recommended_next_action": action,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            int(row["priority"]) if row.get("priority") is not None else 99,
            str(row.get("race_date") or ""),
            str(row.get("race_id") or ""),
        ),
    )


def build_review_packet(
    *,
    crosswalk_csv_path: Path,
    lookup_packet_paths: Sequence[Path],
    winner_only_rows_paths: Sequence[Path],
) -> dict[str, Any]:
    failures: list[str] = []
    crosswalk_resolved = crosswalk_csv_path.expanduser().resolve()
    crosswalk_rows = _load_csv_rows(crosswalk_resolved)
    lookup_by_race, lookup_status_counts = _lookup_index(
        lookup_packet_paths,
        failures=failures,
    )
    winner_rows_by_race = _winner_only_index(
        winner_only_rows_paths,
        failures=failures,
    )
    rows = _review_rows(
        crosswalk_rows=crosswalk_rows,
        lookup_by_race=lookup_by_race,
        winner_rows_by_race=winner_rows_by_race,
    )
    lane_counts: Counter[str] = Counter(str(row.get("review_lane")) for row in rows)
    top1_miss_rows = [row for row in rows if row.get("top1_hit") is not True]
    top3_miss_rows = [row for row in rows if row.get("top3_hit") is not True]
    parsed_rows = [row for row in rows if row.get("result_parse_ready") is True]
    winner_only_rows = [row for row in rows if row.get("winner_only_materialized") is True]
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
            "crosswalk_csv": str(crosswalk_resolved),
            "lookup_packets": [str(path.expanduser().resolve()) for path in lookup_packet_paths],
            "winner_only_rows": [str(path.expanduser().resolve()) for path in winner_only_rows_paths],
        },
        "summary": {
            "rolling_race_count": len(rows),
            "top1_miss_count": len(top1_miss_rows),
            "top3_miss_count": len(top3_miss_rows),
            "lookup_match_count": sum(1 for row in rows if row.get("lookup_status") != "DATA_MISSING"),
            "result_parse_ready_count": len(parsed_rows),
            "label_write_ready_count": sum(1 for row in rows if row.get("label_write_ready") is True),
            "winner_only_materialized_race_count": len(winner_only_rows),
            "top1_miss_result_parse_ready_count": sum(
                1 for row in top1_miss_rows if row.get("result_parse_ready") is True
            ),
            "top1_miss_winner_only_materialized_count": sum(
                1 for row in top1_miss_rows if row.get("winner_only_materialized") is True
            ),
            "top3_miss_result_parse_ready_count": sum(
                1 for row in top3_miss_rows if row.get("result_parse_ready") is True
            ),
            "review_lane_counts": dict(sorted(lane_counts.items())),
            "lookup_status_counts_seen": dict(sorted(lookup_status_counts.items())),
            "next_review_queue": rows[0]["review_lane"] if rows else None,
            "next_review_race_id": rows[0]["race_id"] if rows else None,
        },
        "review_rows": rows,
    }


def write_outputs(
    output_dir: Path,
    packet: Mapping[str, Any],
    *,
    root: Path | None = None,
) -> None:
    output_dir = _assert_output_dir_safe(output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_payload = {key: value for key, value in packet.items() if key != "review_rows"}
    (output_dir / "no_box_pairwise_rolling_failure_official_review_packet.json").write_text(
        json.dumps(json_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "rolling_failure_official_review_queue.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(packet.get("review_rows") or [])
    summary = _mapping(packet.get("summary"))
    lines = [
        "# No-Box Pairwise Rolling Failure Official Review Packet",
        "",
        f"Status: `{packet.get('status')}`.",
        "",
        "No DB rows, labels, snapshots, manifests, datasets, models, registries, TGR settings, betting decisions, EV actions, or official fetches were changed or performed.",
        "",
        "## Summary",
        "",
        f"- Rolling races: `{summary.get('rolling_race_count')}`",
        f"- Top1 misses: `{summary.get('top1_miss_count')}`",
        f"- Top3 misses: `{summary.get('top3_miss_count')}`",
        f"- Lookup matches: `{summary.get('lookup_match_count')}`",
        f"- Parsed official results: `{summary.get('result_parse_ready_count')}`",
        f"- Direct label-write-ready rows: `{summary.get('label_write_ready_count')}`",
        f"- Winner-only materialized races: `{summary.get('winner_only_materialized_race_count')}`",
        f"- Review lane counts: `{summary.get('review_lane_counts')}`",
        "",
        "## Next Safe Action",
        "",
        f"Start with `{summary.get('next_review_queue')}` for `{summary.get('next_review_race_id')}`. Any label or DB write still requires explicit approval, an exact row allowlist, and a pre-op backup.",
        "",
    ]
    (output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crosswalk-csv", required=True)
    parser.add_argument("--lookup-packet", action="append", required=True)
    parser.add_argument("--winner-only-rows-jsonl", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Iterable[str] | None = None, *, root: Path | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    packet = build_review_packet(
        crosswalk_csv_path=Path(args.crosswalk_csv),
        lookup_packet_paths=[Path(path) for path in args.lookup_packet],
        winner_only_rows_paths=[Path(path) for path in args.winner_only_rows_jsonl],
    )
    write_outputs(Path(args.output_dir), packet, root=root)
    print(
        json.dumps(
            {"status": packet["status"], "summary": packet["summary"]},
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if packet["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
