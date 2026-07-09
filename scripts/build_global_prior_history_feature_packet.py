#!/usr/bin/env python3
"""Build a report-only global-prior dog-history feature packet.

This formalizes the current-214 global prior-history prototype. It reads an
existing no-box actual-win feature row packet, augments dog-form fields from
csv_dog_history_staging using only prior raw DATE rows, and writes report-local
artifacts. It does not fetch, mutate labels, update metadata, train, persist, or
promote any model.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_no_box_actual_win_feature_join_packet import (
    HISTORY_FILL_POLICIES,
    HISTORY_OUTCOME_PROXY_FEATURES,
    WRITES_PERFORMED,
    _assert_output_dir_safe,
    _history_feature_bundle,
    _history_lookup_key_candidates,
    _history_time,
    _load_jsonl,
    _name_key,
    _parse_date,
    _safe_float,
    _safe_int,
    _safe_number,
)


SCHEMA_VERSION = "no_box_actual_win_dog_form_feature_join_global_prior_history_suffix_normalized_v1"
ROWS_SCHEMA_VERSION = "no_box_actual_win_dog_form_feature_rows_global_prior_suffix_normalized_v1"
FEATURE_JOIN_STATUS_FIELD = "feature_join_status"
GLOBAL_PRIOR_STATUS_EXACT = "MATCHED_GLOBAL_PRIOR_DOG_HISTORY"
GLOBAL_PRIOR_STATUS_SUFFIX = "MATCHED_SUFFIX_STRIPPED_GLOBAL_PRIOR_DOG_HISTORY"
GLOBAL_PRIOR_STATUS_MISSING = "NO_GLOBAL_PRIOR_DOG_HISTORY"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("json_root_not_object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _feature_names(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted(
        {
            key.removeprefix("feature_")
            for row in rows
            for key in row
            if key.startswith("feature_") and key != FEATURE_JOIN_STATUS_FIELD
        }
    )


def _history_row_from_db(mapped: Mapping[str, Any], raw: Mapping[str, Any], prior_date: date) -> dict[str, Any]:
    return {
        "race_date": prior_date.isoformat(),
        "venue": raw.get("TRACK"),
        "distance": _safe_float(raw.get("DIST")),
        "grade": str(raw.get("G") or "").strip().upper(),
        "finish_position": _safe_int(mapped.get("finish_position") or raw.get("PLC")),
        "individual_time": _history_time(mapped, raw, "individual_time", "TIME"),
        "sectional_1st": _history_time(mapped, raw, "sectional_1st", "1 SEC"),
        "margin": _history_time(mapped, raw, "margin", "MGN"),
        "weight": _history_time(mapped, raw, "weight", "WGT"),
    }


def load_global_history_index_from_db(
    db_path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    resolved = db_path.expanduser().resolve()
    conn = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    history_by_key: dict[str, list[dict[str, Any]]] = {}
    rows_seen = 0
    rows_used = 0
    skipped_no_prior_date = 0
    skipped_no_dog_key = 0
    try:
        conn.execute("pragma query_only=ON")
        quick_check = str(conn.execute("pragma quick_check").fetchone()[0])
        for row in conn.execute(
            """
            select race_id, dog_name, dog_clean_name, finish_position, weight,
                   individual_time, sectional_1st, margin, raw_row_json
            from csv_dog_history_staging
            """
        ):
            rows_seen += 1
            mapped = dict(row)
            raw = json.loads(mapped.get("raw_row_json") or "{}")
            prior_date = _parse_date(raw.get("DATE"))
            if prior_date is None:
                skipped_no_prior_date += 1
                continue
            dog_key = _name_key(mapped.get("dog_clean_name") or mapped.get("dog_name"))
            if not dog_key:
                skipped_no_dog_key += 1
                continue
            history_by_key.setdefault(dog_key, []).append(
                _history_row_from_db(mapped, raw, prior_date)
            )
            rows_used += 1
    finally:
        conn.close()

    for rows in history_by_key.values():
        rows.sort(
            key=lambda item: _parse_date(item.get("race_date")) or date.min,
            reverse=True,
        )
    return history_by_key, {
        "history_db_path": str(resolved),
        "db_quick_check": quick_check,
        "history_db_rows_seen": rows_seen,
        "history_db_rows_used": rows_used,
        "history_db_rows_skipped_no_prior_date": skipped_no_prior_date,
        "history_db_rows_skipped_no_dog_key": skipped_no_dog_key,
        "history_db_dog_keys_with_prior_rows": len(history_by_key),
    }


def _prior_rows_for_target(
    *,
    row: Mapping[str, Any],
    history_by_key: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[Mapping[str, Any]], str, str | None]:
    target_date = _parse_date(row.get("race_date"))
    if target_date is None:
        return [], GLOBAL_PRIOR_STATUS_MISSING, None
    primary_key = str(row.get("dog_name_key") or _name_key(row.get("dog_name")))
    for dog_key, match_status in _history_lookup_key_candidates(row, primary_key):
        prior_rows = [
            history_row
            for history_row in history_by_key.get(dog_key, [])
            if (_parse_date(history_row.get("race_date")) or date.max) < target_date
        ]
        if prior_rows:
            status = (
                GLOBAL_PRIOR_STATUS_SUFFIX
                if match_status == "MATCHED_SUFFIX_STRIPPED_TARGET_NAME"
                else GLOBAL_PRIOR_STATUS_EXACT
            )
            return prior_rows, status, dog_key
    return [], GLOBAL_PRIOR_STATUS_MISSING, None


def _target_meta(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "race_date": row.get("race_date"),
        "venue": row.get("venue"),
        "distance": row.get("target_distance") or row.get("distance"),
        "grade": row.get("target_grade") or row.get("grade"),
    }


def _should_fill(current_value: Any, candidate_value: Any) -> bool:
    if candidate_value is None:
        return False
    if current_value in (None, ""):
        return True
    current_number = _safe_number(current_value)
    candidate_number = _safe_number(candidate_value)
    return current_number == 0 and candidate_number not in (None, 0)


def build_global_prior_history_packet(
    *,
    base_packet: Mapping[str, Any],
    base_rows: Sequence[Mapping[str, Any]],
    history_by_key: Mapping[str, Sequence[Mapping[str, Any]]],
    history_summary: Mapping[str, Any],
    base_packet_path: str | None = None,
    base_rows_path: str | None = None,
    history_fill_policy: str = "all",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if history_fill_policy not in HISTORY_FILL_POLICIES:
        raise ValueError(f"unsupported_history_fill_policy:{history_fill_policy}")

    feature_names = _feature_names(base_rows)
    output_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    candidate_kind_counts: Counter[str] = Counter()
    feature_fill_counts: Counter[str] = Counter()
    policy_skipped_counts: Counter[str] = Counter()
    rows_with_history = 0
    winner_rows_with_history = 0
    races_with_history: set[str] = set()
    match_examples: list[dict[str, Any]] = []

    for row in base_rows:
        joined = dict(row)
        joined["schema_version"] = ROWS_SCHEMA_VERSION
        prior_rows, status, matched_key = _prior_rows_for_target(
            row=row,
            history_by_key=history_by_key,
        )
        status_counts[status] += 1
        candidate_kind_counts[str(row.get("candidate_kind") or "UNKNOWN")] += 1
        joined["history_feature_join_status"] = status
        joined["global_prior_history_count"] = len(prior_rows)
        joined["global_prior_history_latest_date"] = (
            prior_rows[0].get("race_date") if prior_rows else None
        )
        joined["global_prior_history_values_filled"] = 0
        if matched_key is not None:
            joined["global_prior_history_matched_key"] = matched_key

        if prior_rows:
            rows_with_history += 1
            races_with_history.add(str(row.get("race_id") or ""))
            if int(row.get("actual_win") or 0) == 1:
                winner_rows_with_history += 1
            bundle = _history_feature_bundle(
                target_meta=_target_meta(row),
                history_rows=prior_rows,
            )
            for name in feature_names:
                if (
                    history_fill_policy == "no_outcome_proxy_fields"
                    and name in HISTORY_OUTCOME_PROXY_FEATURES
                ):
                    if _safe_number(bundle.get(name)) is not None:
                        policy_skipped_counts[name] += 1
                    continue
                feature_key = f"feature_{name}"
                value = _safe_number(bundle.get(name))
                if not _should_fill(joined.get(feature_key), value):
                    continue
                joined[feature_key] = value
                joined["global_prior_history_values_filled"] += 1
                feature_fill_counts[name] += 1
            if len(match_examples) < 20:
                match_examples.append(
                    {
                        "race_id": row.get("race_id"),
                        "race_date": row.get("race_date"),
                        "venue": row.get("venue"),
                        "dog_name": row.get("dog_name"),
                        "dog_name_key": row.get("dog_name_key"),
                        "matched_key": matched_key,
                        "status": status,
                        "prior_history_count": len(prior_rows),
                        "latest_prior_date": prior_rows[0].get("race_date"),
                        "latest_prior_venue": prior_rows[0].get("venue"),
                        "latest_prior_distance": prior_rows[0].get("distance"),
                        "latest_prior_grade": prior_rows[0].get("grade"),
                    }
                )
        output_rows.append(joined)

    feature_non_null_counts = {
        name: sum(
            1
            for row in output_rows
            if _safe_number(row.get(f"feature_{name}")) is not None
        )
        for name in feature_names
    }
    summary = {
        "base_rows_seen": len(base_rows),
        "joined_rows": len(output_rows),
        "race_count": len({str(row.get("race_id") or "") for row in output_rows}),
        "candidate_kind_counts": dict(sorted(candidate_kind_counts.items())),
        "history_feature_match_status_counts": dict(sorted(status_counts.items())),
        "rows_with_global_prior_history": rows_with_history,
        "rows_without_global_prior_history": len(output_rows) - rows_with_history,
        "suffix_stripped_global_prior_history_rows": status_counts.get(
            GLOBAL_PRIOR_STATUS_SUFFIX,
            0,
        ),
        "exact_global_prior_history_rows": status_counts.get(GLOBAL_PRIOR_STATUS_EXACT, 0),
        "winner_rows_with_global_prior_history": winner_rows_with_history,
        "winner_rows_without_global_prior_history": sum(
            1 for row in output_rows if int(row.get("actual_win") or 0) == 1
        )
        - winner_rows_with_history,
        "races_with_any_global_prior_history": len(races_with_history),
        "feature_column_count": len(feature_names),
        "features_with_non_null_values": sum(
            1 for value in feature_non_null_counts.values() if value > 0
        ),
        "all_null_feature_count": sum(
            1 for value in feature_non_null_counts.values() if value == 0
        ),
        "feature_non_null_counts": dict(sorted(feature_non_null_counts.items())),
        "global_prior_feature_fill_counts": dict(sorted(feature_fill_counts.items())),
        "global_prior_feature_value_fill_count": sum(feature_fill_counts.values()),
        "global_prior_policy_skipped_feature_counts": dict(
            sorted(policy_skipped_counts.items())
        ),
        "global_prior_policy_skipped_feature_value_count": sum(policy_skipped_counts.values()),
        "history_db_feature_summary": dict(history_summary),
        "history_db_fill_policy": history_fill_policy,
        "label_proxy_audit": {"status": "NOT_RUN_GLOBAL_PRIOR_PRIOR_DATE_ONLY"},
        "failures": [],
    }
    packet = {
        "schema_version": SCHEMA_VERSION,
        "status": "REPORT_ONLY_DOG_FORM_FEATURE_JOIN_READY",
        "report_only": True,
        "writes_performed": dict(WRITES_PERFORMED),
        "source_packet_status": base_packet.get("status"),
        "inputs": {
            "base_feature_join_packet": base_packet_path,
            "base_feature_rows": base_rows_path,
            "history_db": history_summary.get("history_db_path"),
            "history_source": "csv_dog_history_staging matched by normalized dog name and prior raw DATE",
        },
        "summary": summary,
        "global_prior_history_method": "suffix_normalized_target_name_prior_raw_date_only",
        "history_match_examples": match_examples,
        "limitations": [
            "prototype_only_not_canonical_dataset_regeneration",
            "target_distance_and_target_grade_remain_missing_for_current_official_race_metadata",
            "same_distance_features_remain_unavailable_without_target_distance",
            "suffix_stripping_is_target_side_only_no_fuzzy_alias_matching",
        ],
        "recommended_next_action": "run_no_box_pairwise_rolling_windows_on_suffix_normalized_global_prior_history",
    }
    return packet, output_rows


def write_outputs(output_dir: Path, packet: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    output_dir = _assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "no_box_actual_win_feature_join_packet.json", packet)
    _write_jsonl(output_dir / "no_box_actual_win_feature_rows.jsonl", rows)
    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "no_box_actual_win_feature_rows.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    summary = packet.get("summary") or {}
    lines = [
        "# No-Box Global Prior-History Feature Packet",
        "",
        f"Status: `{packet.get('status')}`.",
        "",
        "No DB writes, label writes, metadata writes, official fetches, snapshot or manifest mutations, model training, registry updates, promotions, TGR enablement, betting decisions, or EV actions occurred.",
        "",
        f"- Rows: `{summary.get('joined_rows')}`",
        f"- Races: `{summary.get('race_count')}`",
        f"- Rows with global prior history: `{summary.get('rows_with_global_prior_history')}`",
        f"- Suffix-stripped global prior rows: `{summary.get('suffix_stripped_global_prior_history_rows')}`",
        f"- Winner rows with global prior history: `{summary.get('winner_rows_with_global_prior_history')}`",
        f"- Feature values filled: `{summary.get('global_prior_feature_value_fill_count')}`",
        f"- Match status counts: `{summary.get('history_feature_match_status_counts')}`",
        f"- DB quick_check: `{(summary.get('history_db_feature_summary') or {}).get('db_quick_check')}`",
        "",
    ]
    (output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-feature-join-packet", required=True)
    parser.add_argument("--base-feature-rows", required=True)
    parser.add_argument("--history-db", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--expected-races", type=int)
    parser.add_argument(
        "--history-fill-policy",
        choices=sorted(HISTORY_FILL_POLICIES),
        default="all",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_packet_path = Path(args.base_feature_join_packet).expanduser().resolve()
    base_rows_path = Path(args.base_feature_rows).expanduser().resolve()
    history_by_key, history_summary = load_global_history_index_from_db(Path(args.history_db))
    base_rows = _load_jsonl(base_rows_path)
    if args.expected_rows is not None and len(base_rows) != args.expected_rows:
        raise ValueError(f"expected_rows_mismatch:{args.expected_rows}:{len(base_rows)}")
    if args.expected_races is not None:
        race_count = len({str(row.get("race_id") or "") for row in base_rows})
        if race_count != args.expected_races:
            raise ValueError(f"expected_races_mismatch:{args.expected_races}:{race_count}")
    packet, rows = build_global_prior_history_packet(
        base_packet=_load_json(base_packet_path),
        base_rows=base_rows,
        history_by_key=history_by_key,
        history_summary=history_summary,
        base_packet_path=str(base_packet_path),
        base_rows_path=str(base_rows_path),
        history_fill_policy=args.history_fill_policy,
    )
    write_outputs(Path(args.output_dir), packet, rows)
    print(json.dumps({"status": packet["status"], "summary": packet["summary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
