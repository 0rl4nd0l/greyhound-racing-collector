import csv
import json
import sqlite3
from pathlib import Path

import pytest

from scripts.build_rolling_failure_repair_triage_packet import (
    build_repair_triage_packet,
    main,
)


def _db(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        create table race_metadata (
            race_id text,
            race_date text,
            venue text,
            race_number integer,
            distance text,
            grade text,
            field_size integer,
            actual_field_size integer,
            results_status text,
            winner_name text,
            winner_source text,
            url text,
            data_source text
        )
        """
    )
    conn.execute(
        """
        create table dog_race_data (
            race_id text,
            dog_name text,
            box_number integer,
            finish_position integer,
            placing integer,
            scraped_finish_position text,
            data_source text
        )
        """
    )
    conn.executemany(
        "insert into race_metadata values (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            ("R1", "2025-07-01", "TEST", 1, "400", "5", 2, None, "complete", "Alpha", None, None, None),
            ("R2", "2025-07-02", "TEST", 2, "400", "5", 2, None, "complete", "Delta", None, None, None),
        ],
    )
    conn.executemany(
        "insert into dog_race_data values (?,?,?,?,?,?,?)",
        [
            ("R1", "1. Alpha", 4, 1, None, None, None),
            ("R1", "2. Beta", 2, 3, None, None, None),
            ("R2", "1. Delta", 1, 1, None, None, None),
            ("R2", "2. Echo", 2, 2, None, None, None),
        ],
    )
    conn.commit()
    conn.close()
    return path


def _lookup(path: Path) -> Path:
    payload = {
        "schema_version": "official_reverify_lookup_dry_run_v1",
        "status": "REPORT_ONLY",
        "writes_performed": {
            "db_write": False,
            "label_write": False,
            "official_fetch": True,
        },
        "results": [
            {
                "legacy_race_id": "R1",
                "lookup_status": "OFFICIAL_RESULT_PARSED",
                "result_parse_ready": True,
                "label_write_ready": False,
                "skip_reasons": ["official_positions_incomplete_for_legacy_runner_count"],
                "source_url": "https://example.invalid/r1",
                "positions": [
                    {"dog_name": "Alpha", "box_number": 1, "finish_position": 1},
                    {"dog_name": "Beta", "box_number": 2, "finish_position": 2},
                    {"dog_name": "Gamma", "box_number": 3, "finish_position": 3},
                ],
                "terminal_statuses": [],
            },
            {
                "legacy_race_id": "R2",
                "lookup_status": "OFFICIAL_RESULT_PARSED",
                "result_parse_ready": True,
                "label_write_ready": False,
                "skip_reasons": [],
                "source_url": "https://example.invalid/r2",
                "positions": [
                    {"dog_name": "Delta", "box_number": 1, "finish_position": 1},
                    {"dog_name": "Echo", "box_number": 2, "finish_position": 2},
                ],
                "terminal_statuses": [],
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _predictions(path: Path) -> Path:
    rows = []
    for race_id, names in [("R1", ["Alpha", "Beta"]), ("R2", ["Delta", "Echo"])]:
        for rank, name in enumerate(names, start=1):
            rows.append(
                {
                    "race_id": race_id,
                    "dog_name": name,
                    "actual_win": 1 if rank == 1 else 0,
                    "predicted_rank": rank,
                    "score": 10 - rank,
                    "box_features_allowed": False,
                    "finish_order_labels_allowed": False,
                    "top3_labels_allowed": False,
                    "official_safe_label_candidate": False,
                    "label_write_approved": False,
                }
            )
    return _jsonl(path, rows)


def _winner_only(path: Path) -> Path:
    rows = []
    for race_id, names in [("R1", ["Alpha", "Beta"]), ("R2", ["Delta", "Echo"])]:
        for rank, name in enumerate(names, start=1):
            rows.append(
                {
                    "race_id": race_id,
                    "dog_name": name,
                    "actual_win": 1 if rank == 1 else 0,
                    "box_features_allowed": False,
                    "finish_order_labels_allowed": False,
                    "top3_labels_allowed": False,
                    "official_safe_label_candidate": False,
                    "label_write_approved": False,
                }
            )
    return _jsonl(path, rows)


def _queue(path: Path) -> Path:
    rows = [
        {
            "priority": "0",
            "review_lane": "P0_TOP3_MISS_PARSED_OFFICIAL_REVIEW",
            "race_id": "R1",
            "race_date": "2025-07-01",
            "top1_hit": "False",
            "top3_hit": "False",
            "winner_rank": "4",
            "field_scope": "partial_db_name_subset_of_official_finishers",
            "terminal_status_count": "0",
            "source_url": "https://example.invalid/r1",
        },
        {
            "priority": "1",
            "review_lane": "P1_TOP1_MISS_PARSED_WINNER_ONLY_READY_REVIEW",
            "race_id": "R2",
            "race_date": "2025-07-02",
            "top1_hit": "False",
            "top3_hit": "True",
            "winner_rank": "2",
            "field_scope": "partial_db_name_subset_of_official_finishers",
            "terminal_status_count": "0",
            "source_url": "https://example.invalid/r2",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_rolling_failure_repair_triage_classifies_queue_lanes(tmp_path: Path):
    packet = build_repair_triage_packet(
        failure_review_csv_path=_queue(tmp_path / "queue.csv"),
        lookup_packet_paths=[_lookup(tmp_path / "lookup.json")],
        db_path=_db(tmp_path / "greyhound.sqlite"),
        prediction_rows_path=_predictions(tmp_path / "predictions.jsonl"),
        winner_only_rows_path=_winner_only(tmp_path / "winner_only.jsonl"),
    )

    assert packet["schema_version"] == "rolling_failure_repair_triage_packet_v1"
    assert packet["status"] == "REPORT_ONLY_ROLLING_FAILURE_REPAIR_TRIAGE_PACKET"
    assert packet["writes_performed"]["db_write"] is False
    assert packet["safe_to_write_now"] is False
    assert packet["summary"]["races_considered"] == 2
    assert packet["summary"]["safe_to_write_now_count"] == 0
    assert packet["summary"]["repair_lane_counts"] == {
        "existing_runner_update_policy_required": 1,
        "missing_runner_insert_policy_required": 1,
    }
    r1 = next(row for row in packet["triage_rows"] if row["race_id"] == "R1")
    assert r1["missing_runner_insert_candidate_count"] == 1
    assert r1["repair_lane"] == "missing_runner_insert_policy_required"
    r2 = next(row for row in packet["triage_rows"] if row["race_id"] == "R2")
    assert r2["missing_runner_insert_candidate_count"] == 0
    assert r2["changed_dog_update_candidate_count"] == 2


def test_rolling_failure_repair_triage_cli_writes_outputs(tmp_path: Path):
    output_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/triage"
    exit_code = main(
        [
            "--failure-review-csv",
            str(_queue(tmp_path / "queue.csv")),
            "--lookup-packet",
            str(_lookup(tmp_path / "lookup.json")),
            "--db",
            str(_db(tmp_path / "greyhound.sqlite")),
            "--predictions-jsonl",
            str(_predictions(tmp_path / "predictions.jsonl")),
            "--winner-only-rows-jsonl",
            str(_winner_only(tmp_path / "winner_only.jsonl")),
            "--output-dir",
            str(output_dir),
        ],
        root=tmp_path,
    )

    assert exit_code == 0
    assert (output_dir / "rolling_failure_repair_triage_packet.json").exists()
    assert (output_dir / "rolling_failure_repair_triage.csv").exists()
    assert "No DB rows" in (output_dir / "SUMMARY.md").read_text(encoding="utf-8")


def test_rolling_failure_repair_triage_cli_rejects_output_outside_repo(tmp_path: Path):
    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        main(
            [
                "--failure-review-csv",
                str(_queue(tmp_path / "queue.csv")),
                "--lookup-packet",
                str(_lookup(tmp_path / "lookup.json")),
                "--db",
                str(_db(tmp_path / "greyhound.sqlite")),
                "--predictions-jsonl",
                str(_predictions(tmp_path / "predictions.jsonl")),
                "--winner-only-rows-jsonl",
                str(_winner_only(tmp_path / "winner_only.jsonl")),
                "--output-dir",
                str(tmp_path.parent / "triage-outside"),
            ],
            root=tmp_path,
        )


def test_rolling_failure_repair_triage_cli_rejects_output_outside_artifacts(tmp_path: Path):
    with pytest.raises(ValueError, match="output_dir_must_be_under_artifacts"):
        main(
            [
                "--failure-review-csv",
                str(_queue(tmp_path / "queue.csv")),
                "--lookup-packet",
                str(_lookup(tmp_path / "lookup.json")),
                "--db",
                str(_db(tmp_path / "greyhound.sqlite")),
                "--predictions-jsonl",
                str(_predictions(tmp_path / "predictions.jsonl")),
                "--winner-only-rows-jsonl",
                str(_winner_only(tmp_path / "winner_only.jsonl")),
                "--output-dir",
                str(tmp_path / "reports/triage"),
            ],
            root=tmp_path,
        )
