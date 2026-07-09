import json
import sqlite3
from pathlib import Path

import pytest

from scripts.build_single_race_official_gap_review_packet import (
    build_gap_review_packet,
    main,
)


def _lookup(path: Path, *, write_claim: bool = False) -> Path:
    payload = {
        "schema_version": "official_reverify_lookup_dry_run_v1",
        "status": "REPORT_ONLY",
        "writes_performed": {
            "db_write": False,
            "label_write": write_claim,
            "official_fetch": True,
        },
        "results": [
            {
                "legacy_race_id": "TEST_2025-07-01_1",
                "lookup_status": "OFFICIAL_RESULT_PARSED",
                "result_parse_ready": True,
                "label_write_ready": False,
                "skip_reasons": ["official_positions_incomplete_for_legacy_runner_count"],
                "positions": [
                    {"box_number": 1, "dog_name": "Alpha", "finish_position": 1},
                    {"box_number": 2, "dog_name": "Beta", "finish_position": 2},
                    {"box_number": 3, "dog_name": "Gamma", "finish_position": 3},
                ],
                "terminal_statuses": [],
                "source_url": "https://example.invalid/results",
            }
        ],
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


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
            winner_source text
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
    conn.execute(
        "insert into race_metadata values (?,?,?,?,?,?,?,?,?,?,?)",
        ("TEST_2025-07-01_1", "2025-07-01", "TEST", 1, "400", "5", 2, None, "complete", "Alpha", None),
    )
    conn.executemany(
        "insert into dog_race_data values (?,?,?,?,?,?,?)",
        [
            ("TEST_2025-07-01_1", "1. Alpha", 4, 1, None, None, None),
            ("TEST_2025-07-01_1", "2. Beta", 2, 3, None, None, None),
        ],
    )
    conn.commit()
    conn.close()
    return path


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _predictions(path: Path) -> Path:
    rows = [
        {
            "race_id": "TEST_2025-07-01_1",
            "dog_name": "Alpha",
            "actual_win": 1,
            "predicted_rank": 2,
            "score": 1.5,
            "box_features_allowed": False,
            "finish_order_labels_allowed": False,
            "top3_labels_allowed": False,
            "official_safe_label_candidate": False,
            "label_write_approved": False,
            "feature_join_status": "MATCHED",
            "history_feature_join_status": "MATCHED",
            "history_feature_values_filled": 12,
        },
        {
            "race_id": "TEST_2025-07-01_1",
            "dog_name": "Beta",
            "actual_win": 0,
            "predicted_rank": 1,
            "score": 2.0,
            "box_features_allowed": False,
            "finish_order_labels_allowed": False,
            "top3_labels_allowed": False,
            "official_safe_label_candidate": False,
            "label_write_approved": False,
            "feature_join_status": "MATCHED",
            "history_feature_join_status": "MATCHED",
            "history_feature_values_filled": 10,
        },
    ]
    return _jsonl(path, rows)


def _winner_only(path: Path) -> Path:
    rows = [
        {
            "race_id": "TEST_2025-07-01_1",
            "dog_name": "Alpha",
            "actual_win": 1,
            "box_features_allowed": False,
            "finish_order_labels_allowed": False,
            "top3_labels_allowed": False,
            "official_safe_label_candidate": False,
            "label_write_approved": False,
        },
        {
            "race_id": "TEST_2025-07-01_1",
            "dog_name": "Beta",
            "actual_win": 0,
            "box_features_allowed": False,
            "finish_order_labels_allowed": False,
            "top3_labels_allowed": False,
            "official_safe_label_candidate": False,
            "label_write_approved": False,
        },
    ]
    return _jsonl(path, rows)


def test_single_race_gap_review_identifies_missing_runner_and_drift(tmp_path: Path):
    packet = build_gap_review_packet(
        race_id="TEST_2025-07-01_1",
        lookup_packet_paths=[_lookup(tmp_path / "lookup.json")],
        db_path=_db(tmp_path / "greyhound.sqlite"),
        prediction_rows_path=_predictions(tmp_path / "predictions.jsonl"),
        winner_only_rows_path=_winner_only(tmp_path / "winner_only.jsonl"),
    )

    assert packet["status"] == "REPORT_ONLY_SINGLE_RACE_OFFICIAL_GAP_REVIEW"
    assert packet["writes_performed"]["db_write"] is False
    assert packet["db_state"]["quick_check"] == "ok"
    assert packet["summary"]["official_runner_count"] == 3
    assert packet["summary"]["db_runner_count"] == 2
    assert packet["summary"]["missing_db_runner_count"] == 1
    assert packet["summary"]["db_box_drift_count"] == 1
    assert packet["summary"]["db_finish_drift_count"] == 1
    assert packet["feature_summary"]["winner_predicted_rank"] == 2
    by_name = {row["name_key"]: row for row in packet["runner_review_rows"]}
    assert "missing_db_runner" in by_name["gamma"]["gap_flags"]
    assert "db_box_differs_from_official" in by_name["alpha"]["gap_flags"]
    assert "db_finish_differs_from_official" in by_name["beta"]["gap_flags"]


def test_single_race_gap_review_fails_closed_on_lookup_write_claim(tmp_path: Path):
    packet = build_gap_review_packet(
        race_id="TEST_2025-07-01_1",
        lookup_packet_paths=[_lookup(tmp_path / "lookup.json", write_claim=True)],
        db_path=_db(tmp_path / "greyhound.sqlite"),
        prediction_rows_path=_predictions(tmp_path / "predictions.jsonl"),
        winner_only_rows_path=_winner_only(tmp_path / "winner_only.jsonl"),
    )

    assert packet["status"] == "REPORT_ONLY_SINGLE_RACE_OFFICIAL_GAP_REVIEW_WITH_FAILURES"
    assert any("lookup_packet_write_flag_true" in item for item in packet["failures"])


def test_single_race_gap_review_cli_writes_outputs(tmp_path: Path):
    output_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/single"
    status = main(
        [
            "--race-id",
            "TEST_2025-07-01_1",
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

    assert status == 0
    assert (output_dir / "single_race_official_gap_review_packet.json").exists()
    assert (output_dir / "single_race_official_gap_runner_review.csv").exists()
    assert (output_dir / "SUMMARY.md").exists()


def test_single_race_gap_review_cli_rejects_output_outside_repo(tmp_path: Path):
    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        main(
            [
                "--race-id",
                "TEST_2025-07-01_1",
                "--lookup-packet",
                str(_lookup(tmp_path / "lookup.json")),
                "--db",
                str(_db(tmp_path / "greyhound.sqlite")),
                "--predictions-jsonl",
                str(_predictions(tmp_path / "predictions.jsonl")),
                "--winner-only-rows-jsonl",
                str(_winner_only(tmp_path / "winner_only.jsonl")),
                "--output-dir",
                str(tmp_path.parent / "gap-review-outside"),
            ],
            root=tmp_path,
        )


def test_single_race_gap_review_cli_rejects_output_outside_artifacts(tmp_path: Path):
    with pytest.raises(ValueError, match="output_dir_must_be_under_artifacts"):
        main(
            [
                "--race-id",
                "TEST_2025-07-01_1",
                "--lookup-packet",
                str(_lookup(tmp_path / "lookup.json")),
                "--db",
                str(_db(tmp_path / "greyhound.sqlite")),
                "--predictions-jsonl",
                str(_predictions(tmp_path / "predictions.jsonl")),
                "--winner-only-rows-jsonl",
                str(_winner_only(tmp_path / "winner_only.jsonl")),
                "--output-dir",
                str(tmp_path / "reports/gap-review"),
            ],
            root=tmp_path,
        )
