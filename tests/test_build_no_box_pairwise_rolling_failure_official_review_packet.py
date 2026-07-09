import csv
import json
from pathlib import Path

import pytest

from scripts.build_no_box_pairwise_rolling_failure_official_review_packet import (
    build_review_packet,
    main,
)


def _crosswalk(path: Path) -> Path:
    rows = [
        {
            "race_id": "MISS_2025-07-01_1",
            "identity_key": "2025-07-01|MISS|R01",
            "race_date": "2025-07-01",
            "venue": "MISS",
            "race_number": "1",
            "winner_rank": "4",
            "top1_hit": "False",
            "top3_hit": "False",
            "field_size": "5",
            "field_scope": "partial_db_name_subset_of_official_finishers",
            "distance_bucket": "sprint_lt_400",
            "winner_box_bucket": "middle_3_5",
            "source_gap_status": "WINNER_SOURCE_MISSING",
            "best_queue_policy_key": "latest_source_agreement_strict_train_ranks_51_100",
            "best_queue_key": "canonical_identity_ready",
        },
        {
            "race_id": "HIT_2025-07-02_2",
            "identity_key": "2025-07-02|HIT|R02",
            "race_date": "2025-07-02",
            "venue": "HIT",
            "race_number": "2",
            "winner_rank": "1",
            "top1_hit": "True",
            "top3_hit": "True",
            "field_size": "4",
            "field_scope": "partial_db_name_subset_of_official_finishers",
            "distance_bucket": "sprint_400_499",
            "winner_box_bucket": "inside_1_2",
            "source_gap_status": "WINNER_SOURCE_MISSING",
            "best_queue_policy_key": "latest_source_agreement_strict_train_ranks_51_100",
            "best_queue_key": "canonical_identity_ready",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


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
                "legacy_race_id": "MISS_2025-07-01_1",
                "lookup_status": "OFFICIAL_RESULT_PARSED",
                "result_parse_ready": True,
                "label_write_ready": False,
                "skip_reasons": ["official_positions_incomplete_for_legacy_runner_count"],
                "positions": [
                    {"box_number": 1, "finish_position": 2},
                    {"box_number": 4, "finish_position": 1},
                ],
                "terminal_statuses": [],
                "source_url": "https://example.invalid/results",
            },
            {
                "legacy_race_id": "HIT_2025-07-02_2",
                "lookup_status": "OFFICIAL_RESULT_NOT_PARSED",
                "result_parse_ready": False,
                "label_write_ready": False,
                "skip_reasons": ["official_http_404"],
                "positions": [],
                "terminal_statuses": [],
                "source_url": "https://example.invalid/missing",
            },
        ],
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _winner_rows(path: Path) -> Path:
    rows = [
        {
            "race_id": "MISS_2025-07-01_1",
            "dog_name": "Winner",
            "actual_win": 1,
            "box_features_allowed": False,
            "finish_order_labels_allowed": False,
            "top3_labels_allowed": False,
            "official_safe_label_candidate": False,
            "label_write_approved": False,
            "race_grouped_actual_win_ranking_allowed": False,
        }
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_failure_review_packet_prioritizes_top3_miss_with_existing_winner_rows(tmp_path: Path):
    packet = build_review_packet(
        crosswalk_csv_path=_crosswalk(tmp_path / "crosswalk.csv"),
        lookup_packet_paths=[_lookup(tmp_path / "lookup.json")],
        winner_only_rows_paths=[_winner_rows(tmp_path / "winner_rows.jsonl")],
    )

    assert packet["status"] == "REPORT_ONLY_ROLLING_FAILURE_OFFICIAL_REVIEW_PACKET"
    assert packet["writes_performed"]["label_write"] is False
    assert packet["summary"]["top1_miss_count"] == 1
    assert packet["summary"]["top3_miss_count"] == 1
    assert packet["summary"]["result_parse_ready_count"] == 1
    assert packet["summary"]["winner_only_materialized_race_count"] == 1
    first = packet["review_rows"][0]
    assert first["race_id"] == "MISS_2025-07-01_1"
    assert first["review_lane"] == "P0_TOP3_MISS_PARSED_OFFICIAL_REVIEW"


def test_failure_review_packet_fails_closed_on_lookup_write_claim(tmp_path: Path):
    packet = build_review_packet(
        crosswalk_csv_path=_crosswalk(tmp_path / "crosswalk.csv"),
        lookup_packet_paths=[_lookup(tmp_path / "lookup.json", write_claim=True)],
        winner_only_rows_paths=[],
    )

    assert packet["status"] == "REPORT_ONLY_ROLLING_FAILURE_OFFICIAL_REVIEW_PACKET_WITH_FAILURES"
    assert any("lookup_packet_write_flag_true" in item for item in packet["failures"])


def test_failure_review_packet_cli_writes_outputs(tmp_path: Path):
    output_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/review"
    status = main(
        [
            "--crosswalk-csv",
            str(_crosswalk(tmp_path / "crosswalk.csv")),
            "--lookup-packet",
            str(_lookup(tmp_path / "lookup.json")),
            "--winner-only-rows-jsonl",
            str(_winner_rows(tmp_path / "winner_rows.jsonl")),
            "--output-dir",
            str(output_dir),
        ],
        root=tmp_path,
    )

    assert status == 0
    assert (output_dir / "no_box_pairwise_rolling_failure_official_review_packet.json").exists()
    assert (output_dir / "rolling_failure_official_review_queue.csv").exists()
    assert (output_dir / "SUMMARY.md").exists()


def test_failure_review_packet_cli_rejects_output_outside_repo(tmp_path: Path):
    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        main(
            [
                "--crosswalk-csv",
                str(_crosswalk(tmp_path / "crosswalk.csv")),
                "--lookup-packet",
                str(_lookup(tmp_path / "lookup.json")),
                "--output-dir",
                str(tmp_path.parent / "failure-review-outside"),
            ],
            root=tmp_path,
        )


def test_failure_review_packet_cli_rejects_output_outside_artifacts(tmp_path: Path):
    with pytest.raises(ValueError, match="output_dir_must_be_under_artifacts"):
        main(
            [
                "--crosswalk-csv",
                str(_crosswalk(tmp_path / "crosswalk.csv")),
                "--lookup-packet",
                str(_lookup(tmp_path / "lookup.json")),
                "--output-dir",
                str(tmp_path / "reports/failure-review"),
            ],
            root=tmp_path,
        )
