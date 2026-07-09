import json
import sqlite3
from pathlib import Path

import pytest

from scripts.build_no_box_pairwise_rolling_source_label_packet import (
    build_source_label_packet,
    main,
)


def _analysis() -> dict:
    return {
        "status": "REPORT_ONLY_PAIRWISE_ROLLING_STRATIFIED_ERROR_ANALYSIS",
        "report_only": True,
        "write_ready": False,
        "label_write_approved": False,
        "model_training_performed": False,
        "model_promotion_allowed": False,
        "writes_performed": {"db_write": False, "label_write": False},
        "source_sample_size_status": "UNDERPOWERED_BELOW_50_ACTUAL_WIN_RACES",
        "source_reserved_final_races": 5,
        "source_reserved_races_predicted": False,
        "race_records": [
            {
                "race_id": "TEST_2025-07-01_1",
                "race_date": "2025-07-01",
                "venue": "TEST",
                "window_id": "window_01",
                "winner_rank": 2,
                "top1_hit": False,
                "top3_hit": True,
                "field_size": 4,
                "field_scope": "partial_db_name_subset_of_official_finishers",
                "field_complete_for_ranking": "False",
                "distance_bucket": "sprint_400_499",
                "winner_box_bucket": "inside_1_2",
                "source_bucket": "DATA_MISSING",
            },
            {
                "race_id": "MISS_2025-07-02_2",
                "race_date": "2025-07-02",
                "venue": "MISS",
                "window_id": "window_01",
                "winner_rank": 1,
                "top1_hit": True,
                "top3_hit": True,
                "field_size": 4,
                "field_scope": "partial_db_name_subset_of_official_finishers",
                "field_complete_for_ranking": "False",
                "distance_bucket": "sprint_lt_400",
                "winner_box_bucket": "outside_6_plus",
                "source_bucket": "DATA_MISSING",
            },
        ],
    }


def _subpacket(path: Path, *, write_claim: bool = False) -> Path:
    payload = {
        "schema_version": "fixture",
        "status": "READY_FOR_QUEUE_REVIEW",
        "report_only": True,
        "label_write_approved": False,
        "label_writes_performed": write_claim,
        "model_promotion_allowed": False,
        "selected_policy_key": "latest_source_agreement_strict_train_ranks_51_100",
        "queues": {
            "canonical_identity_ready": {
                "candidate_count": 1,
                "strict_protocol_train_candidate_count": 1,
                "rows_with_manual_review_flags": 0,
                "approval_required_before_label_write": True,
                "approval_request_possible_after_manual_review": True,
                "projected_if_queue_reviewed_and_explicitly_approved": {
                    "current_official_safe_races": 217,
                    "current_strict_protocol_official_train_races": 82,
                    "projected_official_safe_races": 218,
                    "projected_strict_protocol_official_train_races": 83,
                    "second_holdout_untouched": True,
                },
                "packet_rows": [
                    {
                        "identity_key": "2025-07-01|TEST|R01",
                        "selected_source_race_id": "TEST_2025-07-01_1",
                        "selected_metadata_race_id": "TEST_2025-07-01_1",
                        "projected_strict_protocol_train_if_approved": True,
                        "policy_rank": 51,
                        "required_action": "verify_official_result_distance_and_full_finish_order_before_any_label_write",
                    }
                ],
            },
            "identity_lookup_required": {
                "candidate_count": 1,
                "strict_protocol_train_candidate_count": 1,
                "rows_with_manual_review_flags": 1,
                "approval_required_before_label_write": True,
                "approval_request_possible_after_manual_review": False,
                "projected_if_queue_reviewed_and_explicitly_approved": {
                    "current_official_safe_races": 217,
                    "current_strict_protocol_official_train_races": 82,
                    "projected_official_safe_races": 218,
                    "projected_strict_protocol_official_train_races": 83,
                    "second_holdout_untouched": True,
                },
                "packet_rows": [
                    {
                        "identity_key": "AMBIGUOUS|FIXTURE",
                        "selected_source_race_id": "LOOKUP_2025-07-01_0",
                        "projected_strict_protocol_train_if_approved": True,
                        "policy_rank": 52,
                        "required_action": "resolve_official_race_number_before_manual_label_review_or_approval_request",
                    }
                ],
            },
        },
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
            winner_source text,
            results_status text
        )
        """
    )
    conn.execute(
        """
        create table dog_race_data (
            race_id text,
            dog_name text,
            data_source text
        )
        """
    )
    conn.executemany(
        "insert into race_metadata values (?,?,?,?,?,?)",
        [
            ("TEST_2025-07-01_1", "2025-07-01", "TEST", 1, None, "complete"),
            ("MISS_2025-07-02_2", "2025-07-02", "MISS", 2, "thedogs_official", "complete"),
        ],
    )
    conn.executemany(
        "insert into dog_race_data values (?,?,?)",
        [
            ("TEST_2025-07-01_1", "Alpha", "current_db"),
            ("TEST_2025-07-01_1", "Beta", None),
            ("MISS_2025-07-02_2", "Gamma", "thedogs_official"),
        ],
    )
    conn.commit()
    conn.close()
    return path


def test_source_label_packet_crosswalks_rolling_misses_to_manual_queue(tmp_path: Path):
    packet = build_source_label_packet(
        rolling_analysis=_analysis(),
        manual_subpacket_paths=[_subpacket(tmp_path / "subpackets.json")],
        db_path=_db(tmp_path / "greyhound.sqlite"),
    )

    assert packet["status"] == "REPORT_ONLY_ROLLING_SOURCE_LABEL_EXPANSION_PACKET"
    assert packet["writes_performed"]["db_write"] is False
    assert packet["summary"]["rolling_source_bucket_missing_count"] == 2
    assert packet["summary"]["unique_canonical_ready_strict_train_candidates"] == 1
    assert packet["summary"]["unique_identity_lookup_strict_train_candidates"] == 1
    assert packet["summary"]["rolling_queue_overlap_count"] == 1
    assert packet["summary"]["rolling_top1_miss_queue_overlap_count"] == 1
    crosswalk = {row["race_id"]: row for row in packet["crosswalk_rows"]}
    assert crosswalk["TEST_2025-07-01_1"]["best_queue_key"] == "canonical_identity_ready"
    assert crosswalk["TEST_2025-07-01_1"]["source_gap_status"] == "WINNER_SOURCE_MISSING"
    assert crosswalk["MISS_2025-07-02_2"]["source_gap_status"] == (
        "SOURCE_BUCKET_MISSING_IN_STRATIFIED_ANALYSIS"
    )


def test_source_label_packet_fails_closed_on_write_claim(tmp_path: Path):
    packet = build_source_label_packet(
        rolling_analysis=_analysis(),
        manual_subpacket_paths=[_subpacket(tmp_path / "subpackets.json", write_claim=True)],
    )

    assert packet["status"] == "REPORT_ONLY_ROLLING_SOURCE_LABEL_EXPANSION_PACKET_WITH_FAILURES"
    assert any("packet_flag_not_false" in failure for failure in packet["failures"])


def test_source_label_packet_cli_writes_outputs(tmp_path: Path):
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps(_analysis(), sort_keys=True), encoding="utf-8")
    output_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/source_label"

    status = main(
        [
            "--rolling-stratified-analysis",
            str(analysis_path),
            "--manual-subpacket",
            str(_subpacket(tmp_path / "subpackets.json")),
            "--db",
            str(_db(tmp_path / "greyhound.sqlite")),
            "--output-dir",
            str(output_dir),
        ],
        root=tmp_path,
    )

    assert status == 0
    assert (output_dir / "no_box_pairwise_rolling_source_label_expansion_packet.json").exists()
    assert (output_dir / "rolling_source_label_crosswalk.csv").exists()
    assert (output_dir / "official_review_queue_summary.csv").exists()
    assert (output_dir / "SUMMARY.md").exists()


def test_source_label_packet_cli_rejects_output_outside_repo(tmp_path: Path):
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps(_analysis(), sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        main(
            [
                "--rolling-stratified-analysis",
                str(analysis_path),
                "--manual-subpacket",
                str(_subpacket(tmp_path / "subpackets.json")),
                "--output-dir",
                str(tmp_path.parent / "source-label-outside"),
            ],
            root=tmp_path,
        )


def test_source_label_packet_cli_rejects_output_outside_artifacts(tmp_path: Path):
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps(_analysis(), sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="output_dir_must_be_under_artifacts"):
        main(
            [
                "--rolling-stratified-analysis",
                str(analysis_path),
                "--manual-subpacket",
                str(_subpacket(tmp_path / "subpackets.json")),
                "--output-dir",
                str(tmp_path / "reports/source-label"),
            ],
            root=tmp_path,
        )
