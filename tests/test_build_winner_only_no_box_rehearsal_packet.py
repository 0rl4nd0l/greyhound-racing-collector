import json
from pathlib import Path

import pytest

from scripts.build_winner_only_no_box_rehearsal_packet import (
    FORBIDDEN_ROW_FIELDS,
    build_winner_only_no_box_rehearsal_packet,
    write_rehearsal_outputs,
)


def _winner_only_packet(*, confirmed: bool = True) -> dict:
    status = (
        "winner_only_no_box_research_candidate_metadata_confirmed"
        if confirmed
        else "blocked_winner_only_metadata_missing_or_differs"
    )
    return {
        "schema_version": "official_winner_only_label_eligibility_packet_v1",
        "report_only": True,
        "write_ready": False,
        "label_write_approved": False,
        "records": [
            {
                "race_id": "TEST_2025-08-24_3",
                "legacy_race_id": "TEST_2025-08-24_3",
                "identity_key": "2025-08-24|TEST|R03",
                "status": status,
                "winner_only_precheck_winner_matches_without_full_finish": True,
                "winner_only_no_box_research_candidate": confirmed,
                "partial_field_winner_only_no_box_research_candidate": False,
                "actual_win_no_box_research_candidate": confirmed,
                "full_name_identity_ready": True,
                "name_set_result": "exact_match",
                "primary_bucket": "box_identity_drift",
                "strict_full_finish_label_candidate": False,
                "official_safe_label_candidate": False,
                "label_write_approved": False,
                "requires_no_box_feature_policy": confirmed,
                "requires_actual_win_only_target": confirmed,
                "forbidden_for_top3_or_finish_order_training": True,
                "forbidden_for_box_feature_training": True,
                "field_scope": "complete_name_set_box_drift" if confirmed else "not_eligible",
                "field_complete_for_ranking": confirmed,
                "name_set_subset_scope": {
                    "db_name_subset_of_official": False,
                    "db_only_name_count": 0,
                    "official_only_name_count": 0,
                    "db_not_in_official": [],
                    "official_not_in_db": [],
                    "duplicate_db_name_keys": [],
                    "duplicate_official_name_keys": [],
                    "partial_field_skip_reasons_allowed": False,
                },
                "lookup_skip_reasons": [],
                "winner_alignment": {
                    "official_winner_name": "Winner Dog",
                    "metadata_winner_name": "Winner Dog" if confirmed else "Different Dog",
                    "db_winner_name": "1. Winner Dog",
                    "official_winner_matches_db_winner": True,
                    "official_winner_matches_metadata_winner": confirmed,
                },
                "finish_signature_alignment": {
                    "official_signature_matches_db_signature": False,
                    "official_finish_signature": "box:1=1|box:2=2|box:3=3",
                    "db_finish_signature": "box:1=1|box:4=3|box:8=2",
                },
            }
        ],
    }


def _p1_recovery_packet() -> dict:
    payload = _winner_only_packet(confirmed=False)
    record = payload["records"][0]
    record.update(
        {
            "status": "blocked_name_set_not_exact",
            "full_name_identity_ready": False,
            "name_set_result": "mismatch",
            "lookup_skip_reasons": [
                "official_positions_incomplete_for_legacy_runner_count",
            ],
            "winner_alignment": {
                "official_winner_name": "Winner Dog",
                "metadata_winner_name": "Winner Dog",
                "db_winner_name": "1. Winner Dog",
                "official_winner_matches_db_winner": True,
                "official_winner_matches_metadata_winner": True,
            },
        }
    )
    return payload


def _partial_field_winner_only_packet() -> dict:
    payload = _winner_only_packet(confirmed=True)
    record = payload["records"][0]
    record.update(
        {
            "status": "partial_field_winner_only_no_box_research_candidate_metadata_confirmed",
            "winner_only_no_box_research_candidate": False,
            "partial_field_winner_only_no_box_research_candidate": True,
            "actual_win_no_box_research_candidate": True,
            "full_name_identity_ready": False,
            "name_set_result": "mismatch",
            "field_scope": "partial_db_name_subset_of_official_finishers",
            "field_complete_for_ranking": False,
            "lookup_skip_reasons": [
                "official_positions_incomplete_for_legacy_runner_count",
            ],
            "name_set_subset_scope": {
                "db_name_subset_of_official": True,
                "db_only_name_count": 0,
                "official_only_name_count": 1,
                "db_not_in_official": [],
                "official_not_in_db": ["extra dog"],
                "duplicate_db_name_keys": [],
                "duplicate_official_name_keys": [],
                "partial_field_skip_reasons_allowed": True,
            },
        }
    )
    return payload


def _terminal_exclusion_winner_only_packet() -> dict:
    payload = _partial_field_winner_only_packet()
    record = payload["records"][0]
    record.update(
        {
            "field_scope": "partial_db_name_subset_after_nonstarter_terminal_exclusions",
            "lookup_skip_reasons": [
                "official_positions_incomplete_for_legacy_runner_count",
                "official_terminal_statuses_present",
            ],
            "name_set_subset_scope": {
                "db_name_subset_of_official": False,
                "db_name_subset_after_terminal_exclusions": True,
                "db_only_name_count": 1,
                "official_only_name_count": 1,
                "db_not_in_official": ["scratched dog"],
                "official_not_in_db": ["extra dog"],
                "duplicate_db_name_keys": [],
                "duplicate_official_name_keys": [],
                "db_only_terminal_exclusion_count": 1,
                "db_only_terminal_exclusions": [
                    {
                        "db_box_number": 5,
                        "db_dog_name": "5. Scratched Dog",
                        "db_name_prefix_box": 5,
                        "dog_name_key": "scratched dog",
                        "terminal_box_number": 5,
                        "terminal_status": "SCR",
                    }
                ],
                "terminal_statuses": [{"box_number": 5, "status": "SCR"}],
                "partial_field_skip_reasons_allowed": True,
            },
        }
    )
    return payload


def _terminal_complete_winner_only_packet() -> dict:
    payload = _terminal_exclusion_winner_only_packet()
    record = payload["records"][0]
    record.update(
        {
            "status": "winner_only_no_box_research_candidate_metadata_confirmed",
            "winner_only_no_box_research_candidate": True,
            "partial_field_winner_only_no_box_research_candidate": False,
            "terminal_exclusion_complete_field_winner_only_no_box_research_candidate": True,
            "full_name_identity_ready": False,
            "field_scope": "complete_name_set_after_nonstarter_terminal_exclusions",
            "field_complete_for_ranking": True,
            "name_set_subset_scope": {
                "db_name_subset_of_official": False,
                "db_name_subset_after_terminal_exclusions": False,
                "db_name_set_complete_after_terminal_exclusions": True,
                "db_only_name_count": 1,
                "official_only_name_count": 0,
                "db_not_in_official": ["scratched dog"],
                "official_not_in_db": [],
                "duplicate_db_name_keys": [],
                "duplicate_official_name_keys": [],
                "db_only_terminal_exclusion_count": 1,
                "db_only_terminal_exclusions": [
                    {
                        "db_box_number": 5,
                        "db_dog_name": "5. Scratched Dog",
                        "db_name_prefix_box": 5,
                        "dog_name_key": "scratched dog",
                        "terminal_box_number": 5,
                        "terminal_status": "SCR",
                    }
                ],
                "terminal_statuses": [{"box_number": 5, "status": "SCR"}],
                "partial_field_skip_reasons_allowed": True,
            },
        }
    )
    return payload


def _identity_packet() -> dict:
    return {
        "schema_version": "official_identity_reconciliation_full_v1",
        "records": [
            {
                "race_id": "TEST_2025-08-24_3",
                "legacy_race_id": "TEST_2025-08-24_3",
                "lookup_key": {
                    "race_date": "2025-08-24",
                    "venue": "TEST",
                    "race_number": 3,
                },
                "matches_by_name": [
                    {
                        "dog_name_key": "winner dog",
                        "official_dog_name": "Winner Dog",
                        "db_dog_name": "1. Winner Dog",
                        "official_box_number": 1,
                        "db_box_number": 1,
                        "official_finish_position": 1,
                        "db_result_position": 1,
                    },
                    {
                        "dog_name_key": "place dog",
                        "official_dog_name": "Place Dog",
                        "db_dog_name": "8. Place Dog",
                        "official_box_number": 2,
                        "db_box_number": 8,
                        "official_finish_position": 2,
                        "db_result_position": 3,
                    },
                    {
                        "dog_name_key": "third dog",
                        "official_dog_name": "Third Dog",
                        "db_dog_name": "4. Third Dog",
                        "official_box_number": 3,
                        "db_box_number": 4,
                        "official_finish_position": 3,
                        "db_result_position": 2,
                    },
                ],
            }
        ],
    }


def _partial_field_identity_packet() -> dict:
    packet = _identity_packet()
    packet["records"][0]["matches_by_name"] = [
        {
            "dog_name_key": "winner dog",
            "official_dog_name": "Winner Dog",
            "db_dog_name": "1. Winner Dog",
            "official_box_number": 1,
            "db_box_number": 7,
            "official_finish_position": 1,
            "db_result_position": 1,
        },
        {
            "dog_name_key": "place dog",
            "official_dog_name": "Place Dog",
            "db_dog_name": "2. Place Dog",
            "official_box_number": 2,
            "db_box_number": 8,
            "official_finish_position": 2,
            "db_result_position": 2,
        },
    ]
    return packet


def test_winner_only_rehearsal_materializes_no_box_actual_win_rows(tmp_path: Path):
    packet, rows = build_winner_only_no_box_rehearsal_packet(
        winner_only_packet=_winner_only_packet(),
        identity_packet=_identity_packet(),
        expected_candidates=1,
    )

    assert packet["status"] == "REPORT_ONLY_SINGLE_CANDIDATE_CONTRACT_REHEARSAL"
    assert packet["label_write_approved"] is False
    assert packet["model_training_performed"] is False
    assert packet["summary"]["confirmed_winner_only_candidate_count"] == 1
    assert packet["summary"]["materialized_rows"] == 3
    assert packet["summary"]["actual_win_positive_rows"] == 1
    assert packet["summary"]["actual_win_negative_rows"] == 2
    assert packet["summary"]["no_box_row_policy_pass"] is True
    assert packet["summary"]["can_evaluate_model"] is False
    assert packet["summary"]["recovery_queue_count"] == 0
    assert packet["recovery_queue"]["summary"]["recovery_queue_count"] == 0
    assert packet["summary"]["sample_size_gate"]["minimum_smoke_actual_win_eval"][
        "additional_confirmed_candidates_needed"
    ] == 19

    assert {row["dog_name_key"]: row["actual_win"] for row in rows} == {
        "place dog": 0,
        "third dog": 0,
        "winner dog": 1,
    }
    for row in rows:
        assert not (set(row) & FORBIDDEN_ROW_FIELDS)
        assert row["box_features_allowed"] is False
        assert row["finish_order_labels_allowed"] is False
        assert row["top3_labels_allowed"] is False
        assert row["official_safe_label_candidate"] is False
        assert row["label_write_approved"] is False

    output_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/rehearsal"
    write_rehearsal_outputs(output_dir, packet, rows, root=tmp_path)
    written = json.loads(
        (output_dir / "winner_only_no_box_actual_win_rehearsal_packet.json").read_text(
            encoding="utf-8"
        )
    )
    assert written["schema_version"] == "winner_only_no_box_actual_win_rehearsal_v1"
    assert (output_dir / "winner_only_no_box_actual_win_rows.jsonl").exists()
    assert (output_dir / "winner_only_no_box_actual_win_rows.csv").exists()
    assert (output_dir / "winner_only_no_box_recovery_queue.json").exists()
    assert (output_dir / "winner_only_no_box_recovery_queue.csv").exists()
    assert (output_dir / "SUMMARY.md").exists()


def test_winner_only_rehearsal_relative_output_resolves_under_repo_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    packet, rows = build_winner_only_no_box_rehearsal_packet(
        winner_only_packet=_winner_only_packet(),
        identity_packet=_identity_packet(),
        expected_candidates=1,
    )
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    relative_output = Path("artifacts/full_evidence_orchestration_20260525/rehearsal")
    write_rehearsal_outputs(relative_output, packet, rows, root=tmp_path)

    assert (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525/rehearsal/SUMMARY.md"
    ).exists()
    assert not (
        cwd / "artifacts/full_evidence_orchestration_20260525/rehearsal/SUMMARY.md"
    ).exists()


def test_winner_only_rehearsal_rejects_output_outside_repo(tmp_path: Path):
    packet, rows = build_winner_only_no_box_rehearsal_packet(
        winner_only_packet=_winner_only_packet(),
        identity_packet=_identity_packet(),
        expected_candidates=1,
    )

    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        write_rehearsal_outputs(
            tmp_path.parent / "outside-rehearsal",
            packet,
            rows,
            root=tmp_path,
        )


def test_winner_only_rehearsal_rejects_output_outside_artifacts(tmp_path: Path):
    packet, rows = build_winner_only_no_box_rehearsal_packet(
        winner_only_packet=_winner_only_packet(),
        identity_packet=_identity_packet(),
        expected_candidates=1,
    )

    with pytest.raises(ValueError, match="output_dir_must_be_under_artifacts"):
        write_rehearsal_outputs(tmp_path / "reports/rehearsal", packet, rows, root=tmp_path)


def test_winner_only_rehearsal_materializes_partial_field_actual_win_rows():
    packet, rows = build_winner_only_no_box_rehearsal_packet(
        winner_only_packet=_partial_field_winner_only_packet(),
        identity_packet=_partial_field_identity_packet(),
        expected_candidates=1,
    )

    assert packet["status"] == "REPORT_ONLY_SINGLE_CANDIDATE_CONTRACT_REHEARSAL"
    assert packet["summary"]["confirmed_winner_only_candidate_count"] == 1
    assert packet["summary"]["complete_field_confirmed_winner_only_candidate_count"] == 0
    assert packet["summary"]["partial_field_confirmed_winner_only_candidate_count"] == 1
    assert packet["summary"]["race_grouped_ranking_ready_candidate_count"] == 0
    assert packet["summary"]["can_evaluate_race_grouped_model"] is False
    assert packet["summary"]["recovery_queue_count"] == 0
    assert packet["summary"]["materialized_rows"] == 2
    assert {row["dog_name_key"]: row["actual_win"] for row in rows} == {
        "place dog": 0,
        "winner dog": 1,
    }
    for row in rows:
        assert row["candidate_kind"] == "partial_field"
        assert row["field_scope"] == "partial_db_name_subset_of_official_finishers"
        assert row["field_complete_for_ranking"] is False
        assert row["race_grouped_actual_win_ranking_allowed"] is False
        assert not (set(row) & FORBIDDEN_ROW_FIELDS)
    race_summary = packet["race_summaries"][0]
    assert race_summary["field_complete_for_ranking"] is False
    assert race_summary["partial_field_official_only_names"] == ["extra dog"]
    assert (
        packet["summary"]["sample_size_gate"]["minimum_ranking_model_comparison"][
            "current_confirmed_races"
        ]
        == 0
    )


def test_winner_only_rehearsal_excludes_partial_field_terminal_db_only_rows():
    packet, rows = build_winner_only_no_box_rehearsal_packet(
        winner_only_packet=_terminal_exclusion_winner_only_packet(),
        identity_packet=_partial_field_identity_packet(),
        expected_candidates=1,
    )

    assert packet["status"] == "REPORT_ONLY_SINGLE_CANDIDATE_CONTRACT_REHEARSAL"
    assert packet["summary"]["confirmed_winner_only_candidate_count"] == 1
    assert packet["summary"]["partial_field_confirmed_winner_only_candidate_count"] == 1
    assert packet["summary"]["materialized_rows"] == 2
    assert {row["dog_name_key"]: row["actual_win"] for row in rows} == {
        "place dog": 0,
        "winner dog": 1,
    }
    assert "scratched dog" not in {row["dog_name_key"] for row in rows}
    for row in rows:
        assert row["candidate_kind"] == "partial_field"
        assert row["field_scope"] == "partial_db_name_subset_after_nonstarter_terminal_exclusions"
        assert row["field_complete_for_ranking"] is False
        assert row["race_grouped_actual_win_ranking_allowed"] is False
        assert not (set(row) & FORBIDDEN_ROW_FIELDS)
    race_summary = packet["race_summaries"][0]
    assert race_summary["partial_field_db_only_terminal_exclusions"] == [
        {
            "db_box_number": 5,
            "db_dog_name": "5. Scratched Dog",
            "db_name_prefix_box": 5,
            "dog_name_key": "scratched dog",
            "terminal_box_number": 5,
            "terminal_status": "SCR",
        }
    ]


def test_winner_only_rehearsal_allows_complete_field_terminal_exclusion_rows():
    packet, rows = build_winner_only_no_box_rehearsal_packet(
        winner_only_packet=_terminal_complete_winner_only_packet(),
        identity_packet=_partial_field_identity_packet(),
        expected_candidates=1,
    )

    assert packet["status"] == "REPORT_ONLY_SINGLE_CANDIDATE_CONTRACT_REHEARSAL"
    assert packet["summary"]["confirmed_winner_only_candidate_count"] == 1
    assert packet["summary"]["complete_field_confirmed_winner_only_candidate_count"] == 1
    assert packet["summary"]["partial_field_confirmed_winner_only_candidate_count"] == 0
    assert packet["summary"]["race_grouped_ranking_ready_candidate_count"] == 1
    assert packet["summary"]["materialized_rows"] == 2
    assert {row["dog_name_key"]: row["actual_win"] for row in rows} == {
        "place dog": 0,
        "winner dog": 1,
    }
    assert "scratched dog" not in {row["dog_name_key"] for row in rows}
    for row in rows:
        assert row["candidate_kind"] == "complete_field"
        assert row["field_scope"] == "complete_name_set_after_nonstarter_terminal_exclusions"
        assert row["field_complete_for_ranking"] is True
        assert row["race_grouped_actual_win_ranking_allowed"] is True
        assert not (set(row) & FORBIDDEN_ROW_FIELDS)
    race_summary = packet["race_summaries"][0]
    assert race_summary["field_complete_for_ranking"] is True
    assert race_summary["partial_field_db_only_terminal_exclusions"] == [
        {
            "db_box_number": 5,
            "db_dog_name": "5. Scratched Dog",
            "db_name_prefix_box": 5,
            "dog_name_key": "scratched dog",
            "terminal_box_number": 5,
            "terminal_status": "SCR",
        }
    ]
    assert (
        packet["summary"]["sample_size_gate"]["minimum_ranking_model_comparison"][
            "current_confirmed_races"
        ]
        == 1
    )


def test_winner_only_rehearsal_blocks_metadata_unconfirmed_candidates():
    packet, rows = build_winner_only_no_box_rehearsal_packet(
        winner_only_packet=_winner_only_packet(confirmed=False),
        identity_packet=_identity_packet(),
        expected_candidates=0,
    )

    assert packet["status"] == "REPORT_ONLY_NO_CONFIRMED_WINNER_ONLY_CANDIDATES"
    assert packet["summary"]["confirmed_winner_only_candidate_count"] == 0
    assert packet["summary"]["blocked_or_non_candidate_records_seen"] == 1
    assert packet["summary"]["materialized_rows"] == 0
    assert rows == []
    blocked = packet["blocked_or_non_candidate_records"][0]
    assert "metadata_confirmed_status_missing" in blocked["blockers"]
    assert "not_winner_only_no_box_research_candidate" in blocked["blockers"]
    recovery = packet["recovery_queue"]
    assert recovery["summary"]["recovery_lane_counts"] == {
        "metadata_winner_recheck_candidate": 1
    }
    assert recovery["records"][0]["priority"] == "P3"


def test_winner_only_recovery_queue_prioritizes_name_set_and_parser_repairs():
    packet, rows = build_winner_only_no_box_rehearsal_packet(
        winner_only_packet=_p1_recovery_packet(),
        identity_packet=_identity_packet(),
        expected_candidates=0,
    )

    assert rows == []
    recovery = packet["recovery_queue"]
    assert recovery["summary"]["recovery_queue_count"] == 1
    assert recovery["summary"]["p1_name_set_and_parser_repair_candidates"] == 1
    assert (
        packet["summary"]["next_recommended_gate"]
        == "repair_official_name_set_or_terminal_status_parse_for_P1_candidates"
    )
    record = recovery["records"][0]
    assert record["priority"] == "P1"
    assert record["recovery_lane"] == "name_set_and_parser_repair_candidate"
    assert record["winner_alignment_summary"]["official_winner_matches_db_winner"] is True
    assert record["winner_alignment_summary"]["official_winner_matches_metadata_winner"] is True
    assert record["lookup_skip_reasons"] == [
        "official_positions_incomplete_for_legacy_runner_count"
    ]
