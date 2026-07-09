import json
from pathlib import Path

import pytest

from scripts.build_official_reverify_queue_window_packet import (
    _safe_output_path,
    build_window_packet,
    write_window_outputs,
)


def _manual_row(index: int, *, race_number: int | None = None, ambiguous: bool = False) -> dict:
    race_date = f"2025-08-{index:02d}"
    identity_key = (
        f"AMBIGUOUS|TEST202508{index:02d}450MGRADE5"
        if ambiguous
        else f"{race_date}|TEST|R{race_number:02d}"
    )
    return {
        "identity_key": identity_key,
        "race_date": race_date,
        "venue": "TEST",
        "race_number": race_number,
        "consensus_sub_bucket": "CURRENT_DB_CORROBORATED_BY_BACKUP",
        "consensus_sub_bucket_tags": ["CURRENT_DB_CORROBORATED_BY_BACKUP"],
        "target_distance": 450,
        "winner_name": "Alpha",
        "winner_key": "box:1",
        "runner_count": 3,
        "source_count": 2,
        "matching_source_count": 2,
        "matching_source_roles": ["current_db", "db_backup"],
        "all_source_roles": ["current_db", "db_backup"],
        "full_finish_signature": "box:1=1|box:2=2|box:3=3",
        "full_finish_signature_agreement_status": "PASS",
        "distance_values": [450],
        "distance_agreement_status": "PASS",
        "source_agreement_status": "PASS",
        "selected_source_role": "current_db",
        "selected_source_path": "/fixture/current.sqlite",
        "source_paths": ["/fixture/current.sqlite", "/fixture/backup.sqlite"],
        "manual_verification_required": True,
        "selected_source_race_id": f"TEST_{race_date}_{race_number or 0}",
        "selected_metadata_race_id": f"TEST_{race_date}_{race_number or 0}",
        "selected_metadata_grade": "Grade 5",
        "selected_metadata_results_status": "complete",
        "selected_metadata_data_source": "fixture",
        "selected_source_initial_bucket": "SCRAPED_RESULT_ONLY",
        "selected_source_initial_reasons": ["fixture"],
        "source_observation_ids": [f"obs-{index}-a", f"obs-{index}-b"],
    }


def test_build_window_packet_slices_next_ranks_and_keeps_no_write_outputs(tmp_path):
    evaluation_dir = tmp_path / "eval"
    output_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/window"
    evaluation_dir.mkdir()
    rows = [
        _manual_row(1, race_number=1),
        _manual_row(2, race_number=2),
        _manual_row(3, race_number=None, ambiguous=True),
        _manual_row(4, race_number=4),
        _manual_row(5, race_number=5),
        _manual_row(6, race_number=6),
    ]
    (evaluation_dir / "official_label_expansion_report.json").write_text(
        json.dumps(
            {
                "schema_version": "fixture",
                "current_official_safe_races": 10,
                "strict_protocol_official_train_races": 6,
                "distance_known_manual_verification_queue": rows,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (evaluation_dir / "temporal_holdout_report.json").write_text(
        json.dumps(
            {
                "shared_holdout_protocol": {
                    "status": "PASS",
                    "holdout_min_date": "2025-09-01",
                    "calibration_holdout_race_ids": ["2025-09-01|TEST|R01"],
                    "second_holdout_race_ids": ["2025-09-02|TEST|R01"],
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    bundle = build_window_packet(
        evaluation_dir=evaluation_dir,
        output_dir=output_dir,
        selection_policy="latest_source_agreement_strict_train",
        start_rank=3,
        limit=2,
    )

    summary = bundle["summary"]
    assert summary["candidate_count"] == 2
    assert summary["eligible_candidates_for_policy"] == 6
    assert summary["queue_counts"] == {
        "canonical_identity_ready": 1,
        "identity_lookup_required": 1,
        "other_manual_flags": 0,
    }
    assert summary["parse_ready_count"] == 1
    assert summary["parse_blocked_count"] == 1
    assert summary["writes_performed"]["label_write"] is False
    assert "mutate_db" in summary["forbidden_without_explicit_approval"]

    packet_rows = bundle["manual_verification_packet"]["packet_rows"]
    assert [row["race_date"] for row in packet_rows] == ["2025-08-04", "2025-08-03"]
    assert [row["policy_rank"] for row in packet_rows] == [3, 4]
    assert [row["window_rank"] for row in packet_rows] == [1, 2]
    assert packet_rows[0]["identity_resolution_status"] == (
        "CANONICAL_IDENTITY_READY_FOR_MANUAL_CONFIRMATION"
    )
    assert packet_rows[1]["identity_resolution_status"] == "NEEDS_OFFICIAL_RACE_NUMBER_LOOKUP"

    write_window_outputs(output_dir, bundle)
    assert (output_dir / "official_label_manual_verification_window_summary.json").exists()
    assert (output_dir / "official_label_manual_verification_window_packet.json").exists()
    assert (output_dir / "official_label_manual_verification_window_subpackets.json").exists()
    assert (output_dir / "official_label_manual_verification_window_reverify_queue_report.json").exists()
    reverify_jsonl = output_dir / summary["reverify_queue_jsonl"]
    assert reverify_jsonl.exists()
    assert len(reverify_jsonl.read_text(encoding="utf-8").splitlines()) == 2


def test_queue_window_rejects_absolute_output_outside_repo(tmp_path: Path) -> None:
    output_dir = tmp_path / "outside" / "artifacts" / "full_evidence_orchestration_20260525" / "window"

    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        _safe_output_path(output_dir, root=tmp_path / "repo")


def test_queue_window_rejects_in_repo_non_artifact_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="output_dir_must_be_under_artifacts"):
        _safe_output_path(Path("reports/window"), root=tmp_path)
