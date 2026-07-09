import csv
import json
from pathlib import Path

import pytest

from scripts.build_single_race_official_repair_plan import build_repair_plan, main


def _gap_packet(path: Path) -> Path:
    payload = {
        "schema_version": "single_race_official_gap_review_packet_v1",
        "status": "REPORT_ONLY_SINGLE_RACE_OFFICIAL_GAP_REVIEW",
        "report_only": True,
        "writes_performed": {
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
        },
        "source_evidence": {
            "lookup_packet_matched": "/tmp/lookup.json",
            "db": "/tmp/greyhound.db",
        },
        "failure_review_row": {
            "source_url": "https://www.thedogs.com.au/racing/test/2025-07-01/1/results",
        },
        "summary": {
            "race_id": "TEST_2025-07-01_1",
            "official_runner_count": 3,
            "db_runner_count": 2,
            "label_write_ready": False,
        },
        "db_metadata": {
            "race_id": "TEST_2025-07-01_1",
            "results_status": "complete",
            "winner_name": "Alpha",
            "winner_source": None,
            "field_size": 2,
            "actual_field_size": None,
            "url": None,
            "data_source": None,
        },
        "official_rows": [
            {"dog_name": "Alpha", "box_number": 1, "finish_position": 1, "name_key": "alpha"},
            {"dog_name": "Beta", "box_number": 2, "finish_position": 2, "name_key": "beta"},
            {"dog_name": "Gamma", "box_number": 3, "finish_position": 3, "name_key": "gamma"},
        ],
        "db_rows": [
            {
                "dog_name": "1. Alpha",
                "box_number": 4,
                "finish_position": 1,
                "placing": None,
                "scraped_finish_position": None,
                "data_source": None,
            },
            {
                "dog_name": "2. Beta",
                "box_number": 2,
                "finish_position": 3,
                "placing": None,
                "scraped_finish_position": None,
                "data_source": None,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _runner_csv(path: Path) -> Path:
    rows = [
        {
            "race_id": "TEST_2025-07-01_1",
            "official_finish_position": "1",
            "official_box_number": "1",
            "official_dog_name": "Alpha",
            "name_key": "alpha",
            "db_matched": "True",
            "db_box_number": "4",
            "db_finish_position": "1",
            "gap_flags": "db_box_differs_from_official",
        },
        {
            "race_id": "TEST_2025-07-01_1",
            "official_finish_position": "2",
            "official_box_number": "2",
            "official_dog_name": "Beta",
            "name_key": "beta",
            "db_matched": "True",
            "db_box_number": "2",
            "db_finish_position": "3",
            "gap_flags": "db_finish_differs_from_official",
        },
        {
            "race_id": "TEST_2025-07-01_1",
            "official_finish_position": "3",
            "official_box_number": "3",
            "official_dog_name": "Gamma",
            "name_key": "gamma",
            "db_matched": "False",
            "gap_flags": "missing_db_runner",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_single_race_repair_plan_builds_no_write_operations(tmp_path: Path):
    packet = build_repair_plan(
        gap_review_packet_path=_gap_packet(tmp_path / "gap.json"),
        runner_review_csv_path=_runner_csv(tmp_path / "review.csv"),
    )

    assert packet["schema_version"] == "single_race_official_repair_plan_v1"
    assert packet["status"] == "REPORT_ONLY_SINGLE_RACE_OFFICIAL_REPAIR_PLAN"
    assert packet["safe_to_write_now"] is False
    assert packet["writes_performed"]["db_write"] is False
    assert packet["summary"]["dog_row_update_candidate_count"] == 2
    assert packet["summary"]["missing_runner_insert_candidate_count"] == 1
    assert packet["summary"]["metadata_update_candidate_count"] == 1
    assert packet["summary"]["field_size_policy_decision_required"] is True

    alpha = next(row for row in packet["dog_row_update_candidates"] if row["name_key"] == "alpha")
    assert alpha["selector"] == {"race_id": "TEST_2025-07-01_1", "dog_name": "1. Alpha"}
    assert alpha["before"]["box_number"] == 4
    assert alpha["after"]["box_number"] == 1
    assert "box_number" in alpha["changed_fields"]
    assert "WHERE race_id=? AND dog_name=?" in alpha["write_sql_shape_if_later_approved"]

    missing = packet["missing_runner_insert_candidates"][0]
    assert missing["official_dog_name"] == "Gamma"
    assert missing["after"]["box_number"] == 3
    assert "schema_default_policy_required_for_missing_runner_insert" in missing["blockers"]

    metadata = packet["metadata_update_candidate"]
    assert metadata["after_patch"]["winner_source"] == "thedogs_official"
    assert metadata["after_patch"]["actual_field_size"] == 3
    assert metadata["after_patch"]["url"].startswith("https://www.thedogs.com.au/")
    assert metadata["deferred_policy_candidates"][0]["field"] == "field_size"


def test_single_race_repair_plan_cli_writes_outputs(tmp_path: Path):
    output_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/repair_plan"

    exit_code = main(
        [
            "--gap-review-packet",
            str(_gap_packet(tmp_path / "gap.json")),
            "--runner-review-csv",
            str(_runner_csv(tmp_path / "review.csv")),
            "--output-dir",
            str(output_dir),
        ],
        root=tmp_path,
    )

    assert exit_code == 0
    assert (output_dir / "single_race_official_repair_plan.json").exists()
    assert (output_dir / "single_race_official_repair_operations.csv").exists()
    assert "No DB rows" in (output_dir / "SUMMARY.md").read_text(encoding="utf-8")


def test_single_race_repair_plan_cli_rejects_output_outside_repo(tmp_path: Path):
    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        main(
            [
                "--gap-review-packet",
                str(_gap_packet(tmp_path / "gap.json")),
                "--runner-review-csv",
                str(_runner_csv(tmp_path / "review.csv")),
                "--output-dir",
                str(tmp_path.parent / "repair-plan-outside"),
            ],
            root=tmp_path,
        )


def test_single_race_repair_plan_cli_rejects_output_outside_artifacts(tmp_path: Path):
    with pytest.raises(ValueError, match="output_dir_must_be_under_artifacts"):
        main(
            [
                "--gap-review-packet",
                str(_gap_packet(tmp_path / "gap.json")),
                "--runner-review-csv",
                str(_runner_csv(tmp_path / "review.csv")),
                "--output-dir",
                str(tmp_path / "reports/repair-plan"),
            ],
            root=tmp_path,
        )
