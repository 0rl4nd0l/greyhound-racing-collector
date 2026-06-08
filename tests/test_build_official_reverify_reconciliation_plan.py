import json
from pathlib import Path

from scripts.build_official_reverify_reconciliation_plan import (
    build_reconciliation_plan,
    main,
)


def _preflight(path: Path) -> Path:
    payload = {
        "schema_version": "official_reverify_label_preflight_v1",
        "status": "PREFLIGHT_READY_WITH_BLOCKERS",
        "writes_performed": {
            "db_write": False,
            "label_write": False,
            "official_fetch": False,
            "snapshot_mutation": False,
            "manifest_mutation": False,
            "model_training": False,
            "registry_mutation": False,
            "promotion": False,
            "betting_decision": False,
        },
        "candidate_preflight": [
            {
                "legacy_race_id": "exact_update",
                "resolved_db_race_id": "AP_K_2025-07-08_3",
                "lookup_key": {"venue": "AP_K", "race_date": "2025-07-08", "race_number": 3},
                "blockers": ["db_has_existing_result_rows"],
                "row_alignment": {
                    "box_set_matches_official": True,
                    "official_box_numbers": [1, 2, 4, 5, 7, 8],
                    "existing_box_numbers": [1, 2, 4, 5, 7, 8],
                },
            },
            {
                "legacy_race_id": "exact_complete",
                "resolved_db_race_id": "AP_K_2025-07-21_10",
                "lookup_key": {"venue": "AP_K", "race_date": "2025-07-21", "race_number": 10},
                "blockers": [
                    "db_has_existing_result_rows",
                    "race_metadata_not_pending",
                    "race_metadata_winner_present",
                ],
                "row_alignment": {
                    "box_set_matches_official": True,
                    "official_box_numbers": [1, 2, 4, 5, 7, 8],
                    "existing_box_numbers": [1, 2, 4, 5, 7, 8],
                },
            },
            {
                "legacy_race_id": "mismatch",
                "resolved_db_race_id": "GEE_2025-07-22_5",
                "lookup_key": {"venue": "GEE", "race_date": "2025-07-22", "race_number": 5},
                "blockers": ["db_has_existing_result_rows"],
                "row_alignment": {
                    "box_set_matches_official": False,
                    "official_box_numbers": [1, 2, 3, 4, 5, 6, 7, 8],
                    "existing_box_numbers": [1, 4, 5, 7, 8],
                },
            },
            {
                "legacy_race_id": "missing",
                "resolved_db_race_id": None,
                "lookup_key": {"venue": "AP_K", "race_date": "2025-07-01", "race_number": 6},
                "blockers": ["db_dog_rows_missing", "race_metadata_missing"],
                "row_alignment": {
                    "box_set_matches_official": False,
                    "official_box_numbers": [1, 2, 4, 5, 7, 8],
                    "existing_box_numbers": [],
                },
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_reconciliation_plan_classifies_blocked_preflight_lanes(tmp_path):
    packet = build_reconciliation_plan(
        preflight_packet_path=_preflight(tmp_path / "preflight.json"),
        sample_limit=2,
    )

    assert packet["schema_version"] == "official_reverify_reconciliation_plan_v1"
    assert packet["status"] == "REPORT_ONLY_RECONCILIATION_REQUIRED"
    assert packet["writes_performed"] == {
        "db_write": False,
        "label_write": False,
        "metadata_write": False,
        "official_fetch": False,
        "snapshot_mutation": False,
        "manifest_mutation": False,
        "model_training": False,
        "registry_mutation": False,
        "promotion": False,
        "betting_decision": False,
    }
    assert packet["summary"]["lane_counts"] == {
        "existing_rows_box_set_mismatch": 1,
        "existing_rows_exact_box_set_update_rehearsal_candidate": 1,
        "existing_rows_exact_box_set_metadata_complete_review": 1,
        "metadata_and_dog_rows_missing": 1,
    }
    assert packet["summary"]["safe_to_write_now_count"] == 0
    assert packet["summary"]["first_executable_lane"] == (
        "existing_rows_exact_box_set_update_rehearsal_candidate"
    )
    assert packet["lanes"]["existing_rows_exact_box_set_update_rehearsal_candidate"][
        "count"
    ] == 1
    assert packet["lanes"]["existing_rows_box_set_mismatch"]["recommended_action"] == (
        "manual_or_source-backed_runner_reconciliation_required"
    )


def test_reconciliation_plan_cli_writes_json_and_report(tmp_path):
    output = tmp_path / "plan.json"
    report = tmp_path / "plan.md"

    exit_code = main(
        [
            "--preflight-packet",
            str(_preflight(tmp_path / "preflight.json")),
            "--output",
            str(output),
            "--report",
            str(report),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"]["safe_to_write_now_count"] == 0
    assert report.exists()
