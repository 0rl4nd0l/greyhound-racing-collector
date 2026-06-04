import json
import sqlite3
from pathlib import Path

from scripts.build_official_reverify_label_preflight import build_preflight_packet, main


def _make_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "greyhound.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE race_metadata (
                race_id TEXT PRIMARY KEY,
                venue TEXT,
                race_number INTEGER,
                race_date TEXT,
                results_status TEXT,
                winner_name TEXT,
                winner_source TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE dog_race_data (
                race_id TEXT,
                box_number INTEGER,
                finish_position INTEGER,
                placing INTEGER,
                scraped_finish_position TEXT,
                data_source TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO race_metadata
                (race_id, venue, race_number, race_date, results_status, winner_name, winner_source)
            VALUES ('GEE_2025-07-22_5', 'GEE', 5, '2025-07-22', 'pending', NULL, NULL)
            """
        )
        conn.execute(
            """
            INSERT INTO race_metadata
                (race_id, venue, race_number, race_date, results_status, winner_name, winner_source)
            VALUES ('AP_K_2025-07-01_2', 'AP_K', 2, '2025-07-01', 'resulted', 'Fast One', 'thedogs_official')
            """
        )
        conn.execute(
            """
            INSERT INTO dog_race_data
                (race_id, box_number, finish_position, data_source)
            VALUES ('AP_K_2025-07-01_2', 4, 1, 'thedogs_official')
            """
        )
    return db_path


def _lookup_packet(path: Path) -> Path:
    payload = {
        "schema_version": "official_reverify_lookup_dry_run_v1",
        "status": "REPORT_ONLY",
        "writes_performed": {
            "db_write": False,
            "label_write": False,
            "official_fetch": True,
            "snapshot_mutation": False,
            "manifest_mutation": False,
            "model_training": False,
            "registry_mutation": False,
            "promotion": False,
            "betting_decision": False,
        },
        "results": [
            {
                "legacy_race_id": "GEE_5_22_July_2025",
                "legacy_runner_rows": 2,
                "lookup_key": {
                    "venue": "GEE",
                    "race_number": 5,
                    "race_date": "2025-07-22",
                },
                "lookup_status": "OFFICIAL_RESULT_PARSED",
                "result_parse_ready": True,
                "label_write_ready": True,
                "skip_reasons": [],
                "source_url": "https://www.thedogs.com.au/racing/geelong/2025-07-22/5/results",
                "positions": [
                    {"box_number": 4, "finish_position": 1},
                    {"box_number": 8, "finish_position": 2},
                ],
                "terminal_statuses": [],
            },
            {
                "legacy_race_id": "ap_k_2025-07-01_2",
                "legacy_runner_rows": 1,
                "lookup_key": {
                    "venue": "AP_K",
                    "race_number": 2,
                    "race_date": "2025-07-01",
                },
                "lookup_status": "OFFICIAL_RESULT_PARSED",
                "result_parse_ready": True,
                "label_write_ready": True,
                "skip_reasons": [],
                "source_url": "https://www.thedogs.com.au/racing/angle-park/2025-07-01/2/results",
                "positions": [{"box_number": 4, "finish_position": 1}],
                "terminal_statuses": [],
            },
            {
                "legacy_race_id": "MISSING_2025-07-01_9",
                "legacy_runner_rows": 1,
                "lookup_key": {
                    "venue": "MISSING",
                    "race_number": 9,
                    "race_date": "2025-07-01",
                },
                "lookup_status": "OFFICIAL_RESULT_PARSED",
                "result_parse_ready": True,
                "label_write_ready": True,
                "skip_reasons": [],
                "source_url": "https://www.thedogs.com.au/racing/missing/2025-07-01/9/results",
                "positions": [{"box_number": 1, "finish_position": 1}],
                "terminal_statuses": [],
            },
            {
                "legacy_race_id": "SKIPPED",
                "legacy_runner_rows": 1,
                "lookup_key": {"venue": "GEE", "race_number": 6, "race_date": "2025-07-22"},
                "label_write_ready": False,
                "skip_reasons": ["official_http_404"],
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_reverify_label_preflight_checks_db_without_writes(tmp_path):
    packet = build_preflight_packet(
        lookup_packet_path=_lookup_packet(tmp_path / "lookup.json"),
        db_path=_make_db(tmp_path),
    )

    assert packet["schema_version"] == "official_reverify_label_preflight_v1"
    assert packet["status"] == "PREFLIGHT_READY_WITH_BLOCKERS"
    assert packet["writes_performed"] == {
        "db_write": False,
        "label_write": False,
        "official_fetch": False,
        "snapshot_mutation": False,
        "manifest_mutation": False,
        "model_training": False,
        "registry_mutation": False,
        "promotion": False,
        "betting_decision": False,
    }
    assert packet["db_state"]["quick_check"] == "ok"
    assert packet["summary"]["lookup_label_write_ready_count"] == 3
    assert packet["summary"]["preflight_ready_count"] == 0
    assert packet["summary"]["blocked_count"] == 3
    assert packet["summary"]["skipped_not_lookup_ready_count"] == 1
    assert packet["summary"]["blocker_counts"] == {
        "db_has_existing_official_rows": 1,
        "db_has_existing_result_rows": 1,
        "db_dog_rows_missing": 2,
        "race_metadata_missing": 1,
        "race_metadata_not_pending": 1,
        "race_metadata_winner_present": 1,
        "race_metadata_winner_source_present": 1,
    }

    by_legacy = {row["legacy_race_id"]: row for row in packet["candidate_preflight"]}
    assert by_legacy["GEE_5_22_July_2025"]["preflight_status"] == "BLOCKED"
    assert by_legacy["GEE_5_22_July_2025"]["resolved_db_race_id"] == "GEE_2025-07-22_5"
    assert by_legacy["GEE_5_22_July_2025"]["blockers"] == ["db_dog_rows_missing"]
    assert by_legacy["GEE_5_22_July_2025"]["row_alignment"] == {
        "official_box_numbers": [4, 8],
        "existing_box_numbers": [],
        "box_set_matches_official": False,
        "missing_existing_boxes": [4, 8],
        "extra_existing_boxes": [],
    }
    assert by_legacy["ap_k_2025-07-01_2"]["preflight_status"] == "BLOCKED"
    assert by_legacy["ap_k_2025-07-01_2"]["row_alignment"] == {
        "official_box_numbers": [4],
        "existing_box_numbers": [4],
        "box_set_matches_official": True,
        "missing_existing_boxes": [],
        "extra_existing_boxes": [],
    }
    assert by_legacy["MISSING_2025-07-01_9"]["blockers"] == [
        "db_dog_rows_missing",
        "race_metadata_missing",
    ]


def test_reverify_label_preflight_cli_writes_json_and_markdown(tmp_path):
    output = tmp_path / "preflight.json"
    report = tmp_path / "preflight.md"

    exit_code = main(
        [
            "--lookup-packet",
            str(_lookup_packet(tmp_path / "lookup.json")),
            "--db",
            str(_make_db(tmp_path)),
            "--output",
            str(output),
            "--report",
            str(report),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"]["preflight_ready_count"] == 0
    assert report.exists()
