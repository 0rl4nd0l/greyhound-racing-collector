import json
import sqlite3
from pathlib import Path

from scripts.build_official_identity_reconciliation_packet import (
    build_identity_reconciliation_packet,
)


def _make_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "greyhound.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE dog_race_data (
                race_id TEXT,
                dog_name TEXT,
                box_number INTEGER,
                finish_position INTEGER,
                placing INTEGER,
                scraped_finish_position TEXT,
                data_source TEXT
            )
            """
        )
        rows = [
            (1, "5. Gold Loki", 4),
            (2, "4. No Idea", 4),
            (4, "7. Victa Monty", 2),
            (5, "2. Kota Lambai", 1),
            (7, "8. South King", 5),
            (8, "1. Flying Embers", 2),
        ]
        for box_number, dog_name, finish_position in rows:
            conn.execute(
                """
                INSERT INTO dog_race_data
                    (race_id, dog_name, box_number, finish_position)
                VALUES ('AP_K_2025-07-21_10', ?, ?, ?)
                """,
                (dog_name, box_number, finish_position),
            )
    return db_path


def _lookup(path: Path) -> Path:
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
                "legacy_race_id": "ap_k_2025-07-21_10",
                "lookup_key": {
                    "race_date": "2025-07-21",
                    "race_number": 10,
                    "venue": "AP_K",
                },
                "lookup_status": "OFFICIAL_RESULT_PARSED",
                "result_parse_ready": True,
                "label_write_ready": True,
                "official_runner_rows": [
                    {"box_number": 7, "dog_name": "Victa Monty", "finish_position": 1},
                    {"box_number": 2, "dog_name": "Kota Lambai", "finish_position": 2},
                    {"box_number": 5, "dog_name": "Gold Loki", "finish_position": 3},
                    {"box_number": 4, "dog_name": "No Idea", "finish_position": 4},
                    {"box_number": 1, "dog_name": "Flying Embers", "finish_position": 5},
                    {"box_number": 8, "dog_name": "South King", "finish_position": 6},
                ],
                "positions": [],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_identity_packet_detects_same_names_different_boxes(tmp_path):
    packet = build_identity_reconciliation_packet(
        lookup_packet_path=_lookup(tmp_path / "lookup.json"),
        db_path=_make_db(tmp_path),
    )

    assert packet["schema_version"] == "official_identity_reconciliation_v1"
    assert packet["status"] == "REPORT_ONLY_IDENTITY_RECONCILIATION_REQUIRED"
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
    assert packet["summary"] == {
        "lookup_results_seen": 1,
        "identity_ready_count": 1,
        "exact_identity_and_position_match_count": 0,
        "same_names_different_boxes_count": 1,
        "name_set_mismatch_count": 0,
        "db_rows_missing_count": 0,
        "official_names_missing_count": 0,
    }
    race = packet["races"][0]
    assert race["race_id"] == "AP_K_2025-07-21_10"
    assert race["status"] == "BOX_IDENTITY_DRIFT"
    assert race["all_official_names_found_in_db"] is True
    assert race["box_mismatch_count"] == 6
    assert race["finish_position_mismatch_count"] == 5
    assert race["matches_by_name"][0] == {
        "official_dog_name": "Victa Monty",
        "db_dog_name": "7. Victa Monty",
        "official_box_number": 7,
        "db_box_number": 4,
        "official_finish_position": 1,
        "db_finish_position": 2,
        "box_matches": False,
        "finish_position_matches": False,
    }
