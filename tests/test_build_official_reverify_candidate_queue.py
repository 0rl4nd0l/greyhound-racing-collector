import json
from pathlib import Path

from scripts.build_official_reverify_candidate_queue import build_queue, main


def _packet(path: Path) -> Path:
    payload = {
        "schema_version": "legacy_label_verification_packet_v1",
        "status": "REPORT_ONLY",
        "race_classifications": [
            {
                "race_id": "DAPT_2_22_August_2025",
                "legacy_db_path": "/tmp/legacy.sqlite",
                "source": "completed_race_update",
                "classification": "result_like_reverify_candidate",
                "legacy_runner_rows": 10,
                "metadata": {
                    "race_date": "22 August 2025",
                    "winner_name": "6. Rilla Dream",
                },
                "verification": {
                    "status": "OFFICIAL_REFERENCE_MISSING",
                    "legacy_rows": 10,
                    "official_reference_rows": 0,
                    "mismatches": [],
                },
            },
            {
                "race_id": "R001_2025-02-18_AP_K",
                "legacy_db_path": "/tmp/legacy.sqlite",
                "source": "navigator_results",
                "classification": "result_like_reverify_candidate",
                "legacy_runner_rows": 4,
                "metadata": {
                    "race_date": "2025-02-18",
                    "winner_name": "COOL IT MISS",
                },
                "verification": {
                    "status": "OFFICIAL_REFERENCE_MISSING",
                    "legacy_rows": 4,
                    "official_reference_rows": 0,
                    "mismatches": [],
                },
            },
            {
                "race_id": "ap_k_2025-07-01_1",
                "legacy_db_path": "/tmp/legacy.sqlite",
                "source": "enhanced_processor_with_results",
                "classification": "result_like_reverify_candidate",
                "legacy_runner_rows": 8,
                "metadata": {"race_date": "2025-07-01", "winner_name": "A"},
                "verification": {"status": "OFFICIAL_REFERENCE_MISSING"},
            },
            {
                "race_id": "NOT_PARSEABLE",
                "legacy_db_path": "/tmp/legacy.sqlite",
                "source": "enhanced_processor_with_results",
                "classification": "result_like_reverify_candidate",
                "legacy_runner_rows": 6,
                "metadata": {"race_date": "Unknown", "winner_name": "B"},
                "verification": {"status": "OFFICIAL_REFERENCE_MISSING"},
            },
            {
                "race_id": "APFR_2024-12-27_520m_5",
                "classification": "embedded_history_only",
                "legacy_runner_rows": 1,
                "metadata": {"race_date": "2024-12-27"},
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_candidate_queue_normalizes_parse_ready_result_like_races(tmp_path):
    report = build_queue(
        verification_packet_path=_packet(tmp_path / "verification.json"),
        queue_output_path=tmp_path / "queue.jsonl",
    )

    assert report["schema_version"] == "official_reverify_candidate_queue_v1"
    assert report["status"] == "REPORT_ONLY"
    assert report["writes_performed"] == {
        "db_write": False,
        "label_write": False,
        "official_fetch": False,
        "snapshot_mutation": False,
        "model_training": False,
        "registry_mutation": False,
    }
    assert report["summary"]["candidate_count"] == 4
    assert report["summary"]["parse_ready_count"] == 3
    assert report["summary"]["parse_blocked_count"] == 1
    assert report["summary"]["source_counts"] == {
        "completed_race_update": 1,
        "enhanced_processor_with_results": 2,
        "navigator_results": 1,
    }

    queued = [
        json.loads(line)
        for line in (tmp_path / "queue.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [item["legacy_race_id"] for item in queued] == [
        "DAPT_2_22_August_2025",
        "R001_2025-02-18_AP_K",
        "ap_k_2025-07-01_1",
        "NOT_PARSEABLE",
    ]
    assert queued[0]["lookup_key"] == {
        "venue": "DAPT",
        "race_number": 2,
        "race_date": "2025-08-22",
    }
    assert queued[1]["lookup_key"] == {
        "venue": "AP_K",
        "race_number": 1,
        "race_date": "2025-02-18",
    }
    assert queued[2]["lookup_key"] == {
        "venue": "AP_K",
        "race_number": 1,
        "race_date": "2025-07-01",
    }
    assert queued[3]["lookup_status"] == "PARSE_BLOCKED"
    assert queued[3]["blockers"] == ["legacy_race_id_not_parseable"]


def test_candidate_queue_cli_writes_packet_and_jsonl(tmp_path):
    output = tmp_path / "queue_report.json"
    queue = tmp_path / "queue.jsonl"

    exit_code = main(
        [
            "--verification-packet",
            str(_packet(tmp_path / "verification.json")),
            "--queue-output",
            str(queue),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "REPORT_ONLY"
    assert payload["summary"]["candidate_count"] == 4
    assert queue.exists()
