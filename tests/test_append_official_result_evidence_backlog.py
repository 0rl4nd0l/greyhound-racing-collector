import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

from scripts import append_official_result_evidence_backlog as backlog
from scripts import autonomous_official_result_capture as capture


def _official_artifact_rows(
    *,
    race_id: str = "Race 1 - WPK - 2026-06-10",
    race_number: int = 1,
    source_url: str = (
        "https://www.thedogs.com.au/racing/"
        "wentworth-park/2026-06-10/1/test-race?trial=false"
    ),
) -> dict[str, list[dict[str, object]]]:
    generated_at = datetime.fromisoformat("2026-06-10T14:20:00+10:00")
    return capture.build_artifact_rows(
        {
            "scope": {
                "date": "2026-06-10",
                "db_path": "/tmp/labels.sqlite",
            },
            "ingested": [
                {
                    "race_id": race_id,
                    "venue": "WPK",
                    "race_number": race_number,
                    "race_date": "2026-06-10",
                    "race_time": "14:00",
                    "start_datetime": "2026-06-10T14:00:00+10:00",
                    "source": "thedogs_official",
                    "source_url": source_url,
                    "status": "resulted",
                    "winner_name": "Alpha",
                    "winner_box": 1,
                    "box_order": [1, 2],
                    "participant_source": "shadow_run_predictions",
                    "positions": [
                        {
                            "box_number": 1,
                            "finish_position": 1,
                            "dog_name": "Alpha",
                        },
                        {
                            "box_number": 2,
                            "finish_position": 2,
                            "dog_name": "Bravo",
                        },
                    ],
                    "participants": [
                        {"box_number": 1, "dog_name": "Alpha"},
                        {"box_number": 2, "dog_name": "Bravo"},
                    ],
                }
            ],
            "failed": [],
            "skipped": [],
        },
        generated_at=generated_at,
    )


def _write_official_artifact_dir(
    root: Path,
    *,
    name: str,
    rows: dict[str, list[dict[str, object]]],
) -> Path:
    artifact_dir = root / name
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "official_result_races.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows["race_rows"]),
        encoding="utf-8",
    )
    (artifact_dir / "official_result_runners.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows["runner_rows"]),
        encoding="utf-8",
    )
    (artifact_dir / "official_result_quarantine.jsonl").write_text("", encoding="utf-8")
    return artifact_dir


def _patch_roots(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(backlog, "ROOT", root)
    monkeypatch.setattr(capture, "ROOT", root)
    monkeypatch.setattr(
        capture,
        "DEFAULT_EVIDENCE_ROOT",
        root / "artifacts/full_evidence_orchestration_20260525",
    )


def test_append_backlog_appends_multiple_artifacts_when_lock_free(tmp_path, monkeypatch):
    _patch_roots(monkeypatch, tmp_path)
    db_path = tmp_path / "labels.sqlite"
    with sqlite3.connect(db_path):
        pass
    artifact_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    race1 = _write_official_artifact_dir(
        artifact_root,
        name="autonomous_official_result_capture_race1",
        rows=_official_artifact_rows(),
    )
    race2 = _write_official_artifact_dir(
        artifact_root,
        name="autonomous_official_result_capture_race2",
        rows=_official_artifact_rows(
            race_id="Race 2 - WPK - 2026-06-10",
            race_number=2,
            source_url=(
                "https://www.thedogs.com.au/racing/"
                "wentworth-park/2026-06-10/2/test-race?trial=false"
            ),
        ),
    )
    output_dir = (
        artifact_root
        / "official_result_evidence_append_backlog_lock_free"
    )

    result = backlog.main(
        [
            "--artifact-dir",
            str(race1),
            "--artifact-dir",
            str(race2),
            "--db",
            str(db_path),
            "--output-dir",
            str(output_dir),
            "--execute-db-ingest",
            "--require-lock-free",
            "--lock-path",
            str(tmp_path / "missing.lock"),
        ]
    )

    assert result == 0
    report = json.loads(
        (output_dir / "official_result_evidence_append_backlog_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["final_status"] == "APPENDED_OFFICIAL_RESULT_EVIDENCE_BACKLOG"
    assert report["db_write_performed"] is True
    assert report["inserted_race_rows"] == 2
    assert report["inserted_runner_rows"] == 4
    assert report["status_counts"] == {"APPENDED_OFFICIAL_RESULT_EVIDENCE": 2}
    assert report["label_write_performed"] is False
    assert report["shared_lock_status"]["status"] == "acquired_by_backlog_append"
    assert report["shared_lock_release"] == {
        "released": True,
        "reason": "released_by_owner",
    }
    with sqlite3.connect(db_path) as conn:
        race_count = conn.execute(
            f"SELECT COUNT(*) FROM {capture.OFFICIAL_RESULT_EVIDENCE_RACES_TABLE}"
        ).fetchone()[0]
        runner_count = conn.execute(
            f"SELECT COUNT(*) FROM {capture.OFFICIAL_RESULT_EVIDENCE_RUNNERS_TABLE}"
        ).fetchone()[0]
        label_table_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM sqlite_master
            WHERE type = 'table'
              AND name IN ('race_metadata', 'dog_race_data')
            """
        ).fetchone()[0]
    assert race_count == 2
    assert runner_count == 4
    assert label_table_count == 0


def test_append_backlog_discovers_child_capture_dirs_from_parent_root(tmp_path, monkeypatch):
    _patch_roots(monkeypatch, tmp_path)
    db_path = tmp_path / "labels.sqlite"
    with sqlite3.connect(db_path):
        pass
    artifact_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    _write_official_artifact_dir(
        artifact_root,
        name="autonomous_official_result_capture_race1",
        rows=_official_artifact_rows(),
    )
    _write_official_artifact_dir(
        artifact_root,
        name="autonomous_official_result_capture_race2",
        rows=_official_artifact_rows(
            race_id="Race 2 - WPK - 2026-06-10",
            race_number=2,
            source_url=(
                "https://www.thedogs.com.au/racing/"
                "wentworth-park/2026-06-10/2/test-race?trial=false"
            ),
        ),
    )
    output_dir = (
        artifact_root
        / "official_result_evidence_append_backlog_parent_discovery"
    )

    result = backlog.main(
        [
            "--artifact-dir",
            str(artifact_root),
            "--db",
            str(db_path),
            "--output-dir",
            str(output_dir),
            "--execute-db-ingest",
            "--require-lock-free",
            "--lock-path",
            str(tmp_path / "missing.lock"),
        ]
    )

    assert result == 0
    report = json.loads(
        (output_dir / "official_result_evidence_append_backlog_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["input_artifact_count"] == 1
    assert report["artifact_count"] == 2
    assert report["processed_count"] == 2
    assert report["artifact_discovery"][0]["mode"] == "recursive_parent_discovery"
    assert report["artifact_discovery"][0]["discovered_child_artifact_count"] == 2
    assert report["final_status"] == "APPENDED_OFFICIAL_RESULT_EVIDENCE_BACKLOG"
    assert report["inserted_race_rows"] == 2
    assert report["inserted_runner_rows"] == 4
    assert report["shared_lock_status"]["status"] == "acquired_by_backlog_append"
    assert report["shared_lock_release"] == {
        "released": True,
        "reason": "released_by_owner",
    }
    with sqlite3.connect(db_path) as conn:
        race_count = conn.execute(
            f"SELECT COUNT(*) FROM {capture.OFFICIAL_RESULT_EVIDENCE_RACES_TABLE}"
        ).fetchone()[0]
        runner_count = conn.execute(
            f"SELECT COUNT(*) FROM {capture.OFFICIAL_RESULT_EVIDENCE_RUNNERS_TABLE}"
        ).fetchone()[0]
    assert race_count == 2
    assert runner_count == 4


def test_append_backlog_blocks_all_artifacts_on_live_lock(tmp_path, monkeypatch):
    _patch_roots(monkeypatch, tmp_path)
    db_path = tmp_path / "labels.sqlite"
    with sqlite3.connect(db_path):
        pass
    artifact_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    artifact_dir = _write_official_artifact_dir(
        artifact_root,
        name="autonomous_official_result_capture_race1",
        rows=_official_artifact_rows(),
    )
    lock_path = tmp_path / "shadow_autopilot.lock"
    lock_path.write_text(
        json.dumps(
            {
                "schema_version": "shadow_autopilot_daemon_lock_v1",
                "run_id": "test_live_lock",
                "pid": os.getpid(),
            }
        ),
        encoding="utf-8",
    )
    output_dir = (
        artifact_root
        / "official_result_evidence_append_backlog_lock_blocked"
    )

    result = backlog.main(
        [
            "--artifact-dir",
            str(artifact_dir),
            "--db",
            str(db_path),
            "--output-dir",
            str(output_dir),
            "--execute-db-ingest",
            "--require-lock-free",
            "--lock-path",
            str(lock_path),
        ]
    )

    assert result == 0
    report = json.loads(
        (output_dir / "official_result_evidence_append_backlog_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["final_status"] == "BLOCKED_SHARED_LOCK_HELD"
    assert report["db_write_performed"] is False
    assert report["inserted_race_rows"] == 0
    assert report["inserted_runner_rows"] == 0
    assert report["status_counts"] == {"BLOCKED_SHARED_LOCK_HELD": 1}
    assert report["items"][0]["valid_race_rows"] == 1
    assert report["items"][0]["valid_runner_rows"] == 2
    assert report["items"][0]["label_write_performed"] is False
    assert report["shared_lock_status"]["status"] == "present_live_pid"
    with sqlite3.connect(db_path) as conn:
        table_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM sqlite_master
            WHERE type = 'table'
              AND name = ?
            """,
            (capture.OFFICIAL_RESULT_EVIDENCE_RACES_TABLE,),
        ).fetchone()[0]
    assert table_count == 0


def test_append_backlog_requires_lock_path_when_lock_free_required(tmp_path, monkeypatch):
    _patch_roots(monkeypatch, tmp_path)
    db_path = tmp_path / "labels.sqlite"
    with sqlite3.connect(db_path):
        pass
    artifact_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    artifact_dir = _write_official_artifact_dir(
        artifact_root,
        name="autonomous_official_result_capture_race1",
        rows=_official_artifact_rows(),
    )
    output_dir = artifact_root / "official_result_evidence_append_backlog_missing_lock_path"

    result = backlog.main(
        [
            "--artifact-dir",
            str(artifact_dir),
            "--db",
            str(db_path),
            "--output-dir",
            str(output_dir),
            "--execute-db-ingest",
            "--require-lock-free",
        ]
    )

    assert result == 0
    report = json.loads(
        (output_dir / "official_result_evidence_append_backlog_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["final_status"] == "BLOCKED_SHARED_LOCK_HELD"
    assert report["db_write_performed"] is False
    assert report["shared_lock_status"]["status"] == "lock_path_missing_required"
    assert report["shared_lock_status"]["write_allowed"] is False
    with sqlite3.connect(db_path) as conn:
        table_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM sqlite_master
            WHERE type = 'table'
              AND name = ?
            """,
            (capture.OFFICIAL_RESULT_EVIDENCE_RACES_TABLE,),
        ).fetchone()[0]
    assert table_count == 0
