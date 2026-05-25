import json
import sqlite3
from pathlib import Path

from scripts import populate_race_metadata_from_upcoming as seed


def _write_snapshot(root: Path, *, race_date: str, venue: str, race_id: str) -> Path:
    path = root / race_date / venue / "snapshot.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "prediction_snapshot_v1",
                "race_id": race_id,
                "race_date": race_date,
                "venue": venue,
                "race_number": 1,
                "jump_time": "14:30",
                "jump_datetime": f"{race_date}T14:30:00+10:00",
                "source_file_path": None,
                "lifecycle_status": "upcoming_not_jumped",
                "snapshot_state": "pre_jump_feature_freeze",
                "is_pre_jump_snapshot": True,
                "prediction_timestamp": f"{race_date}T10:00:00",
                "feature_freeze_timestamp": f"{race_date}T10:00:00",
                "model_version": "test-model",
                "predictions": [
                    {"dog_name": "Alpha", "box_number": 1, "win_prob_norm": 0.4},
                    {"dog_name": "Bravo", "box_number": 2, "win_prob_norm": 0.3},
                    {"dog_name": "Charlie", "box_number": 3, "win_prob_norm": 0.2},
                    {"dog_name": "Delta", "box_number": 4, "win_prob_norm": 0.1},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_snapshot_limited_metadata_seed_preserves_snapshot_race_id(tmp_path):
    snapshot_root = tmp_path / "snapshots"
    target = _write_snapshot(
        snapshot_root,
        race_date="2026-05-25",
        venue="WAR",
        race_id="Race 1 - WAR - 2026-05-25",
    )
    _write_snapshot(
        snapshot_root,
        race_date="2026-05-26",
        venue="TRA",
        race_id="Race 1 - TRA - 2026-05-26",
    )
    db_path = tmp_path / "metadata.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE race_metadata (
            race_id TEXT PRIMARY KEY,
            venue TEXT,
            race_number INTEGER,
            race_date TEXT,
            grade TEXT,
            distance TEXT,
            field_size INTEGER,
            extraction_timestamp TEXT,
            data_source TEXT
        )
        """
    )

    files = seed._snapshot_files(snapshot_root, "2026-05-25")
    assert files == [target]
    pm, enrich, skip_reason = seed._meta_from_snapshot(target)
    assert skip_reason is None
    inserted, _updated = seed.upsert_race_meta(conn, pm, enrich)
    conn.commit()

    rows = conn.execute(
        "SELECT race_id, venue, race_number, race_date, field_size, data_source FROM race_metadata"
    ).fetchall()
    conn.close()

    assert inserted is True
    assert rows == [
        (
            "Race 1 - WAR - 2026-05-25",
            "WARRNAMBOOL",
            1,
            "2026-05-25",
            4,
            "frozen_prediction_snapshot",
        )
    ]
