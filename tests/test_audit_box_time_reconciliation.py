import sqlite3

from scripts.audit_box_time_reconciliation import build_report


def _create_schema(db_path):
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE race_metadata (
            race_id TEXT PRIMARY KEY,
            venue TEXT,
            race_number INTEGER,
            race_date TEXT,
            data_source TEXT
        );
        CREATE TABLE dog_race_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT NOT NULL,
            dog_name TEXT NOT NULL,
            dog_clean_name TEXT,
            box_number INTEGER,
            individual_time TEXT,
            winning_time TEXT,
            best_time REAL,
            data_source TEXT
        );
        CREATE TABLE csv_dog_history_staging (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT,
            venue TEXT,
            race_number INTEGER,
            race_date TEXT,
            dog_name TEXT,
            dog_clean_name TEXT,
            box_number INTEGER,
            individual_time TEXT,
            data_source TEXT
        );
        """
    )
    connection.commit()
    return connection


def test_audit_reports_data_missing_when_production_timing_has_no_strict_box_join(tmp_path):
    db_path = tmp_path / "audit.db"
    connection = _create_schema(db_path)
    connection.executescript(
        """
        INSERT INTO race_metadata (race_id, venue, race_number, race_date, data_source)
        VALUES
          ('TIME_RACE', 'SHEP', NULL, '2026-01-01', 'embedded_form_guide'),
          ('BOX_RACE', 'SHEP', 1, '2026-01-01', 'official');
        INSERT INTO dog_race_data
          (race_id, dog_name, dog_clean_name, box_number, individual_time, data_source)
        VALUES
          ('TIME_RACE', 'Fast Dog', 'Fast Dog', NULL, '22.31', 'embedded_form_guide'),
          ('BOX_RACE', '1. Fast Dog', 'Fast Dog', 1, NULL, 'thedogs_official');
        INSERT INTO csv_dog_history_staging
          (race_id, venue, race_number, race_date, dog_name, dog_clean_name,
           box_number, individual_time, data_source)
        VALUES
          ('STAGE_RACE', 'SHEP', 1, '2026-01-01', '1. Fast Dog', 'Fast Dog',
           1, '22.31', 'csv_stage');
        """
    )
    connection.commit()
    connection.close()

    report = build_report(db_path)

    assert report["verdict"] == "DATA_MISSING"
    assert report["safe_recovery_count"] == 0
    assert report["staging_self_contained_safe_rows"] == 1
    assert "No strict join recovered boxes" in report["blocked_reason"]


def test_audit_flags_safe_candidate_when_production_timing_matches_unique_box(tmp_path):
    db_path = tmp_path / "audit.db"
    connection = _create_schema(db_path)
    connection.executescript(
        """
        INSERT INTO race_metadata (race_id, venue, race_number, race_date, data_source)
        VALUES
          ('TIME_RACE', 'SHEP', 1, '2026-01-01', 'embedded_form_guide'),
          ('BOX_RACE', 'SHEP', 1, '2026-01-01', 'official');
        INSERT INTO dog_race_data
          (race_id, dog_name, dog_clean_name, box_number, individual_time, data_source)
        VALUES
          ('TIME_RACE', 'Fast Dog', 'Fast Dog', NULL, '22.31', 'embedded_form_guide'),
          ('BOX_RACE', '1. Fast Dog', 'Fast Dog', 1, NULL, 'thedogs_official');
        """
    )
    connection.commit()
    connection.close()

    report = build_report(db_path)
    canonical_join = next(
        item
        for item in report["join_audits"]
        if item["name"] == "production_timing_to_production_box_by_canonical_identity"
    )

    assert report["verdict"] == "SAFE_RECOVERY_CANDIDATE_REQUIRES_FEATURE_PIPELINE_REVIEW"
    assert report["safe_recovery_count"] == 1
    assert canonical_join["safe_recoverable_timing_rows"] == 1
    assert canonical_join["safe_box_band_counts"] == {"inside": 1}


def test_audit_rejects_ambiguous_box_matches(tmp_path):
    db_path = tmp_path / "audit.db"
    connection = _create_schema(db_path)
    connection.executescript(
        """
        INSERT INTO race_metadata (race_id, venue, race_number, race_date, data_source)
        VALUES
          ('TIME_RACE', 'SHEP', 1, '2026-01-01', 'embedded_form_guide'),
          ('BOX_RACE_A', 'SHEP', 1, '2026-01-01', 'official'),
          ('BOX_RACE_B', 'SHEP', 1, '2026-01-01', 'official');
        INSERT INTO dog_race_data
          (race_id, dog_name, dog_clean_name, box_number, individual_time, data_source)
        VALUES
          ('TIME_RACE', 'Fast Dog', 'Fast Dog', NULL, '22.31', 'embedded_form_guide'),
          ('BOX_RACE_A', '1. Fast Dog', 'Fast Dog', 1, NULL, 'thedogs_official'),
          ('BOX_RACE_B', '2. Fast Dog', 'Fast Dog', 2, NULL, 'thedogs_official');
        """
    )
    connection.commit()
    connection.close()

    report = build_report(db_path)
    canonical_join = next(
        item
        for item in report["join_audits"]
        if item["name"] == "production_timing_to_production_box_by_canonical_identity"
    )

    assert report["verdict"] == "DATA_MISSING"
    assert report["safe_recovery_count"] == 0
    assert canonical_join["matched_timing_rows"] == 1
    assert canonical_join["safe_recoverable_timing_rows"] == 0
    assert canonical_join["ambiguous_timing_rows"] == 1
