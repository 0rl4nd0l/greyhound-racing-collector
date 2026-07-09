import sqlite3

import pytest

from scripts.audit_stage2_target_metadata_coverage import audit_rows, safe_output_dir


def _connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE race_metadata (
            race_id TEXT,
            venue TEXT,
            race_number INTEGER,
            race_date TEXT,
            grade TEXT,
            distance TEXT,
            data_source TEXT,
            url TEXT
        );
        """
    )
    return connection


def test_target_metadata_audit_marks_exact_race_number_context_recoverable():
    connection = _connection()
    connection.execute(
        """
        INSERT INTO race_metadata
            (race_id, venue, race_number, race_date, grade, distance, data_source, url)
        VALUES
            ('Race 1 - TEST - 2026-06-01', 'TEST', 1, '2026-06-01',
             'Grade 5', '450', 'canonical_pre_race_page', 'https://example.test/r1')
        """
    )

    report = audit_rows(
        [
            {
                "race_id": "Race 1 - TEST - 2026-06-01",
                "race_date": "2026-06-01",
                "venue": "TEST",
                "target_distance_safe": "",
                "target_grade_safe": "",
            }
        ],
        connection,
    )

    assert report["verdict"] == "SAFE_REPAIR_AVAILABLE"
    assert report["safe_recoverable_rows_from_existing_sources"] == 1
    assert report["db_exact_race_number_status_counts"] == {
        "safe_exact_race_number_context_available": 1
    }


def test_target_metadata_audit_blocks_unmapped_embedded_form_context():
    connection = _connection()
    connection.executescript(
        """
        INSERT INTO race_metadata
            (race_id, venue, race_number, race_date, grade, distance, data_source, url)
        VALUES
            ('Race 1 - TEST - 2026-06-01', 'TEST', 1, '2026-06-01',
             NULL, NULL, 'frozen_snapshot', 'https://example.test/r1'),
            ('TEST_2026-06-01_450m_GRADE5', 'TEST', NULL, '2026-06-01',
             'Grade 5', '450', 'embedded_form_guide', NULL);
        """
    )

    report = audit_rows(
        [
            {
                "race_id": "Race 1 - TEST - 2026-06-01",
                "race_date": "2026-06-01",
                "venue": "TEST",
                "target_distance_safe": "",
                "target_grade_safe": "",
            }
        ],
        connection,
    )

    assert report["verdict"] == "DATA_MISSING"
    assert report["safe_recoverable_rows_from_existing_sources"] == 0
    assert report["db_exact_race_number_status_counts"] == {
        "exact_race_number_row_missing_distance_or_grade": 1
    }
    assert report["embedded_context_status_counts"] == {
        "unsafe_unmapped_embedded_form_context_present": 1
    }


def test_target_metadata_audit_output_dir_guard_rejects_symlink_escape(
    tmp_path, monkeypatch
):
    import scripts.audit_stage2_target_metadata_coverage as audit

    monkeypatch.setattr(audit, "ROOT", tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}_outside"
    outside.mkdir()
    link = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525/"
        "target_metadata_coverage_symlink_report_only"
    )
    link.parent.mkdir(parents=True)
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        safe_output_dir(link)
