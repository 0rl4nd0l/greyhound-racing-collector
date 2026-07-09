import json
import sqlite3
from pathlib import Path

import pytest

import scripts.build_no_box_actual_win_feature_join_packet as feature_join_guard
from scripts.build_global_prior_history_feature_packet import (
    GLOBAL_PRIOR_STATUS_EXACT,
    GLOBAL_PRIOR_STATUS_SUFFIX,
    build_global_prior_history_packet,
    load_global_history_index_from_db,
    write_outputs,
)


def _base_row(race_id: str, dog_name_key: str, dog_name: str, actual_win: int = 1) -> dict:
    return {
        "race_id": race_id,
        "legacy_race_id": race_id,
        "identity_key": "2025-07-14|WAR|R06",
        "race_date": "2025-07-14",
        "venue": "WAR",
        "race_number": 6,
        "dog_name_key": dog_name_key,
        "dog_name": dog_name,
        "actual_win": actual_win,
        "candidate_kind": "complete_field",
        "field_scope": "official_safe",
        "field_complete_for_ranking": True,
        "race_grouped_actual_win_ranking_allowed": True,
        "target_source": "thedogs_official",
        "target_distance": 450,
        "target_grade": "5",
        "label_scope": "actual_win_only",
        "box_features_allowed": False,
        "finish_order_labels_allowed": False,
        "top3_labels_allowed": False,
        "official_safe_label_candidate": False,
        "label_write_approved": False,
        "feature_prior_start_count": 0,
        "feature_days_since_last_start": None,
        "feature_recent_win_rate_5": None,
        "feature_recent_place_rate_5": None,
        "feature_win_rate_same_distance": None,
        "feature_starts_same_venue": 0,
        "history_feature_join_status": "DISABLED",
        "history_feature_values_filled": 0,
    }


def _create_history_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        create table csv_dog_history_staging (
            race_id text,
            dog_name text,
            dog_clean_name text,
            finish_position integer,
            weight real,
            individual_time text,
            sectional_1st text,
            margin real,
            raw_row_json text
        )
        """
    )
    rows = [
        ("H1", "Paw Maddox", "Paw Maddox", 1, "2025-06-30"),
        ("H2", "Paw Maddox", "Paw Maddox", 3, "2025-06-23"),
        ("H3", "Paw Maddox", "Paw Maddox", 4, "2025-07-20"),
        ("H4", "Fast Dog", "Fast Dog", 2, "2025-07-01"),
    ]
    for race_id, dog_name, clean_name, place, race_date in rows:
        raw = {
            "DATE": race_date,
            "TRACK": "WAR",
            "DIST": "450",
            "G": "5",
            "TIME": "25.10",
            "1 SEC": "4.90",
            "MGN": "1.5",
            "WGT": "30.0",
            "PLC": str(place),
        }
        conn.execute(
            "insert into csv_dog_history_staging values (?,?,?,?,?,?,?,?,?)",
            (
                race_id,
                dog_name,
                clean_name,
                place,
                30.0,
                "25.10",
                "4.90",
                1.5,
                json.dumps(raw),
            ),
        )
    conn.commit()
    conn.close()


def test_global_prior_history_suffix_strips_target_status_and_filters_future_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(feature_join_guard, "ROOT", tmp_path)
    db_path = tmp_path / "history.db"
    _create_history_db(db_path)
    history_by_key, history_summary = load_global_history_index_from_db(db_path)
    base_rows = [_base_row("R1", "paw maddox nbt", "Paw Maddox NBT")]

    packet, rows = build_global_prior_history_packet(
        base_packet={"status": "REPORT_ONLY_DOG_FORM_FEATURE_JOIN_READY"},
        base_rows=base_rows,
        history_by_key=history_by_key,
        history_summary=history_summary,
        base_packet_path="/fixture/packet.json",
        base_rows_path="/fixture/rows.jsonl",
    )

    assert packet["status"] == "REPORT_ONLY_DOG_FORM_FEATURE_JOIN_READY"
    assert all(value is False for value in packet["writes_performed"].values())
    assert packet["summary"]["history_db_feature_summary"]["db_quick_check"] == "ok"
    assert packet["summary"]["rows_with_global_prior_history"] == 1
    assert packet["summary"]["suffix_stripped_global_prior_history_rows"] == 1
    assert packet["summary"]["history_feature_match_status_counts"] == {
        GLOBAL_PRIOR_STATUS_SUFFIX: 1
    }
    assert rows[0]["history_feature_join_status"] == GLOBAL_PRIOR_STATUS_SUFFIX
    assert rows[0]["global_prior_history_matched_key"] == "paw maddox"
    assert rows[0]["global_prior_history_count"] == 2
    assert rows[0]["global_prior_history_latest_date"] == "2025-06-30"
    assert rows[0]["feature_prior_start_count"] == 2
    assert rows[0]["feature_days_since_last_start"] == 14
    assert rows[0]["feature_recent_win_rate_5"] == 0.5
    assert rows[0]["feature_recent_place_rate_5"] == 1.0
    assert rows[0]["feature_win_rate_same_distance"] == 0.5

    output_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/global_prior"
    write_outputs(output_dir, packet, rows)
    assert (output_dir / "no_box_actual_win_feature_join_packet.json").exists()
    assert (output_dir / "no_box_actual_win_feature_rows.jsonl").exists()
    assert (output_dir / "no_box_actual_win_feature_rows.csv").exists()


def test_global_prior_history_uses_exact_match_before_suffix_match(tmp_path: Path):
    db_path = tmp_path / "history.db"
    _create_history_db(db_path)
    history_by_key, history_summary = load_global_history_index_from_db(db_path)
    base_rows = [_base_row("R1", "fast dog", "Fast Dog")]

    packet, rows = build_global_prior_history_packet(
        base_packet={"status": "REPORT_ONLY_DOG_FORM_FEATURE_JOIN_READY"},
        base_rows=base_rows,
        history_by_key=history_by_key,
        history_summary=history_summary,
    )

    assert packet["summary"]["history_feature_match_status_counts"] == {
        GLOBAL_PRIOR_STATUS_EXACT: 1
    }
    assert rows[0]["history_feature_join_status"] == GLOBAL_PRIOR_STATUS_EXACT
    assert rows[0]["global_prior_history_matched_key"] == "fast dog"
    assert rows[0]["global_prior_history_count"] == 1


def test_global_prior_history_rejects_absolute_output_outside_repo(tmp_path: Path):
    outside = tmp_path.parent / "artifacts/full_evidence_orchestration_20260525/global_prior"

    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        feature_join_guard._repo_output_path(outside, root=tmp_path)


def test_global_prior_history_rejects_in_repo_non_artifact_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(feature_join_guard, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="output_dir_must_be_under_artifacts"):
        feature_join_guard._assert_output_dir_safe(tmp_path / "reports/global_prior")
