import json
import sqlite3
from pathlib import Path

import pytest

from scripts.build_no_box_actual_win_feature_join_packet import (
    build_feature_join_packet,
    load_history_feature_index_from_db,
    write_outputs,
)


def _smoke_row(race_id: str, dog_name_key: str, actual_win: int) -> dict:
    return {
        "race_id": race_id,
        "legacy_race_id": race_id,
        "identity_key": "2025-01-01|TEST|R01",
        "race_date": "2025-01-01",
        "venue": "TEST",
        "race_number": 1,
        "dog_name_key": dog_name_key,
        "dog_name": dog_name_key.title(),
        "actual_win": actual_win,
        "candidate_kind": "partial_field",
        "field_scope": "partial_db_name_subset_of_official_finishers",
        "field_complete_for_ranking": False,
        "race_grouped_actual_win_ranking_allowed": False,
        "target_source": "official_winner_name_metadata_confirmed",
        "label_scope": "actual_win_only",
        "box_features_allowed": False,
        "finish_order_labels_allowed": False,
        "top3_labels_allowed": False,
        "official_safe_label_candidate": False,
        "label_write_approved": False,
    }


def _dataset_row(race_id: str, dog_name: str) -> dict:
    return {
        "race_id": race_id,
        "dog_name": dog_name,
        "dog_key": dog_name.upper().replace(" ", ""),
        "features": {
            "box_number": 8,
            "box_band_outside": 1,
            "race_number": 1,
            "target_day_of_week": 2,
            "target_month": 1,
            "field_size": 2,
            "target_distance_safe": 450.0,
            "prior_start_count": 2,
            "recent_win_rate_5": 0.4,
            "recent_place_rate_5": 0.6,
            "recent_finish_best_5": 2,
            "recent_avg_time_5": 25.5,
            "days_since_last_start": 8,
            "starts_same_venue": 3,
            "win_rate_same_distance": 0.25,
            "same_grade_start_count": 2,
            "grade_change_indicator": -1,
            "career_win_rate": 0.33,
            "weather": 1,
        },
    }


def test_feature_join_matches_prefixed_names_and_excludes_leaky_features(
    tmp_path: Path, monkeypatch
):
    import scripts.build_no_box_actual_win_feature_join_packet as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    smoke_rows = [
        _smoke_row("R1", "carry on kim", 1),
        _smoke_row("R1", "binnaway wizard", 0),
    ]
    dataset_rows = [
        _dataset_row("R1", "8. Carry On Kim"),
        _dataset_row("R1", "4. Binnaway Wizard"),
    ]

    packet, rows = build_feature_join_packet(
        smoke_rows=smoke_rows,
        dataset_rows=dataset_rows,
        smoke_rows_path="/fixture/smoke.jsonl",
        dataset_path="/fixture/dataset.jsonl",
    )

    assert packet["status"] == "REPORT_ONLY_DOG_FORM_FEATURE_JOIN_READY"
    assert packet["writes_performed"]["label_write"] is False
    assert packet["writes_performed"]["model_training"] is False
    assert packet["summary"]["match_status_counts"] == {"MATCHED": 2}
    assert packet["summary"]["no_box_features_selected"] is True
    assert packet["summary"]["no_race_number_feature_selected"] is True
    assert packet["summary"]["no_calendar_features_selected"] is True
    assert rows[0]["feature_join_status"] == "MATCHED"
    assert rows[0]["feature_recent_win_rate_5"] == 0.4
    assert rows[0]["feature_starts_same_venue"] == 3
    assert rows[0]["feature_grade_change_indicator"] == -1
    assert "feature_box_number" not in rows[0]
    assert "feature_race_number" not in rows[0]
    assert "feature_target_day_of_week" not in rows[0]
    assert "feature_target_distance_safe" not in rows[0]
    assert "box_number" not in rows[0]

    output_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/feature_join"
    write_outputs(output_dir, packet, rows)
    assert (output_dir / "no_box_actual_win_feature_join_packet.json").exists()
    assert (output_dir / "no_box_actual_win_feature_rows.jsonl").exists()
    assert (output_dir / "no_box_actual_win_feature_rows.csv").exists()
    written = json.loads((output_dir / "no_box_actual_win_feature_join_packet.json").read_text())
    assert written["schema_version"] == "no_box_actual_win_dog_form_feature_join_v1"

    cwd = tmp_path / "caller_cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    relative_output_dir = Path(
        "artifacts/full_evidence_orchestration_20260525/relative_feature_join"
    )
    write_outputs(relative_output_dir, packet, rows)
    assert (
        tmp_path / relative_output_dir / "no_box_actual_win_feature_join_packet.json"
    ).exists()
    assert not (
        cwd / relative_output_dir / "no_box_actual_win_feature_join_packet.json"
    ).exists()


def test_feature_join_output_guard_fails_closed(tmp_path: Path, monkeypatch):
    import scripts.build_no_box_actual_win_feature_join_packet as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    packet, rows = build_feature_join_packet(
        smoke_rows=[_smoke_row("R1", "missing dog", 1)],
        dataset_rows=[],
    )

    with pytest.raises(ValueError, match="output_dir_must_be_under_artifacts"):
        write_outputs(tmp_path / "outside", packet, rows)
    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        write_outputs(
            tmp_path.parent
            / "outside"
            / "artifacts/full_evidence_orchestration_20260525/feature_join",
            packet,
            rows,
        )


def test_feature_join_fails_closed_when_dataset_match_missing():
    packet, rows = build_feature_join_packet(
        smoke_rows=[_smoke_row("R1", "missing dog", 1)],
        dataset_rows=[],
    )

    assert packet["status"] == "REPORT_ONLY_DOG_FORM_FEATURE_JOIN_FAILED"
    assert rows[0]["feature_join_status"] == "MISSING_DATASET_ROW"
    assert packet["summary"]["failures"]


def test_feature_join_fills_missing_dog_form_from_staged_history_db(tmp_path: Path):
    db_path = tmp_path / "history.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        create table race_metadata (
            race_id text,
            race_date text,
            venue text,
            race_number integer,
            distance text,
            grade text
        )
        """
    )
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
    conn.execute(
        "insert into race_metadata values (?,?,?,?,?,?)",
        ("R1", "2025-01-10", "TEST", 1, "450", "5"),
    )
    history_rows = [
        ("R1", "1. Fast Dog", "1. Fast Dog", 1, 30.0, "25.10", "4.90", 0.0, "2025-01-03", "450", "5"),
        ("R1", "1. Fast Dog", "1. Fast Dog", 4, 31.0, "26.20", "5.10", 3.5, "2024-12-20", "400", "6"),
    ]
    for row in history_rows:
        raw = {
            "DATE": row[8],
            "TRACK": "TEST",
            "DIST": row[9],
            "G": row[10],
            "TIME": row[5],
            "1 SEC": row[6],
            "MGN": str(row[7]),
            "WGT": str(row[4]),
            "PLC": str(row[3]),
        }
        conn.execute(
            "insert into csv_dog_history_staging values (?,?,?,?,?,?,?,?,?)",
            (*row[:8], json.dumps(raw)),
        )
    conn.commit()
    conn.close()

    smoke_rows = [_smoke_row("R1", "fast dog", 1)]
    dataset = _dataset_row("R1", "1. Fast Dog")
    for name in list(dataset["features"]):
        if "box" not in name and name not in {"race_number", "target_day_of_week", "target_month", "field_size"}:
            dataset["features"][name] = None
    history_index, history_summary = load_history_feature_index_from_db(db_path, smoke_rows)

    packet, rows = build_feature_join_packet(
        smoke_rows=smoke_rows,
        dataset_rows=[dataset],
        history_feature_index=history_index,
        history_feature_summary=history_summary,
    )

    assert packet["status"] == "REPORT_ONLY_DOG_FORM_FEATURE_JOIN_READY"
    assert packet["summary"]["history_db_features_enabled"] is True
    assert packet["summary"]["history_db_feature_summary"]["history_db_rows_used"] == 2
    assert packet["summary"]["history_db_filled_rows"] == 1
    assert rows[0]["history_feature_join_status"] == "MATCHED"
    assert rows[0]["history_feature_prior_start_count"] == 2
    assert rows[0]["feature_prior_start_count"] == 2
    assert rows[0]["feature_recent_win_rate_5"] == 0.5
    assert rows[0]["feature_recent_place_rate_5"] == 0.5
    assert rows[0]["feature_days_since_last_start"] == 7
    assert rows[0]["feature_win_rate_same_distance"] == 1.0
    assert rows[0]["feature_same_grade_start_count"] == 1
    assert "feature_box_number" not in rows[0]
    assert "feature_race_number" not in rows[0]


def test_feature_join_fills_history_when_target_name_has_terminal_status_suffix(tmp_path: Path):
    db_path = tmp_path / "history.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        create table race_metadata (
            race_id text,
            race_date text,
            venue text,
            race_number integer,
            distance text,
            grade text
        )
        """
    )
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
    conn.execute(
        "insert into race_metadata values (?,?,?,?,?,?)",
        ("R1", "2025-07-14", "WAR", 6, "450", "5"),
    )
    raw = {
        "DATE": "2025-06-30",
        "TRACK": "WAR",
        "DIST": "450",
        "G": "5",
        "TIME": "25.10",
        "1 SEC": "4.90",
        "MGN": "0.0",
        "WGT": "30.0",
        "PLC": "1",
    }
    conn.execute(
        "insert into csv_dog_history_staging values (?,?,?,?,?,?,?,?,?)",
        ("R1", "Paw Maddox", "Paw Maddox", 1, 30.0, "25.10", "4.90", 0.0, json.dumps(raw)),
    )
    conn.commit()
    conn.close()

    smoke_row = _smoke_row("R1", "paw maddox nbt", 1)
    smoke_row["dog_name"] = "Paw Maddox NBT"
    dataset = _dataset_row("R1", "Paw Maddox NBT")
    for name in list(dataset["features"]):
        if "box" not in name and name not in {"race_number", "target_day_of_week", "target_month", "field_size"}:
            dataset["features"][name] = None
    history_index, history_summary = load_history_feature_index_from_db(db_path, [smoke_row])

    packet, rows = build_feature_join_packet(
        smoke_rows=[smoke_row],
        dataset_rows=[dataset],
        history_feature_index=history_index,
        history_feature_summary=history_summary,
    )

    assert packet["status"] == "REPORT_ONLY_DOG_FORM_FEATURE_JOIN_READY"
    assert packet["summary"]["history_feature_match_status_counts"] == {
        "MATCHED_SUFFIX_STRIPPED_TARGET_NAME": 1
    }
    assert packet["summary"]["history_db_filled_rows"] == 1
    assert rows[0]["history_feature_join_status"] == "MATCHED_SUFFIX_STRIPPED_TARGET_NAME"
    assert rows[0]["history_feature_matched_key"] == "paw maddox"
    assert rows[0]["history_feature_prior_start_count"] == 1
    assert rows[0]["feature_prior_start_count"] == 1
    assert rows[0]["feature_days_since_last_start"] == 14
    assert rows[0]["feature_win_rate_same_distance"] == 1.0
    assert "feature_box_number" not in rows[0]


def test_feature_join_flags_history_label_proxy_risk():
    smoke_rows = []
    dataset_rows = []
    history_index = {}
    for race_number in range(1, 21):
        race_id = f"R{race_number}"
        winner = f"winner {race_number}"
        loser = f"loser {race_number}"
        smoke_rows.extend([
            _smoke_row(race_id, winner, 1),
            _smoke_row(race_id, loser, 0),
        ])
        winner_row = _dataset_row(race_id, winner)
        loser_row = _dataset_row(race_id, loser)
        for dataset_row in (winner_row, loser_row):
            for name in list(dataset_row["features"]):
                if "box" not in name and name not in {"race_number", "target_day_of_week", "target_month", "field_size"}:
                    dataset_row["features"][name] = None
        dataset_rows.extend([winner_row, loser_row])
        history_index[(race_id, winner)] = {
            "recent_win_rate_5": 1.0,
            "career_win_rate": 1.0,
            "recent_finish_best_5": 1,
            "recent_finish_mean_3": 1.0,
        }
        history_index[(race_id, loser)] = {
            "recent_win_rate_5": 0.0,
            "career_win_rate": 0.0,
            "recent_finish_best_5": 4,
            "recent_finish_mean_3": 4.0,
        }

    packet, _ = build_feature_join_packet(
        smoke_rows=smoke_rows,
        dataset_rows=dataset_rows,
        history_feature_index=history_index,
        history_feature_summary={"fixture": True},
    )

    assert packet["status"] == "REPORT_ONLY_DOG_FORM_FEATURE_JOIN_LEAKAGE_RISK"
    audit = packet["summary"]["label_proxy_audit"]
    assert audit["status"] == "POTENTIAL_LABEL_PROXY"
    assert "feature_recent_win_rate_5" in audit["risk_features"]
    assert packet["recommended_next_action"].startswith("do_not_use_history_db_enriched_metrics")


def test_feature_join_masked_history_policy_skips_outcome_proxy_fields():
    smoke_rows = [_smoke_row("R1", "fast dog", 1), _smoke_row("R1", "slow dog", 0)]
    dataset_rows = [_dataset_row("R1", "Fast Dog"), _dataset_row("R1", "Slow Dog")]
    for dataset_row in dataset_rows:
        for name in list(dataset_row["features"]):
            if "box" not in name and name not in {"race_number", "target_day_of_week", "target_month", "field_size"}:
                dataset_row["features"][name] = None
    history_index = {
        ("R1", "fast dog"): {
            "recent_win_rate_5": 1.0,
            "career_win_rate": 1.0,
            "recent_finish_best_5": 1,
            "days_since_last_start": 7,
            "recent_avg_time_5": 25.1,
            "starts_same_venue": 2,
        },
        ("R1", "slow dog"): {
            "recent_win_rate_5": 0.0,
            "career_win_rate": 0.0,
            "recent_finish_best_5": 4,
            "days_since_last_start": 7,
            "recent_avg_time_5": 26.2,
            "starts_same_venue": 2,
        },
    }

    packet, rows = build_feature_join_packet(
        smoke_rows=smoke_rows,
        dataset_rows=dataset_rows,
        history_feature_index=history_index,
        history_feature_summary={"fixture": True},
        history_fill_policy="no_outcome_proxy_fields",
    )

    assert packet["status"] == "REPORT_ONLY_DOG_FORM_FEATURE_JOIN_READY"
    assert packet["summary"]["history_db_fill_policy"] == "no_outcome_proxy_fields"
    assert packet["summary"]["history_db_policy_skipped_feature_value_count"] == 6
    assert packet["summary"]["label_proxy_audit"]["status"] == "PASS"
    fast = next(row for row in rows if row["dog_name_key"] == "fast dog")
    assert fast["feature_recent_win_rate_5"] is None
    assert fast["feature_career_win_rate"] is None
    assert fast["feature_recent_finish_best_5"] is None
    assert fast["feature_days_since_last_start"] == 7
    assert fast["feature_recent_avg_time_5"] == 25.1
    assert fast["feature_starts_same_venue"] == 2
