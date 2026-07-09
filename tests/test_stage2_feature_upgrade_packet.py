import json
from pathlib import Path

import pytest

from scripts import run_shadow_non_tgr_rf_evaluation as shadow_eval


STAGE2_FEATURE_COLUMNS = [
    "field_size",
    "box_number",
    "box_band_outside",
    "target_distance_safe",
    "target_grade_safe",
    "safe_grade_rank",
    "safe_field_strength",
    "last_start_grade_rank",
    "recent_avg_grade_rank_5",
    "last_start_field_size",
    "recent_avg_field_size_5",
    "last_start_race_strength",
    "recent_avg_race_strength_5",
    "prior_race_strength_delta_to_target",
    "recent_avg_speed_mps_5",
    "same_grade_avg_speed_mps",
    "grade_normalized_recent_speed_index",
    "target_box_band_prior_start_count",
    "target_box_band_win_rate",
    "target_box_band_place_rate",
    "target_box_band_avg_finish",
    "target_box_band_avg_time",
    "venue_box_band_start_count",
    "venue_box_band_win_rate",
    "venue_box_band_place_rate",
    "venue_box_band_avg_finish",
    "distance_box_band_start_count",
    "distance_box_band_win_rate",
    "distance_box_band_place_rate",
    "distance_box_band_avg_time",
    "same_distance_same_grade_start_count",
    "same_distance_same_grade_best_time",
    "same_distance_same_grade_avg_time",
]


def _prior_history_rows() -> list[dict]:
    return [
        {
            "race_date": "2026-06-01",
            "venue": "TRA",
            "box_number": 8,
            "distance_num": 350,
            "grade_normalized": "Grade 5",
            "time_num": 18.20,
            "finish_num": 2,
            "field_size": 8,
        },
        {
            "race_date": "2026-06-05",
            "venue": "WBL",
            "box_number": 7,
            "distance_num": 390,
            "grade_normalized": "Grade 4",
            "time_num": 18.00,
            "finish_num": 1,
            "field_size": 7,
        },
        {
            "race_date": "2026-06-08",
            "venue": "TRA",
            "box_number": 2,
            "distance_num": 350,
            "grade_normalized": "Grade 5",
            "time_num": 19.10,
            "finish_num": 5,
            "field_size": 6,
        },
        {
            "race_date": "2026-06-12",
            "venue": "TRA",
            "box_number": 7,
            "distance_num": 450,
            "grade_normalized": "Grade 6",
            "time_num": 23.00,
            "finish_num": 3,
            "field_size": 8,
        },
        {
            "race_date": "2026-06-15",
            "venue": "TRA",
            "box_number": 7,
            "distance_num": 350,
            "grade_normalized": "Grade 5",
            "time_num": 10.00,
            "finish_num": 1,
            "field_size": 8,
        },
        {
            "race_date": "2026-06-20",
            "venue": "TRA",
            "box_number": 7,
            "distance_num": 350,
            "grade_normalized": "Grade 5",
            "time_num": 9.50,
            "finish_num": 1,
            "field_size": 8,
        },
    ]


def _build_stage2_live_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    race_file = tmp_path / "Race 7 - TRA - 2026-06-15.csv"
    race_file.write_text(
        "Dog Name|BOX\n"
        "1. One Runner|1\n"
        "2. Two Runner|2\n"
        "3. Three Runner|3\n"
        "4. Four Runner|4\n"
        "5. Five Runner|5\n"
        "6. Six Runner|6\n"
        "7. Alpha Runner|7\n"
        "8. Eight Runner|8\n",
        encoding="utf-8",
    )
    race_file.with_name(race_file.name + ".metadata.json").write_text(
        json.dumps(
            {
                "metadata_is_leakage_safe": True,
                "metadata_source_url": "https://www.thedogs.com.au/racing/traralgon/2026-06-15/7/test?trial=false",
                "target_distance": "350m",
                "target_distance_source": "canonical_pre_race_page",
                "target_grade": "Grade 5",
                "target_grade_source": "canonical_pre_race_page",
                "race_info": {
                    "date": "2026-06-15",
                    "venue": "TRA",
                    "race_number": "7",
                    "race_time": "11:15 AM",
                    "url": "https://www.thedogs.com.au/racing/traralgon/2026-06-15/7/test?trial=false",
                },
            }
        ),
        encoding="utf-8",
    )

    class DummyConnection:
        def close(self):
            return None

    monkeypatch.setattr(shadow_eval, "sqlite_ro", lambda _path: DummyConnection())
    monkeypatch.setattr(
        shadow_eval,
        "load_db_history",
        lambda _connection: {"alpha runner": _prior_history_rows()},
    )

    rows = shadow_eval.build_live_feature_rows(
        input_paths=[race_file],
        schema={"feature_columns": STAGE2_FEATURE_COLUMNS},
        db_path=Path("unused.db"),
    )

    return next(row for row in rows if row["dog_name"] == "Alpha Runner")


def test_stage2_live_feature_rows_expose_prior_draw_adjusted_history_slices(
    tmp_path, monkeypatch
):
    row = _build_stage2_live_row(tmp_path, monkeypatch)

    assert row["box_number"] == 7
    assert row["box_band_outside"] == 1
    assert row["prior_start_count"] == 4
    assert row["target_box_band_prior_start_count"] == 3
    assert row["target_box_band_win_rate"] == pytest.approx(1 / 3)
    assert row["target_box_band_place_rate"] == pytest.approx(1.0)
    assert row["target_box_band_avg_finish"] == pytest.approx(2.0)
    assert row["target_box_band_avg_time"] == pytest.approx((18.20 + 18.00 + 23.00) / 3)

    assert row["venue_box_band_start_count"] == 2
    assert row["venue_box_band_win_rate"] == pytest.approx(0.0)
    assert row["venue_box_band_place_rate"] == pytest.approx(1.0)
    assert row["venue_box_band_avg_finish"] == pytest.approx(2.5)

    assert row["distance_box_band_start_count"] == 2
    assert row["distance_box_band_win_rate"] == pytest.approx(0.5)
    assert row["distance_box_band_place_rate"] == pytest.approx(1.0)
    assert row["distance_box_band_avg_time"] == pytest.approx((18.20 + 18.00) / 2)

    assert row["same_distance_same_grade_start_count"] == 2
    assert row["same_distance_same_grade_best_time"] == pytest.approx(18.20)
    assert row["same_distance_same_grade_avg_time"] == pytest.approx((18.20 + 19.10) / 2)
    assert row["same_distance_same_grade_target_race_rows_used"] == 0
    assert row["same_distance_same_grade_post_outcome_rows_used"] == 0


def test_stage2_live_feature_rows_expose_prior_class_strength_features(
    tmp_path, monkeypatch
):
    row = _build_stage2_live_row(tmp_path, monkeypatch)

    prior_speeds = [
        350 / 18.20,
        390 / 18.00,
        350 / 19.10,
        450 / 23.00,
    ]
    same_grade_speeds = [350 / 18.20, 350 / 19.10]

    assert row["safe_grade_rank"] == 7
    assert row["safe_field_strength"] == 56
    assert row["last_start_grade_rank"] == 6
    assert row["recent_avg_grade_rank_5"] == pytest.approx((7 + 8 + 7 + 6) / 4)
    assert row["last_start_field_size"] == 8
    assert row["recent_avg_field_size_5"] == pytest.approx((8 + 7 + 6 + 8) / 4)
    assert row["last_start_race_strength"] == 48
    assert row["recent_avg_race_strength_5"] == pytest.approx((56 + 56 + 42 + 48) / 4)
    assert row["prior_race_strength_delta_to_target"] == pytest.approx(56 - 50.5)
    assert row["recent_avg_speed_mps_5"] == pytest.approx(sum(prior_speeds) / len(prior_speeds))
    assert row["same_grade_avg_speed_mps"] == pytest.approx(
        sum(same_grade_speeds) / len(same_grade_speeds)
    )
    assert row["grade_normalized_recent_speed_index"] == pytest.approx(
        (sum(prior_speeds) / len(prior_speeds))
        / (sum(same_grade_speeds) / len(same_grade_speeds))
    )
