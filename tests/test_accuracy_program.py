from datetime import datetime

import pytest

from accuracy_program.bet_readiness import apply_bet_readiness_gates
from accuracy_program.evaluation import (
    blend_probabilities,
    market_implied_probabilities,
    score_predictions,
    validate_feature_columns,
    validate_temporal_holdout,
)
from accuracy_program.snapshots import (
    assert_no_result_fields,
    build_prediction_snapshot,
    persist_prediction_snapshot,
)


def test_snapshot_excludes_result_labels_and_preserves_ev_contract():
    prediction_result = {
        "race_id": "Race 4 - WRGL - 2026-05-21",
        "model_version": "model-v1",
        "predictions": [
            {
                "dog_clean_name": "Alpha Runner",
                "box_number": 1,
                "win_prob_raw": 0.44,
                "win_prob_norm": 0.4,
                "predicted_rank": 1,
                "confidence_score": 0.62,
                "odds_win": 3.0,
                "odds_timestamp": "2026-05-21T15:44:00",
                "odds_source": "sportsbet",
                "odds_source_url": "https://www.sportsbet.com.au/greyhound-racing/race-4",
                "odds_race_id": "Race 4 - WRGL - 2026-05-21",
                "odds_dog_name": "Alpha Runner",
                "odds_box_number": 1,
                "odds_match_method": "race_id_box_name",
                "odds_match_confidence": 1.0,
                "ev_win": 0.2,
            }
        ],
        "actual_results": {"winner": "Beta Runner"},
    }

    snapshot = build_prediction_snapshot(
        prediction_result,
        source_file_path="Race 4 - WRGL - 2026-05-21.csv",
        lifecycle={
            "status": "upcoming_not_jumped",
            "status_reason": "jump_time_after_now_no_result",
        },
        prediction_timestamp="2026-05-21T15:45:00",
    )

    assert snapshot["is_pre_jump_snapshot"] is True
    assert snapshot["schema_version"] == "prediction_snapshot_v1"
    assert snapshot["predictions"][0]["predicted_rank"] == 1
    assert snapshot["predictions"][0]["confidence_score"] == pytest.approx(0.62)
    assert snapshot["predictions"][0]["odds"] == pytest.approx(3.0)
    assert snapshot["predictions"][0]["ev_win"] == pytest.approx(0.2)
    assert (
        snapshot["predictions"][0]["odds_snapshot"]["odds_timestamp"]
        == "2026-05-21T15:44:00"
    )
    assert snapshot["predictions"][0]["odds_match_status"] == "valid_pre_jump_dog_odds"
    assert "actual_results" not in snapshot
    assert "actual_results" not in snapshot["predictions"][0]


def test_snapshot_can_attach_report_only_calibration_without_reranking():
    prediction_result = {
        "race_id": "Race 4 - WRGL - 2026-05-21",
        "model_version": "model-v1",
        "predictions": [
            {
                "dog_clean_name": "Alpha Runner",
                "box_number": 1,
                "win_prob_norm": 0.64,
                "predicted_rank": 1,
            },
            {
                "dog_clean_name": "Beta Runner",
                "box_number": 2,
                "win_prob_norm": 0.25,
                "predicted_rank": 2,
            },
            {
                "dog_clean_name": "Gamma Runner",
                "box_number": 3,
                "win_prob_norm": 0.11,
                "predicted_rank": 3,
            },
        ],
    }

    snapshot = build_prediction_snapshot(
        prediction_result,
        source_file_path="Race 4 - WRGL - 2026-05-21.csv",
        lifecycle={"status": "upcoming_not_jumped"},
        prediction_timestamp="2026-05-21T15:45:00",
        report_only_calibration={
            "algorithm": "power_normalize_per_race",
            "alpha": 0.5,
            "input_probability_key": "win_prob_norm",
            "output_probability_key": "calibrated_win_prob_report_only",
        },
    )

    rows = snapshot["predictions"]
    assert [row["predicted_rank"] for row in rows] == [1, 2, 3]
    assert [row["win_prob_norm"] for row in rows] == [0.64, 0.25, 0.11]
    assert sum(row["calibrated_win_prob_report_only"] for row in rows) == pytest.approx(
        1.0
    )
    assert [row["calibrated_predicted_rank_report_only"] for row in rows] == [
        1,
        2,
        3,
    ]
    assert snapshot["report_only_calibration"] == {
        "algorithm": "power_normalize_per_race",
        "alpha": 0.5,
        "input_probability_key": "win_prob_norm",
        "output_probability_key": "calibrated_win_prob_report_only",
        "output_rank_key": "calibrated_predicted_rank_report_only",
        "status": "APPLIED_REPORT_ONLY",
        "canonical_probabilities_unchanged": True,
        "canonical_ranks_unchanged": True,
        "uses_labels_at_runtime": False,
        "uses_odds_at_runtime": False,
        "model_artifact_written": False,
        "registry_mutation_allowed": False,
        "promotion_allowed": False,
        "betting_allowed": False,
    }


def test_snapshot_ev_stays_null_without_valid_pre_jump_odds():
    snapshot = build_prediction_snapshot(
        {
            "race_id": "Race 4 - WRGL - 2026-05-21",
            "model_version": "model-v1",
            "predictions": [
                {
                    "dog_clean_name": "Alpha Runner",
                    "box_number": 1,
                    "win_prob_norm": 0.4,
                    "predicted_rank": 1,
                    "odds_win": 3.0,
                    "ev_win": 0.2,
                }
            ],
        },
        lifecycle={
            "status": "upcoming_not_jumped",
            "jump_datetime": "2026-05-21T16:00:00",
        },
        prediction_timestamp="2026-05-21T15:45:00",
    )

    runner = snapshot["predictions"][0]
    assert runner["ev_win"] is None
    assert runner["odds_match_status"] == "missing_timestamp"
    assert runner["odds_exclusion_reason"] == "missing_timestamp"
    assert "missing_odds_timestamp" in runner["data_quality_flags"]
    assert "invalid_pre_jump_odds" in runner["data_quality_flags"]


def test_snapshot_odds_provenance_gate_rejects_unsafe_ev_inputs():
    base = {
        "dog_clean_name": "Alpha Runner",
        "box_number": 1,
        "win_prob_norm": 0.4,
        "predicted_rank": 1,
        "odds_win": 3.0,
        "odds_source": "sportsbet",
        "odds_source_url": "https://www.sportsbet.com.au/greyhound-racing/race-4",
        "odds_race_id": "Race 4 - WRGL - 2026-05-21",
        "odds_dog_name": "Alpha Runner",
        "odds_box_number": 1,
        "odds_match_method": "race_id_box_name",
        "odds_match_confidence": 1.0,
    }
    cases = [
        (
            "valid_pre_jump_dog_odds",
            {"odds_timestamp": "2026-05-21T15:44:00"},
            pytest.approx(0.2),
        ),
        (
            "duplicate_odds_rows",
            {
                "odds_timestamp": "2026-05-21T15:44:00",
                "odds_candidate_count": 2,
            },
            None,
        ),
        ("missing_timestamp", {}, None),
        (
            "timestamp_after_prediction",
            {"odds_timestamp": "2026-05-21T15:46:00"},
            None,
        ),
        (
            "timestamp_after_feature_freeze",
            {
                "odds_timestamp": "2026-05-21T15:44:00",
                "feature_freeze_timestamp": "2026-05-21T15:43:00",
            },
            None,
        ),
        (
            "timestamp_after_jump",
            {
                "odds_timestamp": "2026-05-21T15:59:30",
                "prediction_timestamp": "2026-05-21T16:00:00",
            },
            None,
        ),
        (
            "stale_beyond_ttl",
            {"odds_timestamp": "2026-05-21T15:00:00"},
            None,
        ),
        (
            "box_mismatch",
            {"odds_timestamp": "2026-05-21T15:44:00", "odds_box_number": 2},
            None,
        ),
        (
            "ambiguous_box_source",
            {
                "odds_timestamp": "2026-05-21T15:44:00",
                "odds_sportsbet_box_source": "list_position_fallback",
            },
            None,
        ),
        (
            "post_race_or_sp_only",
            {"odds_timestamp": "2026-05-21T15:44:00", "odds_market_type": "sp"},
            None,
        ),
        (
            "post_race_or_sp_only",
            {
                "odds_timestamp": "2026-05-21T15:44:00",
                "odds_source_url": "https://www.sportsbet.com.au/greyhound-racing/results/race-4",
            },
            None,
        ),
        (
            "post_race_or_sp_only",
            {
                "odds_timestamp": "2026-05-21T15:44:00",
                "odds_source_url": "https://www.sportsbet.com.au/greyhound-racing/race-4?market=sp",
            },
            None,
        ),
    ]

    for expected_status, override, expected_ev in cases:
        row = {**base, **override}
        prediction_timestamp = row.pop("prediction_timestamp", "2026-05-21T15:45:00")
        feature_freeze_timestamp = row.pop("feature_freeze_timestamp", prediction_timestamp)
        snapshot = build_prediction_snapshot(
            {
                "race_id": "Race 4 - WRGL - 2026-05-21",
                "model_version": "model-v1",
                "predictions": [row],
            },
            lifecycle={
                "status": "upcoming_not_jumped",
                "jump_datetime": "2026-05-21T15:58:00",
            },
            prediction_timestamp=prediction_timestamp,
            feature_freeze_timestamp=feature_freeze_timestamp,
            stale_odds_after_minutes=30.0,
        )
        runner = snapshot["predictions"][0]
        assert runner["odds_match_status"] == expected_status
        if expected_status == "timestamp_after_feature_freeze":
            assert "odds_not_captured_before_feature_freeze" in runner["data_quality_flags"]
            readiness = snapshot["snapshot_readiness"]
            assert readiness["requirements"]["priced_runners_captured_before_prediction"] is True
            assert readiness["requirements"]["priced_runners_captured_before_feature_freeze"] is False
            assert readiness["counts"]["odds_not_captured_before_prediction_count"] == 0
            assert readiness["counts"]["odds_not_captured_before_feature_freeze_count"] == 1
        if expected_ev is None:
            assert runner["ev_win"] is None
            assert runner["odds_exclusion_reason"] == expected_status
        else:
            assert runner["ev_win"] == expected_ev
            assert runner["odds_exclusion_reason"] is None
        assert_no_result_fields(snapshot)


def test_snapshot_ev_accepts_canonical_race_id_equivalence_only_for_same_race():
    base_runner = {
        "dog_clean_name": "Alpha Runner",
        "box_number": 1,
        "win_prob_norm": 0.4,
        "predicted_rank": 1,
        "odds_win": 3.0,
        "odds_timestamp": "2026-05-21T15:44:00",
        "odds_source": "sportsbet",
        "odds_source_url": "https://www.sportsbet.com.au/greyhound-racing/race-4",
        "odds_dog_name": "Alpha Runner",
        "odds_box_number": 1,
        "odds_match_method": "race_id_box_name",
        "odds_match_confidence": 1.0,
        "odds_sportsbet_box_source": "explicit_dom",
    }

    def snapshot_for(odds_race_id: str):
        return build_prediction_snapshot(
            {
                "race_id": "Race 4 - WRGL - 2026-05-21",
                "model_version": "model-v1",
                "predictions": [{**base_runner, "odds_race_id": odds_race_id}],
            },
            lifecycle={
                "status": "upcoming_not_jumped",
                "jump_datetime": "2026-05-21T15:58:00",
            },
            prediction_timestamp="2026-05-21T15:45:00",
            stale_odds_after_minutes=30.0,
        )

    accepted = snapshot_for("WRGL_2026-05-21_4")
    accepted_runner = accepted["predictions"][0]
    assert accepted_runner["odds_match_status"] == "valid_pre_jump_dog_odds"
    assert accepted_runner["odds_match_method"] == "canonical_race_id_box_dog"
    assert accepted_runner["ev_win"] == pytest.approx(0.2)
    assert (
        accepted_runner["odds_snapshot"]["odds_provenance"]["sportsbet_box_source"]
        == "explicit_dom"
    )

    rejected = snapshot_for("WRGL_2026-05-22_4")
    rejected_runner = rejected["predictions"][0]
    assert rejected_runner["odds_match_status"] == "race_id_mismatch"
    assert rejected_runner["ev_win"] is None
    assert_no_result_fields(accepted)
    assert_no_result_fields(rejected)


def test_snapshot_carries_history_and_target_metadata_provenance_result_free():
    snapshot = build_prediction_snapshot(
        {
            "race_id": "Race 4 - WRGL - 2026-05-21",
            "model_version": "model-v1",
            "predictions": [
                {
                    "dog_clean_name": "Alpha Runner",
                    "box_number": 1,
                    "win_prob_norm": 0.4,
                    "predicted_rank": 1,
                    "history_source": "embedded_csv_form_history",
                    "history_match_status": "embedded_history_only",
                    "db_history_match_status": "matched_identity_rows_missing_finish_position",
                    "db_result_history_count": 0,
                    "runner_inclusion_reason": "model_scored_low_confidence_retained",
                    "distance_source": "default_missing_target",
                    "grade_source": "default_missing_target",
                    "metadata_source_detail": {
                        "distance": "default_missing_target:no_safe_pre_race_distance",
                        "grade": "default_missing_target:no_safe_pre_race_grade",
                    },
                    "metadata_is_leakage_safe": False,
                    "rejected_metadata_sources": [
                        "embedded_form_history:DIST",
                        "embedded_form_history:G",
                    ],
                    "quality_flags": [
                        "optimizer_retained_low_quality_for_runner_alignment"
                    ],
                }
            ],
        },
        lifecycle={"status": "upcoming_not_jumped"},
        prediction_timestamp="2026-05-21T15:45:00",
    )

    runner = snapshot["predictions"][0]
    assert runner["history_source"] == "embedded_csv_form_history"
    assert runner["history_match_status"] == "embedded_history_only"
    assert runner["db_history_match_status"] == "matched_identity_rows_missing_finish_position"
    assert runner["runner_inclusion_reason"] == "model_scored_low_confidence_retained"
    assert runner["metadata_is_leakage_safe"] is False
    assert runner["ev_win"] is None
    assert_no_result_fields(snapshot)


def test_snapshot_persistence_is_explicit_result_free_and_dry_runnable(tmp_path):
    snapshot = build_prediction_snapshot(
        {
            "race_id": "Race 4 - WRGL - 2026-05-21",
            "model_version": "model-v1",
            "predictions": [
                {
                    "dog_clean_name": "Alpha Runner",
                    "box_number": 1,
                    "win_prob_norm": 1.0,
                    "predicted_rank": 1,
                }
            ],
        },
        lifecycle={"status": "upcoming_not_jumped"},
        prediction_timestamp="2026-05-21T15:45:00",
    )

    dry_run = persist_prediction_snapshot(snapshot, tmp_path, dry_run=True)
    assert dry_run["status"] == "dry_run"
    assert not list(tmp_path.glob("**/*.json"))

    persisted = persist_prediction_snapshot(snapshot, tmp_path)
    assert persisted["status"] == "persisted"
    assert "manifest_path" in persisted
    assert list(tmp_path.glob("**/*.json"))


def test_snapshot_rejects_result_fields_inside_runner_rows():
    with pytest.raises(ValueError):
        build_prediction_snapshot(
            {
                "race_id": "Race 4 - WRGL - 2026-05-21",
                "predictions": [{"dog_clean_name": "Alpha", "finish_position": 1}],
            },
            lifecycle={"status": "upcoming_not_jumped"},
        )


@pytest.mark.parametrize("field", ["official_result", "winner", "race_result"])
def test_snapshot_rejects_broader_result_aliases(field):
    with pytest.raises(ValueError):
        assert_no_result_fields({"predictions": [{"dog_name": "Alpha", field: "won"}]})


def test_snapshot_readiness_marks_incomplete_source_runner_set_not_ready():
    snapshot = build_prediction_snapshot(
        {
            "race_id": "Race 1 - SHEP - 2026-05-25",
            "model_version": "model-v1",
            "predictions": [
                {"dog_clean_name": "Shima Lexie", "box_number": 2, "win_prob_norm": 0.6},
                {"dog_clean_name": "Sekiro", "box_number": 4, "win_prob_norm": 0.4},
            ],
        },
        lifecycle={
            "status": "upcoming_not_jumped",
            "race_date": "2026-05-25",
            "venue": "SHEP",
            "race_number": 1,
        },
        source_runner_completeness={
            "schema_version": "runner_completeness_v1",
            "status": "INCOMPLETE",
            "runner_count": 2,
            "min_complete_runners": 4,
            "boxes": [2, 4],
            "dog_names": ["Shima Lexie", "Sekiro"],
            "participants": [
                {"box_number": 2, "dog_name": "Shima Lexie"},
                {"box_number": 4, "dog_name": "Sekiro"},
            ],
            "duplicate_boxes": [],
            "duplicate_dog_names": [],
            "invalid_runner_rows": 0,
            "reasons": ["runner_count_below_min:2<4"],
        },
        prediction_timestamp="2026-05-24T21:38:53",
    )

    assert snapshot["runner_set_complete"] is False
    assert snapshot["snapshot_readiness"]["status"] == "NOT_READY"
    assert (
        snapshot["snapshot_readiness"]["requirements"]["source_runner_set_complete"]
        is False
    )


def test_snapshot_readiness_carries_verified_final_runner_set():
    snapshot = build_prediction_snapshot(
        {
            "race_id": "Race 4 - BEN - 2026-05-27",
            "model_version": "model-v1",
            "predictions": [
                {
                    "dog_clean_name": "Alpha Runner",
                    "box_number": 1,
                    "win_prob_norm": 0.55,
                    "quality_flags": ["missing_live_odds"],
                },
                {
                    "dog_clean_name": "Bravo Runner",
                    "box_number": 2,
                    "win_prob_norm": 0.45,
                    "quality_flags": ["missing_live_odds"],
                },
            ],
        },
        lifecycle={"status": "upcoming_not_jumped"},
        source_runner_completeness={
            "schema_version": "runner_completeness_v1",
            "status": "COMPLETE",
            "runner_count": 2,
            "min_complete_runners": 2,
            "boxes": [1, 2],
            "dog_names": ["Alpha Runner", "Bravo Runner"],
            "participants": [
                {"box_number": 1, "dog_name": "Alpha Runner"},
                {"box_number": 2, "dog_name": "Bravo Runner"},
            ],
            "duplicate_boxes": [],
            "duplicate_dog_names": [],
            "invalid_runner_rows": 0,
            "reasons": [],
        },
        final_runner_set_verification={
            "schema_version": "final_runner_set_verification_v1",
            "final_runner_set_status": "verified",
            "final_runner_set_source": "canonical_pre_race_page",
            "final_runner_set_source_url": "https://www.thedogs.com.au/racing/bendigo/2026-05-27/4/example",
            "canonical_active_boxes": [1, 2],
            "source_active_boxes": [1, 2],
            "canonical_scratch_boxes": [],
            "source_reserve_boxes": [],
            "mismatch_reason": None,
        },
        prediction_timestamp="2026-05-27T15:45:00",
    )

    assert snapshot["final_runner_set_status"] == "verified"
    assert snapshot["canonical_active_boxes"] == [1, 2]
    assert snapshot["source_active_boxes"] == [1, 2]
    assert snapshot["snapshot_readiness"]["requirements"]["final_runner_set_verified"] is True
    assert snapshot["snapshot_readiness"]["status"] == "READY"


def test_snapshot_readiness_marks_unverified_final_runner_set_not_ready():
    snapshot = build_prediction_snapshot(
        {
            "race_id": "Race 4 - BEN - 2026-05-27",
            "model_version": "model-v1",
            "predictions": [
                {
                    "dog_clean_name": "Alpha Runner",
                    "box_number": 1,
                    "win_prob_norm": 1.0,
                    "quality_flags": ["missing_live_odds"],
                },
            ],
        },
        lifecycle={"status": "upcoming_not_jumped"},
        source_runner_completeness={
            "schema_version": "runner_completeness_v1",
            "status": "COMPLETE",
            "runner_count": 1,
            "min_complete_runners": 1,
            "boxes": [1],
            "dog_names": ["Alpha Runner"],
            "participants": [{"box_number": 1, "dog_name": "Alpha Runner"}],
            "duplicate_boxes": [],
            "duplicate_dog_names": [],
            "invalid_runner_rows": 0,
            "reasons": [],
        },
        final_runner_set_verification={
            "schema_version": "final_runner_set_verification_v1",
            "final_runner_set_status": "mismatch",
            "final_runner_set_source": "canonical_pre_race_page",
            "canonical_active_boxes": [1, 2],
            "source_active_boxes": [1],
            "canonical_scratch_boxes": [],
            "source_reserve_boxes": [],
            "mismatch_reason": "source_missing_active_boxes:2",
        },
        prediction_timestamp="2026-05-27T15:45:00",
    )

    assert snapshot["final_runner_set_status"] == "mismatch"
    assert snapshot["final_runner_set_mismatch_reason"] == "source_missing_active_boxes:2"
    assert snapshot["snapshot_readiness"]["requirements"]["final_runner_set_verified"] is False
    assert snapshot["snapshot_readiness"]["status"] == "NOT_READY"


def test_snapshot_persistence_requires_final_runner_set_when_requested(tmp_path):
    snapshot = build_prediction_snapshot(
        {
            "race_id": "Race 4 - BEN - 2026-05-27",
            "model_version": "model-v1",
            "predictions": [
                {
                    "dog_clean_name": "Alpha Runner",
                    "box_number": 1,
                    "win_prob_norm": 1.0,
                    "quality_flags": ["missing_live_odds"],
                },
            ],
        },
        lifecycle={"status": "upcoming_not_jumped"},
        final_runner_set_verification={
            "schema_version": "final_runner_set_verification_v1",
            "final_runner_set_status": "unavailable",
            "final_runner_set_source": "canonical_pre_race_page",
            "mismatch_reason": "no_race_runner_rows",
        },
        prediction_timestamp="2026-05-27T15:45:00",
    )

    report = persist_prediction_snapshot(
        snapshot,
        tmp_path,
        require_final_runner_verification=True,
    )

    assert report["status"] == "skipped_pre_jump_runner_set_unverified"
    assert not list(tmp_path.rglob("*.json"))
    assert not (tmp_path / "manifest.jsonl").exists()


def test_bet_readiness_marks_abstain_without_reranking_or_ev_changes():
    prediction_result = {
        "race_id": "Race 4 - WRGL - 2026-05-21",
        "model_version": "unknown",
        "ensemble_models_used": 1,
        "market_context": {"large_disagreement_count": 1},
        "predictions": [
            {
                "dog_clean_name": "Alpha",
                "win_prob_norm": 0.251,
                "predicted_rank": 1,
                "ev_win": None,
                "quality_flags": ["large_model_market_disagreement"],
            },
            {
                "dog_clean_name": "Beta",
                "win_prob_norm": 0.249,
                "predicted_rank": 2,
                "odds_win": 4.0,
                "odds_timestamp": "2026-05-21T14:30:00",
                "ev_win": -0.004,
            },
            {"dog_clean_name": "Gamma", "win_prob_norm": 0.25, "predicted_rank": 3},
            {"dog_clean_name": "Delta", "win_prob_norm": 0.25, "predicted_rank": 4},
        ],
    }

    readiness = apply_bet_readiness_gates(
        prediction_result,
        lifecycle={"status": "jumped_pending_results"},
        uniform_threshold=0.005,
        now=datetime.fromisoformat("2026-05-21T15:45:00"),
    )

    assert readiness["status"] == "prediction_available_not_bet_qualified"
    assert "jumped_pending_results" in readiness["abstain_flags"]
    assert "missing_live_odds" in readiness["abstain_flags"]
    assert "probabilities_too_uniform" in readiness["abstain_flags"]
    assert "single_model_only" in readiness["abstain_flags"]
    assert "market_model_disagreement" in readiness["abstain_flags"]
    assert "low_calibration_confidence" in readiness["abstain_flags"]
    assert "stale_live_odds" in readiness["abstain_flags"]
    assert [p["predicted_rank"] for p in prediction_result["predictions"]] == [
        1,
        2,
        3,
        4,
    ]
    assert prediction_result["predictions"][1]["ev_win"] == pytest.approx(-0.004)


def test_evaluation_checks_leakage_temporal_holdout_and_scores_market_metrics():
    assert validate_feature_columns(["dog_name", "winner_name", "finish_position"]) == [
        "finish_position",
        "winner_name",
    ]

    holdout = validate_temporal_holdout(
        [{"race_id": "R1", "race_date": "2026-05-20"}],
        [{"race_id": "R2", "race_date": "2026-05-21"}],
    )
    assert holdout.ok is True

    bad_holdout = validate_temporal_holdout(
        [{"race_id": "R1", "race_date": "2026-05-21"}],
        [{"race_id": "R1", "race_date": "2026-05-21"}],
    )
    assert bad_holdout.ok is False
    assert "race_id_overlap" in bad_holdout.violations
    assert "test_not_strictly_after_train" in bad_holdout.violations

    metrics = score_predictions(
        [
            {"race_id": "R1", "dog_name": "A", "win_prob_norm": 0.6, "actual_win": 1, "odds_win": 2.0},
            {"race_id": "R1", "dog_name": "B", "win_prob_norm": 0.4, "actual_win": 0, "odds_win": 3.0},
            {"race_id": "R2", "dog_name": "C", "win_prob_norm": 0.7, "actual_win": 0, "odds_win": 1.5},
            {"race_id": "R2", "dog_name": "D", "win_prob_norm": 0.3, "actual_win": 1, "odds_win": 5.0},
        ]
    )
    assert metrics["races_evaluated"] == 2
    assert metrics["top1"] == pytest.approx(0.5)
    assert metrics["top2"] == pytest.approx(1.0)
    assert metrics["probability_sum"]["max_abs_error"] == pytest.approx(0.0)
    assert metrics["roi_ev_by_decile"]


def test_score_predictions_can_group_frozen_snapshot_instances_separately():
    metrics = score_predictions(
        [
            {
                "race_id": "R1",
                "snapshot_instance_id": "R1|snapshot:a",
                "dog_name": "A",
                "win_prob_norm": 0.6,
                "actual_win": 1,
            },
            {
                "race_id": "R1",
                "snapshot_instance_id": "R1|snapshot:a",
                "dog_name": "B",
                "win_prob_norm": 0.4,
                "actual_win": 0,
            },
            {
                "race_id": "R1",
                "snapshot_instance_id": "R1|snapshot:b",
                "dog_name": "A",
                "win_prob_norm": 0.4,
                "actual_win": 1,
            },
            {
                "race_id": "R1",
                "snapshot_instance_id": "R1|snapshot:b",
                "dog_name": "B",
                "win_prob_norm": 0.6,
                "actual_win": 0,
            },
        ],
        race_id_key="snapshot_instance_id",
    )

    assert metrics["races_evaluated"] == 2
    assert metrics["top1"] == pytest.approx(0.5)
    assert metrics["top2"] == pytest.approx(1.0)
    assert metrics["winner_ranks"] == [1, 2]


def test_market_implied_and_blend_are_normalized_experiment_helpers():
    market = market_implied_probabilities({"A": 2.0, "B": 4.0})
    assert sum(market.values()) == pytest.approx(1.0)
    blend = blend_probabilities({"A": 0.6, "B": 0.4}, market, model_weight=0.5)
    assert sum(blend.values()) == pytest.approx(1.0)
    assert blend["A"] > blend["B"]
