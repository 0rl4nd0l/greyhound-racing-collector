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
    assert "actual_results" not in snapshot
    assert "actual_results" not in snapshot["predictions"][0]


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
    assert "missing_odds_timestamp" in runner["data_quality_flags"]
    assert "invalid_pre_jump_odds" in runner["data_quality_flags"]


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


def test_market_implied_and_blend_are_normalized_experiment_helpers():
    market = market_implied_probabilities({"A": 2.0, "B": 4.0})
    assert sum(market.values()) == pytest.approx(1.0)
    blend = blend_probabilities({"A": 0.6, "B": 0.4}, market, model_weight=0.5)
    assert sum(blend.values()) == pytest.approx(1.0)
    assert blend["A"] > blend["B"]
