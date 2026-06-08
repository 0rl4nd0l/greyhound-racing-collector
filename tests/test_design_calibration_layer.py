import json
from pathlib import Path

from scripts.design_calibration_layer import build_design


def _row(
    *,
    race_id: str,
    race_date: str,
    box_number: int,
    actual_win: int,
) -> dict:
    probs = {1: 0.7, 2: 0.2, 3: 0.1}
    return {
        "race_id": race_id,
        "race_date": race_date,
        "dog_name": f"Dog {race_id} {box_number}",
        "box_number": box_number,
        "win_prob_norm": probs[box_number],
        "actual_win": actual_win,
        "finish_position": 1 if actual_win else box_number,
        "label_quality": "official_or_complete_result",
        "result_detail_quality": "finish_position",
    }


def _write_dataset(path: Path) -> None:
    rows = []
    for race_number in range(1, 13):
        race_date = "2026-01-01" if race_number <= 6 else "2026-01-02"
        race_id = f"Race {race_number} - TEST - {race_date}"
        for box_number in (1, 2, 3):
            rows.append(
                _row(
                    race_id=race_id,
                    race_date=race_date,
                    box_number=box_number,
                    actual_win=int(box_number == 2),
                )
            )
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _review(dataset: Path) -> dict:
    return {
        "schema_version": "snapshot_challenger_review_v1",
        "status": "SUCCESS",
        "failures": [],
        "warnings": [],
        "source_evidence": {
            "evaluation_dataset": str(dataset.resolve()),
        },
        "promotion_control": {
            "action_taken": "none",
            "model_artifact_written": False,
            "registry_mutation_allowed": False,
            "promotion_allowed": False,
            "required_gate": "APPROVE_MODEL_PROMOTION",
        },
    }


def _packet(dataset: Path, review_path: Path, *, promotion_allowed=False) -> dict:
    return {
        "schema_version": "model_review_packet_v1",
        "status": "READY_FOR_CHALLENGER_REVIEW",
        "failures": [],
        "warnings": [],
        "source_evidence": {
            "evaluation_dataset": str(dataset.resolve()),
            "evaluation_dataset_rows_written": 36,
            "evaluation_dataset_rows_observed": 36,
        },
        "promotion_control": {
            "action_taken": "none",
            "registry_mutation_allowed": False,
            "promotion_allowed": promotion_allowed,
        },
        "challenger_review_gate": {
            "provided": True,
            "path": str(review_path.resolve()),
            "status": "READY",
            "failures": [],
            "candidate_arm": "power_calibrated_baseline",
            "stability_status": "STABLE_REPORT_ONLY",
            "split_count": 2,
            "failed_split_count": 0,
            "all_log_loss_improved": True,
            "all_brier_improved": True,
            "all_ranking_preserved": True,
            "selected_alpha": 0.5,
            "promotion_allowed": False,
            "registry_mutation_allowed": False,
            "model_artifact_written": False,
        },
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_calibration_layer_design_is_report_only_and_rank_preserving(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    review_path = tmp_path / "challenger_review.json"
    packet_path = tmp_path / "model_review_packet.json"
    _write_dataset(dataset)
    _write_json(review_path, _review(dataset))
    _write_json(packet_path, _packet(dataset, review_path))

    report = build_design(model_review_packet_path=packet_path)

    assert report["status"] == "READY_FOR_OPERATOR_DESIGN_REVIEW"
    assert report["failures"] == []
    assert report["runtime_transform_spec"] == {
        "candidate_arm": "power_calibrated_baseline",
        "algorithm": "power_normalize_per_race",
        "alpha": 0.5,
        "input_probability_key": "win_prob_norm",
        "output_probability_key": "calibrated_win_prob_report_only",
        "formula": "p_cal_i = p_i ** alpha / sum_j(p_j ** alpha)",
        "rank_preserving_when_alpha_positive": True,
        "uses_labels_at_runtime": False,
        "uses_odds_at_runtime": False,
        "requires_runner_complete_race_group": True,
    }
    assert report["comparison_to_baseline"]["log_loss_improved"] is True
    assert report["comparison_to_baseline"]["brier_improved"] is True
    assert report["comparison_to_baseline"]["top1_preserved"] is True
    assert report["comparison_to_baseline"]["top2_preserved"] is True
    assert report["comparison_to_baseline"]["top3_preserved"] is True
    assert report["deployment_control"] == {
        "action_taken": "none",
        "model_artifact_written": False,
        "registry_mutation_allowed": False,
        "production_config_write_allowed": False,
        "promotion_allowed": False,
        "required_gate": "APPROVE_MODEL_PROMOTION",
        "betting_allowed": False,
    }


def test_calibration_layer_design_fails_closed_when_packet_allows_promotion(
    tmp_path,
):
    dataset = tmp_path / "dataset.jsonl"
    review_path = tmp_path / "challenger_review.json"
    packet_path = tmp_path / "model_review_packet.json"
    _write_dataset(dataset)
    _write_json(review_path, _review(dataset))
    _write_json(packet_path, _packet(dataset, review_path, promotion_allowed=True))

    report = build_design(model_review_packet_path=packet_path)

    assert report["status"] == "NOT_READY"
    assert "model_review_packet_promotion_not_blocked" in report["failures"]
    assert report["deployment_control"]["promotion_allowed"] is False


def test_calibration_layer_design_fails_closed_on_dataset_scope_mismatch(
    tmp_path,
):
    dataset = tmp_path / "dataset.jsonl"
    other_dataset = tmp_path / "other_dataset.jsonl"
    review_path = tmp_path / "challenger_review.json"
    packet_path = tmp_path / "model_review_packet.json"
    _write_dataset(dataset)
    _write_dataset(other_dataset)
    _write_json(review_path, _review(other_dataset))
    _write_json(packet_path, _packet(dataset, review_path))

    report = build_design(model_review_packet_path=packet_path)

    assert report["status"] == "NOT_READY"
    assert "challenger_review_dataset_scope_mismatch" in report["failures"]
    assert report["deployment_control"]["model_artifact_written"] is False
