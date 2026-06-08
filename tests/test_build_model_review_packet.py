import json
from pathlib import Path

from scripts.build_model_review_packet import build_packet


def _evaluation_report(dataset_path: Path, clean_races: int = 105) -> dict:
    return {
        "status": "SUCCESS",
        "runner_rows_scored": 773,
        "evaluation_dataset_output": str(dataset_path),
        "evaluation_dataset_rows_written": 3,
        "clean_official_evaluation": {
            "races_evaluated": clean_races,
            "snapshot_instances_evaluated": clean_races,
            "runner_rows_evaluated": 735,
            "metrics_by_arm": {
                "model_only": {
                    "top1": 0.16,
                    "top2": 0.33,
                    "top3": 0.48,
                    "log_loss": 1.95,
                    "brier": 0.123,
                    "mean_winner_rank": 3.9,
                    "races_evaluated": clean_races,
                    "dog_predictions_evaluated": 735,
                }
            },
        },
        "model_quality_diagnosis": {
            "status": "SUCCESS",
            "retrain_gate": {
                "status": "READY_FOR_REVIEW",
                "minimum_clean_evaluated_races": 100,
                "clean_official_evaluated_races": clean_races,
                "reason": None,
                "action_taken": "none",
            },
            "promotion_gate": {
                "status": "REPORT_ONLY",
                "reason": "promotion requires a separately approved challenger evaluation",
                "action_taken": "none",
            },
        },
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _challenger_review(dataset_path: Path) -> dict:
    return {
        "schema_version": "snapshot_challenger_review_v1",
        "status": "SUCCESS",
        "failures": [],
        "warnings": [],
        "source_evidence": {
            "evaluation_dataset": str(dataset_path.resolve()),
            "rows_loaded": 3,
            "clean_official_rows": 3,
            "clean_official_races": 3,
        },
        "challenger_training": {
            "model_artifact_written": False,
            "registry_mutation_allowed": False,
            "power_calibration": {
                "selected_alpha": 0.5,
                "model_artifact_written": False,
                "registry_mutation_allowed": False,
            },
        },
        "stability_review": {
            "status": "STABLE_REPORT_ONLY",
            "candidate_arm": "power_calibrated_baseline",
            "minimum_split_count": 2,
            "split_count": 2,
            "failed_split_count": 0,
            "all_log_loss_improved": True,
            "all_brier_improved": True,
            "all_ranking_preserved": True,
            "promotion_allowed": False,
        },
        "promotion_control": {
            "action_taken": "none",
            "model_artifact_written": False,
            "registry_mutation_allowed": False,
            "promotion_allowed": False,
            "required_gate": "APPROVE_MODEL_PROMOTION",
        },
    }


def test_build_packet_ready_for_challenger_review(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n{}\n{}\n", encoding="utf-8")
    report_path = tmp_path / "eval.json"
    _write_json(report_path, _evaluation_report(dataset))

    packet = build_packet(evaluation_report_path=report_path, repo_root=tmp_path)

    assert packet["status"] == "READY_FOR_CHALLENGER_REVIEW"
    assert packet["failures"] == []
    assert packet["review_gate"]["clean_official_evaluated_races"] == 105
    assert packet["baseline_model_metrics"]["top1"] == 0.16
    assert packet["promotion_control"] == {
        "action_taken": "none",
        "registry_mutation_allowed": False,
        "promotion_allowed": False,
        "reason": (
            "challenger training/evaluation must run separately and beat this "
            "baseline on clean held-out evidence before any promotion approval"
        ),
    }
    assert packet["next_review_steps"][2]["status"] == (
        "WAITING_FOR_STABLE_REPORT_ONLY_CHALLENGER_REVIEW"
    )
    assert packet["next_review_steps"][3]["status"] == "BLOCKED"


def test_build_packet_embeds_stable_report_only_challenger_review(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n{}\n{}\n", encoding="utf-8")
    report_path = tmp_path / "eval.json"
    review_path = tmp_path / "challenger_review.json"
    _write_json(report_path, _evaluation_report(dataset))
    _write_json(review_path, _challenger_review(dataset))

    packet = build_packet(
        evaluation_report_path=report_path,
        dataset_path=dataset,
        challenger_review_path=review_path,
        repo_root=tmp_path,
    )

    gate = packet["challenger_review_gate"]
    assert packet["status"] == "READY_FOR_CHALLENGER_REVIEW"
    assert gate["status"] == "READY"
    assert gate["candidate_arm"] == "power_calibrated_baseline"
    assert gate["split_count"] == 2
    assert gate["all_ranking_preserved"] is True
    assert gate["promotion_allowed"] is False
    assert gate["registry_mutation_allowed"] is False
    assert gate["model_artifact_written"] is False
    assert packet["next_review_steps"][2]["status"] == (
        "READY_FOR_SEPARATE_DESIGN_REVIEW"
    )
    assert packet["next_review_steps"][3]["status"] == "BLOCKED"


def test_build_packet_fails_closed_on_challenger_dataset_mismatch(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    other_dataset = tmp_path / "other.jsonl"
    dataset.write_text("{}\n{}\n{}\n", encoding="utf-8")
    other_dataset.write_text("{}\n{}\n{}\n", encoding="utf-8")
    report_path = tmp_path / "eval.json"
    review_path = tmp_path / "challenger_review.json"
    _write_json(report_path, _evaluation_report(dataset))
    _write_json(review_path, _challenger_review(other_dataset))

    packet = build_packet(
        evaluation_report_path=report_path,
        dataset_path=dataset,
        challenger_review_path=review_path,
        repo_root=tmp_path,
    )

    assert packet["status"] == "NOT_READY"
    assert "challenger_review_gate_not_ready" in packet["failures"]
    assert "challenger_review_dataset_scope_mismatch" in packet[
        "challenger_review_gate"
    ]["failures"]
    assert packet["promotion_control"]["promotion_allowed"] is False


def test_build_packet_fails_closed_when_clean_race_gate_not_met(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n{}\n{}\n", encoding="utf-8")
    report_path = tmp_path / "eval.json"
    _write_json(report_path, _evaluation_report(dataset, clean_races=99))

    packet = build_packet(evaluation_report_path=report_path, repo_root=tmp_path)

    assert packet["status"] == "NOT_READY"
    assert "insufficient_clean_official_races" in packet["failures"]
    assert "insufficient_clean_snapshot_instances" in packet["failures"]
    assert packet["promotion_control"]["promotion_allowed"] is False


def test_build_packet_fails_closed_when_dataset_count_mismatches(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n{}\n", encoding="utf-8")
    report_path = tmp_path / "eval.json"
    _write_json(report_path, _evaluation_report(dataset))

    packet = build_packet(evaluation_report_path=report_path, repo_root=tmp_path)

    assert packet["status"] == "NOT_READY"
    assert "evaluation_dataset_row_count_mismatch" in packet["failures"]
    assert packet["source_evidence"]["evaluation_dataset_rows_observed"] == 2
