import csv
import json
from pathlib import Path

import pytest

from scripts import build_scorecard_failure_attribution_packet as packet


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_scorecard_failure_attribution_writes_report_only_packet(tmp_path, monkeypatch):
    monkeypatch.setattr(packet, "ROOT", tmp_path)
    scorecard_csv = tmp_path / "scorecard.csv"
    inventory_csv = tmp_path / "inventory.csv"
    report_json = tmp_path / "source_report.json"
    _write_csv(
        scorecard_csv,
        [
            {
                "race_id": "Race 1 - WPK - 2026-06-10",
                "race_date": "2026-06-10",
                "venue": "WPK",
                "race_number": "1",
                "runner_count": "8",
                "winner_box": "1",
                "winner_dog_name": "Alpha",
                "model_winner_rank": "4",
                "model_top1_correct": "False",
                "model_top3_correct": "False",
                "model_winner_probability": "0.10",
                "model_logloss": "2.302585092994046",
                "market_winner_rank": "1",
                "market_top1_correct": "True",
                "market_top3_correct": "True",
                "market_winner_probability": "0.50",
                "market_logloss": "0.6931471805599453",
                "model_top_box": "2",
                "market_top_box": "1",
                "winner_prediction_source_path": "shadow/a.jsonl",
                "winner_prediction_raw_probability": "0.10",
            },
            {
                "race_id": "Race 2 - WPK - 2026-06-10",
                "race_date": "2026-06-10",
                "venue": "WPK",
                "race_number": "2",
                "runner_count": "8",
                "winner_box": "2",
                "winner_dog_name": "Bravo",
                "model_winner_rank": "1",
                "model_top1_correct": "True",
                "model_top3_correct": "True",
                "model_winner_probability": "0.40",
                "model_logloss": "0.916290731874155",
                "market_winner_rank": "3",
                "market_top1_correct": "False",
                "market_top3_correct": "True",
                "market_winner_probability": "0.20",
                "market_logloss": "1.6094379124341003",
                "model_top_box": "2",
                "market_top_box": "1",
                "winner_prediction_source_path": "shadow/b.jsonl",
                "winner_prediction_raw_probability": "0.40",
            },
        ],
    )
    _write_csv(
        inventory_csv,
        [
            {
                "race_id": "Race 1 - WPK - 2026-06-10",
                "official_result_duplicate_certification": "NO_OFFICIAL_RESULT_DUPLICATES",
            },
            {
                "race_id": "Race 2 - WPK - 2026-06-10",
                "official_result_duplicate_certification": "NO_OFFICIAL_RESULT_DUPLICATES",
            },
        ],
    )
    report_json.write_text(
        json.dumps(
            {
                "final_status": "RACE_EVIDENCE_INVENTORY_READY_FOR_EVALUATION",
                "recommended_decision": "RUN_POST_BACKLOG_UNIFIED_EVALUATION",
            }
        ),
        encoding="utf-8",
    )

    report = packet.build_packet(
        scorecard_csv=scorecard_csv,
        inventory_csv=inventory_csv,
        report_json=report_json,
        output_dir=tmp_path
        / "artifacts/full_evidence_orchestration_20260525/scorecard_failure_attribution_test_report_only",
        min_cluster_races=1,
    )

    assert report["final_status"] == "SCORECARD_FAILURE_ATTRIBUTION_READY"
    assert report["overall_metrics"]["race_count"] == 2
    assert report["overall_metrics"]["model_top1_accuracy"] == 0.5
    assert report["overall_metrics"]["market_top1_accuracy"] == 0.5
    assert report["box_bias_summary"]["max_model_top_box_overpick_box"] == "2"
    assert report["box_bias_summary"]["max_model_top_box_overpick_share"] == 0.5
    assert report["no_write_guarantees"]["db_write"] is False
    assert (tmp_path / report["dimension_summary_csv"]).exists()
    assert (tmp_path / report["top_error_clusters_csv"]).exists()
    assert (tmp_path / report["output_dir"] / "SUMMARY.md").exists()


def test_scorecard_failure_attribution_output_dir_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(packet, "ROOT", tmp_path)

    with pytest.raises(ValueError, match="output_dir_must_be_scorecard_failure_attribution"):
        packet.assert_output_dir_safe(
            tmp_path / "artifacts/full_evidence_orchestration_20260525/wrong_report_only"
        )


def test_scorecard_failure_attribution_output_dir_guard_rejects_symlink_escape(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(packet, "ROOT", tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}_outside"
    outside.mkdir()
    link = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525/"
        "scorecard_failure_attribution_symlink_report_only"
    )
    link.parent.mkdir(parents=True)
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        packet.assert_output_dir_safe(link)
