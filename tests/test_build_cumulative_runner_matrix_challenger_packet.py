import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.build_cumulative_runner_matrix_challenger_packet import build_packet


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_runner_matrix(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_build_packet_accepts_current_cumulative_runner_matrix(tmp_path):
    matrix = _write_runner_matrix(
        tmp_path / "runner_matrix.csv",
        [
            {
                "race_id": "Race 1 - TEST - 2026-07-01",
                "source_report": "unified_a/unified_evidence_dataset_report.json",
                "venue": "TEST",
                "race_number": 1,
                "race_date": "2026-07-01",
                "dog_name": "Alpha",
                "box_number": 1,
                "finish_position": 1,
                "odds_decimal": 2.5,
                "market_probability": 0.4,
                "primary_shadow_probability_norm": 0.35,
                "stage2_shadow_probability_norm": 0.35,
                "stage2_shadow_uncalibrated_probability_norm": 0.36,
            },
            {
                "race_id": "Race 1 - TEST - 2026-07-01",
                "source_report": "unified_a/unified_evidence_dataset_report.json",
                "venue": "TEST",
                "race_number": 1,
                "race_date": "2026-07-01",
                "dog_name": "Beta",
                "box_number": 2,
                "finish_position": 2,
                "odds_decimal": 3.0,
                "market_probability": 0.333333,
                "primary_shadow_probability_norm": 0.4,
                "stage2_shadow_probability_norm": 0.4,
                "stage2_shadow_uncalibrated_probability_norm": 0.38,
            },
            {
                "race_id": "Race 2 - TEST - 2026-07-01",
                "source_report": "unified_b/unified_evidence_dataset_report.json",
                "venue": "TEST",
                "race_number": 2,
                "race_date": "2026-07-01",
                "dog_name": "Gamma",
                "box_number": 1,
                "finish_position": 2,
                "odds_decimal": 4.0,
                "market_probability": 0.25,
                "primary_shadow_probability_norm": 0.2,
                "stage2_shadow_probability_norm": 0.2,
                "stage2_shadow_uncalibrated_probability_norm": 0.22,
            },
            {
                "race_id": "Race 2 - TEST - 2026-07-01",
                "source_report": "unified_b/unified_evidence_dataset_report.json",
                "venue": "TEST",
                "race_number": 2,
                "race_date": "2026-07-01",
                "dog_name": "Delta",
                "box_number": 2,
                "finish_position": 1,
                "odds_decimal": 1.8,
                "market_probability": 0.555556,
                "primary_shadow_probability_norm": 0.6,
                "stage2_shadow_probability_norm": 0.6,
                "stage2_shadow_uncalibrated_probability_norm": 0.58,
            },
        ],
    )
    rolling_report = _write_json(
        tmp_path / "rolling_model_comparison_report.json",
        {
            "schema_version": "rolling_model_comparison_report_v1",
            "final_status": "ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW",
            "sample_floor_met": True,
            "sample_race_count": 2,
            "sample_runner_rows": 4,
            "minimum_races_for_review": 100,
            "candidate_count": 54,
            "market_candidate_key": "market_only_implied",
            "best_non_market_candidate_key": "stage2_uncalibrated_market_blend_70",
            "best_non_market_minus_market": {"top1": 0.0},
            "market_residual_runner_matrix_csv": str(matrix),
            "source_unified_evidence_reports": [
                "unified_a/unified_evidence_dataset_report.json",
                "unified_b/unified_evidence_dataset_report.json",
            ],
        },
    )

    packet = build_packet(
        rolling_report_path=rolling_report,
        output_dir=tmp_path / "adapter",
        now=datetime(2026, 7, 2, tzinfo=timezone.utc),
    )

    assert packet["schema_version"] == "cumulative_runner_matrix_challenger_packet_v1"
    assert packet["status"] == "READY_FOR_REPORT_ONLY_CHALLENGER"
    assert packet["blockers"] == []
    assert packet["no_write_guarantees"]["model_fit"] is False
    assert packet["counts"]["runner_matrix_race_count"] == 2
    assert packet["counts"]["runner_matrix_rows"] == 4
    assert packet["counts"]["complete_valid_odds_races"] == 2
    assert packet["counts"]["official_result_joined_races"] == 2
    assert packet["counts"]["source_report_count"] == 2
    assert packet["counts"]["source_unified_evidence_report_count"] == 2
    assert packet["readiness"] == {
        "race_count_match": True,
        "runner_row_count_match": True,
        "complete_market_comparable_status": "READY",
        "missing_required_runner_fields": [],
    }
    assert Path(packet["paths"]["race_table_csv"]).exists()
    assert Path(packet["paths"]["race_table_jsonl"]).exists()


def test_build_packet_fails_closed_when_matrix_does_not_match_rolling_counts(tmp_path):
    matrix = _write_runner_matrix(
        tmp_path / "runner_matrix.csv",
        [
            {
                "race_id": "Race 1 - TEST - 2026-07-01",
                "source_report": "unified_a/unified_evidence_dataset_report.json",
                "venue": "TEST",
                "race_number": 1,
                "race_date": "2026-07-01",
                "dog_name": "Alpha",
                "box_number": 1,
                "finish_position": "",
                "odds_decimal": "",
                "market_probability": 0.4,
                "primary_shadow_probability_norm": 0.35,
                "stage2_shadow_probability_norm": 0.35,
                "stage2_shadow_uncalibrated_probability_norm": 0.36,
            }
        ],
    )
    rolling_report = _write_json(
        tmp_path / "rolling_model_comparison_report.json",
        {
            "sample_race_count": 2,
            "sample_runner_rows": 2,
            "market_residual_runner_matrix_csv": str(matrix),
            "source_unified_evidence_reports": [],
        },
    )

    packet = build_packet(
        rolling_report_path=rolling_report,
        output_dir=tmp_path / "adapter",
        now=datetime(2026, 7, 2, tzinfo=timezone.utc),
    )

    assert packet["status"] == "DATA_MISSING"
    assert packet["blockers"] == [
        "runner_matrix_required_fields_missing",
        "runner_matrix_row_count_mismatch",
        "runner_matrix_race_count_mismatch",
        "runner_matrix_not_complete_market_comparable",
    ]
    assert packet["readiness"]["complete_market_comparable_status"] == "DATA_MISSING"


def test_build_packet_rejects_protected_output_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    matrix = _write_runner_matrix(
        tmp_path / "runner_matrix.csv",
        [
            {
                "race_id": "Race 1 - TEST - 2026-07-01",
                "source_report": "unified_a/unified_evidence_dataset_report.json",
                "venue": "TEST",
                "race_number": 1,
                "race_date": "2026-07-01",
                "dog_name": "Alpha",
                "box_number": 1,
                "finish_position": 1,
                "odds_decimal": 2.5,
                "market_probability": 0.4,
                "primary_shadow_probability_norm": 0.35,
                "stage2_shadow_probability_norm": 0.35,
                "stage2_shadow_uncalibrated_probability_norm": 0.36,
            }
        ],
    )
    rolling_report = _write_json(
        tmp_path / "rolling_model_comparison_report.json",
        {
            "sample_race_count": 1,
            "sample_runner_rows": 1,
            "market_residual_runner_matrix_csv": str(matrix),
        },
    )

    with pytest.raises(ValueError, match="output_dir_protected"):
        build_packet(
            rolling_report_path=rolling_report,
            output_dir=tmp_path / "model_registry/bad",
        )
