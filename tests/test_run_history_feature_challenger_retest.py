import json
from pathlib import Path

import pytest

from scripts.run_history_feature_challenger_retest import (
    _clone_probability,
    _load_csv,
    _load_jsonl,
    _prepare_rows,
    _select_features,
    _top_and_winner_boxes,
    assert_output_dir_safe,
    main,
)


CLEAN_DATASET = Path(
    "artifacts/full_evidence_orchestration_20260525/"
    "isolated_challenger_box_bias_study_20260602/clean_official_dataset.jsonl"
)
PACKET_CSV = Path(
    "artifacts/full_evidence_orchestration_20260525/"
    "clean_history_feature_packet_20260602/pre_race_history_feature_packet.csv"
)


def _real_joined_rows():
    if not CLEAN_DATASET.exists() or not PACKET_CSV.exists():
        pytest.skip("clean history packet artifacts are not present")
    rows, audit = _prepare_rows(_load_jsonl(CLEAN_DATASET), _load_csv(PACKET_CSV))
    assert audit["join_status"] == "PASS"
    return rows


def test_output_dir_rejects_protected_surfaces(tmp_path):
    with pytest.raises(ValueError, match="output_dir_protected:artifacts/prediction_snapshots"):
        assert_output_dir_safe(tmp_path / "artifacts" / "prediction_snapshots" / "x", tmp_path)
    with pytest.raises(ValueError, match="output_dir_protected:model_registry"):
        assert_output_dir_safe(tmp_path / "model_registry" / "x", tmp_path)
    with pytest.raises(ValueError, match="output_dir_protected:advanced_models"):
        assert_output_dir_safe(tmp_path / "advanced_models" / "x", tmp_path)


def test_output_dir_must_stay_under_full_evidence(tmp_path):
    with pytest.raises(ValueError, match="output_dir_must_be_under"):
        assert_output_dir_safe(tmp_path / "reports" / "history-study", tmp_path)


def test_prepare_rows_requires_exact_clean_packet_join():
    clean = _load_jsonl(CLEAN_DATASET)
    packet = _load_csv(PACKET_CSV)

    rows, audit = _prepare_rows(clean[:1], packet[:1])

    assert len(rows) == len(packet[:1])
    assert audit["join_status"] == "PASS"
    assert rows[0]["field_size"] >= 1


def test_prepare_rows_reports_join_mismatch():
    clean = _load_jsonl(CLEAN_DATASET)
    packet = _load_csv(PACKET_CSV)
    mismatched_packet = [dict(packet[0], dog_name=f"{packet[0]['dog_name']} mismatch")]

    rows, audit = _prepare_rows(clean[:1], mismatched_packet)

    assert rows == []
    assert audit["join_status"] == "FAIL"
    assert audit["missing_clean_key_count"] == 1


def test_feature_selection_excludes_low_coverage_without_imputation():
    rows = [row for row in _real_joined_rows() if row.get("packet") == "historical"]

    selected, details = _select_features(
        rows,
        ["prior_start_count", "avg_time_same_distance"],
        min_present=10,
    )

    assert "prior_start_count" in selected
    assert details["avg_time_same_distance"]["status"] == "EXCLUDED_TRAIN_COVERAGE_TOO_LOW"


def test_box1_collapse_is_reported():
    rows = _clone_probability(
        [row for row in _real_joined_rows() if row.get("packet") == "rolling"],
        input_key="win_prob_norm",
    )

    result = _top_and_winner_boxes(rows, probability_key="study_prob")

    assert result["top_pick_box_distribution"]["1"] >= 1
    assert result["winner_box_distribution"]
    assert result["box1_top_pick_share"] > 0.5


def test_main_refuses_label_write_approval_env(monkeypatch, tmp_path):
    monkeypatch.setenv("APPROVE_RESULT_LABEL_WRITE", "1")

    with pytest.raises(SystemExit, match="refusing_to_run_with_APPROVE_RESULT_LABEL_WRITE_set"):
        main(
            [
                "--output-dir",
                str(tmp_path / "artifacts" / "full_evidence_orchestration_20260525" / "x"),
                "--endpoint-health",
                json.dumps({}),
            ]
        )


def test_report_documents_closeout_mutation_guards():
    report = Path(
        "artifacts/full_evidence_orchestration_20260525/"
        "history_feature_challenger_retest_20260602/report.md"
    )
    if not report.exists():
        pytest.skip("history-feature challenger report is not present")

    text = report.read_text(encoding="utf-8", errors="replace").lower()

    assert "history_features_do_not_fix_box_bias" in text
    assert ("no promotion" in text) or ("not promote" in text) or ("promotion" in text)
    assert ("no betting" in text) or ("used for betting" in text)
    assert "no production writes" in text
    assert "live result-ingest writes" in text
    assert "model registry" in text
    assert "box-bias production gate remains red" in text
