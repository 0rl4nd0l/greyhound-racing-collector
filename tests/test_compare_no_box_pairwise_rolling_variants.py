import json
from pathlib import Path

import pytest

from scripts.compare_no_box_pairwise_rolling_variants import (
    build_variant_comparison_packet,
    main,
    write_outputs,
)


WRITES_PERFORMED = {
    "db_write": False,
    "label_write": False,
    "metadata_write": False,
    "official_fetch": False,
    "snapshot_mutation": False,
    "manifest_mutation": False,
    "dataset_regeneration": False,
    "model_training": False,
    "model_persistence": False,
    "registry_mutation": False,
    "promotion": False,
    "tgr_enablement": False,
    "betting_decision": False,
    "ev_action": False,
}


def _report(
    *,
    status: str = "REPORT_ONLY_PAIRWISE_ROLLING_WINDOWS_EVALUATED",
    top1: float | None = 0.2,
    top3: float | None = 0.6,
    mean_rank: float | None = 2.5,
    usable_features: int = 4,
    write_flag: bool = False,
) -> dict:
    writes = dict(WRITES_PERFORMED)
    if write_flag:
        writes["model_persistence"] = True
    return {
        "schema_version": "no_box_pairwise_rolling_windows_v1",
        "status": status,
        "report_only": True,
        "writes_performed": writes,
        "source_packet": {
            "status": "REPORT_ONLY_DOG_FORM_FEATURE_JOIN_READY",
            "history_db_fill_policy": "no_outcome_proxy_fields",
            "label_proxy_audit_status": "PASS",
        },
        "validation": {
            "status": "PASS",
            "race_count": 37,
            "row_count": 167,
            "usable_feature_count": usable_features,
            "complete_field_races": 2,
        },
        "aggregate_metrics": None
        if top1 is None
        else {
            "race_count": 20,
            "top1_accuracy": top1,
            "top3_hit_rate": top3,
            "mean_winner_rank": mean_rank,
            "expected_random_top1": 0.25,
            "expected_random_top3": 0.75,
        },
        "window_metric_summary": None
        if top1 is None
        else {
            "window_count": 4,
            "top1_range": 0.2,
            "top3_range": 0.2,
        },
        "race_grouped_complete_field_gate": {
            "status": "SKIPPED_INSUFFICIENT_COMPLETE_FIELD_RACES",
            "complete_field_races": 2,
        },
        "rolling_window_policy": {
            "reserved_final_races": 5,
            "reserved_races_predicted": False,
        },
        "sample_size_status": "UNDERPOWERED_BELOW_50_ACTUAL_WIN_RACES",
    }


def test_variant_comparison_ranks_masked_history_over_plain_without_promotion():
    packet = build_variant_comparison_packet(
        variant_reports={
            "plain": ("plain.json", _report(top1=0.2, top3=0.6, mean_rank=2.5)),
            "history_masked": (
                "history_masked.json",
                _report(top1=0.5, top3=0.95, mean_rank=1.8, usable_features=40),
            ),
            "history_enriched_rejected": (
                "history_enriched_rejected.json",
                _report(
                    status="REPORT_ONLY_PAIRWISE_ROLLING_WINDOWS_REJECTED_LEAKAGE_RISK",
                    top1=None,
                    top3=None,
                    mean_rank=None,
                    usable_features=42,
                ),
            ),
        },
        baseline_key="plain",
    )

    assert packet["status"] == "REPORT_ONLY_NO_BOX_PAIRWISE_ROLLING_VARIANT_COMPARISON"
    assert packet["safe_to_write_now"] is False
    assert packet["model_promotion_allowed"] is False
    assert packet["writes_performed"]["model_persistence"] is False
    assert packet["summary"]["best_diagnostic_variant_key"] == "history_masked"
    assert packet["summary"]["best_top1_delta_vs_baseline"] == 0.3
    assert packet["summary"]["best_top3_delta_vs_baseline"] == 0.35
    assert (
        packet["summary"]["history_feature_gain_status"]
        == "PROMISING_UNDERPOWERED_DIAGNOSTIC"
    )
    by_key = {row["variant_key"]: row for row in packet["variant_rows"]}
    assert by_key["history_masked"]["diagnostic_rank"] == 1
    assert by_key["plain"]["diagnostic_rank"] == 2
    assert by_key["history_enriched_rejected"]["diagnostic_rank"] is None


def test_variant_comparison_records_write_flag_failures():
    packet = build_variant_comparison_packet(
        variant_reports={"bad": ("bad.json", _report(write_flag=True))},
    )

    assert (
        packet["status"]
        == "REPORT_ONLY_NO_BOX_PAIRWISE_ROLLING_VARIANT_COMPARISON_WITH_FAILURES"
    )
    assert packet["failures"] == ["bad:write_flags_true:model_persistence"]
    assert "write_flags_true:model_persistence" in packet["variant_rows"][0]["blocking_reasons"]


def test_variant_comparison_cli_writes_outputs(tmp_path: Path, monkeypatch):
    import scripts.evaluate_no_box_pairwise_ranking_smoke as ranking_module

    monkeypatch.setattr(ranking_module, "ROOT", tmp_path)
    plain = tmp_path / "plain.json"
    masked = tmp_path / "masked.json"
    plain.write_text(json.dumps(_report(top1=0.2, top3=0.6)), encoding="utf-8")
    masked.write_text(json.dumps(_report(top1=0.5, top3=0.95)), encoding="utf-8")
    output_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/comparison"

    exit_code = main(
        [
            "--variant",
            f"plain={plain}",
            "--variant",
            f"history_masked={masked}",
            "--baseline-key",
            "plain",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    payload = json.loads(
        (output_dir / "no_box_pairwise_rolling_variant_comparison.json").read_text()
    )
    assert payload["summary"]["best_diagnostic_variant_key"] == "history_masked"
    assert (output_dir / "no_box_pairwise_rolling_variant_comparison.csv").exists()
    assert "No DB rows" in (output_dir / "SUMMARY.md").read_text(encoding="utf-8")

    cwd = tmp_path / "caller_cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    relative_output_dir = Path(
        "artifacts/full_evidence_orchestration_20260525/relative_comparison"
    )
    exit_code = main(
        [
            "--variant",
            f"plain={plain}",
            "--variant",
            f"history_masked={masked}",
            "--baseline-key",
            "plain",
            "--output-dir",
            str(relative_output_dir),
        ]
    )

    assert exit_code == 0
    assert (
        tmp_path
        / relative_output_dir
        / "no_box_pairwise_rolling_variant_comparison.json"
    ).exists()
    assert not (
        cwd
        / relative_output_dir
        / "no_box_pairwise_rolling_variant_comparison.json"
    ).exists()


def test_variant_comparison_output_guard_fails_closed(tmp_path: Path, monkeypatch):
    import scripts.evaluate_no_box_pairwise_ranking_smoke as ranking_module

    monkeypatch.setattr(ranking_module, "ROOT", tmp_path)
    packet = build_variant_comparison_packet(
        variant_reports={"plain": ("plain.json", _report(top1=0.2, top3=0.6))},
        baseline_key="plain",
    )

    with pytest.raises(ValueError, match="output_dir_must_be_under_artifacts"):
        write_outputs(tmp_path / "outside", packet)
    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        write_outputs(
            tmp_path.parent
            / "outside"
            / "artifacts/full_evidence_orchestration_20260525/comparison",
            packet,
        )
