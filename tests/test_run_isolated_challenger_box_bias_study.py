import json
from pathlib import Path

import pytest

from scripts.run_isolated_challenger_box_bias_study import (
    _leakage_audit,
    _not_run,
    _top_and_winner_boxes,
    _write_report,
    assert_output_dir_safe,
    build_primary_split,
    main,
    parse_snapshot_manifest_line,
    rows_from_evaluation_datasets,
)


def test_parse_snapshot_manifest_line_accepts_raw_path():
    assert parse_snapshot_manifest_line("artifacts/prediction_snapshots/a.json\n") == (
        "artifacts/prediction_snapshots/a.json"
    )


def test_parse_snapshot_manifest_line_accepts_snapshot_jsonl():
    line = json.dumps(
        {
            "schema_version": "prediction_snapshot_manifest_v1",
            "snapshot_path": "artifacts/prediction_snapshots/b.json",
        }
    )
    assert parse_snapshot_manifest_line(line) == "artifacts/prediction_snapshots/b.json"


def test_output_dir_rejects_production_surfaces(tmp_path):
    repo_root = tmp_path
    with pytest.raises(ValueError, match="output_dir_protected:model_registry"):
        assert_output_dir_safe(repo_root / "model_registry" / "study", repo_root)
    with pytest.raises(ValueError, match="output_dir_protected:docs/model_registry"):
        assert_output_dir_safe(repo_root / "docs" / "model_registry" / "study", repo_root)
    with pytest.raises(ValueError, match="output_dir_protected:ml_models_v4"):
        assert_output_dir_safe(repo_root / "ml_models_v4" / "study", repo_root)
    with pytest.raises(ValueError, match="output_dir_protected:artifacts/prediction_snapshots"):
        assert_output_dir_safe(
            repo_root / "artifacts" / "prediction_snapshots" / "study",
            repo_root,
        )


def test_output_dir_must_stay_under_full_evidence_artifacts(tmp_path):
    with pytest.raises(ValueError, match="output_dir_must_be_under"):
        assert_output_dir_safe(tmp_path / "reports" / "study", tmp_path)


def test_primary_split_uses_historical_train_and_rolling_eval():
    rows = [
        {
            "snapshot_path": "artifacts/full_evidence_orchestration_20260525/historical_replay_backfill_20260531T210114AEST/snapshots/a.json",
            "snapshot_instance_id": "hist-a",
            "race_date": "2026-05-21",
        },
        {
            "snapshot_path": "artifacts/prediction_snapshots/2026-05-31/VENUE/b.json",
            "snapshot_instance_id": "roll-b",
            "race_date": "2026-05-31",
        },
    ]

    split = build_primary_split(rows)

    assert split["strategy"] == "historical_train_rolling_holdout"
    assert [row["snapshot_instance_id"] for row in split["train_rows"]] == ["hist-a"]
    assert [row["snapshot_instance_id"] for row in split["eval_rows"]] == ["roll-b"]


def test_rows_from_evaluation_datasets_loads_jsonl(tmp_path):
    path = tmp_path / "dataset.jsonl"
    path.write_text(
        json.dumps({"snapshot_instance_id": "a", "dog_name": "Dog A"}) + "\n",
        encoding="utf-8",
    )

    rows = rows_from_evaluation_datasets([path])

    assert rows == [{"snapshot_instance_id": "a", "dog_name": "Dog A"}]


def test_main_refuses_label_write_approval_env(monkeypatch):
    monkeypatch.setenv("APPROVE_RESULT_LABEL_WRITE", "1")

    with pytest.raises(SystemExit, match="refusing_to_run_with_APPROVE_RESULT_LABEL_WRITE_set"):
        main(["--output-dir", "artifacts/full_evidence_orchestration_20260525/test_study"])


def test_leakage_audit_reports_clean_temporal_split():
    train_rows = [
        {
            "race_id": "train-race",
            "snapshot_instance_id": "train",
            "race_date": "2026-05-21",
        }
    ]
    eval_rows = [
        {
            "race_id": "eval-race",
            "snapshot_instance_id": "eval",
            "race_date": "2026-05-26",
        }
    ]

    audit = _leakage_audit(
        train_rows=train_rows,
        eval_rows=eval_rows,
        feature_families={
            "champion_current_production_scoring": ["production_snapshot_win_prob_norm"]
        },
    )

    assert audit["status"] == "PASS"
    assert audit["temporal_holdout"]["train_max_date"] == "2026-05-21"
    assert audit["temporal_holdout"]["test_min_date"] == "2026-05-26"
    assert audit["temporal_holdout"]["race_id_overlap"] == []
    assert audit["post_outcome_feature_columns_used"] == []
    assert audit["snapshot_policy"] == (
        "read existing frozen snapshots only; no snapshot writes or rewrites"
    )


def test_not_run_variant_blocks_artifacts_and_registry_mutation():
    blocked = _not_run("history_only_model", "missing clean historical feature columns")

    assert blocked == {
        "status": "NOT_RUN",
        "variant": "history_only_model",
        "blocker": "missing clean historical feature columns",
        "promotion_allowed": False,
        "registry_mutation_allowed": False,
        "model_artifact_written": False,
    }


def test_box1_collapse_is_reported_not_hidden():
    rows = [
        {
            "snapshot_instance_id": "race-a",
            "dog_name": "Inside A",
            "box_number": 1,
            "actual_win": 0,
            "study_prob": 0.70,
        },
        {
            "snapshot_instance_id": "race-a",
            "dog_name": "Outside A",
            "box_number": 8,
            "actual_win": 1,
            "study_prob": 0.30,
        },
        {
            "snapshot_instance_id": "race-b",
            "dog_name": "Inside B",
            "box_number": 1,
            "actual_win": 1,
            "study_prob": 0.80,
        },
        {
            "snapshot_instance_id": "race-b",
            "dog_name": "Middle B",
            "box_number": 4,
            "actual_win": 0,
            "study_prob": 0.20,
        },
    ]

    box_bias = _top_and_winner_boxes(rows, probability_key="study_prob")

    assert box_bias["top_pick_box_distribution"] == {"1": 2}
    assert box_bias["winner_box_distribution"] == {"1": 1, "8": 1}
    assert box_bias["box1_top_pick_share"] == 1.0


def test_report_preserves_no_mutation_and_no_promotion_conclusion(tmp_path):
    report = tmp_path / "report.md"

    _write_report(
        report,
        manifest={
            "race_count": 132,
            "snapshot_instance_count": 134,
            "runner_row_count": 943,
            "date_range": {"min": "2026-05-01", "max": "2026-06-02"},
            "source_groups": {
                "historical_clean_official_packet": 105,
                "rolling_persisted_snapshot_corpus": 29,
            },
            "excluded_reason_counts": {"partial_official_result_positions": 3},
        },
        baseline={
            "historical_clean_official_packet": {
                "metrics": {
                    "top1": 0.1619,
                    "top3": 0.4857,
                    "mean_winner_rank": 3.9143,
                },
                "box_bias": {"top_pick_box_distribution": {"1": 105}},
            },
            "rolling_clean_official_packet": {
                "metrics": {
                    "top1": 0.1379,
                    "top3": 0.4828,
                    "mean_winner_rank": 3.9655,
                },
                "box_bias": {"top_pick_box_distribution": {"1": 27, "2": 2}},
            },
        },
        challengers={
            "history_only_model": _not_run(
                "history_only_model",
                "clean snapshot evaluation rows do not contain historical performance feature columns",
            )
        },
        comparison_rows=[
            {
                "variant": "history_only_model",
                "status": "NOT_RUN",
                "scope": "DATA_MISSING",
                "races": "DATA_MISSING",
                "top1": "DATA_MISSING",
                "top3": "DATA_MISSING",
                "brier": "DATA_MISSING",
                "log_loss": "DATA_MISSING",
                "box1_top_pick_share": "DATA_MISSING",
                "blocker": (
                    "clean snapshot evaluation rows do not contain historical "
                    "performance feature columns"
                ),
            }
        ],
        leakage={
            "status": "PASS",
            "temporal_holdout": {
                "ok": True,
                "train_max_date": "2026-05-21",
                "test_min_date": "2026-05-26",
                "race_id_overlap": [],
                "violations": [],
            },
        },
        endpoint={
            "api_health_error": "connection refused",
            "api_model_health_error": "connection refused",
        },
        sqlite_quick_check="ok",
    )

    text = report.read_text(encoding="utf-8")
    assert "Recommendation: `NO_PROMOTION_MORE_DATA_NEEDED`." in text
    assert "Do not promote, bet, or infer EV edge from this report." in text
    assert "No production model registry" in text
    assert "snapshot JSON, manifest, labels, odds capture" in text
    assert "Any `NOT_RUN` variant is blocked rather than substituted with fake data." in text
