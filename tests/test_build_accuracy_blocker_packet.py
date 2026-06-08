import json
from pathlib import Path

from scripts.build_accuracy_blocker_packet import (
    LEGACY_MODEL_FIT_ROWS_KEY,
    LEGACY_VALID_ODDS_FIT_RACES_KEY,
    build_packet,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_accuracy_blocker_packet_separates_readiness_and_challenger_inputs(tmp_path):
    snapshot_audit = _write_json(
        tmp_path / "snapshot_audit.json",
        {
            "schema_version": "prediction_snapshot_readiness_box_bias_audit_v1",
            "manifest_path": "artifacts/prediction_snapshots/manifest.jsonl",
            "counts": {
                "manifest_rows": 123,
                "latest_ready_races": 123,
                "latest_ready_result_label_candidate_like": 31,
            },
            "gate": {
                "status": "FAIL",
                "reason": "box1_share_over_threshold",
                "box1_share": 112 / 123,
                "box1_max_share": 0.5,
                "evaluated_latest_ready_races": 123,
            },
            "skip_reason_counts": {"snapshot_readiness_not_ready": 4},
            "latest_ready_records": [
                {
                    "race_id": "Race 1 - TEST - 2026-06-01",
                    "snapshot_path": "snapshots/race1.json",
                    "top_pick_box": 1,
                    "target_distance": 520,
                }
            ],
        },
    )
    result_report = _write_json(
        tmp_path / "result_dry_run.json",
        {
            "schema_version": "official_result_ingest_report_v1",
            "status": "SUCCESS",
            "dry_run": True,
            "clean_for_label_write": False,
            "candidate_count": 4,
            "ingested_count": 1,
            "failed_count": 1,
            "skipped": [
                {
                    "race_id": "Race 2 - TEST - 2026-06-01",
                    "reason": "race_not_jumped:upcoming_not_jumped",
                }
            ],
            "failed": [
                {
                    "race_id": "Race 3 - TEST - 2026-06-01",
                    "error": "missing_first_place_result",
                    "attempts": [
                        {
                            "source": "thedogs_official",
                            "source_url": "https://www.thedogs.test/race3",
                            "error": "missing_first_place_result",
                        }
                    ],
                }
            ],
            "label_write_blockers": [
                {
                    "race_id": "Race 3 - TEST - 2026-06-01",
                    "source": "thedogs_official",
                    "reason": "label_write_requires_complete_official_result",
                }
            ],
            "ingested": [
                {
                    "race_id": "Race 1 - TEST - 2026-06-01",
                    "source": "thedogs_official",
                    "status": "resulted",
                    "box_order": [1, 2, 3, 4],
                    "dry_run": True,
                }
            ],
        },
    )
    label_readiness = _write_json(
        tmp_path / "label_readiness.json",
        {
            "schema_version": "result_label_write_readiness_validation_v1",
            "status": "NOT_READY",
            "write_performed": False,
            "candidate_count_loaded_for_write_scope": 1,
            "skipped_before_write_scope_validation": [
                {
                    "race_id": "Race 4 - TEST - 2026-06-01",
                    "reason": "ready_prejump_snapshot_required",
                }
            ],
            "dry_run_report_gate": {
                "approved": False,
                "reason": "report_not_clean_for_label_write",
            },
        },
    )
    challenger_inputs = _write_json(
        tmp_path / "challenger_inputs.json",
        {
            "schema_version": "box_bias_study_data_inventory_v1",
            "status": "SUCCESS",
            "clean_official_races": 132,
            "runner_rows": 943,
            "near_duplicate_non_box_peer_rows": 763,
            "all_zero_feature_columns": ["tgr_win_rate", "tgr_place_rate"],
            "unsupported_variants": {"market_blend": "BLOCKED_UNDERPOWERED"},
            "output_paths": {
                "evaluation_dataset": "artifacts/eval/evaluation_dataset.jsonl",
                "pre_race_history_feature_packet": (
                    "artifacts/full_evidence_orchestration_20260525/"
                    "pre_race_history_feature_packet.csv"
                ),
            },
        },
    )

    packet = build_packet(
        snapshot_audit_path=snapshot_audit,
        result_dry_run_report_path=result_report,
        label_readiness_path=label_readiness,
        challenger_inputs_path=challenger_inputs,
    )

    assert packet["schema_version"] == "accuracy_blocker_packet_v1"
    assert packet["status"] == "BLOCKED"
    assert packet["writes_performed"] == {
        "snapshot_persist": False,
        "result_label_write": False,
        "live_odds_capture": False,
        "model_artifact_write": False,
        "registry_mutation": False,
        "model_refit": False,
        "promotion": False,
        "betting": False,
    }
    assert packet["prediction_ready_inventory"]["latest_ready_races"] == 123
    assert packet["prediction_ready_inventory"]["box1_share"] == 112 / 123
    assert packet["readiness_by_stage"] == {
        "prediction_ready": {
            "status": "READY",
            "count": 123,
            "skip_reason_counts": {"snapshot_readiness_not_ready": 4},
        },
        "result_parse_ready": {
            "status": "NOT_READY",
            "count": 1,
            "skip_reason_counts": {"missing_first_place_result": 1},
        },
        "label_write_ready": {
            "status": "NOT_READY",
            "count": 1,
            "skip_reason_counts": {
                "label_write_requires_complete_official_result": 1,
                "ready_prejump_snapshot_required": 1,
                "report_not_clean_for_label_write": 1,
            },
        },
    }
    assert packet["official_parser_failure_examples"] == [
        {
            "race_id": "Race 3 - TEST - 2026-06-01",
            "source": "thedogs_official",
            "source_url": "https://www.thedogs.test/race3",
            "error": "missing_first_place_result",
        }
    ]
    assert packet["challenger_matrix_inputs"]["clean_official_races"] == 132
    assert packet["challenger_matrix_inputs"]["all_zero_feature_columns"] == [
        "tgr_win_rate",
        "tgr_place_rate",
    ]
    assert packet["promotion_gate"]["status"] == "BLOCKED"
    assert packet["promotion_gate"]["required_human_approval"] is True


def test_accuracy_blocker_packet_fails_closed_when_required_artifact_is_missing(tmp_path):
    snapshot_audit = _write_json(
        tmp_path / "snapshot_audit.json",
        {
            "schema_version": "prediction_snapshot_readiness_box_bias_audit_v1",
            "counts": {"manifest_rows": 0, "latest_ready_races": 0},
            "gate": {"status": "DATA_MISSING", "box1_share": None},
            "skip_reason_counts": {},
        },
    )

    packet = build_packet(
        snapshot_audit_path=snapshot_audit,
        result_dry_run_report_path=tmp_path / "missing_result_report.json",
        label_readiness_path=tmp_path / "missing_label_readiness.json",
        challenger_inputs_path=tmp_path / "missing_challenger_inputs.json",
    )

    assert packet["status"] == "DATA_MISSING"
    assert packet["failures"] == [
        "result_dry_run_report_missing",
        "label_readiness_missing",
        "challenger_inputs_missing",
    ]
    assert packet["readiness_by_stage"]["result_parse_ready"] == {
        "status": "DATA_MISSING",
        "count": 0,
        "skip_reason_counts": {},
    }
    assert packet["readiness_by_stage"]["label_write_ready"] == {
        "status": "DATA_MISSING",
        "count": 0,
        "skip_reason_counts": {},
    }
    assert packet["writes_performed"]["result_label_write"] is False
    assert packet["promotion_gate"]["status"] == "BLOCKED"


def test_accuracy_blocker_packet_accepts_history_feature_challenger_inventory(tmp_path):
    snapshot_audit = _write_json(
        tmp_path / "snapshot_audit.json",
        {
            "schema_version": "prediction_snapshot_readiness_box_bias_audit_v1",
            "counts": {"manifest_rows": 1, "latest_ready_races": 1},
            "gate": {"status": "FAIL", "box1_share": 1.0},
            "skip_reason_counts": {},
        },
    )
    result_report = _write_json(
        tmp_path / "result_dry_run.json",
        {
            "schema_version": "official_result_ingest_report_v1",
            "status": "SUCCESS",
            "ingested_count": 1,
            "failed_count": 0,
            "skipped": [],
            "failed": [],
            "label_write_blockers": [],
        },
    )
    label_readiness = _write_json(
        tmp_path / "label_readiness.json",
        {
            "schema_version": "result_label_write_readiness_validation_v1",
            "status": "NOT_READY",
            "candidate_count_loaded_for_write_scope": 1,
            "skipped_before_write_scope_validation": [],
            "dry_run_report_gate": {"approved": True},
        },
    )
    challenger_inputs = _write_json(
        tmp_path / "data_inventory.json",
        {
            "schema_version": "history_feature_challenger_data_inventory_v1",
            "status": "SUCCESS",
            "clean_races": 132,
            "clean_runner_rows": 943,
            "clean_snapshot_instances": 134,
            LEGACY_MODEL_FIT_ROWS_KEY: 735,
            "eval_rows": 208,
            LEGACY_VALID_ODDS_FIT_RACES_KEY: 0,
            "complete_valid_odds_eval_races": 4,
            "clean_dataset": "artifacts/clean_official_dataset.jsonl",
            "packet_csv": "artifacts/pre_race_history_feature_packet.csv",
        },
    )

    packet = build_packet(
        snapshot_audit_path=snapshot_audit,
        result_dry_run_report_path=result_report,
        label_readiness_path=label_readiness,
        challenger_inputs_path=challenger_inputs,
    )

    assert packet["challenger_matrix_inputs"]["clean_official_races"] == 132
    assert packet["challenger_matrix_inputs"]["runner_rows"] == 943
    assert packet["challenger_matrix_inputs"]["clean_snapshot_instances"] == 134
    assert packet["challenger_matrix_inputs"]["model_fit_rows"] == 735
    assert packet["challenger_matrix_inputs"]["eval_rows"] == 208
    assert packet["challenger_matrix_inputs"]["complete_valid_odds_model_fit_races"] == 0
    assert packet["challenger_matrix_inputs"]["complete_valid_odds_eval_races"] == 4
    assert packet["challenger_matrix_inputs"]["output_paths"] == {
        "clean_dataset": "artifacts/clean_official_dataset.jsonl",
        "packet_csv": "artifacts/pre_race_history_feature_packet.csv",
    }


def test_accuracy_blocker_packet_includes_non_box_feature_audit_blockers(tmp_path):
    snapshot_audit = _write_json(
        tmp_path / "snapshot_audit.json",
        {
            "schema_version": "prediction_snapshot_readiness_box_bias_audit_v1",
            "counts": {"manifest_rows": 123, "latest_ready_races": 123},
            "gate": {"status": "FAIL", "box1_share": 112 / 123},
            "skip_reason_counts": {},
        },
    )
    result_report = _write_json(
        tmp_path / "result_dry_run.json",
        {
            "schema_version": "official_result_ingest_report_v1",
            "status": "SUCCESS",
            "ingested_count": 2,
            "failed_count": 0,
            "skipped": [],
            "failed": [],
            "label_write_blockers": [],
        },
    )
    label_readiness = _write_json(
        tmp_path / "label_readiness.json",
        {
            "schema_version": "result_label_write_readiness_validation_v1",
            "status": "NOT_READY",
            "candidate_count_loaded_for_write_scope": 2,
            "skipped_before_write_scope_validation": [],
            "dry_run_report_gate": {"approved": True},
        },
    )
    challenger_inputs = _write_json(
        tmp_path / "challenger_inputs.json",
        {
            "schema_version": "box_bias_study_data_inventory_v1",
            "status": "SUCCESS",
            "clean_official_races": 132,
            "runner_rows": 943,
        },
    )
    feature_audit = _write_json(
        tmp_path / "feature_audit.json",
        {
            "schema_version": "non_box_feature_quality_audit_v1",
            "races": 123,
            "runner_rows": 915,
            "top_pick_box_distribution": {"1": 112},
            "near_duplicate_rows_ge80pct_equal_peer": 763,
            "near_duplicate_rows_ge90pct_equal_peer": 399,
            "exact_non_box_duplicate_rows": 26,
            "mean_most_similar_non_box_equal_share": 0.8728142076502732,
            "mean_constant_non_box_feature_share": 0.7343579234972678,
            "distance_source_counts": {
                "canonical_pre_race_page": 596,
                "default_missing_target": 293,
                "DATA_MISSING": 26,
            },
            "grade_source_counts": {"canonical_pre_race_page": 596},
            "source_errors": [
                {
                    "race_id": "Race 1 - OLD - 2026-05-25",
                    "error": "source_file_missing",
                }
            ],
        },
    )

    packet = build_packet(
        snapshot_audit_path=snapshot_audit,
        result_dry_run_report_path=result_report,
        label_readiness_path=label_readiness,
        challenger_inputs_path=challenger_inputs,
        feature_audit_path=feature_audit,
    )

    assert packet["source_evidence"]["feature_audit"] == str(feature_audit)
    assert packet["feature_quality_blockers"] == {
        "schema_version": "non_box_feature_quality_audit_v1",
        "races": 123,
        "runner_rows": 915,
        "top_pick_box_distribution": {"1": 112},
        "near_duplicate_rows_ge80pct_equal_peer": 763,
        "near_duplicate_rows_ge90pct_equal_peer": 399,
        "exact_non_box_duplicate_rows": 26,
        "mean_most_similar_non_box_equal_share": 0.8728142076502732,
        "mean_constant_non_box_feature_share": 0.7343579234972678,
        "distance_source_counts": {
            "canonical_pre_race_page": 596,
            "default_missing_target": 293,
            "DATA_MISSING": 26,
        },
        "grade_source_counts": {"canonical_pre_race_page": 596},
        "source_error_count": 1,
        "source_error_examples": [
            {
                "race_id": "Race 1 - OLD - 2026-05-25",
                "error": "source_file_missing",
            }
        ],
    }
    assert "non_box_feature_quality_blocked" in packet["blocker_reasons"]


def test_accuracy_blocker_packet_includes_feature_missingness_tgr_blocker(tmp_path):
    snapshot_audit = _write_json(
        tmp_path / "snapshot_audit.json",
        {
            "schema_version": "prediction_snapshot_readiness_box_bias_audit_v1",
            "counts": {"manifest_rows": 123, "latest_ready_races": 123},
            "gate": {"status": "FAIL", "box1_share": 112 / 123},
            "skip_reason_counts": {},
        },
    )
    result_report = _write_json(
        tmp_path / "result_dry_run.json",
        {
            "schema_version": "official_result_ingest_report_v1",
            "status": "SUCCESS",
            "ingested_count": 2,
            "failed_count": 0,
            "skipped": [],
            "failed": [],
            "label_write_blockers": [],
        },
    )
    label_readiness = _write_json(
        tmp_path / "label_readiness.json",
        {
            "schema_version": "result_label_write_readiness_validation_v1",
            "status": "NOT_READY",
            "candidate_count_loaded_for_write_scope": 2,
            "skipped_before_write_scope_validation": [],
            "dry_run_report_gate": {"approved": True},
        },
    )
    challenger_inputs = _write_json(
        tmp_path / "challenger_inputs.json",
        {
            "schema_version": "box_bias_study_data_inventory_v1",
            "status": "SUCCESS",
            "clean_official_races": 132,
            "runner_rows": 943,
        },
    )
    feature_missingness = _write_json(
        tmp_path / "feature_missingness.json",
        {
            "schema_version": "feature_missingness_by_variant_v1",
            "all_clean_official": {
                "history_only_model": {
                    "field_stats": {
                        "historical_win_rate": {
                            "row_present_count": 0,
                            "row_present_pct": 0,
                        },
                        "tgr_total_races": {
                            "row_present_count": 0,
                            "row_present_pct": 0,
                        },
                        "tgr_win_rate": {
                            "row_present_count": 0,
                            "row_present_pct": 0,
                        },
                        "tgr_place_rate": {
                            "row_present_count": 0,
                            "row_present_pct": 0,
                        },
                        "recent_win_rate_5": {
                            "row_present_count": 656,
                            "row_present_pct": 0.695652,
                        },
                        "malformed_without_row_count": {
                            "row_present_pct": 0,
                        },
                    },
                    "required_features": [
                        "historical_win_rate",
                        "tgr_total_races",
                        "tgr_win_rate",
                        "tgr_place_rate",
                    ],
                    "scope": "clean official holdout",
                }
            },
        },
    )

    packet = build_packet(
        snapshot_audit_path=snapshot_audit,
        result_dry_run_report_path=result_report,
        label_readiness_path=label_readiness,
        challenger_inputs_path=challenger_inputs,
        feature_missingness_path=feature_missingness,
    )

    assert packet["source_evidence"]["feature_missingness"] == str(feature_missingness)
    assert packet["challenger_matrix_inputs"]["all_zero_feature_columns"] == [
        "historical_win_rate",
        "tgr_place_rate",
        "tgr_total_races",
        "tgr_win_rate",
    ]
    assert "malformed_without_row_count" not in packet["challenger_matrix_inputs"][
        "all_zero_feature_columns"
    ]
    assert packet["challenger_matrix_inputs"]["tgr_zero_coverage_columns"] == [
        "tgr_place_rate",
        "tgr_total_races",
        "tgr_win_rate",
    ]
    assert packet["challenger_matrix_inputs"]["feature_missingness_schema_version"] == (
        "feature_missingness_by_variant_v1"
    )
    assert "all_zero_tgr_feature_coverage" in packet["blocker_reasons"]


def test_accuracy_blocker_packet_reports_thedogs_incomplete_position_blockers(tmp_path):
    snapshot_audit = _write_json(
        tmp_path / "snapshot_audit.json",
        {
            "schema_version": "prediction_snapshot_readiness_box_bias_audit_v1",
            "counts": {"manifest_rows": 1, "latest_ready_races": 1},
            "gate": {"status": "FAIL", "box1_share": 1.0},
            "skip_reason_counts": {},
        },
    )
    result_report = _write_json(
        tmp_path / "result_dry_run.json",
        {
            "schema_version": "official_result_ingest_report_v1",
            "status": "SUCCESS",
            "ingested_count": 2,
            "failed_count": 0,
            "skipped": [],
            "failed": [],
            "label_write_blockers": [
                {
                    "race_id": "Race 7 - LADBROKES-Q1-LAKESIDE - 2026-06-03",
                    "source": "thedogs_official",
                    "status": "resulted",
                    "reason": "label_write_requires_complete_official_result_positions",
                    "expected_box_count": 8,
                    "result_box_count": 4,
                    "missing_result_boxes": [1, 2, 4, 6],
                    "unexpected_result_boxes": [],
                }
            ],
        },
    )
    label_readiness = _write_json(
        tmp_path / "label_readiness.json",
        {
            "schema_version": "result_label_write_readiness_validation_v1",
            "status": "NOT_READY",
            "candidate_count_loaded_for_write_scope": 2,
            "skipped_before_write_scope_validation": [],
            "dry_run_report_gate": {"approved": True},
        },
    )
    challenger_inputs = _write_json(
        tmp_path / "challenger_inputs.json",
        {
            "schema_version": "box_bias_study_data_inventory_v1",
            "status": "SUCCESS",
            "clean_official_races": 132,
            "runner_rows": 943,
        },
    )

    packet = build_packet(
        snapshot_audit_path=snapshot_audit,
        result_dry_run_report_path=result_report,
        label_readiness_path=label_readiness,
        challenger_inputs_path=challenger_inputs,
    )

    assert packet["official_parser_failure_examples"] == [
        {
            "race_id": "Race 7 - LADBROKES-Q1-LAKESIDE - 2026-06-03",
            "source": "thedogs_official",
            "source_url": None,
            "error": "label_write_requires_complete_official_result_positions",
            "status": "resulted",
            "expected_box_count": 8,
            "result_box_count": 4,
            "missing_result_boxes": [1, 2, 4, 6],
            "unexpected_result_boxes": [],
        }
    ]
    assert packet["readiness_by_stage"]["result_parse_ready"] == {
        "status": "NOT_READY",
        "count": 2,
        "skip_reason_counts": {
            "label_write_requires_complete_official_result_positions": 1
        },
    }
