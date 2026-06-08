import json
from datetime import datetime

from scripts import forward_shadow_status_report as status


def _db_report(state="PASS"):
    return {"status": state}


def _metrics(**overrides):
    payload = {
        "safe_joined_race_count": 25,
        "pending_race_count": 0,
        "unsafe_match_count": 0,
        "probability_sum_max_error_joined_races": 0.0,
    }
    payload.update(overrides)
    return payload


def _activation(**overrides):
    payload = {"kept_quarantined_features": []}
    payload.update(overrides)
    return payload


def test_status_collects_more_when_joined_sample_is_too_small():
    final_status, reasons = status.decide_status(
        db_report=_db_report(),
        metrics=_metrics(safe_joined_race_count=4, pending_race_count=12),
        activation=_activation(kept_quarantined_features=["same_distance_same_grade_best_time"]),
        min_joined_races=20,
    )

    assert final_status == "CONTINUE_FORWARD_SHADOW_COLLECTION"
    assert "safe_joined_race_count_below_review_min" in reasons
    assert "pending_official_results_remain" in reasons
    assert "features_remain_quarantined" in reasons


def test_status_blocks_on_db_failure():
    final_status, reasons = status.decide_status(
        db_report=_db_report("FAIL"),
        metrics=_metrics(),
        activation=_activation(),
        min_joined_races=20,
    )

    assert final_status == "BLOCKED_DB_STATE"
    assert reasons == ["db_state_not_pass"]


def test_status_ready_for_report_only_review_when_sample_and_gates_pass():
    final_status, reasons = status.decide_status(
        db_report=_db_report(),
        metrics=_metrics(),
        activation=_activation(),
        min_joined_races=20,
    )

    assert final_status == "READY_FOR_FORWARD_SHADOW_REVIEW_REPORT_ONLY"
    assert reasons == []


def test_metric_summary_handles_missing_metrics():
    summary = status.metric_summary(None)

    assert summary["safe_joined_race_count"] == 0
    assert summary["pending_race_count"] == 0
    assert summary["winner_ranks"] == []


def test_artifact_final_status_reads_status_file(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "final_status.txt").write_text("READY\n", encoding="utf-8")

    assert status.artifact_final_status(artifact) == "READY"


def test_status_prefers_aggregate_metrics_over_latest_single_join(tmp_path, monkeypatch):
    aggregate_dir = tmp_path / "forward_shadow_result_aggregate_20260608T123000+1000"
    aggregate_dir.mkdir()
    (aggregate_dir / "aggregate_forward_metrics.json").write_text(
        json.dumps(
            {
                "safe_joined_race_count": 6,
                "pending_race_count": 10,
                "unsafe_match_count": 0,
                "probability_sum_max_error_joined_races": 0.0,
                "winner_ranks": [7, 1, 5, 5, 2, 8],
            }
        ),
        encoding="utf-8",
    )
    single_dir = tmp_path / "forward_shadow_result_join_20260608T124000+1000"
    single_dir.mkdir()
    (single_dir / "shadow_forward_metrics.json").write_text(
        json.dumps(
            {
                "safe_joined_race_count": 1,
                "pending_race_count": 10,
                "unsafe_match_count": 0,
                "probability_sum_max_error_joined_races": 0.0,
                "winner_ranks": [5],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(status, "db_state", lambda _path: {"status": "PASS"})

    report = status.build_status_report(
        evidence_root=tmp_path,
        db_path=tmp_path / "db.sqlite",
        min_joined_races=20,
        generated_at=datetime.fromisoformat("2026-06-08T12:45:00+10:00"),
    )

    assert report["result_metric_source"] == "aggregate_forward_metrics"
    assert report["forward_metrics"]["safe_joined_race_count"] == 6
    assert report["coverage_gap"]["latest_forward_metrics_summary"][
        "safe_joined_race_count"
    ] == 6
    assert report["source_dirs"]["aggregate_result_dir"].endswith(
        "forward_shadow_result_aggregate_20260608T123000+1000"
    )


def test_coverage_summary_uses_selected_forward_metrics_over_stale_audit_metrics():
    summary = status.coverage_summary(
        {
            "latest_forward_metrics_summary": {
                "safe_joined_race_count": 4,
                "pending_race_count": 12,
            },
            "blocked_reasons": ["features_remain_quarantined"],
        },
        selected_metrics={
            "safe_joined_race_count": 7,
            "pending_race_count": 9,
        },
    )

    assert summary["latest_forward_metrics_summary"] == {
        "safe_joined_race_count": 7,
        "pending_race_count": 9,
    }
    assert summary["blocked_reasons"] == ["features_remain_quarantined"]
