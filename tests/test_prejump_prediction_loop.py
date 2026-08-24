import json
import os
import sys
import time
import types
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from scripts import prejump_prediction_loop as loop
from scripts.refresh_prejump_upcoming import (
    _metadata_record_for_csv,
    current_index_metadata_selection,
    expand_excluded_race_ids,
    refresh_prejump_upcoming,
    refresh_timing_summary,
    select_prejump_races,
    sidecar_metadata_coverage,
    stable_race_id,
)


READY_PREDICTIONS = [
    {"dog_name": "Alpha Runner", "box_number": 1, "win_prob_norm": 0.4},
]

JUNE2_SECOND_BATCH_RACE_IDS = [
    "Race 1 - GEE - 2026-06-02",
    "Race 10 - AP_K - 2026-06-02",
    "Race 11 - AP_K - 2026-06-02",
    "Race 3 - HOR - 2026-06-02",
    "Race 3 - LADBROKES-Q1-LAKESIDE - 2026-06-02",
]


def _june2_authoritative_persist_report(
    *,
    include_capture_ev_status: bool = True,
    odds_exclusion_counts: dict | None = None,
) -> dict:
    captures = []
    priced_counts = [7, 5, 5, 7, 8]
    for race_id, priced_count in zip(JUNE2_SECOND_BATCH_RACE_IDS, priced_counts):
        capture = {
            "race_id": race_id,
            "snapshot_readiness": {"status": "READY"},
            "priced_ev_runner_count": priced_count,
            "odds_exclusion_counts": {},
        }
        if include_capture_ev_status:
            capture["ev_readiness_status"] = "EV_READY"
        captures.append(capture)
    return {
        "status": "SUCCESS",
        "dry_run": False,
        "persist_requested": True,
        "persist_approved": True,
        "capture_count": 5,
        "ev_readiness_counts": {"EV_READY": 5},
        "priced_ev_runner_count": 32,
        "odds_exclusion_counts": odds_exclusion_counts or {},
        "captures": captures,
    }


def _june2_dry_run_not_ready_report() -> dict:
    return {
        "status": "SUCCESS",
        "dry_run": True,
        "persist_requested": False,
        "persist_approved": False,
        "candidate_files": 5,
        "capture_count": 5,
        "metadata_missing_count": 0,
        "metadata_unsafe_count": 0,
        "metadata_mismatch_count": 0,
        "ev_readiness_counts": {"EV_NOT_READY": 5},
        "captures": [
            {
                "race_id": race_id,
                "snapshot_readiness": {"status": "READY"},
                "ev_readiness": {"status": "EV_NOT_READY"},
            }
            for race_id in JUNE2_SECOND_BATCH_RACE_IDS
        ],
    }


def _clean_dry_run_capture_report() -> dict:
    return {
        "status": "SUCCESS",
        "dry_run": True,
        "persist_requested": False,
        "persist_approved": False,
        "candidate_files": 1,
        "capture_count": 1,
        "metadata_missing_count": 0,
        "metadata_unsafe_count": 0,
        "metadata_mismatch_count": 0,
        "captures": [
            {
                "race_id": "Race 1 - TEST - 2026-05-29",
                "runner_count": 1,
                "snapshot_readiness": {"status": "READY"},
                "ev_readiness": {"status": "EV_NOT_READY"},
                "probability_sum_check": {
                    "runner_count": 1,
                    "probability_sum": 1.0,
                    "abs_error": 0.0,
                },
                "prediction_preview": [
                    {
                        "predicted_rank": 1,
                        "box_number": 1,
                        "dog_name": "Alpha Runner",
                        "win_prob_norm": 1.0,
                        "odds_match_status": "no_odds_row",
                        "market_odds_win": None,
                        "ev_win": None,
                        "quality_flags": [],
                    }
                ],
            }
        ],
    }


def _clean_result_dry_run_report(scope: dict) -> dict:
    return {
        "schema_version": "official_result_ingest_report_v1",
        "status": "SUCCESS",
        "dry_run": True,
        "clean_for_label_write": True,
        "candidate_count": 1,
        "ingested_count": 1,
        "failed_count": 0,
        "scope": scope,
    }


def _clean_label_write_readiness_report(scope: dict) -> dict:
    return {
        "schema_version": "result_label_write_readiness_validation_v1",
        "status": "READY_FOR_EXPLICIT_APPROVAL",
        "scope": scope,
        "candidate_count_loaded_for_write_scope": 1,
        "candidate_race_ids_loaded_for_write_scope": scope.get("race_ids") or [],
        "skipped_before_write_scope_validation": [],
        "dry_run_report_gate": {"approved": True},
        "result_label_write_approval": {"approved": False},
        "approval_required": True,
        "required_cli_flag": "--write-labels-approved",
        "required_env_var": "APPROVE_RESULT_LABEL_WRITE",
        "planned_command_if_approved": ["write-labels-command"],
        "write_performed": False,
    }


def _clean_label_write_preflight_packet(
    scope: dict,
    *,
    label_readiness_path,
    result_report_path,
    db_path,
) -> dict:
    return {
        "schema_version": "label_write_preflight_packet_v1",
        "status": "READY_FOR_EXPLICIT_LABEL_WRITE_APPROVAL",
        "failures": [],
        "warnings": [],
        "race_scope": scope,
        "source_evidence": {
            "label_readiness": str(label_readiness_path),
            "result_dry_run_report": str(result_report_path),
            "db": str(db_path),
        },
        "approval_gate": {
            "approved": False,
            "required": True,
            "required_cli_flag": "--write-labels-approved",
            "required_env_var": "APPROVE_RESULT_LABEL_WRITE",
        },
        "writes_performed": {
            "result_label_write": False,
            "snapshot_persist": False,
            "live_odds_capture": False,
            "model_artifact_write": False,
            "registry_mutation": False,
            "production_config_write": False,
            "refresh_signal_write": False,
            "retrain": False,
            "betting": False,
        },
        "pre_write_db_state": {
            "quick_check": "ok",
            "result_free_before_write": True,
        },
        "no_write_preflight_only": True,
    }


def _clean_evaluation_report(dataset_path) -> dict:
    return {
        "status": "SUCCESS",
        "runner_rows_scored": 4,
        "evaluation_dataset_output": str(dataset_path),
        "evaluation_dataset_rows_written": 4,
        "snapshot_corpus_readiness": {"status": "READY"},
        "clean_official_evaluation": {
            "races_evaluated": 1,
            "runner_rows_evaluated": 4,
            "metrics_by_arm": {
                "model_only": {
                    "top1": 1.0,
                    "top3": 1.0,
                    "brier": 0.1,
                    "log_loss": 0.2,
                    "calibration": {"bins": []},
                }
            },
        },
        "model_quality_diagnosis": {
            "status": "SUCCESS",
            "retrain_gate": {"status": "NOT_READY", "action_taken": "none"},
            "promotion_gate": {"status": "REPORT_ONLY", "action_taken": "none"},
        },
    }


def _clean_model_review_packet(
    eval_report,
    dataset_path,
    challenger_review=None,
) -> dict:
    packet = {
        "schema_version": "model_review_packet_v1",
        "status": "READY_FOR_CHALLENGER_REVIEW",
        "failures": [],
        "warnings": [],
        "source_evidence": {
            "evaluation_report": str(eval_report.resolve()),
            "evaluation_dataset": str(dataset_path.resolve()),
            "evaluation_dataset_rows_written": 4,
            "evaluation_dataset_rows_observed": 4,
        },
        "promotion_control": {
            "action_taken": "none",
            "registry_mutation_allowed": False,
            "promotion_allowed": False,
        },
        "next_review_steps": [
            {
                "name": "promotion",
                "status": "BLOCKED",
                "required_gate": "APPROVE_MODEL_PROMOTION",
            }
        ],
    }
    if challenger_review is not None:
        packet["challenger_review_gate"] = {
            "provided": True,
            "path": str(challenger_review.resolve()),
            "status": "READY",
            "failures": [],
            "candidate_arm": "power_calibrated_baseline",
            "stability_status": "STABLE_REPORT_ONLY",
            "split_count": 2,
            "failed_split_count": 0,
            "all_log_loss_improved": True,
            "all_brier_improved": True,
            "all_ranking_preserved": True,
            "promotion_allowed": False,
            "registry_mutation_allowed": False,
            "model_artifact_written": False,
        }
    return packet


def _clean_calibration_design_report(review_packet) -> dict:
    return {
        "schema_version": "calibration_layer_design_v1",
        "status": "READY_FOR_OPERATOR_DESIGN_REVIEW",
        "failures": [],
        "warnings": [],
        "source_evidence": {
            "model_review_packet": str(review_packet.resolve()),
            "clean_official_races": 1,
            "clean_official_rows": 4,
        },
        "runtime_transform_spec": {
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
        },
        "comparison_to_baseline": {
            "log_loss_improved": True,
            "brier_improved": True,
            "top1_preserved": True,
            "top2_preserved": True,
            "top3_preserved": True,
            "mean_winner_rank_preserved": True,
        },
        "deployment_control": {
            "action_taken": "none",
            "model_artifact_written": False,
            "registry_mutation_allowed": False,
            "production_config_write_allowed": False,
            "promotion_allowed": False,
            "required_gate": "APPROVE_MODEL_PROMOTION",
            "betting_allowed": False,
        },
    }


def _clean_snapshot_challenger_review_report(dataset_path) -> dict:
    return {
        "schema_version": "snapshot_challenger_review_v1",
        "status": "SUCCESS",
        "failures": [],
        "warnings": [],
        "source_evidence": {
            "evaluation_dataset": str(dataset_path.resolve()),
            "rows_loaded": 4,
            "clean_official_rows": 4,
            "clean_official_races": 1,
        },
        "stability_review": {
            "status": "STABLE_REPORT_ONLY",
            "candidate_arm": "power_calibrated_baseline",
            "split_count": 2,
            "failed_split_count": 0,
            "all_log_loss_improved": True,
            "all_brier_improved": True,
            "all_ranking_preserved": True,
            "promotion_allowed": False,
        },
        "challenger_training": {
            "model_family": "LogisticRegression",
            "model_artifact_written": False,
            "registry_mutation_allowed": False,
            "power_calibration": {
                "selected_alpha": 0.5,
                "model_artifact_written": False,
                "registry_mutation_allowed": False,
            },
        },
        "promotion_control": {
            "action_taken": "none",
            "model_artifact_written": False,
            "registry_mutation_allowed": False,
            "promotion_allowed": False,
            "required_gate": "APPROVE_MODEL_PROMOTION",
        },
    }


def test_refresh_selection_filters_prejump_window():
    now = datetime(2026, 5, 29, 13, 0, tzinfo=ZoneInfo("Australia/Melbourne"))
    races = [
        {
            "url": "https://example.test/r1",
            "date": "2026-05-29",
            "race_time": "1:10 PM",
            "race_number": 1,
            "venue": "TEST",
        },
        {
            "url": "https://example.test/r2",
            "date": "2026-05-29",
            "race_time": "1:45 PM",
            "race_number": 2,
            "venue": "TEST",
        },
        {
            "url": "https://example.test/r3",
            "date": "2026-05-29",
            "race_time": "4:30 PM",
            "race_number": 3,
            "venue": "TEST",
        },
    ]

    selected, records = select_prejump_races(
        races,
        now=now,
        min_minutes=20,
        max_minutes=160,
        limit=0,
    )

    assert [race["race_number"] for race in selected] == [2]
    assert [record["bucket"] for record in records] == [
        "past_or_too_close",
        "preferred_window",
        "future_outside_preferred_window",
    ]

    timing = refresh_timing_summary(records, min_minutes=20, max_minutes=160)

    assert timing["status"] == "OPEN_NOW"
    assert timing["next_race"]["race_number"] == 2
    assert timing["minutes_until_window_closes"] == 25.0


def test_refresh_selection_prioritizes_nearest_due_races_before_limit():
    now = datetime(2026, 5, 29, 13, 0, tzinfo=ZoneInfo("Australia/Melbourne"))
    races = [
        {
            "url": "https://example.test/r1",
            "date": "2026-05-29",
            "race_time": "1:55 PM",
            "race_number": 1,
            "venue": "TEST",
        },
        {
            "url": "https://example.test/r2",
            "date": "2026-05-29",
            "race_time": "1:05 PM",
            "race_number": 2,
            "venue": "TEST",
        },
        {
            "url": "https://example.test/r3",
            "date": "2026-05-29",
            "race_time": "1:30 PM",
            "race_number": 3,
            "venue": "TEST",
        },
    ]

    selected, records = select_prejump_races(
        races,
        now=now,
        min_minutes=0,
        max_minutes=60,
        limit=2,
    )

    assert [race["race_number"] for race in selected] == [2, 3]
    selected_records = [record for record in records if record.get("selection_order")]
    assert [(record["race_number"], record["selection_order"]) for record in selected_records] == [
        (2, 1),
        (3, 2),
    ]


def test_refresh_selection_keeps_exact_manual_request_inside_limit():
    now = datetime(2026, 5, 29, 13, 0, tzinfo=ZoneInfo("Australia/Melbourne"))
    races = [
        {
            "url": f"https://example.test/r{race_number}",
            "date": "2026-05-29",
            "race_time": race_time,
            "race_number": race_number,
            "venue": "TEST",
        }
        for race_number, race_time in ((1, "1:05 PM"), (2, "1:30 PM"))
    ]

    selected, records = select_prejump_races(
        races,
        now=now,
        min_minutes=0,
        max_minutes=60,
        limit=1,
        priority_race_id="Race 2 - TEST - 2026-05-29",
    )

    assert [race["race_number"] for race in selected] == [2]
    assert next(
        record for record in records if record["race_number"] == 2
    )["selection_order"] == 1


def test_refresh_timing_summary_reports_next_future_window():
    now = datetime(2026, 5, 29, 13, 0, tzinfo=ZoneInfo("Australia/Melbourne"))
    races = [
        {
            "url": "https://example.test/r3",
            "date": "2026-05-29",
            "race_time": "4:30 PM",
            "race_number": 3,
            "venue": "TEST",
        },
        {
            "url": "https://example.test/r4",
            "date": "2026-05-29",
            "race_time": "5:00 PM",
            "race_number": 4,
            "venue": "TEST",
        },
    ]

    _, records = select_prejump_races(
        races,
        now=now,
        min_minutes=20,
        max_minutes=160,
        limit=0,
    )
    timing = refresh_timing_summary(records, min_minutes=20, max_minutes=160)

    assert timing["status"] == "WAITING_FOR_FUTURE_WINDOW"
    assert timing["next_race"]["race_number"] == 3
    assert timing["recommended_rerun_after_local"] == "2026-05-29T13:50:00+10:00"
    assert timing["minutes_until_window_opens"] == 50.0


def test_select_prejump_races_can_exclude_already_attempted_race_ids():
    now = datetime(2026, 5, 29, 13, 0, tzinfo=ZoneInfo("Australia/Melbourne"))
    races = [
        {
            "url": "https://example.test/r1",
            "date": "2026-05-29",
            "race_time": "1:45 PM",
            "race_number": 1,
            "venue": "TEST",
        },
        {
            "url": "https://example.test/r2",
            "date": "2026-05-29",
            "race_time": "2:00 PM",
            "race_number": 2,
            "venue": "TEST",
        },
    ]
    excluded = {stable_race_id(races[0])}

    selected, records = select_prejump_races(
        races,
        now=now,
        min_minutes=20,
        max_minutes=160,
        limit=0,
        exclude_race_ids=excluded,
    )

    assert [race["race_number"] for race in selected] == [2]
    assert records[0]["race_id"] == "Race 1 - TEST - 2026-05-29"
    assert records[0]["bucket"] == "excluded_race_id"
    assert records[0]["excluded_reason"] == "excluded_race_id"
    assert records[1]["bucket"] == "preferred_window"


def test_select_prejump_races_excludes_known_venue_alias_ids():
    now = datetime(2026, 6, 8, 17, 30, tzinfo=ZoneInfo("Australia/Melbourne"))
    races = [
        {
            "url": "https://example.test/qot-r2",
            "date": "2026-06-08",
            "race_time": "6:41 PM",
            "race_number": 2,
            "venue": "QOT",
        },
        {
            "url": "https://example.test/lctn-r1",
            "date": "2026-06-08",
            "race_time": "7:06 PM",
            "race_number": 1,
            "venue": "LCTN",
        },
        {
            "url": "https://example.test/new",
            "date": "2026-06-08",
            "race_time": "7:20 PM",
            "race_number": 9,
            "venue": "TEST",
        },
    ]
    excluded = {
        "Race 2 - LADBROKES-Q-STRAIGHT - 2026-06-08",
        "Race 1 - LAUNCESTON - 2026-06-08",
    }

    selected, records = select_prejump_races(
        races,
        now=now,
        min_minutes=20,
        max_minutes=160,
        limit=0,
        exclude_race_ids=excluded,
    )

    assert [race["race_number"] for race in selected] == [9]
    assert records[0]["race_id"] == "Race 2 - QOT - 2026-06-08"
    assert records[0]["bucket"] == "excluded_race_id"
    assert "Race 2 - LADBROKES-Q-STRAIGHT - 2026-06-08" in records[0]["race_id_aliases"]
    assert records[1]["race_id"] == "Race 1 - LCTN - 2026-06-08"
    assert records[1]["bucket"] == "excluded_race_id"
    assert "Race 1 - LAUNCESTON - 2026-06-08" in records[1]["race_id_aliases"]
    assert records[2]["bucket"] == "preferred_window"


def test_expand_excluded_race_ids_adds_known_venue_aliases():
    expanded = expand_excluded_race_ids(
        {
            "Race 7 - QOT - 2026-06-08",
            "Race 5 - LAUNCESTON - 2026-06-08",
        }
    )

    assert "Race 7 - LADBROKES-Q-STRAIGHT - 2026-06-08" in expanded
    assert "Race 5 - LCTN - 2026-06-08" in expanded


def test_select_prejump_races_excludes_url_slug_venue_alias_ids():
    now = datetime(2026, 6, 9, 14, 0, tzinfo=ZoneInfo("Australia/Melbourne"))
    races = [
        {
            "url": "https://www.thedogs.com.au/racing/ladbrokes-q1-lakeside/2026-06-09/3/garrard-s-horse-and-hound?trial=false",
            "date": "2026-06-09",
            "race_time": "3:18 PM",
            "race_number": 3,
            "venue": "QOT",
        },
        {
            "url": "https://www.thedogs.com.au/racing/horsham/2026-06-09/2/chs-group?trial=false",
            "date": "2026-06-09",
            "race_time": "3:20 PM",
            "race_number": 2,
            "venue": "HOR",
        },
    ]
    excluded = {"Race 3 - LADBROKES-Q1-LAKESIDE - 2026-06-09"}

    selected, records = select_prejump_races(
        races,
        now=now,
        min_minutes=20,
        max_minutes=160,
        limit=0,
        exclude_race_ids=excluded,
    )

    assert [race["venue"] for race in selected] == ["HOR"]
    assert records[0]["race_id"] == "Race 3 - QOT - 2026-06-09"
    assert records[0]["bucket"] == "excluded_race_id"
    assert "Race 3 - LADBROKES-Q1-LAKESIDE - 2026-06-09" in records[0]["race_id_aliases"]
    assert records[1]["bucket"] == "preferred_window"


def test_refresh_prejump_upcoming_reports_top_level_artifact_counts(tmp_path, monkeypatch):
    refresh_module = sys.modules[refresh_prejump_upcoming.__module__]
    now = datetime(2026, 5, 29, 13, 0, tzinfo=ZoneInfo("Australia/Melbourne"))
    monkeypatch.setattr(refresh_module, "melbourne_now", lambda: now)

    class FakeUpcomingRaceBrowser:
        def get_upcoming_races(self, days_ahead):
            assert days_ahead == 0
            return [
                {
                    "url": "https://example.test/racing/test/2026-05-29/1/example?trial=false",
                    "date": "2026-05-29",
                    "race_time": "1:45 PM",
                    "race_number": 1,
                    "venue": "TEST",
                    "distance": "400",
                    "grade": "Grade 5",
                    "target_grade_context_schema": (
                        "thedogs_meeting_card_exact_race_v1"
                    ),
                    "target_grade_equivalence_key": "GRADE:5",
                    "target_grade_exact_value": "Grade 5",
                    "target_grade_race_url": (
                        "https://example.test/racing/test/2026-05-29/1/example"
                    ),
                    "target_grade_source_url": (
                        "https://www.thedogs.com.au/racing/2026-05-29"
                    ),
                    "target_grade_source_sha256": "c" * 64,
                }
            ]

        def download_race_csv(self, url, *, race_info_hint=None):
            assert url.endswith("trial=false")
            assert race_info_hint == {
                "url": url,
                "date": "2026-05-29",
                "race_time": "1:45 PM",
                "race_number": 1,
                "venue": "TEST",
                "distance": "400",
                "grade": "Grade 5",
                "target_grade_context_schema": (
                    "thedogs_meeting_card_exact_race_v1"
                ),
                "target_grade_equivalence_key": "GRADE:5",
                "target_grade_exact_value": "Grade 5",
                "target_grade_race_url": (
                    "https://example.test/racing/test/2026-05-29/1/example"
                ),
                "target_grade_source_url": (
                    "https://www.thedogs.com.au/racing/2026-05-29"
                ),
                "target_grade_source_sha256": "c" * 64,
                "jump_datetime": "2026-05-29T13:45:00+10:00",
            }
            upcoming_dir = tmp_path / "upcoming"
            csv_path = upcoming_dir / "Race 1 - TEST - 2026-05-29.csv"
            csv_path.write_text("box|dog_name\n1|Alpha Runner\n", encoding="utf-8")
            csv_path.with_suffix(".csv.metadata.json").write_text(
                json.dumps({"race_url": url}),
                encoding="utf-8",
            )
            raw_dir = upcoming_dir / "raw_exports"
            raw_dir.mkdir()
            (raw_dir / csv_path.name).write_text("box,dog_name\n1,Alpha Runner\n", encoding="utf-8")
            quarantine_dir = upcoming_dir / "quarantine"
            quarantine_dir.mkdir()
            (quarantine_dir / "rejected.csv").write_text("bad\n", encoding="utf-8")
            return {"success": True, "path": str(csv_path)}

    fake_module = types.SimpleNamespace(UpcomingRaceBrowser=FakeUpcomingRaceBrowser)
    monkeypatch.setitem(sys.modules, "upcoming_race_browser", fake_module)

    report = refresh_prejump_upcoming(
        Namespace(
            upcoming_dir=str(tmp_path / "upcoming"),
            days_ahead=0,
            min_minutes=20,
            max_minutes=160,
            limit=16,
            exclude_race_id=[],
            exclude_race_ids_file=None,
            dry_run=False,
        )
    )

    assert report["accepted_csv_count"] == 1
    assert report["sidecar_count"] == 1
    assert report["raw_export_count"] == 1
    assert report["quarantine_count"] == 1
    assert report["artifact_counts"] == {
        "accepted_csv_count": 1,
        "sidecar_count": 1,
        "raw_export_count": 1,
        "quarantine_count": 1,
    }
    assert report["metadata_collection_status"] == "PARTIAL"
    assert report["sidecar_metadata_coverage"]["safe_weather_race_count"] == 0


def test_refresh_prejump_upcoming_honors_supplied_current_time(tmp_path, monkeypatch):
    refresh_module = sys.modules[refresh_prejump_upcoming.__module__]
    wall_clock = datetime(2026, 6, 24, 0, 47, tzinfo=ZoneInfo("Australia/Melbourne"))
    monkeypatch.setattr(refresh_module, "melbourne_now", lambda: wall_clock)

    class FakeUpcomingRaceBrowser:
        def get_upcoming_races(self, days_ahead):
            assert days_ahead == 1
            return [
                {
                    "url": "https://www.thedogs.com.au/racing/murray-bridge-straight/2026-06-24/1/test?trial=false",
                    "date": "2026-06-24",
                    "race_time": "11:29 AM",
                    "race_number": 1,
                    "venue": "MURR",
                }
            ]

        def download_race_csv(self, url, *, race_info_hint=None):
            csv_path = tmp_path / "upcoming" / "Race 1 - MURR - 2026-06-24.csv"
            csv_path.write_text("box|dog_name\n1|Alpha Runner\n", encoding="utf-8")
            csv_path.with_name(csv_path.name + ".metadata.json").write_text(
                json.dumps({"race_url": url}),
                encoding="utf-8",
            )
            return {"success": True, "path": str(csv_path)}

    monkeypatch.setitem(
        sys.modules,
        "upcoming_race_browser",
        types.SimpleNamespace(UpcomingRaceBrowser=FakeUpcomingRaceBrowser),
    )

    report = refresh_prejump_upcoming(
        Namespace(
            upcoming_dir=str(tmp_path / "upcoming"),
            days_ahead=1,
            min_minutes=20,
            max_minutes=160,
            limit=16,
            exclude_race_id=[],
            exclude_race_ids_file=None,
            dry_run=False,
            current_time="2026-06-24T09:30:00+10:00",
        )
    )

    assert report["generated_at"] == "2026-06-24T09:30:00+10:00"
    assert report["selected_count"] == 1
    assert report["bucket_counts"] == {"preferred_window": 1}


def _write_safe_collection_sidecar(csv_path: Path, *, expert_form: bool = True):
    csv_path.write_text(
        "box|dog_name\n1|Alpha Runner\n2|Bravo Runner\n", encoding="utf-8"
    )
    expert = {
        "schema_version": "thedogs_expert_form_metadata_v1",
        "source": "thedogs_expert_form_page",
        "source_url": "https://www.thedogs.com.au/racing/sale/2026-06-17/9/test/expert-form",
        "captured_at": "2026-06-17T03:00:00Z",
        "metadata_is_leakage_safe": True,
        "runner_count": 1,
        "runners": [{"dog_name": "Alpha Runner"}],
        "rejected_reasons": [],
    }
    if not expert_form:
        expert["metadata_is_leakage_safe"] = False
        expert["runner_count"] = 0
        expert["runners"] = []
        expert["rejected_reasons"] = ["expert_form_runner_metadata_missing"]
    payload = {
        "schema_version": "form_guide_download_provenance_v1",
        "metadata_is_leakage_safe": True,
        "metadata_captured_at": "2026-06-17T02:45:00Z",
        "prejump_shadow_metadata": {
            "metadata_captured_at": "2026-06-17T02:45:00Z",
            "source_native_race_id": "15900",
            "runner_box_name_list": [
                {
                    "box_number": 1,
                    "dog_name": "Alpha Runner",
                    "source_native_runner_id": "159001",
                },
                {
                    "box_number": 2,
                    "dog_name": "Bravo Runner",
                    "source_native_runner_id": "159002",
                },
            ],
        },
        "race_url": "https://www.thedogs.com.au/racing/sale/2026-06-17/9/test?trial=false",
        "race_info": {
            "date": "2026-06-17",
            "race_time": "1:57 PM",
            "venue": "SAL",
            "race_number": "9",
            "url": "https://www.thedogs.com.au/racing/sale/2026-06-17/9/test?trial=false",
        },
        "weather": "Clear",
        "track_condition": "Good",
        "weather_track_metadata_source": "open_meteo_forecast_api+sportsbet_pre_race_page",
        "weather_track_metadata_source_url": {
            "open_meteo_forecast_api": "https://api.open-meteo.com/v1/forecast?latitude=-38.1",
            "sportsbet_pre_race_page": "https://www.sportsbet.com.au/apigw/sportsbook-racing/Sportsbook/Racing/NextEvents",
        },
        "weather_track_metadata_is_leakage_safe": True,
        "expert_form_metadata": expert,
    }
    csv_path.with_name(csv_path.name + ".metadata.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_sidecar_metadata_coverage_accepts_safe_weather_track_and_expert_form(tmp_path):
    upcoming_dir = tmp_path / "upcoming"
    upcoming_dir.mkdir()
    csv_path = upcoming_dir / "Race 9 - SAL - 2026-06-17.csv"
    _write_safe_collection_sidecar(csv_path)

    coverage = sidecar_metadata_coverage(
        upcoming_dir,
        [
            {
                "race_id": "Race 9 - SAL - 2026-06-17",
                "race_url": "https://www.thedogs.com.au/racing/sale/2026-06-17/9/test?trial=false",
            }
        ],
    )

    assert coverage["status"] == "READY"
    assert coverage["safe_weather_race_count"] == 1
    assert coverage["safe_track_condition_race_count"] == 1
    assert coverage["safe_expert_form_race_count"] == 1
    assert coverage["safe_all_weather_track_expert_form_race_count"] == 1


def test_current_index_metadata_selection_rejects_conflicting_sidecar_url():
    selected = [{
        "race_id": "Race 9 - SAL - 2026-06-17",
        "race_id_aliases": ["Race 9 - SAL - 2026-06-17"],
        "race_url": (
            "https://www.thedogs.com.au/racing/sale/2026-06-17/9/test?trial=false"
        ),
        "jump_datetime": "2026-06-17T13:57:00+10:00",
    }]
    coverage = {
        "races": [{
            "race_id": "Race 9 - SAL - 2026-06-17",
            "race_url": (
                "https://www.thedogs.com.au/racing/gunnedah/2026-06-17/9/test"
            ),
            "safe_all_weather_track_expert_form_present": True,
        }]
    }

    eligible, selection = current_index_metadata_selection(selected, coverage)

    assert eligible == []
    assert selection["status"] == "INCOMPLETE"
    assert selection["exclusions"][0]["missing_safe_metadata"] == [
        "metadata_alignment"
    ]


def test_current_index_metadata_selection_requires_complete_consistent_identity():
    race_url = "https://www.thedogs.com.au/racing/sale/2026-06-17/9/test"
    selected = [{
        "race_id": "Race 9 - SAL - 2026-06-17",
        "race_id_aliases": ["Race 9 - SAL - 2026-06-17"],
        "race_url": race_url,
        "jump_datetime": "2026-06-17T13:57:00+10:00",
    }]
    row = {
        "race_id": "",
        "race_url": race_url,
        "csv_path": "/evidence/Race 9 - SAL - 2026-06-17.csv",
        "safe_weather_present": True,
        "safe_track_condition_present": True,
        "safe_expert_form_present": True,
        "safe_all_weather_track_expert_form_present": True,
        "runner_source_observed_at": "2026-06-17T12:45:00+10:00",
        "source_native_race_id": "15900",
        "source_native_runner_ids": ["159001", "159002"],
    }

    eligible, selection = current_index_metadata_selection(
        selected,
        {"races": [row]},
        source_generated_at="2026-06-17T12:30:00+10:00",
    )
    assert eligible == []
    assert selection["exclusions"][0]["missing_safe_metadata"] == [
        "metadata_alignment"
    ]

    row["race_id"] = selected[0]["race_id"]
    row["safe_weather_present"] = False
    eligible, selection = current_index_metadata_selection(
        selected,
        {"races": [row]},
        source_generated_at="2026-06-17T12:30:00+10:00",
    )
    assert eligible == []
    assert selection["exclusions"][0]["missing_safe_metadata"] == ["weather"]


def test_current_index_metadata_selection_rejects_unsafe_native_identity():
    race_url = "https://www.thedogs.com.au/racing/sale/2026-06-17/9/test"
    race = {
        "race_id": "Race 9 - SAL - 2026-06-17",
        "race_id_aliases": ["Race 9 - SAL - 2026-06-17"],
        "race_url": race_url,
        "jump_datetime": "2026-06-17T13:57:00+10:00",
    }
    safe_row = {
        "race_id": race["race_id"],
        "race_url": race_url,
        "csv_path": "/evidence/Race 9 - SAL - 2026-06-17.csv",
        "safe_weather_present": True,
        "safe_track_condition_present": True,
        "safe_expert_form_present": True,
        "safe_all_weather_track_expert_form_present": True,
        "runner_source_observed_at": "2026-06-17T12:45:00+10:00",
    }
    for race_id, runner_ids in (
        ("not-numeric", ["159001", "159002"]),
        ("15900", ["159001", "not-numeric"]),
        ("15900", ["159001", "159001"]),
        ("15900", ["159001", []]),
    ):
        row = {
            **safe_row,
            "source_native_race_id": race_id,
            "source_native_runner_ids": runner_ids,
        }
        eligible, selection = current_index_metadata_selection(
            [race],
            {"races": [row]},
            source_generated_at="2026-06-17T12:30:00+10:00",
        )

        assert eligible == []
        assert selection["exclusions"][0]["missing_safe_metadata"] == [
            "native_source_identity"
        ]


def test_current_index_metadata_selection_excludes_postjump_runner_observation():
    race_url = "https://www.thedogs.com.au/racing/sale/2026-06-17/9/test"
    race = {
        "race_id": "Race 9 - SAL - 2026-06-17",
        "race_id_aliases": ["Race 9 - SAL - 2026-06-17"],
        "race_url": race_url,
        "jump_datetime": "2026-06-17T13:00:00+10:00",
    }
    row = {
        "race_id": race["race_id"],
        "race_url": race_url,
        "csv_path": "/evidence/Race 9 - SAL - 2026-06-17.csv",
        "safe_weather_present": True,
        "safe_track_condition_present": True,
        "safe_expert_form_present": True,
        "safe_all_weather_track_expert_form_present": True,
        "runner_source_observed_at": "2026-06-17T13:00:06+10:00",
        "source_native_race_id": "15900",
        "source_native_runner_ids": ["159001", "159002"],
    }

    eligible, selection = current_index_metadata_selection(
        [race],
        {"races": [row]},
        source_generated_at="2026-06-17T12:59:00+10:00",
    )

    assert eligible == []
    assert selection["status"] == "INCOMPLETE"
    assert selection["exclusions"][0]["missing_safe_metadata"] == [
        "runner_source_timing"
    ]


def test_metadata_record_does_not_substitute_top_level_runner_timestamp(tmp_path):
    csv_path = tmp_path / "Race 9 - SAL - 2026-06-17.csv"
    csv_path.write_text("box|dog_name\n1|Alpha Runner\n", encoding="utf-8")
    csv_path.with_name(csv_path.name + ".metadata.json").write_text(
        json.dumps({"metadata_captured_at": "2026-06-17T12:45:00+10:00"}),
        encoding="utf-8",
    )

    record = _metadata_record_for_csv(csv_path)

    assert record["runner_source_observed_at"] is None


def test_current_index_metadata_selection_enforces_runner_source_age_boundary():
    race_url = "https://www.thedogs.com.au/racing/sale/2026-06-17/9/test"
    race = {
        "race_id": "Race 9 - SAL - 2026-06-17",
        "race_id_aliases": ["Race 9 - SAL - 2026-06-17"],
        "race_url": race_url,
        "jump_datetime": "2026-06-17T13:30:00+10:00",
    }
    row = {
        "race_id": race["race_id"],
        "race_url": race_url,
        "csv_path": "/evidence/Race 9 - SAL - 2026-06-17.csv",
        "safe_weather_present": True,
        "safe_track_condition_present": True,
        "safe_expert_form_present": True,
        "safe_all_weather_track_expert_form_present": True,
        "source_native_race_id": "15900",
        "source_native_runner_ids": ["159001", "159002"],
    }
    cases = [
        ("2026-06-17T12:50:00+10:00", True),
        ("2026-06-17T12:50:01+10:00", False),
        ("2026-06-17T12:45:00", False),
    ]
    for observed_at, expected_eligible in cases:
        row["runner_source_observed_at"] = observed_at
        eligible, selection = current_index_metadata_selection(
            [race],
            {"races": [row]},
            source_generated_at="2026-06-17T12:30:00+10:00",
        )
        assert bool(eligible) is expected_eligible
        assert selection["status"] == ("READY" if expected_eligible else "INCOMPLETE")


def test_refresh_prejump_upcoming_can_fail_closed_on_incomplete_safe_metadata(
    tmp_path,
    monkeypatch,
):
    refresh_module = sys.modules[refresh_prejump_upcoming.__module__]
    now = datetime(2026, 6, 17, 12, 30, tzinfo=ZoneInfo("Australia/Melbourne"))
    monkeypatch.setattr(refresh_module, "melbourne_now", lambda: now)

    class FakeUpcomingRaceBrowser:
        def get_upcoming_races(self, days_ahead):
            return [
                {
                    "url": "https://www.thedogs.com.au/racing/sale/2026-06-17/9/test?trial=false",
                    "date": "2026-06-17",
                    "race_time": "1:57 PM",
                    "race_number": 9,
                    "venue": "SAL",
                }
            ]

        def download_race_csv(self, url, *, race_info_hint=None):
            csv_path = tmp_path / "upcoming" / "Race 9 - SAL - 2026-06-17.csv"
            _write_safe_collection_sidecar(csv_path, expert_form=False)
            return {"success": True, "filepath": str(csv_path)}

    monkeypatch.setitem(
        sys.modules,
        "upcoming_race_browser",
        types.SimpleNamespace(UpcomingRaceBrowser=FakeUpcomingRaceBrowser),
    )

    report = refresh_prejump_upcoming(
        Namespace(
            upcoming_dir=str(tmp_path / "upcoming"),
            days_ahead=0,
            min_minutes=20,
            max_minutes=160,
            limit=16,
            exclude_race_id=[],
            exclude_race_ids_file=None,
            dry_run=False,
            require_safe_metadata=True,
        )
    )

    assert report["status"] == "METADATA_COVERAGE_INCOMPLETE"
    assert report["metadata_collection_status"] == "PARTIAL"
    assert report["current_index_race_count"] == 0
    assert report["current_index_races"] == []
    assert report["current_index_metadata_selection"]["status"] == "INCOMPLETE"
    assert report["sidecar_metadata_coverage"]["safe_both_weather_track_race_count"] == 1
    assert report["sidecar_metadata_coverage"]["safe_expert_form_race_count"] == 0
    assert report["reason"] == "missing_safe_expert_form"


def test_refresh_prejump_upcoming_exposes_only_safe_current_index_subset(
    tmp_path,
    monkeypatch,
):
    refresh_module = sys.modules[refresh_prejump_upcoming.__module__]
    now = datetime(2026, 6, 17, 12, 30, tzinfo=ZoneInfo("Australia/Melbourne"))
    monkeypatch.setattr(refresh_module, "melbourne_now", lambda: now)
    safe_url = (
        "https://www.thedogs.com.au/racing/sale/2026-06-17/9/test?trial=false"
    )
    incomplete_url = (
        "https://www.thedogs.com.au/racing/wagga/2026-06-17/4/test?trial=false"
    )

    class FakeUpcomingRaceBrowser:
        def get_upcoming_races(self, days_ahead):
            return [
                {
                    "url": safe_url,
                    "date": "2026-06-17",
                    "race_time": "1:57 PM",
                    "race_number": 9,
                    "venue": "SAL",
                },
                {
                    "url": incomplete_url,
                    "date": "2026-06-17",
                    "race_time": "2:00 PM",
                    "race_number": 4,
                    "venue": "WAG",
                },
            ]

        def download_race_csv(self, url, *, race_info_hint=None):
            if url == safe_url:
                csv_path = tmp_path / "upcoming" / "Race 9 - SAL - 2026-06-17.csv"
                _write_safe_collection_sidecar(csv_path)
                return {"success": True, "filepath": str(csv_path)}
            return {"success": False, "reason": "source_unavailable"}

    monkeypatch.setitem(
        sys.modules,
        "upcoming_race_browser",
        types.SimpleNamespace(UpcomingRaceBrowser=FakeUpcomingRaceBrowser),
    )

    report = refresh_prejump_upcoming(
        Namespace(
            upcoming_dir=str(tmp_path / "upcoming"),
            days_ahead=0,
            min_minutes=20,
            max_minutes=160,
            limit=16,
            exclude_race_id=[],
            exclude_race_ids_file=None,
            dry_run=False,
            require_safe_metadata=True,
        )
    )

    assert report["status"] == "SUCCESS"
    assert report["selected_count"] == 2
    assert len(report["selected_races"]) == 2
    assert report["current_index_race_count"] == 1
    assert [row["race_url"] for row in report["current_index_races"]] == [safe_url]
    selection = report["current_index_metadata_selection"]
    assert selection["status"] == "READY_WITH_EXCLUSIONS"
    assert selection["candidate_race_count"] == 2
    assert selection["eligible_race_count"] == 1
    assert selection["excluded_race_count"] == 1
    assert selection["exclusions"][0]["race_url"] == incomplete_url
    assert selection["exclusions"][0]["reason"] == "safe_metadata_incomplete"


def test_prejump_loop_operator_action_surfaces_refresh_rerun_window(
    tmp_path, monkeypatch
):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    refresh_report = {
        "status": "SUCCESS",
        "generated_at": "2026-06-02T00:25:54+10:00",
        "total_races_found": 129,
        "selected_count": 0,
        "bucket_counts": {"future_outside_preferred_window": 129},
        "window": {"min_minutes": 20.0, "max_minutes": 160.0},
        "next_preferred_window": {
            "status": "WAITING_FOR_FUTURE_WINDOW",
            "reason": "next_race_not_yet_inside_preferred_window",
            "recommended_rerun_after_local": "2026-06-02T09:22:00+10:00",
            "next_race": {
                "race_number": "1",
                "venue": "AP_K",
                "jump_datetime": "2026-06-02T12:02:00+10:00",
                "minutes_to_jump": 696.0,
                "bucket": "future_outside_preferred_window",
            },
        },
    }
    (run_dir / "refresh_report.json").write_text(
        json.dumps(refresh_report),
        encoding="utf-8",
    )
    args = loop.build_parser().parse_args(
        [
            "--db",
            str(tmp_path / "test.db"),
            "--snapshot-dir",
            str(tmp_path / "snapshots"),
            "--run-dir",
            str(run_dir),
            "--date",
            "2026-06-02",
        ]
    )

    plan = loop.build_loop_plan(args)
    action = plan["operator_next_action"]

    assert plan["refresh_report_gate"]["status"] == "WAITING_FOR_FUTURE_WINDOW"
    assert plan["refresh_report_gate"]["selected_count"] == 0
    assert action["next_step_status"] == "WAIT_FOR_NEXT_PREFERRED_PREJUMP_WINDOW"
    assert action["next_step_reason"] == "2026-06-02T09:22:00+10:00"
    assert action["required_gate"] is None
    assert action["approval_required"] is False
    assert action["persist_approval_window_status"] == "WAITING_FOR_FUTURE_WINDOW"
    assert action["persist_approval_window_urgency"] == "WAITING_FOR_WINDOW"
    assert action["refresh_report_gate_status"] == "WAITING_FOR_FUTURE_WINDOW"
    assert action["refresh_recommended_rerun_after_local"] == (
        "2026-06-02T09:22:00+10:00"
    )
    assert action["refresh_next_preferred_window"]["next_race"]["venue"] == "AP_K"


def test_persist_approval_window_urgency():
    assert (
        loop._persist_approval_window_urgency("OPEN_AWAITING_APPROVAL", 121)
        == "OPEN"
    )
    assert (
        loop._persist_approval_window_urgency("OPEN_AWAITING_APPROVAL", 120)
        == "CLOSING_SOON"
    )
    assert (
        loop._persist_approval_window_urgency("OPEN_AWAITING_APPROVAL", 0)
        == "REFRESH_REQUIRED"
    )
    assert (
        loop._persist_approval_window_urgency("REFRESH_REQUIRED", None)
        == "REFRESH_REQUIRED"
    )


def test_prejump_loop_plan_blocks_gated_writes_without_approval(monkeypatch):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    args = loop.build_parser().parse_args(
        [
            "--db",
            "test.db",
            "--snapshot-dir",
            "snapshots",
            "--run-dir",
            "artifacts/test-loop",
            "--date",
            "2026-05-29",
        ]
    )

    plan = loop.build_loop_plan(args)
    by_name = {step["name"]: step for step in plan["steps"]}
    audit_by_milestone = {
        item["milestone"]: item
        for item in plan["milestone_completion_audit"]["items"]
    }

    assert plan["steps"][4]["command"][3] == "artifacts/test-loop/upcoming_races"
    assert by_name["approved_persist_ready_subset"]["status"] == (
        "WAITING_FOR_READY_PERSIST_PACKET"
    )
    assert "persist_readiness_gate_not_clean" in by_name[
        "approved_persist_ready_subset"
    ]["reason"]
    assert "dry_run_capture_report_not_fresh" in by_name[
        "approved_persist_ready_subset"
    ]["reason"]
    assert by_name["opt_in_live_odds_capture"]["status"] == (
        "WAITING_FOR_READY_ODDS_PACKET"
    )
    assert "dry_run_capture_report_not_fresh" in by_name[
        "opt_in_live_odds_capture"
    ]["reason"]
    assert by_name["official_result_ingest_dry_run"]["status"] == (
        "WAITING_FOR_PERSISTED_PREJUMP_SNAPSHOTS"
    )
    assert by_name["approved_official_label_write"]["status"] == (
        "WAITING_FOR_PERSISTED_PREJUMP_SNAPSHOTS"
    )
    assert "--approve-live-persist" not in by_name["approved_persist_ready_subset"]["command"]
    assert "--approve-live-odds-capture" not in by_name["opt_in_live_odds_capture"]["command"]
    assert "--write-labels-approved" not in by_name["approved_official_label_write"]["command"]
    assert by_name["rolling_evaluation_dataset"]["status"] == "READY_TO_RUN"
    assert by_name["runner_set_mismatch_reduction"]["status"] == (
        "IMPLEMENTED_IN_REFRESH_AND_CAPTURE_GATES"
    )
    assert by_name["venue_filename_contract"]["status"] == (
        "IMPLEMENTED_IN_VALIDATORS_AND_PARSERS"
    )
    assert "expert-form duplicate tracking" in by_name[
        "venue_filename_contract"
    ]["reason"]
    assert "quarantines source CSVs missing active canonical runners" in by_name[
        "runner_set_mismatch_reduction"
    ]["reason"]
    assert "Non Graded" in by_name["target_metadata_coverage"]["reason"]
    assert "Special Event" in by_name["target_metadata_coverage"]["reason"]
    assert plan["current_corpus"]["status"] == "NO_READY_PERSISTED_PREJUMP_SNAPSHOTS_FOR_DATE"
    assert plan["current_corpus"]["ready_persisted_prediction_snapshot_count_for_date"] == 0
    assert plan["persist_readiness_gate"]["status"] == "DATA_MISSING"
    assert plan["persist_approval_packet"]["status"] == "NOT_READY"
    assert plan["persist_approval_packet"]["can_execute_persist_now"] is False
    assert "persist_readiness_gate_not_clean" in plan["persist_approval_packet"][
        "hard_stops"
    ]
    assert plan["persist_approval_packet"]["approved_persist_command_template"] is None
    assert plan["persist_approval_packet"]["approval_command_template_status"] == (
        "BLOCKED_BY_HARD_STOPS"
    )
    assert plan["live_odds_approval_packet"]["status"] == "NOT_READY"
    assert plan["live_odds_approval_packet"]["can_capture_live_odds_now"] is False
    assert "persist_readiness_gate_not_clean" in plan[
        "live_odds_approval_packet"
    ]["hard_stops"]
    assert plan["live_odds_approval_packet"]["approved_odds_command_template"] is None
    assert plan["result_label_approval_packet"]["status"] == "NOT_READY"
    assert plan["result_label_approval_packet"]["can_write_labels_now"] is False
    assert plan["result_label_approval_packet"]["hard_stops"] == [
        "persisted_prejump_corpus_missing",
        "result_dry_run_report_missing",
    ]
    assert plan["result_label_approval_packet"]["approval_required"] is True
    assert plan["result_label_approval_packet"][
        "approved_label_write_command_template"
    ] is None
    assert plan["result_label_approval_packet"]["official_first_policy"][
        "participant_alignment_required"
    ] is True
    assert plan["prediction_preview_report"]["status"] == "DATA_MISSING"
    assert plan["prediction_preview_report"]["reason"] == (
        "dry_run_capture_report_missing"
    )
    assert plan["latest_prediction_preview_report_phase"] == "initial_plan"
    assert plan["latest_prediction_preview_report"] == plan[
        "prediction_preview_report"
    ]
    assert plan["operator_next_action"]["next_step_status"] == (
        "REFRESH_DRY_RUN_REQUIRED_FOR_PERSIST_PACKET"
    )
    assert plan["operator_next_action"]["full_objective_complete"] is False
    assert plan["operator_next_action"]["completed_milestone_count"] == 5
    assert plan["operator_next_action"]["incomplete_milestone_count"] == 5
    assert [
        item["milestone"]
        for item in plan["operator_next_action"]["incomplete_milestones"]
    ] == [5, 7, 8, 9, 10]
    assert plan["operator_next_action"]["required_gate"] == "APPROVE_LIVE_PERSIST"
    assert plan["operator_next_action"]["approval_required"] is True
    assert plan["operator_next_action"]["command_template"] is None
    assert plan["operator_next_action"][
        "safe_no_approval_persist_packet_refresh_sequence_status"
    ] == "AVAILABLE"
    refresh_sequence = plan["operator_next_action"][
        "safe_no_approval_persist_packet_refresh_sequence"
    ]
    assert [item["name"] for item in refresh_sequence] == [
        "fresh_refresh_current_window",
        "validate_current_upcoming_contract",
        "dry_run_prejump_capture",
    ]
    assert "--approve-live-persist" not in json.dumps(refresh_sequence)
    assert "--approve-live-odds-capture" not in json.dumps(refresh_sequence)
    assert "--write-labels-approved" not in json.dumps(refresh_sequence)
    assert plan["operator_next_action"]["persist_dry_run_fresh_for_plan"] is False
    assert plan["operator_next_action"]["persist_approval_window_status"] == (
        "REFRESH_REQUIRED"
    )
    assert plan["operator_next_action"]["persist_approval_window_urgency"] == (
        "REFRESH_REQUIRED"
    )
    assert plan["operator_next_action"][
        "persist_approval_command_template_status"
    ] == "BLOCKED_BY_HARD_STOPS"
    assert "snapshot_persist" in plan["operator_next_action"][
        "forbidden_without_explicit_approval"
    ]
    assert plan["approval_provenance"]["live_persist"] == {
        "approved": False,
        "sources": [],
        "cli_flag": "--approve-live-persist",
        "cli_approved": False,
        "env_var": "APPROVE_LIVE_PERSIST",
        "env_approved": False,
    }
    assert plan["milestone_completion_audit"]["overall_status"] == "INCOMPLETE"
    assert audit_by_milestone[1]["complete"] is True
    assert audit_by_milestone[5]["complete"] is False
    assert audit_by_milestone[5]["status"] == "APPROVAL_REQUIRED"
    assert audit_by_milestone[6]["complete"] is True
    assert audit_by_milestone[6]["status"] == (
        "IMPLEMENTED_APPROVAL_REQUIRED_FOR_LIVE_CAPTURE"
    )
    assert audit_by_milestone[7]["status"] == (
        "WAITING_FOR_PERSISTED_PREJUMP_SNAPSHOTS"
    )
    assert audit_by_milestone[8]["status"] == (
        "DATA_MISSING_NO_PERSISTED_PREJUMP_CORPUS"
    )
    assert audit_by_milestone[10]["status"] == "REPORT_ONLY_NO_PROMOTION"
    assert plan["guarantees"]["no_fake_odds_or_ev"] is True
    assert plan["guarantees"]["no_retrain"] is True


def test_prejump_loop_plan_attaches_challenger_review_to_packet_command(
    tmp_path,
    monkeypatch,
):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    run_dir = tmp_path / "run"
    challenger_review = tmp_path / "snapshot_challenger_review.json"
    args = loop.build_parser().parse_args(
        [
            "--run-dir",
            str(run_dir),
            "--snapshot-dir",
            str(tmp_path / "snapshots"),
            "--date",
            "2026-05-29",
            "--challenger-review",
            str(challenger_review),
        ]
    )

    plan = loop.build_loop_plan(args)
    by_name = {step["name"]: step for step in plan["steps"]}
    command = by_name["model_review_packet"]["command"]
    design_command = by_name["calibration_layer_design"]["command"]

    assert by_name["snapshot_challenger_review"]["status"] == "PROVIDED_EXTERNALLY"
    assert by_name["snapshot_challenger_review"]["command"] is None
    assert "--challenger-review" in command
    assert command[command.index("--challenger-review") + 1] == str(
        challenger_review
    )
    assert by_name["calibration_layer_design"]["status"] == "READY_TO_RUN"
    assert "scripts/design_calibration_layer.py" in design_command
    assert "--model-review-packet" in design_command
    assert design_command[
        design_command.index("--model-review-packet") + 1
    ].endswith("run/model_review_packet.json")
    assert "--output" in design_command
    assert design_command[design_command.index("--output") + 1].endswith(
        "run/calibration_layer_design.json"
    )
    assert plan["model_review_packet_gate"]["challenger_review_path"] == str(
        challenger_review
    )
    assert plan["model_review_packet_gate"]["status"] == "DATA_MISSING"
    assert plan["calibration_design_gate"]["status"] == "DATA_MISSING"


def test_prejump_loop_plan_waits_for_challenger_before_calibration_design(
    tmp_path,
    monkeypatch,
):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    args = loop.build_parser().parse_args(
        [
            "--run-dir",
            str(tmp_path / "run"),
            "--date",
            "2026-05-29",
        ]
    )

    plan = loop.build_loop_plan(args)
    by_name = {step["name"]: step for step in plan["steps"]}

    assert by_name["calibration_layer_design"]["command"] is None
    assert by_name["calibration_layer_design"]["status"] == (
        "WAITING_FOR_STABLE_REPORT_ONLY_CHALLENGER_REVIEW"
    )
    assert by_name["snapshot_challenger_review"]["command"] is None
    assert by_name["snapshot_challenger_review"]["status"] == (
        "WAITING_FOR_RUN_CHALLENGER_REVIEW_OR_EXTERNAL_REVIEW"
    )


def test_prejump_loop_plan_runs_same_execution_challenger_review(
    tmp_path,
    monkeypatch,
):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    run_dir = tmp_path / "run"
    args = loop.build_parser().parse_args(
        [
            "--run-dir",
            str(run_dir),
            "--date",
            "2026-05-29",
            "--run-challenger-review",
        ]
    )

    plan = loop.build_loop_plan(args)
    by_name = {step["name"]: step for step in plan["steps"]}
    challenger_command = by_name["snapshot_challenger_review"]["command"]
    packet_command = by_name["model_review_packet"]["command"]

    assert by_name["snapshot_challenger_review"]["status"] == "READY_TO_RUN"
    assert "scripts/review_snapshot_challenger.py" in challenger_command
    assert "--dataset" in challenger_command
    assert challenger_command[challenger_command.index("--dataset") + 1].endswith(
        "run/evaluation_dataset.jsonl"
    )
    assert "--output" in challenger_command
    assert challenger_command[challenger_command.index("--output") + 1].endswith(
        "run/snapshot_challenger_review.json"
    )
    assert "--challenger-review" in packet_command
    assert packet_command[packet_command.index("--challenger-review") + 1].endswith(
        "run/snapshot_challenger_review.json"
    )
    assert plan["snapshot_challenger_review_gate"]["status"] == "DATA_MISSING"
    assert plan["model_review_packet_gate"]["challenger_review_path"].endswith(
        "run/snapshot_challenger_review.json"
    )


def test_prejump_loop_plan_reflects_explicit_approvals(tmp_path, monkeypatch):
    monkeypatch.delenv("APPROVE_LIVE_PERSIST", raising=False)
    monkeypatch.delenv("APPROVE_LIVE_ODDS_CAPTURE", raising=False)
    monkeypatch.delenv("APPROVE_RESULT_LABEL_WRITE", raising=False)
    args = loop.build_parser().parse_args(
        [
            "--snapshot-dir",
            str(tmp_path / "snapshots"),
            "--date",
            "2026-05-29",
            "--approve-live-persist",
            "--approve-live-odds-capture",
            "--write-labels-approved",
        ]
    )

    plan = loop.build_loop_plan(args)
    by_name = {step["name"]: step for step in plan["steps"]}
    audit_by_milestone = {
        item["milestone"]: item
        for item in plan["milestone_completion_audit"]["items"]
    }

    assert by_name["approved_persist_ready_subset"]["status"] == (
        "WAITING_FOR_READY_PERSIST_PACKET"
    )
    assert "persist_readiness_gate_not_clean" in by_name[
        "approved_persist_ready_subset"
    ]["reason"]
    assert by_name["opt_in_live_odds_capture"]["status"] == (
        "COVERED_BY_APPROVED_PERSIST_WITH_LIVE_ODDS"
    )
    assert by_name["opt_in_live_odds_capture"]["reason"] == (
        "approved persist command captures live odds before prediction "
        "and persistence"
    )
    assert by_name["approved_official_label_write"]["status"] == (
        "WAITING_FOR_PERSISTED_PREJUMP_SNAPSHOTS"
    )
    assert "--approve-live-persist" in by_name["approved_persist_ready_subset"]["command"]
    assert "--capture-live-odds" in by_name["approved_persist_ready_subset"]["command"]
    assert "--approve-live-odds-capture" in by_name[
        "approved_persist_ready_subset"
    ]["command"]
    assert by_name["opt_in_live_odds_capture"]["command"] is None
    assert "--write-labels-approved" in by_name["approved_official_label_write"]["command"]
    assert plan["approvals"] == {
        "live_persist": True,
        "live_odds_capture": True,
        "result_label_write": True,
        "promotion": False,
    }
    assert plan["approval_provenance"]["live_persist"]["sources"] == ["cli"]
    assert plan["approval_provenance"]["live_odds_capture"]["sources"] == ["cli"]
    assert plan["approval_provenance"]["result_label_write"]["sources"] == ["cli"]
    assert plan["approval_provenance"]["promotion"]["sources"] == []
    assert plan["live_odds_approval_packet"]["status"] == "NOT_READY"
    assert plan["live_odds_approval_packet"]["approval_sources"] == ["cli"]
    assert plan["live_odds_approval_packet"]["can_capture_live_odds_now"] is False
    assert plan["result_label_approval_packet"]["status"] == "NOT_READY"
    assert plan["result_label_approval_packet"]["approval_sources"] == ["cli"]
    assert plan["result_label_approval_packet"]["can_write_labels_now"] is False
    assert plan["result_label_approval_packet"]["hard_stops"] == [
        "persisted_prejump_corpus_missing",
        "result_dry_run_report_missing",
    ]
    assert audit_by_milestone[5]["status"] == (
        "PENDING_APPROVED_FRESH_SAME_RUN_PERSIST"
    )
    assert audit_by_milestone[6]["status"] == "IMPLEMENTED_APPROVAL_PRESENT"


def test_prejump_loop_plan_reflects_env_approval_provenance(monkeypatch):
    monkeypatch.setenv("APPROVE_LIVE_PERSIST", "approved")
    monkeypatch.delenv("APPROVE_LIVE_ODDS_CAPTURE", raising=False)
    monkeypatch.delenv("APPROVE_RESULT_LABEL_WRITE", raising=False)
    monkeypatch.delenv("APPROVE_MODEL_PROMOTION", raising=False)
    args = loop.build_parser().parse_args(
        [
            "--date",
            "2026-05-29",
        ]
    )

    plan = loop.build_loop_plan(args)
    by_name = {step["name"]: step for step in plan["steps"]}

    assert plan["approvals"]["live_persist"] is True
    assert plan["approval_provenance"]["live_persist"]["sources"] == ["env"]
    assert plan["approval_provenance"]["live_persist"]["env_approved"] is True
    assert plan["approval_provenance"]["live_persist"]["cli_approved"] is False


def test_promotion_approval_does_not_override_not_ready_evidence_gates():
    approvals = {
        "live_persist": False,
        "live_odds_capture": False,
        "result_label_write": False,
        "promotion": True,
    }
    promotion_gate = loop._promotion_readiness_gate(
        approvals=approvals,
        evaluation_report_gate={
            "status": "PARTIAL_READY",
            "clean_official_races_evaluated": 27,
        },
        snapshot_challenger_review_gate={
            "status": "NOT_READY",
            "reason": "snapshot_challenger_not_stable_report_only",
            "stability_status": "NOT_STABLE",
        },
        model_review_packet_gate={
            "status": "DATA_MISSING",
            "reason": "model_review_packet_missing",
        },
        calibration_design_gate={
            "status": "DATA_MISSING",
            "reason": "calibration_design_report_missing",
        },
    )

    assert promotion_gate["approval_present"] is True
    assert promotion_gate["status"] == "APPROVAL_PRESENT_EVIDENCE_NOT_READY"
    assert promotion_gate["ready_for_separate_promotion_review"] is False
    assert promotion_gate["promotion_allowed_by_loop"] is False
    assert promotion_gate["promotion_action_taken"] == "none"
    assert promotion_gate["historical_clean_official_races_can_satisfy_minimum"] is True
    assert promotion_gate["current_day_races_required_for_minimum"] is False
    assert "clean_official_evaluated_races_below_minimum" in promotion_gate[
        "blockers"
    ]
    assert "snapshot_challenger_review_gate_not_ready" in promotion_gate["blockers"]
    assert "model_review_packet_gate_not_ready" in promotion_gate["blockers"]

    operator_action = loop._operator_next_action_report(
        approvals=approvals,
        current_corpus={
            "status": "NO_READY_PERSISTED_PREJUMP_SNAPSHOTS_FOR_DATE",
            "ready_persisted_prediction_snapshot_count_for_date": 0,
        },
        persist_packet={
            "status": "NOT_READY",
            "hard_stops": ["ready_count_zero"],
            "ready_count": 0,
            "not_ready_count": 0,
        },
        live_odds_packet={"status": "NOT_READY"},
        result_label_packet={"status": "NOT_READY", "hard_stops": []},
        milestone_audit={
            "overall_status": "INCOMPLETE",
            "completed_count": 0,
            "incomplete_count": 1,
            "items": [],
        },
        promotion_readiness_gate=promotion_gate,
    )

    assert "APPROVE_MODEL_PROMOTION" not in operator_action[
        "blocked_approval_gates"
    ]
    assert operator_action["promotion_next_step_status"] == (
        "PROMOTION_APPROVAL_ACCEPTED_EVIDENCE_NOT_READY"
    )
    assert operator_action["promotion_ready_for_separate_review"] is False
    assert operator_action["promotion_action_taken"] == "none"


def test_prejump_loop_uses_historical_100plus_packet_for_promotion_evidence(
    tmp_path,
    monkeypatch,
):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)

    evidence_dir = tmp_path / "historical"
    evidence_dir.mkdir()
    eval_report = evidence_dir / "evaluation_report.json"
    dataset_path = evidence_dir / "evaluation_dataset.jsonl"
    challenger_review = evidence_dir / "snapshot_challenger_review.json"
    model_packet = evidence_dir / "model_review_packet.json"
    calibration_design = evidence_dir / "calibration_layer_design.json"
    dataset_path.write_text("{}\n{}\n{}\n{}\n", encoding="utf-8")
    eval_report.write_text(
        json.dumps(_clean_evaluation_report(dataset_path)),
        encoding="utf-8",
    )
    challenger_review.write_text(
        json.dumps(_clean_snapshot_challenger_review_report(dataset_path)),
        encoding="utf-8",
    )
    packet = _clean_model_review_packet(
        eval_report,
        dataset_path,
        challenger_review,
    )
    packet["review_gate"] = {
        "minimum_clean_evaluated_races": 100,
        "clean_official_evaluated_races": 105,
        "clean_official_snapshot_instances": 105,
        "clean_official_runner_rows": 735,
        "retrain_gate": {
            "status": "READY_FOR_REVIEW",
            "action_taken": "none",
        },
        "promotion_gate": {
            "status": "REPORT_ONLY",
            "action_taken": "none",
        },
    }
    model_packet.write_text(json.dumps(packet), encoding="utf-8")
    calibration_design.write_text(
        json.dumps(_clean_calibration_design_report(model_packet)),
        encoding="utf-8",
    )

    args = loop.build_parser().parse_args(
        [
            "--snapshot-dir",
            str(tmp_path / "snapshots"),
            "--run-dir",
            str(tmp_path / "run"),
            "--date",
            "2026-05-29",
            "--approve-promotion",
            "--promotion-model-review-packet",
            str(model_packet),
            "--promotion-calibration-design",
            str(calibration_design),
        ]
    )

    plan = loop.build_loop_plan(args)
    gate = plan["promotion_readiness_gate"]
    operator_action = plan["operator_next_action"]
    by_name = {step["name"]: step for step in plan["steps"]}
    audit_by_milestone = {
        item["milestone"]: item
        for item in plan["milestone_completion_audit"]["items"]
    }

    assert plan["promotion_model_review_packet_gate"]["status"] == "READY"
    assert plan["promotion_calibration_design_gate"]["status"] == "READY"
    assert gate["status"] == "APPROVAL_PRESENT_EVIDENCE_READY_REPORT_ONLY"
    assert gate["promotion_evidence_source"] == (
        "external_report_only_promotion_evidence"
    )
    assert gate["promotion_evidence_clean_official_evaluated_races"] == 105
    assert gate["ready_for_separate_promotion_review"] is True
    assert gate["promotion_action_taken"] == "none"
    assert gate["promotion_allowed_by_loop"] is False
    assert "clean_official_evaluated_races_below_minimum" not in gate["blockers"]
    assert operator_action["promotion_next_step_status"] == (
        "PROMOTION_APPROVAL_ACCEPTED_READY_FOR_SEPARATE_REVIEW"
    )
    assert operator_action["promotion_evidence_clean_official_evaluated_races"] == 105
    assert operator_action["promotion_ready_for_separate_review"] is True
    assert by_name["promotion_controlled_loop"]["status"] == (
        "APPROVAL_PRESENT_EVIDENCE_READY_REPORT_ONLY"
    )
    assert audit_by_milestone[10]["status"] == (
        "PROMOTION_EVIDENCE_READY_FOR_SEPARATE_REVIEW_REPORT_ONLY"
    )
    assert "promotion readiness gate: APPROVAL_PRESENT_EVIDENCE_READY_REPORT_ONLY" in (
        audit_by_milestone[10]["evidence"]
    )
    assert plan["guarantees"]["no_model_promotion"] is True


def test_prejump_loop_allows_result_steps_after_persisted_corpus_exists(tmp_path, monkeypatch):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    snapshot_dir = tmp_path / "snapshots"
    date_dir = snapshot_dir / "2026-05-29" / "TEST"
    date_dir.mkdir(parents=True)
    snapshot_path = date_dir / "race-1.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "schema_version": "prediction_snapshot_v1",
                "race_id": "Race 1 - TEST - 2026-05-29",
                "is_pre_jump_snapshot": True,
                "snapshot_state": "pre_jump_feature_freeze",
                "snapshot_readiness": {"status": "READY"},
                "predictions": READY_PREDICTIONS,
            }
        ),
        encoding="utf-8",
    )
    args = loop.build_parser().parse_args(
        [
            "--snapshot-dir",
            str(snapshot_dir),
            "--date",
            "2026-05-29",
            "--write-labels-approved",
        ]
    )

    plan = loop.build_loop_plan(args)
    by_name = {step["name"]: step for step in plan["steps"]}
    audit_by_milestone = {
        item["milestone"]: item
        for item in plan["milestone_completion_audit"]["items"]
    }

    assert plan["current_corpus"]["status"] == "READY_PERSISTED_PREJUMP_SNAPSHOTS_PRESENT"
    assert plan["current_corpus"]["ready_persisted_prediction_snapshot_count_for_date"] == 1
    assert plan["current_corpus"]["ready_persisted_prediction_snapshot_examples"] == [
        str(snapshot_path)
    ]
    assert by_name["official_result_ingest_dry_run"]["status"] == "READY_TO_RUN"
    assert "--output" in by_name["official_result_ingest_dry_run"]["command"]
    assert "--require-ready-snapshot" in by_name["official_result_ingest_dry_run"]["command"]
    assert by_name["approved_official_label_write"]["status"] == (
        "WAITING_FOR_CLEAN_RESULT_DRY_RUN"
    )
    assert plan["result_label_approval_packet"]["status"] == "NOT_READY"
    assert plan["result_label_approval_packet"]["can_write_labels_now"] is False
    assert plan["result_label_approval_packet"]["hard_stops"] == [
        "result_dry_run_report_missing"
    ]
    assert plan["result_label_approval_packet"][
        "ready_persisted_prediction_snapshot_count_for_date"
    ] == 1
    assert audit_by_milestone[5]["complete"] is True
    assert audit_by_milestone[5]["status"] == (
        "COMPLETE_READY_PERSISTED_PREJUMP_SNAPSHOTS_PRESENT"
    )
    assert audit_by_milestone[7]["status"] == (
        "WAITING_FOR_CLEAN_OFFICIAL_RESULT_DRY_RUN"
    )
    assert "--approved-dry-run-report" in by_name["approved_official_label_write"]["command"]
    assert "--require-ready-snapshot" in by_name["approved_official_label_write"]["command"]


def test_prejump_loop_waits_for_persisted_future_races_before_result_dry_run(
    tmp_path,
    monkeypatch,
):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    snapshot_dir = tmp_path / "snapshots"
    date_dir = snapshot_dir / "2999-06-01" / "TEST"
    date_dir.mkdir(parents=True)
    snapshot_path = date_dir / "race-1.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "schema_version": "prediction_snapshot_v1",
                "race_id": "Race 1 - TEST - 2999-06-01",
                "venue": "TEST",
                "race_number": 1,
                "race_date": "2999-06-01",
                "jump_datetime": "2999-06-01T12:00:00+10:00",
                "is_pre_jump_snapshot": True,
                "snapshot_state": "pre_jump_feature_freeze",
                "snapshot_readiness": {"status": "READY"},
                "predictions": READY_PREDICTIONS,
            }
        ),
        encoding="utf-8",
    )
    args = loop.build_parser().parse_args(
        [
            "--snapshot-dir",
            str(snapshot_dir),
            "--date",
            "2999-06-01",
            "--write-labels-approved",
        ]
    )

    plan = loop.build_loop_plan(args)
    by_name = {step["name"]: step for step in plan["steps"]}
    audit_by_milestone = {
        item["milestone"]: item
        for item in plan["milestone_completion_audit"]["items"]
    }

    jump_status = plan["current_corpus"]["persisted_snapshot_jump_status"]
    assert jump_status["known_future_not_jumped_count"] == 1
    assert jump_status["wait_for_known_future_jumps_before_result_dry_run"] is True
    assert plan["current_corpus"][
        "result_dry_run_waiting_for_known_future_jumps"
    ] is True
    assert by_name["official_result_ingest_dry_run"]["status"] == (
        "WAITING_FOR_PERSISTED_RACES_TO_JUMP"
    )
    assert by_name["official_result_ingest_dry_run"]["reason"] == (
        "known persisted pre-jump snapshots have future jump times"
    )
    assert by_name["approved_official_label_write"]["status"] == (
        "WAITING_FOR_PERSISTED_RACES_TO_JUMP"
    )
    assert plan["result_label_approval_packet"]["status"] == "NOT_READY"
    assert "persisted_prejump_races_not_jumped_yet" in plan[
        "result_label_approval_packet"
    ]["hard_stops"]
    assert audit_by_milestone[7]["status"] == (
        "WAITING_FOR_PERSISTED_RACES_TO_JUMP"
    )
    assert plan["operator_next_action"]["next_step_status"] == (
        "WAIT_FOR_PERSISTED_RACES_TO_JUMP_BEFORE_RESULT_DRY_RUN"
    )
    assert plan["operator_next_action"]["command_template"] is None
    assert plan["operator_next_action"][
        "result_dry_run_waiting_for_known_future_jumps"
    ] is True
    assert plan["operator_next_action"][
        "result_dry_run_safe_after_latest_known_jump_local"
    ] == "2999-06-01T12:00:00+10:00"
    assert plan["current_corpus"]["ready_persisted_prediction_snapshot_examples"] == [
        str(snapshot_path)
    ]


def test_prejump_loop_allows_result_dry_run_after_known_jump_time_has_passed(
    tmp_path,
    monkeypatch,
):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    snapshot_dir = tmp_path / "snapshots"
    date_dir = snapshot_dir / "2000-01-01" / "TEST"
    date_dir.mkdir(parents=True)
    (date_dir / "race-1.json").write_text(
        json.dumps(
            {
                "schema_version": "prediction_snapshot_v1",
                "race_id": "Race 1 - TEST - 2000-01-01",
                "venue": "TEST",
                "race_number": 1,
                "race_date": "2000-01-01",
                "jump_datetime": "2000-01-01T12:00:00+10:00",
                "is_pre_jump_snapshot": True,
                "snapshot_state": "pre_jump_feature_freeze",
                "snapshot_readiness": {"status": "READY"},
                "predictions": READY_PREDICTIONS,
            }
        ),
        encoding="utf-8",
    )
    args = loop.build_parser().parse_args(
        [
            "--snapshot-dir",
            str(snapshot_dir),
            "--date",
            "2000-01-01",
        ]
    )

    plan = loop.build_loop_plan(args)
    by_name = {step["name"]: step for step in plan["steps"]}

    jump_status = plan["current_corpus"]["persisted_snapshot_jump_status"]
    assert jump_status["known_future_not_jumped_count"] == 0
    assert jump_status["known_jumped_or_due_count"] == 1
    assert by_name["official_result_ingest_dry_run"]["status"] == "READY_TO_RUN"
    assert plan["operator_next_action"]["next_step_status"] == (
        "RUN_OR_REFRESH_OFFICIAL_RESULT_DRY_RUN"
    )


def test_prejump_loop_allows_label_write_after_clean_result_dry_run_report(
    tmp_path,
    monkeypatch,
):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    db_path = tmp_path / "labels.sqlite"
    upcoming_dir = tmp_path / "upcoming"
    upcoming_dir.mkdir()
    snapshot_dir = tmp_path / "snapshots"
    date_dir = snapshot_dir / "2026-05-29" / "TEST"
    date_dir.mkdir(parents=True)
    (date_dir / "race-1.json").write_text(
        json.dumps(
            {
                "schema_version": "prediction_snapshot_v1",
                "race_id": "Race 1 - TEST - 2026-05-29",
                "is_pre_jump_snapshot": True,
                "snapshot_state": "pre_jump_feature_freeze",
                "snapshot_readiness": {"status": "READY"},
                "predictions": READY_PREDICTIONS,
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    scope = {
        "db_path": str(db_path.resolve()),
        "date": "2026-05-29",
        "upcoming_dir": str(upcoming_dir.resolve()),
        "snapshot_dir": str(snapshot_dir.resolve()),
        "race_ids": [],
        "require_ready_snapshot": True,
    }
    (run_dir / "result_ingest_dry_run_report.json").write_text(
        json.dumps(_clean_result_dry_run_report(scope)),
        encoding="utf-8",
    )
    (run_dir / "label_write_readiness_validation.json").write_text(
        json.dumps(_clean_label_write_readiness_report(scope)),
        encoding="utf-8",
    )
    (run_dir / "label_write_readiness_validation.json").write_text(
        json.dumps(_clean_label_write_readiness_report(scope)),
        encoding="utf-8",
    )
    args = loop.build_parser().parse_args(
        [
            "--db",
            str(db_path),
            "--upcoming-dir",
            str(upcoming_dir),
            "--snapshot-dir",
            str(snapshot_dir),
            "--run-dir",
            str(run_dir),
            "--date",
            "2026-05-29",
            "--write-labels-approved",
        ]
    )

    plan = loop.build_loop_plan(args)
    by_name = {step["name"]: step for step in plan["steps"]}
    audit_by_milestone = {
        item["milestone"]: item
        for item in plan["milestone_completion_audit"]["items"]
    }

    assert plan["result_dry_run_report_gate"]["status"] == "READY"
    assert by_name["approved_official_label_write"]["status"] == "READY_TO_RUN"
    assert plan["result_label_approval_packet"]["status"] == (
        "APPROVAL_PRESENT_READY_TO_WRITE_OFFICIAL_LABELS"
    )
    assert plan["result_label_approval_packet"]["can_write_labels_now"] is True
    assert plan["result_label_approval_packet"]["approval_sources"] == ["cli"]
    assert plan["result_label_approval_packet"]["hard_stops"] == []
    assert plan["result_label_approval_packet"]["result_dry_run_report_age_seconds"] >= 0
    assert (
        plan["result_label_approval_packet"]["result_dry_run_report_max_age_seconds"]
        == loop.RESULT_DRY_RUN_REPORT_MAX_AGE_SECONDS
    )
    assert plan["result_label_approval_packet"]["result_dry_run_report_expires_at_utc"]
    assert plan["result_label_approval_packet"]["result_dry_run_report_expires_at_local"]
    assert plan["result_label_approval_packet"][
        "result_dry_run_report_expires_at_local_timezone"
    ] == loop.LOCAL_OPERATOR_TIMEZONE
    assert (
        0
        < plan["result_label_approval_packet"][
            "result_dry_run_report_seconds_until_expiry"
        ]
        <= loop.RESULT_DRY_RUN_REPORT_MAX_AGE_SECONDS
    )
    assert plan["result_label_approval_packet"][
        "approval_must_arrive_before_report_expiry"
    ] is True
    assert plan["result_label_approval_packet"]["rerun_required_after_expiry"] is True
    assert "--write-labels-approved" in plan["result_label_approval_packet"][
        "planned_label_write_command"
    ]
    assert plan["result_label_approval_packet"]["official_first_policy"] == {
        "require_ready_prejump_snapshot": True,
        "dry_run_must_be_clean_and_fresh": True,
        "participant_alignment_required": True,
        "official_or_complete_result_required": True,
        "winner_only_or_partial_results_not_label_ready": True,
    }
    assert audit_by_milestone[7]["status"] == "READY_FOR_APPROVED_LABEL_WRITE"
    assert "--approved-dry-run-report" in by_name["approved_official_label_write"]["command"]
    assert "--write-labels-approved" in by_name["approved_official_label_write"]["command"]


def test_prejump_loop_label_packet_waits_for_approval_after_clean_result_dry_run(
    tmp_path,
    monkeypatch,
):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    db_path = tmp_path / "labels.sqlite"
    upcoming_dir = tmp_path / "upcoming"
    upcoming_dir.mkdir()
    snapshot_dir = tmp_path / "snapshots"
    date_dir = snapshot_dir / "2026-05-29" / "TEST"
    date_dir.mkdir(parents=True)
    (date_dir / "race-1.json").write_text(
        json.dumps(
            {
                "schema_version": "prediction_snapshot_v1",
                "race_id": "Race 1 - TEST - 2026-05-29",
                "is_pre_jump_snapshot": True,
                "snapshot_state": "pre_jump_feature_freeze",
                "snapshot_readiness": {"status": "READY"},
                "predictions": READY_PREDICTIONS,
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    scope = {
        "db_path": str(db_path.resolve()),
        "date": "2026-05-29",
        "upcoming_dir": str(upcoming_dir.resolve()),
        "snapshot_dir": str(snapshot_dir.resolve()),
        "race_ids": [],
        "require_ready_snapshot": True,
    }
    (run_dir / "result_ingest_dry_run_report.json").write_text(
        json.dumps(_clean_result_dry_run_report(scope)),
        encoding="utf-8",
    )
    (run_dir / "label_write_readiness_validation.json").write_text(
        json.dumps(_clean_label_write_readiness_report(scope)),
        encoding="utf-8",
    )
    args = loop.build_parser().parse_args(
        [
            "--db",
            str(db_path),
            "--upcoming-dir",
            str(upcoming_dir),
            "--snapshot-dir",
            str(snapshot_dir),
            "--run-dir",
            str(run_dir),
            "--date",
            "2026-05-29",
        ]
    )

    plan = loop.build_loop_plan(args)
    packet = plan["result_label_approval_packet"]
    by_name = {step["name"]: step for step in plan["steps"]}

    assert packet["status"] == "AWAITING_EXPLICIT_APPROVAL_READY_FOR_LABEL_WRITE"
    assert packet["can_write_labels_now"] is False
    assert packet["approval_required"] is True
    assert packet["hard_stops"] == []
    assert "--write-labels-approved" not in packet["planned_label_write_command"]
    assert "--write-labels-approved" in packet[
        "approved_label_write_command_template"
    ]
    assert packet["approval_command_requires_explicit_operator_confirmation"] is True
    same_run_command = packet["approved_same_run_execute_ready_command_template"]
    assert "scripts/prejump_prediction_loop.py" in same_run_command
    assert "--execute-ready" in same_run_command
    assert "--write-labels-approved" in same_run_command
    assert "--approve-live-persist" not in same_run_command
    assert "--approve-live-odds-capture" not in same_run_command
    assert "--output" in same_run_command
    assert same_run_command[same_run_command.index("--output") + 1].endswith(
        "loop_plan_execute_approved_label_write.json"
    )
    assert packet["same_run_execute_ready_command_template_status"] == (
        "READY_FOR_EXPLICIT_APPROVAL_AND_FRESH_RECHECK"
    )
    assert packet[
        "same_run_execute_ready_command_requires_explicit_operator_confirmation"
    ] is True
    assert packet["same_run_execute_ready_rechecks"] == [
        "current_persisted_prejump_corpus",
        "official_result_ingest_dry_run",
        "result_label_approval_gate",
        "official_first_scope_match",
    ]
    assert packet["label_write_preflight_packet_status"] == "DATA_MISSING"
    assert packet["planned_label_write_preflight_packet_command"][:2] == [
        loop._repo_python(),
        "scripts/build_label_write_preflight_packet.py",
    ]
    assert "--label-readiness" in packet["planned_label_write_preflight_packet_command"]
    assert "--result-dry-run-report" in packet[
        "planned_label_write_preflight_packet_command"
    ]
    assert by_name["label_write_preflight_packet"]["status"] == (
        "READY_FOR_OPERATOR_RUN_AFTER_LOOP_PLAN_WRITE"
    )
    assert by_name["label_write_preflight_packet"]["write_scope"] == (
        "report_only_no_label_write"
    )
    operator_command = plan["operator_next_action"]["command_template"]
    assert plan["operator_next_action"]["next_step_status"] == (
        "RUN_LABEL_WRITE_PREFLIGHT_PACKET"
    )
    assert plan["operator_next_action"]["approval_required"] is False
    assert plan["operator_next_action"]["required_gate"] is None
    assert "scripts/build_label_write_preflight_packet.py" in operator_command
    assert "--label-readiness" in operator_command
    assert "--result-dry-run-report" in operator_command
    assert plan["operator_next_action"][
        "result_label_same_run_execute_ready_command_template_status"
    ] == "READY_FOR_EXPLICIT_APPROVAL_AND_FRESH_RECHECK"
    assert plan["operator_next_action"][
        "result_label_same_run_execute_ready_rechecks"
    ] == [
        "current_persisted_prejump_corpus",
        "official_result_ingest_dry_run",
        "result_label_approval_gate",
        "official_first_scope_match",
    ]
    assert plan["operator_next_action"]["label_write_preflight_packet_status"] == (
        "DATA_MISSING"
    )
    assert "scripts/build_label_write_preflight_packet.py" in plan[
        "operator_next_action"
    ]["label_write_preflight_packet_command_template"]


def test_prejump_loop_rejects_stale_result_dry_run_report_for_label_write(
    tmp_path,
    monkeypatch,
):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    db_path = tmp_path / "labels.sqlite"
    upcoming_dir = tmp_path / "upcoming"
    upcoming_dir.mkdir()
    snapshot_dir = tmp_path / "snapshots"
    date_dir = snapshot_dir / "2026-05-29" / "TEST"
    date_dir.mkdir(parents=True)
    (date_dir / "race-1.json").write_text(
        json.dumps(
            {
                "schema_version": "prediction_snapshot_v1",
                "race_id": "Race 1 - TEST - 2026-05-29",
                "is_pre_jump_snapshot": True,
                "snapshot_state": "pre_jump_feature_freeze",
                "snapshot_readiness": {"status": "READY"},
                "predictions": READY_PREDICTIONS,
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    report_path = run_dir / "result_ingest_dry_run_report.json"
    scope = {
        "db_path": str(db_path.resolve()),
        "date": "2026-05-29",
        "upcoming_dir": str(upcoming_dir.resolve()),
        "snapshot_dir": str(snapshot_dir.resolve()),
        "race_ids": [],
        "require_ready_snapshot": True,
    }
    report_path.write_text(
        json.dumps(_clean_result_dry_run_report(scope)),
        encoding="utf-8",
    )
    stale_time = time.time() - loop.RESULT_DRY_RUN_REPORT_MAX_AGE_SECONDS - 60
    os.utime(report_path, (stale_time, stale_time))
    args = loop.build_parser().parse_args(
        [
            "--db",
            str(db_path),
            "--upcoming-dir",
            str(upcoming_dir),
            "--snapshot-dir",
            str(snapshot_dir),
            "--run-dir",
            str(run_dir),
            "--date",
            "2026-05-29",
            "--write-labels-approved",
        ]
    )

    plan = loop.build_loop_plan(args)
    by_name = {step["name"]: step for step in plan["steps"]}

    assert plan["result_dry_run_report_gate"]["status"] == "NOT_READY"
    assert plan["result_dry_run_report_gate"]["clean"] is False
    assert plan["result_dry_run_report_gate"]["fresh_for_plan"] is False
    assert "result_dry_run_report_stale" in plan["result_dry_run_report_gate"][
        "reason"
    ]
    assert plan["result_label_approval_packet"]["status"] == "NOT_READY"
    assert plan["result_label_approval_packet"]["can_write_labels_now"] is False
    assert "result_dry_run_report_not_fresh" in plan[
        "result_label_approval_packet"
    ]["hard_stops"]
    assert by_name["approved_official_label_write"]["status"] == (
        "WAITING_FOR_CLEAN_RESULT_DRY_RUN"
    )
    assert plan["operator_next_action"]["next_step_status"] == (
        "RUN_OR_REFRESH_OFFICIAL_RESULT_DRY_RUN"
    )
    assert plan["operator_next_action"]["required_gate"] is None
    assert plan["operator_next_action"]["approval_required"] is False
    assert plan["operator_next_action"]["result_dry_run_fresh_for_plan"] is False
    assert plan["operator_next_action"]["result_label_approval_window_status"] == (
        "REFRESH_REQUIRED"
    )
    assert plan["operator_next_action"][
        "result_label_approval_command_template_status"
    ] == "BLOCKED_BY_HARD_STOPS"
    assert "--dry-run" in plan["operator_next_action"]["command_template"]
    assert (
        "scripts/ingest_results_for_date.py"
        in plan["operator_next_action"]["command_template"]
    )


def test_prejump_loop_marks_evaluation_and_diagnosis_complete_from_clean_report(
    tmp_path,
    monkeypatch,
):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    snapshot_dir = tmp_path / "snapshots"
    date_dir = snapshot_dir / "2026-05-29" / "TEST"
    date_dir.mkdir(parents=True)
    (date_dir / "race-1.json").write_text(
        json.dumps(
            {
                "schema_version": "prediction_snapshot_v1",
                "race_id": "Race 1 - TEST - 2026-05-29",
                "is_pre_jump_snapshot": True,
                "snapshot_state": "pre_jump_feature_freeze",
                "snapshot_readiness": {"status": "READY"},
                "predictions": READY_PREDICTIONS,
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    dataset_path = run_dir / "evaluation_dataset.jsonl"
    dataset_path.write_text("{}\n{}\n{}\n{}\n", encoding="utf-8")
    (run_dir / "evaluation_report.json").write_text(
        json.dumps(_clean_evaluation_report(dataset_path)),
        encoding="utf-8",
    )
    args = loop.build_parser().parse_args(
        [
            "--snapshot-dir",
            str(snapshot_dir),
            "--run-dir",
            str(run_dir),
            "--date",
            "2026-05-29",
        ]
    )

    plan = loop.build_loop_plan(args)
    audit_by_milestone = {
        item["milestone"]: item
        for item in plan["milestone_completion_audit"]["items"]
    }

    assert plan["evaluation_report_gate"]["status"] == "READY"
    assert plan["evaluation_report_gate"]["dataset_ready"] is True
    assert plan["evaluation_report_gate"]["clean_official_metrics_ready"] is True
    assert audit_by_milestone[8]["complete"] is True
    assert audit_by_milestone[8]["status"] == "COMPLETE_ROLLING_EVALUATION_READY"
    assert audit_by_milestone[9]["complete"] is True
    assert audit_by_milestone[9]["status"] == "COMPLETE_MODEL_QUALITY_DIAGNOSIS_READY"


def test_prejump_loop_uses_evaluation_snapshot_manifest(
    tmp_path,
    monkeypatch,
):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    manifest = tmp_path / "clean_snapshot_manifest.txt"
    manifest.write_text(
        str(snapshot_dir / "race-1.json") + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    args = loop.build_parser().parse_args(
        [
            "--snapshot-dir",
            str(snapshot_dir),
            "--run-dir",
            str(run_dir),
            "--date",
            "2026-05-29",
            "--evaluation-snapshots-manifest",
            str(manifest),
        ]
    )

    plan = loop.build_loop_plan(args)
    steps = {step["name"]: step for step in plan["steps"]}
    command = steps["rolling_evaluation_dataset"]["command"]

    assert plan["evaluation_snapshot_scope"] == "manifest"
    assert plan["evaluation_snapshots_manifest"].endswith(
        "clean_snapshot_manifest.txt"
    )
    assert "--snapshots-manifest" in command
    assert str(manifest) in command
    assert "--snapshots" not in command


def test_prejump_loop_reports_historical_evaluation_without_current_corpus(
    tmp_path,
    monkeypatch,
):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    snapshot_dir = tmp_path / "snapshots"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    dataset_path = run_dir / "evaluation_dataset.jsonl"
    dataset_path.write_text("{}\n{}\n{}\n{}\n", encoding="utf-8")
    (run_dir / "evaluation_report.json").write_text(
        json.dumps(_clean_evaluation_report(dataset_path)),
        encoding="utf-8",
    )
    args = loop.build_parser().parse_args(
        [
            "--snapshot-dir",
            str(snapshot_dir),
            "--run-dir",
            str(run_dir),
            "--date",
            "2026-05-29",
        ]
    )

    plan = loop.build_loop_plan(args)
    audit_by_milestone = {
        item["milestone"]: item
        for item in plan["milestone_completion_audit"]["items"]
    }

    assert plan["current_corpus"]["status"] == (
        "NO_READY_PERSISTED_PREJUMP_SNAPSHOTS_FOR_DATE"
    )
    assert plan["evaluation_report_gate"]["status"] == "READY"
    assert audit_by_milestone[8]["complete"] is False
    assert audit_by_milestone[8]["status"] == (
        "REPORT_ONLY_EVALUATION_READY_AWAITING_PERSISTED_CURRENT_CORPUS"
    )
    assert "historical/report-only" in audit_by_milestone[8]["remaining"][-1]
    assert audit_by_milestone[9]["complete"] is False
    assert audit_by_milestone[9]["status"] == (
        "REPORT_ONLY_MODEL_QUALITY_READY_AWAITING_PERSISTED_CURRENT_CORPUS"
    )
    assert "current target corpus" in audit_by_milestone[9]["remaining"][-1]


def test_prejump_loop_evaluation_gate_rejects_missing_official_metrics(
    tmp_path,
    monkeypatch,
):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    dataset_path = run_dir / "evaluation_dataset.jsonl"
    dataset_path.write_text("{}\n", encoding="utf-8")
    report = _clean_evaluation_report(dataset_path)
    report["evaluation_dataset_rows_written"] = 1
    report["clean_official_evaluation"]["races_evaluated"] = 0
    report["clean_official_evaluation"]["metrics_by_arm"]["model_only"]["top1"] = None
    (run_dir / "evaluation_report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    args = loop.build_parser().parse_args(
        [
            "--run-dir",
            str(run_dir),
            "--date",
            "2026-05-29",
        ]
    )

    plan = loop.build_loop_plan(args)

    assert plan["evaluation_report_gate"]["status"] == "NOT_READY"
    assert "clean_official_races_evaluated_zero" in plan[
        "evaluation_report_gate"
    ]["reason"]
    assert "clean_official_metrics_missing:top1" in plan[
        "evaluation_report_gate"
    ]["reason"]


def test_prejump_loop_evaluation_gate_rejects_stale_report(
    tmp_path,
    monkeypatch,
):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    snapshot_dir = tmp_path / "snapshots"
    date_dir = snapshot_dir / "2026-05-29" / "TEST"
    date_dir.mkdir(parents=True)
    (date_dir / "race-1.json").write_text(
        json.dumps(
            {
                "schema_version": "prediction_snapshot_v1",
                "race_id": "Race 1 - TEST - 2026-05-29",
                "is_pre_jump_snapshot": True,
                "snapshot_state": "pre_jump_feature_freeze",
                "snapshot_readiness": {"status": "READY"},
                "predictions": READY_PREDICTIONS,
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    dataset_path = run_dir / "evaluation_dataset.jsonl"
    dataset_path.write_text("{}\n{}\n{}\n{}\n", encoding="utf-8")
    report_path = run_dir / "evaluation_report.json"
    report_path.write_text(
        json.dumps(_clean_evaluation_report(dataset_path)),
        encoding="utf-8",
    )
    stale_time = time.time() - loop.EVALUATION_REPORT_MAX_AGE_SECONDS - 60
    os.utime(report_path, (stale_time, stale_time))
    args = loop.build_parser().parse_args(
        [
            "--snapshot-dir",
            str(snapshot_dir),
            "--run-dir",
            str(run_dir),
            "--date",
            "2026-05-29",
        ]
    )

    plan = loop.build_loop_plan(args)
    audit_by_milestone = {
        item["milestone"]: item
        for item in plan["milestone_completion_audit"]["items"]
    }

    assert plan["evaluation_report_gate"]["status"] == "NOT_READY"
    assert plan["evaluation_report_gate"]["fresh_for_plan"] is False
    assert "evaluation_report_stale" in plan["evaluation_report_gate"]["reason"]
    assert audit_by_milestone[8]["complete"] is False
    assert audit_by_milestone[9]["complete"] is False


def test_prejump_loop_evaluation_gate_rejects_dataset_scope_mismatch(
    tmp_path,
    monkeypatch,
):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    dataset_path = run_dir / "evaluation_dataset.jsonl"
    dataset_path.write_text("{}\n{}\n{}\n{}\n", encoding="utf-8")
    report = _clean_evaluation_report(run_dir / "other_dataset.jsonl")
    (run_dir / "evaluation_report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    args = loop.build_parser().parse_args(
        [
            "--run-dir",
            str(run_dir),
            "--date",
            "2026-05-29",
        ]
    )

    plan = loop.build_loop_plan(args)

    assert plan["evaluation_report_gate"]["status"] == "NOT_READY"
    assert "evaluation_dataset_output_scope_mismatch" in plan[
        "evaluation_report_gate"
    ]["reason"]


def test_prejump_loop_ignores_not_ready_prejump_snapshots_for_result_steps(
    tmp_path,
    monkeypatch,
):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    snapshot_dir = tmp_path / "snapshots"
    date_dir = snapshot_dir / "2026-05-29" / "TEST"
    date_dir.mkdir(parents=True)
    (date_dir / "race-1.json").write_text(
        json.dumps(
            {
                "schema_version": "prediction_snapshot_v1",
                "race_id": "Race 1 - TEST - 2026-05-29",
                "is_pre_jump_snapshot": True,
                "snapshot_state": "pre_jump_feature_freeze",
                "snapshot_readiness": {"status": "NOT_READY"},
                "predictions": READY_PREDICTIONS,
            }
        ),
        encoding="utf-8",
    )
    args = loop.build_parser().parse_args(
        [
            "--snapshot-dir",
            str(snapshot_dir),
            "--date",
            "2026-05-29",
            "--write-labels-approved",
        ]
    )

    plan = loop.build_loop_plan(args)
    by_name = {step["name"]: step for step in plan["steps"]}

    assert plan["current_corpus"]["status"] == "NO_READY_PERSISTED_PREJUMP_SNAPSHOTS_FOR_DATE"
    assert plan["current_corpus"]["ready_persisted_prediction_snapshot_count_for_date"] == 0
    assert by_name["official_result_ingest_dry_run"]["status"] == (
        "WAITING_FOR_PERSISTED_PREJUMP_SNAPSHOTS"
    )
    assert by_name["approved_official_label_write"]["status"] == (
        "WAITING_FOR_PERSISTED_PREJUMP_SNAPSHOTS"
    )


def test_prejump_loop_rejects_empty_ready_snapshot_as_current_corpus(
    tmp_path,
    monkeypatch,
):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    snapshot_dir = tmp_path / "snapshots"
    date_dir = snapshot_dir / "2026-05-29" / "TEST"
    date_dir.mkdir(parents=True)
    (date_dir / "race-1.json").write_text(
        json.dumps(
            {
                "schema_version": "prediction_snapshot_v1",
                "race_id": "Race 1 - TEST - 2026-05-29",
                "is_pre_jump_snapshot": True,
                "snapshot_state": "pre_jump_feature_freeze",
                "snapshot_readiness": {"status": "READY"},
                "predictions": [],
            }
        ),
        encoding="utf-8",
    )
    args = loop.build_parser().parse_args(
        [
            "--snapshot-dir",
            str(snapshot_dir),
            "--date",
            "2026-05-29",
        ]
    )

    plan = loop.build_loop_plan(args)

    assert plan["current_corpus"]["status"] == "NO_READY_PERSISTED_PREJUMP_SNAPSHOTS_FOR_DATE"
    assert plan["current_corpus"]["ready_persisted_prediction_snapshot_count_for_date"] == 0


def test_prejump_loop_rejects_result_contaminated_ready_snapshot_as_current_corpus(
    tmp_path,
    monkeypatch,
):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    snapshot_dir = tmp_path / "snapshots"
    date_dir = snapshot_dir / "2026-05-29" / "TEST"
    date_dir.mkdir(parents=True)
    (date_dir / "race-1.json").write_text(
        json.dumps(
            {
                "schema_version": "prediction_snapshot_v1",
                "race_id": "Race 1 - TEST - 2026-05-29",
                "is_pre_jump_snapshot": True,
                "snapshot_state": "pre_jump_feature_freeze",
                "snapshot_readiness": {"status": "READY"},
                "predictions": [
                    {
                        "dog_name": "Alpha Runner",
                        "box_number": 1,
                        "win_prob_norm": 0.4,
                        "actual_win": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    args = loop.build_parser().parse_args(
        [
            "--snapshot-dir",
            str(snapshot_dir),
            "--date",
            "2026-05-29",
        ]
    )

    plan = loop.build_loop_plan(args)

    assert plan["current_corpus"]["status"] == "NO_READY_PERSISTED_PREJUMP_SNAPSHOTS_FOR_DATE"
    assert plan["current_corpus"]["ready_persisted_prediction_snapshot_count_for_date"] == 0
    assert plan["current_corpus"]["result_contaminated_snapshot_rejection_count"] == 1
    assert "actual_win" in plan["current_corpus"][
        "result_contaminated_snapshot_rejection_examples"
    ][0]["reason"]


def test_prejump_loop_default_date_uses_melbourne_racing_date(monkeypatch):
    fixed_now = datetime(2026, 5, 30, 0, 30, tzinfo=ZoneInfo("Australia/Melbourne"))
    monkeypatch.setattr(loop, "melbourne_now", lambda: fixed_now)
    args = loop.build_parser().parse_args([])

    plan = loop.build_loop_plan(args)

    assert plan["current_corpus"]["target_date"] == "2026-05-30"


def test_prejump_loop_reports_protected_resource_counters(tmp_path, monkeypatch):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    snapshot_dir = tmp_path / "snapshots"
    date_dir = snapshot_dir / "2026-05-29" / "TEST"
    date_dir.mkdir(parents=True)
    (snapshot_dir / "manifest.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    (date_dir / "race-1.json").write_text("{}", encoding="utf-8")
    (date_dir / "race-2.json").write_text("{}", encoding="utf-8")
    args = loop.build_parser().parse_args(
        [
            "--snapshot-dir",
            str(snapshot_dir),
            "--date",
            "2026-05-29",
        ]
    )

    plan = loop.build_loop_plan(args)
    counters = plan["protected_resource_counters"]

    assert counters["schema_version"] == "protected_resource_counters_v1"
    assert counters["snapshot_dir"] == str(snapshot_dir)
    assert counters["target_date"] == "2026-05-29"
    assert counters["manifest_path"] == str(snapshot_dir / "manifest.jsonl")
    assert counters["manifest_line_count"] == 2
    assert counters["target_date_snapshot_json_count"] == 2
    assert plan["current_corpus"]["status"] == (
        "NO_READY_PERSISTED_PREJUMP_SNAPSHOTS_FOR_DATE"
    )


def test_execute_ready_rechecks_current_corpus_after_steps(
    tmp_path,
    monkeypatch,
    capsys,
):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    snapshot_dir = tmp_path / "snapshots"
    run_dir = tmp_path / "run"
    output = run_dir / "loop.json"

    def fake_execute_ready_steps(plan):
        date_dir = snapshot_dir / "2026-05-29" / "TEST"
        date_dir.mkdir(parents=True)
        (date_dir / "race-1.json").write_text(
            json.dumps(
                {
                    "schema_version": "prediction_snapshot_v1",
                    "race_id": "Race 1 - TEST - 2026-05-29",
                    "is_pre_jump_snapshot": True,
                    "snapshot_state": "pre_jump_feature_freeze",
                    "snapshot_readiness": {"status": "READY"},
                    "predictions": READY_PREDICTIONS,
                }
            ),
            encoding="utf-8",
        )
        return [{"name": "approved_persist_ready_subset", "returncode": 0}]

    monkeypatch.setattr(loop, "execute_ready_steps", fake_execute_ready_steps)

    assert (
        loop.main(
            [
                "--snapshot-dir",
                str(snapshot_dir),
                "--run-dir",
                str(run_dir),
                "--date",
                "2026-05-29",
                "--execute-ready",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    capsys.readouterr()

    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["current_corpus"]["status"] == (
        "NO_READY_PERSISTED_PREJUMP_SNAPSHOTS_FOR_DATE"
    )
    assert payload["post_execution_current_corpus"]["status"] == (
        "READY_PERSISTED_PREJUMP_SNAPSHOTS_PRESENT"
    )
    assert (
        payload["post_execution_current_corpus"][
            "ready_persisted_prediction_snapshot_count_for_date"
        ]
        == 1
    )


def test_execute_ready_rechecks_protected_resource_counters_after_steps(
    tmp_path,
    monkeypatch,
    capsys,
):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    snapshot_dir = tmp_path / "snapshots"
    run_dir = tmp_path / "run"
    output = run_dir / "loop.json"

    def fake_execute_ready_steps(plan):
        date_dir = snapshot_dir / "2026-05-29" / "TEST"
        date_dir.mkdir(parents=True)
        (snapshot_dir / "manifest.jsonl").write_text("{}\n", encoding="utf-8")
        (date_dir / "race-1.json").write_text("{}", encoding="utf-8")
        return [{"name": "approved_persist_ready_subset", "returncode": 0}]

    monkeypatch.setattr(loop, "execute_ready_steps", fake_execute_ready_steps)

    assert (
        loop.main(
            [
                "--snapshot-dir",
                str(snapshot_dir),
                "--run-dir",
                str(run_dir),
                "--date",
                "2026-05-29",
                "--execute-ready",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    capsys.readouterr()

    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["protected_resource_counters"]["manifest_line_count"] is None
    assert payload["protected_resource_counters"][
        "target_date_snapshot_json_count"
    ] == 0
    assert payload["post_execution_protected_resource_counters"][
        "manifest_line_count"
    ] == 1
    assert payload["post_execution_protected_resource_counters"][
        "target_date_snapshot_json_count"
    ] == 1
    assert payload["post_execution_protected_resource_delta"]["status"] == (
        "UNAPPROVED_PROTECTED_RESOURCE_CHANGE"
    )
    assert payload["post_execution_protected_resource_delta"]["changed"] is True
    assert payload["post_execution_protected_resource_delta"][
        "live_persist_approved"
    ] is False
    assert payload["post_execution_protected_resource_delta"][
        "target_date_snapshot_json_count_delta"
    ] == 1


def test_execute_ready_reports_approved_protected_resource_delta(
    tmp_path,
    monkeypatch,
    capsys,
):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    snapshot_dir = tmp_path / "snapshots"
    run_dir = tmp_path / "run"
    output = run_dir / "loop.json"

    def fake_execute_ready_steps(plan):
        date_dir = snapshot_dir / "2026-05-29" / "TEST"
        date_dir.mkdir(parents=True)
        (snapshot_dir / "manifest.jsonl").write_text("{}\n", encoding="utf-8")
        (date_dir / "race-1.json").write_text(
            json.dumps(
                {
                    "schema_version": "prediction_snapshot_v1",
                    "race_id": "Race 1 - TEST - 2026-05-29",
                    "is_pre_jump_snapshot": True,
                    "snapshot_state": "pre_jump_feature_freeze",
                    "snapshot_readiness": {"status": "READY"},
                    "predictions": READY_PREDICTIONS,
                }
            ),
            encoding="utf-8",
        )
        return [{"name": "approved_persist_ready_subset", "returncode": 0}]

    monkeypatch.setattr(loop, "execute_ready_steps", fake_execute_ready_steps)

    assert (
        loop.main(
            [
                "--snapshot-dir",
                str(snapshot_dir),
                "--run-dir",
                str(run_dir),
                "--date",
                "2026-05-29",
                "--approve-live-persist",
                "--execute-ready",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    capsys.readouterr()

    payload = json.loads(output.read_text(encoding="utf-8"))
    delta = payload["post_execution_protected_resource_delta"]

    assert delta["status"] == "CHANGED_AFTER_APPROVED_PERSIST"
    assert delta["reason"] is None
    assert delta["changed"] is True
    assert delta["live_persist_approved"] is True
    assert delta["persist_step_succeeded"] is True
    assert delta["manifest_line_count_delta"] is None
    assert delta["target_date_snapshot_json_count_delta"] == 1


def test_execute_ready_rechecks_persist_readiness_after_dry_run_report(
    tmp_path,
    monkeypatch,
    capsys,
):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    run_dir = tmp_path / "run"
    output = run_dir / "loop.json"

    def fake_execute_ready_steps(plan):
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "dry_run_capture_report.json").write_text(
            json.dumps(_clean_dry_run_capture_report()),
            encoding="utf-8",
        )
        return [{"name": "dry_run_prejump_capture", "returncode": 0}]

    monkeypatch.setattr(loop, "execute_ready_steps", fake_execute_ready_steps)

    assert (
        loop.main(
            [
                "--run-dir",
                str(run_dir),
                "--date",
                "2026-05-29",
                "--execute-ready",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    capsys.readouterr()

    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["persist_readiness_gate"]["status"] == "DATA_MISSING"
    assert payload["post_execution_persist_readiness_gate"]["status"] == "READY"
    assert payload["post_execution_persist_readiness_gate"]["ready_count"] == 1
    assert payload["post_execution_persist_readiness_gate"][
        "clean_for_ready_subset_persist"
    ] is True
    assert payload["post_execution_persist_approval_packet"]["status"] == (
        "AWAITING_EXPLICIT_APPROVAL_READY_SUBSET"
    )
    post_steps = {
        step["name"]: step
        for step in payload["post_execution_steps"]
    }
    assert post_steps["approved_persist_ready_subset"]["status"] == (
        "APPROVAL_REQUIRED"
    )
    assert post_steps["approved_persist_ready_subset"]["reason"] == (
        "snapshot persistence is blocked until approved"
    )
    assert post_steps["opt_in_live_odds_capture"]["status"] == (
        "APPROVAL_REQUIRED"
    )
    assert post_steps["opt_in_live_odds_capture"]["reason"] == (
        "live odds capture is blocked until approved"
    )
    assert payload["post_execution_persist_approval_packet"][
        "can_execute_persist_now"
    ] is False
    assert payload["post_execution_persist_approval_packet"]["hard_stops"] == []
    assert payload["post_execution_persist_approval_packet"]["ready_count"] == 1
    assert payload["post_execution_live_odds_approval_packet"]["status"] == (
        "AWAITING_EXPLICIT_APPROVAL_READY_FOR_LIVE_ODDS"
    )
    assert payload["post_execution_live_odds_approval_packet"][
        "can_capture_live_odds_now"
    ] is False
    assert payload["post_execution_live_odds_approval_packet"]["hard_stops"] == []
    assert payload["post_execution_live_odds_approval_packet"][
        "current_ev_readiness_counts"
    ] == {"EV_NOT_READY": 1}
    assert payload["prediction_preview_report"]["status"] == "DATA_MISSING"
    assert payload["post_execution_prediction_preview_report"]["status"] == "READY"
    assert payload["post_execution_prediction_preview_report"][
        "preview_runner_count"
    ] == 1
    assert payload["latest_prediction_preview_report_phase"] == "post_execution"
    assert payload["latest_prediction_preview_report"] == payload[
        "post_execution_prediction_preview_report"
    ]
    assert payload["post_execution_protected_resource_delta"]["status"] == (
        "UNCHANGED_NO_APPROVAL"
    )
    assert payload["post_execution_protected_resource_delta"]["changed"] is False
    assert payload["post_execution_milestone_completion_audit"]["overall_status"] == (
        "INCOMPLETE"
    )
    assert payload["post_execution_operator_next_action"]["next_step_status"] == (
        "APPROVAL_REQUIRED_FOR_READY_PERSIST_SUBSET"
    )
    assert payload["post_execution_operator_next_action"][
        "full_objective_complete"
    ] is False
    assert payload["post_execution_operator_next_action"][
        "completed_milestone_count"
    ] == 5
    assert payload["post_execution_operator_next_action"][
        "incomplete_milestone_count"
    ] == 5
    assert [
        item["milestone"]
        for item in payload["post_execution_operator_next_action"][
            "incomplete_milestones"
        ]
    ] == [5, 7, 8, 9, 10]
    assert payload["post_execution_operator_next_action"]["required_gate"] == (
        "APPROVE_LIVE_PERSIST"
    )
    assert payload["post_execution_operator_next_action"]["approval_required"] is True
    assert payload["post_execution_operator_next_action"]["ready_count"] == 1
    assert payload["post_execution_operator_next_action"][
        "safe_no_approval_persist_packet_refresh_sequence_status"
    ] == "NOT_REQUIRED"
    assert payload["post_execution_operator_next_action"][
        "safe_no_approval_persist_packet_refresh_sequence"
    ] == []
    assert payload["post_execution_operator_next_action"][
        "persist_dry_run_fresh_for_plan"
    ] is True
    assert payload["post_execution_operator_next_action"][
        "persist_approval_window_status"
    ] == "OPEN_AWAITING_APPROVAL"
    assert payload["post_execution_operator_next_action"][
        "persist_approval_window_urgency"
    ] == "OPEN"
    assert payload["post_execution_operator_next_action"][
        "dry_run_report_expires_at_local"
    ]
    assert payload["post_execution_operator_next_action"][
        "dry_run_report_expires_at_local_timezone"
    ] == loop.LOCAL_OPERATOR_TIMEZONE
    assert payload["post_execution_operator_next_action"][
        "persist_approval_command_template_status"
    ] == "READY_FOR_EXPLICIT_APPROVAL"
    assert payload["post_execution_operator_next_action"][
        "live_odds_next_step_status"
    ] == "APPROVAL_REQUIRED_FOR_LIVE_ODDS_CAPTURE"
    assert payload["post_execution_operator_next_action"][
        "live_odds_current_ev_readiness_counts"
    ] == {"EV_NOT_READY": 1}
    assert "--approve-live-persist" in payload["post_execution_operator_next_action"][
        "command_template"
    ]


def test_execute_ready_rechecks_result_label_packet_after_result_dry_run(
    tmp_path,
    monkeypatch,
    capsys,
):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    db_path = tmp_path / "labels.sqlite"
    upcoming_dir = tmp_path / "upcoming"
    upcoming_dir.mkdir()
    snapshot_dir = tmp_path / "snapshots"
    date_dir = snapshot_dir / "2026-05-29" / "TEST"
    date_dir.mkdir(parents=True)
    (date_dir / "race-1.json").write_text(
        json.dumps(
            {
                "schema_version": "prediction_snapshot_v1",
                "race_id": "Race 1 - TEST - 2026-05-29",
                "is_pre_jump_snapshot": True,
                "snapshot_state": "pre_jump_feature_freeze",
                "snapshot_readiness": {"status": "READY"},
                "predictions": READY_PREDICTIONS,
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    output = run_dir / "loop.json"
    scope = {
        "db_path": str(db_path.resolve()),
        "date": "2026-05-29",
        "upcoming_dir": str(upcoming_dir.resolve()),
        "snapshot_dir": str(snapshot_dir.resolve()),
        "race_ids": [],
        "require_ready_snapshot": True,
    }

    def fake_execute_ready_steps(plan):
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "result_ingest_dry_run_report.json").write_text(
            json.dumps(_clean_result_dry_run_report(scope)),
            encoding="utf-8",
        )
        (run_dir / "label_write_readiness_validation.json").write_text(
            json.dumps(_clean_label_write_readiness_report(scope)),
            encoding="utf-8",
        )
        return [{"name": "official_result_ingest_dry_run", "returncode": 0}]

    monkeypatch.setattr(loop, "execute_ready_steps", fake_execute_ready_steps)

    assert (
        loop.main(
            [
                "--db",
                str(db_path),
                "--upcoming-dir",
                str(upcoming_dir),
                "--snapshot-dir",
                str(snapshot_dir),
                "--run-dir",
                str(run_dir),
                "--date",
                "2026-05-29",
                "--execute-ready",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    capsys.readouterr()

    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["result_label_approval_packet"]["status"] == "NOT_READY"
    assert payload["result_label_approval_packet"]["hard_stops"] == [
        "result_dry_run_report_missing"
    ]
    assert payload["operator_next_action"]["next_step_status"] == (
        "RUN_OR_REFRESH_OFFICIAL_RESULT_DRY_RUN"
    )
    assert payload["operator_next_action"]["required_gate"] is None
    assert payload["operator_next_action"]["approval_required"] is False
    assert payload["operator_next_action"]["result_dry_run_fresh_for_plan"] is False
    assert payload["operator_next_action"]["result_label_approval_window_status"] == (
        "DRY_RUN_REQUIRED"
    )
    assert payload["operator_next_action"][
        "result_label_approval_command_template_status"
    ] == "BLOCKED_BY_HARD_STOPS"
    assert "--dry-run" in payload["operator_next_action"]["command_template"]
    assert payload["post_execution_result_dry_run_report_gate"]["status"] == "READY"
    assert payload["post_execution_result_label_approval_packet"]["status"] == (
        "AWAITING_EXPLICIT_APPROVAL_READY_FOR_LABEL_WRITE"
    )
    post_steps = {
        step["name"]: step
        for step in payload["post_execution_steps"]
    }
    assert post_steps["approved_official_label_write"]["status"] == (
        "APPROVAL_REQUIRED"
    )
    assert post_steps["approved_official_label_write"]["reason"] == (
        "official result label writes are blocked until approved"
    )
    assert payload["post_execution_result_label_approval_packet"][
        "can_write_labels_now"
    ] is False
    assert payload["post_execution_result_label_approval_packet"]["hard_stops"] == []
    assert payload["post_execution_label_write_readiness_validation_gate"][
        "status"
    ] == "READY"
    assert post_steps["label_write_preflight_packet"]["status"] == (
        "READY_FOR_OPERATOR_RUN_AFTER_LOOP_PLAN_WRITE"
    )
    assert payload["post_execution_operator_next_action"]["next_step_status"] == (
        "RUN_LABEL_WRITE_PREFLIGHT_PACKET"
    )
    assert payload["post_execution_operator_next_action"]["required_gate"] is None
    assert payload["post_execution_operator_next_action"]["approval_required"] is False
    assert "scripts/build_label_write_preflight_packet.py" in payload[
        "post_execution_operator_next_action"
    ]["command_template"]
    assert (
        payload["post_execution_operator_next_action"][
            "result_label_approval_window_status"
        ]
        == "OPEN_AWAITING_APPROVAL"
    )
    assert payload["post_execution_operator_next_action"][
        "result_dry_run_fresh_for_plan"
    ] is True
    assert payload["post_execution_operator_next_action"][
        "result_label_approval_command_template_status"
    ] == "READY_FOR_EXPLICIT_APPROVAL"
    assert "--output" in payload["post_execution_operator_next_action"][
        "command_template"
    ]


def test_prejump_loop_threads_result_race_ids_into_readiness_validation(
    tmp_path,
    monkeypatch,
):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    db_path = tmp_path / "labels.sqlite"
    upcoming_dir = tmp_path / "upcoming"
    upcoming_dir.mkdir()
    snapshot_dir = tmp_path / "snapshots"
    date_dir = snapshot_dir / "2026-05-29" / "TEST"
    date_dir.mkdir(parents=True)
    race_id = "Race 1 - TEST - 2026-05-29"
    (date_dir / "race-1.json").write_text(
        json.dumps(
            {
                "schema_version": "prediction_snapshot_v1",
                "race_id": race_id,
                "is_pre_jump_snapshot": True,
                "snapshot_state": "pre_jump_feature_freeze",
                "snapshot_readiness": {"status": "READY"},
                "predictions": READY_PREDICTIONS,
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    scope = {
        "db_path": str(db_path.resolve()),
        "date": "2026-05-29",
        "upcoming_dir": str(upcoming_dir.resolve()),
        "snapshot_dir": str(snapshot_dir.resolve()),
        "race_ids": [race_id],
        "require_ready_snapshot": True,
    }
    (run_dir / "result_ingest_dry_run_report.json").write_text(
        json.dumps(_clean_result_dry_run_report(scope)),
        encoding="utf-8",
    )

    plan = loop.build_loop_plan(
        loop.build_parser().parse_args(
            [
                "--db",
                str(db_path),
                "--upcoming-dir",
                str(upcoming_dir),
                "--snapshot-dir",
                str(snapshot_dir),
                "--run-dir",
                str(run_dir),
                "--date",
                "2026-05-29",
                "--result-race-id",
                race_id,
            ]
        )
    )

    steps = {step["name"]: step for step in plan["steps"]}
    assert plan["result_dry_run_report_gate"]["status"] == "READY"
    assert plan["result_dry_run_report_gate"]["expected_scope"]["race_ids"] == [
        race_id
    ]
    assert steps["result_label_write_readiness_validation"]["status"] == (
        "READY_TO_RUN"
    )
    assert "--validate-label-write-readiness" in steps[
        "result_label_write_readiness_validation"
    ]["command"]
    assert steps["result_label_write_readiness_validation"]["command"].count(
        "--race-id"
    ) == 1
    assert race_id in steps["result_label_write_readiness_validation"]["command"]
    assert steps["approved_official_label_write"]["status"] == (
        "WAITING_FOR_LABEL_WRITE_READINESS_VALIDATION"
    )
    assert plan["operator_next_action"]["next_step_status"] == (
        "RUN_LABEL_WRITE_READINESS_VALIDATION"
    )
    assert plan["operator_next_action"][
        "label_write_readiness_validation_command_template"
    ] == steps["result_label_write_readiness_validation"]["command"]
    assert "--result-race-id" in plan["result_label_approval_packet"][
        "approved_same_run_execute_ready_command_template"
    ]


def test_prejump_loop_reports_clean_dry_run_persist_readiness(tmp_path, monkeypatch):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "dry_run_capture_report.json").write_text(
        json.dumps(
            {
                "status": "SUCCESS",
                "dry_run": True,
                "persist_requested": False,
                "persist_approved": False,
                "candidate_files": 2,
                "capture_count": 2,
                "metadata_missing_count": 0,
                "metadata_unsafe_count": 0,
                "metadata_mismatch_count": 0,
                "lifecycle_counts": {"upcoming_not_jumped": 2},
                "final_runner_set_counts": {"verified": 2},
                "target_metadata_counts": {"verified": 2},
                "ev_readiness_counts": {"EV_NOT_READY": 2},
                "captures": [
                    {
                        "race_id": "Race 1 - TEST - 2026-05-29",
                        "snapshot_readiness": {"status": "READY"},
                    },
                    {
                        "race_id": "Race 2 - TEST - 2026-05-29",
                        "snapshot_readiness": {"status": "READY"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    args = loop.build_parser().parse_args(
        [
            "--run-dir",
            str(run_dir),
            "--date",
            "2026-05-29",
        ]
    )

    plan = loop.build_loop_plan(args)
    by_name = {step["name"]: step for step in plan["steps"]}

    assert plan["persist_readiness_gate"]["status"] == "READY"
    assert plan["persist_readiness_gate"]["clean_for_ready_subset_persist"] is True
    assert plan["persist_readiness_gate"]["ready_count"] == 2
    assert plan["persist_readiness_gate"]["not_ready_count"] == 0
    assert plan["persist_readiness_gate"]["ev_readiness_counts"] == {"EV_NOT_READY": 2}
    assert plan["persist_readiness_gate"]["fresh_for_plan"] is True
    assert plan["persist_approval_packet"]["status"] == (
        "AWAITING_EXPLICIT_APPROVAL_READY_SUBSET"
    )
    assert by_name["approved_persist_ready_subset"]["status"] == "APPROVAL_REQUIRED"
    assert by_name["approved_persist_ready_subset"]["reason"] == (
        "snapshot persistence is blocked until approved"
    )
    assert plan["persist_approval_packet"]["approval_required"] is True
    assert plan["persist_approval_packet"]["can_execute_persist_now"] is False
    assert plan["persist_approval_packet"]["ready_count"] == 2
    assert plan["persist_approval_packet"]["not_ready_count"] == 0
    assert plan["persist_approval_packet"]["hard_stops"] == []
    assert plan["persist_approval_packet"]["dry_run_report_expires_at_utc"]
    assert plan["persist_approval_packet"]["dry_run_report_expires_at_local"]
    assert plan["persist_approval_packet"][
        "dry_run_report_expires_at_local_timezone"
    ] == loop.LOCAL_OPERATOR_TIMEZONE
    assert (
        0
        < plan["persist_approval_packet"]["dry_run_report_seconds_until_expiry"]
        <= loop.PERSIST_DRY_RUN_REPORT_MAX_AGE_SECONDS
    )
    assert plan["persist_approval_packet"][
        "approval_must_arrive_before_report_expiry"
    ] is True
    assert plan["persist_approval_packet"]["rerun_required_after_expiry"] is True
    assert "--approve-live-persist" not in plan["persist_approval_packet"][
        "planned_persist_command"
    ]
    assert "--approve-live-persist" in plan["persist_approval_packet"][
        "approved_persist_command_template"
    ]
    assert plan["persist_approval_packet"][
        "approval_command_requires_explicit_operator_confirmation"
    ] is True
    assert plan["persist_approval_packet"]["expected_protected_delta_upper_bounds"] == {
        "manifest_line_count_delta_max": 2,
        "target_date_snapshot_json_count_delta_max": 2,
    }
    assert plan["live_odds_approval_packet"]["status"] == (
        "AWAITING_EXPLICIT_APPROVAL_READY_FOR_LIVE_ODDS"
    )
    assert by_name["opt_in_live_odds_capture"]["status"] == "APPROVAL_REQUIRED"
    assert by_name["opt_in_live_odds_capture"]["reason"] == (
        "live odds capture is blocked until approved"
    )
    assert plan["live_odds_approval_packet"]["approval_required"] is True
    assert plan["live_odds_approval_packet"]["can_capture_live_odds_now"] is False
    assert plan["live_odds_approval_packet"]["ready_count"] == 2
    assert plan["live_odds_approval_packet"]["current_ev_readiness_counts"] == {
        "EV_NOT_READY": 2
    }
    assert plan["live_odds_approval_packet"]["dry_run_report_expires_at_utc"]
    assert plan["live_odds_approval_packet"]["dry_run_report_expires_at_local"]
    assert plan["live_odds_approval_packet"][
        "dry_run_report_expires_at_local_timezone"
    ] == loop.LOCAL_OPERATOR_TIMEZONE
    assert (
        0
        < plan["live_odds_approval_packet"]["dry_run_report_seconds_until_expiry"]
        <= loop.PERSIST_DRY_RUN_REPORT_MAX_AGE_SECONDS
    )
    assert plan["live_odds_approval_packet"][
        "approval_must_arrive_before_report_expiry"
    ] is True
    assert plan["live_odds_approval_packet"]["rerun_required_after_expiry"] is True
    assert "--approve-live-odds-capture" not in plan[
        "live_odds_approval_packet"
    ]["planned_odds_command"]
    assert "--approve-live-odds-capture" in plan["live_odds_approval_packet"][
        "approved_odds_command_template"
    ]
    assert plan["live_odds_approval_packet"][
        "approval_command_requires_explicit_operator_confirmation"
    ] is True
    assert plan["live_odds_approval_packet"]["ev_policy"][
        "ev_must_remain_null_unless_all_requirements_pass"
    ] is True


def test_authoritative_persist_capture_report_ev_summary_uses_explicit_ready_counts(
    tmp_path,
):
    report_path = tmp_path / "persist_capture_report.json"
    report_path.write_text(
        json.dumps(
            _june2_authoritative_persist_report(
                include_capture_ev_status=False,
                odds_exclusion_counts={"late_scratched_without_price": 2},
            )
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("existing-manifest-line\n", encoding="utf-8")
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text('{"existing": true}\n', encoding="utf-8")
    before_manifest = manifest_path.read_text(encoding="utf-8")
    before_snapshot = snapshot_path.read_text(encoding="utf-8")

    summary = loop._authoritative_capture_report_ev_summary(report_path)

    assert summary["ev_summary_source"] == "authoritative_persist_capture_report"
    assert summary["authoritative_capture_report_path"].endswith(
        "persist_capture_report.json"
    )
    assert summary["ev_readiness_counts"] == {"EV_READY": 5}
    assert summary["ev_ready_count"] == 5
    assert summary["ev_not_ready_count"] == 0
    assert summary["priced_ev_runner_count"] == 32
    assert summary["odds_exclusion_counts"] == {"late_scratched_without_price": 2}
    assert summary["odds_exclusion_count"] == 2
    assert summary["ev_summary_consistency_check"] == "SOURCE_COUNTS_USED"
    assert summary["ev_summary_failure_reason"] is None
    assert manifest_path.read_text(encoding="utf-8") == before_manifest
    assert snapshot_path.read_text(encoding="utf-8") == before_snapshot


def test_dry_run_capture_report_is_not_authoritative_persisted_ev_evidence(
    tmp_path,
):
    report_path = tmp_path / "dry_run_capture_report.json"
    report = _june2_authoritative_persist_report()
    report.update(
        {
            "dry_run": True,
            "persist_requested": False,
            "persist_approved": False,
        }
    )
    report_path.write_text(json.dumps(report), encoding="utf-8")

    summary = loop._authoritative_capture_report_ev_summary(report_path)

    assert summary["ev_summary_source"] == "NOT_AUTHORITATIVE_CAPTURE_REPORT"
    assert summary["ev_readiness_counts"] == {}
    assert summary["ev_ready_count"] == 0
    assert summary["ev_not_ready_count"] == 0
    assert summary["priced_ev_runner_count"] == 0
    assert summary["ev_summary_consistency_check"] == "REJECTED_NON_PERSISTED_REPORT"
    assert summary["ev_summary_failure_reason"] == (
        "capture_report_is_not_approved_persist_report"
    )


def test_authoritative_persist_capture_report_fallback_aggregates_capture_fields(
    tmp_path,
):
    report_path = tmp_path / "persist_capture_report.json"
    report = _june2_authoritative_persist_report()
    report.pop("ev_readiness_counts")
    report.pop("priced_ev_runner_count")
    report.pop("odds_exclusion_counts")
    report["captures"][0]["odds_exclusion_counts"] = {"late_scratched": 1}
    report["captures"][2]["odds_exclusion_counts"] = {"missing_live_odds": 2}
    report_path.write_text(json.dumps(report), encoding="utf-8")

    summary = loop._authoritative_capture_report_ev_summary(report_path)

    assert summary["ev_summary_source"] == "authoritative_persist_capture_report"
    assert summary["ev_readiness_counts"] == {"EV_READY": 5}
    assert summary["ev_ready_count"] == 5
    assert summary["ev_not_ready_count"] == 0
    assert summary["priced_ev_runner_count"] == 32
    assert summary["odds_exclusion_counts"] == {
        "late_scratched": 1,
        "missing_live_odds": 2,
    }
    assert summary["odds_exclusion_count"] == 3
    assert summary["ev_summary_consistency_check"] == "CAPTURE_COUNTS_USED"
    assert summary["ev_summary_failure_reason"] is None


def test_live_odds_packet_keeps_non_authoritative_summary_diagnostics():
    persist_gate = {
        "clean_for_ready_subset_persist": True,
        "fresh_for_plan": True,
        "ready_count": 5,
        "ready_race_ids": JUNE2_SECOND_BATCH_RACE_IDS,
        "ev_readiness_counts": {"EV_NOT_READY": 5},
        "path": "artifacts/run/dry_run_capture_report.json",
    }
    ev_summary = {
        "ev_summary_source": "NOT_AUTHORITATIVE_CAPTURE_REPORT",
        "ev_readiness_counts": {},
        "ev_ready_count": 0,
        "ev_not_ready_count": 0,
        "priced_ev_runner_count": 0,
        "odds_exclusion_counts": {},
        "odds_exclusion_count": 0,
        "authoritative_capture_report_path": (
            "artifacts/run/persist_capture_report.json"
        ),
        "ev_summary_consistency_check": "REJECTED_NON_PERSISTED_REPORT",
        "ev_summary_failure_reason": (
            "capture_report_is_not_approved_persist_report"
        ),
    }

    packet = loop._live_odds_approval_packet(
        persist_readiness_gate=persist_gate,
        approvals={"live_odds_capture": False},
        approval_details={"live_odds_capture": {"sources": []}},
        odds_command=["capture-odds"],
        odds_report_path=loop.ROOT / "artifacts/run/live_odds_report.json",
        ev_summary=ev_summary,
    )

    assert packet["current_ev_readiness_counts"] == {}
    assert packet["ev_summary_source"] == "NOT_AUTHORITATIVE_CAPTURE_REPORT"
    assert packet["ev_ready_count"] == 0
    assert packet["ev_not_ready_count"] == 0
    assert packet["priced_ev_runner_count"] == 0
    assert packet["authoritative_capture_report_path"].endswith(
        "persist_capture_report.json"
    )
    assert packet["ev_summary_consistency_check"] == "REJECTED_NON_PERSISTED_REPORT"
    assert packet["ev_summary_failure_reason"] == (
        "capture_report_is_not_approved_persist_report"
    )


def test_execute_ready_operator_ev_summary_prefers_authoritative_persist_report(
    tmp_path,
    monkeypatch,
    capsys,
):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    run_dir = tmp_path / "run"
    output = run_dir / "loop.json"

    def fake_execute_ready_steps(plan):
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "dry_run_capture_report.json").write_text(
            json.dumps(_june2_dry_run_not_ready_report()),
            encoding="utf-8",
        )
        (run_dir / "persist_capture_report.json").write_text(
            json.dumps(_june2_authoritative_persist_report()),
            encoding="utf-8",
        )
        return [
            {"name": "dry_run_prejump_capture", "returncode": 0},
            {"name": "approved_persist_ready_subset", "returncode": 0},
        ]

    monkeypatch.setattr(loop, "execute_ready_steps", fake_execute_ready_steps)

    assert (
        loop.main(
            [
                "--run-dir",
                str(run_dir),
                "--date",
                "2026-06-02",
                "--approve-live-persist",
                "--approve-live-odds-capture",
                "--execute-ready",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    capsys.readouterr()

    payload = json.loads(output.read_text(encoding="utf-8"))
    summary = payload["post_execution_ev_readiness_summary"]
    live_odds_packet = payload["post_execution_live_odds_approval_packet"]
    operator_action = payload["post_execution_operator_next_action"]

    assert payload["post_execution_persist_readiness_gate"]["ev_readiness_counts"] == {
        "EV_NOT_READY": 5
    }
    assert summary["ev_summary_source"] == "authoritative_persist_capture_report"
    assert summary["ev_ready_count"] == 5
    assert summary["ev_not_ready_count"] == 0
    assert summary["priced_ev_runner_count"] == 32
    assert summary["odds_exclusion_count"] == 0
    assert live_odds_packet["current_ev_readiness_counts"] == {"EV_READY": 5}
    assert live_odds_packet["ev_summary_source"] == (
        "authoritative_persist_capture_report"
    )
    assert live_odds_packet["ev_ready_count"] == 5
    assert live_odds_packet["ev_not_ready_count"] == 0
    assert live_odds_packet["priced_ev_runner_count"] == 32
    assert live_odds_packet["authoritative_capture_report_path"].endswith(
        "persist_capture_report.json"
    )
    assert live_odds_packet["ev_summary_consistency_check"] == "MATCH"
    assert live_odds_packet["ev_summary_failure_reason"] is None
    assert operator_action["live_odds_current_ev_readiness_counts"] == {
        "EV_READY": 5
    }
    assert operator_action["ev_summary_source"] == (
        "authoritative_persist_capture_report"
    )
    assert operator_action["ev_ready_count"] == 5
    assert operator_action["ev_not_ready_count"] == 0
    assert operator_action["priced_ev_runner_count"] == 32
    assert operator_action["odds_exclusion_count"] == 0
    assert operator_action["authoritative_capture_report_path"].endswith(
        "persist_capture_report.json"
    )


def test_prejump_loop_reports_same_run_approval_command_templates(
    tmp_path,
    monkeypatch,
):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    calibration_design = tmp_path / "calibration_layer_design.json"
    calibration_design.write_text(
        json.dumps(
            {
                "schema_version": "calibration_layer_design_v1",
                "status": "READY_FOR_OPERATOR_DESIGN_REVIEW",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "dry_run_capture_report.json").write_text(
        json.dumps(_clean_dry_run_capture_report()),
        encoding="utf-8",
    )
    args = loop.build_parser().parse_args(
        [
            "--run-dir",
            str(run_dir),
            "--date",
            "2026-05-29",
            "--min-minutes",
            "20",
            "--max-minutes",
            "160",
            "--limit",
            "16",
            "--report-only-calibration-design",
            str(calibration_design),
        ]
    )

    plan = loop.build_loop_plan(args)
    by_name = {step["name"]: step for step in plan["steps"]}
    persist_packet = plan["persist_approval_packet"]
    odds_packet = plan["live_odds_approval_packet"]
    dry_run_command = by_name["dry_run_prejump_capture"]["command"]
    persist_capture_command = by_name["approved_persist_ready_subset"]["command"]
    odds_capture_command = by_name["opt_in_live_odds_capture"]["command"]

    assert dry_run_command[dry_run_command.index("--limit") + 1] == "16"
    assert persist_capture_command[persist_capture_command.index("--limit") + 1] == "16"
    assert odds_capture_command[odds_capture_command.index("--limit") + 1] == "16"
    for command in (
        dry_run_command,
        persist_capture_command,
        odds_capture_command,
    ):
        assert "--report-only-calibration-design" in command
        assert command[command.index("--report-only-calibration-design") + 1] == str(
            calibration_design
        )

    persist_command = persist_packet[
        "approved_same_run_execute_ready_command_template"
    ]
    assert "scripts/prejump_prediction_loop.py" in persist_command
    assert "--execute-ready" in persist_command
    assert "--approve-live-persist" in persist_command
    assert "--approve-live-odds-capture" not in persist_command
    assert "--run-dir" in persist_command
    assert "--date" in persist_command
    assert "--upcoming-dir" in persist_command
    assert "--snapshot-dir" in persist_command
    assert "--db" in persist_command
    assert "--report-only-calibration-design" in persist_command
    assert persist_command[
        persist_command.index("--report-only-calibration-design") + 1
    ] == str(calibration_design)
    assert "--output" in persist_command
    assert persist_command[persist_command.index("--output") + 1].endswith(
        "loop_plan_execute_approved_persist.json"
    )
    assert persist_packet["same_run_execute_ready_command_template_status"] == (
        "READY_FOR_EXPLICIT_APPROVAL_AND_FRESH_RECHECK"
    )
    assert persist_packet[
        "same_run_execute_ready_command_requires_explicit_operator_confirmation"
    ] is True
    assert persist_packet["same_run_execute_ready_rechecks"] == [
        "fresh_refresh_current_window",
        "validate_current_upcoming_contract",
        "dry_run_prejump_capture",
        "persist_readiness_gate",
        "protected_resource_delta",
    ]
    operator_command = plan["operator_next_action"]["command_template"]
    assert "scripts/prejump_prediction_loop.py" in operator_command
    assert "scripts/capture_prediction_snapshot.py" not in operator_command
    assert "--execute-ready" in operator_command
    assert "--approve-live-persist" in operator_command
    assert "--report-only-calibration-design" in operator_command
    assert plan["operator_next_action"][
        "persist_same_run_execute_ready_command_template_status"
    ] == "READY_FOR_EXPLICIT_APPROVAL_AND_FRESH_RECHECK"
    assert plan["operator_next_action"]["persist_same_run_execute_ready_rechecks"] == [
        "fresh_refresh_current_window",
        "validate_current_upcoming_contract",
        "dry_run_prejump_capture",
        "persist_readiness_gate",
        "protected_resource_delta",
    ]

    odds_command = odds_packet["approved_same_run_execute_ready_command_template"]
    assert "scripts/prejump_prediction_loop.py" in odds_command
    assert "--execute-ready" in odds_command
    assert "--approve-live-odds-capture" in odds_command
    assert "--approve-live-persist" not in odds_command
    assert "--report-only-calibration-design" in odds_command
    assert odds_command[
        odds_command.index("--report-only-calibration-design") + 1
    ] == str(calibration_design)
    assert "--output" in odds_command
    assert odds_command[odds_command.index("--output") + 1].endswith(
        "loop_plan_execute_approved_live_odds.json"
    )
    assert odds_packet["same_run_execute_ready_command_template_status"] == (
        "READY_FOR_EXPLICIT_APPROVAL_AND_FRESH_RECHECK"
    )
    assert odds_packet[
        "same_run_execute_ready_command_requires_explicit_operator_confirmation"
    ] is True
    assert odds_packet["same_run_execute_ready_rechecks"] == [
        "fresh_refresh_current_window",
        "validate_current_upcoming_contract",
        "dry_run_prejump_capture",
        "persist_readiness_gate",
        "live_odds_readiness_gate",
    ]


def test_prejump_loop_persist_approval_packet_ready_with_cli_approval(
    tmp_path,
    monkeypatch,
):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "dry_run_capture_report.json").write_text(
        json.dumps(_clean_dry_run_capture_report()),
        encoding="utf-8",
    )
    args = loop.build_parser().parse_args(
        [
            "--run-dir",
            str(run_dir),
            "--date",
            "2026-05-29",
            "--approve-live-persist",
        ]
    )

    plan = loop.build_loop_plan(args)
    by_name = {step["name"]: step for step in plan["steps"]}
    packet = plan["persist_approval_packet"]

    assert packet["status"] == "APPROVAL_PRESENT_READY_TO_EXECUTE_READY_SUBSET"
    assert packet["can_execute_persist_now"] is True
    assert packet["approval_required"] is False
    assert packet["approval_sources"] == ["cli"]
    assert packet["hard_stops"] == []
    assert packet["ready_count"] == 1
    assert packet["expected_protected_delta_upper_bounds"] == {
        "manifest_line_count_delta_max": 1,
        "target_date_snapshot_json_count_delta_max": 1,
    }
    assert "--approve-live-persist" in packet["planned_persist_command"]
    assert packet["post_execute_delta_report"] == (
        "post_execution_protected_resource_delta"
    )


def test_prejump_loop_live_odds_approval_packet_ready_with_cli_approval(
    tmp_path,
    monkeypatch,
):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "dry_run_capture_report.json").write_text(
        json.dumps(_clean_dry_run_capture_report()),
        encoding="utf-8",
    )
    args = loop.build_parser().parse_args(
        [
            "--run-dir",
            str(run_dir),
            "--date",
            "2026-05-29",
            "--approve-live-odds-capture",
        ]
    )

    plan = loop.build_loop_plan(args)
    packet = plan["live_odds_approval_packet"]

    assert packet["status"] == "APPROVAL_PRESENT_READY_TO_CAPTURE_LIVE_ODDS"
    assert packet["can_capture_live_odds_now"] is True
    assert packet["approval_required"] is False
    assert packet["approval_sources"] == ["cli"]
    assert packet["hard_stops"] == []
    assert packet["ready_count"] == 1
    assert "--approve-live-odds-capture" in packet["planned_odds_command"]
    assert packet["write_scope"] == "append_only_live_odds_rows"
    assert packet["no_snapshot_persist"] is True
    assert packet["no_result_labels"] is True


def test_prejump_loop_reports_stale_prediction_preview_without_freshness_claim(
    tmp_path,
    monkeypatch,
):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    report_path = run_dir / "dry_run_capture_report.json"
    report_path.write_text(
        json.dumps(_clean_dry_run_capture_report()),
        encoding="utf-8",
    )
    stale_time = time.time() - loop.PERSIST_DRY_RUN_REPORT_MAX_AGE_SECONDS - 60
    os.utime(report_path, (stale_time, stale_time))
    args = loop.build_parser().parse_args(
        [
            "--run-dir",
            str(run_dir),
            "--date",
            "2026-05-29",
        ]
    )

    plan = loop.build_loop_plan(args)
    preview_report = plan["prediction_preview_report"]

    assert preview_report["status"] == "STALE_AVAILABLE"
    assert preview_report["fresh_for_plan"] is False
    assert preview_report["reason"] == "dry_run_capture_report_stale"
    assert preview_report["preview_race_count"] == 1
    assert preview_report["preview_runner_count"] == 1
    assert preview_report["races"][0]["race_id"] == "Race 1 - TEST - 2026-05-29"
    assert preview_report["races"][0]["prediction_preview"] == [
        {
            "predicted_rank": 1,
            "box_number": 1,
            "dog_name": "Alpha Runner",
            "win_prob_norm": 1.0,
            "odds_match_status": "no_odds_row",
            "market_odds_win": None,
            "ev_win": None,
            "quality_flags": [],
        }
    ]


def test_prejump_loop_rejects_result_contaminated_prediction_preview(
    tmp_path,
    monkeypatch,
):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    report = _clean_dry_run_capture_report()
    report["captures"][0]["prediction_preview"][0]["actual_win"] = 1
    (run_dir / "dry_run_capture_report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    args = loop.build_parser().parse_args(
        [
            "--run-dir",
            str(run_dir),
            "--date",
            "2026-05-29",
        ]
    )

    plan = loop.build_loop_plan(args)
    preview_report = plan["prediction_preview_report"]

    assert preview_report["status"] == "NOT_READY"
    assert "result_field_leakage_detected" in preview_report["reason"]
    assert preview_report["races"] == []
    assert preview_report["preview_race_count"] == 0
    assert preview_report["preview_runner_count"] == 0
    assert preview_report["result_contaminated_capture_rejection_count"] == 1
    assert "actual_win" in preview_report[
        "result_contaminated_capture_rejection_examples"
    ][0]["reason"]


def test_prediction_preview_scans_for_result_leakage_beyond_display_limit(tmp_path):
    report_path = tmp_path / "dry_run_capture_report.json"
    report = _clean_dry_run_capture_report()
    second_capture = json.loads(json.dumps(report["captures"][0]))
    second_capture["race_id"] = "Race 2 - TEST - 2026-05-29"
    second_capture["prediction_preview"][0]["actual_win"] = 1
    report["captures"].append(second_capture)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    gate = {
        "status": "READY",
        "reason": None,
        "fresh_for_plan": True,
    }

    preview_report = loop._dry_run_prediction_preview_report(
        report_path,
        gate,
        max_races=1,
    )

    assert preview_report["status"] == "NOT_READY"
    assert "result_field_leakage_detected" in preview_report["reason"]
    assert preview_report["races"] == []
    assert preview_report["preview_race_count"] == 0
    assert preview_report["preview_runner_count"] == 0
    assert preview_report["result_contaminated_capture_rejection_count"] == 1


def test_prejump_loop_rejects_stale_dry_run_persist_readiness(tmp_path, monkeypatch):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    report_path = run_dir / "dry_run_capture_report.json"
    report_path.write_text(
        json.dumps(_clean_dry_run_capture_report()),
        encoding="utf-8",
    )
    stale_time = time.time() - loop.PERSIST_DRY_RUN_REPORT_MAX_AGE_SECONDS - 60
    os.utime(report_path, (stale_time, stale_time))
    args = loop.build_parser().parse_args(
        [
            "--run-dir",
            str(run_dir),
            "--date",
            "2026-05-29",
        ]
    )

    plan = loop.build_loop_plan(args)
    by_name = {step["name"]: step for step in plan["steps"]}

    assert plan["persist_readiness_gate"]["status"] == "NOT_READY"
    assert plan["persist_readiness_gate"]["clean_for_ready_subset_persist"] is False
    assert plan["persist_readiness_gate"]["fresh_for_plan"] is False
    assert "dry_run_capture_report_stale" in plan["persist_readiness_gate"]["reason"]
    assert plan["persist_approval_packet"]["status"] == "NOT_READY"
    assert "dry_run_capture_report_not_fresh" in plan["persist_approval_packet"][
        "hard_stops"
    ]
    assert plan["persist_approval_packet"]["approved_persist_command_template"] is None
    assert plan["persist_approval_packet"]["approval_command_template_status"] == (
        "BLOCKED_BY_HARD_STOPS"
    )
    assert by_name["approved_persist_ready_subset"]["status"] == (
        "WAITING_FOR_READY_PERSIST_PACKET"
    )
    assert "dry_run_capture_report_not_fresh" in by_name[
        "approved_persist_ready_subset"
    ]["reason"]
    assert plan["live_odds_approval_packet"]["status"] == "NOT_READY"
    assert "dry_run_capture_report_not_fresh" in plan[
        "live_odds_approval_packet"
    ]["hard_stops"]
    assert plan["live_odds_approval_packet"]["approved_odds_command_template"] is None
    assert by_name["opt_in_live_odds_capture"]["status"] == (
        "WAITING_FOR_READY_ODDS_PACKET"
    )
    assert "dry_run_capture_report_not_fresh" in by_name[
        "opt_in_live_odds_capture"
    ]["reason"]
    assert plan["operator_next_action"]["next_step_status"] == (
        "REFRESH_DRY_RUN_REQUIRED_FOR_PERSIST_PACKET"
    )
    assert plan["operator_next_action"]["command_template"] is None
    assert plan["operator_next_action"][
        "safe_no_approval_persist_packet_refresh_sequence_status"
    ] == "AVAILABLE"
    assert [
        item["name"]
        for item in plan["operator_next_action"][
            "safe_no_approval_persist_packet_refresh_sequence"
        ]
    ] == [
        "fresh_refresh_current_window",
        "validate_current_upcoming_contract",
        "dry_run_prejump_capture",
    ]
    assert plan["operator_next_action"]["persist_dry_run_fresh_for_plan"] is False
    assert plan["operator_next_action"]["persist_approval_window_status"] == (
        "REFRESH_REQUIRED"
    )
    assert plan["operator_next_action"]["persist_approval_window_urgency"] == (
        "REFRESH_REQUIRED"
    )
    assert plan["operator_next_action"][
        "persist_approval_command_template_status"
    ] == "BLOCKED_BY_HARD_STOPS"


def test_prejump_loop_reports_partial_dry_run_persist_readiness(tmp_path, monkeypatch):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "dry_run_capture_report.json").write_text(
        json.dumps(
            {
                "status": "SUCCESS",
                "dry_run": True,
                "persist_requested": False,
                "persist_approved": False,
                "candidate_files": 2,
                "capture_count": 2,
                "metadata_missing_count": 0,
                "metadata_unsafe_count": 0,
                "metadata_mismatch_count": 0,
                "captures": [
                    {
                        "race_id": "Race 1 - TEST - 2026-05-29",
                        "snapshot_readiness": {"status": "READY"},
                    },
                    {
                        "race_id": "Race 2 - TEST - 2026-05-29",
                        "snapshot_readiness": {"status": "NOT_READY"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    args = loop.build_parser().parse_args(
        [
            "--run-dir",
            str(run_dir),
            "--date",
            "2026-05-29",
        ]
    )

    plan = loop.build_loop_plan(args)

    assert plan["persist_readiness_gate"]["status"] == "PARTIAL_READY"
    assert plan["persist_readiness_gate"]["clean_for_ready_subset_persist"] is True
    assert plan["persist_readiness_gate"]["ready_count"] == 1
    assert plan["persist_readiness_gate"]["not_ready_count"] == 1
    assert plan["persist_readiness_gate"]["not_ready_race_ids"] == [
        "Race 2 - TEST - 2026-05-29"
    ]
    assert plan["persist_approval_packet"]["status"] == (
        "AWAITING_EXPLICIT_APPROVAL_READY_SUBSET"
    )
    assert plan["persist_approval_packet"]["ready_count"] == 1
    assert plan["persist_approval_packet"]["not_ready_count"] == 1
    assert plan["persist_approval_packet"]["expected_protected_delta_upper_bounds"] == {
        "manifest_line_count_delta_max": 1,
        "target_date_snapshot_json_count_delta_max": 1,
    }
    assert plan["live_odds_approval_packet"]["status"] == (
        "AWAITING_EXPLICIT_APPROVAL_READY_FOR_LIVE_ODDS"
    )
    assert plan["live_odds_approval_packet"]["ready_count"] == 1


def test_prejump_loop_rejects_persist_readiness_from_non_dry_run_report(
    tmp_path,
    monkeypatch,
):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "dry_run_capture_report.json").write_text(
        json.dumps(
            {
                "status": "SUCCESS",
                "dry_run": False,
                "persist_requested": True,
                "persist_approved": True,
                "candidate_files": 1,
                "metadata_missing_count": 0,
                "metadata_unsafe_count": 0,
                "metadata_mismatch_count": 0,
                "captures": [
                    {
                        "race_id": "Race 1 - TEST - 2026-05-29",
                        "snapshot_readiness": {"status": "READY"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    args = loop.build_parser().parse_args(
        [
            "--run-dir",
            str(run_dir),
            "--date",
            "2026-05-29",
        ]
    )

    plan = loop.build_loop_plan(args)

    assert plan["persist_readiness_gate"]["status"] == "NOT_READY"
    assert plan["persist_readiness_gate"]["clean_for_ready_subset_persist"] is False
    assert "report_is_not_dry_run" in plan["persist_readiness_gate"]["reason"]
    assert "dry_run_precheck_should_not_request_persist" in plan[
        "persist_readiness_gate"
    ]["reason"]


def test_execute_ready_skips_approved_persist_when_readiness_gate_not_clean(
    tmp_path,
    monkeypatch,
):
    dry_run_report = tmp_path / "dry_run_capture_report.json"

    class Completed:
        returncode = 0
        stdout = "dry run ok"
        stderr = ""

    def fake_run(command, **kwargs):
        if command == ["dry-run-command"]:
            dry_run_report.write_text(
                json.dumps(
                    {
                        "status": "SUCCESS",
                        "dry_run": True,
                        "persist_requested": False,
                        "persist_approved": False,
                        "candidate_files": 0,
                        "capture_count": 0,
                        "metadata_missing_count": 0,
                        "metadata_unsafe_count": 0,
                        "metadata_mismatch_count": 0,
                        "captures": [],
                    }
                ),
                encoding="utf-8",
            )
            return Completed()
        raise AssertionError("approved persist command should not run")

    monkeypatch.setattr(loop.subprocess, "run", fake_run)
    plan = {
        "run_dir": str(tmp_path / "run"),
        "persist_readiness_gate": {"path": str(dry_run_report)},
        "steps": [
            {
                "name": "dry_run_prejump_capture",
                "status": "READY_TO_RUN",
                "command": ["dry-run-command"],
            },
            {
                "name": "approved_persist_ready_subset",
                "status": "READY_TO_RUN",
                "command": ["should-not-run"],
            }
        ],
    }

    results = loop.execute_ready_steps(plan)

    assert results[0]["name"] == "dry_run_prejump_capture"
    assert results[0]["returncode"] == 0
    assert results[0]["output_report_freshness"]["fresh_for_current_execution"] is True
    assert results[1]["name"] == "approved_persist_ready_subset"
    assert results[1]["returncode"] is None
    assert results[1]["status"] == "SKIPPED"
    assert results[1]["reason"] == "persist_readiness_gate_not_clean"
    assert results[1]["persist_readiness_gate"]["status"] == "NOT_READY"
    assert "capture_count_zero" in results[1]["persist_readiness_gate"]["reason"]


def test_execute_ready_skips_approved_persist_without_same_run_dry_capture(
    tmp_path,
    monkeypatch,
):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("approved persist command should not run")

    monkeypatch.setattr(loop.subprocess, "run", fail_if_called)
    plan = {
        "run_dir": str(tmp_path / "run"),
        "persist_readiness_gate": {"path": str(tmp_path / "dry_run_capture_report.json")},
        "steps": [
            {
                "name": "approved_persist_ready_subset",
                "status": "READY_TO_RUN",
                "command": ["should-not-run"],
            }
        ],
    }

    results = loop.execute_ready_steps(plan)

    assert results == [
        {
            "name": "approved_persist_ready_subset",
            "returncode": None,
            "status": "SKIPPED",
            "reason": "dry_run_prejump_capture_not_completed_in_this_execution",
        }
    ]


def test_execute_ready_rejects_stale_preexisting_dry_run_report(
    tmp_path,
    monkeypatch,
):
    dry_run_report = tmp_path / "dry_run_capture_report.json"
    dry_run_report.write_text(
        json.dumps(_clean_dry_run_capture_report()),
        encoding="utf-8",
    )

    class Completed:
        returncode = 0
        stdout = "dry run ok"
        stderr = ""

    def fake_run(command, **kwargs):
        if command == ["dry-run-command"]:
            return Completed()
        raise AssertionError("approved persist command should not run")

    monkeypatch.setattr(loop.subprocess, "run", fake_run)
    plan = {
        "run_dir": str(tmp_path / "run"),
        "persist_readiness_gate": {"path": str(dry_run_report)},
        "steps": [
            {
                "name": "dry_run_prejump_capture",
                "status": "READY_TO_RUN",
                "command": ["dry-run-command"],
            },
            {
                "name": "approved_persist_ready_subset",
                "status": "READY_TO_RUN",
                "command": ["should-not-run"],
            }
        ],
    }

    results = loop.execute_ready_steps(plan)

    assert len(results) == 1
    assert results[0]["name"] == "dry_run_prejump_capture"
    assert results[0]["returncode"] == 0
    assert results[0]["status"] == "FAILED_REPORT_FRESHNESS"
    assert results[0]["reason"] == (
        "dry_run_capture_report_not_fresh_for_current_execution"
    )
    assert results[0]["output_report_freshness"]["fresh_for_current_execution"] is False


def test_execute_ready_runs_approved_persist_when_readiness_gate_is_clean(
    tmp_path,
    monkeypatch,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    dry_run_report = run_dir / "dry_run_capture_report.json"
    calls = []

    class Completed:
        def __init__(self, stdout: str):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command == ["dry-run-command"]:
            dry_run_report.write_text(
                json.dumps(_clean_dry_run_capture_report()),
                encoding="utf-8",
            )
            return Completed("dry run ok")
        return Completed("persist ok")

    monkeypatch.setattr(loop.subprocess, "run", fake_run)
    plan = {
        "run_dir": str(run_dir),
        "persist_readiness_gate": {"path": str(dry_run_report)},
        "steps": [
            {
                "name": "dry_run_prejump_capture",
                "status": "READY_TO_RUN",
                "command": ["dry-run-command"],
            },
            {
                "name": "approved_persist_ready_subset",
                "status": "READY_TO_RUN",
                "command": ["persist-command"],
            }
        ],
    }

    results = loop.execute_ready_steps(plan)

    assert calls[0][0] == ["dry-run-command"]
    assert calls[1][0] == ["persist-command"]
    assert calls[0][1]["cwd"] == loop.ROOT
    assert calls[1][1]["cwd"] == loop.ROOT
    assert results[0]["output_report_freshness"]["fresh_for_current_execution"] is True
    assert results[1]["name"] == "approved_persist_ready_subset"
    assert results[1]["returncode"] == 0
    assert results[1]["persist_readiness_gate"]["status"] == "READY"
    assert results[1]["persist_readiness_gate"]["ready_count"] == 1


def test_execute_ready_does_not_run_waiting_persist_without_approval(
    tmp_path,
    monkeypatch,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    dry_run_report = run_dir / "dry_run_capture_report.json"
    calls = []

    class Completed:
        returncode = 0
        stdout = "dry run ok"
        stderr = ""

    def fake_run(command, **kwargs):
        calls.append(command)
        if command == ["dry-run-command"]:
            dry_run_report.write_text(
                json.dumps(_clean_dry_run_capture_report()),
                encoding="utf-8",
            )
            return Completed()
        raise AssertionError("unapproved waiting persist command should not run")

    monkeypatch.setattr(loop.subprocess, "run", fake_run)
    plan = {
        "run_dir": str(run_dir),
        "approvals": {"live_persist": False},
        "persist_readiness_gate": {"path": str(dry_run_report)},
        "steps": [
            {
                "name": "dry_run_prejump_capture",
                "status": "READY_TO_RUN",
                "command": ["dry-run-command"],
            },
            {
                "name": "approved_persist_ready_subset",
                "status": "WAITING_FOR_READY_PERSIST_PACKET",
                "command": ["persist-command"],
            },
        ],
    }

    results = loop.execute_ready_steps(plan)

    assert calls == [["dry-run-command"]]
    assert [result["name"] for result in results] == ["dry_run_prejump_capture"]


def test_execute_ready_runs_approved_waiting_persist_after_same_run_dry_run(
    tmp_path,
    monkeypatch,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    dry_run_report = run_dir / "dry_run_capture_report.json"
    calls = []

    class Completed:
        def __init__(self, stdout: str):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    def fake_run(command, **kwargs):
        calls.append(command)
        if command == ["dry-run-command"]:
            dry_run_report.write_text(
                json.dumps(_clean_dry_run_capture_report()),
                encoding="utf-8",
            )
            return Completed("dry run ok")
        if command == ["persist-command"]:
            return Completed("persist ok")
        raise AssertionError("unexpected command")

    monkeypatch.setattr(loop.subprocess, "run", fake_run)
    plan = {
        "run_dir": str(run_dir),
        "approvals": {"live_persist": True},
        "persist_readiness_gate": {"path": str(dry_run_report)},
        "steps": [
            {
                "name": "dry_run_prejump_capture",
                "status": "READY_TO_RUN",
                "command": ["dry-run-command"],
            },
            {
                "name": "approved_persist_ready_subset",
                "status": "WAITING_FOR_READY_PERSIST_PACKET",
                "command": ["persist-command"],
            },
        ],
    }

    results = loop.execute_ready_steps(plan)

    assert calls == [["dry-run-command"], ["persist-command"]]
    assert results[1]["name"] == "approved_persist_ready_subset"
    assert results[1]["returncode"] == 0
    assert results[1]["persist_readiness_gate"]["status"] == "READY"
    assert results[1]["persist_readiness_gate"]["ready_count"] == 1


def test_execute_ready_skips_live_odds_without_same_run_dry_capture(
    tmp_path,
    monkeypatch,
):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("live odds command should not run")

    monkeypatch.setattr(loop.subprocess, "run", fail_if_called)
    plan = {
        "run_dir": str(tmp_path / "run"),
        "persist_readiness_gate": {"path": str(tmp_path / "dry_run_capture_report.json")},
        "live_odds_approval_packet": {
            "odds_capture_report_path": str(tmp_path / "odds_report.json"),
        },
        "steps": [
            {
                "name": "opt_in_live_odds_capture",
                "status": "READY_TO_RUN",
                "command": ["should-not-run"],
            }
        ],
    }

    results = loop.execute_ready_steps(plan)

    assert results == [
        {
            "name": "opt_in_live_odds_capture",
            "returncode": None,
            "status": "SKIPPED",
            "reason": "dry_run_prejump_capture_not_completed_in_this_execution",
        }
    ]


def test_execute_ready_runs_live_odds_after_fresh_clean_dry_run(
    tmp_path,
    monkeypatch,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    dry_run_report = run_dir / "dry_run_capture_report.json"
    odds_report = run_dir / "odds_capture_report.json"
    calls = []

    class Completed:
        def __init__(self, stdout: str):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command == ["dry-run-command"]:
            dry_run_report.write_text(
                json.dumps(_clean_dry_run_capture_report()),
                encoding="utf-8",
            )
            return Completed("dry run ok")
        if command == ["live-odds-command"]:
            odds_report.write_text(
                json.dumps(
                    {
                        "status": "SUCCESS",
                        "dry_run": True,
                        "odds_capture_requested": True,
                        "odds_capture_approved": True,
                    }
                ),
                encoding="utf-8",
            )
            return Completed("odds ok")
        raise AssertionError("unexpected command")

    monkeypatch.setattr(loop.subprocess, "run", fake_run)
    plan = {
        "run_dir": str(run_dir),
        "persist_readiness_gate": {"path": str(dry_run_report)},
        "live_odds_approval_packet": {
            "odds_capture_report_path": str(odds_report),
        },
        "steps": [
            {
                "name": "dry_run_prejump_capture",
                "status": "READY_TO_RUN",
                "command": ["dry-run-command"],
            },
            {
                "name": "opt_in_live_odds_capture",
                "status": "READY_TO_RUN",
                "command": ["live-odds-command"],
            }
        ],
    }

    results = loop.execute_ready_steps(plan)

    assert calls[0][0] == ["dry-run-command"]
    assert calls[1][0] == ["live-odds-command"]
    assert results[0]["output_report_freshness"]["fresh_for_current_execution"] is True
    assert results[1]["name"] == "opt_in_live_odds_capture"
    assert results[1]["returncode"] == 0
    assert results[1]["live_odds_readiness_gate"]["status"] == "READY"
    assert results[1]["output_report_freshness"]["fresh_for_current_execution"] is True


def test_execute_ready_runs_approved_waiting_live_odds_after_same_run_dry_run(
    tmp_path,
    monkeypatch,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    dry_run_report = run_dir / "dry_run_capture_report.json"
    odds_report = run_dir / "odds_capture_report.json"
    calls = []

    class Completed:
        def __init__(self, stdout: str):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    def fake_run(command, **kwargs):
        calls.append(command)
        if command == ["dry-run-command"]:
            dry_run_report.write_text(
                json.dumps(_clean_dry_run_capture_report()),
                encoding="utf-8",
            )
            return Completed("dry run ok")
        if command == ["live-odds-command"]:
            odds_report.write_text(
                json.dumps(
                    {
                        "status": "SUCCESS",
                        "dry_run": True,
                        "odds_capture_requested": True,
                        "odds_capture_approved": True,
                    }
                ),
                encoding="utf-8",
            )
            return Completed("odds ok")
        raise AssertionError("unexpected command")

    monkeypatch.setattr(loop.subprocess, "run", fake_run)
    plan = {
        "run_dir": str(run_dir),
        "approvals": {"live_odds_capture": True},
        "persist_readiness_gate": {"path": str(dry_run_report)},
        "live_odds_approval_packet": {
            "odds_capture_report_path": str(odds_report),
        },
        "steps": [
            {
                "name": "dry_run_prejump_capture",
                "status": "READY_TO_RUN",
                "command": ["dry-run-command"],
            },
            {
                "name": "opt_in_live_odds_capture",
                "status": "WAITING_FOR_READY_ODDS_PACKET",
                "command": ["live-odds-command"],
            },
        ],
    }

    results = loop.execute_ready_steps(plan)

    assert calls == [["dry-run-command"], ["live-odds-command"]]
    assert results[1]["name"] == "opt_in_live_odds_capture"
    assert results[1]["returncode"] == 0
    assert results[1]["live_odds_readiness_gate"]["status"] == "READY"


def test_execute_ready_skips_label_write_without_same_run_result_dry_run(
    tmp_path,
    monkeypatch,
):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("label write command should not run")

    monkeypatch.setattr(loop.subprocess, "run", fail_if_called)
    plan = {
        "run_dir": str(tmp_path / "run"),
        "result_dry_run_report_gate": {
            "path": str(tmp_path / "result_ingest_dry_run_report.json"),
            "expected_scope": {},
        },
        "steps": [
            {
                "name": "approved_official_label_write",
                "status": "READY_TO_RUN",
                "command": ["should-not-run"],
            }
        ],
    }

    results = loop.execute_ready_steps(plan)

    assert results == [
        {
            "name": "approved_official_label_write",
            "returncode": None,
            "status": "SKIPPED",
            "reason": "official_result_ingest_dry_run_not_completed_in_this_execution",
        }
    ]


def test_execute_ready_runs_label_write_after_fresh_clean_result_dry_run(
    tmp_path,
    monkeypatch,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result_report = run_dir / "result_ingest_dry_run_report.json"
    readiness_report = run_dir / "label_write_readiness_validation.json"
    preflight_report = run_dir / "label_write_preflight_packet.json"
    db_path = tmp_path / "labels.sqlite"
    expected_scope = {
        "db_path": str(db_path.resolve()),
        "date": "2026-05-29",
        "upcoming_dir": str((tmp_path / "upcoming").resolve()),
        "snapshot_dir": str((tmp_path / "snapshots").resolve()),
        "race_ids": [],
        "require_ready_snapshot": True,
    }
    readiness_report.write_text(
        json.dumps(_clean_label_write_readiness_report(expected_scope)),
        encoding="utf-8",
    )
    preflight_report.write_text(
        json.dumps(
            _clean_label_write_preflight_packet(
                expected_scope,
                label_readiness_path=readiness_report,
                result_report_path=result_report,
                db_path=db_path,
            )
        ),
        encoding="utf-8",
    )
    calls = []

    class Completed:
        def __init__(self, stdout: str):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command == ["result-dry-run-command"]:
            result_report.write_text(
                json.dumps(_clean_result_dry_run_report(expected_scope)),
                encoding="utf-8",
            )
            return Completed("result dry run ok")
        return Completed("label write ok")

    monkeypatch.setattr(loop.subprocess, "run", fake_run)
    plan = {
        "run_dir": str(run_dir),
        "result_dry_run_report_gate": {
            "path": str(result_report),
            "expected_scope": expected_scope,
        },
        "label_write_readiness_validation_gate": {
            "path": str(readiness_report),
        },
        "label_write_preflight_packet_gate": {
            "path": str(preflight_report),
            "db_path": str(db_path),
        },
        "steps": [
            {
                "name": "official_result_ingest_dry_run",
                "status": "READY_TO_RUN",
                "command": ["result-dry-run-command"],
            },
            {
                "name": "approved_official_label_write",
                "status": "READY_TO_RUN",
                "command": ["label-write-command"],
            }
        ],
    }

    results = loop.execute_ready_steps(plan)

    assert calls[0][0] == ["result-dry-run-command"]
    assert calls[1][0] == ["label-write-command"]
    assert results[0]["output_report_freshness"]["fresh_for_current_execution"] is True
    assert results[1]["name"] == "approved_official_label_write"
    assert results[1]["returncode"] == 0
    assert results[1]["result_dry_run_report_gate"]["status"] == "READY"
    assert results[1]["label_write_preflight_packet_gate"]["status"] == "READY"


def test_execute_ready_blocks_label_write_without_preflight_packet(
    tmp_path,
    monkeypatch,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result_report = run_dir / "result_ingest_dry_run_report.json"
    readiness_report = run_dir / "label_write_readiness_validation.json"
    preflight_report = run_dir / "missing_label_write_preflight_packet.json"
    db_path = tmp_path / "labels.sqlite"
    expected_scope = {
        "db_path": str(db_path.resolve()),
        "date": "2026-05-29",
        "upcoming_dir": str((tmp_path / "upcoming").resolve()),
        "snapshot_dir": str((tmp_path / "snapshots").resolve()),
        "race_ids": [],
        "require_ready_snapshot": True,
    }
    readiness_report.write_text(
        json.dumps(_clean_label_write_readiness_report(expected_scope)),
        encoding="utf-8",
    )
    calls = []

    class Completed:
        def __init__(self, stdout: str):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    def fake_run(command, **kwargs):
        calls.append(command)
        if command == ["result-dry-run-command"]:
            result_report.write_text(
                json.dumps(_clean_result_dry_run_report(expected_scope)),
                encoding="utf-8",
            )
            return Completed("result dry run ok")
        raise AssertionError("label write should not run without preflight packet")

    monkeypatch.setattr(loop.subprocess, "run", fake_run)
    plan = {
        "run_dir": str(run_dir),
        "approvals": {"result_label_write": True},
        "result_dry_run_report_gate": {
            "path": str(result_report),
            "expected_scope": expected_scope,
        },
        "label_write_readiness_validation_gate": {
            "path": str(readiness_report),
        },
        "label_write_preflight_packet_gate": {
            "path": str(preflight_report),
            "db_path": str(db_path),
        },
        "steps": [
            {
                "name": "official_result_ingest_dry_run",
                "status": "READY_TO_RUN",
                "command": ["result-dry-run-command"],
            },
            {
                "name": "approved_official_label_write",
                "status": "WAITING_FOR_CLEAN_RESULT_DRY_RUN",
                "command": ["label-write-command"],
            },
        ],
    }

    results = loop.execute_ready_steps(plan)

    assert calls == [["result-dry-run-command"]]
    assert results[1]["name"] == "approved_official_label_write"
    assert results[1]["status"] == "SKIPPED"
    assert results[1]["reason"] == "label_write_preflight_packet_not_ready"
    assert results[1]["label_write_preflight_packet_gate"]["status"] == "DATA_MISSING"


def test_execute_ready_runs_approved_waiting_label_write_after_result_dry_run(
    tmp_path,
    monkeypatch,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result_report = run_dir / "result_ingest_dry_run_report.json"
    readiness_report = run_dir / "label_write_readiness_validation.json"
    preflight_report = run_dir / "label_write_preflight_packet.json"
    db_path = tmp_path / "labels.sqlite"
    expected_scope = {
        "db_path": str(db_path.resolve()),
        "date": "2026-05-29",
        "upcoming_dir": str((tmp_path / "upcoming").resolve()),
        "snapshot_dir": str((tmp_path / "snapshots").resolve()),
        "race_ids": [],
        "require_ready_snapshot": True,
    }
    readiness_report.write_text(
        json.dumps(_clean_label_write_readiness_report(expected_scope)),
        encoding="utf-8",
    )
    preflight_report.write_text(
        json.dumps(
            _clean_label_write_preflight_packet(
                expected_scope,
                label_readiness_path=readiness_report,
                result_report_path=result_report,
                db_path=db_path,
            )
        ),
        encoding="utf-8",
    )
    calls = []

    class Completed:
        def __init__(self, stdout: str):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    def fake_run(command, **kwargs):
        calls.append(command)
        if command == ["result-dry-run-command"]:
            result_report.write_text(
                json.dumps(_clean_result_dry_run_report(expected_scope)),
                encoding="utf-8",
            )
            return Completed("result dry run ok")
        if command == ["label-write-command"]:
            return Completed("label write ok")
        raise AssertionError("unexpected command")

    monkeypatch.setattr(loop.subprocess, "run", fake_run)
    plan = {
        "run_dir": str(run_dir),
        "approvals": {"result_label_write": True},
        "result_dry_run_report_gate": {
            "path": str(result_report),
            "expected_scope": expected_scope,
        },
        "label_write_readiness_validation_gate": {
            "path": str(readiness_report),
        },
        "label_write_preflight_packet_gate": {
            "path": str(preflight_report),
            "db_path": str(db_path),
        },
        "steps": [
            {
                "name": "official_result_ingest_dry_run",
                "status": "READY_TO_RUN",
                "command": ["result-dry-run-command"],
            },
            {
                "name": "approved_official_label_write",
                "status": "WAITING_FOR_CLEAN_RESULT_DRY_RUN",
                "command": ["label-write-command"],
            },
        ],
    }

    results = loop.execute_ready_steps(plan)

    assert calls == [["result-dry-run-command"], ["label-write-command"]]
    assert results[1]["name"] == "approved_official_label_write"
    assert results[1]["returncode"] == 0
    assert results[1]["result_dry_run_report_gate"]["status"] == "READY"
    assert results[1]["label_write_preflight_packet_gate"]["status"] == "READY"


def test_execute_ready_runs_label_write_readiness_after_result_dry_run(
    tmp_path,
    monkeypatch,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result_report = run_dir / "result_ingest_dry_run_report.json"
    readiness_report = run_dir / "label_write_readiness_validation.json"
    expected_scope = {
        "db_path": str((tmp_path / "labels.sqlite").resolve()),
        "date": "2026-05-29",
        "upcoming_dir": str((tmp_path / "upcoming").resolve()),
        "snapshot_dir": str((tmp_path / "snapshots").resolve()),
        "race_ids": ["Race 1 - TEST - 2026-05-29"],
        "require_ready_snapshot": True,
    }
    calls = []

    class Completed:
        def __init__(self, stdout: str):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    def fake_run(command, **kwargs):
        calls.append(command)
        if command == ["result-dry-run-command"]:
            result_report.write_text(
                json.dumps(_clean_result_dry_run_report(expected_scope)),
                encoding="utf-8",
            )
            return Completed("result dry run ok")
        if command == ["readiness-command"]:
            readiness_report.write_text(
                json.dumps(_clean_label_write_readiness_report(expected_scope)),
                encoding="utf-8",
            )
            return Completed("readiness ok")
        raise AssertionError("unexpected command")

    monkeypatch.setattr(loop.subprocess, "run", fake_run)
    plan = {
        "run_dir": str(run_dir),
        "result_dry_run_report_gate": {
            "path": str(result_report),
            "expected_scope": expected_scope,
        },
        "label_write_readiness_validation_gate": {
            "path": str(readiness_report),
        },
        "steps": [
            {
                "name": "official_result_ingest_dry_run",
                "status": "READY_TO_RUN",
                "command": ["result-dry-run-command"],
            },
            {
                "name": "result_label_write_readiness_validation",
                "status": "WAITING_FOR_CLEAN_RESULT_DRY_RUN",
                "command": ["readiness-command"],
            },
            {
                "name": "approved_official_label_write",
                "status": "APPROVAL_REQUIRED",
                "command": ["label-write-command"],
            },
        ],
    }

    results = loop.execute_ready_steps(plan)

    assert calls == [["result-dry-run-command"], ["readiness-command"]]
    assert [result["name"] for result in results] == [
        "official_result_ingest_dry_run",
        "result_label_write_readiness_validation",
    ]
    assert results[1]["returncode"] == 0
    assert results[1]["label_write_readiness_validation_gate"]["status"] == "READY"
    assert results[1]["output_report_freshness"][
        "fresh_for_current_execution"
    ] is True


def test_execute_ready_rejects_stale_preexisting_evaluation_report(
    tmp_path,
    monkeypatch,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    eval_report = run_dir / "evaluation_report.json"
    dataset_path = run_dir / "evaluation_dataset.jsonl"
    dataset_path.write_text("{}\n{}\n{}\n{}\n", encoding="utf-8")
    eval_report.write_text(
        json.dumps(_clean_evaluation_report(dataset_path)),
        encoding="utf-8",
    )

    class Completed:
        returncode = 0
        stdout = "evaluation ok"
        stderr = ""

    def fake_run(command, **kwargs):
        if command == ["evaluation-command"]:
            return Completed()
        raise AssertionError("unexpected command")

    monkeypatch.setattr(loop.subprocess, "run", fake_run)
    plan = {
        "run_dir": str(run_dir),
        "evaluation_report_gate": {
            "path": str(eval_report),
            "dataset_path": str(dataset_path),
        },
        "steps": [
            {
                "name": "rolling_evaluation_dataset",
                "status": "READY_TO_RUN",
                "command": ["evaluation-command"],
            }
        ],
    }

    results = loop.execute_ready_steps(plan)

    assert len(results) == 1
    assert results[0]["name"] == "rolling_evaluation_dataset"
    assert results[0]["returncode"] == 0
    assert results[0]["status"] == "FAILED_REPORT_FRESHNESS"
    assert results[0]["reason"] == "evaluation_report_not_fresh_for_current_execution"
    assert results[0]["output_report_freshness"]["fresh_for_current_execution"] is False


def test_execute_ready_rechecks_evaluation_gate_after_execution(
    tmp_path,
    monkeypatch,
    capsys,
):
    for name in (
        "APPROVE_LIVE_PERSIST",
        "APPROVE_LIVE_ODDS_CAPTURE",
        "APPROVE_RESULT_LABEL_WRITE",
        "APPROVE_MODEL_PROMOTION",
    ):
        monkeypatch.delenv(name, raising=False)
    snapshot_dir = tmp_path / "snapshots"
    date_dir = snapshot_dir / "2026-05-29" / "TEST"
    date_dir.mkdir(parents=True)
    (date_dir / "race-1.json").write_text(
        json.dumps(
            {
                "schema_version": "prediction_snapshot_v1",
                "race_id": "Race 1 - TEST - 2026-05-29",
                "is_pre_jump_snapshot": True,
                "snapshot_state": "pre_jump_feature_freeze",
                "snapshot_readiness": {"status": "READY"},
                "predictions": READY_PREDICTIONS,
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    output = run_dir / "loop.json"

    def fake_execute_ready_steps(plan):
        run_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = run_dir / "evaluation_dataset.jsonl"
        dataset_path.write_text("{}\n{}\n{}\n{}\n", encoding="utf-8")
        (run_dir / "evaluation_report.json").write_text(
            json.dumps(_clean_evaluation_report(dataset_path)),
            encoding="utf-8",
        )
        return [{"name": "rolling_evaluation_dataset", "returncode": 0}]

    monkeypatch.setattr(loop, "execute_ready_steps", fake_execute_ready_steps)

    assert (
        loop.main(
            [
                "--snapshot-dir",
                str(snapshot_dir),
                "--run-dir",
                str(run_dir),
                "--date",
                "2026-05-29",
                "--execute-ready",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    capsys.readouterr()

    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["evaluation_report_gate"]["status"] == "DATA_MISSING"
    assert payload["post_execution_evaluation_report_gate"]["status"] == "READY"
    assert payload["post_execution_milestone_completion_audit"]["items"][7][
        "complete"
    ] is True
    assert payload["post_execution_milestone_completion_audit"]["items"][8][
        "complete"
    ] is True


def test_execute_ready_builds_model_review_packet_after_fresh_evaluation(
    tmp_path,
    monkeypatch,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    eval_report = run_dir / "evaluation_report.json"
    dataset_path = run_dir / "evaluation_dataset.jsonl"
    review_packet = run_dir / "model_review_packet.json"
    calls = []

    class Completed:
        def __init__(self, stdout: str):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    def fake_run(command, **kwargs):
        calls.append(command)
        if command == ["evaluation-command"]:
            dataset_path.write_text("{}\n{}\n{}\n{}\n", encoding="utf-8")
            eval_report.write_text(
                json.dumps(_clean_evaluation_report(dataset_path)),
                encoding="utf-8",
            )
            return Completed("evaluation ok")
        if command == ["review-packet-command"]:
            review_packet.write_text(
                json.dumps(_clean_model_review_packet(eval_report, dataset_path)),
                encoding="utf-8",
            )
            return Completed("review packet ok")
        raise AssertionError("unexpected command")

    monkeypatch.setattr(loop.subprocess, "run", fake_run)
    plan = {
        "run_dir": str(run_dir),
        "evaluation_report_gate": {
            "path": str(eval_report),
            "dataset_path": str(dataset_path),
        },
        "model_review_packet_gate": {
            "path": str(review_packet),
            "evaluation_report_path": str(eval_report),
            "dataset_path": str(dataset_path),
        },
        "steps": [
            {
                "name": "rolling_evaluation_dataset",
                "status": "READY_TO_RUN",
                "command": ["evaluation-command"],
            },
            {
                "name": "model_review_packet",
                "status": "READY_TO_RUN",
                "command": ["review-packet-command"],
            },
        ],
    }

    results = loop.execute_ready_steps(plan)

    assert calls == [["evaluation-command"], ["review-packet-command"]]
    assert results[0]["output_report_freshness"]["fresh_for_current_execution"] is True
    assert results[1]["output_report_freshness"]["fresh_for_current_execution"] is True
    assert results[1]["model_review_packet_gate"]["status"] == "READY"
    assert results[1]["model_review_packet_gate"]["packet_status"] == (
        "READY_FOR_CHALLENGER_REVIEW"
    )
    assert results[1]["model_review_packet_gate"]["promotion_allowed"] is False
    assert results[1]["model_review_packet_gate"]["registry_mutation_allowed"] is False
    assert results[1]["model_review_packet_gate"]["required_promotion_gate"] == (
        "APPROVE_MODEL_PROMOTION"
    )


def test_execute_ready_validates_challenger_review_gate_after_model_packet(
    tmp_path,
    monkeypatch,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    eval_report = run_dir / "evaluation_report.json"
    dataset_path = run_dir / "evaluation_dataset.jsonl"
    review_packet = run_dir / "model_review_packet.json"
    challenger_review = run_dir / "snapshot_challenger_review.json"
    calls = []

    class Completed:
        def __init__(self, stdout: str):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    def fake_run(command, **kwargs):
        calls.append(command)
        if command == ["evaluation-command"]:
            dataset_path.write_text("{}\n{}\n{}\n{}\n", encoding="utf-8")
            eval_report.write_text(
                json.dumps(_clean_evaluation_report(dataset_path)),
                encoding="utf-8",
            )
            return Completed("evaluation ok")
        if command == ["review-packet-command"]:
            challenger_review.write_text(
                json.dumps({"schema_version": "snapshot_challenger_review_v1"}),
                encoding="utf-8",
            )
            review_packet.write_text(
                json.dumps(
                    _clean_model_review_packet(
                        eval_report,
                        dataset_path,
                        challenger_review,
                    )
                ),
                encoding="utf-8",
            )
            return Completed("review packet ok")
        raise AssertionError("unexpected command")

    monkeypatch.setattr(loop.subprocess, "run", fake_run)
    plan = {
        "run_dir": str(run_dir),
        "evaluation_report_gate": {
            "path": str(eval_report),
            "dataset_path": str(dataset_path),
        },
        "model_review_packet_gate": {
            "path": str(review_packet),
            "evaluation_report_path": str(eval_report),
            "dataset_path": str(dataset_path),
            "challenger_review_path": str(challenger_review),
        },
        "steps": [
            {
                "name": "rolling_evaluation_dataset",
                "status": "READY_TO_RUN",
                "command": ["evaluation-command"],
            },
            {
                "name": "model_review_packet",
                "status": "READY_TO_RUN",
                "command": ["review-packet-command"],
            },
        ],
    }

    results = loop.execute_ready_steps(plan)
    packet_gate = results[1]["model_review_packet_gate"]

    assert calls == [["evaluation-command"], ["review-packet-command"]]
    assert packet_gate["status"] == "READY"
    assert packet_gate["challenger_review_path"] == str(challenger_review)
    assert packet_gate["challenger_review_gate_status"] == "READY"
    assert packet_gate["challenger_review_candidate_arm"] == (
        "power_calibrated_baseline"
    )
    assert packet_gate["challenger_review_stability_status"] == "STABLE_REPORT_ONLY"
    assert packet_gate["challenger_review_promotion_allowed"] is False
    assert packet_gate["promotion_allowed"] is False
    assert packet_gate["registry_mutation_allowed"] is False


def test_execute_ready_builds_calibration_design_after_fresh_model_packet(
    tmp_path,
    monkeypatch,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    eval_report = run_dir / "evaluation_report.json"
    dataset_path = run_dir / "evaluation_dataset.jsonl"
    review_packet = run_dir / "model_review_packet.json"
    challenger_review = run_dir / "snapshot_challenger_review.json"
    calibration_design = run_dir / "calibration_layer_design.json"
    calls = []

    class Completed:
        def __init__(self, stdout: str):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    def fake_run(command, **kwargs):
        calls.append(command)
        if command == ["evaluation-command"]:
            dataset_path.write_text("{}\n{}\n{}\n{}\n", encoding="utf-8")
            eval_report.write_text(
                json.dumps(_clean_evaluation_report(dataset_path)),
                encoding="utf-8",
            )
            return Completed("evaluation ok")
        if command == ["review-packet-command"]:
            challenger_review.write_text(
                json.dumps({"schema_version": "snapshot_challenger_review_v1"}),
                encoding="utf-8",
            )
            review_packet.write_text(
                json.dumps(
                    _clean_model_review_packet(
                        eval_report,
                        dataset_path,
                        challenger_review,
                    )
                ),
                encoding="utf-8",
            )
            return Completed("review packet ok")
        if command == ["calibration-design-command"]:
            calibration_design.write_text(
                json.dumps(_clean_calibration_design_report(review_packet)),
                encoding="utf-8",
            )
            return Completed("calibration design ok")
        raise AssertionError("unexpected command")

    monkeypatch.setattr(loop.subprocess, "run", fake_run)
    plan = {
        "run_dir": str(run_dir),
        "evaluation_report_gate": {
            "path": str(eval_report),
            "dataset_path": str(dataset_path),
        },
        "model_review_packet_gate": {
            "path": str(review_packet),
            "evaluation_report_path": str(eval_report),
            "dataset_path": str(dataset_path),
            "challenger_review_path": str(challenger_review),
        },
        "calibration_design_gate": {
            "path": str(calibration_design),
            "model_review_packet_path": str(review_packet),
        },
        "steps": [
            {
                "name": "rolling_evaluation_dataset",
                "status": "READY_TO_RUN",
                "command": ["evaluation-command"],
            },
            {
                "name": "model_review_packet",
                "status": "READY_TO_RUN",
                "command": ["review-packet-command"],
            },
            {
                "name": "calibration_layer_design",
                "status": "READY_TO_RUN",
                "command": ["calibration-design-command"],
            },
        ],
    }

    results = loop.execute_ready_steps(plan)
    design_gate = results[2]["calibration_design_gate"]

    assert calls == [
        ["evaluation-command"],
        ["review-packet-command"],
        ["calibration-design-command"],
    ]
    assert results[2]["output_report_freshness"]["fresh_for_current_execution"] is True
    assert design_gate["status"] == "READY"
    assert design_gate["runtime_transform_spec"]["candidate_arm"] == (
        "power_calibrated_baseline"
    )
    assert design_gate["deployment_control"]["promotion_allowed"] is False
    assert design_gate["deployment_control"]["registry_mutation_allowed"] is False


def test_execute_ready_runs_same_execution_challenger_before_packet(
    tmp_path,
    monkeypatch,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    eval_report = run_dir / "evaluation_report.json"
    dataset_path = run_dir / "evaluation_dataset.jsonl"
    challenger_review = run_dir / "snapshot_challenger_review.json"
    review_packet = run_dir / "model_review_packet.json"
    calls = []

    class Completed:
        def __init__(self, stdout: str):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    def fake_run(command, **kwargs):
        calls.append(command)
        if command == ["evaluation-command"]:
            dataset_path.write_text("{}\n{}\n{}\n{}\n", encoding="utf-8")
            eval_report.write_text(
                json.dumps(_clean_evaluation_report(dataset_path)),
                encoding="utf-8",
            )
            return Completed("evaluation ok")
        if command == ["challenger-review-command"]:
            challenger_review.write_text(
                json.dumps(_clean_snapshot_challenger_review_report(dataset_path)),
                encoding="utf-8",
            )
            return Completed("challenger review ok")
        if command == ["review-packet-command"]:
            review_packet.write_text(
                json.dumps(
                    _clean_model_review_packet(
                        eval_report,
                        dataset_path,
                        challenger_review,
                    )
                ),
                encoding="utf-8",
            )
            return Completed("review packet ok")
        raise AssertionError("unexpected command")

    monkeypatch.setattr(loop.subprocess, "run", fake_run)
    plan = {
        "run_dir": str(run_dir),
        "evaluation_report_gate": {
            "path": str(eval_report),
            "dataset_path": str(dataset_path),
        },
        "snapshot_challenger_review_gate": {
            "path": str(challenger_review),
            "dataset_path": str(dataset_path),
        },
        "model_review_packet_gate": {
            "path": str(review_packet),
            "evaluation_report_path": str(eval_report),
            "dataset_path": str(dataset_path),
            "challenger_review_path": str(challenger_review),
        },
        "steps": [
            {
                "name": "rolling_evaluation_dataset",
                "status": "READY_TO_RUN",
                "command": ["evaluation-command"],
            },
            {
                "name": "snapshot_challenger_review",
                "status": "READY_TO_RUN",
                "command": ["challenger-review-command"],
            },
            {
                "name": "model_review_packet",
                "status": "READY_TO_RUN",
                "command": ["review-packet-command"],
            },
        ],
    }

    results = loop.execute_ready_steps(plan)

    assert calls == [
        ["evaluation-command"],
        ["challenger-review-command"],
        ["review-packet-command"],
    ]
    assert results[1]["snapshot_challenger_review_gate"]["status"] == "READY"
    assert results[1]["output_report_freshness"]["fresh_for_current_execution"] is True
    assert results[2]["model_review_packet_gate"]["status"] == "READY"


def test_model_review_packet_gate_rejects_challenger_review_scope_mismatch(
    tmp_path,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    eval_report = run_dir / "evaluation_report.json"
    dataset_path = run_dir / "evaluation_dataset.jsonl"
    review_packet = run_dir / "model_review_packet.json"
    expected_challenger = run_dir / "expected_challenger_review.json"
    observed_challenger = run_dir / "observed_challenger_review.json"
    dataset_path.write_text("{}\n{}\n{}\n{}\n", encoding="utf-8")
    eval_report.write_text(
        json.dumps(_clean_evaluation_report(dataset_path)),
        encoding="utf-8",
    )
    review_packet.write_text(
        json.dumps(
            _clean_model_review_packet(
                eval_report,
                dataset_path,
                observed_challenger,
            )
        ),
        encoding="utf-8",
    )

    gate = loop._model_review_packet_gate(
        packet_path=review_packet,
        evaluation_report_path=eval_report,
        dataset_path=dataset_path,
        challenger_review_path=expected_challenger,
    )

    assert gate["status"] == "NOT_READY"
    assert "challenger_review_scope_mismatch" in gate["reason"]
