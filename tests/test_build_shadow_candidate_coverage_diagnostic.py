import json
from datetime import datetime

from scripts import build_shadow_candidate_coverage_diagnostic as diagnostic


def test_build_diagnostic_classifies_absent_shadow_candidate():
    report = diagnostic.build_diagnostic(
        recovery_queue={
            "items": [
                {
                    "race_id": "Race 1 - GRDN - 2026-06-13",
                    "venue": "GRDN",
                    "race_number": 1,
                    "canonical_live_odds_race_id": "GRDN_2026-06-13_1",
                    "latest_capture": "2026-06-13T18:50:23+10:00",
                    "live_odds_row_count": 8,
                    "live_odds_box_count": 8,
                    "reason": "no_matching_shadow_run_candidate_found",
                    "recovery_action": "inspect_shadow_run_candidate_coverage",
                    "authorized_action": "diagnostic_review_only",
                }
            ]
        },
        shadow_prediction_race_ids={"Race 2 - GRDN - 2026-06-13"},
        stage2_prediction_race_ids={"Race 2 - GRDN - 2026-06-13"},
        refreshed_upcoming_race_ids={"Race 2 - GRDN - 2026-06-13"},
        candidate_source_race_ids={"Race 2 - GRDN - 2026-06-13"},
        generated_at=datetime.fromisoformat("2026-06-13T19:30:00+10:00"),
        source_paths={},
    )

    assert report["diagnostic_only"] is True
    assert report["summary"]["diagnostic_race_count"] == 1
    assert report["summary"]["coverage_cause_counts"] == {
        "absent_from_shadow_candidate_sources": 1
    }
    assert report["items"][0]["in_latest_stage2_predictions"] is False
    assert report["items"][0]["nearby_latest_shadow_races"] == [
        "Race 2 - GRDN - 2026-06-13"
    ]
    assert report["items"][0]["db_write_performed"] is False
    assert report["items"][0]["join_acceptance_changed"] is False


def test_extract_recovery_items_supports_nested_legacy_queue():
    items = diagnostic.extract_recovery_items(
        {
            "queues": {
                "no_exact_shadow_match": [
                    {
                        "race_id": "Race 1 - WPK - 2026-06-13",
                        "recovery_action": "inspect_shadow_run_candidate_coverage",
                    }
                ]
            }
        }
    )

    assert items == [
        {
            "race_id": "Race 1 - WPK - 2026-06-13",
            "recovery_action": "inspect_shadow_run_candidate_coverage",
            "queue": "no_exact_shadow_match",
        }
    ]


def test_main_writes_report_only_artifacts(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    script_dir = repo_root / "scripts"
    output_dir = (
        repo_root
        / "artifacts/full_evidence_orchestration_20260525"
        / "autonomous_accuracy_odds_status_test_shadow_candidate_coverage"
    )
    script_dir.mkdir(parents=True)
    monkeypatch.setattr(diagnostic, "ROOT", repo_root)

    recovery_queue = tmp_path / "queue.json"
    recovery_queue.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "race_id": "Race 1 - GRDN - 2026-06-13",
                        "venue": "GRDN",
                        "race_number": 1,
                        "live_odds_row_count": 8,
                        "live_odds_box_count": 8,
                        "recovery_action": "inspect_shadow_run_candidate_coverage",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    shadow_predictions = tmp_path / "shadow.jsonl"
    shadow_predictions.write_text(
        json.dumps({"race_id": "Race 2 - GRDN - 2026-06-13"}) + "\n",
        encoding="utf-8",
    )
    stage2_predictions = tmp_path / "stage2.jsonl"
    stage2_predictions.write_text(
        json.dumps({"race_id": "Race 2 - GRDN - 2026-06-13"}) + "\n",
        encoding="utf-8",
    )
    upcoming_dir = tmp_path / "upcoming"
    upcoming_dir.mkdir()
    (upcoming_dir / "Race 2 - GRDN - 2026-06-13.csv").write_text(
        "Dog Name,Box\nAlpha,1\n",
        encoding="utf-8",
    )
    candidate_source = tmp_path / "source.json"
    candidate_source.write_text(
        json.dumps({"candidate_race_ids": ["Race 2 - GRDN - 2026-06-13"]}),
        encoding="utf-8",
    )

    assert (
        diagnostic.main(
            [
                "--recovery-queue",
                str(recovery_queue),
                "--shadow-predictions-jsonl",
                str(shadow_predictions),
                "--stage2-shadow-predictions-jsonl",
                str(stage2_predictions),
                "--refreshed-upcoming-dir",
                str(upcoming_dir),
                "--shadow-candidate-source-report",
                str(candidate_source),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    report = json.loads(
        (output_dir / "shadow_candidate_coverage_diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["summary"]["diagnostic_race_count"] == 1
    assert report["no_write_guarantees"]["db_write"] is False
    assert (output_dir / "SUMMARY.md").exists()
    assert (output_dir / "manifest.json").exists()
