import json

from scripts import run_odds_augmented_challenger_report as odds_aug


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_odds_augmented_report_runs_after_gate_and_remains_report_only(tmp_path, monkeypatch):
    monkeypatch.setattr(odds_aug, "ROOT", tmp_path)
    monkeypatch.setattr(odds_aug, "DEFAULT_PROTECTED_PATHS", ())
    joined_path = tmp_path / "joined_shadow_predictions.jsonl"
    odds_path = tmp_path / "shadow_odds_snapshot.jsonl"
    gate_path = tmp_path / "odds_research_gate_report.json"
    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525"
        / "odds_augmented_challenger_20260610T010000+0000"
    )
    joined_rows = []
    odds_rows = []
    for index in range(30):
        race_id = f"Race {index + 1}"
        box1_wins = index % 2 == 0
        market_favors_winner = index < 20
        for box in (1, 2):
            dog_name = f"Dog {index + 1}-{box}"
            is_winner = (box == 1 and box1_wins) or (box == 2 and not box1_wins)
            is_market_favorite = is_winner if market_favors_winner else not is_winner
            joined_rows.append(
                {
                    "race_id": race_id,
                    "box": box,
                    "dog_name": dog_name,
                    "is_winner": is_winner,
                    "shadow_rf_calibrated_probability": 0.6 if box == 1 else 0.4,
                }
            )
            odds_rows.append(
                {
                    "race_id": race_id,
                    "box": box,
                    "dog_name": dog_name,
                    "odds_match_status": "valid_pre_jump_dog_odds",
                    "odds_snapshot": {
                        "market_odds_win": 2.0 if is_market_favorite else 10.0
                    },
                }
            )
    _write_jsonl(joined_path, joined_rows)
    _write_jsonl(odds_path, odds_rows)
    gate_path.write_text(
        json.dumps(
            {
                "status": odds_aug.ODDS_RESEARCH_READY_REPORT_ONLY,
                "complete_valid_prejump_odds_races": 100,
                "source_url_rows_missing": 0,
                "source_url_coverage_pct": 100.0,
            }
        ),
        encoding="utf-8",
    )

    result = odds_aug.run_odds_augmented_report(
        joined_predictions=joined_path,
        odds_snapshot=odds_path,
        odds_gate_report_path=gate_path,
        output_dir=output_dir,
        max_box1_top_pick_share=0.6,
    )

    assert result["final_status"] == odds_aug.ODDS_AUGMENTED_MODEL_READY_FOR_PR_REVIEW
    assert result["best_rank_accuracy_candidate"] in {
        "market_only_implied_probability_baseline",
        "odds_augmented_challenger",
        "probability_blend_calibration_candidate",
    }
    report = json.loads((output_dir / "odds_augmented_challenger_report.json").read_text())
    ev = json.loads((output_dir / "report_only_ev_diagnostics.json").read_text())
    assert report["promotion_boundary"]["direct_registry_mutation_allowed"] is False
    assert report["promotion_boundary"]["ev_can_override_failed_accuracy_gate"] is False
    assert report["odds_used_for_shadow_scoring"] is False
    assert report["betting_action_allowed"] is False
    assert report["no_write_guarantees"]["db_write"] is False
    assert ev["status"] == "EV_DIAGNOSTICS_REPORT_ONLY"
    assert ev["betting_advice"] is False
    assert ev["stakes"] is False
    assert ev["betting_action_allowed"] is False


def test_odds_augmented_report_blocks_before_odds_gate_ready():
    report, ev = odds_aug.build_report(
        joined_rows=[],
        odds_rows=[],
        odds_gate_report={"status": "ODDS_RESEARCH_BLOCKED_PROVENANCE"},
        protected_before={},
        protected_after={},
    )

    assert report["final_status"] == odds_aug.ODDS_AUGMENTED_MODEL_BLOCKED
    assert "odds_research_gate_not_ready" in report["activation_blockers"]
    assert report["promotion_boundary"]["promotion_pr_allowed"] is False
    assert ev["betting_action_allowed"] is False
