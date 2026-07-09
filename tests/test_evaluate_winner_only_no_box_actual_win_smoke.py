import json
from pathlib import Path

import pytest

from scripts.evaluate_winner_only_no_box_actual_win_smoke import (
    evaluate_smoke_packet,
    write_outputs,
)


def _packet(can_evaluate: bool = True) -> dict:
    return {
        "schema_version": "winner_only_no_box_actual_win_rehearsal_v1",
        "report_only": True,
        "summary": {
            "can_evaluate_model": can_evaluate,
            "no_box_row_policy_pass": True,
        },
    }


def _row(race: int, dog: str, actual_win: int, *, field_complete: bool = False) -> dict:
    return {
        "race_id": f"R{race}",
        "legacy_race_id": f"R{race}",
        "identity_key": f"2025-01-0{race}|TEST|R0{race}",
        "race_date": f"2025-01-0{race}",
        "venue": "TEST",
        "race_number": race,
        "dog_name_key": dog.lower(),
        "dog_name": dog,
        "actual_win": actual_win,
        "candidate_kind": "complete_field" if field_complete else "partial_field",
        "field_scope": "complete_name_set_box_drift" if field_complete else "partial_db_name_subset_of_official_finishers",
        "field_complete_for_ranking": field_complete,
        "race_grouped_actual_win_ranking_allowed": field_complete,
        "target_source": "official_winner_name_metadata_confirmed",
        "label_scope": "actual_win_only",
        "box_features_allowed": False,
        "finish_order_labels_allowed": False,
        "top3_labels_allowed": False,
        "official_safe_label_candidate": False,
        "label_write_approved": False,
    }


def _rows() -> list[dict]:
    return [
        _row(1, "Alpha", 1, field_complete=True),
        _row(1, "Bravo", 0, field_complete=True),
        _row(2, "Alpha", 0),
        _row(2, "Charlie", 1),
        _row(3, "Alpha", 1),
        _row(3, "Delta", 0),
    ]


def test_smoke_eval_scores_baselines_without_training_or_box_fields(
    tmp_path: Path,
    monkeypatch,
):
    import scripts.evaluate_winner_only_no_box_actual_win_smoke as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    report, predictions = evaluate_smoke_packet(
        rehearsal_packet=_packet(),
        rows=_rows(),
        expected_races=3,
        min_smoke_races=3,
    )

    assert report["status"] == "REPORT_ONLY_NO_BOX_ACTUAL_WIN_SMOKE_EVALUATED"
    assert report["writes_performed"]["model_training"] is False
    assert report["writes_performed"]["label_write"] is False
    assert report["validation"]["status"] == "PASS"
    assert report["validation"]["race_count"] == 3
    assert report["validation"]["complete_field_races"] == 1
    assert report["validation"]["partial_field_races"] == 2
    assert report["feature_model_status"] == "SKIPPED_NO_PREDICTIVE_FEATURE_COLUMNS_IN_REHEARSAL_ROWS"
    assert report["race_grouped_ranking_status"] == "SKIPPED_INSUFFICIENT_COMPLETE_FIELD_RACES"
    assert set(report["baselines"]) == {"uniform_field", "rolling_dog_name_prior"}
    assert report["baselines"]["uniform_field"]["top1_accuracy"] == 2 / 3
    assert report["baselines"]["uniform_field"]["top3_hit_rate"] == 1.0
    assert report["baselines"]["uniform_field"]["brier"] == 0.25
    assert len(predictions) == 12
    assert all("box_number" not in row for row in predictions)

    output_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/smoke"
    write_outputs(output_dir, report, predictions)
    assert (output_dir / "no_box_actual_win_smoke_eval_report.json").exists()
    written = json.loads((output_dir / "no_box_actual_win_smoke_eval_report.json").read_text())
    assert written["schema_version"] == "winner_only_no_box_actual_win_smoke_eval_v1"
    assert (output_dir / "no_box_actual_win_smoke_predictions.jsonl").exists()
    assert (output_dir / "no_box_actual_win_smoke_predictions.csv").exists()
    assert (output_dir / "SUMMARY.md").exists()

    cwd = tmp_path / "caller_cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    relative_output_dir = Path(
        "artifacts/full_evidence_orchestration_20260525/relative_smoke"
    )
    write_outputs(relative_output_dir, report, predictions)
    assert (
        tmp_path / relative_output_dir / "no_box_actual_win_smoke_eval_report.json"
    ).exists()
    assert not (
        cwd / relative_output_dir / "no_box_actual_win_smoke_eval_report.json"
    ).exists()


def test_smoke_eval_output_guard_fails_closed(tmp_path: Path, monkeypatch):
    import scripts.evaluate_winner_only_no_box_actual_win_smoke as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    report, predictions = evaluate_smoke_packet(
        rehearsal_packet=_packet(),
        rows=_rows(),
        expected_races=3,
        min_smoke_races=3,
    )

    with pytest.raises(ValueError, match="output_dir_must_be_under_artifacts"):
        write_outputs(tmp_path / "outside", report, predictions)
    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        write_outputs(
            tmp_path.parent
            / "outside"
            / "artifacts/full_evidence_orchestration_20260525/smoke",
            report,
            predictions,
        )


def test_smoke_eval_fails_closed_on_forbidden_box_field():
    rows = _rows()
    rows[0]["box_number"] = 1

    report, _ = evaluate_smoke_packet(
        rehearsal_packet=_packet(),
        rows=rows,
        expected_races=3,
        min_smoke_races=3,
    )

    assert report["status"] == "REPORT_ONLY_NO_BOX_ACTUAL_WIN_SMOKE_FAILED_CONTRACT"
    assert report["validation"]["status"] == "FAIL"
    assert any("forbidden_fields_present:box_number" in item for item in report["validation"]["failures"])


def test_smoke_eval_adds_dog_form_heuristic_when_feature_columns_exist():
    rows = _rows()
    for row in rows:
        row["feature_recent_win_rate_5"] = 1.0 if row["actual_win"] else 0.0
        row["feature_recent_place_rate_5"] = 1.0 if row["actual_win"] else 0.0
        row["feature_prior_start_count"] = 3

    report, predictions = evaluate_smoke_packet(
        rehearsal_packet=_packet(),
        rows=rows,
        expected_races=3,
        min_smoke_races=3,
    )

    assert report["validation"]["status"] == "PASS"
    assert report["feature_model_status"] == "READY_FOR_FEATURE_MODEL"
    assert "dog_form_heuristic" in report["baselines"]
    assert report["baselines"]["dog_form_heuristic"]["top1_accuracy"] == 1.0
    heuristic_predictions = [row for row in predictions if row["baseline"] == "dog_form_heuristic"]
    assert len(heuristic_predictions) == len(rows)
    assert all("box_number" not in row for row in heuristic_predictions)
