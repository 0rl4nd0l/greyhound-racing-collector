import json
from datetime import datetime, timezone
from pathlib import Path

from scripts import run_forward_shadow_challenger_calibration as challenger


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _runner(
    race_id: str,
    box: int,
    dog_name: str,
    probability: float,
    *,
    is_winner: bool = False,
    identity_status: str = "exact_box_and_normalized_name",
) -> dict:
    return {
        "race_id": race_id,
        "race_date": "2026-06-08",
        "venue": "TEST",
        "race_number": int(race_id.rsplit(" ", 1)[-1]),
        "box": box,
        "dog_name": dog_name,
        "identity_match_status": identity_status,
        "is_winner": is_winner,
        "shadow_rf_calibrated_probability": probability,
        "shadow_rf_uncalibrated_probability": probability,
        "calibration_method": "power_gamma_2.4",
        "tgr_enabled": False,
    }


def _safe_race(race_id: str, *, winner_box: int = 1) -> list[dict]:
    return [
        _runner(race_id, 1, f"{race_id} Alpha", 0.60, is_winner=winner_box == 1),
        _runner(race_id, 2, f"{race_id} Bravo", 0.25, is_winner=winner_box == 2),
        _runner(race_id, 3, f"{race_id} Charlie", 0.15, is_winner=winner_box == 3),
    ]


def _join_dir(evidence_root: Path, name: str, source_shadow_run: str, rows: list[dict]) -> Path:
    join_dir = evidence_root / name
    _write_json(
        join_dir / "shadow_forward_metrics.json",
        {
            "schema_version": "forward_shadow_metrics_v1",
            "source_shadow_run": source_shadow_run,
        },
    )
    _write_jsonl(join_dir / "joined_shadow_predictions.jsonl", rows)
    return join_dir


def test_challenger_uses_only_exact_joined_races_and_writes_report_only(tmp_path, monkeypatch):
    monkeypatch.setattr(challenger, "ROOT", tmp_path)
    monkeypatch.setattr(challenger, "DEFAULT_PROTECTED_PATHS", ())
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    safe_rows = _safe_race("Race 1")
    safe_rows.extend(_safe_race("Race 2", winner_box=2))
    unsafe_rows = [
        _runner(
            "Race 3",
            1,
            "Race 3 Alpha",
            0.70,
            is_winner=True,
            identity_status="fuzzy_name_only",
        ),
        _runner("Race 3", 2, "Race 3 Bravo", 0.30),
    ]
    _join_dir(
        evidence_root,
        "forward_shadow_result_join_20260608T120000+1000",
        "daily_race_ingest_shadow_a",
        safe_rows + unsafe_rows,
    )
    output_dir = (
        evidence_root / "forward_shadow_challenger_calibration_20260608T120100+1000"
    )

    result = challenger.run_challenger_calibration(
        evidence_root=evidence_root,
        output_dir=output_dir,
        alpha_grid=(1.0,),
        thresholds=challenger.ChallengerThresholds(
            min_total_safe_joined_races=2,
            min_train_races=1,
            min_eval_races=1,
            train_fraction=0.5,
        ),
    )

    assert result["final_status"] == challenger.FINAL_READY
    report = json.loads((output_dir / "challenger_calibration_report.json").read_text())
    assert report["safe_exact_joined_race_count"] == 2
    assert report["production_activation_allowed"] is False
    assert report["no_write_guarantees"]["db_write"] is False
    assert report["no_write_guarantees"]["registry_mutation"] is False
    assert report["rejected_joined_races"] == [
        {
            "race_id": "Race 3",
            "reasons": ["non_exact_identity_match_status"],
            "row_count": 2,
        }
    ]
    predictions = [
        json.loads(line)
        for line in (output_dir / "challenger_predictions_report_only.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["race_id"] for row in predictions} == {"Race 1", "Race 2"}
    assert all(
        row["identity_match_status"] == "exact_box_and_normalized_name"
        for row in predictions
    )


def test_challenger_blocks_activation_when_sample_is_below_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(challenger, "ROOT", tmp_path)
    monkeypatch.setattr(challenger, "DEFAULT_PROTECTED_PATHS", ())
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    _join_dir(
        evidence_root,
        "forward_shadow_result_join_20260608T120000+1000",
        "daily_race_ingest_shadow_a",
        _safe_race("Race 1") + _safe_race("Race 2", winner_box=2),
    )

    report, _ = challenger.build_report(
        evidence_root=evidence_root,
        input_probability_key="shadow_rf_calibrated_probability",
        alpha_grid=(1.0,),
        thresholds=challenger.ChallengerThresholds(
            min_total_safe_joined_races=3,
            min_train_races=2,
            min_eval_races=2,
            train_fraction=0.5,
        ),
        generated_at=datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc),
        protected_before={},
        protected_after={},
    )

    assert report["final_status"] == challenger.FINAL_BLOCKED
    assert "safe_joined_race_count_below_min_total" in report["activation_blockers"]
    assert "train_race_count_below_min" in report["activation_blockers"]
    assert "eval_race_count_below_min" in report["activation_blockers"]
    assert report["report_only"] is True
