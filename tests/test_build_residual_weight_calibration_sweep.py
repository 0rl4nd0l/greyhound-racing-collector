import csv
from pathlib import Path

import pytest

from scripts.build_residual_weight_calibration_sweep import (
    FINAL_BLOCKED,
    FINAL_DATA_MISSING,
    FINAL_SEGMENT_DESIGN_ONLY,
    build_report,
)


FIELDS = [
    "candidate_key",
    "market_candidate_key",
    "race_id",
    "venue",
    "race_number",
    "race_date",
    "dog_name",
    "box_number",
    "is_winner",
    "finish_position",
    "market_probability",
    "candidate_probability",
    "runner_count",
]


def _write_matrix(path: Path, rows: list[dict[str, object]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            base = {
                "candidate_key": "stage2_uncalibrated_market_blend_25",
                "market_candidate_key": "market_only_implied",
                "venue": "HEA",
                "race_number": 1,
                "race_date": "2026-07-08",
                "runner_count": 2,
            }
            base.update(row)
            writer.writerow(base)
    return path


def _race_rows(
    race_id: str,
    *,
    winner_market_probability: float,
    winner_candidate_probability: float,
    race_date: str = "2026-07-08",
    venue: str = "HEA",
) -> list[dict[str, object]]:
    return [
        {
            "race_id": race_id,
            "race_date": race_date,
            "venue": venue,
            "dog_name": f"{race_id} winner",
            "box_number": 1,
            "is_winner": True,
            "finish_position": 1,
            "market_probability": winner_market_probability,
            "candidate_probability": winner_candidate_probability,
        },
        {
            "race_id": race_id,
            "race_date": race_date,
            "venue": venue,
            "dog_name": f"{race_id} loser",
            "box_number": 2,
            "is_winner": False,
            "finish_position": 2,
            "market_probability": 1.0 - winner_market_probability,
            "candidate_probability": 1.0 - winner_candidate_probability,
        },
    ]


def _build(
    tmp_path: Path,
    freeze_rows: list[dict[str, object]],
    oos_rows: list[dict[str, object]],
    *,
    output_dir: Path | None = None,
    min_oos_races: int = 2,
) -> dict[str, object]:
    freeze = _write_matrix(tmp_path / "freeze.csv", freeze_rows)
    oos = _write_matrix(tmp_path / "oos.csv", oos_rows)
    return build_report(
        freeze_runner_matrix_csv=freeze,
        oos_runner_matrix_csv=oos,
        candidate_key="stage2_uncalibrated_market_blend_25",
        weights=(0.0, 0.25),
        movement_caps=(None,),
        modes=("linear_residual",),
        output_dir=output_dir,
        min_oos_races=min_oos_races,
        promotion_review_races=4,
        min_race_dates_for_stability=1,
        min_venues_for_stability=1,
    )


def test_selects_candidate_from_freeze_then_validates_oos(tmp_path: Path) -> None:
    freeze_rows = [
        *_race_rows(
            "Freeze A",
            winner_market_probability=0.49,
            winner_candidate_probability=0.75,
        ),
        *_race_rows(
            "Freeze B",
            winner_market_probability=0.49,
            winner_candidate_probability=0.75,
        ),
    ]
    oos_rows = [
        *freeze_rows,
        *_race_rows(
            "OOS A",
            winner_market_probability=0.49,
            winner_candidate_probability=0.75,
            race_date="2026-07-09",
        ),
        *_race_rows(
            "OOS B",
            winner_market_probability=0.49,
            winner_candidate_probability=0.75,
            race_date="2026-07-10",
            venue="WAR",
        ),
    ]

    report = _build(tmp_path, freeze_rows, oos_rows)

    assert report["final_status"] == FINAL_SEGMENT_DESIGN_ONLY
    assert report["oos_race_count"] == 2
    assert report["freeze_selected_candidate"]["candidate_key"] == (
        "linear_residual_w0_25_cap_uncapped"
    )
    assert report["oos_validation"]["candidate_minus_market"]["top1"] == 1.0
    assert report["frozen_candidate_manifest"]["selection_rule"] == (
        "freeze_metrics_only; oos_metrics_not_used_for_selection"
    )


def test_blocks_when_freeze_selected_candidate_fails_oos(tmp_path: Path) -> None:
    freeze_rows = [
        *_race_rows(
            "Freeze A",
            winner_market_probability=0.49,
            winner_candidate_probability=0.75,
        ),
        *_race_rows(
            "Freeze B",
            winner_market_probability=0.49,
            winner_candidate_probability=0.75,
        ),
    ]
    oos_rows = [
        *freeze_rows,
        *_race_rows(
            "OOS A",
            winner_market_probability=0.75,
            winner_candidate_probability=0.49,
            race_date="2026-07-09",
        ),
        *_race_rows(
            "OOS B",
            winner_market_probability=0.75,
            winner_candidate_probability=0.49,
            race_date="2026-07-10",
        ),
    ]

    report = _build(tmp_path, freeze_rows, oos_rows)

    assert report["final_status"] == FINAL_BLOCKED
    assert "brier_delta_failed" in report["oos_validation"]["blockers"]
    assert "logloss_delta_failed" in report["oos_validation"]["blockers"]


def test_missing_candidate_probability_fails_closed_as_data_missing(tmp_path: Path) -> None:
    freeze_rows = _race_rows(
        "Freeze A",
        winner_market_probability=0.49,
        winner_candidate_probability=0.75,
    )
    oos_rows = _race_rows(
        "OOS A",
        winner_market_probability=0.49,
        winner_candidate_probability=0.75,
    )
    oos_rows[0]["candidate_probability"] = ""

    report = _build(tmp_path, freeze_rows, oos_rows, min_oos_races=1)

    assert report["final_status"] == FINAL_DATA_MISSING
    assert report["oos_race_count"] == 0


def test_writes_report_artifacts_and_calibration_bins(tmp_path: Path) -> None:
    packet_dir = tmp_path / "packet"
    packet_dir.mkdir()
    freeze_rows = [
        *_race_rows(
            "Freeze A",
            winner_market_probability=0.49,
            winner_candidate_probability=0.75,
        )
    ]
    oos_rows = [
        *freeze_rows,
        *_race_rows(
            "OOS A",
            winner_market_probability=0.49,
            winner_candidate_probability=0.75,
        ),
    ]
    output_dir = tmp_path / "report"

    report = build_report(
        freeze_runner_matrix_csv=_write_matrix(packet_dir / "freeze.csv", freeze_rows),
        oos_runner_matrix_csv=_write_matrix(packet_dir / "oos.csv", oos_rows),
        candidate_key="stage2_uncalibrated_market_blend_25",
        weights=(0.0, 0.25),
        movement_caps=(None,),
        modes=("linear_residual",),
        output_dir=output_dir,
        min_oos_races=1,
        promotion_review_races=4,
        min_race_dates_for_stability=1,
        min_venues_for_stability=1,
    )

    assert report["calibration_bin_count"] > 0
    assert (output_dir / "residual_weight_calibration_sweep_report.json").exists()
    assert (output_dir / "candidate_metrics.csv").exists()
    assert (output_dir / "calibration_bins.csv").exists()
    assert (output_dir / "frozen_candidate_manifest.json").exists()


def test_rejects_output_inside_input_packet_directory(tmp_path: Path) -> None:
    packet_dir = tmp_path / "rolling_model_comparison_20260707T010000+1000_daemon_autopilot"
    packet_dir.mkdir()
    rows = _race_rows(
        "Race A",
        winner_market_probability=0.49,
        winner_candidate_probability=0.75,
    )
    matrix = _write_matrix(packet_dir / "market_residual_runner_matrix.csv", rows)

    with pytest.raises(ValueError, match="output_dir_must_not_write_inside_input_packet"):
        build_report(
            freeze_runner_matrix_csv=matrix,
            oos_runner_matrix_csv=matrix,
            candidate_key="stage2_uncalibrated_market_blend_25",
            output_dir=packet_dir / "sweep",
            min_oos_races=1,
        )


def test_rejects_protected_output_when_invoked_outside_repo(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / ".git").mkdir(parents=True)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    rows = _race_rows(
        "Race A",
        winner_market_probability=0.49,
        winner_candidate_probability=0.75,
    )
    matrix = _write_matrix(input_dir / "matrix.csv", rows)

    with pytest.raises(ValueError, match="output_dir_protected"):
        build_report(
            freeze_runner_matrix_csv=matrix,
            oos_runner_matrix_csv=matrix,
            candidate_key="stage2_uncalibrated_market_blend_25",
            output_dir=repo_root / "artifacts/full_evidence_orchestration_20260525/sweep",
            min_oos_races=1,
        )
