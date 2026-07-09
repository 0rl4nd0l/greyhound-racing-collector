import csv
import json
import os
from pathlib import Path

import pytest

from scripts.build_frozen_oos_gate_evaluator import (
    FINAL_BLOCKED,
    FINAL_DATA_MISSING,
    FINAL_SEGMENT_DESIGN_ONLY,
    build_grid_report,
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
    "market_favourite_odds_band",
    "market_probability",
    "candidate_probability",
    "market_rank",
    "candidate_rank",
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
                "market_favourite_odds_band": "market_favourite_odds_lte_2",
                "runner_count": 7,
            }
            base.update(row)
            writer.writerow(base)
    return path


def _write_packet(
    root: Path,
    name: str,
    rows: list[dict[str, object]],
    *,
    generated_at: str,
) -> Path:
    packet_dir = root / name
    packet_dir.mkdir()
    _write_matrix(packet_dir / "market_residual_runner_matrix.csv", rows)
    (packet_dir / "rolling_model_comparison_report.json").write_text(
        json.dumps({"generated_at": generated_at}) + "\n",
        encoding="utf-8",
    )
    return packet_dir


def _race_rows(
    race_id: str,
    *,
    winner_market_probability: float,
    winner_candidate_probability: float,
    winner_market_rank: int,
    winner_candidate_rank: int,
    race_date: str = "2026-07-08",
    venue: str = "HEA",
) -> list[dict[str, object]]:
    loser_market_probability = 1.0 - winner_market_probability
    loser_candidate_probability = 1.0 - winner_candidate_probability
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
            "market_rank": winner_market_rank,
            "candidate_rank": winner_candidate_rank,
        },
        {
            "race_id": race_id,
            "race_date": race_date,
            "venue": venue,
            "dog_name": f"{race_id} loser",
            "box_number": 2,
            "is_winner": False,
            "finish_position": 2,
            "market_probability": loser_market_probability,
            "candidate_probability": loser_candidate_probability,
            "market_rank": 1 if winner_market_rank == 2 else 2,
            "candidate_rank": 1 if winner_candidate_rank == 2 else 2,
        },
    ]


def _build(
    tmp_path: Path,
    freeze_rows: list[dict[str, object]],
    oos_rows: list[dict[str, object]],
    *,
    min_oos_races: int = 2,
) -> dict[str, object]:
    freeze_path = _write_matrix(tmp_path / "freeze.csv", freeze_rows)
    oos_path = _write_matrix(tmp_path / "oos.csv", oos_rows)
    return build_report(
        freeze_runner_matrix_csv=freeze_path,
        oos_runner_matrix_csv=oos_path,
        gate_id="runner_count7_oos_gate_v1",
        selector_field="runner_count",
        selector_operator="eq",
        selector_value=7,
        candidate_key="stage2_uncalibrated_market_blend_25",
        min_oos_races=min_oos_races,
        stability_races=3,
        promotion_review_races=4,
    )


def test_excludes_freeze_race_ids_before_sample_floor(tmp_path: Path) -> None:
    freeze_rows = _race_rows(
        "Race A",
        winner_market_probability=0.4,
        winner_candidate_probability=0.7,
        winner_market_rank=2,
        winner_candidate_rank=1,
    )
    oos_rows = [
        *freeze_rows,
        *_race_rows(
            "Race B",
            winner_market_probability=0.4,
            winner_candidate_probability=0.7,
            winner_market_rank=2,
            winner_candidate_rank=1,
        ),
    ]

    report = _build(tmp_path, freeze_rows, oos_rows, min_oos_races=2)

    assert report["final_status"] == FINAL_DATA_MISSING
    assert report["oos_selected_race_count"] == 1
    assert report["metrics"]["skipped_counts"]["freeze_race_id_excluded"] == 1
    assert "oos_race_count_below_floor" in report["decision"]["blockers"]


def test_blocks_when_hard_probability_gates_fail(tmp_path: Path) -> None:
    freeze_rows = _race_rows(
        "Race A",
        winner_market_probability=0.4,
        winner_candidate_probability=0.7,
        winner_market_rank=2,
        winner_candidate_rank=1,
    )
    oos_rows = [
        *_race_rows(
            "Race B",
            winner_market_probability=0.7,
            winner_candidate_probability=0.4,
            winner_market_rank=1,
            winner_candidate_rank=2,
        ),
        *_race_rows(
            "Race C",
            winner_market_probability=0.7,
            winner_candidate_probability=0.4,
            winner_market_rank=1,
            winner_candidate_rank=2,
        ),
    ]

    report = _build(tmp_path, freeze_rows, oos_rows, min_oos_races=2)

    assert report["final_status"] == FINAL_BLOCKED
    assert report["decision"]["hard_gates_pass"] is False
    assert "brier_delta_failed" in report["decision"]["blockers"]
    assert "logloss_delta_failed" in report["decision"]["blockers"]


def test_tie_with_market_passes_hard_gates_but_fails_materiality(
    tmp_path: Path,
) -> None:
    freeze_rows = _race_rows(
        "Race A",
        winner_market_probability=0.4,
        winner_candidate_probability=0.7,
        winner_market_rank=2,
        winner_candidate_rank=1,
    )
    oos_rows = _race_rows(
        "Race B",
        winner_market_probability=0.4,
        winner_candidate_probability=0.4,
        winner_market_rank=1,
        winner_candidate_rank=1,
    )

    report = _build(tmp_path, freeze_rows, oos_rows, min_oos_races=1)

    assert report["final_status"] == FINAL_BLOCKED
    assert report["decision"]["hard_gates_pass"] is True
    assert report["decision"]["materiality_gates_pass"] is False
    assert report["metrics"]["candidate_minus_market"]["brier"] == 0.0
    assert report["metrics"]["candidate_minus_market"]["logloss"] == 0.0
    assert "brier_delta_failed" not in report["decision"]["blockers"]
    assert "logloss_delta_failed" not in report["decision"]["blockers"]


def test_passes_hard_and_materiality_gates_as_design_only_below_promotion_floor(
    tmp_path: Path,
) -> None:
    freeze_rows = _race_rows(
        "Race A",
        winner_market_probability=0.4,
        winner_candidate_probability=0.7,
        winner_market_rank=2,
        winner_candidate_rank=1,
    )
    oos_rows = [
        *_race_rows(
            "Race B",
            winner_market_probability=0.4,
            winner_candidate_probability=0.7,
            winner_market_rank=2,
            winner_candidate_rank=1,
            race_date="2026-07-08",
            venue="HEA",
        ),
        *_race_rows(
            "Race C",
            winner_market_probability=0.4,
            winner_candidate_probability=0.7,
            winner_market_rank=2,
            winner_candidate_rank=1,
            race_date="2026-07-09",
            venue="WAR",
        ),
    ]

    report = _build(tmp_path, freeze_rows, oos_rows, min_oos_races=2)

    assert report["final_status"] == FINAL_SEGMENT_DESIGN_ONLY
    assert report["decision"]["hard_gates_pass"] is True
    assert report["decision"]["materiality_gates_pass"] is True
    assert report["decision"]["promotion_review_floor_met"] is False
    assert report["metrics"]["candidate_minus_market"]["top1"] == 1.0
    assert report["metrics"]["winner_movements"]["candidate_promoted_winner_count"] == 2


def test_grid_report_evaluates_source_safe_freeze_to_latest_gate(tmp_path: Path) -> None:
    packet_root = tmp_path / "packets"
    packet_root.mkdir()
    freeze_rows = _race_rows(
        "Race A",
        winner_market_probability=0.4,
        winner_candidate_probability=0.7,
        winner_market_rank=2,
        winner_candidate_rank=1,
    )
    latest_rows = [
        *freeze_rows,
        *_race_rows(
            "Race B",
            winner_market_probability=0.7,
            winner_candidate_probability=0.4,
            winner_market_rank=1,
            winner_candidate_rank=2,
        ),
    ]
    _write_packet(
        packet_root,
        "rolling_model_comparison_20260707T010000+1000_daemon_autopilot",
        freeze_rows,
        generated_at="2026-07-07T01:00:00+10:00",
    )
    _write_packet(
        packet_root,
        "rolling_model_comparison_20260707T020000+1000_daemon_autopilot",
        latest_rows,
        generated_at="2026-07-07T02:00:00+10:00",
    )

    report = build_grid_report(
        packet_root=packet_root,
        packet_name_regex=r"^rolling_model_comparison_20260707T\d{6}\+1000_daemon_autopilot$",
        selector_fields=("runner_count",),
        candidate_key="stage2_uncalibrated_market_blend_25",
        min_oos_races=1,
        stability_races=2,
        promotion_review_races=3,
    )

    assert report["final_status"] == FINAL_BLOCKED
    assert report["packet_count"] == 2
    assert report["evaluated_gate_count"] == 1
    assert report["eligible_gate_count"] == 1
    assert report["status_counts"] == {FINAL_BLOCKED: 1}
    assert report["gate_results"][0]["selected_race_count"] == 1
    assert report["gate_results"][0]["selector_field"] == "runner_count"
    assert report["gate_results"][0]["selector_value"] == 7


def test_rejects_rank_or_result_derived_selector(tmp_path: Path) -> None:
    freeze_rows = _race_rows(
        "Race A",
        winner_market_probability=0.4,
        winner_candidate_probability=0.7,
        winner_market_rank=2,
        winner_candidate_rank=1,
    )
    freeze_path = _write_matrix(tmp_path / "freeze.csv", freeze_rows)
    oos_path = _write_matrix(tmp_path / "oos.csv", freeze_rows)

    with pytest.raises(ValueError, match="selector_field_not_source_safe"):
        build_report(
            freeze_runner_matrix_csv=freeze_path,
            oos_runner_matrix_csv=oos_path,
            gate_id="bad_gate",
            selector_field="candidate_rank",
            selector_operator="eq",
            selector_value=1,
            candidate_key="stage2_uncalibrated_market_blend_25",
        )


def test_missing_candidate_or_market_key_blocks_as_data_missing(tmp_path: Path) -> None:
    freeze_rows = _race_rows(
        "Race A",
        winner_market_probability=0.4,
        winner_candidate_probability=0.7,
        winner_market_rank=2,
        winner_candidate_rank=1,
    )
    oos_rows = _race_rows(
        "Race B",
        winner_market_probability=0.4,
        winner_candidate_probability=0.7,
        winner_market_rank=2,
        winner_candidate_rank=1,
    )
    oos_rows[0]["candidate_key"] = ""
    oos_rows[1]["market_candidate_key"] = ""

    report = _build(tmp_path, freeze_rows, oos_rows, min_oos_races=1)

    assert report["final_status"] == FINAL_DATA_MISSING
    assert "candidate_key_mismatch_requires_new_freeze" in report["decision"]["blockers"]
    assert "market_candidate_key_mismatch" in report["decision"]["blockers"]


def test_rejects_output_inside_input_packet_directory(tmp_path: Path) -> None:
    packet_dir = tmp_path / "rolling_model_comparison_20260707T155535+1000_daemon_autopilot"
    packet_dir.mkdir()
    rows = _race_rows(
        "Race A",
        winner_market_probability=0.4,
        winner_candidate_probability=0.7,
        winner_market_rank=2,
        winner_candidate_rank=1,
    )
    matrix_path = _write_matrix(packet_dir / "market_residual_runner_matrix.csv", rows)

    with pytest.raises(ValueError, match="output_dir_must_not_write_inside_input_packet"):
        build_report(
            freeze_runner_matrix_csv=matrix_path,
            oos_runner_matrix_csv=matrix_path,
            gate_id="bad_output",
            selector_field="runner_count",
            selector_operator="eq",
            selector_value=7,
            candidate_key="stage2_uncalibrated_market_blend_25",
            output_dir=packet_dir / "new_report",
        )


def test_rejects_protected_output_when_invoked_outside_repo(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / ".git").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    rows = _race_rows(
        "Race A",
        winner_market_probability=0.4,
        winner_candidate_probability=0.7,
        winner_market_rank=2,
        winner_candidate_rank=1,
    )
    matrix_path = _write_matrix(tmp_path / "market_residual_runner_matrix.csv", rows)
    protected_output = repo_root / "artifacts" / "full_evidence_orchestration_20260525" / "gate"
    original_cwd = Path.cwd()

    try:
        os.chdir(outside)
        with pytest.raises(ValueError, match="output_dir_protected"):
            build_report(
                freeze_runner_matrix_csv=matrix_path,
                oos_runner_matrix_csv=matrix_path,
                gate_id="bad_output",
                selector_field="runner_count",
                selector_operator="eq",
                selector_value=7,
                candidate_key="stage2_uncalibrated_market_blend_25",
                output_dir=protected_output,
            )
    finally:
        os.chdir(original_cwd)


def test_grid_rejects_output_inside_packet_directory(tmp_path: Path) -> None:
    packet_root = tmp_path / "packets"
    packet_root.mkdir()
    rows = _race_rows(
        "Race A",
        winner_market_probability=0.4,
        winner_candidate_probability=0.7,
        winner_market_rank=2,
        winner_candidate_rank=1,
    )
    _write_packet(
        packet_root,
        "rolling_model_comparison_20260707T010000+1000_daemon_autopilot",
        rows,
        generated_at="2026-07-07T01:00:00+10:00",
    )
    latest_packet = _write_packet(
        packet_root,
        "rolling_model_comparison_20260707T020000+1000_daemon_autopilot",
        rows,
        generated_at="2026-07-07T02:00:00+10:00",
    )

    with pytest.raises(ValueError, match="output_dir_must_not_write_inside_input_packet"):
        build_grid_report(
            packet_root=packet_root,
            packet_name_regex=r"^rolling_model_comparison_20260707T\d{6}\+1000_daemon_autopilot$",
            selector_fields=("runner_count",),
            candidate_key="stage2_uncalibrated_market_blend_25",
            output_dir=latest_packet / "grid",
        )
