import importlib.util
from pathlib import Path


PATH = Path(__file__).parents[1] / "scripts/audit_fast_nonfavourite_mechanism.py"
SPEC = importlib.util.spec_from_file_location("fast_nonfavourite", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def history_row(day, track="MEAD", distance=525, elapsed=30.0, source_row_id=1):
    return {
        "source_performance_date": day,
        "source_track": track,
        "source_distance_m": distance,
        "individual_time_seconds": elapsed,
        "source_row_id": source_row_id,
    }


def test_history_times_enforces_exact_context_recency_and_last_five():
    rows = [
        history_row("2025-06-30", elapsed=99, source_row_id=1),
        history_row("2025-07-01", track="SAND", elapsed=99, source_row_id=2),
        history_row("2025-07-02", distance=600, elapsed=99, source_row_id=3),
        history_row("2026-07-01", elapsed=29.9, source_row_id=4),
        history_row("2026-07-02", elapsed=29.8, source_row_id=5),
        history_row("2026-07-03", elapsed=29.7, source_row_id=6),
        history_row("2026-07-04", elapsed=29.6, source_row_id=7),
        history_row("2026-07-05", elapsed=29.5, source_row_id=8),
        history_row("2026-07-06", elapsed=29.4, source_row_id=9),
        history_row("2026-07-10", elapsed=10, source_row_id=10),
    ]
    got = MODULE.history_times(rows, cutoff="2026-07-10", track="MEAD", distance_m=525)
    assert [row["source_row_id"] for row in got] == [5, 6, 7, 8, 9]


def test_history_times_excludes_target_day_and_invalid_times():
    rows = [
        history_row("2026-07-09", elapsed=0, source_row_id=1),
        history_row("2026-07-09", elapsed=29.5, source_row_id=2),
        history_row("2026-07-10", elapsed=29.0, source_row_id=3),
    ]
    got = MODULE.history_times(rows, cutoff="2026-07-10", track="MEAD", distance_m=525)
    assert [row["source_row_id"] for row in got] == [2]


def test_folds_are_disjoint_and_before_forward_boundary():
    assert MODULE.FOLDS[0]["end"] < MODULE.FOLDS[1]["start"]
    assert MODULE.FOLDS[1]["end"] < MODULE.FOLDS[2]["start"]
    assert MODULE.FOLDS[-1]["end"] < MODULE.FORWARD_START


def test_empty_group_is_explicit_not_nan():
    result = MODULE.empty_group("B_MARKET_RANK_2")
    assert result["count"] == 0
    assert result["calibration_residual_mean"] is None
    assert result["economic"]["roi"] is None
