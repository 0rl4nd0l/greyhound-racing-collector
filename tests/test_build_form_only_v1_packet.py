from __future__ import annotations

import importlib.util
import json
from datetime import date
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_form_only_v1_packet.py"
SPEC = importlib.util.spec_from_file_location("build_form_only_v1_packet", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_grade_aliases_are_explicit_and_stable() -> None:
    assert MODULE.canonical_grade("5th Grade") == "GRADE_5"
    assert MODULE.canonical_grade("Grade 5") == "GRADE_5"
    assert MODULE.canonical_grade("Tier 3 - Restricted Win") == "RESTRICTED_WIN"
    assert MODULE.canonical_grade("Bottom Up - Restricted Win") == "RESTRICTED_WIN"
    assert MODULE.canonical_grade("5/6") == "MIXED_5_6"
    assert MODULE.canonical_grade(None) == "__MISSING__"


def test_history_guard_rejects_target_and_forward_rows() -> None:
    rows = [
        {"DATE": "2026-07-08", "TRACK": "BAL", "DIST": "450", "G": "Grade 5", "PLC": "2", "BOX": "1", "MGN": "1.5"},
        {"DATE": "2026-07-09", "TRACK": "BAL", "DIST": "450", "G": "Grade 5", "PLC": "1", "BOX": "1", "MGN": "0"},
        {"DATE": "2026-07-10", "TRACK": "BAL", "DIST": "450", "G": "Grade 5", "PLC": "3", "BOX": "1", "MGN": "2"},
    ]
    accepted, rejected = MODULE.accepted_history(rows, date(2026, 7, 9))
    assert [row["date"].isoformat() for row in accepted] == ["2026-07-08"]
    assert [reason for reason, _row in rejected] == [
        "TARGET_OR_POST_TARGET_HISTORY", "TARGET_OR_POST_TARGET_HISTORY"
    ]


def test_feature_row_has_no_forbidden_feature_columns() -> None:
    history = [{
        "date": date(2026, 7, 1), "venue": "BAL", "distance": 450,
        "grade": "GRADE_5", "finish": 2, "margin": 1.25,
    }]
    row = MODULE.feature_row(
        "Race 1 - BAL - 2026-07-09", date(2026, 7, 9), "BAL", 450,
        "GRADE_5", 8, 1, "Example Dog", history,
    )
    assert not set(row).intersection(MODULE.FORBIDDEN_FEATURE_TOKENS)
    assert row["prior_start_count"] == 1
    assert row["days_since_last_start"] == 8
    assert row["same_venue_start_count"] == 1


def test_dog_name_changes_only_opaque_row_key() -> None:
    args = (
        "Race 1 - BAL - 2026-07-09", date(2026, 7, 9), "BAL", 450,
        "GRADE_5", 8, 1,
    )
    left = MODULE.feature_row(*args, "Dog A", [])
    right = MODULE.feature_row(*args, "Dog B", [])
    assert left["row_id"] != right["row_id"]
    left.pop("row_id")
    right.pop("row_id")
    assert left == right


def test_out_of_time_path_filter_rejects_reconstruction_and_results() -> None:
    assert MODULE.out_of_time_path_allowed(Path("/x/run/refreshed_upcoming/Race 1.csv.metadata.json"))
    assert not MODULE.out_of_time_path_allowed(Path("/x/reconstructed/upcoming/Race 1.csv.metadata.json"))
    assert not MODULE.out_of_time_path_allowed(Path("/x/official_result/upcoming/Race 1.csv.metadata.json"))


def test_market_coverage_is_separate_data_missing_carry_forward(tmp_path: Path) -> None:
    MODULE.write_market_coverage(tmp_path)
    coverage = json.loads((tmp_path / "market_coverage.json").read_text(encoding="utf-8"))

    assert coverage["status"] == "DATA_MISSING"
    assert coverage["paired_race_counts"] == {
        "T-10": 497,
        "T-2": 501,
        "T-30": 402,
        "T-60": 2,
    }
    assert coverage["market_fields_in_packet"] is False
    assert coverage["independent_frozen_cohort_manifest_bound"] is False
