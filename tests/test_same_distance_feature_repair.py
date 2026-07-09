from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from scripts.rebuild_same_distance_feature_packet import (
    assert_output_dir_safe,
    _load_csv,
    _resolve_target_metadata,
    repair_packet,
)


PACKET_CSV = Path(
    "artifacts/full_evidence_orchestration_20260525/"
    "clean_history_feature_packet_20260602/pre_race_history_feature_packet.csv"
)
DB_PATH = Path("greyhound_racing_data_writable.db")


def _packet_rows() -> list[dict[str, object]]:
    if not PACKET_CSV.exists():
        pytest.skip("clean history packet artifact is not present")
    return _load_csv(PACKET_CSV)


def _present_count(rows: list[dict[str, object]], field: str, *, packet: str) -> int:
    return sum(
        1
        for row in rows
        if row.get("packet") == packet and row.get(field) not in (None, "")
    )


def test_safe_target_metadata_resolves_for_unique_meeting():
    rows = _packet_rows()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        unique_row = None
        for row in rows:
            if row.get("packet") != "historical":
                continue
            count = conn.execute(
                "SELECT COUNT(*) FROM race_metadata WHERE race_date=? AND venue=?",
                (str(row.get("race_date")), str(row.get("venue"))),
            ).fetchone()[0]
            if count == 1:
                unique_row = row
                break

        if unique_row is None:
            pytest.skip("no historical row with a unique canonical meeting was found")

        meta = _resolve_target_metadata(conn, unique_row)

    assert meta["status"] == "UNIQUE_DATE_VENUE"
    assert meta["target_distance"] is not None
    assert meta["target_venue"] == unique_row["venue"]


def test_ambiguous_meeting_remains_unsafe():
    rows = _packet_rows()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        ambiguous_row = None
        for row in rows:
            if row.get("packet") != "historical":
                continue
            count = conn.execute(
                "SELECT COUNT(*) FROM race_metadata WHERE race_date=? AND venue=?",
                (str(row.get("race_date")), str(row.get("venue"))),
            ).fetchone()[0]
            if count > 1:
                ambiguous_row = row
                break

        if ambiguous_row is None:
            pytest.skip("no historical row with an ambiguous canonical meeting was found")

        meta = _resolve_target_metadata(conn, ambiguous_row)

    assert meta["status"] == "AMBIGUOUS_OR_MISSING"
    assert meta["target_distance"] is None


def test_repair_packet_populates_historical_same_distance_fields(tmp_path):
    if not PACKET_CSV.exists() or not DB_PATH.exists():
        pytest.skip("packet or database artifact is not present")

    output_dir = Path(
        "artifacts/full_evidence_orchestration_20260525/"
        f"bounded_target_grade_repair_20260603/{tmp_path.name}_same_distance"
    )
    result = repair_packet(
        input_packet=PACKET_CSV,
        output_dir=output_dir,
        db_path=DB_PATH,
    )
    assert result["leakage_status"] == "PASS"
    assert result["parity_status"] == "PASS"

    repaired_rows = _load_csv(output_dir / "repaired_pre_race_history_feature_packet.csv")
    original_rows = _packet_rows()

    for field in (
        "starts_same_distance",
        "prior_same_distance_start_count",
        "best_time_same_distance",
        "avg_time_same_distance",
    ):
        assert _present_count(repaired_rows, field, packet="rolling") >= _present_count(
            original_rows, field, packet="rolling"
        )

    if _present_count(repaired_rows, "starts_same_distance", packet="historical") == 0:
        historical_safe_target_rows = [
            row
            for row in repaired_rows
            if row.get("packet") == "historical"
            and row.get("target_grade_safe") not in (None, "")
            and row.get("target_distance_safe") not in (None, "")
        ]
        if not historical_safe_target_rows:
            pytest.skip(
                "current frozen packet has no leakage-safe historical target metadata"
            )

    for field in (
        "starts_same_distance",
        "prior_same_distance_start_count",
        "best_time_same_distance",
        "avg_time_same_distance",
        "median_time_same_distance",
        "recent_best_time_same_distance_5",
        "recent_avg_time_same_distance_5",
        "days_since_last_same_distance_start",
        "win_rate_same_distance",
        "place_rate_same_distance",
        "same_distance_venue_start_count",
        "same_distance_venue_best_time",
    ):
        assert _present_count(repaired_rows, field, packet="historical") > 0

    repaired_historical = [
        row for row in repaired_rows if row.get("packet") == "historical"
    ]
    repaired_rolling = [row for row in repaired_rows if row.get("packet") == "rolling"]
    for field in (
        "median_time_same_distance",
        "recent_best_time_same_distance_5",
        "recent_avg_time_same_distance_5",
        "days_since_last_same_distance_start",
        "place_rate_same_distance",
        "same_distance_venue_start_count",
        "same_distance_venue_best_time",
    ):
        assert any(row.get(field) not in (None, "") for row in repaired_historical)
        assert all(field in row for row in repaired_historical)
        assert all(field in row for row in repaired_rolling)


def test_same_distance_output_dir_guard_rejects_artifact_symlink_escape(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}_outside"
    outside.mkdir()
    link = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525/"
        "bounded_target_grade_repair_symlink_report_only"
    )
    link.parent.mkdir(parents=True)
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        assert_output_dir_safe(link, repo_root=tmp_path)
