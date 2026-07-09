import json
from pathlib import Path

from scripts import build_expert_form_shadow_feature_row_backfill_packet as packet


def _write_csv(path: Path, dog_name: str = "Alpha Runner") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"Dog Name,Box\n1. {dog_name},1\n", encoding="utf-8")


def _write_sidecar(path: Path, *, race_number: int = 1, dog_name: str = "Alpha Runner") -> None:
    payload = {
        "race_info": {
            "date": "2026-06-17",
            "race_time": "4:00 PM",
            "venue": "CAPALABA",
            "race_number": str(race_number),
        },
        "expert_form_metadata": {
            "schema_version": "thedogs_expert_form_metadata_v1",
            "source": "thedogs_expert_form_page",
            "source_url": f"https://www.thedogs.com.au/racing/capalaba/2026-06-17/{race_number}/test/expert-form",
            "captured_at": "2026-06-17T05:00:00Z",
            "metadata_is_leakage_safe": True,
            "runner_count": 1,
            "rejected_reasons": [],
            "runners": [
                {
                    "dog_name": dog_name,
                    "grade": "5",
                    "career": {"starts": 10, "wins": 3, "seconds": 1, "thirds": 2},
                    "track_distance": {
                        "starts": 4,
                        "wins": 2,
                        "seconds": 0,
                        "thirds": 1,
                        "best_time": 19.8,
                    },
                    "win_percent": 30.0,
                    "place_percent": 60.0,
                    "prize_money": 12345.0,
                    "greyhound": {"sex": "D", "sire": "Sire", "dam": "Dam"},
                    "trainer": {"name": "Trainer", "district": "- Test"},
                    "winning_distance_counts": {"<400": 2, "400+": 1},
                    "box_history": {"1": {"starts": 5, "wins": 2, "places": 3}},
                    "best_win_times_other_tracks": [
                        {"track": "QST", "distance": "350m", "time": 18.8}
                    ],
                }
            ],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_backfill_packet_flattens_safe_expert_form_sidecar_rows(tmp_path):
    root = tmp_path / "artifacts"
    csv_path = root / "Race 1 - CAPALABA - 2026-06-17.csv"
    sidecar_path = Path(f"{csv_path}.metadata.json")
    _write_csv(csv_path)
    _write_sidecar(sidecar_path)

    report = packet.build_report(
        artifact_roots=[root],
        min_source_races=1,
        min_source_runner_rows=1,
        min_feature_rows=1,
    )

    assert report["final_status"] == packet.FINAL_BACKFILL_READY
    assert report["activation_allowed"] is False
    assert report["training_run"] is False
    assert report["model_scoring"] is False
    assert report["coverage_summary"]["selected_safe_source_races"] == 1
    assert report["coverage_summary"]["safe_expert_form_feature_rows"] == 1
    row = report["feature_rows"][0]
    assert row["expert_form_metadata_from_sidecar"] is True
    assert row["expert_form_career_starts"] == 10
    assert row["expert_form_current_box_starts"] == 5


def test_backfill_packet_requires_source_coverage(tmp_path):
    report = packet.build_report(
        artifact_roots=[tmp_path / "missing"],
        min_source_races=1,
        min_source_runner_rows=1,
        min_feature_rows=1,
    )

    assert report["final_status"] == packet.FINAL_SOURCE_LOW
    assert "safe_expert_form_source_races_below_min" in report["blockers"]


def test_backfill_packet_writes_only_report_artifacts(tmp_path):
    root = tmp_path / "artifacts"
    csv_path = root / "Race 1 - CAPALABA - 2026-06-17.csv"
    sidecar_path = Path(f"{csv_path}.metadata.json")
    _write_csv(csv_path)
    _write_sidecar(sidecar_path)
    report = packet.build_report(
        artifact_roots=[root],
        min_source_races=1,
        min_source_runner_rows=1,
        min_feature_rows=1,
    )
    output_dir = (
        packet.ROOT
        / "artifacts/full_evidence_orchestration_20260525"
        / f"expert_form_shadow_feature_row_backfill_test_{tmp_path.name}_report_only"
    )
    if output_dir.exists():
        for child in output_dir.iterdir():
            child.unlink()
        output_dir.rmdir()

    packet.write_packet(
        report,
        output_dir,
        {"protected_paths_unchanged": True, "protected_paths": []},
    )

    try:
        assert (output_dir / "shadow_feature_rows.json").exists()
        assert (output_dir / "expert_form_shadow_feature_row_backfill_report.json").exists()
        assert (output_dir / "protected_path_verification.json").exists()
        assert json.loads((output_dir / "output_manifest.json").read_text(encoding="utf-8"))[
            "no_write_guarantees"
        ]["db_write"] is False
    finally:
        for child in output_dir.iterdir():
            child.unlink()
        output_dir.rmdir()
