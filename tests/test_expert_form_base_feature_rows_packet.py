import json
from pathlib import Path

from scripts import build_expert_form_base_feature_rows_report_only as packet


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    csv_path = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525/source/Race 1 - TEST - 2026-06-17.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("Dog Name\n1. Alpha Runner\n2. Bravo Runner\n", encoding="utf-8")
    expert_rows = [
        {
            "race_id": "Race 1 - TEST - 2026-06-17",
            "dog_name": "Alpha Runner",
            "box_number": 1,
            "source_csv_path": str(csv_path.relative_to(tmp_path)),
        },
        {
            "race_id": "Race 1 - TEST - 2026-06-17",
            "dog_name": "Bravo Runner",
            "box_number": 2,
            "source_csv_path": str(csv_path.relative_to(tmp_path)),
        },
    ]
    expert_path = tmp_path / "expert_rows.json"
    expert_path.write_text(json.dumps(expert_rows), encoding="utf-8")
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "schema_version": "test",
                "feature_columns": ["field_size", "box_number"],
            }
        ),
        encoding="utf-8",
    )
    db_path = tmp_path / "history.db"
    db_path.write_bytes(b"not-empty")
    return expert_path, schema_path, db_path


def test_base_feature_rows_packet_writes_report_only_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(packet, "ROOT", tmp_path)
    expert_path, schema_path, db_path = _write_inputs(tmp_path)

    def fake_build_live_feature_rows(*, input_paths, schema, db_path):
        assert len(input_paths) == 1
        assert schema["schema_version"] == "test"
        return [
            {
                "race_id": "Race 1 - TEST - 2026-06-17",
                "dog_name": "Alpha Runner",
                "box_number": 1,
                "field_size": 2,
            },
            {
                "race_id": "Race 1 - TEST - 2026-06-17",
                "dog_name": "Bravo Runner",
                "box_number": 2,
                "field_size": 2,
            },
        ]

    monkeypatch.setattr(packet, "build_live_feature_rows", fake_build_live_feature_rows)
    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525/expert_form_base_feature_rows_test_report_only"
    )
    report = packet.build_report(
        expert_feature_rows_path=expert_path,
        schema_path=schema_path,
        db_path=db_path,
    )
    packet.write_packet(report, output_dir, {"protected_paths_unchanged": True})

    assert report["final_status"] == packet.FINAL_READY
    assert report["coverage_summary"]["base_feature_rows"] == 2
    assert report["no_write_guarantees"]["training_run"] is False
    assert report["no_write_guarantees"]["db_write"] is False
    rows = json.loads((output_dir / "base_feature_rows.json").read_text())
    assert rows[0]["field_size"] == 2
    manifest = json.loads((output_dir / "output_manifest.json").read_text())
    assert manifest["files"]["base_feature_rows"].endswith("base_feature_rows.json")


def test_base_feature_rows_packet_fails_closed_for_zero_byte_db(tmp_path, monkeypatch):
    monkeypatch.setattr(packet, "ROOT", tmp_path)
    expert_path, schema_path, db_path = _write_inputs(tmp_path)
    db_path.write_bytes(b"")

    report = packet.build_report(
        expert_feature_rows_path=expert_path,
        schema_path=schema_path,
        db_path=db_path,
    )

    assert report["final_status"] == packet.FINAL_DATA_MISSING
    assert report["build_error"] == "db_path_zero_bytes"
    assert report["coverage_summary"]["base_feature_rows"] == 0
