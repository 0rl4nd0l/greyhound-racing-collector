import json
from pathlib import Path

from scripts import collect_expert_form_official_result_labels_report_only as packet


RACE_ID = "Race 1 - GUNNEDAH - 2026-06-17"


def _write_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "Dog Name",
                "1. Alpha Runner",
                "2. Bravo Runner",
                "3. Charlie Runner",
                "4. Delta Runner",
            ]
        ),
        encoding="utf-8",
    )


def _sidecar_payload(*, safe: bool = True) -> dict:
    source_url = (
        "https://www.thedogs.com.au/racing/gunnedah/2026-06-17/1/"
        "test-race/expert-form"
    )
    return {
        "race_info": {
            "date": "2026-06-17",
            "venue": "GUNNEDAH",
            "race_number": "1",
            "race_time": "3:07 PM",
            "race_time_source": "canonical_race_url",
            "race_time_mapping_status": "exact_url_match",
            "url": source_url.removesuffix("/expert-form") + "?trial=false",
        },
        "expert_form_metadata": {
            "schema_version": "thedogs_expert_form_metadata_v1",
            "source": "thedogs_expert_form_page",
            "source_url": source_url,
            "captured_at": "2026-06-17T00:00:00Z",
            "metadata_is_leakage_safe": safe,
            "rejected_reasons": [] if safe else ["unsafe_test_payload"],
            "runners": [{"dog_name": "Alpha Runner"}],
        },
    }


def _feature_rows(tmp_path: Path, *, safe: bool = True) -> Path:
    csv_path = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525/source/Race 1 - GUNNEDAH - 2026-06-17.csv"
    )
    sidecar_path = Path(str(csv_path) + ".metadata.json")
    _write_csv(csv_path)
    sidecar_path.write_text(json.dumps(_sidecar_payload(safe=safe)), encoding="utf-8")
    rows = [
        {
            "race_id": RACE_ID,
            "dog_name": dog,
            "box_number": box,
            "source_csv_path": str(csv_path.relative_to(tmp_path)),
            "source_sidecar_path": str(sidecar_path.relative_to(tmp_path)),
        }
        for box, dog in enumerate(
            ["Alpha Runner", "Bravo Runner", "Charlie Runner", "Delta Runner"],
            start=1,
        )
    ]
    rows_path = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525/expert_form_shadow_feature_row_backfill_test_report_only/shadow_feature_rows.json"
    )
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.write_text(json.dumps(rows), encoding="utf-8")
    return rows_path


def test_collect_labels_writes_ablation_compatible_official_runner_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(packet, "ROOT", tmp_path)
    rows_path = _feature_rows(tmp_path)

    def fake_fetch(candidate, *, use_browser_fallback):
        assert use_browser_fallback is False
        return (
            packet.ingest.SourceResult(
                source="thedogs_official",
                status=packet.RESULTED,
                source_url="https://www.thedogs.com.au/racing/gunnedah/2026-06-17/1/test-race/results?trial=false",
                positions_by_box={1: 1, 2: 2, 3: 3, 4: 4},
                raw_order=[1, 2, 3, 4],
            ),
            None,
        )

    monkeypatch.setattr(packet, "fetch_official_result", fake_fetch)
    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525/expert_form_official_result_labels_test_report_only"
    )
    report = packet.collect_labels(
        expert_feature_rows_path=rows_path,
        output_dir=output_dir,
    )
    packet.write_packet(report, output_dir, {"protected_paths_unchanged": True})

    assert report["final_status"] == packet.FINAL_READY
    assert report["coverage_summary"]["safe_official_result_races"] == 1
    assert report["coverage_summary"]["safe_official_result_runner_rows"] == 4
    assert report["coverage_summary"]["winner_label_runner_rows"] == 4
    runner_rows = [
        json.loads(line)
        for line in (output_dir / "official_result_runners.jsonl").read_text().splitlines()
    ]
    assert runner_rows[0]["race_id"] == RACE_ID
    assert runner_rows[0]["box_number"] == 1
    assert runner_rows[0]["finish_position"] == 1
    assert runner_rows[0]["is_winner"] is True
    winner_rows = [
        json.loads(line)
        for line in (
            output_dir / "official_result_winner_label_runners.jsonl"
        ).read_text().splitlines()
    ]
    assert winner_rows[0]["schema_version"] == (
        "expert_form_official_result_winner_label_runner_v1"
    )
    assert winner_rows[0]["label_scope"] == "winner_only_full_frozen_field"
    manifest = json.loads((output_dir / "output_manifest.json").read_text())
    assert manifest["no_write_guarantees"]["db_write"] is False
    assert manifest["no_write_guarantees"]["canonical_label_write"] is False


def test_collect_labels_rejects_unsafe_expert_form_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(packet, "ROOT", tmp_path)
    rows_path = _feature_rows(tmp_path, safe=False)
    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525/expert_form_official_result_labels_unsafe_report_only"
    )

    report = packet.collect_labels(
        expert_feature_rows_path=rows_path,
        output_dir=output_dir,
    )

    assert report["final_status"] == packet.FINAL_DATA_MISSING
    assert report["coverage_summary"]["candidate_races"] == 0
    assert report["quarantine_rows"][0]["reason"] == "expert_form_sidecar_not_leakage_safe"


def test_collect_labels_quarantines_result_validation_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(packet, "ROOT", tmp_path)
    rows_path = _feature_rows(tmp_path)

    def fake_fetch(candidate, *, use_browser_fallback):
        return (
            packet.ingest.SourceResult(
                source="thedogs_official",
                status=packet.RESULTED,
                source_url="https://www.thedogs.com.au/racing/gunnedah/2026-06-17/1/test-race/results?trial=false",
                positions_by_box={9: 1},
                raw_order=[9],
            ),
            None,
        )

    monkeypatch.setattr(packet, "fetch_official_result", fake_fetch)
    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525/expert_form_official_result_labels_mismatch_report_only"
    )
    report = packet.collect_labels(
        expert_feature_rows_path=rows_path,
        output_dir=output_dir,
    )

    assert report["final_status"] == packet.FINAL_DATA_MISSING
    assert report["coverage_summary"]["safe_official_result_races"] == 0
    assert report["quarantine_rows"][0]["reason"] == "official_result_validation_failed"
    assert "result_boxes_not_in_participants:9" in report["quarantine_rows"][0]["errors"]


def test_result_dog_name_validation_ignores_terminal_nbt_suffix(tmp_path):
    csv_path = tmp_path / "race.csv"
    candidate = packet.ingest.RaceCandidate(
        race_id=RACE_ID,
        venue="GUNNEDAH",
        race_number=1,
        race_date="2026-06-17",
        race_time="3:07 PM",
        start_datetime=None,
        sportsbet_url=None,
        csv_path=csv_path,
        participants=[{"box_number": 1, "dog_name": "Alpha Runner"}],
        lifecycle_status="test",
    )
    result = packet.ingest.SourceResult(
        source="thedogs_official",
        status=packet.RESULTED,
        source_url="https://www.thedogs.com.au/racing/gunnedah/2026-06-17/1/test-race/results?trial=false",
        positions_by_box={1: 1},
        raw_order=[1],
        dog_names_by_box={1: "Alpha Runner NBT"},
    )

    assert packet.result_dog_name_validation_error(candidate, result) is None

    result.dog_names_by_box = {1: "Different Runner NBT"}
    assert packet.result_dog_name_validation_error(candidate, result) == (
        "result_dog_name_mismatch_for_box:1"
    )


def test_winner_label_rows_cover_frozen_field_without_fabricating_finish_positions(tmp_path):
    candidate = packet.ingest.RaceCandidate(
        race_id=RACE_ID,
        venue="GUNNEDAH",
        race_number=1,
        race_date="2026-06-17",
        race_time="3:07 PM",
        start_datetime=None,
        sportsbet_url=None,
        csv_path=tmp_path / "race.csv",
        participants=[
            {"box_number": 1, "dog_name": "Alpha Runner"},
            {"box_number": 2, "dog_name": "Bravo Runner"},
            {"box_number": 3, "dog_name": "Charlie Runner"},
        ],
        lifecycle_status="test",
    )
    result = packet.ingest.SourceResult(
        source="thedogs_official",
        status=packet.RESULTED,
        source_url="https://www.thedogs.com.au/racing/gunnedah/2026-06-17/1/test-race/results?trial=false",
        positions_by_box={2: 1, 1: 2},
        raw_order=[2, 1],
    )

    rows = packet.winner_label_rows_for_result(
        candidate,
        result,
        captured_at="2026-06-17T18:00:00+10:00",
    )

    assert len(rows) == 3
    assert [row["is_winner"] for row in rows] == [False, True, False]
    assert rows[2]["finish_position"] is None
    assert rows[2]["finish_position_available"] is False
