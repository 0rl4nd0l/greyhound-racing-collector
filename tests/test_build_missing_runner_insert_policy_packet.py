import json
from pathlib import Path

import pytest

import scripts.build_missing_runner_insert_policy_packet as policy
from scripts.build_missing_runner_insert_policy_packet import build_policy_packet, main
from tests.test_build_rolling_failure_repair_triage_packet import (
    _db,
    _lookup,
    _predictions,
    _queue,
    _winner_only,
)


def test_missing_runner_insert_policy_builds_exact_no_write_candidates(tmp_path: Path):
    packet = build_policy_packet(
        failure_review_csv_path=_queue(tmp_path / "queue.csv"),
        lookup_packet_paths=[_lookup(tmp_path / "lookup.json")],
        db_path=_db(tmp_path / "greyhound.sqlite"),
        prediction_rows_path=_predictions(tmp_path / "predictions.jsonl"),
        winner_only_rows_path=_winner_only(tmp_path / "winner_only.jsonl"),
    )

    assert packet["schema_version"] == "missing_runner_insert_policy_packet_v1"
    assert packet["status"] == "REPORT_ONLY_MISSING_RUNNER_INSERT_POLICY_PACKET"
    assert packet["safe_to_write_now"] is False
    assert packet["writes_performed"]["db_write"] is False
    assert packet["summary"]["races_considered"] == 2
    assert packet["summary"]["races_with_missing_runner_insert_candidates"] == 1
    assert packet["summary"]["missing_runner_insert_candidate_count"] == 1
    assert packet["insert_policy"]["missing_required_columns"] == []
    assert packet["insert_policy"]["proposed_insert_columns"] == [
        "race_id",
        "dog_name",
        "dog_clean_name",
        "box_number",
        "finish_position",
        "placing",
        "scraped_finish_position",
        "extraction_timestamp",
        "data_source",
    ]
    candidate = packet["candidate_rows"][0]
    assert candidate["race_id"] == "R1"
    assert candidate["official_dog_name"] == "Gamma"
    assert candidate["insert_values"]["dog_name"] == "Gamma"
    assert candidate["insert_values"]["data_source"] == "thedogs_official"
    assert "duplicate_guard_required_before_each_insert" in candidate["blockers"]


def test_missing_runner_insert_policy_cli_writes_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(policy, "ROOT", tmp_path)
    output_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/insert_policy"

    exit_code = main(
        [
            "--failure-review-csv",
            str(_queue(tmp_path / "queue.csv")),
            "--lookup-packet",
            str(_lookup(tmp_path / "lookup.json")),
            "--db",
            str(_db(tmp_path / "greyhound.sqlite")),
            "--predictions-jsonl",
            str(_predictions(tmp_path / "predictions.jsonl")),
            "--winner-only-rows-jsonl",
            str(_winner_only(tmp_path / "winner_only.jsonl")),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    payload = json.loads((output_dir / "missing_runner_insert_policy_packet.json").read_text())
    assert payload["summary"]["missing_runner_insert_candidate_count"] == 1
    assert (output_dir / "missing_runner_insert_candidates.csv").exists()
    assert "No DB rows" in (output_dir / "SUMMARY.md").read_text(encoding="utf-8")


def test_missing_runner_insert_policy_rejects_absolute_output_outside_repo(tmp_path: Path):
    outside = tmp_path.parent / "artifacts/full_evidence_orchestration_20260525/outside"

    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        policy._assert_output_dir_safe(outside, root=tmp_path)


def test_missing_runner_insert_policy_rejects_in_repo_non_artifact_output(tmp_path: Path):
    with pytest.raises(ValueError, match="output_dir_must_be_under_artifacts"):
        policy._assert_output_dir_safe(tmp_path / "reports/not_allowed", root=tmp_path)
