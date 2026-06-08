import json
from datetime import datetime
from pathlib import Path

from scripts import aggregate_forward_shadow_results as aggregate


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _runner(race_id, box, rank, prob, winner=False):
    return {
        "race_id": race_id,
        "race_date": "2026-06-08",
        "venue": "TEST",
        "race_number": int(race_id.rsplit(" ", 1)[-1]),
        "box": box,
        "dog_name": f"Dog {box}",
        "predicted_rank": rank,
        "shadow_rf_calibrated_probability": prob,
        "calibration_method": "power_gamma_2.4",
        "tgr_enabled": False,
        "is_winner": winner,
        "result_url": f"https://example.test/{race_id}",
    }


def _write_join_dir(root: Path, name: str, joined_rows, pending_ids=(), source_shadow_run=None, unsafe_rows=()):
    join_dir = root / name
    _write_json(
        join_dir / "shadow_forward_metrics.json",
        {"safe_joined_race_count": 1, "source_shadow_run": source_shadow_run},
    )
    _write_jsonl(join_dir / "joined_shadow_predictions.jsonl", joined_rows)
    _write_json(
        join_dir / "pending_results.json",
        {
            "pending_race_count": len(pending_ids),
            "pending_results": [
                {"race_id": race_id, "status": "PENDING_OFFICIAL_OUTCOME"}
                for race_id in pending_ids
            ],
        },
    )
    _write_json(join_dir / "unsafe_result_matches.json", {"unsafe_result_matches": list(unsafe_rows)})
    return join_dir


def test_aggregate_deduplicates_joined_races_with_latest_artifact_selection(tmp_path):
    _write_join_dir(
        tmp_path,
        "forward_shadow_result_join_20260608T120000+1000",
        [
            _runner("Race 1", 1, 1, 0.6, winner=False),
            _runner("Race 1", 2, 2, 0.4, winner=True),
            _runner("Race 2", 1, 1, 0.7, winner=True),
            _runner("Race 2", 2, 2, 0.3, winner=False),
        ],
        pending_ids=["Race 3"],
    )
    _write_join_dir(
        tmp_path,
        "forward_shadow_result_join_20260608T121000+1000",
        [
            _runner("Race 1", 1, 2, 0.2, winner=False),
            _runner("Race 1", 2, 1, 0.8, winner=True),
        ],
        pending_ids=["Race 3"],
    )

    report = aggregate.build_aggregate_report(
        evidence_root=tmp_path,
        generated_at=datetime.fromisoformat("2026-06-08T12:30:00+10:00"),
    )

    metrics = report["aggregate_forward_metrics"]
    assert report["final_status"] == "PARTIAL_AGGREGATE_PENDING_MORE_RESULTS"
    assert metrics["safe_joined_race_count"] == 2
    assert metrics["pending_race_count"] == 1
    assert metrics["unsafe_match_count"] == 0
    assert metrics["top1"] == 1.0
    assert metrics["winner_ranks"] == [1, 1]
    assert report["duplicate_joined_race_count"] == 1
    duplicate = report["duplicate_joined_races"][0]
    assert duplicate["selected_source"].endswith("forward_shadow_result_join_20260608T121000+1000")
    assert len(duplicate["previous_sources"]) == 1
    assert duplicate["previous_sources"][0].endswith("forward_shadow_result_join_20260608T120000+1000")
    assert duplicate["selected_source"] not in duplicate["previous_sources"]
    assert report["pending_results"]["pending_results"][0]["race_id"] == "Race 3"


def test_output_guard_rejects_non_shadow_aggregate_paths():
    try:
        aggregate.assert_output_dir_safe(Path("model_registry/forward_shadow_result_aggregate_bad"))
    except ValueError as exc:
        assert "output_dir_must_be_forward_shadow_result_aggregate_artifact" in str(exc)
    else:
        raise AssertionError("expected output path guard to reject production path")


def test_aggregate_uses_latest_join_per_source_shadow_run(tmp_path):
    _write_join_dir(
        tmp_path,
        "forward_shadow_result_join_20260608T120000+1000",
        [],
        source_shadow_run="daily_shadow_a",
        unsafe_rows=[{"race_id": "Race 1", "status": "UNSAFE_RESULT_MATCH_QUARANTINED"}],
    )
    _write_join_dir(
        tmp_path,
        "forward_shadow_result_join_20260608T121000+1000",
        [
            _runner("Race 1", 1, 1, 0.7, winner=True),
            _runner("Race 1", 2, 2, 0.3, winner=False),
        ],
        source_shadow_run="daily_shadow_a",
    )

    report = aggregate.build_aggregate_report(
        evidence_root=tmp_path,
        generated_at=datetime.fromisoformat("2026-06-08T12:30:00+10:00"),
    )

    assert report["discovered_join_artifact_count"] == 2
    assert report["source_join_artifact_count"] == 1
    assert report["aggregate_forward_metrics"]["safe_joined_race_count"] == 1
    assert report["unsafe_result_matches"]["unsafe_match_count"] == 0


def test_aggregate_collapses_storage_and_repo_aliases_for_same_daily_shadow_run(tmp_path):
    source_name = "daily_race_ingest_shadow_20260608T202503+1000_urgent_nearjump"
    _write_join_dir(
        tmp_path,
        "forward_shadow_result_join_20260608T120000+1000",
        [],
        pending_ids=["Race 1"],
        source_shadow_run=f"artifacts/full_evidence_orchestration_20260525/{source_name}",
        unsafe_rows=[{"race_id": "Race 2", "status": "UNSAFE_RESULT_MATCH_QUARANTINED"}],
    )
    _write_join_dir(
        tmp_path,
        "forward_shadow_result_join_20260608T121000+1000",
        [
            _runner("Race 1", 1, 1, 0.7, winner=True),
            _runner("Race 1", 2, 2, 0.3, winner=False),
        ],
        source_shadow_run=(
            "../../../greyhound_racing_collector_storage/artifacts/"
            f"full_evidence_orchestration_20260525/{source_name}"
        ),
    )

    report = aggregate.build_aggregate_report(
        evidence_root=tmp_path,
        generated_at=datetime.fromisoformat("2026-06-08T12:30:00+10:00"),
    )

    assert report["discovered_join_artifact_count"] == 2
    assert report["source_join_artifact_count"] == 1
    assert report["aggregate_forward_metrics"]["safe_joined_race_count"] == 1
    assert report["aggregate_forward_metrics"]["pending_race_count"] == 0
    assert report["unsafe_result_matches"]["unsafe_match_count"] == 0


def test_aggregate_keeps_distinct_daily_shadow_runs_separate(tmp_path):
    _write_join_dir(
        tmp_path,
        "forward_shadow_result_join_20260608T120000+1000",
        [
            _runner("Race 1", 1, 1, 0.7, winner=True),
            _runner("Race 1", 2, 2, 0.3, winner=False),
        ],
        source_shadow_run=(
            "artifacts/full_evidence_orchestration_20260525/"
            "daily_race_ingest_shadow_20260608T202503+1000_urgent_nearjump"
        ),
    )
    _write_join_dir(
        tmp_path,
        "forward_shadow_result_join_20260608T121000+1000",
        [
            _runner("Race 2", 1, 1, 0.6, winner=False),
            _runner("Race 2", 2, 2, 0.4, winner=True),
        ],
        source_shadow_run=(
            "artifacts/full_evidence_orchestration_20260525/"
            "daily_race_ingest_shadow_20260608T202922+1000_urgent_nearjump_second_pass"
        ),
    )

    report = aggregate.build_aggregate_report(
        evidence_root=tmp_path,
        generated_at=datetime.fromisoformat("2026-06-08T12:30:00+10:00"),
    )

    assert report["source_join_artifact_count"] == 2
    assert report["aggregate_forward_metrics"]["safe_joined_race_count"] == 2
