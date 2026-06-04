import json
from datetime import datetime, timezone

from scripts.audit_prediction_snapshot_readiness import build_audit


def _write_snapshot(path, *, race_id, box_probs, ready=True, race_date="2026-06-01"):
    predictions = [
        {
            "dog_name": f"Dog {box}",
            "box_number": box,
            "win_prob_norm": prob,
            "predicted_rank": rank,
        }
        for rank, (box, prob) in enumerate(box_probs, start=1)
    ]
    snapshot = {
        "schema_version": "prediction_snapshot_v1",
        "race_id": race_id,
        "stable_race_key": f"{race_date}|TEST|{race_id[-1]}",
        "race_date": race_date,
        "venue": "TEST",
        "race_number": int(race_id[-1]),
        "lifecycle_status": "upcoming_not_jumped",
        "is_pre_jump_snapshot": True,
        "feature_freeze_timestamp": f"{race_date}T12:00:00",
        "prediction_timestamp": f"{race_date}T12:00:00",
        "jump_datetime": f"{race_date}T12:30:00+00:00",
        "snapshot_readiness": {"status": "READY" if ready else "NOT_READY"},
        "predictions": predictions,
    }
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    return snapshot


def _append_manifest(manifest, snapshot_path, *, race_id, race_date="2026-06-01"):
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "schema_version": "prediction_snapshot_manifest_v1",
                    "race_id": race_id,
                    "stable_race_key": f"{race_date}|TEST|{race_id[-1]}",
                    "snapshot_path": str(snapshot_path),
                    "lifecycle_status": "upcoming_not_jumped",
                    "feature_freeze_timestamp": f"{race_date}T12:00:00",
                }
            )
            + "\n"
        )


def test_manifest_backed_audit_uses_ready_latest_snapshots_and_computes_box_share(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    s1 = tmp_path / "race1.json"
    s2 = tmp_path / "race2.json"
    _write_snapshot(s1, race_id="Race 1", box_probs=[(1, 0.8), (2, 0.2)])
    _write_snapshot(s2, race_id="Race 2", box_probs=[(2, 0.7), (1, 0.3)])
    _append_manifest(manifest, s1, race_id="Race 1")
    _append_manifest(manifest, s2, race_id="Race 2")

    report = build_audit(
        manifest_path=manifest,
        repo_root=tmp_path,
        now=datetime(2026, 6, 2, tzinfo=timezone.utc),
    )

    assert report["counts"]["latest_ready_races"] == 2
    assert report["latest_ready_races_summary"]["top_pick_box_distribution"] == {"1": 1, "2": 1}
    assert report["latest_ready_races_summary"]["box1_share"] == 0.5
    assert report["gate"]["status"] == "PASS"


def test_manifest_backed_audit_excludes_not_ready_and_low_runner_snapshots(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    not_ready = tmp_path / "not_ready.json"
    one_runner = tmp_path / "one_runner.json"
    _write_snapshot(not_ready, race_id="Race 1", box_probs=[(1, 0.8), (2, 0.2)], ready=False)
    _write_snapshot(one_runner, race_id="Race 2", box_probs=[(1, 1.0)], ready=True)
    _append_manifest(manifest, not_ready, race_id="Race 1")
    _append_manifest(manifest, one_runner, race_id="Race 2")

    report = build_audit(
        manifest_path=manifest,
        repo_root=tmp_path,
        min_runners=2,
        now=datetime(2026, 6, 2, tzinfo=timezone.utc),
    )

    assert report["counts"]["latest_ready_races"] == 0
    assert report["gate"]["status"] == "DATA_MISSING"
    assert report["skip_reason_counts"]["snapshot_readiness_not_ready"] == 1
    assert report["skip_reason_counts"]["runner_count_below_minimum"] == 1


def test_manifest_backed_audit_fails_when_box1_share_exceeds_threshold(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    s1 = tmp_path / "race1.json"
    s2 = tmp_path / "race2.json"
    _write_snapshot(s1, race_id="Race 1", box_probs=[(1, 0.8), (2, 0.2)])
    _write_snapshot(s2, race_id="Race 2", box_probs=[(1, 0.7), (2, 0.3)])
    _append_manifest(manifest, s1, race_id="Race 1")
    _append_manifest(manifest, s2, race_id="Race 2")

    report = build_audit(
        manifest_path=manifest,
        repo_root=tmp_path,
        box1_max_share=0.5,
        now=datetime(2026, 6, 2, tzinfo=timezone.utc),
    )

    assert report["latest_ready_races_summary"]["box1_share"] == 1.0
    assert report["gate"]["status"] == "FAIL"
