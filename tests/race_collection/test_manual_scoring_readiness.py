from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

import race_collection.synchronous_manual_capture as capture
from race_collection.manual_scoring_readiness import (
    GHU_MERGE_COMMITS,
    READINESS_AUTHORITATIVE_FILES,
    READINESS_SCHEMA,
    manual_scoring_readiness_index_path,
    publish_manual_scoring_readiness_index,
)
from race_collection.synchronous_manual_capture import publish_current_race_index
from src.predictor.on_demand import canonical_bytes

NOW = datetime.fromisoformat("2026-08-07T10:00:00+10:00")
ROOT = Path(__file__).resolve().parents[2]


def _identity() -> dict:
    return {
        "repository": {"commit": "1" * 40, "tree": "2" * 40},
        "readiness_authoritative": {
            "members": [
                {"path": path, "bytes": 1, "sha256": "b" * 64}
                for path in READINESS_AUTHORITATIVE_FILES
            ]
        },
        "model": {
            "model_id": "market_form_residual_v1",
            "model_sha256": "3" * 64,
            "manifest_sha256": "4" * 64,
            "effective_state_schema": "market_form_residual_effective_state_v2",
            "effective_state_sha256": "5" * 64,
        },
        "config": {
            "manual_capture": {
                "path": "configs/prediction/manual-independent-capture-v1/example-config.json",
                "sha256": "6" * 64,
            },
            "prediction": {
                "path": "configs/prediction/manual-default.json",
                "sha256": "7" * 64,
            },
            "minimum_prejump_margin_seconds": 120,
        },
        "scoring_contract": {
            "schema_version": "market_form_residual_scoring_config_v1",
            "sha256": "8" * 64,
            "numeric_canonicalization_sha256": "9" * 64,
        },
        "ghu_050_056": {
            ticket: {
                "merge_commit": commit,
                "members": [
                    {"path": f"fixture/{ticket}.json", "bytes": 1, "sha256": "a" * 64}
                ],
            }
            for ticket, commit in GHU_MERGE_COMMITS.items()
        },
    }


def _race(index: int, *, jump: datetime | None = None, url: str | None = None) -> dict:
    date = "2026-08-07"
    venue = "TEST"
    race_number = index
    return {
        "race_id": f"Race {index} - {venue} - {date}",
        "race_id_aliases": [f"Race {index} - {venue} - {date}"],
        "race_url": url or f"https://www.thedogs.com.au/racing/test/{date}/{index}",
        "race_number": race_number,
        "venue": venue,
        "date": date,
        "race_time": (jump or NOW + timedelta(minutes=20)).strftime("%H:%M"),
        "jump_datetime": (jump or NOW + timedelta(minutes=20)).isoformat(),
    }


def _source_files(root: Path, race: dict, *, bad_csv: bool = False) -> dict:
    csv_path = root / "upcoming" / f"Race {race['race_id']}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(
        "box|dog_name\n1|Alpha\n2|Beta\n" if not bad_csv else "box|dog_name\n1|Alpha\n",
        encoding="utf-8",
    )
    sidecar_path = csv_path.with_name(csv_path.name + ".metadata.json")
    sidecar_path.write_bytes(
        canonical_bytes(
            {
                "runner_completeness_after_canonical_alignment": {
                    "status": "COMPLETE",
                    "runner_count": 2,
                    "participants": [
                        {"box_number": 1, "dog_name": "Alpha", "scratch_state": "ACTIVE"},
                        {"box_number": 2, "dog_name": "Beta", "scratch_state": "ACTIVE"},
                    ],
                },
                "prejump_shadow_metadata": {
                    "status": "PASS",
                    "metadata_is_leakage_safe": True,
                    "race_date": race["date"],
                    "venue": race["venue"],
                    "race_number": race["race_number"],
                    "source_url": race["race_url"],
                    "metadata_captured_at": "2026-08-07T09:50:00+10:00",
                    "runner_box_name_list": [
                        {"box_number": 1, "dog_name": "Alpha"},
                        {"box_number": 2, "dog_name": "Beta"},
                    ],
                    "canonical_final_runner_alignment": {
                        "status": "aligned",
                        "canonical_runner_set_status": "available",
                    },
                },
            }
        )
    )
    return {"race_url": race["race_url"], "csv_path": str(csv_path), "sidecar_path": str(sidecar_path)}


def _missing_coverage(race: dict) -> dict:
    return {
        "race_id": race["race_id"],
        "race_url": race["race_url"],
        "csv_path": None,
        "sidecar_path": None,
        "sidecar_status": "accepted_csv_missing",
    }


def _write_source(root: Path, races: list[dict], coverage: list[dict], *, status: str = "METADATA_COVERAGE_INCOMPLETE") -> Path:
    source = root / "reports" / "odds_capture_refresh_report.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(
        canonical_bytes(
            {
                "status": status,
                "dry_run": False,
                "generated_at": NOW.isoformat(),
                "selected_count": len(races),
                "selected_races": races,
                "sidecar_metadata_coverage": {
                    "schema_version": "prejump_sidecar_metadata_coverage_v1",
                    "status": "PARTIAL" if status != "SUCCESS" else "READY",
                    "selected_race_count": len(races),
                    "accepted_selected_csv_count": sum(1 for row in coverage if row.get("csv_path")),
                    "safe_weather_race_count": 0,
                    "safe_track_condition_race_count": 0,
                    "safe_both_weather_track_race_count": 0,
                    "safe_expert_form_race_count": 0,
                    "safe_all_weather_track_expert_form_race_count": 0,
                    "races": coverage,
                },
            }
        )
    )
    return source


def _publish(tmp_path: Path, source: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setattr(
        "race_collection.manual_scoring_readiness._global_identity", lambda _root: _identity()
    )
    evidence = tmp_path / "evidence"
    state = evidence / "shadow_autopilot_daemon_runtime" / "odds_capture_state.json"
    return publish_manual_scoring_readiness_index(
        state_path=state,
        evidence_root=evidence,
        source_refresh_report_path=source,
        now=NOW,
        repo_root=ROOT,
    )


def _published_index(tmp_path: Path) -> Path:
    return manual_scoring_readiness_index_path(
        tmp_path / "evidence" / "shadow_autopilot_daemon_runtime" / "state.json"
    )


def test_mixed_metadata_packet_publishes_only_race_with_capture_prerequisites(tmp_path, monkeypatch):
    root = tmp_path / "evidence"
    first = _race(1)
    second = _race(2, jump=NOW + timedelta(minutes=25))
    coverage = [_source_files(root, first), _missing_coverage(second)]
    source = _write_source(root, [first, second], coverage)

    result = _publish(tmp_path, source, monkeypatch)

    assert result["status"] == "PUBLISHED"
    assert result["eligible_race_count"] == 1
    packet = json.loads(_published_index(tmp_path).read_bytes())
    assert [row["race_id"] for row in packet["races"]] == [first["race_id"]]
    assert packet["races"][0]["odds"]["status"] == "PENDING_GHU_051"
    assert packet["exclusions"][0]["reason_code"] == "FORM_SOURCE_MISSING"
    legacy = publish_current_race_index(
        state_path=root / "shadow_autopilot_daemon_runtime" / "odds_capture_state.json",
        evidence_root=root,
        source_refresh_report_path=source,
        run_id="legacy-incomplete",
    )
    assert legacy["status"] == "PUBLISHED"
    assert legacy["race_count"] == 1
    legacy_packet = json.loads(
        (
            root
            / "shadow_autopilot_daemon_runtime"
            / "manual_prediction_current_race_index.json"
        ).read_bytes()
    )
    assert [row["race_id"] for row in legacy_packet["races"]] == [first["race_id"]]


@pytest.mark.parametrize(
    "mutation,expected",
    [
        (lambda race: race.update(race_id="not-the-canonical-id"), "RACE_IDENTITY_INVALID"),
        (lambda race: race.update(jump_datetime=(NOW - timedelta(minutes=1)).isoformat(), race_time="09:59"), "PREJUMP_TIMING_INVALID"),
    ],
)
def test_race_local_identity_and_prejump_defects_exclude_only_affected_race(tmp_path, monkeypatch, mutation, expected):
    root = tmp_path / "evidence"
    bad = _race(1)
    mutation(bad)
    good = _race(2, jump=NOW + timedelta(minutes=25))
    coverage = [_source_files(root, bad), _source_files(root, good)]
    source = _write_source(root, [bad, good], coverage)

    result = _publish(tmp_path, source, monkeypatch)

    assert result["status"] == "PUBLISHED"
    packet = json.loads(_published_index(tmp_path).read_bytes())
    assert [row["race_id"] for row in packet["races"]] == [good["race_id"]]
    assert packet["exclusions"][0]["reason_code"] == expected


def test_global_noncanonical_source_preserves_prior_readiness_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "race_collection.manual_scoring_readiness._global_identity", lambda _root: _identity()
    )
    evidence = tmp_path / "evidence"
    state = evidence / "runtime" / "state.json"
    index = manual_scoring_readiness_index_path(state)
    index.parent.mkdir(parents=True)
    prior = b"prior-readiness-bytes"
    index.write_bytes(prior)
    source = evidence / "reports" / "source.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(b'{"status":"SUCCESS","status":"TAMPERED"}')

    result = publish_manual_scoring_readiness_index(
        state_path=state,
        evidence_root=evidence,
        source_refresh_report_path=source,
        now=NOW,
        repo_root=ROOT,
    )

    assert result == {
        "schema_version": "manual_prediction_scoring_readiness_publish_v1",
        "status": "REJECTED",
        "index_path": str(index),
        "source_refresh_report_path": str(source),
        "reason": "GLOBAL_SOURCE_PACKET_NOT_CANONICAL",
    }
    assert index.read_bytes() == prior


def test_global_ambiguous_identity_preserves_prior_readiness_bytes(tmp_path, monkeypatch):
    root = tmp_path / "evidence"
    first = _race(1)
    second = _race(2, url=first["race_url"].replace("/1", "/2"))
    second["race_id"] = first["race_id"]
    source = _write_source(
        root, [first, second], [_missing_coverage(first), _missing_coverage(second)]
    )
    index = _published_index(tmp_path)
    index.parent.mkdir(parents=True)
    prior = b"prior-readiness-bytes"
    index.write_bytes(prior)

    result = _publish(tmp_path, source, monkeypatch)

    assert result["status"] == "REJECTED"
    assert result["reason"] == "GLOBAL_PACKET_IDENTITY_AMBIGUOUS"
    assert index.read_bytes() == prior


@pytest.mark.parametrize("malformed", [None, 7, [], {"race_id": "only"}])
def test_malformed_selected_race_member_is_global_and_preserves_prior_bytes(
    tmp_path, monkeypatch, malformed
):
    root = tmp_path / "evidence"
    source = _write_source(
        root,
        [malformed],
        [{"race_url": None, "csv_path": None, "sidecar_path": None}],
    )
    index = _published_index(tmp_path)
    index.parent.mkdir(parents=True)
    prior = b"exact-prior-readiness-bytes"
    index.write_bytes(prior)

    result = _publish(tmp_path, source, monkeypatch)

    assert result["status"] == "REJECTED"
    assert result["reason"] == "GLOBAL_SOURCE_PACKET_INVALID"
    assert index.read_bytes() == prior


def test_atomic_publication_failure_preserves_prior_readiness_bytes(tmp_path, monkeypatch):
    root = tmp_path / "evidence"
    race = _race(1)
    source = _write_source(root, [race], [_source_files(root, race)], status="SUCCESS")
    index = _published_index(tmp_path)
    index.parent.mkdir(parents=True)
    prior = b"prior-readiness-bytes"
    index.write_bytes(prior)
    def partial_replace(path, _payload, *, evidence_root, **_kwargs):
        del evidence_root
        Path(path).write_bytes(b"new")
        raise OSError("atomic failure after replace")

    monkeypatch.setattr(
        "race_collection.manual_scoring_readiness._atomic_replace_canonical",
        partial_replace,
    )

    result = _publish(tmp_path, source, monkeypatch)

    assert result["status"] == "REJECTED"
    assert result["reason"] == "OSError"
    assert index.read_bytes() == prior


def test_readiness_packet_matches_own_schema_and_does_not_change_parity_hashes(tmp_path, monkeypatch):
    root = tmp_path / "evidence"
    race = _race(1)
    source = _write_source(root, [race], [_source_files(root, race)], status="SUCCESS")
    result = _publish(tmp_path, source, monkeypatch)
    assert result["status"] == "PUBLISHED"
    packet_path = _published_index(tmp_path)
    packet = json.loads(packet_path.read_bytes())
    schema = json.loads((ROOT / "configs/prediction/manual-readiness-v1/scoring-readiness.schema.json").read_bytes())
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(packet)
    assert packet["schema_version"] == READINESS_SCHEMA
    assert packet["safety"]["canonical"] is False


def test_real_global_identity_rejects_corrupt_pinned_member_and_preserves_prior_bytes(tmp_path):
    clone = tmp_path / "repo"
    subprocess.run(
        ["git", "clone", "--no-local", str(ROOT), str(clone)],
        check=True,
        capture_output=True,
        text=True,
    )
    member = clone / "src/predictor/manual_independent_capture.py"
    member.write_bytes(member.read_bytes() + b"\ncorrupt\n")
    root = tmp_path / "evidence"
    source = _write_source(root, [], [])
    index = _published_index(tmp_path)
    state = root / "shadow_autopilot_daemon_runtime" / "odds_capture_state.json"
    index.parent.mkdir(parents=True)
    prior = b"exact-prior-readiness-bytes"
    index.write_bytes(prior)

    result = publish_manual_scoring_readiness_index(
        state_path=state,
        evidence_root=root,
        source_refresh_report_path=source,
        now=NOW,
        repo_root=clone,
    )

    assert result["status"] == "REJECTED"
    assert result["reason"] in {
        "GLOBAL_REPOSITORY_DIRTY",
        "GLOBAL_PINNED_IDENTITY_MISMATCH",
    }
    assert index.read_bytes() == prior


def test_dirty_synchronous_capture_provenance_is_global_and_preserves_prior_bytes(tmp_path):
    clone = tmp_path / "repo"
    subprocess.run(
        ["git", "clone", "--no-local", str(ROOT), str(clone)],
        check=True,
        capture_output=True,
        text=True,
    )
    member = clone / "race_collection/synchronous_manual_capture.py"
    member.write_bytes(member.read_bytes() + b"\ncorrupt\n")
    root = tmp_path / "evidence"
    source = _write_source(root, [], [])
    index = _published_index(tmp_path)
    index.parent.mkdir(parents=True)
    prior = b"exact-prior-readiness-bytes"
    index.write_bytes(prior)

    result = publish_manual_scoring_readiness_index(
        state_path=root / "shadow_autopilot_daemon_runtime" / "odds_capture_state.json",
        evidence_root=root,
        source_refresh_report_path=source,
        now=NOW,
        repo_root=clone,
    )

    assert result["status"] == "REJECTED"
    assert result["reason"] == "GLOBAL_REPOSITORY_DIRTY"
    assert index.read_bytes() == prior


def test_global_tampered_coverage_schema_preserves_prior_bytes(tmp_path, monkeypatch):
    root = tmp_path / "evidence"
    race = _race(1)
    source = _write_source(root, [race], [_missing_coverage(race)])
    payload = json.loads(source.read_bytes())
    payload["sidecar_metadata_coverage"]["schema_version"] = "TAMPERED"
    source.write_bytes(canonical_bytes(payload))
    index = _published_index(tmp_path)
    index.parent.mkdir(parents=True)
    prior = b"exact-prior-readiness-bytes"
    index.write_bytes(prior)

    result = _publish(tmp_path, source, monkeypatch)

    assert result["status"] == "REJECTED"
    assert result["reason"] == "GLOBAL_SOURCE_PACKET_COVERAGE_INVALID"
    assert index.read_bytes() == prior


def test_unsafe_race_source_path_is_global_and_preserves_prior_bytes(tmp_path, monkeypatch):
    root = tmp_path / "evidence"
    good = _race(1)
    bad = _race(2, jump=NOW + timedelta(minutes=25))
    coverage = [
        _source_files(root, good),
        {
            "race_url": bad["race_url"],
            "csv_path": "../outside.csv",
            "sidecar_path": "../outside.csv.metadata.json",
        },
    ]
    source = _write_source(root, [good, bad], coverage)
    index = _published_index(tmp_path)
    index.parent.mkdir(parents=True)
    prior = b"exact-prior-readiness-bytes"
    index.write_bytes(prior)

    result = _publish(tmp_path, source, monkeypatch)

    assert result["status"] == "REJECTED"
    assert result["reason"] == "GLOBAL_SOURCE_PATH_UNSAFE"
    assert index.read_bytes() == prior


def test_cross_race_alias_collision_is_global_and_preserves_prior_bytes(tmp_path, monkeypatch):
    root = tmp_path / "evidence"
    first = _race(1)
    second = _race(2, jump=NOW + timedelta(minutes=25))
    second["race_id_aliases"] = [first["race_id"]]
    coverage = [_missing_coverage(first), _missing_coverage(second)]
    source = _write_source(root, [first, second], coverage)
    index = _published_index(tmp_path)
    index.parent.mkdir(parents=True)
    prior = b"exact-prior-readiness-bytes"
    index.write_bytes(prior)

    result = _publish(tmp_path, source, monkeypatch)

    assert result["status"] == "REJECTED"
    assert result["reason"] == "GLOBAL_PACKET_IDENTITY_AMBIGUOUS"
    assert index.read_bytes() == prior


def test_post_replace_atomic_failure_rolls_back_exact_prior_bytes(tmp_path, monkeypatch):
    root = tmp_path / "evidence"
    race = _race(1)
    source = _write_source(root, [race], [_source_files(root, race)], status="SUCCESS")
    index = _published_index(tmp_path)
    index.parent.mkdir(parents=True)
    prior = b"exact-prior-readiness-bytes"
    index.write_bytes(prior)
    calls = 0
    real_fsync = capture.os.fsync

    def fail_after_replace(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("post-replace fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr(capture.os, "fsync", fail_after_replace)

    result = _publish(tmp_path, source, monkeypatch)

    assert result["status"] == "REJECTED"
    assert result["reason"] == "OSError"
    assert index.read_bytes() == prior


def test_persistent_post_replace_fsync_fault_rejects_with_exact_prior_bytes(
    tmp_path, monkeypatch
):
    root = tmp_path / "evidence"
    race = _race(1)
    source = _write_source(root, [race], [_source_files(root, race)], status="SUCCESS")
    index = _published_index(tmp_path)
    index.parent.mkdir(parents=True)
    prior = b"exact-prior-readiness-bytes"
    index.write_bytes(prior)
    calls = 0
    real_fsync = capture.os.fsync

    def fail_persistently_after_temp_fsync(fd):
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise OSError("persistent post-replace fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr(capture.os, "fsync", fail_persistently_after_temp_fsync)

    result = _publish(tmp_path, source, monkeypatch)

    assert result["status"] == "REJECTED"
    assert result["reason"] == "OSError"
    assert index.read_bytes() == prior


def test_replace_failure_after_effect_rejects_with_exact_prior_bytes(tmp_path, monkeypatch):
    root = tmp_path / "evidence"
    race = _race(1)
    source = _write_source(root, [race], [_source_files(root, race)], status="SUCCESS")
    index = _published_index(tmp_path)
    index.parent.mkdir(parents=True)
    prior = b"exact-prior-readiness-bytes"
    index.write_bytes(prior)
    real_replace = capture.os.replace

    def replace_then_fail(*args, **kwargs):
        real_replace(*args, **kwargs)
        raise OSError("persistent replace failure after effect")

    monkeypatch.setattr(capture.os, "replace", replace_then_fail)

    result = _publish(tmp_path, source, monkeypatch)

    assert result["status"] == "REJECTED"
    assert result["reason"] == "OSError"
    assert index.read_bytes() == prior


def test_pr125_scoring_files_remain_byte_identical_to_base():
    paths = {
        "src/predictor/scoring_parity.py",
        "configs/prediction/market-form-residual-v1/scoring-input.schema.json",
        "configs/prediction/market-form-residual-v1/scoring-core-output.schema.json",
    }
    for relative in paths:
        expected = subprocess.run(
            [
                "git",
                "show",
                f"5e9a370477a905a67bdcb26c9b9315ef0050b362:{relative}",
            ],
            check=True,
            capture_output=True,
        ).stdout
        assert (ROOT / relative).read_bytes() == expected
