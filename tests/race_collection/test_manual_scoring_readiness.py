from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from race_collection.manual_scoring_readiness import (
    GHU_MERGE_COMMITS,
    READINESS_SCHEMA,
    manual_scoring_readiness_index_path,
    publish_manual_scoring_readiness_index,
)
from src.predictor.on_demand import canonical_bytes

NOW = datetime.fromisoformat("2026-08-07T10:00:00+10:00")
ROOT = Path(__file__).resolve().parents[2]


def _identity() -> dict:
    return {
        "repository": {"commit": "1" * 40, "tree": "2" * 40},
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
    coverage = [_source_files(root, first)]
    source = _write_source(root, [first, second], coverage)

    result = _publish(tmp_path, source, monkeypatch)

    assert result["status"] == "PUBLISHED"
    assert result["eligible_race_count"] == 1
    packet = json.loads(_published_index(tmp_path).read_bytes())
    assert [row["race_id"] for row in packet["races"]] == [first["race_id"]]
    assert packet["races"][0]["odds"]["status"] == "PENDING_GHU_051"
    assert packet["exclusions"][0]["reason_code"] == "FORM_SOURCE_MISSING"


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
    source = _write_source(root, [first, second], [])
    state = root / "runtime" / "state.json"
    index = manual_scoring_readiness_index_path(state)
    index.parent.mkdir(parents=True)
    prior = b"prior-readiness-bytes"
    index.write_bytes(prior)

    result = _publish(tmp_path, source, monkeypatch)

    assert result["status"] == "REJECTED"
    assert result["reason"] == "GLOBAL_PACKET_IDENTITY_AMBIGUOUS"
    assert index.read_bytes() == prior


def test_atomic_publication_failure_preserves_prior_readiness_bytes(tmp_path, monkeypatch):
    root = tmp_path / "evidence"
    race = _race(1)
    source = _write_source(root, [race], [_source_files(root, race)], status="SUCCESS")
    state = root / "runtime" / "state.json"
    index = manual_scoring_readiness_index_path(state)
    index.parent.mkdir(parents=True)
    prior = b"prior-readiness-bytes"
    index.write_bytes(prior)
    monkeypatch.setattr(
        "race_collection.manual_scoring_readiness._atomic_replace_canonical",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("atomic failure")),
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
    Draft202012Validator(schema).validate(packet)
    assert packet["schema_version"] == READINESS_SCHEMA
    assert packet["safety"]["canonical"] is False
