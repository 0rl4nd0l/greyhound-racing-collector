from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from race_collection.domain import ArtifactChecksum
from race_collection.evaluation import PromotionPolicy
from race_collection.operator import main

NOW = datetime(2026, 7, 25, 1, 2, 3, tzinfo=timezone.utc)


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def operation(number: int) -> str:
    return f"op_{number:032x}"


def common(database: Path, legacy: Path) -> list[str]:
    return ["--operations-db", str(database), "--legacy-db", str(legacy)]


def write_document(path: Path, document: object) -> Path:
    path.write_bytes(canonical(document))
    return path


def python_311_executable() -> str:
    """Resolve and authenticate a real Python 3.11 independently of the test runner."""
    unavailable = "operator tests require a uv-resolvable exact Python 3.11 executable"
    try:
        probe = subprocess.run(
            (
                "uv",
                "run",
                "--no-project",
                "--python",
                "3.11",
                "python",
                "-c",
                "import json,sys; print(json.dumps("
                "{'executable':sys.executable,"
                "'version':[sys.version_info.major,sys.version_info.minor]},"
                "sort_keys=True,separators=(',',':')))",
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        identity = json.loads(probe.stdout)
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
    ) as error:
        raise AssertionError(unavailable) from error
    if type(identity) is not dict:
        raise AssertionError(unavailable)
    executable = identity.get("executable")
    if (
        type(executable) is not str
        or not Path(executable).is_absolute()
        or identity != {"executable": executable, "version": [3, 11]}
    ):
        raise AssertionError(unavailable)
    return executable


def test_migrate_rejects_legacy_and_non_operations_databases_and_ends_at_29(tmp_path, capsys):
    legacy = tmp_path / "greyhound_racing_data.db"
    with sqlite3.connect(legacy) as db:
        db.execute("CREATE TABLE race_metadata(race_id TEXT PRIMARY KEY)")

    assert main(["migrate", *common(legacy, legacy)]) == 2
    symlink_alias = tmp_path / "legacy-symlink.db"
    symlink_alias.symlink_to(legacy)
    assert main(["migrate", *common(symlink_alias, legacy)]) == 2
    hardlink_alias = tmp_path / "legacy-hardlink.db"
    hardlink_alias.hardlink_to(legacy)
    assert main(["migrate", *common(hardlink_alias, legacy)]) == 2
    with sqlite3.connect(legacy) as db:
        assert db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall() == [("race_metadata",)]

    mislabeled = tmp_path / "mislabeled.sqlite3"
    with sqlite3.connect(mislabeled) as db:
        db.execute("CREATE TABLE dogs(dog_id TEXT PRIMARY KEY)")
    assert main(["migrate", *common(mislabeled, legacy)]) == 2
    with sqlite3.connect(mislabeled) as db:
        assert db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall() == [("dogs",)]

    operations = tmp_path / "operations.sqlite3"
    assert main(["migrate", *common(operations, legacy)]) == 0
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert result == {"command": "migrate", "schema_version": 30, "status": "ok"}
    with sqlite3.connect(operations) as db:
        assert [
            row[0] for row in db.execute("SELECT version FROM schema_migrations ORDER BY version")
        ] == list(range(1, 31))
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    with sqlite3.connect(legacy) as db:
        assert db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall() == [("race_metadata",)]
    with sqlite3.connect(operations) as db:
        db.execute(
            "UPDATE schema_migrations SET checksum=? WHERE version=30",
            ("0" * 64,),
        )
    assert (
        main(
            [
                "register-policy",
                *common(operations, legacy),
                "--artifacts-root",
                str(tmp_path / "artifacts"),
                "--document",
                str(tmp_path / "missing-policy.json"),
                "--operation-id",
                operation(99),
                "--at",
                NOW.isoformat(),
            ]
        )
        == 2
    )
    assert "exact checked-in schema" in capsys.readouterr().err


def test_registration_and_observation_commands_preserve_exact_immutable_authority(tmp_path, capsys):
    legacy = tmp_path / "legacy.db"
    with sqlite3.connect(legacy) as db:
        db.execute("CREATE TABLE race_metadata(race_id TEXT PRIMARY KEY)")
    operations = tmp_path / "operations.sqlite3"
    artifacts = tmp_path / "artifacts"
    assert main(["migrate", *common(operations, legacy)]) == 0

    policy_document = asdict(PromotionPolicy())
    policy = write_document(tmp_path / "policy.json", policy_document)
    configuration_document = {
        "schema_version": "phase7-config-v1",
        "service_root": "/opt/race-collection/current",
        "artifact_root": str(artifacts),
        "operations_database": str(operations),
        "sources": ["official"],
        "schedule_policy": "adaptive-odds-v1",
        "promotion_policy": policy_document["policy_id"],
        "bundle_versions": ["runner-win-probability-v1"],
        "runtime_adapter": "race_collection.runtime_adapters:checked_in",
        "runtime_input_checksum": "sha256:" + "9" * 64,
    }
    configuration = write_document(tmp_path / "configuration.json", configuration_document)
    config_checksum = "sha256:" + hashlib.sha256(canonical(configuration_document)).hexdigest()
    release_document = {
        "schema_version": "phase7-release-v1",
        "release_id": "candidate-release",
        "code_commit": "a" * 40,
        "config_checksum": config_checksum,
        "database_schema": 30,
        "artifact_contract": "canonical-artifacts-v1",
        "policy_version": policy_document["policy_id"],
        "supported_bundle_versions": ["runner-win-probability-v1"],
        "service_root": "/opt/race-collection/current",
    }
    release = write_document(tmp_path / "release.json", release_document)
    legacy_release = write_document(
        tmp_path / "legacy-release.json",
        {**release_document, "release_id": "legacy-release"},
    )
    authority = [*common(operations, legacy), "--artifacts-root", str(artifacts)]

    assert (
        main(
            [
                "register-policy",
                *authority,
                "--document",
                str(policy),
                "--operation-id",
                operation(1),
                "--at",
                NOW.isoformat(),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "register-config",
                *authority,
                "--document",
                str(configuration),
                "--operation-id",
                operation(2),
                "--at",
                NOW.isoformat(),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "register-release",
                *authority,
                "--document",
                str(release),
                "--operation-id",
                operation(3),
                "--at",
                NOW.isoformat(),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "register-release",
                *authority,
                "--document",
                str(legacy_release),
                "--operation-id",
                operation(4),
                "--at",
                NOW.isoformat(),
            ]
        )
        == 0
    )
    initialize = [
        "initialize-legacy",
        *authority,
        "--release-id",
        "legacy-release",
        "--actor",
        "operator",
        "--reason",
        "record exact rollback target",
        "--operation-id",
        operation(5),
        "--at",
        NOW.isoformat(),
    ]
    authorize = [
        "authorize-observation",
        *authority,
        "--release-id",
        "candidate-release",
        "--actor",
        "operator",
        "--reason",
        "future result-blind canary",
        "--operation-id",
        operation(6),
        "--at",
        NOW.isoformat(),
    ]
    assert main(initialize) == 0
    assert main(authorize) == 0
    assert main(authorize) == 0
    conflicting = list(authorize)
    conflicting[conflicting.index("--reason") + 1] = "different reason"
    assert main(conflicting) == 2
    assert (
        main(
            [
                "revoke-observation",
                *authority,
                "--release-id",
                "candidate-release",
                "--actor",
                "operator",
                "--reason",
                "stop before runtime",
                "--operation-id",
                operation(7),
                "--at",
                NOW.isoformat(),
            ]
        )
        == 0
    )

    with sqlite3.connect(operations) as db:
        db.row_factory = sqlite3.Row
        config = db.execute(
            "SELECT config_checksum,operation_id FROM phase7_release_configurations"
        ).fetchone()
        release_rows = db.execute(
            "SELECT release_id,config_checksum,code_commit,operation_id "
            "FROM phase7_release_manifests ORDER BY release_id"
        ).fetchall()
        pointer = db.execute(
            "SELECT release_id,authority,legacy_preserved FROM phase7_release_pointer"
        ).fetchone()
        events = db.execute(
            "SELECT candidate_release_id,action,actor,reason,operation_id "
            "FROM phase7_observation_authority_events ORDER BY event_id"
        ).fetchall()
        assert tuple(config) == (config_checksum, operation(2))
        assert [tuple(row) for row in release_rows] == [
            ("candidate-release", config_checksum, "a" * 40, operation(3)),
            ("legacy-release", config_checksum, "a" * 40, operation(4)),
        ]
        assert tuple(pointer) == ("legacy-release", "legacy", 1)
        assert [tuple(row) for row in events] == [
            (
                "candidate-release",
                "authorize",
                "operator",
                "future result-blind canary",
                operation(6),
            ),
            (
                "candidate-release",
                "revoke",
                "operator",
                "stop before runtime",
                operation(7),
            ),
        ]
    assert "different intent" in capsys.readouterr().err

    python_executable = python_311_executable()
    service_arguments = [
        "generate-user-service",
        "--release-document",
        str(release),
        "--configuration-document",
        str(configuration),
        "--config-path",
        str(configuration),
    ]
    if sys.version_info[:2] != (3, 11):
        assert main([*service_arguments, "--python-executable", sys.executable]) == 2
        assert "requires the exact supplied Python 3.11" in capsys.readouterr().err
    assert main([*service_arguments, "--python-executable", python_executable]) == 0
    generated = json.loads(capsys.readouterr().out)
    assert generated["command"] == "generate-user-service"
    assert set(generated["units"]) == {"race-collection.service"}
    unit = generated["units"]["race-collection.service"]
    assert f"ExecStart={python_executable} " in unit
    assert "WantedBy=default.target" in unit
    assert "timer" not in unit
    different_configuration = write_document(
        tmp_path / "different-configuration.json",
        {"not": "the authenticated configuration"},
    )
    assert (
        main(
            [
                "generate-user-service",
                "--release-document",
                str(release),
                "--configuration-document",
                str(configuration),
                "--config-path",
                str(different_configuration),
                "--python-executable",
                python_executable,
            ]
        )
        == 2
    )
    assert "configuration bytes disagree" in capsys.readouterr().err


def test_transition_and_recovery_commands_preserve_explicit_cli_identities(
    tmp_path, capsys, monkeypatch
):
    legacy = tmp_path / "legacy.db"
    with sqlite3.connect(legacy) as db:
        db.execute("CREATE TABLE race_metadata(race_id TEXT PRIMARY KEY)")
    operations = tmp_path / "operations.sqlite3"
    artifacts = tmp_path / "artifacts"
    authority = [*common(operations, legacy), "--artifacts-root", str(artifacts)]
    assert main(["migrate", *common(operations, legacy)]) == 0

    calls = {}

    def activate(self, operation_id, **arguments):
        calls["activate"] = (self.store.path, str(operation_id), arguments)
        calls["activate_clock"] = self._OperationalAuthority__clock()
        return True

    def rollback(self, operation_id, **arguments):
        calls["rollback"] = (self.store.path, str(operation_id), arguments)
        return True

    def backup(self, operation_id, **arguments):
        calls["backup"] = (
            self.store.path,
            self.artifacts.root,
            str(operation_id),
            arguments,
        )
        return ArtifactChecksum("sha256:" + "a" * 64)

    def restore(self, operation_id, **arguments):
        calls["restore"] = (
            self.store.path,
            self.artifacts.root,
            str(operation_id),
            arguments,
        )
        return True

    monkeypatch.setattr("race_collection.operator.OperationalAuthority.activate", activate)
    monkeypatch.setattr("race_collection.operator.OperationalAuthority.rollback", rollback)
    monkeypatch.setattr("race_collection.operator.RecoveryAuthority.backup", backup)
    monkeypatch.setattr("race_collection.operator.RecoveryAuthority.restore_drill", restore)

    trusted_window_start = datetime.now(timezone.utc)
    assert (
        main(
            [
                "activate",
                *authority,
                "--release-id",
                "candidate-release",
                "--boundary-day-id",
                "day_" + "b" * 32,
                "--actor",
                "operator",
                "--reason",
                "separately authorized cutover",
                "--operation-id",
                operation(20),
                "--at",
                NOW.isoformat(),
            ]
        )
        == 0
    )
    trusted_window_end = datetime.now(timezone.utc)
    assert (
        main(
            [
                "rollback",
                *authority,
                "--actor",
                "operator",
                "--reason",
                "exact legacy rollback",
                "--operation-id",
                operation(21),
                "--at",
                NOW.isoformat(),
            ]
        )
        == 0
    )

    isolated = tmp_path / "isolated"
    isolated.mkdir()
    snapshot = isolated / "snapshot.sqlite3"
    replica = tmp_path / "replica"
    recovery = [
        *authority,
        "--backup-id",
        "backup-1",
        "--snapshot",
        str(snapshot),
        "--replica-root",
        str(replica),
    ]
    assert (
        main(
            [
                "backup",
                *recovery,
                "--racing-day-id",
                "day_" + "c" * 32,
                "--operation-id",
                operation(22),
                "--at",
                NOW.isoformat(),
            ]
        )
        == 0
    )
    snapshot.write_bytes(b"isolated snapshot fixture")
    replica.mkdir()
    for offset, database in enumerate((operations, legacy), 30):
        alias = isolated / f"database-alias-{offset}.sqlite3"
        alias.hardlink_to(database)
        alias_recovery = list(recovery)
        alias_recovery[alias_recovery.index("--snapshot") + 1] = str(alias)
        assert (
            main(
                [
                    "validate-restore",
                    *alias_recovery,
                    "--drill-id",
                    f"alias-drill-{offset}",
                    "--operation-id",
                    operation(offset),
                    "--at",
                    NOW.isoformat(),
                ]
            )
            == 2
        )
    assert (
        main(
            [
                "validate-restore",
                *recovery,
                "--drill-id",
                "drill-1",
                "--operation-id",
                operation(23),
                "--at",
                NOW.isoformat(),
            ]
        )
        == 0
    )

    assert calls["activate"] == (
        operations,
        operation(20),
        {
            "release_id": "candidate-release",
            "boundary_day_id": "day_" + "b" * 32,
            "actor": "operator",
            "reason": "separately authorized cutover",
            "at": NOW,
        },
    )
    assert trusted_window_start <= calls["activate_clock"] <= trusted_window_end
    assert calls["activate_clock"] != NOW
    assert calls["rollback"] == (
        operations,
        operation(21),
        {
            "actor": "operator",
            "reason": "exact legacy rollback",
            "at": NOW,
        },
    )
    assert calls["backup"] == (
        operations,
        artifacts,
        operation(22),
        {
            "backup_id": "backup-1",
            "racing_day_id": "day_" + "c" * 32,
            "snapshot_path": snapshot,
            "replica": calls["backup"][3]["replica"],
            "at": NOW,
        },
    )
    assert calls["backup"][3]["replica"].root == replica
    assert calls["restore"] == (
        operations,
        artifacts,
        operation(23),
        {
            "drill_id": "drill-1",
            "backup_id": "backup-1",
            "snapshot_path": snapshot,
            "replica": calls["restore"][3]["replica"],
            "at": NOW,
        },
    )
    assert calls["restore"][3]["replica"].root == replica
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == {
        "command": "validate-restore",
        "operation_id": operation(23),
        "status": "ok",
        "successful": True,
    }
