"""Fail-closed operator CLI for the Phase 7 repository authority surfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import LocalArtifactStore
from .domain import ArtifactChecksum, OperationId, require_aware
from .evaluation import EvaluationAuthority, PromotionPolicy
from .operational import OperationalAuthority, ReleaseConfiguration, ReleaseManifest
from .operations import SQLiteOperationsStore
from .recovery import RecoveryAuthority


class OperatorRejected(RuntimeError):
    """An operator request is unsafe, ambiguous, or outside the supported contract."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _absolute(path: Path, label: str, *, must_exist: bool = False) -> Path:
    if not path.is_absolute():
        raise OperatorRejected(f"{label} must be an absolute path")
    try:
        resolved = path.resolve(strict=must_exist)
    except (OSError, RuntimeError) as error:
        raise OperatorRejected(f"{label} cannot be resolved exactly") from error
    if any(character in str(path) for character in ("\n", "\r", "\x00")):
        raise OperatorRejected(f"{label} contains unsafe characters")
    return resolved


def _sqlite_tables(path: Path) -> set[str]:
    try:
        with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True) as db:
            return {
                row[0]
                for row in db.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            }
    except sqlite3.Error as error:
        raise OperatorRejected("operations database is not readable SQLite") from error


def _database_paths(
    operations_path: Path,
    legacy_path: Path,
    *,
    allow_new_operations: bool,
) -> tuple[Path, Path]:
    legacy = _absolute(legacy_path, "legacy database", must_exist=True)
    if not legacy.is_file():
        raise OperatorRejected("legacy database must be an existing file")
    operations = _absolute(operations_path, "operations database")
    if operations == legacy or (
        operations_path.exists() and legacy_path.exists() and operations_path.samefile(legacy_path)
    ):
        raise OperatorRejected("legacy database cannot be used as the operations database")
    if not operations.parent.is_dir():
        raise OperatorRejected("operations database parent must already exist")
    if operations_path.is_symlink():
        raise OperatorRejected("operations database must not be a symlink")
    if not operations_path.exists():
        if not allow_new_operations:
            raise OperatorRejected("operations database does not exist")
        return operations, legacy
    if not operations.is_file():
        raise OperatorRejected("operations database must be a regular file")
    tables = _sqlite_tables(operations)
    if tables and not {"schema_migrations", "operations"} <= tables:
        raise OperatorRejected("existing database is not a Race Collection operations database")
    if not tables and not allow_new_operations:
        raise OperatorRejected("operations database is empty and unmigrated")
    return operations, legacy


def _verify_schema_identity(store: SQLiteOperationsStore) -> None:
    try:
        with store._connect() as db:
            recorded = [
                tuple(row)
                for row in db.execute(
                    "SELECT version,checksum FROM schema_migrations ORDER BY version"
                )
            ]
        expected = [
            (version, hashlib.sha256(content).hexdigest())
            for version, _, content in store._migration_scripts()
        ]
    except (OSError, sqlite3.Error) as error:
        raise OperatorRejected("operations database schema authority is unavailable") from error
    if recorded != expected or [version for version, _ in recorded] != list(range(1, 32)):
        raise OperatorRejected(
            "operations database must have the exact checked-in schema 1-31 identity"
        )


def _store(args: argparse.Namespace, *, allow_new: bool = False) -> SQLiteOperationsStore:
    operations, _ = _database_paths(
        args.operations_db,
        args.legacy_db,
        allow_new_operations=allow_new,
    )
    store = SQLiteOperationsStore(operations)
    if not allow_new:
        _verify_schema_identity(store)
    return store


def _artifact_root(path: Path, label: str, *, must_exist: bool = False) -> Path:
    resolved = _absolute(path, label, must_exist=must_exist)
    if path.is_symlink():
        raise OperatorRejected(f"{label} must not be a symlink")
    if path.exists() and not resolved.is_dir():
        raise OperatorRejected(f"{label} must be a directory")
    if not path.exists() and not resolved.parent.is_dir():
        raise OperatorRejected(f"{label} parent must already exist")
    return resolved


def _artifacts(args: argparse.Namespace) -> LocalArtifactStore:
    return LocalArtifactStore(_artifact_root(args.artifacts_root, "artifact root"))


def _document(path: Path, label: str) -> Mapping[str, Any]:
    resolved = _absolute(path, f"{label} document", must_exist=True)
    if path.is_symlink() or not resolved.is_file():
        raise OperatorRejected(f"{label} document must be a regular non-symlink file")
    try:
        content = resolved.read_bytes()
        value = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OperatorRejected(f"{label} document is unreadable or malformed") from error
    if type(value) is not dict or content != _canonical(value):
        raise OperatorRejected(f"{label} document must be exact canonical JSON")
    return value


def _configuration(path: Path) -> ReleaseConfiguration:
    value = _document(path, "configuration")
    expected = {
        "schema_version",
        "service_root",
        "artifact_root",
        "operations_database",
        "sources",
        "schedule_policy",
        "promotion_policy",
        "bundle_versions",
        "runtime_adapter",
        "runtime_input_checksum",
    }
    if set(value) != expected:
        raise OperatorRejected("configuration document has unknown or missing keys")
    configuration = ReleaseConfiguration(
        schema_version=value["schema_version"],
        service_root=value["service_root"],
        artifact_root=value["artifact_root"],
        operations_database=value["operations_database"],
        sources=tuple(value["sources"]),
        schedule_policy=value["schedule_policy"],
        promotion_policy=value["promotion_policy"],
        bundle_versions=tuple(value["bundle_versions"]),
        runtime_adapter=value["runtime_adapter"],
        runtime_input_checksum=ArtifactChecksum(value["runtime_input_checksum"]),
    )
    if configuration.document() != value:
        raise OperatorRejected("configuration document disagrees with its typed identity")
    return configuration


def _release(path: Path) -> ReleaseManifest:
    value = _document(path, "release")
    expected = {
        "schema_version",
        "release_id",
        "code_commit",
        "config_checksum",
        "database_schema",
        "artifact_contract",
        "policy_version",
        "supported_bundle_versions",
        "service_root",
    }
    if set(value) != expected:
        raise OperatorRejected("release document has unknown or missing keys")
    manifest = ReleaseManifest(
        schema_version=value["schema_version"],
        release_id=value["release_id"],
        code_commit=value["code_commit"],
        config_checksum=ArtifactChecksum(value["config_checksum"]),
        database_schema=value["database_schema"],
        artifact_contract=value["artifact_contract"],
        policy_version=value["policy_version"],
        supported_bundle_versions=tuple(value["supported_bundle_versions"]),
        service_root=value["service_root"],
    )
    if manifest.document() != value:
        raise OperatorRejected("release document disagrees with its typed identity")
    return manifest


def _policy(path: Path) -> PromotionPolicy:
    value = _document(path, "policy")
    expected = {field.name for field in fields(PromotionPolicy)}
    if set(value) != expected:
        raise OperatorRejected("policy document has unknown or missing keys")
    policy = PromotionPolicy(**value)
    if asdict(policy) != value:
        raise OperatorRejected("policy document disagrees with its typed identity")
    return policy


def _timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
        require_aware(parsed, "operator timestamp")
        return parsed
    except (TypeError, ValueError) as error:
        raise OperatorRejected("operator timestamp must be an aware ISO instant") from error


def _operation(value: str) -> OperationId:
    try:
        return OperationId(value)
    except ValueError as error:
        raise OperatorRejected("operation identity is malformed") from error


def _authority(
    args: argparse.Namespace,
) -> tuple[SQLiteOperationsStore, LocalArtifactStore, OperationalAuthority]:
    store = _store(args)
    artifacts = _artifacts(args)
    return store, artifacts, OperationalAuthority(store, artifacts)


def _add_databases(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--operations-db", required=True, type=Path)
    parser.add_argument("--legacy-db", required=True, type=Path)


def _add_artifacts(parser: argparse.ArgumentParser) -> None:
    _add_databases(parser)
    parser.add_argument("--artifacts-root", required=True, type=Path)


def _add_operation(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--operation-id", required=True)
    parser.add_argument("--at", required=True)


def _add_actor(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--actor", required=True)
    parser.add_argument("--reason", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="race-collection-operator",
        description="Apply one explicit Phase 7 authority command to a separate operations DB.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    migrate = commands.add_parser("migrate")
    _add_databases(migrate)

    for name in ("register-config", "register-release", "register-policy"):
        command = commands.add_parser(name)
        _add_artifacts(command)
        _add_operation(command)
        command.add_argument("--document", required=True, type=Path)

    initialize = commands.add_parser("initialize-legacy")
    _add_artifacts(initialize)
    _add_operation(initialize)
    _add_actor(initialize)
    initialize.add_argument("--release-id", required=True)

    for name in ("authorize-observation", "revoke-observation"):
        command = commands.add_parser(name)
        _add_artifacts(command)
        _add_operation(command)
        _add_actor(command)
        command.add_argument("--release-id", required=True)

    activate = commands.add_parser("activate")
    _add_artifacts(activate)
    _add_operation(activate)
    _add_actor(activate)
    activate.add_argument("--release-id", required=True)
    activate.add_argument("--boundary-day-id", required=True)

    rollback = commands.add_parser("rollback")
    _add_artifacts(rollback)
    _add_operation(rollback)
    _add_actor(rollback)

    backup = commands.add_parser("backup")
    _add_artifacts(backup)
    _add_operation(backup)
    backup.add_argument("--backup-id", required=True)
    backup.add_argument("--racing-day-id", required=True)
    backup.add_argument("--snapshot", required=True, type=Path)
    backup.add_argument("--replica-root", required=True, type=Path)

    restore = commands.add_parser("validate-restore")
    _add_artifacts(restore)
    _add_operation(restore)
    restore.add_argument("--drill-id", required=True)
    restore.add_argument("--backup-id", required=True)
    restore.add_argument("--snapshot", required=True, type=Path)
    restore.add_argument("--replica-root", required=True, type=Path)

    service = commands.add_parser("generate-user-service")
    service.add_argument("--release-document", required=True, type=Path)
    service.add_argument("--configuration-document", required=True, type=Path)
    service.add_argument("--config-path", required=True)
    service.add_argument("--python-executable", required=True)
    return parser


def _recovery_paths(
    args: argparse.Namespace,
    store: SQLiteOperationsStore,
    artifacts: LocalArtifactStore,
    *,
    snapshot_must_exist: bool,
) -> tuple[Path, LocalArtifactStore]:
    def contains(parent: Path, child: Path) -> bool:
        try:
            child.relative_to(parent)
        except ValueError:
            return False
        return True

    snapshot = _absolute(
        args.snapshot,
        "isolated snapshot",
        must_exist=snapshot_must_exist,
    )
    if args.snapshot.is_symlink():
        raise OperatorRejected("isolated snapshot must not be a symlink")
    database_paths = (store.path.resolve(), args.legacy_db.resolve(strict=True))
    try:
        database_alias = snapshot_must_exist and any(
            snapshot.samefile(database) for database in database_paths
        )
    except OSError as error:
        raise OperatorRejected("database and snapshot identities cannot be compared") from error
    if snapshot in set(database_paths) or database_alias:
        raise OperatorRejected("isolated snapshot must not alias either database")
    if not snapshot_must_exist and not snapshot.parent.is_dir():
        raise OperatorRejected("isolated snapshot parent must already exist")
    if snapshot_must_exist and not snapshot.is_file():
        raise OperatorRejected("isolated snapshot must be an existing file")
    replica_root = _artifact_root(
        args.replica_root,
        "replica artifact root",
        must_exist=snapshot_must_exist,
    )
    if contains(artifacts.root, replica_root) or contains(replica_root, artifacts.root):
        raise OperatorRejected("replica artifact root must be isolated from the primary")
    if contains(artifacts.root, snapshot):
        raise OperatorRejected("isolated snapshot must be outside the primary artifact root")
    return snapshot, LocalArtifactStore(replica_root)


def _dispatch(args: argparse.Namespace) -> tuple[Mapping[str, Any], int]:
    if args.command == "migrate":
        store = _store(args, allow_new=True)
        store.migrate()
        _verify_schema_identity(store)
        return {"command": "migrate", "schema_version": 31, "status": "ok"}, 0

    if args.command == "generate-user-service":
        manifest = _release(args.release_document)
        configuration = _configuration(args.configuration_document)
        units = OperationalAuthority.generate_units(
            manifest,
            configuration,
            config_path=args.config_path,
            python_executable=args.python_executable,
        )
        return {"command": args.command, "status": "ok", "units": units}, 0

    at = _timestamp(args.at)
    operation_id = _operation(args.operation_id)
    store, artifacts, authority = _authority(args)
    base = {
        "command": args.command,
        "operation_id": str(operation_id),
        "status": "ok",
    }

    if args.command == "register-config":
        configuration = _configuration(args.document)
        if (
            configuration.operations_database != str(store.path)
            or Path(configuration.artifact_root).resolve() != artifacts.root
        ):
            raise OperatorRejected(
                "configuration database or artifact root disagrees with explicit CLI paths"
            )
        checksum = authority.register_configuration(operation_id, configuration, at)
        return {**base, "checksum": str(checksum)}, 0
    if args.command == "register-release":
        checksum = authority.register_release(operation_id, _release(args.document), at)
        return {**base, "checksum": str(checksum)}, 0
    if args.command == "register-policy":
        checksum = EvaluationAuthority(store, artifacts).register_policy(
            operation_id,
            _policy(args.document),
            at,
        )
        return {**base, "checksum": str(checksum)}, 0
    if args.command == "initialize-legacy":
        changed = authority.initialize_legacy_authority(
            operation_id,
            release_id=args.release_id,
            actor=args.actor,
            reason=args.reason,
            at=at,
        )
        return {**base, "changed": changed}, 0
    if args.command == "authorize-observation":
        changed = authority.authorize_observation(
            operation_id,
            candidate_release_id=args.release_id,
            actor=args.actor,
            reason=args.reason,
            at=at,
        )
        return {**base, "changed": changed}, 0
    if args.command == "revoke-observation":
        changed = authority.revoke_observation(
            operation_id,
            candidate_release_id=args.release_id,
            actor=args.actor,
            reason=args.reason,
            at=at,
        )
        return {**base, "changed": changed}, 0
    if args.command == "activate":
        changed = authority.activate(
            operation_id,
            release_id=args.release_id,
            boundary_day_id=args.boundary_day_id,
            actor=args.actor,
            reason=args.reason,
            at=at,
        )
        return {**base, "changed": changed}, 0
    if args.command == "rollback":
        changed = authority.rollback(
            operation_id,
            actor=args.actor,
            reason=args.reason,
            at=at,
        )
        return {**base, "changed": changed}, 0
    if args.command == "backup":
        snapshot, replica = _recovery_paths(
            args,
            store,
            artifacts,
            snapshot_must_exist=False,
        )
        checksum = RecoveryAuthority(store, artifacts).backup(
            operation_id,
            backup_id=args.backup_id,
            racing_day_id=args.racing_day_id,
            snapshot_path=snapshot,
            replica=replica,
            at=at,
        )
        return {**base, "database_checksum": str(checksum)}, 0
    if args.command == "validate-restore":
        snapshot, replica = _recovery_paths(
            args,
            store,
            artifacts,
            snapshot_must_exist=True,
        )
        successful = RecoveryAuthority(store, artifacts).restore_drill(
            operation_id,
            drill_id=args.drill_id,
            backup_id=args.backup_id,
            snapshot_path=snapshot,
            replica=replica,
            at=at,
        )
        return {
            **base,
            "status": "ok" if successful else "rejected",
            "successful": successful,
        }, (0 if successful else 3)
    raise OperatorRejected("unsupported operator command")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result, status = _dispatch(args)
    except Exception as error:
        print(f"race-collection-operator rejected: {error}", file=sys.stderr)
        return 2
    print(_canonical(result).decode())
    return status


if __name__ == "__main__":
    raise SystemExit(main())
