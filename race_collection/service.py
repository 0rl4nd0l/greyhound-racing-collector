"""Runnable composition root for the single authoritative Race Collection Service."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Callable, Protocol, Sequence

from .artifacts import ArtifactStoreError, LocalArtifactStore
from .domain import ArtifactChecksum, OperationId
from .operational import (
    ApplicationCommand,
    ClosedCommandDispatcher,
    OperationalAuthority,
    OperationalRejected,
    PhaseHandlerRegistration,
    RaceCollectionService,
    ReleaseConfiguration,
    _canonical,
    _checksum,
    verify_release_authority,
)
from .operations import SQLiteOperationsStore


class ServiceUnavailable(RuntimeError):
    """The release cannot form one complete, trusted runtime composition."""


@dataclass(frozen=True, slots=True)
class RacingDayCycle:
    """One immutable nine-command Racing Day plan supplied by a trusted adapter."""

    racing_day_id: str
    commands: tuple[ApplicationCommand, ...]
    plan_operation_id: OperationId
    advancement_operation_ids: tuple[OperationId, ...]
    at: datetime

    def __post_init__(self) -> None:
        phases = tuple(command.phase for command in self.commands)
        if (
            not isinstance(self.racing_day_id, str)
            or not self.racing_day_id.strip()
            or phases != RaceCollectionService.ORDER
            or len(self.advancement_operation_ids) != len(self.commands)
            or any(command.racing_day_id != self.racing_day_id for command in self.commands)
            or len(
                {
                    self.plan_operation_id,
                    *self.advancement_operation_ids,
                    *(command.operation_id for command in self.commands),
                }
            )
            != 19
            or self.at.tzinfo is None
            or self.at.utcoffset() is None
        ):
            raise ValueError("runtime cycle is not one exact ordered Racing Day plan")


class RuntimeAdapter(Protocol):
    """Live capability plugin; it supplies bindings and plans, never a scheduler."""

    def registrations(self) -> Sequence[PhaseHandlerRegistration]:
        """Return exactly one trusted handler for every closed command type."""

    def next_cycle(self, *, now: datetime) -> RacingDayCycle | None:
        """Return the next plan, rehydrating any durable command IDs for a partial day."""

    def close(self) -> None:
        """Idempotently release every adapter-owned resource."""


AdapterFactory = Callable[
    [ReleaseConfiguration, SQLiteOperationsStore, LocalArtifactStore],
    RuntimeAdapter,
]


def load_configuration(path: Path) -> tuple[ReleaseConfiguration, bytes]:
    content = path.read_bytes()
    try:
        document = json.loads(content)
        configuration = ReleaseConfiguration(
            schema_version=document["schema_version"],
            service_root=document["service_root"],
            artifact_root=document["artifact_root"],
            operations_database=document["operations_database"],
            sources=tuple(document["sources"]),
            schedule_policy=document["schedule_policy"],
            promotion_policy=document["promotion_policy"],
            bundle_versions=tuple(document["bundle_versions"]),
            runtime_adapter=document["runtime_adapter"],
            runtime_input_checksum=ArtifactChecksum(document["runtime_input_checksum"]),
        )
    except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ServiceUnavailable("release configuration is malformed or unsupported") from error
    if content != _canonical(configuration.document()):
        raise ServiceUnavailable("release configuration bytes are not canonical")
    return configuration, content


def _load_factory(binding: str) -> AdapterFactory:
    module_name, factory_name = binding.split(":", 1)
    try:
        module: ModuleType = importlib.import_module(module_name)
        factory = getattr(module, factory_name)
    except (ImportError, AttributeError) as error:
        raise ServiceUnavailable(f"runtime adapter {binding!r} is unavailable") from error
    if not callable(factory):
        raise ServiceUnavailable(f"runtime adapter {binding!r} is not callable")
    return factory


def _git_release_output(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ServiceUnavailable("declared release Git identity cannot be proven") from error
    return completed.stdout.strip()


def verify_release_source_identity(
    service_root: Path | str,
    code_commit: str,
    *,
    source_file: Path | None = None,
) -> None:
    """Bind resolved source and executable bytes to one clean immutable Git release."""
    try:
        root = Path(service_root).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ServiceUnavailable("declared release root is unavailable") from error
    if not root.is_dir():
        raise ServiceUnavailable("declared release root is unavailable")

    source = (source_file or Path(__file__)).resolve(strict=True)
    try:
        source_relative = source.relative_to(root)
    except ValueError as error:
        raise ServiceUnavailable(
            "resolved service source is outside the approved release root"
        ) from error
    if source_relative.as_posix() != "race_collection/service.py":
        raise ServiceUnavailable("resolved service source is outside the approved release module")

    executable = root / "bin" / "race-collection-service"
    if executable.is_symlink():
        raise ServiceUnavailable("release executable must not be a retargetable symlink")
    try:
        resolved_executable = executable.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ServiceUnavailable("release executable is unavailable") from error
    if resolved_executable != executable or not executable.is_file():
        raise ServiceUnavailable("release executable is outside the approved release root")
    if executable.stat().st_mode & 0o111 == 0:
        raise ServiceUnavailable("release executable is not executable")

    if _git_release_output(root, "rev-parse", "--show-toplevel") != str(root):
        raise ServiceUnavailable("approved release root is not the exact Git worktree root")
    try:
        _git_release_output(root, "cat-file", "-e", f"{code_commit}^{{commit}}")
    except ServiceUnavailable as error:
        raise ServiceUnavailable("declared release commit does not exist") from error
    if _git_release_output(root, "rev-parse", "HEAD") != code_commit:
        raise ServiceUnavailable("checked-out release does not match the declared commit")
    if _git_release_output(root, "rev-parse", "HEAD^{tree}") != _git_release_output(
        root, "rev-parse", f"{code_commit}^{{tree}}"
    ):
        raise ServiceUnavailable("checked-out release tree does not match the declared commit")

    for relative, path, label in (
        (source_relative.as_posix(), source, "service source"),
        ("bin/race-collection-service", executable, "release executable"),
    ):
        expected_blob = _git_release_output(root, "rev-parse", f"{code_commit}:{relative}")
        actual_blob = _git_release_output(root, "hash-object", str(path))
        if actual_blob != expected_blob:
            raise ServiceUnavailable(f"{label} bytes do not match the declared release")
    executable_entry = _git_release_output(
        root, "ls-tree", code_commit, "--", "bin/race-collection-service"
    )
    if not executable_entry.startswith("100755 blob "):
        raise ServiceUnavailable("release executable mode does not match the declared release")
    if _git_release_output(root, "status", "--porcelain", "--untracked-files=all"):
        raise ServiceUnavailable("declared release worktree is dirty or tampered")


def _operation_id(label: str) -> OperationId:
    digest = hashlib.sha256(label.encode()).hexdigest()[:32]
    return OperationId(f"op_{digest}")


class ServiceComposition:
    """One store, one dispatcher, one scheduler lease, and one command surface."""

    def __init__(
        self,
        configuration: ReleaseConfiguration,
        store: SQLiteOperationsStore,
        artifacts: LocalArtifactStore,
        adapter: RuntimeAdapter,
        *,
        owner: str,
        token: str,
        lease_ttl: timedelta,
        clock: Callable[[], datetime] | None = None,
        release_id: str | None = None,
        mode: str | None = None,
    ):
        if (
            not isinstance(owner, str)
            or not owner.strip()
            or not isinstance(token, str)
            or not token.strip()
            or lease_ttl <= timedelta(0)
        ):
            raise ValueError("owner, token and positive lease TTL are required")
        try:
            dispatcher = ClosedCommandDispatcher(tuple(adapter.registrations()))
        except (TypeError, ValueError) as error:
            raise ServiceUnavailable("runtime adapter handler set is incomplete") from error
        if not callable(getattr(adapter, "close", None)):
            raise ServiceUnavailable("runtime adapter close contract is unavailable")
        self.configuration = configuration
        self.store = store
        self.artifacts = artifacts
        self.adapter = adapter
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._last_timestamp: datetime | None = None
        self.authority = OperationalAuthority(
            store,
            artifacts,
            command_executor=dispatcher,
            clock=self.trusted_timestamp,
        )
        self.owner = owner
        self.token = token
        self.lease_ttl = lease_ttl
        self.release_id = release_id
        self.mode = mode
        self.generation: int | None = None
        self._closed = False

    def close(self) -> None:
        """Idempotently release adapter-owned resources."""
        if self._closed:
            return
        self.adapter.close()
        self._closed = True

    def _revalidate_release_mode(self) -> None:
        if self.release_id is None or self.mode is None:
            return
        with self.store._connect() as db:
            pointer = db.execute(
                "SELECT release_id,authority,legacy_preserved "
                "FROM phase7_release_pointer WHERE singleton=1"
            ).fetchone()
            observation = db.execute(
                "SELECT candidate_release_id,action FROM "
                "phase7_observation_authority_events ORDER BY event_id DESC LIMIT 1"
            ).fetchone()
            valid = (
                self.mode == "active"
                and pointer is not None
                and pointer["authority"] == "race_collection_service"
                and pointer["release_id"] == self.release_id
            ) or (
                self.mode == "observation"
                and pointer is not None
                and pointer["authority"] == "legacy"
                and pointer["legacy_preserved"]
                and observation is not None
                and observation["action"] == "authorize"
                and observation["candidate_release_id"] == self.release_id
            )
            if not valid:
                raise OperationalRejected("service release mode is no longer authorized")
            verify_release_authority(db, self.artifacts, self.release_id)

    def trusted_timestamp(self) -> datetime:
        """Return the public monotonic authority time for all service work."""
        now = self.clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ServiceUnavailable("trusted service clock must return an aware timestamp")
        now = now.astimezone(timezone.utc)
        if self._last_timestamp is not None and now <= self._last_timestamp:
            now = self._last_timestamp + timedelta(microseconds=1)
        self._last_timestamp = now
        return now

    def maintain_lease(self, now: datetime) -> int:
        if self.generation is None:
            self.generation = self.authority.acquire_lease(
                _operation_id(f"acquire:{self.owner}:{self.token}:{now.isoformat()}"),
                owner=self.owner,
                token=self.token,
                now=now,
                ttl=self.lease_ttl,
            )
            return self.generation
        self.authority.renew_lease(
            _operation_id(
                f"heartbeat:{self.owner}:{self.token}:{self.generation}:{now.isoformat()}"
            ),
            token=self.token,
            generation=self.generation,
            now=now,
            ttl=self.lease_ttl,
        )
        return self.generation

    def run_cycle(self, cycle: RacingDayCycle) -> tuple[object, ...]:
        self._revalidate_release_mode()
        now = self.trusted_timestamp()
        if cycle.at.astimezone(timezone.utc) > now:
            raise OperationalRejected("runtime cycle schedule time is in the future")
        generation = self.maintain_lease(now)
        with self.store._connect() as read:
            existing_plan = read.execute(
                "SELECT p.planned_at,p.operation_id,p.lease_generation,o.kind "
                "FROM phase7_day_command_plan p "
                "JOIN operations o ON o.operation_id=p.operation_id "
                "WHERE p.racing_day_id=? ORDER BY p.phase_ordinal LIMIT 1",
                (cycle.racing_day_id,),
            ).fetchone()
        plan_at = (
            datetime.fromisoformat(existing_plan["planned_at"])
            if existing_plan is not None
            and existing_plan["operation_id"] == str(cycle.plan_operation_id)
            else now
        )
        planning_operation_id = cycle.plan_operation_id
        if (
            existing_plan is not None
            and existing_plan["lease_generation"] != generation
            and existing_plan["kind"]
            in ("phase7_plan_racing_day", "phase7_migrate_v27_day_command_plan")
        ):
            planning_operation_id = _operation_id(
                f"adopt:{cycle.racing_day_id}:{existing_plan['operation_id']}:"
                f"{generation}:{self.token}"
            )
            if str(planning_operation_id) == existing_plan["operation_id"]:
                planning_operation_id = _operation_id(
                    f"adopt-distinct:{cycle.racing_day_id}:{existing_plan['operation_id']}:"
                    f"{generation}:{self.token}"
                )
        self.authority.plan_racing_day(
            planning_operation_id,
            racing_day_id=cycle.racing_day_id,
            lease_token=self.token,
            lease_generation=generation,
            commands=cycle.commands,
            at=plan_at,
        )
        service = RaceCollectionService(
            self.authority,
            token=self.token,
            generation=generation,
        )
        with self.store._connect() as read:
            completed = {
                row["phase_ordinal"]: row
                for row in read.execute(
                    "SELECT progress.*,receipt.result_json AS receipt_result_json,"
                    "receipt.result_checksum AS receipt_result_checksum,"
                    "receipt.racing_day_id AS receipt_racing_day_id,"
                    "receipt.phase_name AS receipt_phase_name,"
                    "receipt.committed_at AS receipt_committed_at,"
                    "plan.phase_ordinal AS plan_phase_ordinal,"
                    "plan.phase_name AS plan_phase_name,"
                    "plan.command_operation_id AS plan_command_operation_id "
                    "FROM phase7_scheduler_progress progress "
                    "LEFT JOIN phase7_application_command_receipts receipt "
                    "ON receipt.command_operation_id=progress.command_operation_id "
                    "LEFT JOIN phase7_day_command_plan plan "
                    "ON plan.racing_day_id=progress.racing_day_id "
                    "AND plan.phase_ordinal=progress.phase_ordinal "
                    "WHERE progress.racing_day_id=? "
                    "ORDER BY phase_ordinal",
                    (cycle.racing_day_id,),
                )
            }
        if tuple(sorted(completed)) != tuple(range(1, len(completed) + 1)):
            raise OperationalRejected("completed scheduler progress is not a contiguous prefix")
        results = []
        for ordinal, (command, advancement_id) in enumerate(
            zip(cycle.commands, cycle.advancement_operation_ids, strict=True),
            1,
        ):
            prior = completed.get(ordinal)
            if prior is not None:
                if (
                    prior["phase_name"] != command.phase
                    or prior["command_operation_id"] != str(command.operation_id)
                    or prior["racing_day_id"] != cycle.racing_day_id
                    or prior["phase_ordinal"] != ordinal
                    or prior["receipt_racing_day_id"] != cycle.racing_day_id
                    or prior["receipt_phase_name"] != command.phase
                    or prior["receipt_committed_at"] != prior["completed_at"]
                    or prior["plan_phase_ordinal"] != ordinal
                    or prior["plan_phase_name"] != command.phase
                    or prior["plan_command_operation_id"] != str(command.operation_id)
                    or prior["receipt_result_json"] != prior["result_json"]
                    or prior["receipt_result_checksum"] != prior["result_checksum"]
                    or _canonical(json.loads(prior["result_json"])).decode() != prior["result_json"]
                    or str(_checksum(json.loads(prior["result_json"]))) != prior["result_checksum"]
                ):
                    raise OperationalRejected(
                        "completed prefix disagrees with rehydrated command identities"
                    )
                results.append(json.loads(prior["result_json"]))
                continue
            renewed_at = self.trusted_timestamp()
            self.authority.renew_lease(
                _operation_id(
                    f"renew:{self.owner}:{self.token}:{self.generation}:"
                    f"{cycle.racing_day_id}:{ordinal}"
                ),
                token=self.token,
                generation=self.generation,
                now=renewed_at,
                ttl=self.lease_ttl,
            )
            results.append(
                service.advance(
                    advancement_id,
                    racing_day_id=cycle.racing_day_id,
                    phase=command.phase,
                    now=renewed_at,
                    command=command,
                )
            )
        return tuple(results)


def compose(
    config_path: Path,
    *,
    owner: str,
    token: str,
    lease_ttl: timedelta,
) -> ServiceComposition:
    configuration, content = load_configuration(config_path)
    database_path = Path(configuration.operations_database)
    artifact_path = Path(configuration.artifact_root)
    if not database_path.is_file() or not artifact_path.is_dir():
        raise ServiceUnavailable("operations database or immutable artifact store is unavailable")
    store = SQLiteOperationsStore(database_path)
    artifacts = LocalArtifactStore(artifact_path)
    checksum = f"sha256:{hashlib.sha256(content).hexdigest()}"
    with store._connect() as db:
        releases = {
            row["release_id"]
            for row in db.execute(
                "SELECT release_id FROM phase7_release_manifests WHERE config_checksum=?",
                (checksum,),
            )
        }
        pointer = db.execute(
            "SELECT release_id,authority,legacy_preserved "
            "FROM phase7_release_pointer WHERE singleton=1"
        ).fetchone()
        observation = db.execute(
            "SELECT candidate_release_id,action FROM "
            "phase7_observation_authority_events ORDER BY event_id DESC LIMIT 1"
        ).fetchone()
        if (
            pointer is not None
            and pointer["authority"] == "race_collection_service"
            and pointer["release_id"] in releases
        ):
            release_id, mode = pointer["release_id"], "active"
        elif (
            pointer is not None
            and pointer["authority"] == "legacy"
            and pointer["legacy_preserved"]
            and observation is not None
            and observation["action"] == "authorize"
            and observation["candidate_release_id"] in releases
        ):
            release_id, mode = observation["candidate_release_id"], "observation"
        else:
            raise ServiceUnavailable("configuration release lacks active or observation authority")
        try:
            release = verify_release_authority(db, artifacts, release_id)
            verify_release_source_identity(
                configuration.service_root,
                release["code_commit"],
            )
        except (ArtifactStoreError, OperationalRejected, ValueError) as error:
            raise ServiceUnavailable("release authority verification failed") from error
    factory = _load_factory(configuration.runtime_adapter)
    adapter: RuntimeAdapter | None = None
    try:
        adapter = factory(configuration, store, artifacts)
        return ServiceComposition(
            configuration,
            store,
            artifacts,
            adapter,
            owner=owner,
            token=token,
            lease_ttl=lease_ttl,
            release_id=release_id,
            mode=mode,
        )
    except Exception as error:
        if adapter is not None:
            try:
                adapter.close()
            except Exception:
                pass
        raise ServiceUnavailable("runtime adapter composition failed") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="race-collection-service",
        description="Run the single authoritative Race Collection Service.",
    )
    parser.add_argument("--config", required=True, type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--once", action="store_true", help="run at most one due Racing Day cycle")
    modes.add_argument("--continuous", action="store_true", help="wait for and run due cycles")
    parser.add_argument("--owner", default="race-collection-service")
    parser.add_argument("--lease-seconds", type=int, default=300)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    composition_loader: Callable[..., ServiceComposition] = compose,
    token_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
) -> int:
    args = _parser().parse_args(argv)
    if args.lease_seconds < 1 or args.poll_seconds <= 0:
        _parser().error("lease and poll intervals must be positive")
    token = token_factory()
    composition: ServiceComposition | None = None
    failed = False
    try:
        composition = composition_loader(
            args.config,
            owner=args.owner,
            token=token,
            lease_ttl=timedelta(seconds=args.lease_seconds),
        )
        if args.once:
            cycle = composition.adapter.next_cycle(now=composition.trusted_timestamp())
            if cycle is not None:
                composition.run_cycle(cycle)
            composition.close()
            return 0
        stopping = False

        def stop(_signum: int, _frame: object) -> None:
            nonlocal stopping
            stopping = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        while not stopping:
            polled_at = composition.trusted_timestamp()
            cycle = composition.adapter.next_cycle(now=polled_at)
            if cycle is not None:
                composition.run_cycle(cycle)
            elif composition.generation is not None:
                composition.maintain_lease(polled_at)
            if not stopping:
                time.sleep(args.poll_seconds)
        composition.close()
        return 0
    except Exception:
        failed = True
        print("race-collection-service unavailable", file=sys.stderr)
        return 69
    finally:
        if composition is not None:
            try:
                composition.close()
            except Exception:
                if not failed and not composition._closed:
                    print("race-collection-service unavailable", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
