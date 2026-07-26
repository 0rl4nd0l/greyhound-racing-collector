"""Phase 7 operational authority; reports are projections, never workflow truth."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .artifacts import ArtifactStore, ArtifactStoreError
from .domain import (
    ADAPTIVE_ODDS_MAX_LATE,
    ADAPTIVE_ODDS_TIMING_POLICY,
    ArtifactChecksum,
    OperationId,
    next_odds_attempt_at,
    require_aware,
)
from .model_bundle import SUPPORTED_FORECAST_CONTRACTS
from .operations import (
    BarrierNotSatisfied,
    ConflictingOperation,
    SQLiteOperationsStore,
    iso_timestamp,
)


class OperationalRejected(RuntimeError):
    """An operational command failed closed."""


@dataclass(frozen=True, slots=True)
class DayForecastCohortMember:
    role: str
    bundle_id: str
    bundle_checksum: ArtifactChecksum
    service_run_id: OperationId
    forecast_operations: tuple[tuple[str, OperationId], ...]

    def __post_init__(self) -> None:
        if type(self.role) is not str or self.role not in {"champion", "challenger"}:
            raise ValueError("day forecast cohort role is invalid")
        _strict_nonempty(self.bundle_id, "bundle identity")
        if not isinstance(self.bundle_checksum, ArtifactChecksum):
            raise ValueError("day forecast bundle checksum is invalid")
        if not isinstance(self.service_run_id, OperationId):
            raise ValueError("day forecast service run identity is invalid")
        if type(self.forecast_operations) is not tuple:
            raise ValueError("day forecast operations must be an immutable tuple")
        if not self.forecast_operations:
            raise ValueError("day forecast member requires race forecast operations")
        if any(
            type(binding) is not tuple
            or len(binding) != 2
            or type(binding[0]) is not str
            or not binding[0].strip()
            or not isinstance(binding[1], OperationId)
            for binding in self.forecast_operations
        ):
            raise ValueError("day forecast operation binding is invalid")
        races = [race_id for race_id, _ in self.forecast_operations]
        operations = [operation for _, operation in self.forecast_operations]
        if len(races) != len(set(races)) or len(operations) != len(set(operations)):
            raise ValueError("day forecast operations must be unique")


def _strict_nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _strict_unique_text(values: Any, field: str) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{field} must be a non-empty tuple")
    checked = tuple(_strict_nonempty(value, field) for value in values)
    if len(set(checked)) != len(checked):
        raise ValueError(f"{field} must contain unique values")
    return checked


@dataclass(frozen=True, slots=True)
class DiscoverProgramme:
    source: str
    programme_checksum: ArtifactChecksum

    def __post_init__(self) -> None:
        _strict_nonempty(self.source, "programme source")


@dataclass(frozen=True, slots=True)
class CollectCardsAndForm:
    pass


@dataclass(frozen=True, slots=True)
class CollectAdaptiveOdds:
    pass


@dataclass(frozen=True, slots=True)
class CloseAndSeal:
    pass


@dataclass(frozen=True, slots=True)
class CommitDeferredPrediction:
    pass


@dataclass(frozen=True, slots=True)
class CollectResults:
    pass


@dataclass(frozen=True, slots=True)
class JoinTrainingExamples:
    pass


@dataclass(frozen=True, slots=True)
class ReconcileRacingDay:
    pass


@dataclass(frozen=True, slots=True)
class RequestTraining:
    request_id: str
    request_operation_id: OperationId
    authorization_operation_id: OperationId
    binding_operation_id: OperationId

    def __post_init__(self) -> None:
        _strict_nonempty(self.request_id, "training request identity")
        if (
            len(
                {
                    self.request_operation_id,
                    self.authorization_operation_id,
                    self.binding_operation_id,
                }
            )
            != 3
        ):
            raise ValueError(
                "training request authorization, producer, and binding operations must be distinct"
            )


CommandPayload = (
    DiscoverProgramme
    | CollectCardsAndForm
    | CollectAdaptiveOdds
    | CloseAndSeal
    | CommitDeferredPrediction
    | CollectResults
    | JoinTrainingExamples
    | ReconcileRacingDay
    | RequestTraining
)

COMMAND_PHASES = {
    DiscoverProgramme: "discover_programme",
    CollectCardsAndForm: "collect_cards_and_form",
    CollectAdaptiveOdds: "collect_adaptive_odds",
    CloseAndSeal: "close_and_seal",
    CommitDeferredPrediction: "deferred_prediction",
    CollectResults: "collect_results",
    JoinTrainingExamples: "join_training_examples",
    ReconcileRacingDay: "reconcile",
    RequestTraining: "request_training",
}


@dataclass(frozen=True, slots=True)
class ApplicationCommand:
    """Closed immutable application command shared by every adapter."""

    operation_id: OperationId
    racing_day_id: str
    payload: CommandPayload

    def __post_init__(self) -> None:
        if type(self.payload) not in COMMAND_PHASES:
            raise ValueError("typed application command is incomplete")
        _strict_nonempty(self.racing_day_id, "racing day identity")

    @property
    def phase(self) -> str:
        return COMMAND_PHASES[type(self.payload)]

    def payload_document(self) -> Mapping[str, Any]:
        payload = self.payload
        if isinstance(payload, DiscoverProgramme):
            return {
                "type": "discover_programme",
                "source": payload.source,
                "programme_checksum": str(payload.programme_checksum),
            }
        if isinstance(payload, RequestTraining):
            return {
                "type": "request_training",
                "request_id": payload.request_id,
                "request_operation_id": str(payload.request_operation_id),
                "authorization_operation_id": str(payload.authorization_operation_id),
                "binding_operation_id": str(payload.binding_operation_id),
            }
        return {"type": COMMAND_PHASES[type(payload)]}

    def payload_sha256(self) -> str:
        return hashlib.sha256(_canonical(self.payload_document())).hexdigest()


class ApplicationCommandExecutor(Protocol):
    """Trusted composition-root executor; adapters can never provide one per call."""

    def execute(self, command: ApplicationCommand, *, at: datetime) -> None:
        """Invoke the existing Phase 1--6 application API for ``command``."""


PhaseHandler = Callable[[ApplicationCommand, datetime], None]


@dataclass(frozen=True, slots=True)
class PhaseHandlerRegistration:
    """One composition-root binding; never accepted through an adapter command."""

    payload_type: type
    handler: PhaseHandler


class ClosedCommandDispatcher:
    """Fixed, exhaustive typed dispatcher installed once at the composition root."""

    def __init__(self, registrations: Sequence[PhaseHandlerRegistration]):
        handlers: dict[type, PhaseHandler] = {}
        supported = set(COMMAND_PHASES)
        for registration in registrations:
            if registration.payload_type not in supported:
                raise ValueError("unknown application-command handler type")
            if registration.payload_type in handlers:
                raise ValueError("duplicate application-command handler")
            if not callable(registration.handler):
                raise ValueError("application-command handler must be callable")
            handlers[registration.payload_type] = registration.handler
        missing = supported - set(handlers)
        if missing:
            names = ",".join(sorted(item.__name__ for item in missing))
            raise ValueError(f"missing application-command handlers: {names}")
        self.__handlers = handlers

    def execute(self, command: ApplicationCommand, *, at: datetime) -> None:
        handler = self.__handlers.get(type(command.payload))
        if handler is None:  # pragma: no cover - constructor and command guards are exhaustive
            raise OperationalRejected("unsupported application command")
        handler(command, at)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _checksum(value: Any) -> ArtifactChecksum:
    return ArtifactChecksum(f"sha256:{hashlib.sha256(_canonical(value)).hexdigest()}")


def _derived_operation_id(namespace: str, *parts: str) -> OperationId:
    digest = hashlib.sha256(_canonical({"namespace": namespace, "parts": list(parts)})).hexdigest()[
        :32
    ]
    return OperationId(f"op_{digest}")


def _adaptive_odds_history_complete(
    odds: Sequence[Mapping[str, Any]],
    *,
    discovery_at: datetime,
    scheduled_jump: datetime,
    cutoff: datetime,
) -> bool:
    """Validate due attempts against the canonical adaptive cadence and retry policy."""
    require_aware(discovery_at, "discovery_at")
    if not odds:
        return False
    try:
        due_times = [datetime.fromisoformat(row["scheduled_due_at"]) for row in odds]
        attempt_times = [datetime.fromisoformat(row["attempted_at"]) for row in odds]
        timing_policies = [row["timing_policy"] for row in odds]
        statuses = [row["status"] for row in odds]
    except (IndexError, KeyError, TypeError, ValueError):
        return False
    if (
        due_times[0] != discovery_at
        or any(policy != ADAPTIVE_ODDS_TIMING_POLICY for policy in timing_policies)
        or any(status not in {"succeeded", "failed"} for status in statuses)
        or any(at >= cutoff for at in attempt_times)
        or any(
            attempted < due or attempted - due > ADAPTIVE_ODDS_MAX_LATE
            for due, attempted in zip(due_times, attempt_times)
        )
        or any(later <= earlier for earlier, later in zip(due_times, due_times[1:]))
        or any(later <= earlier for earlier, later in zip(attempt_times, attempt_times[1:]))
    ):
        return False
    consecutive_failures = 0
    for index, prior in enumerate(odds):
        consecutive_failures = consecutive_failures + 1 if prior["status"] == "failed" else 0
        prior_at = due_times[index]
        required_at = next_odds_attempt_at(
            now=prior_at,
            scheduled_jump=scheduled_jump,
            last_attempt_at=prior_at,
            consecutive_failures=consecutive_failures,
        )
        next_at = due_times[index + 1] if index + 1 < len(due_times) else None
        if next_at is not None:
            if required_at is None or required_at >= cutoff or next_at != required_at:
                return False
        elif required_at is not None and required_at < cutoff:
            return False
    return True


def _odds_snapshot_mismatches(
    successes: Sequence[Mapping[str, Any]],
    seal: Mapping[str, Any] | None,
    artifacts: ArtifactStore,
) -> tuple[str, ...]:
    """Bind a seal to the exact final readable successful pre-freeze snapshot."""
    if seal is None:
        return ("seal_missing",)
    mismatches: list[str] = []
    for column in ("raw_manifest_checksum", "normalized_checksum", "odds_checksum"):
        try:
            artifacts.verify(ArtifactChecksum(seal[column]))
        except ArtifactStoreError:
            mismatches.append(f"{column}_missing_or_corrupt")
    frozen_at = datetime.fromisoformat(seal["frozen_at"])
    pre_freeze = [
        row for row in successes if datetime.fromisoformat(row["attempted_at"]) <= frozen_at
    ]
    post_freeze = [
        row for row in successes if datetime.fromisoformat(row["attempted_at"]) > frozen_at
    ]
    if not pre_freeze:
        mismatches.append("final_pre_jump_odds_missing")
    elif pre_freeze[-1]["artifact_checksum"] != seal["odds_checksum"]:
        mismatches.append("final_pre_jump_odds_checksum_mismatch")
    if post_freeze:
        mismatches.append("post_freeze_odds_contamination")
    return tuple(mismatches)


def _safe_operational_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("operational paths must be stable and outside run worktrees")
    current = Path(path.anchor)
    try:
        for part in path.parts[1:]:
            current /= part
            if current.is_symlink():
                # A broken link is no more stable than a looping one. Resolve each
                # extant link before trusting the final lexical path.
                current.resolve(strict=True)
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise ValueError("operational paths must resolve without symlink ambiguity") from error
    if (
        ".state" in resolved.parts
        or "runs" in resolved.parts
        or any(re.fullmatch(r"20\d{6}T\d{6}Z.*", part) for part in resolved.parts)
        or any(part in {"current-run", "latest-run"} for part in resolved.parts)
    ):
        raise ValueError("operational paths must be stable and outside run worktrees")
    return resolved


def verify_release_authority(
    db: sqlite3.Connection, artifacts: ArtifactStore, release_id: str
) -> sqlite3.Row:
    """Reauthenticate the complete immutable release chain at every trust boundary."""
    release = db.execute(
        "SELECT m.*,c.config_json FROM phase7_release_manifests m "
        "JOIN phase7_release_configurations c USING(config_checksum) WHERE m.release_id=?",
        (release_id,),
    ).fetchone()
    if release is None:
        raise OperationalRejected("release and typed configuration are unavailable")
    try:
        config_document = json.loads(release["config_json"])
        config = ReleaseConfiguration(
            schema_version=config_document["schema_version"],
            service_root=config_document["service_root"],
            artifact_root=config_document["artifact_root"],
            operations_database=config_document["operations_database"],
            sources=tuple(config_document["sources"]),
            schedule_policy=config_document["schedule_policy"],
            promotion_policy=config_document["promotion_policy"],
            bundle_versions=tuple(config_document["bundle_versions"]),
            runtime_adapter=config_document["runtime_adapter"],
            runtime_input_checksum=ArtifactChecksum(config_document["runtime_input_checksum"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise OperationalRejected("release configuration is malformed") from error
    if (
        config.document() != config_document
        or artifacts.read(ArtifactChecksum(release["config_checksum"]))
        != _canonical(config_document)
        or config.service_root != release["service_root"]
        or config.promotion_policy != release["policy_version"]
        or config.bundle_versions != tuple(json.loads(release["bundle_versions_json"]))
    ):
        raise OperationalRejected("release configuration artifact disagrees with authority")
    manifest = ReleaseManifest(
        schema_version="phase7-release-v1",
        release_id=release["release_id"],
        code_commit=release["code_commit"],
        config_checksum=ArtifactChecksum(release["config_checksum"]),
        database_schema=release["schema_version"],
        artifact_contract=release["artifact_contract"],
        policy_version=release["policy_version"],
        supported_bundle_versions=tuple(json.loads(release["bundle_versions_json"])),
        service_root=release["service_root"],
    )
    if artifacts.read(ArtifactChecksum(release["manifest_checksum"])) != _canonical(
        manifest.document()
    ):
        raise OperationalRejected("release manifest artifact disagrees with authority")
    schema = db.execute("SELECT max(version) FROM schema_migrations").fetchone()[0]
    if release["schema_version"] != schema:
        raise OperationalRejected("release schema is stale")
    if release["artifact_contract"] != "canonical-artifacts-v1":
        raise OperationalRejected("release artifact contract is unsupported")
    if not set(manifest.supported_bundle_versions).issubset(SUPPORTED_FORECAST_CONTRACTS):
        raise OperationalRejected("release bundle contract registry binding is invalid")
    policy = db.execute(
        "SELECT artifact_checksum FROM phase6_policy_registry WHERE policy_id=?",
        (release["policy_version"],),
    ).fetchone()
    if policy is None:
        raise OperationalRejected("release policy registry binding is unavailable")
    try:
        policy_document = json.loads(artifacts.read(ArtifactChecksum(policy[0])))
    except (ArtifactStoreError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OperationalRejected("release policy artifact is malformed") from error
    if policy_document.get("policy_id") != release["policy_version"]:
        raise OperationalRejected("release policy artifact version disagrees")
    return release


@dataclass(frozen=True, slots=True)
class ReleaseManifest:
    schema_version: str
    release_id: str
    code_commit: str
    config_checksum: ArtifactChecksum
    database_schema: int
    artifact_contract: str
    policy_version: str
    supported_bundle_versions: tuple[str, ...]
    service_root: str

    def __post_init__(self) -> None:
        if self.schema_version != "phase7-release-v1":
            raise ValueError("unsupported release manifest")
        _strict_nonempty(self.release_id, "release identity")
        _strict_nonempty(self.artifact_contract, "artifact contract")
        _strict_nonempty(self.policy_version, "policy identifier")
        _strict_unique_text(self.supported_bundle_versions, "supported bundle versions")
        if not re.fullmatch(r"[0-9a-f]{40}", _strict_nonempty(self.code_commit, "code commit")):
            raise ValueError("code_commit must be an exact Git commit")
        _safe_operational_path(self.service_root)
        if self.database_schema != 29 or not self.supported_bundle_versions:
            raise ValueError("release schema or bundle contract is unsupported")

    def document(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "release_id": self.release_id,
            "code_commit": self.code_commit,
            "config_checksum": str(self.config_checksum),
            "database_schema": self.database_schema,
            "artifact_contract": self.artifact_contract,
            "policy_version": self.policy_version,
            "supported_bundle_versions": list(self.supported_bundle_versions),
            "service_root": self.service_root,
        }


@dataclass(frozen=True, slots=True)
class ReleaseConfiguration:
    schema_version: str
    service_root: str
    artifact_root: str
    operations_database: str
    sources: tuple[str, ...]
    schedule_policy: str
    promotion_policy: str
    bundle_versions: tuple[str, ...]
    runtime_adapter: str
    runtime_input_checksum: ArtifactChecksum

    def __post_init__(self) -> None:
        if self.schema_version != "phase7-config-v1":
            raise ValueError("unsupported release configuration")
        paths = (self.service_root, self.artifact_root, self.operations_database)
        for index, value in enumerate(paths):
            _strict_nonempty(value, ("service root", "artifact root", "operations database")[index])
            _safe_operational_path(value)
        _strict_unique_text(self.sources, "sources")
        _strict_unique_text(self.bundle_versions, "bundle versions")
        _strict_nonempty(self.schedule_policy, "schedule policy")
        _strict_nonempty(self.promotion_policy, "promotion policy")
        _strict_nonempty(self.runtime_adapter, "runtime adapter")
        if not isinstance(self.runtime_input_checksum, ArtifactChecksum):
            raise ValueError("runtime input must be an immutable artifact checksum")
        if not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*",
            self.runtime_adapter,
        ):
            raise ValueError("runtime adapter must be an explicit module:factory binding")

    def document(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "service_root": self.service_root,
            "artifact_root": self.artifact_root,
            "operations_database": self.operations_database,
            "sources": list(self.sources),
            "schedule_policy": self.schedule_policy,
            "promotion_policy": self.promotion_policy,
            "bundle_versions": list(self.bundle_versions),
            "runtime_adapter": self.runtime_adapter,
            "runtime_input_checksum": str(self.runtime_input_checksum),
        }


class OperationalAuthority:
    """The shared scheduler, administration, cutover, and probation command surface."""

    def __init__(
        self,
        store: SQLiteOperationsStore,
        artifacts: ArtifactStore,
        *,
        command_executor: ApplicationCommandExecutor | None = None,
        clock: Callable[[], datetime] | None = None,
    ):
        self.store, self.artifacts = store, artifacts
        self.__command_executor = command_executor
        self.__clock = clock or (lambda: datetime.now().astimezone())

    def __audit_command_attempt(
        self,
        command: ApplicationCommand,
        *,
        token: str,
        generation: int,
        state: str,
        at: datetime,
        error: BaseException,
    ) -> None:
        """Best-effort append-only failure evidence; never masks the command error."""
        try:
            with self.store._connect() as audit:
                audit.execute(
                    "INSERT INTO phase7_application_command_attempts("
                    "command_operation_id,lease_generation,lease_token,state,"
                    "recorded_at,details) VALUES(?,?,?,?,?,?)",
                    (
                        str(command.operation_id),
                        generation,
                        token,
                        state,
                        iso_timestamp(at),
                        f"command_failure:{type(error).__name__}",
                    ),
                )
        except Exception:
            pass

    ALERT_SCOPES = {
        "source_wide_outage": "joins",
        "day_blocker": "training_requests",
        "checksum_failure": "promotion",
        "post_freeze_contamination": "joins",
        "result_before_prediction": "results",
        "champion_failure": "promotion",
    }

    def authorize_day_forecast_cohort(
        self,
        operation_id: OperationId,
        *,
        racing_day_id: str,
        assignment_id: str,
        members: Sequence[DayForecastCohortMember],
        at: datetime,
    ) -> bool:
        """Authorize exact pre-result Racing-Day forecast coverage, never promotion."""
        require_aware(at, "at")
        _strict_nonempty(racing_day_id, "Racing Day identity")
        _strict_nonempty(assignment_id, "assignment identity")
        if not isinstance(operation_id, OperationId) or any(
            type(member) is not DayForecastCohortMember for member in members
        ):
            raise OperationalRejected("day forecast cohort shape is invalid")
        if len(members) < 2 or sum(m.role == "champion" for m in members) != 1:
            raise OperationalRejected("day forecast cohort requires one champion and a challenger")
        identities = [(m.bundle_id, str(m.bundle_checksum)) for m in members]
        if len(identities) != len(set(identities)):
            raise OperationalRejected("day forecast cohort contains duplicate bundles")
        if len({m.service_run_id for m in members}) != len(members):
            raise OperationalRejected("day forecast service runs must be distinct")
        operations = [
            operation for member in members for _, operation in member.forecast_operations
        ]
        if len(operations) != len(set(operations)):
            raise OperationalRejected("day forecast operations must be distinct across the cohort")
        payload = {
            "racing_day_id": racing_day_id,
            "assignment_id": assignment_id,
            "members": [
                {
                    "role": member.role,
                    "bundle_id": member.bundle_id,
                    "bundle_checksum": str(member.bundle_checksum),
                    "service_run_id": str(member.service_run_id),
                    "forecast_operations": [
                        [race_id, str(forecast_operation)]
                        for race_id, forecast_operation in member.forecast_operations
                    ],
                }
                for member in members
            ],
            "at": iso_timestamp(at),
        }
        with self.store._operation(
            operation_id, "phase7_authorize_day_forecast_cohort", payload
        ) as (db, replay):
            if replay:
                row = db.execute(
                    "SELECT assignment_id FROM phase7_day_forecast_cohorts " "WHERE operation_id=?",
                    (str(operation_id),),
                ).fetchone()
                if row is None or row["assignment_id"] != assignment_id:
                    raise OperationalRejected(
                        "day forecast cohort replay lacks exact durable authority"
                    )
                return False
            assignment = db.execute(
                "SELECT assignment_id,bundle_id,bundle_checksum FROM "
                "canonical_day_assignments WHERE racing_day_id=?",
                (racing_day_id,),
            ).fetchone()
            races = {
                row[0]
                for row in db.execute(
                    "SELECT r.race_id FROM races r WHERE r.racing_day_id=? "
                    "AND NOT EXISTS (SELECT 1 FROM collection_quarantines q "
                    "WHERE q.race_id=r.race_id "
                    "AND q.stage IN ('identity','collection','sealing'))",
                    (racing_day_id,),
                )
            }
            if (
                assignment is None
                or assignment["assignment_id"] != assignment_id
                or not races
                or any(
                    {race for race, _ in member.forecast_operations} != races for member in members
                )
            ):
                raise OperationalRejected(
                    "day forecast cohort is unknown, incomplete, stale, or assignment-mismatched"
                )
            champion = next(member for member in members if member.role == "champion")
            if (
                champion.bundle_id != assignment["bundle_id"]
                or str(champion.bundle_checksum) != assignment["bundle_checksum"]
            ):
                raise OperationalRejected(
                    "day forecast champion disagrees with Racing-Day assignment"
                )
            if (
                db.execute(
                    "SELECT 1 FROM result_attempts r JOIN races x USING(race_id) "
                    "WHERE x.racing_day_id=? LIMIT 1",
                    (racing_day_id,),
                ).fetchone()
                or db.execute(
                    "SELECT 1 FROM phase6_forecast_service_artifacts f "
                    "JOIN races x USING(race_id) WHERE x.racing_day_id=? LIMIT 1",
                    (racing_day_id,),
                ).fetchone()
            ):
                raise OperationalRejected(
                    "day forecast cohort is immutable after forecast work or results"
                )
            authenticated: dict[str, list[sqlite3.Row]] = {}
            for member in members:
                bundle = db.execute(
                    "SELECT bundle_checksum,created_at FROM canonical_model_bundles "
                    "WHERE bundle_id=? AND bundle_checksum=?",
                    (member.bundle_id, str(member.bundle_checksum)),
                ).fetchone()
                components = db.execute(
                    "SELECT component_kind,artifact_checksum,byte_size "
                    "FROM canonical_bundle_components WHERE bundle_id=? "
                    "ORDER BY component_kind",
                    (member.bundle_id,),
                ).fetchall()
                if (
                    bundle is None
                    or len(components) != 9
                    or bundle["created_at"] >= iso_timestamp(at)
                ):
                    raise OperationalRejected(
                        "day forecast bundle is unknown, incomplete, or stale"
                    )
                try:
                    self.artifacts.verify(member.bundle_checksum)
                    for component in components:
                        self.artifacts.verify(ArtifactChecksum(component["artifact_checksum"]))
                except ArtifactStoreError as error:
                    raise OperationalRejected(
                        "day forecast bundle artifact is unavailable"
                    ) from error
                authenticated[member.bundle_id] = components
            db.execute(
                "INSERT INTO phase7_day_forecast_cohorts VALUES(?,?,?,?)",
                (racing_day_id, assignment_id, iso_timestamp(at), str(operation_id)),
            )
            for member in members:
                db.execute(
                    "INSERT INTO phase7_day_forecast_cohort_members VALUES(?,?,?,?,?)",
                    (
                        racing_day_id,
                        member.role,
                        member.bundle_id,
                        str(member.bundle_checksum),
                        str(member.service_run_id),
                    ),
                )
                db.executemany(
                    "INSERT INTO phase7_day_forecast_cohort_components VALUES(?,?,?,?,?)",
                    (
                        (
                            racing_day_id,
                            member.bundle_id,
                            component["component_kind"],
                            component["artifact_checksum"],
                            component["byte_size"],
                        )
                        for component in authenticated[member.bundle_id]
                    ),
                )
                db.executemany(
                    "INSERT INTO phase7_day_forecast_commands VALUES(?,?,?,?)",
                    (
                        (racing_day_id, race_id, member.bundle_id, str(forecast_operation))
                        for race_id, forecast_operation in member.forecast_operations
                    ),
                )
        return True

    def reject_result_before_prediction(
        self,
        operation_id: OperationId,
        *,
        racing_day_id: str,
        race_id: str,
        at: datetime,
    ) -> None:
        """Persist a real result-command barrier rejection, then reject the caller."""
        require_aware(at, "at")
        payload = {
            "racing_day_id": racing_day_id,
            "race_id": race_id,
            "reason": "result_before_prediction",
            "at": iso_timestamp(at),
        }
        with self.store._operation(
            operation_id, "phase7_reject_result_before_prediction", payload
        ) as (db, replay):
            alert_id = f"result-rejection:{operation_id}"
            expected = (
                str(operation_id),
                racing_day_id,
                race_id,
                "result_before_prediction",
                iso_timestamp(at),
            )
            if replay:
                row = db.execute(
                    "SELECT * FROM phase7_rejected_result_commands WHERE operation_id=?",
                    (str(operation_id),),
                ).fetchone()
                if row is None or tuple(row) != expected:
                    raise OperationalRejected("result rejection replay lacks its exact receipt")
                alert = db.execute(
                    "SELECT racing_day_id,operation_id FROM phase7_alerts WHERE alert_id=?",
                    (alert_id,),
                ).fetchone()
                pause = db.execute(
                    "SELECT paused,operation_id FROM phase7_pauses WHERE scope='results'"
                ).fetchone()
                if (
                    alert is None
                    or tuple(alert) != (racing_day_id, str(operation_id))
                    or pause is None
                    or tuple(pause) != (1, str(operation_id))
                ):
                    raise OperationalRejected("result rejection replay lacks alert/pause authority")
            else:
                race = db.execute(
                    "SELECT racing_day_id FROM races WHERE race_id=?", (race_id,)
                ).fetchone()
                if race is None or race[0] != racing_day_id:
                    raise OperationalRejected("result command race/day binding is invalid")
                if db.execute(
                    "SELECT 1 FROM deferred_predictions WHERE race_id=?", (race_id,)
                ).fetchone():
                    raise OperationalRejected("prediction exists; this barrier rejection is false")
                db.execute(
                    "INSERT INTO phase7_alerts VALUES(?,?,?,?,?,NULL,?)",
                    (
                        alert_id,
                        "result_before_prediction",
                        racing_day_id,
                        f"result command rejected for race {race_id}",
                        iso_timestamp(at),
                        str(operation_id),
                    ),
                )
                db.execute(
                    "INSERT INTO phase7_pauses VALUES('results',1,?,?,?) "
                    "ON CONFLICT(scope) DO UPDATE SET paused=1,reason=excluded.reason,"
                    "changed_at=excluded.changed_at,operation_id=excluded.operation_id",
                    (
                        "alert:result_before_prediction",
                        iso_timestamp(at),
                        str(operation_id),
                    ),
                )
                state = db.execute(
                    "SELECT generation FROM phase7_probation_control WHERE singleton=1"
                ).fetchone()
                generation = 1 if state is None else state[0]
                db.execute(
                    "INSERT INTO phase7_probation_control VALUES"
                    "(1,'paused','result_before_prediction',?,?,?) "
                    "ON CONFLICT(singleton) DO UPDATE SET state='paused',"
                    "reason=excluded.reason,changed_at=excluded.changed_at,"
                    "operation_id=excluded.operation_id",
                    (generation, iso_timestamp(at), str(operation_id)),
                )
                db.execute(
                    "INSERT INTO phase7_rejected_result_commands VALUES(?,?,?,?,?)", expected
                )
        raise BarrierNotSatisfied("result collection requires a committed prediction")

    def acquire_lease(
        self, operation_id: OperationId, *, owner: str, token: str, now: datetime, ttl: timedelta
    ) -> int:
        require_aware(now, "now")
        if not owner.strip() or not token.strip() or ttl <= timedelta(0):
            raise ValueError("lease owner, token and positive ttl are required")
        payload = {
            "owner": owner,
            "token": token,
            "now": iso_timestamp(now),
            "ttl": ttl.total_seconds(),
        }
        with self.store._operation(operation_id, "phase7_acquire_scheduler_lease", payload) as (
            db,
            replay,
        ):
            if replay:
                row = db.execute(
                    "SELECT generation FROM phase7_scheduler_history WHERE operation_id=?",
                    (str(operation_id),),
                ).fetchone()
                if row is None:
                    raise OperationalRejected("lease replay has no durable outcome")
                return row[0]
            current = db.execute(
                "SELECT * FROM phase7_scheduler_lease WHERE singleton=1"
            ).fetchone()
            if current is not None and current["expires_at"] > iso_timestamp(now):
                raise OperationalRejected("scheduler lease is owned by another live generation")
            generation = 1 if current is None else current["generation"] + 1
            expires = iso_timestamp(now + ttl)
            db.execute("DELETE FROM phase7_scheduler_lease WHERE singleton=1")
            values = (owner, token, iso_timestamp(now), expires, generation, str(operation_id))
            db.execute("INSERT INTO phase7_scheduler_lease VALUES(1,?,?,?,?,?,?)", values)
            db.execute(
                "INSERT INTO phase7_scheduler_history VALUES(?,?,?,?,?,?)",
                (generation, owner, token, iso_timestamp(now), expires, str(operation_id)),
            )
            return generation

    def renew_lease(
        self,
        operation_id: OperationId,
        *,
        token: str,
        generation: int,
        now: datetime,
        ttl: timedelta,
    ) -> datetime:
        """Extend only the currently fenced generation and retain append-only proof."""
        require_aware(now, "now")
        if not token.strip() or generation < 1 or ttl <= timedelta(0):
            raise ValueError("lease token, generation and positive ttl are required")
        expires_at = now + ttl
        payload = {
            "token": token,
            "generation": generation,
            "now": iso_timestamp(now),
            "expires_at": iso_timestamp(expires_at),
        }
        with self.store._operation(
            operation_id,
            "phase7_renew_scheduler_lease",
            payload,
        ) as (db, replay):
            if replay:
                row = db.execute(
                    "SELECT expires_at FROM phase7_scheduler_renewals WHERE operation_id=?",
                    (str(operation_id),),
                ).fetchone()
                if row is None:
                    raise OperationalRejected("lease renewal replay lacks durable authority")
                return datetime.fromisoformat(row["expires_at"])
            lease = db.execute(
                "SELECT lease_token,generation,acquired_at,expires_at FROM phase7_scheduler_lease "
                "WHERE singleton=1"
            ).fetchone()
            prior = db.execute(
                "SELECT renewed_at,expires_at FROM phase7_scheduler_renewals "
                "WHERE lease_generation=? ORDER BY renewed_at DESC LIMIT 1",
                (generation,),
            ).fetchone()
            if (
                lease is None
                or lease["lease_token"] != token
                or lease["generation"] != generation
                or lease["expires_at"] <= iso_timestamp(now)
                or iso_timestamp(now) < lease["acquired_at"]
                or expires_at <= datetime.fromisoformat(lease["expires_at"])
                or (
                    prior is not None
                    and (
                        iso_timestamp(now) <= prior["renewed_at"]
                        or iso_timestamp(expires_at) <= prior["expires_at"]
                    )
                )
            ):
                raise OperationalRejected("scheduler lease renewal is stale or fenced")
            db.execute(
                "UPDATE phase7_scheduler_lease SET expires_at=? WHERE singleton=1",
                (iso_timestamp(expires_at),),
            )
            db.execute(
                "INSERT INTO phase7_scheduler_renewals VALUES(?,?,?,?,?)",
                (
                    str(operation_id),
                    generation,
                    token,
                    iso_timestamp(now),
                    iso_timestamp(expires_at),
                ),
            )
        return expires_at

    def __commit_application_command(
        self,
        operation_id: OperationId,
        *,
        racing_day_id: str,
        phase: str,
        result: Mapping[str, Any],
        at: datetime,
        token: str,
        generation: int,
        authority_now: datetime,
        command: ApplicationCommand | None = None,
    ) -> Any:
        """Atomically commit phase mutations and their immutable result receipt."""
        require_aware(at, "at")
        if phase not in RaceCollectionService.ORDER:
            raise ValueError("unknown Race Collection Service phase")
        result_json = _canonical(result).decode()
        if command is None:
            raise ValueError("typed application command is required")
        command_payload_sha256 = command.payload_sha256()
        payload = {
            "racing_day_id": racing_day_id,
            "phase": phase,
            "result_checksum": str(_checksum(result)),
            "command_payload_sha256": command_payload_sha256,
            "at": iso_timestamp(at),
        }
        with self.store._operation(operation_id, f"phase7_command_{phase}", payload) as (
            db,
            replay,
        ):
            if replay:
                row = db.execute(
                    "SELECT result_json,command_payload_sha256 "
                    "FROM phase7_application_command_receipts "
                    "WHERE command_operation_id=?",
                    (str(operation_id),),
                ).fetchone()
                if row is None or row["command_payload_sha256"] != command_payload_sha256:
                    raise OperationalRejected("command replay lacks its durable receipt")
                return json.loads(row[0])
            lease = db.execute(
                "SELECT 1 FROM phase7_scheduler_lease WHERE singleton=1 "
                "AND lease_token=? AND generation=? AND acquired_at<=? AND expires_at>?",
                (
                    token,
                    generation,
                    iso_timestamp(authority_now),
                    iso_timestamp(authority_now),
                ),
            ).fetchone()
            claim = db.execute(
                "SELECT 1 FROM phase7_application_command_claims "
                "WHERE command_operation_id=? AND racing_day_id=? AND phase_name=? "
                "AND command_payload_sha256=?",
                (str(operation_id), racing_day_id, phase, command_payload_sha256),
            ).fetchone()
            attempt = db.execute(
                "SELECT lease_generation,lease_token,state "
                "FROM phase7_application_command_attempts "
                "WHERE command_operation_id=? AND state IN ('claimed','recovering') "
                "ORDER BY attempt_id DESC LIMIT 1",
                (str(operation_id),),
            ).fetchone()
            if (
                lease is None
                or claim is None
                or attempt is None
                or attempt["lease_generation"] != generation
                or attempt["lease_token"] != token
                or attempt["state"] not in ("claimed", "recovering")
            ):
                raise OperationalRejected(
                    "lease or authoritative command attempt changed before receipt commit"
                )
            if command is not None:
                committed_result = self._phase_postcondition(db, command)
                if _canonical(committed_result) != _canonical(result):
                    raise OperationalRejected("command postcondition changed before receipt commit")
            db.execute(
                "INSERT INTO phase7_application_command_receipts("
                "command_operation_id,racing_day_id,phase_name,result_json,"
                "result_checksum,committed_at,command_payload_sha256) "
                "VALUES(?,?,?,?,?,?,?)",
                (
                    str(operation_id),
                    racing_day_id,
                    phase,
                    result_json,
                    str(_checksum(result)),
                    iso_timestamp(at),
                    command_payload_sha256,
                ),
            )
            db.execute(
                "INSERT INTO phase7_application_command_attempts("
                "command_operation_id,lease_generation,lease_token,state,recorded_at,details"
                ") VALUES(?,?,?,?,?,?)",
                (
                    str(operation_id),
                    generation,
                    token,
                    "completed",
                    iso_timestamp(authority_now),
                    "postcondition and receipt committed atomically",
                ),
            )
        return result

    def _expected_cohort(self, db: sqlite3.Connection, racing_day_id: str) -> list[sqlite3.Row]:
        rows = db.execute(
            "SELECT e.*,r.discovered_at,r.state FROM expected_races e "
            "JOIN races r USING(race_id) WHERE r.racing_day_id=? "
            "ORDER BY e.scheduled_jump,e.race_id",
            (racing_day_id,),
        ).fetchall()
        if not rows:
            raise BarrierNotSatisfied("expected programme is empty")
        return rows

    @staticmethod
    def _terminal_collection_quarantine(db: sqlite3.Connection, race_id: str) -> sqlite3.Row | None:
        return db.execute(
            "SELECT stage,code,details,operation_id FROM collection_quarantines "
            "WHERE race_id=? AND stage IN ('identity','collection','sealing') "
            "ORDER BY quarantine_id DESC LIMIT 1",
            (race_id,),
        ).fetchone()

    def bind_training_request(
        self,
        operation_id: OperationId,
        *,
        racing_day_id: str,
        request_id: str,
        request_operation_id: OperationId,
        at: datetime,
    ) -> bool:
        """Bind an emitted forecast-service request to one reconciled Racing Day."""
        require_aware(at, "at")
        payload = {
            "racing_day_id": racing_day_id,
            "request_id": request_id,
            "request_operation_id": str(request_operation_id),
            "at": iso_timestamp(at),
        }
        with self.store._operation(operation_id, "phase7_bind_training_request", payload) as (
            db,
            replay,
        ):
            if replay:
                return False
            reconciliation = db.execute(
                "SELECT e.complete,r.mismatch_count,r.reconciled_at "
                "FROM phase7_day_evidence e "
                "JOIN phase7_reconciliation r USING(racing_day_id) "
                "WHERE e.racing_day_id=?",
                (racing_day_id,),
            ).fetchone()
            request = db.execute(
                "SELECT r.operation_id,r.requested_at,s.operation_id service_operation_id "
                "FROM phase6_training_requests r JOIN phase6_service_training_requests s "
                "USING(training_request_id) WHERE r.training_request_id=?",
                (request_id,),
            ).fetchone()
            intent = db.execute(
                "SELECT operation_id,authorized_at FROM phase7_training_request_intents "
                "WHERE racing_day_id=? AND training_request_id=? "
                "AND request_operation_id=?",
                (racing_day_id, request_id, str(request_operation_id)),
            ).fetchone()
            if (
                reconciliation is None
                or reconciliation["complete"] != 1
                or reconciliation["mismatch_count"] != 0
                or request is None
                or intent is None
                or request["operation_id"] != str(request_operation_id)
                or request["service_operation_id"] != str(request_operation_id)
                or iso_timestamp(at) < reconciliation["reconciled_at"]
                or iso_timestamp(at) < request["requested_at"]
                or iso_timestamp(at) < intent["authorized_at"]
            ):
                raise BarrierNotSatisfied(
                    "training request lacks exact reconciled-day service authority"
                )
            db.execute(
                "INSERT INTO phase7_day_training_requests VALUES(?,?,?,?,?)",
                (
                    racing_day_id,
                    request_id,
                    str(request_operation_id),
                    iso_timestamp(at),
                    str(operation_id),
                ),
            )
        return True

    def authorize_training_request(
        self,
        operation_id: OperationId,
        *,
        racing_day_id: str,
        request_id: str,
        request_operation_id: OperationId,
        at: datetime,
    ) -> bool:
        """Authorize one emit-only request identity for one reconciled Racing Day."""
        require_aware(at, "at")
        payload = {
            "racing_day_id": racing_day_id,
            "request_id": request_id,
            "request_operation_id": str(request_operation_id),
            "at": iso_timestamp(at),
        }
        with self.store._operation(operation_id, "phase7_authorize_training_request", payload) as (
            db,
            replay,
        ):
            if replay:
                return False
            reconciliation = db.execute(
                "SELECT e.complete,r.mismatch_count,r.reconciled_at "
                "FROM phase7_day_evidence e JOIN phase7_reconciliation r "
                "USING(racing_day_id) WHERE e.racing_day_id=?",
                (racing_day_id,),
            ).fetchone()
            if (
                reconciliation is None
                or reconciliation["complete"] != 1
                or reconciliation["mismatch_count"] != 0
                or iso_timestamp(at) < reconciliation["reconciled_at"]
            ):
                raise BarrierNotSatisfied(
                    "training request authorization requires complete reconciliation"
                )
            db.execute(
                "INSERT INTO phase7_training_request_intents VALUES(?,?,?,?,?)",
                (
                    racing_day_id,
                    request_id,
                    str(request_operation_id),
                    iso_timestamp(at),
                    str(operation_id),
                ),
            )
        return True

    def _phase_postcondition(
        self, db: sqlite3.Connection, command: ApplicationCommand
    ) -> Mapping[str, Any]:
        """Return the exact durable authority set, or reject an incomplete day barrier."""
        cohort = self._expected_cohort(db, command.racing_day_id)
        race_ids = [row["race_id"] for row in cohort]
        payload = command.payload

        def one(sql: str, race_id: str) -> sqlite3.Row | None:
            return db.execute(sql, (race_id,)).fetchone()

        if isinstance(payload, DiscoverProgramme):
            if (
                not payload.source.strip()
                or len({row["programme_checksum"] for row in cohort}) != 1
                or any(
                    row["source"] != payload.source
                    or row["programme_checksum"] != str(payload.programme_checksum)
                    for row in cohort
                )
            ):
                raise BarrierNotSatisfied("expected programme source/checksum is not exact")
            self.artifacts.verify(payload.programme_checksum)
            return {
                "source": payload.source,
                "programme_checksum": str(payload.programme_checksum),
                "races": [
                    {
                        "race_id": row["race_id"],
                        "source_race_id": row["source_race_id"],
                        "scheduled_jump": row["scheduled_jump"],
                        "operation_id": row["operation_id"],
                    }
                    for row in cohort
                ],
            }

        if isinstance(payload, CollectCardsAndForm):
            authority = []
            for race_id in race_ids:
                quarantine = self._terminal_collection_quarantine(db, race_id)
                fields = db.execute(
                    "SELECT field_name,value_json,distinct_top_values,critical "
                    "FROM phase6_resolved_field_evidence WHERE race_id=? "
                    "ORDER BY field_name",
                    (race_id,),
                ).fetchall()
                names = {row["field_name"] for row in fields}
                if quarantine is None:
                    if not {"runner_set", "box"}.issubset(names) or any(
                        row["distinct_top_values"] != 1
                        for row in fields
                        if row["field_name"] in {"runner_set", "box"}
                    ):
                        raise BarrierNotSatisfied(f"{race_id} lacks complete runner/box authority")
                    resolved = {row["field_name"]: row for row in fields}
                    try:
                        runners = json.loads(resolved["runner_set"]["value_json"])
                        boxes = json.loads(resolved["box"]["value_json"])
                    except (KeyError, TypeError, json.JSONDecodeError) as error:
                        raise BarrierNotSatisfied(
                            f"{race_id} runner/box authority is malformed"
                        ) from error
                    if (
                        not isinstance(runners, list)
                        or not runners
                        or len(runners) != len(set(runners))
                        or not isinstance(boxes, dict)
                        or set(boxes) != set(runners)
                        or len(boxes) != len(runners)
                        or any(type(box) is not int or box <= 0 for box in boxes.values())
                        or len(set(boxes.values())) != len(runners)
                    ):
                        raise BarrierNotSatisfied(
                            f"{race_id} runner cardinality/box mapping is not exact"
                        )
                source_rows = db.execute(
                    "SELECT field_name,authority,value_json,artifact_checksum,operation_id "
                    "FROM field_evidence WHERE race_id=? AND field_name IN ('runner_set','box') "
                    "ORDER BY field_name,authority,operation_id",
                    (race_id,),
                ).fetchall()
                for field in source_rows:
                    self.artifacts.verify(ArtifactChecksum(field["artifact_checksum"]))
                authority.append(
                    {
                        "race_id": race_id,
                        "fields": [dict(row) for row in fields],
                        "source_evidence": [dict(row) for row in source_rows],
                        "quarantine": None if quarantine is None else dict(quarantine),
                    }
                )
            return {"races": authority}

        if isinstance(payload, CollectAdaptiveOdds):
            authority = []
            for expected in cohort:
                race_id = expected["race_id"]
                quarantine = self._terminal_collection_quarantine(db, race_id)
                if quarantine is not None:
                    authority.append({"race_id": race_id, "excluded": dict(quarantine)})
                    continue
                attempts = db.execute(
                    "SELECT scheduled_due_at,attempted_at,timing_policy,status,"
                    "artifact_checksum,runner_mapping_checksum,operation_id "
                    "FROM odds_attempts WHERE race_id=? "
                    "ORDER BY scheduled_due_at,attempted_at",
                    (race_id,),
                ).fetchall()
                jump = datetime.fromisoformat(expected["scheduled_jump"])
                if not _adaptive_odds_history_complete(
                    attempts,
                    discovery_at=datetime.fromisoformat(expected["discovered_at"]),
                    scheduled_jump=jump,
                    cutoff=jump,
                ):
                    raise BarrierNotSatisfied(f"{race_id} adaptive odds cadence is incomplete")
                successes = [row for row in attempts if row["status"] == "succeeded"]
                if not successes:
                    raise BarrierNotSatisfied(f"{race_id} lacks a successful pre-jump snapshot")
                for row in successes:
                    self.artifacts.verify(ArtifactChecksum(row["artifact_checksum"]))
                    self.artifacts.verify(ArtifactChecksum(row["runner_mapping_checksum"]))
                authority.append({"race_id": race_id, "attempts": [dict(row) for row in attempts]})
            return {"races": authority}

        if isinstance(payload, CloseAndSeal):
            day = db.execute(
                "SELECT closed_at FROM racing_days WHERE racing_day_id=?",
                (command.racing_day_id,),
            ).fetchone()
            if day is None or day["closed_at"] is None:
                raise BarrierNotSatisfied("Racing Day is not closed")
            authority = []
            for race_id in race_ids:
                seal = one("SELECT * FROM sealed_evidence WHERE race_id=?", race_id)
                quarantine = self._terminal_collection_quarantine(db, race_id)
                if seal is None and quarantine is None:
                    raise BarrierNotSatisfied(f"{race_id} is neither sealed nor quarantined")
                if seal is not None:
                    attempts = db.execute(
                        "SELECT * FROM odds_attempts WHERE race_id=? AND status='succeeded' "
                        "ORDER BY scheduled_due_at,attempted_at",
                        (race_id,),
                    ).fetchall()
                    mismatch = _odds_snapshot_mismatches(attempts, seal, self.artifacts)
                    if mismatch:
                        raise BarrierNotSatisfied(f"{race_id} invalid seal: {','.join(mismatch)}")
                authority.append(
                    {
                        "race_id": race_id,
                        "seal": (
                            None
                            if seal is None
                            else {
                                "seal_id": seal["seal_id"],
                                "odds_checksum": seal["odds_checksum"],
                                "operation_id": seal["operation_id"],
                            }
                        ),
                        "quarantine": None if quarantine is None else dict(quarantine),
                    }
                )
            return {"closed_at": day["closed_at"], "races": authority}

        if isinstance(payload, CommitDeferredPrediction):
            authority = []
            for race_id in race_ids:
                collection_quarantine = self._terminal_collection_quarantine(db, race_id)
                if collection_quarantine is not None:
                    authority.append({"race_id": race_id, "excluded": dict(collection_quarantine)})
                    continue
                prediction = one(
                    "SELECT prediction_id,artifact_checksum,evidence_checksum,operation_id "
                    "FROM deferred_predictions WHERE race_id=?",
                    race_id,
                )
                quarantine = one(
                    "SELECT prediction_id,evidence_checksum,operation_id "
                    "FROM prediction_quarantines WHERE race_id=?",
                    race_id,
                )
                if (prediction is None) == (quarantine is None):
                    raise BarrierNotSatisfied(
                        f"{race_id} needs exactly one prediction or prediction quarantine"
                    )
                if prediction is not None:
                    self.artifacts.verify(ArtifactChecksum(prediction["artifact_checksum"]))
                authority.append(
                    {
                        "race_id": race_id,
                        "prediction": None if prediction is None else dict(prediction),
                        "quarantine": None if quarantine is None else dict(quarantine),
                    }
                )
            return {
                "races": authority,
                "forecast_cohort": self._phase5_forecast_cohort_authority(
                    db, command.racing_day_id, race_ids
                ),
            }

        if isinstance(payload, CollectResults):
            authority = []
            for race_id in race_ids:
                collection_quarantine = self._terminal_collection_quarantine(db, race_id)
                if collection_quarantine is not None:
                    authority.append({"race_id": race_id, "excluded": dict(collection_quarantine)})
                    continue
                prediction = one(
                    "SELECT prediction_id FROM deferred_predictions WHERE race_id=?", race_id
                )
                prediction_quarantine = one(
                    "SELECT prediction_id,operation_id FROM prediction_quarantines WHERE race_id=?",
                    race_id,
                )
                if prediction_quarantine is not None:
                    authority.append({"race_id": race_id, "excluded": dict(prediction_quarantine)})
                    continue
                if prediction is None:
                    raise BarrierNotSatisfied(f"{race_id} result precedes prediction")
                terminal = one(
                    "SELECT attempt_id,status,artifact_checksum,operation_id FROM result_attempts "
                    "WHERE race_id=? AND status IN ('collected','quarantined') "
                    "ORDER BY attempt_number DESC LIMIT 1",
                    race_id,
                )
                if terminal is None:
                    raise BarrierNotSatisfied(f"{race_id} lacks a terminal result")
                if terminal["artifact_checksum"] is not None:
                    self.artifacts.verify(ArtifactChecksum(terminal["artifact_checksum"]))
                authority.append({"race_id": race_id, "result": dict(terminal)})
            return {"races": authority}

        if isinstance(payload, JoinTrainingExamples):
            authority = []
            for race_id in race_ids:
                collection_quarantine = self._terminal_collection_quarantine(db, race_id)
                if collection_quarantine is not None:
                    authority.append({"race_id": race_id, "excluded": dict(collection_quarantine)})
                    continue
                terminal = one(
                    "SELECT attempt_id,status,operation_id FROM result_attempts WHERE race_id=? "
                    "AND status IN ('collected','quarantined') ORDER BY attempt_number DESC LIMIT 1",
                    race_id,
                )
                prediction_quarantine = one(
                    "SELECT operation_id FROM prediction_quarantines WHERE race_id=?", race_id
                )
                example = one(
                    "SELECT training_example_id,prediction_id,result_attempt_id,"
                    "artifact_checksum,eligibility,reason,operation_id "
                    "FROM training_examples WHERE race_id=?",
                    race_id,
                )
                if terminal is not None and terminal["status"] == "collected" and example is None:
                    raise BarrierNotSatisfied(f"{race_id} eligible result is not joined")
                if terminal is None and prediction_quarantine is None:
                    raise BarrierNotSatisfied(f"{race_id} has no authoritative join exclusion")
                if example is not None:
                    self.artifacts.verify(ArtifactChecksum(example["artifact_checksum"]))
                    prediction = one(
                        "SELECT prediction_id FROM deferred_predictions WHERE race_id=?", race_id
                    )
                    if (
                        prediction is None
                        or example["prediction_id"] != prediction["prediction_id"]
                        or terminal is None
                        or example["result_attempt_id"] != terminal["attempt_id"]
                    ):
                        raise BarrierNotSatisfied(f"{race_id} training join identity is not exact")
                    canonical = db.execute(
                        "SELECT artifact_checksum FROM canonical_training_examples "
                        "WHERE phase3_training_example_id=?",
                        (example["training_example_id"],),
                    ).fetchone()
                    if example["eligibility"] == "eligible":
                        if canonical is None:
                            raise BarrierNotSatisfied(
                                f"{race_id} eligible join lacks canonical training authority"
                            )
                        self.artifacts.verify(ArtifactChecksum(canonical["artifact_checksum"]))
                    elif not example["reason"]:
                        raise BarrierNotSatisfied(
                            f"{race_id} ineligible join lacks an explicit exclusion"
                        )
                authority.append(
                    {
                        "race_id": race_id,
                        "example": None if example is None else dict(example),
                        "excluded": prediction_quarantine is not None
                        or (terminal is not None and terminal["status"] == "quarantined"),
                    }
                )
            return {"races": authority}

        if isinstance(payload, ReconcileRacingDay):
            row = db.execute(
                "SELECT e.complete,e.reconciliation_checksum,e.operation_id,"
                "r.report_checksum,r.mismatch_count,r.report_json "
                "FROM phase7_day_evidence e JOIN phase7_reconciliation r USING(racing_day_id) "
                "WHERE e.racing_day_id=?",
                (command.racing_day_id,),
            ).fetchone()
            if (
                row is None
                or row["complete"] != 1
                or row["mismatch_count"] != 0
                or row["report_checksum"] != row["reconciliation_checksum"]
            ):
                raise BarrierNotSatisfied("reconciliation is incomplete or mismatched")
            report_content = self.artifacts.read(ArtifactChecksum(row["report_checksum"]))
            try:
                report = json.loads(row["report_json"])
            except json.JSONDecodeError as error:
                raise BarrierNotSatisfied("reconciliation report JSON is malformed") from error
            if report_content != _canonical(report):
                raise BarrierNotSatisfied(
                    "reconciliation report artifact disagrees with durable JSON"
                )
            return dict(row)

        if isinstance(payload, RequestTraining):
            row = db.execute(
                "SELECT b.*,s.service_run_id,i.operation_id authorization_operation_id "
                "FROM phase7_day_training_requests b "
                "JOIN phase6_service_training_requests s USING(training_request_id) "
                "JOIN phase7_training_request_intents i USING("
                "racing_day_id,training_request_id,request_operation_id) "
                "WHERE b.racing_day_id=? AND b.training_request_id=? "
                "AND b.request_operation_id=? AND i.operation_id=? AND b.operation_id=?",
                (
                    command.racing_day_id,
                    payload.request_id,
                    str(payload.request_operation_id),
                    str(payload.authorization_operation_id),
                    str(payload.binding_operation_id),
                ),
            ).fetchone()
            if row is None:
                raise BarrierNotSatisfied("exact day training request binding is unavailable")
            return {"request": dict(row)}
        raise OperationalRejected("unsupported application command")

    def _phase5_forecast_cohort_authority(
        self,
        db: sqlite3.Connection,
        racing_day_id: str,
        expected_race_ids: Sequence[str],
    ) -> Mapping[str, Any]:
        """Prove the exact authenticated day cohort and every service forecast."""
        supplied = set(expected_race_ids)
        expected = {
            race_id
            for race_id in supplied
            if self._terminal_collection_quarantine(db, race_id) is None
        }
        excluded = sorted(supplied - expected)
        if not expected:
            stray = (
                db.execute(
                    "SELECT 1 FROM phase7_day_forecast_cohorts " "WHERE racing_day_id=?",
                    (racing_day_id,),
                ).fetchone()
                or db.execute(
                    "SELECT 1 FROM phase6_forecast_service_artifacts f "
                    "JOIN races r USING(race_id) WHERE r.racing_day_id=?",
                    (racing_day_id,),
                ).fetchone()
            )
            if stray is not None:
                raise BarrierNotSatisfied(
                    "fully excluded Racing Day has unauthorized forecast authority"
                )
            return {
                "assignment_id": None,
                "authorization_operation_id": None,
                "members": [],
                "forecasts": [],
                "excluded_race_ids": excluded,
            }
        assignment = db.execute(
            "SELECT assignment_id,bundle_id,bundle_checksum FROM "
            "canonical_day_assignments WHERE racing_day_id=?",
            (racing_day_id,),
        ).fetchone()
        cohort = db.execute(
            "SELECT assignment_id,authorized_at,operation_id FROM "
            "phase7_day_forecast_cohorts WHERE racing_day_id=?",
            (racing_day_id,),
        ).fetchone()
        members = db.execute(
            "SELECT role,bundle_id,bundle_checksum,service_run_id FROM "
            "phase7_day_forecast_cohort_members WHERE racing_day_id=? "
            "ORDER BY role,bundle_id",
            (racing_day_id,),
        ).fetchall()
        if (
            assignment is None
            or cohort is None
            or cohort["assignment_id"] != assignment["assignment_id"]
            or len([member for member in members if member["role"] == "champion"]) != 1
            or not any(member["role"] == "challenger" for member in members)
            or (
                assignment["bundle_id"],
                assignment["bundle_checksum"],
            )
            not in {
                (member["bundle_id"], member["bundle_checksum"])
                for member in members
                if member["role"] == "champion"
            }
        ):
            raise BarrierNotSatisfied("day forecast cohort lacks its exact champion assignment")
        if len({member["service_run_id"] for member in members}) != len(members):
            raise BarrierNotSatisfied("day forecast cohort service runs are ambiguous")
        member_by_bundle = {member["bundle_id"]: member for member in members}
        if len(member_by_bundle) != len(members):
            raise BarrierNotSatisfied("day forecast cohort bundles are ambiguous")
        commands = db.execute(
            "SELECT race_id,bundle_id,operation_id FROM phase7_day_forecast_commands "
            "WHERE racing_day_id=?",
            (racing_day_id,),
        ).fetchall()
        expected_commands = {
            (race_id, member["bundle_id"]) for race_id in expected for member in members
        }
        if {(row["race_id"], row["bundle_id"]) for row in commands} != expected_commands or len(
            {row["operation_id"] for row in commands}
        ) != len(commands):
            raise BarrierNotSatisfied("day forecast cohort lacks exact race/operation coverage")
        command_by_identity = {
            (row["race_id"], row["bundle_id"]): row["operation_id"] for row in commands
        }
        for member in members:
            registered = db.execute(
                "SELECT created_at,forecast_contract_version FROM "
                "canonical_model_bundles WHERE bundle_id=? AND bundle_checksum=?",
                (member["bundle_id"], member["bundle_checksum"]),
            ).fetchone()
            snapshot = db.execute(
                "SELECT component_kind,artifact_checksum,byte_size FROM "
                "phase7_day_forecast_cohort_components "
                "WHERE racing_day_id=? AND bundle_id=? ORDER BY component_kind",
                (racing_day_id, member["bundle_id"]),
            ).fetchall()
            current = db.execute(
                "SELECT component_kind,artifact_checksum,byte_size FROM "
                "canonical_bundle_components WHERE bundle_id=? ORDER BY component_kind",
                (member["bundle_id"],),
            ).fetchall()
            if (
                registered is None
                or registered["created_at"] >= cohort["authorized_at"]
                or [tuple(row) for row in snapshot] != [tuple(row) for row in current]
                or len(snapshot) != 9
            ):
                raise BarrierNotSatisfied(
                    f"{member['bundle_id']} component snapshot is incomplete or altered"
                )
            try:
                self.artifacts.verify(ArtifactChecksum(member["bundle_checksum"]))
                for component in snapshot:
                    self.artifacts.verify(ArtifactChecksum(component["artifact_checksum"]))
            except ArtifactStoreError as error:
                raise BarrierNotSatisfied(
                    f"{member['bundle_id']} bundle artifacts do not verify"
                ) from error
        forecast_rows = db.execute(
            "SELECT f.*,c.racing_day_id authorized_day FROM "
            "phase6_forecast_service_artifacts f LEFT JOIN "
            "phase7_day_forecast_commands c ON c.race_id=f.race_id "
            "AND c.bundle_id=f.bundle_id AND c.operation_id=f.operation_id "
            "WHERE f.race_id IN (SELECT race_id FROM races WHERE racing_day_id=?)",
            (racing_day_id,),
        ).fetchall()
        forecast_by_identity = {(row["race_id"], row["bundle_id"]): row for row in forecast_rows}
        forecast_expected = {
            (race_id, member["bundle_id"])
            for race_id in expected
            if db.execute(
                "SELECT 1 FROM deferred_predictions WHERE race_id=?", (race_id,)
            ).fetchone()
            is not None
            for member in members
        }
        if (
            set(forecast_by_identity) != forecast_expected
            or len(forecast_by_identity) != len(forecast_rows)
            or any(row["authorized_day"] != racing_day_id for row in forecast_rows)
        ):
            raise BarrierNotSatisfied(
                "service forecasts do not exactly cover the authorized cohort"
            )
        authenticated_forecasts = []
        for identity in sorted(forecast_expected):
            race_id, bundle_id = identity
            forecast = forecast_by_identity[identity]
            member = member_by_bundle[bundle_id]
            prediction = db.execute(
                "SELECT prediction_id,evidence_checksum,computed_at FROM "
                "deferred_predictions WHERE race_id=?",
                (race_id,),
            ).fetchone()
            computation = db.execute(
                "SELECT s.* FROM phase6_forecast_computation_bindings b "
                "JOIN phase6_service_computations s USING(computation_id) "
                "WHERE b.forecast_checksum=?",
                (forecast["forecast_checksum"],),
            ).fetchone()
            run = db.execute(
                "SELECT started_at FROM phase6_runs WHERE run_id=? "
                "AND run_kind='forecast_service'",
                (member["service_run_id"],),
            ).fetchone()
            if (
                prediction is None
                or forecast["operation_id"] != command_by_identity[identity]
                or forecast["service_run_id"] != member["service_run_id"]
                or forecast["bundle_checksum"] != member["bundle_checksum"]
                or forecast["deferred_prediction_id"] != prediction["prediction_id"]
                or forecast["evidence_checksum"] != prediction["evidence_checksum"]
                or forecast["generated_at"] != prediction["computed_at"]
                or computation is None
                or any(
                    computation[field] != forecast[field]
                    for field in (
                        "race_id",
                        "bundle_id",
                        "bundle_checksum",
                        "evidence_checksum",
                        "service_run_id",
                    )
                )
                or computation["computed_at"] != forecast["generated_at"]
                or computation["phase3_prediction_id"] != forecast["deferred_prediction_id"]
                or computation["operation_id"] != forecast["operation_id"]
                or run is None
                or run["started_at"] > forecast["generated_at"]
                or db.execute(
                    "SELECT 1 FROM result_attempts WHERE race_id=? AND attempted_at<=?",
                    (race_id, forecast["generated_at"]),
                ).fetchone()
                is not None
            ):
                raise BarrierNotSatisfied(f"{race_id}/{bundle_id} forecast authority disagrees")
            try:
                content = self.artifacts.read(ArtifactChecksum(forecast["artifact_checksum"]))
                document = json.loads(content)
            except (
                ArtifactStoreError,
                json.JSONDecodeError,
                UnicodeDecodeError,
            ) as error:
                raise BarrierNotSatisfied(
                    f"{race_id}/{bundle_id} forecast artifact does not verify"
                ) from error
            if (
                "sha256:" + hashlib.sha256(content).hexdigest() != forecast["forecast_checksum"]
                or document.get("race_id") != race_id
                or document.get("bundle_id") != bundle_id
                or document.get("bundle_checksum") != forecast["bundle_checksum"]
                or document.get("evidence_checksum") != forecast["evidence_checksum"]
                or document.get("computed_at") != forecast["generated_at"]
                or type(document.get("distribution")) is not dict
            ):
                raise BarrierNotSatisfied(f"{race_id}/{bundle_id} forecast document disagrees")
            authenticated_forecasts.append(
                {
                    "race_id": race_id,
                    "bundle_id": bundle_id,
                    "forecast_checksum": forecast["forecast_checksum"],
                    "operation_id": forecast["operation_id"],
                    "service_run_id": forecast["service_run_id"],
                }
            )
        return {
            "assignment_id": cohort["assignment_id"],
            "authorization_operation_id": cohort["operation_id"],
            "members": [dict(member) for member in members],
            "forecasts": authenticated_forecasts,
            "excluded_race_ids": excluded,
        }

    def execute_application_command(
        self,
        command: ApplicationCommand,
        *,
        token: str,
        generation: int,
        at: datetime,
    ) -> Any:
        """Execute the only command value accepted from schedulers and adapters."""
        require_aware(at, "at")
        _strict_nonempty(token, "scheduler lease token")
        if type(generation) is not int or generation < 1:
            raise ValueError("scheduler lease generation must be a positive integer")
        claim_now = self.__clock()
        require_aware(claim_now, "trusted command clock")
        self.assert_lease(token, generation, at)
        self.assert_lease(token, generation, claim_now)
        command_payload_sha256 = command.payload_sha256()
        if isinstance(command.payload, CollectResults):
            with self.store._connect() as read:
                cohort = self._expected_cohort(read, command.racing_day_id)
                phase5_receipt = read.execute(
                    "SELECT r.result_json FROM phase7_day_command_plan p "
                    "JOIN phase7_application_command_receipts r "
                    "ON r.command_operation_id=p.command_operation_id "
                    "WHERE p.racing_day_id=? AND p.phase_ordinal=5 "
                    "AND p.phase_name='deferred_prediction'",
                    (command.racing_day_id,),
                ).fetchone()
                try:
                    received = (
                        None
                        if phase5_receipt is None
                        else json.loads(phase5_receipt["result_json"])
                    )
                    closed_cohort = self._phase5_forecast_cohort_authority(
                        read,
                        command.racing_day_id,
                        [row["race_id"] for row in cohort],
                    )
                except (BarrierNotSatisfied, json.JSONDecodeError) as error:
                    raise BarrierNotSatisfied(
                        "results require the closed Phase-5 cohort receipt"
                    ) from error
                if type(received) is not dict or received.get("forecast_cohort") != closed_cohort:
                    raise BarrierNotSatisfied(
                        "results require the exact closed Phase-5 cohort receipt"
                    )
                violating = next(
                    (
                        row["race_id"]
                        for row in cohort
                        if self._terminal_collection_quarantine(read, row["race_id"]) is None
                        and read.execute(
                            "SELECT 1 FROM deferred_predictions WHERE race_id=? "
                            "UNION ALL SELECT 1 FROM prediction_quarantines WHERE race_id=?",
                            (row["race_id"], row["race_id"]),
                        ).fetchone()
                        is None
                    ),
                    None,
                )
            if violating is not None:
                self.reject_result_before_prediction(
                    _derived_operation_id(
                        "result-before-prediction-attempt-v1",
                        str(command.operation_id),
                        command.racing_day_id,
                        violating,
                    ),
                    racing_day_id=command.racing_day_id,
                    race_id=violating,
                    at=at,
                )
        db = self.store._connect()
        db.execute("BEGIN IMMEDIATE")
        try:
            operation = db.execute(
                "SELECT kind FROM operations WHERE operation_id=?", (str(command.operation_id),)
            ).fetchone()
            receipt = db.execute(
                "SELECT * FROM phase7_application_command_receipts " "WHERE command_operation_id=?",
                (str(command.operation_id),),
            ).fetchone()
            if operation is not None:
                if (
                    receipt is None
                    or operation["kind"] != f"phase7_command_{command.phase}"
                    or receipt["racing_day_id"] != command.racing_day_id
                    or receipt["phase_name"] != command.phase
                    or receipt["command_payload_sha256"] != command_payload_sha256
                    or receipt["committed_at"] != iso_timestamp(at)
                ):
                    raise ConflictingOperation("application operation replay has different intent")
                result = json.loads(receipt["result_json"])
                db.rollback()
                return result
            claim = db.execute(
                "SELECT * FROM phase7_application_command_claims " "WHERE command_operation_id=?",
                (str(command.operation_id),),
            ).fetchone()
            if claim is not None and (
                claim["racing_day_id"] != command.racing_day_id
                or claim["phase_name"] != command.phase
                or claim["command_payload_sha256"] != command_payload_sha256
            ):
                raise ConflictingOperation("application command claim has different intent")
            latest_attempt = db.execute(
                "SELECT lease_generation,lease_token FROM phase7_application_command_attempts "
                "WHERE command_operation_id=? AND state IN ('claimed','recovering') "
                "ORDER BY attempt_id DESC LIMIT 1",
                (str(command.operation_id),),
            ).fetchone()
            if self.__command_executor is None:
                raise OperationalRejected(
                    "no trusted executor is configured for application commands"
                )
            if (
                latest_attempt is not None
                and latest_attempt["lease_generation"] == generation
                and latest_attempt["lease_token"] == token
            ):
                raise OperationalRejected(
                    "application command has an indeterminate claim still owned by this live lease"
                )
            ordinal = RaceCollectionService.ORDER.index(command.phase) + 1
            planned = db.execute(
                "SELECT p.command_operation_id,"
                "(p.lease_generation=? OR EXISTS ("
                " SELECT 1 FROM phase7_day_plan_adoptions adoption"
                " WHERE adoption.racing_day_id=p.racing_day_id"
                " AND adoption.lease_generation=?"
                " AND adoption.lease_token=?)) AS lease_authorized "
                "FROM phase7_day_command_plan p "
                "WHERE p.racing_day_id=? AND p.phase_ordinal=? AND p.phase_name=?",
                (generation, generation, token, command.racing_day_id, ordinal, command.phase),
            ).fetchone()
            completed = db.execute(
                "SELECT count(*) FROM phase7_scheduler_progress WHERE racing_day_id=?",
                (command.racing_day_id,),
            ).fetchone()[0]
            if (
                planned is None
                or planned["command_operation_id"] != str(command.operation_id)
                or not planned["lease_authorized"]
                or completed != ordinal - 1
            ):
                raise OperationalRejected(
                    "application command is outside the exact ordered Racing Day plan"
                )
            if (
                db.execute(
                    "SELECT 1 FROM racing_days WHERE racing_day_id=?", (command.racing_day_id,)
                ).fetchone()
                is None
            ):
                raise OperationalRejected("application command Racing Day is unavailable")
            state = "recovering" if claim is not None else "claimed"
            if claim is None:
                db.execute(
                    "INSERT INTO phase7_application_command_claims VALUES(?,?,?,?,?,?,?)",
                    (
                        str(command.operation_id),
                        command.racing_day_id,
                        command.phase,
                        command_payload_sha256,
                        generation,
                        token,
                        iso_timestamp(claim_now),
                    ),
                )
            db.execute(
                "INSERT INTO phase7_application_command_attempts("
                "command_operation_id,lease_generation,lease_token,state,recorded_at,details"
                ") VALUES(?,?,?,?,?,?)",
                (
                    str(command.operation_id),
                    generation,
                    token,
                    state,
                    iso_timestamp(claim_now),
                    "fresh durable claim" if claim is None else "recovery under fresh live lease",
                ),
            )
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
        if claim is not None:
            try:
                with self.store._connect() as read:
                    result = self._phase_postcondition(read, command)
            except (BarrierNotSatisfied, OperationalRejected):
                result = None
            except Exception as error:
                failed_at = self.__clock()
                self.__audit_command_attempt(
                    command,
                    token=token,
                    generation=generation,
                    state="postcondition_failed",
                    at=failed_at,
                    error=error,
                )
                raise OperationalRejected(
                    "command recovery postcondition could not be proven safely"
                ) from error
            else:
                commit_now = self.__clock()
                try:
                    return self.__commit_application_command(
                        command.operation_id,
                        racing_day_id=command.racing_day_id,
                        phase=command.phase,
                        result=result,
                        at=at,
                        token=token,
                        generation=generation,
                        authority_now=commit_now,
                        command=command,
                    )
                except OperationalRejected as error:
                    state = (
                        "fenced"
                        if "authoritative command attempt" in str(error)
                        else "postcondition_failed"
                    )
                    self.__audit_command_attempt(
                        command,
                        token=token,
                        generation=generation,
                        state=state,
                        at=self.__clock(),
                        error=error,
                    )
                    raise
        try:
            self.__command_executor.execute(command, at=at)
        except Exception as error:
            failure_now = self.__clock()
            self.__audit_command_attempt(
                command,
                token=token,
                generation=generation,
                state="handler_failed",
                at=failure_now,
                error=error,
            )
            raise
        commit_now = self.__clock()
        require_aware(commit_now, "trusted command clock")
        try:
            with self.store._connect() as db:
                result = self._phase_postcondition(db, command)
        except Exception as error:
            self.__audit_command_attempt(
                command,
                token=token,
                generation=generation,
                state="postcondition_failed",
                at=self.__clock(),
                error=error,
            )
            raise
        try:
            return self.__commit_application_command(
                command.operation_id,
                racing_day_id=command.racing_day_id,
                phase=command.phase,
                result=result,
                at=at,
                token=token,
                generation=generation,
                authority_now=commit_now,
                command=command,
            )
        except OperationalRejected as error:
            state = (
                "fenced"
                if "authoritative command attempt" in str(error)
                else "postcondition_failed"
            )
            self.__audit_command_attempt(
                command,
                token=token,
                generation=generation,
                state=state,
                at=self.__clock(),
                error=error,
            )
            raise

    def assert_lease(self, token: str, generation: int, now: datetime) -> None:
        require_aware(now, "now")
        with self.store._connect() as db:
            row = db.execute("SELECT * FROM phase7_scheduler_lease WHERE singleton=1").fetchone()
        if (
            row is None
            or row["lease_token"] != token
            or row["generation"] != generation
            or row["expires_at"] <= iso_timestamp(now)
        ):
            raise OperationalRejected("stale or absent scheduler ownership")

    def trusted_time(self) -> datetime:
        """Read the injected authority clock for an in-transaction fence."""
        now = self.__clock()
        require_aware(now, "trusted authority clock")
        return now

    def plan_racing_day(
        self,
        operation_id: OperationId,
        *,
        racing_day_id: str,
        lease_token: str,
        lease_generation: int,
        commands: Sequence[ApplicationCommand],
        at: datetime,
    ) -> bool:
        """Commit the exact nine command identities before any barrier can advance."""
        require_aware(at, "at")
        _strict_nonempty(lease_token, "scheduler lease token")
        expected = list(RaceCollectionService.ORDER)
        if (
            len(commands) != len(expected)
            or [command.phase for command in commands] != expected
            or any(command.racing_day_id != racing_day_id for command in commands)
            or len({command.operation_id for command in commands}) != len(commands)
        ):
            raise ValueError("Racing Day plan must contain the exact ordered nine commands")
        authority = self.store._connect()
        authority.execute("BEGIN IMMEDIATE")
        try:
            trusted_now = self.__clock()
            require_aware(trusted_now, "trusted planning clock")
            authority_at = iso_timestamp(trusted_now)
            lease = authority.execute(
                "SELECT lease_token,generation,acquired_at,expires_at "
                "FROM phase7_scheduler_lease WHERE singleton=1"
            ).fetchone()
            if (
                lease is None
                or lease["lease_token"] != lease_token
                or lease["generation"] != lease_generation
                or lease["acquired_at"] > authority_at
                or lease["expires_at"] <= authority_at
            ):
                raise OperationalRejected("Racing Day plan lacks the live scheduler lease")
            result = self._plan_racing_day_transaction(
                authority,
                operation_id=operation_id,
                racing_day_id=racing_day_id,
                lease_token=lease_token,
                lease_generation=lease_generation,
                commands=commands,
                authority_at=authority_at,
            )
            authority.commit()
            return result
        finally:
            if authority.in_transaction:
                authority.rollback()
            authority.close()

    def _plan_racing_day_transaction(
        self,
        db: sqlite3.Connection,
        *,
        operation_id: OperationId,
        racing_day_id: str,
        lease_token: str,
        lease_generation: int,
        commands: Sequence[ApplicationCommand],
        authority_at: str,
    ) -> bool:
        existing = db.execute(
            "SELECT p.phase_ordinal,p.phase_name,p.command_operation_id,p.lease_generation,"
            "p.planned_at,p.operation_id,"
            "o.kind FROM phase7_day_command_plan p "
            "JOIN operations o ON o.operation_id=p.operation_id "
            "WHERE p.racing_day_id=? ORDER BY p.phase_ordinal",
            (racing_day_id,),
        ).fetchall()
        payload_base = {
            "racing_day_id": racing_day_id,
            "lease_generation": lease_generation,
            "lease_token": lease_token,
            "commands": [
                {"phase": command.phase, "operation_id": str(command.operation_id)}
                for command in commands
            ],
        }
        if existing:
            identities = [
                (row["phase_ordinal"], row["phase_name"], row["command_operation_id"])
                for row in existing
            ]
            expected_identities = [
                (ordinal, command.phase, str(command.operation_id))
                for ordinal, command in enumerate(commands, 1)
            ]
            if identities != expected_identities:
                raise OperationalRejected(
                    "existing Racing Day plan has different command identities"
                )
            provenance = {(row["operation_id"], row["kind"]) for row in existing}
            if len(provenance) != 1:
                raise OperationalRejected("existing Racing Day plan has mixed provenance")
            plan_operation_id, plan_kind = next(iter(provenance))
            if plan_kind in (
                "phase7_migrate_v27_day_command_plan",
                "phase7_plan_racing_day",
            ) and any(row["lease_generation"] != lease_generation for row in existing):
                resolved_adoption: dict[str, str] = {}

                def adoption_payload(db: sqlite3.Connection) -> Mapping[str, Any]:
                    prior_adoption = db.execute(
                        "SELECT lease_generation,lease_token,adopted_at,"
                        "plan_operation_id,plan_kind "
                        "FROM phase7_day_plan_adoptions WHERE operation_id=?",
                        (str(operation_id),),
                    ).fetchone()
                    if prior_adoption is not None and (
                        prior_adoption["lease_generation"] != lease_generation
                        or prior_adoption["lease_token"] != lease_token
                        or prior_adoption["plan_operation_id"] != plan_operation_id
                        or prior_adoption["plan_kind"] != plan_kind
                    ):
                        raise OperationalRejected(
                            "existing Racing Day adoption has different authority"
                        )
                    if prior_adoption is None:
                        adopted_at = authority_at
                    else:
                        adopted_at = prior_adoption["adopted_at"]
                    resolved_adoption["at"] = adopted_at
                    return {
                        **payload_base,
                        "at": adopted_at,
                        "plan_operation_id": plan_operation_id,
                        "plan_kind": plan_kind,
                    }

                adoption_kind = (
                    "phase7_adopt_migrated_day_command_plan"
                    if plan_kind == "phase7_migrate_v27_day_command_plan"
                    else "phase7_adopt_day_command_plan"
                )
                with self.store._operation(
                    operation_id,
                    adoption_kind,
                    adoption_payload,
                    connection=db,
                ) as (db, replay):
                    adopted_at = resolved_adoption["at"]
                    lease = db.execute(
                        "SELECT 1 FROM phase7_scheduler_lease WHERE singleton=1 "
                        "AND lease_token=? AND generation=? AND acquired_at<=? AND expires_at>?",
                        (lease_token, lease_generation, authority_at, authority_at),
                    ).fetchone()
                    if lease is None:
                        raise OperationalRejected(
                            "Racing Day adoption lacks the live scheduler lease"
                        )
                    self._verify_plan_adoption(
                        db,
                        racing_day_id=racing_day_id,
                        commands=commands,
                        plan_operation_id=plan_operation_id,
                        plan_kind=plan_kind,
                    )
                    if not replay:
                        db.execute(
                            "INSERT INTO phase7_day_plan_adoptions VALUES(?,?,?,?,?,?,?)",
                            (
                                racing_day_id,
                                lease_generation,
                                lease_token,
                                adopted_at,
                                plan_operation_id,
                                plan_kind,
                                str(operation_id),
                            ),
                        )
                    if not replay and plan_kind == "phase7_migrate_v27_day_command_plan":
                        db.execute(
                            "INSERT INTO phase7_migrated_plan_adoptions VALUES(?,?,?,?,?)",
                            (
                                racing_day_id,
                                lease_generation,
                                adopted_at,
                                plan_operation_id,
                                str(operation_id),
                            ),
                        )
                return not replay
            if plan_kind != "phase7_plan_racing_day" or plan_operation_id != str(operation_id):
                raise OperationalRejected("existing Racing Day plan is not adoptable")
        resolved_plan: dict[str, str] = {}

        def plan_payload(db: sqlite3.Connection) -> Mapping[str, Any]:
            authority = db.execute(
                "SELECT lease_generation,lease_token,planned_at "
                "FROM phase7_day_plan_authorities "
                "WHERE racing_day_id=? AND operation_id=?",
                (racing_day_id, str(operation_id)),
            ).fetchone()
            if authority is not None:
                if (
                    authority["lease_generation"] != lease_generation
                    or authority["lease_token"] != lease_token
                    or any(row["planned_at"] != authority["planned_at"] for row in existing)
                ):
                    raise OperationalRejected("existing Racing Day plan has different authority")
                planned_at = authority["planned_at"]
            else:
                planned_at = authority_at
            resolved_plan["at"] = planned_at
            return {**payload_base, "at": planned_at}

        with self.store._operation(
            operation_id,
            "phase7_plan_racing_day",
            plan_payload,
            connection=db,
        ) as (
            db,
            replay,
        ):
            lease = db.execute(
                "SELECT lease_token,generation,acquired_at,expires_at "
                "FROM phase7_scheduler_lease WHERE singleton=1"
            ).fetchone()
            planned_at = resolved_plan["at"]
            if (
                lease is None
                or lease["lease_token"] != lease_token
                or lease["generation"] != lease_generation
                or lease["acquired_at"] > authority_at
                or lease["expires_at"] <= authority_at
            ):
                raise OperationalRejected("Racing Day plan lacks the live scheduler lease")
            if replay:
                durable_plan = db.execute(
                    "SELECT phase_ordinal,phase_name,command_operation_id,"
                    "lease_generation,planned_at,operation_id "
                    "FROM phase7_day_command_plan WHERE racing_day_id=? "
                    "ORDER BY phase_ordinal",
                    (racing_day_id,),
                ).fetchall()
                expected_plan = [
                    (
                        ordinal,
                        command.phase,
                        str(command.operation_id),
                        lease_generation,
                        planned_at,
                        str(operation_id),
                    )
                    for ordinal, command in enumerate(commands, 1)
                ]
                if [tuple(row) for row in durable_plan] != expected_plan:
                    raise OperationalRejected(
                        "existing Racing Day plan has different command identities"
                    )
                return False
            db.execute(
                "INSERT INTO phase7_day_plan_authorities VALUES(?,?,?,?,?)",
                (
                    racing_day_id,
                    lease_generation,
                    lease_token,
                    planned_at,
                    str(operation_id),
                ),
            )
            for ordinal, command in enumerate(commands, 1):
                db.execute(
                    "INSERT INTO phase7_day_command_plan VALUES(?,?,?,?,?,?,?)",
                    (
                        racing_day_id,
                        ordinal,
                        command.phase,
                        str(command.operation_id),
                        lease_generation,
                        planned_at,
                        str(operation_id),
                    ),
                )
        return True

    def _verify_plan_adoption(
        self,
        db: sqlite3.Connection,
        *,
        racing_day_id: str,
        commands: Sequence[ApplicationCommand],
        plan_operation_id: str,
        plan_kind: str,
    ) -> None:
        """Fail closed unless the immutable plan and restart prefix are canonical."""
        rows = db.execute(
            "SELECT phase_ordinal,phase_name,command_operation_id,operation_id "
            "FROM phase7_day_command_plan WHERE racing_day_id=? ORDER BY phase_ordinal",
            (racing_day_id,),
        ).fetchall()
        expected = [
            (ordinal, command.phase, str(command.operation_id), plan_operation_id)
            for ordinal, command in enumerate(commands, 1)
        ]
        if [tuple(row) for row in rows] != expected:
            raise OperationalRejected("existing Racing Day plan is corrupt or mismatched")
        provenance = db.execute(
            "SELECT kind FROM operations WHERE operation_id=?", (plan_operation_id,)
        ).fetchone()
        if provenance is None or provenance["kind"] != plan_kind:
            raise OperationalRejected("existing Racing Day plan provenance is unavailable")
        progress = db.execute(
            "SELECT p.*,r.racing_day_id receipt_day,r.phase_name receipt_phase,"
            "r.result_json receipt_json,r.result_checksum receipt_checksum,"
            "r.committed_at FROM phase7_scheduler_progress p "
            "LEFT JOIN phase7_application_command_receipts r "
            "ON r.command_operation_id=p.command_operation_id "
            "WHERE p.racing_day_id=? ORDER BY p.phase_ordinal",
            (racing_day_id,),
        ).fetchall()
        if [row["phase_ordinal"] for row in progress] != list(range(1, len(progress) + 1)):
            raise OperationalRejected("existing Racing Day progress is not contiguous")
        for row, command in zip(progress, commands, strict=False):
            try:
                canonical = _canonical(json.loads(row["result_json"])).decode()
            except Exception as error:
                raise OperationalRejected(
                    "existing Racing Day progress is not canonical"
                ) from error
            if (
                row["phase_name"] != command.phase
                or row["command_operation_id"] != str(command.operation_id)
                or row["receipt_day"] != racing_day_id
                or row["receipt_phase"] != command.phase
                or row["receipt_json"] != row["result_json"]
                or row["receipt_checksum"] != row["result_checksum"]
                or row["committed_at"] != row["completed_at"]
                or canonical != row["result_json"]
                or str(_checksum(json.loads(row["result_json"]))) != row["result_checksum"]
            ):
                raise OperationalRejected("existing Racing Day progress or receipt is corrupt")
            lease = db.execute(
                "SELECT acquired_at,max(expires_at,COALESCE(("
                " SELECT max(expires_at) FROM phase7_scheduler_renewals"
                " WHERE lease_generation=? AND renewed_at<=?"
                "),expires_at)) authoritative_expiry "
                "FROM phase7_scheduler_history WHERE generation=?",
                (
                    row["lease_generation"],
                    row["completed_at"],
                    row["lease_generation"],
                ),
            ).fetchone()
            if (
                lease is None
                or lease["acquired_at"] > row["completed_at"]
                or lease["authoritative_expiry"] <= row["completed_at"]
            ):
                raise OperationalRejected("existing Racing Day progress lacks lease authority")
        claims = db.execute(
            "SELECT c.*,p.phase_ordinal FROM phase7_application_command_claims c "
            "LEFT JOIN phase7_day_command_plan p "
            "ON p.command_operation_id=c.command_operation_id "
            "WHERE c.racing_day_id=?",
            (racing_day_id,),
        ).fetchall()
        for claim in claims:
            if (
                claim["phase_ordinal"] is None
                or claim["phase_ordinal"] > len(progress) + 1
                or claim["phase_name"] != rows[claim["phase_ordinal"] - 1]["phase_name"]
            ):
                raise OperationalRejected("existing Racing Day claim is corrupt")
            attempts = db.execute(
                "SELECT lease_generation,lease_token,state,recorded_at "
                "FROM phase7_application_command_attempts "
                "WHERE command_operation_id=? ORDER BY attempt_id",
                (claim["command_operation_id"],),
            ).fetchall()
            receipt = db.execute(
                "SELECT 1 FROM phase7_application_command_receipts " "WHERE command_operation_id=?",
                (claim["command_operation_id"],),
            ).fetchone()
            if (
                not attempts
                or attempts[0]["state"] != "claimed"
                or attempts[0]["lease_generation"] != claim["lease_generation"]
                or attempts[0]["lease_token"] != claim["lease_token"]
                or attempts[0]["recorded_at"] != claim["claimed_at"]
                or (attempts[-1]["state"] == "completed") != (receipt is not None)
            ):
                raise OperationalRejected("existing Racing Day interrupted claim is corrupt")

    def register_release(
        self, operation_id: OperationId, manifest: ReleaseManifest, at: datetime
    ) -> ArtifactChecksum:
        require_aware(at, "at")
        content = _canonical(manifest.document())
        artifact = self.artifacts.put(content, media_type="application/vnd.race-release+json")
        payload = {
            **manifest.document(),
            "manifest_checksum": str(artifact.checksum),
            "at": iso_timestamp(at),
        }
        with self.store._operation(operation_id, "phase7_register_release", payload) as (
            db,
            replay,
        ):
            if not replay:
                schema = db.execute("SELECT max(version) FROM schema_migrations").fetchone()[0]
                if schema != manifest.database_schema:
                    raise OperationalRejected(
                        "release manifest schema does not match migrated schema"
                    )
                config = db.execute(
                    "SELECT config_json FROM phase7_release_configurations "
                    "WHERE config_checksum=?",
                    (str(manifest.config_checksum),),
                ).fetchone()
                if config is None:
                    raise OperationalRejected("release configuration is not registered")
                config_document = json.loads(config["config_json"])
                if (
                    self.artifacts.read(manifest.config_checksum) != _canonical(config_document)
                    or config_document["service_root"] != manifest.service_root
                    or config_document["promotion_policy"] != manifest.policy_version
                    or tuple(config_document["bundle_versions"])
                    != manifest.supported_bundle_versions
                    or manifest.artifact_contract != "canonical-artifacts-v1"
                    or not set(manifest.supported_bundle_versions).issubset(
                        SUPPORTED_FORECAST_CONTRACTS
                    )
                ):
                    raise OperationalRejected(
                        "release manifest disagrees with typed configuration contract"
                    )
                policy = db.execute(
                    "SELECT artifact_checksum FROM phase6_policy_registry WHERE policy_id=?",
                    (manifest.policy_version,),
                ).fetchone()
                if policy is None:
                    raise OperationalRejected("release promotion policy is not registered")
                self.artifacts.verify(ArtifactChecksum(policy[0]))
                db.execute(
                    "INSERT INTO phase7_release_manifests VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        manifest.release_id,
                        str(artifact.checksum),
                        manifest.code_commit,
                        str(manifest.config_checksum),
                        schema,
                        manifest.artifact_contract,
                        manifest.policy_version,
                        json.dumps(manifest.supported_bundle_versions),
                        manifest.service_root,
                        iso_timestamp(at),
                        str(operation_id),
                    ),
                )
        return artifact.checksum

    def register_configuration(
        self, operation_id: OperationId, configuration: ReleaseConfiguration, at: datetime
    ) -> ArtifactChecksum:
        require_aware(at, "at")
        content = _canonical(configuration.document())
        artifact = self.artifacts.put(
            content, media_type="application/vnd.race-release-config+json"
        )
        payload = {
            "configuration": configuration.document(),
            "checksum": str(artifact.checksum),
            "at": iso_timestamp(at),
        }
        with self.store._operation(operation_id, "phase7_register_configuration", payload) as (
            db,
            replay,
        ):
            if not replay:
                db.execute(
                    "INSERT INTO phase7_release_configurations VALUES(?,?,?,?,?,?)",
                    (
                        str(artifact.checksum),
                        configuration.schema_version,
                        content.decode(),
                        configuration.service_root,
                        iso_timestamp(at),
                        str(operation_id),
                    ),
                )
        return artifact.checksum

    def _verified_release(self, db: sqlite3.Connection, release_id: str) -> sqlite3.Row:
        return verify_release_authority(db, self.artifacts, release_id)

    def _verify_operational_evidence(
        self,
        db: sqlite3.Connection,
        *,
        racing_day_id: str,
        release_id: str,
        evidence_kind: str,
        checksum: ArtifactChecksum,
        require_complete_ordering: bool = True,
    ) -> Mapping[str, Any]:
        release = self._verified_release(db, release_id)
        row = db.execute(
            "SELECT * FROM phase7_operational_evidence WHERE artifact_checksum=? "
            "AND evidence_kind=? AND racing_day_id=? AND release_id=?",
            (str(checksum), evidence_kind, racing_day_id, release_id),
        ).fetchone()
        if row is None:
            raise OperationalRejected(f"{evidence_kind} evidence authority is missing")
        self.artifacts.verify(checksum)
        manifest_bytes = self.artifacts.read(ArtifactChecksum(row["manifest_checksum"]))
        if manifest_bytes != _canonical(json.loads(row["manifest_json"])):
            raise OperationalRejected(f"{evidence_kind} manifest artifact disagrees")
        manifest = json.loads(manifest_bytes)
        expected_versions = {
            "code": release["code_commit"],
            "config": release["config_checksum"],
            "schema": release["schema_version"],
            "policy": release["policy_version"],
            "bundles": json.loads(release["bundle_versions_json"]),
        }
        if (
            manifest.get("schema_version") != "phase7-operational-evidence-v1"
            or manifest.get("evidence_kind") != evidence_kind
            or manifest.get("racing_day_id") != racing_day_id
            or manifest.get("release_id") != release_id
            or manifest.get("artifact_checksum") != str(checksum)
            or manifest.get("versions") != expected_versions
        ):
            raise OperationalRejected(f"{evidence_kind} manifest binding is invalid")
        checks = manifest.get("checks")
        if not isinstance(checks, dict):
            raise OperationalRejected(f"{evidence_kind} checks are malformed")
        progress = db.execute(
            "SELECT phase_ordinal,phase_name,lease_generation,command_operation_id,"
            "result_checksum FROM phase7_scheduler_progress WHERE racing_day_id=? "
            "ORDER BY phase_ordinal",
            (racing_day_id,),
        ).fetchall()
        if evidence_kind == "restart":
            recovered_phase = checks.get("recovered_phase")
            adoption_operation_id = checks.get("adoption_operation_id")
            adoption = db.execute(
                "SELECT lease_generation FROM phase7_day_plan_adoptions "
                "WHERE racing_day_id=? AND operation_id=?",
                (racing_day_id, adoption_operation_id),
            ).fetchone()
            if (
                type(recovered_phase) is not int
                or recovered_phase < 1
                or recovered_phase > len(progress)
                or adoption is None
            ):
                raise OperationalRejected("restart evidence has no durable recovered prefix")
            recovered = progress[:recovered_phase]
            state = [dict(item) for item in recovered]
            if checks != {
                "adoption_operation_id": adoption_operation_id,
                "lease_generation": adoption["lease_generation"],
                "recovered_phase": recovered_phase,
                "state_checksum": str(_checksum(state)),
            } or any(row["lease_generation"] >= adoption["lease_generation"] for row in recovered):
                raise OperationalRejected("restart evidence disagrees with scheduler state")
        elif evidence_kind == "ordering":
            plan = db.execute(
                "SELECT phase_name,command_operation_id FROM phase7_day_command_plan "
                "WHERE racing_day_id=? ORDER BY phase_ordinal",
                (racing_day_id,),
            ).fetchall()
            planned_operations = [item["command_operation_id"] for item in plan]
            planned_phases = [item["phase_name"] for item in plan]
            committed_operations = [item["command_operation_id"] for item in progress]
            committed_phases = [item["phase_name"] for item in progress]
            rejections = [
                row[0]
                for row in db.execute(
                    "SELECT r.operation_id FROM phase7_rejected_result_commands r "
                    "JOIN phase7_alerts a ON a.operation_id=r.operation_id "
                    "AND a.alert_id='result-rejection:'||r.operation_id "
                    "AND a.category='result_before_prediction' "
                    "AND a.racing_day_id=r.racing_day_id "
                    "WHERE r.racing_day_id=? ORDER BY r.rejected_at,r.operation_id",
                    (racing_day_id,),
                )
            ]
            if (
                planned_phases != list(RaceCollectionService.ORDER)
                or committed_phases != planned_phases[: len(committed_phases)]
                or committed_operations != planned_operations[: len(committed_operations)]
                or (require_complete_ordering and len(progress) != len(plan))
                or checks
                != {
                    "phase_operations": planned_operations,
                    "result_before_prediction_rejection_operations": rejections,
                }
            ):
                raise OperationalRejected("ordering evidence disagrees with durable barriers")
        elif evidence_kind == "determinism":
            try:
                input_checksum = ArtifactChecksum(checks["input_checksum"])
                output = ArtifactChecksum(checks["output_checksum"])
                replay = ArtifactChecksum(checks["replay_output_checksum"])
                first_operation_id = checks["first_execution_operation_id"]
                replay_operation_id = checks["replay_execution_operation_id"]
            except (KeyError, ValueError) as error:
                raise OperationalRejected("determinism checks are invalid") from error
            executions = db.execute(
                "SELECT operation_id,release_manifest_checksum,config_checksum,"
                "bundle_authority_checksum,runner_identity,"
                "runner_implementation_version,input_checksum,output_checksum FROM "
                "phase7_determinism_executions WHERE racing_day_id=? AND release_id=? "
                "AND operation_id IN (?,?)",
                (
                    racing_day_id,
                    release_id,
                    first_operation_id,
                    replay_operation_id,
                ),
            ).fetchall()
            self.artifacts.verify(input_checksum)
            self.artifacts.verify(output)
            self.artifacts.verify(replay)
            bundle_authority = self.artifacts.put(
                _canonical(
                    {
                        "release_id": release_id,
                        "supported_bundle_versions": json.loads(release["bundle_versions_json"]),
                    }
                ),
                media_type="application/vnd.race-determinism-bundle-authority+json",
            )
            if (
                first_operation_id == replay_operation_id
                or len(executions) != 2
                or any(row["input_checksum"] != str(input_checksum) for row in executions)
                or any(
                    row["release_manifest_checksum"] != release["manifest_checksum"]
                    or row["config_checksum"] != release["config_checksum"]
                    or row["bundle_authority_checksum"] != str(bundle_authority.checksum)
                    or row["runner_identity"] != "race_collection.phase7.closed_replay"
                    or row["runner_implementation_version"] != "phase7-determinism-runner-v1"
                    for row in executions
                )
                or {(row["operation_id"], row["output_checksum"]) for row in executions}
                != {
                    (first_operation_id, str(output)),
                    (replay_operation_id, str(replay)),
                }
                or output != replay
            ):
                raise OperationalRejected("determinism replay output differs")
        return manifest

    def verify_operational_evidence(
        self,
        db: sqlite3.Connection,
        *,
        racing_day_id: str,
        release_id: str,
        evidence_kind: str,
        checksum: ArtifactChecksum,
        require_complete_ordering: bool = True,
    ) -> Mapping[str, Any]:
        """Public cross-phase verifier; it never records or projects new evidence."""
        return self._verify_operational_evidence(
            db,
            racing_day_id=racing_day_id,
            release_id=release_id,
            evidence_kind=evidence_kind,
            checksum=checksum,
            require_complete_ordering=require_complete_ordering,
        )

    def verify_day_evidence(
        self,
        *,
        racing_day_id: str,
        release_id: str,
        restart_checksum: ArtifactChecksum,
        ordering_checksum: ArtifactChecksum,
        determinism_checksum: ArtifactChecksum,
    ) -> None:
        """Re-authenticate all operational evidence immediately before consumption."""
        with self.store._connect() as db:
            for kind, checksum in (
                ("restart", restart_checksum),
                ("ordering", ordering_checksum),
                ("determinism", determinism_checksum),
            ):
                self._verify_operational_evidence(
                    db,
                    racing_day_id=racing_day_id,
                    release_id=release_id,
                    evidence_kind=kind,
                    checksum=checksum,
                )

    def record_operational_evidence(
        self,
        operation_id: OperationId,
        *,
        checksum: ArtifactChecksum,
        evidence_kind: str,
        racing_day_id: str,
        release_id: str,
        checks: Mapping[str, Any],
        at: datetime,
    ) -> bool:
        """Authenticate content existence and checksum before it can advance workflow."""
        require_aware(at, "at")
        verified = self.artifacts.verify(checksum)
        required = {
            "restart": {
                "adoption_operation_id",
                "lease_generation",
                "recovered_phase",
                "state_checksum",
            },
            "ordering": {
                "phase_operations",
                "result_before_prediction_rejection_operations",
            },
            "determinism": {
                "input_checksum",
                "first_execution_operation_id",
                "output_checksum",
                "replay_execution_operation_id",
                "replay_output_checksum",
            },
        }
        if evidence_kind not in required or set(checks) != required[evidence_kind]:
            raise ValueError("operational evidence check shape is invalid")
        try:
            proof = json.loads(self.artifacts.read(checksum))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise OperationalRejected("operational evidence artifact must be typed JSON") from error
        if proof != {
            "schema_version": "phase7-check-proof-v1",
            "evidence_kind": evidence_kind,
            "racing_day_id": racing_day_id,
            "release_id": release_id,
            "checks": dict(checks),
        }:
            raise OperationalRejected("operational evidence artifact binding is invalid")
        with self.store._connect() as read:
            release = self._verified_release(read, release_id)
            day = read.execute(
                "SELECT 1 FROM racing_days WHERE racing_day_id=?", (racing_day_id,)
            ).fetchone()
        if release is None or day is None:
            raise OperationalRejected("evidence day or release is not registered")
        manifest = {
            "schema_version": "phase7-operational-evidence-v1",
            "evidence_kind": evidence_kind,
            "racing_day_id": racing_day_id,
            "release_id": release_id,
            "artifact_checksum": str(checksum),
            "checks": dict(checks),
            "versions": {
                "code": release["code_commit"],
                "config": release["config_checksum"],
                "schema": release["schema_version"],
                "policy": release["policy_version"],
                "bundles": json.loads(release["bundle_versions_json"]),
            },
        }
        manifest_content = _canonical(manifest)
        manifest_artifact = self.artifacts.put(
            manifest_content, media_type="application/vnd.race-operational-evidence+json"
        )
        payload = {
            "manifest": manifest,
            "manifest_checksum": str(manifest_artifact.checksum),
            "byte_size": verified.byte_size,
            "at": iso_timestamp(at),
        }
        with self.store._operation(operation_id, "phase7_record_operational_evidence", payload) as (
            db,
            replay,
        ):
            if replay:
                return False
            db.execute(
                "INSERT INTO phase7_operational_evidence VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    str(checksum),
                    evidence_kind,
                    racing_day_id,
                    release_id,
                    str(manifest_artifact.checksum),
                    manifest_content.decode(),
                    verified.byte_size,
                    iso_timestamp(at),
                    str(operation_id),
                ),
            )
            self._verify_operational_evidence(
                db,
                racing_day_id=racing_day_id,
                release_id=release_id,
                evidence_kind=evidence_kind,
                checksum=checksum,
                require_complete_ordering=False,
            )
        return True

    def record_determinism_execution(
        self,
        operation_id: OperationId,
        *,
        racing_day_id: str,
        release_id: str,
        input_checksum: ArtifactChecksum,
        at: datetime,
    ) -> ArtifactChecksum:
        """Execute the closed deterministic replay runner over immutable input."""
        require_aware(at, "at")
        runner_identity = "race_collection.phase7.closed_replay"
        runner_version = "phase7-determinism-runner-v1"
        try:
            input_content = self.artifacts.read(input_checksum)
        except (ArtifactStoreError, ValueError) as error:
            raise OperationalRejected("determinism input artifact is unavailable") from error

        def intent(db: sqlite3.Connection) -> Mapping[str, Any]:
            release = self._verified_release(db, release_id)
            if (
                db.execute(
                    "SELECT 1 FROM racing_days WHERE racing_day_id=?", (racing_day_id,)
                ).fetchone()
                is None
            ):
                raise OperationalRejected("determinism execution Racing Day is unavailable")
            bundle_authority = self.artifacts.put(
                _canonical(
                    {
                        "release_id": release_id,
                        "supported_bundle_versions": json.loads(release["bundle_versions_json"]),
                    }
                ),
                media_type="application/vnd.race-determinism-bundle-authority+json",
            )
            output = self.artifacts.put(
                _canonical(
                    {
                        "schema_version": "phase7-determinism-output-v1",
                        "racing_day_id": racing_day_id,
                        "release_id": release_id,
                        "release_manifest_checksum": release["manifest_checksum"],
                        "config_checksum": release["config_checksum"],
                        "bundle_authority_checksum": str(bundle_authority.checksum),
                        "runner_identity": runner_identity,
                        "runner_implementation_version": runner_version,
                        "input_checksum": str(input_checksum),
                        "input_content_sha256": hashlib.sha256(input_content).hexdigest(),
                    }
                ),
                media_type="application/vnd.race-determinism-output+json",
            )
            return {
                "racing_day_id": racing_day_id,
                "release_id": release_id,
                "release_manifest_checksum": release["manifest_checksum"],
                "config_checksum": release["config_checksum"],
                "bundle_authority_checksum": str(bundle_authority.checksum),
                "runner_identity": runner_identity,
                "runner_implementation_version": runner_version,
                "input_checksum": str(input_checksum),
                "output_checksum": str(output.checksum),
                "at": iso_timestamp(at),
            }

        with self.store._operation(operation_id, "phase7_record_determinism_execution", intent) as (
            db,
            replay,
        ):
            if replay:
                row = db.execute(
                    "SELECT output_checksum FROM phase7_determinism_executions "
                    "WHERE operation_id=?",
                    (str(operation_id),),
                ).fetchone()
                if row is None:
                    raise OperationalRejected("determinism execution replay is missing")
                return ArtifactChecksum(row["output_checksum"])
            payload = intent(db)
            db.execute(
                "INSERT INTO phase7_determinism_executions VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    str(operation_id),
                    racing_day_id,
                    release_id,
                    payload["release_manifest_checksum"],
                    payload["config_checksum"],
                    payload["bundle_authority_checksum"],
                    runner_identity,
                    runner_version,
                    str(input_checksum),
                    payload["output_checksum"],
                    iso_timestamp(at),
                ),
            )
        return ArtifactChecksum(payload["output_checksum"])

    @staticmethod
    def generate_units(
        manifest: ReleaseManifest,
        configuration: ReleaseConfiguration,
        *,
        config_path: str,
        python_executable: str,
    ) -> Mapping[str, str]:
        configuration_path = Path(config_path)
        if not configuration_path.is_absolute():
            raise ValueError("config path must be absolute")
        _safe_operational_path(config_path)
        if configuration_path.is_symlink() or not configuration_path.is_file():
            raise OperationalRejected(
                "service configuration must be an existing regular non-symlink file"
            )
        try:
            configuration_bytes = configuration_path.read_bytes()
        except OSError as error:
            raise OperationalRejected("service configuration is unreadable") from error
        if configuration_bytes != _canonical(configuration.document()):
            raise OperationalRejected(
                "service configuration bytes disagree with the authenticated configuration"
            )
        python_path = Path(python_executable)
        if (
            not python_path.is_absolute()
            or not python_path.is_file()
            or python_path.stat().st_mode & 0o111 == 0
        ):
            raise OperationalRejected("Python interpreter is unavailable or not executable")
        _safe_operational_path(python_executable)
        if any(
            any(
                character.isspace() or character in {"%", "$", "\\", '"', "'"}
                for character in value
            )
            for value in (manifest.service_root, config_path, python_executable)
        ):
            raise ValueError("systemd command paths contain unsupported syntax")
        try:
            probe = subprocess.run(
                (
                    python_executable,
                    "-c",
                    "import json,sys; print(json.dumps("
                    "{'executable':sys.executable,"
                    "'version':[sys.version_info.major,sys.version_info.minor]},"
                    "sort_keys=True,separators=(',',':')))",
                ),
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            raise OperationalRejected(
                "Python interpreter identity could not be verified"
            ) from error
        try:
            python_identity = json.loads(probe.stdout)
        except json.JSONDecodeError as error:
            raise OperationalRejected("Python interpreter identity is malformed") from error
        if python_identity != {
            "executable": python_executable,
            "version": [3, 11],
        }:
            raise OperationalRejected(
                "Race Collection Service requires the exact supplied Python 3.11 environment"
            )
        actual = _checksum(configuration.document())
        if (
            actual != manifest.config_checksum
            or configuration.service_root != manifest.service_root
            or configuration.promotion_policy != manifest.policy_version
            or configuration.bundle_versions != manifest.supported_bundle_versions
        ):
            raise OperationalRejected("unit configuration disagrees with immutable release")
        service = "\n".join(
            (
                "[Unit]",
                "Description=Race Collection Service",
                "After=network-online.target",
                "",
                "[Service]",
                "Type=simple",
                f"WorkingDirectory={manifest.service_root}",
                f"ExecStart={python_executable} "
                f"{manifest.service_root}/bin/race-collection-service "
                f"--config {config_path} --continuous",
                "Restart=on-failure",
                "NoNewPrivileges=true",
                "",
                "[Install]",
                "WantedBy=default.target",
                "",
            )
        )
        return {"race-collection.service": service}  # no workflow-owning timers

    def reconcile_day(
        self,
        operation_id: OperationId,
        *,
        racing_day_id: str,
        release_id: str,
        restart_checksum: ArtifactChecksum,
        ordering_checksum: ArtifactChecksum,
        determinism_checksum: ArtifactChecksum,
        at: datetime,
    ) -> Mapping[str, Any]:
        require_aware(at, "at")
        payload = {
            "day": racing_day_id,
            "release": release_id,
            "restart": str(restart_checksum),
            "ordering": str(ordering_checksum),
            "determinism": str(determinism_checksum),
            "at": iso_timestamp(at),
        }
        with self.store._operation(operation_id, "phase7_reconciliation", payload) as (db, replay):
            if replay:
                row = db.execute(
                    "SELECT report_json FROM phase7_reconciliation WHERE racing_day_id=?",
                    (racing_day_id,),
                ).fetchone()
                if row is None:
                    raise OperationalRejected("reconciliation replay has no durable outcome")
                return json.loads(row[0])
            day = db.execute(
                "SELECT * FROM racing_days WHERE racing_day_id=?", (racing_day_id,)
            ).fetchone()
            if day is None or day["closed_at"] is None:
                raise BarrierNotSatisfied("Racing Day must be durably closed before reconciliation")
            races = db.execute(
                "SELECT * FROM races WHERE racing_day_id=?", (racing_day_id,)
            ).fetchall()
            expected = db.execute(
                "SELECT e.* FROM expected_races e JOIN races r USING(race_id) WHERE r.racing_day_id=?",
                (racing_day_id,),
            ).fetchall()
            ids = [r["race_id"] for r in races]

            def count(table: str, condition: str = "1") -> int:
                if not ids:
                    return 0
                marks = ",".join("?" for _ in ids)
                return db.execute(
                    f"SELECT count(*) FROM {table} WHERE race_id IN ({marks}) AND {condition}", ids
                ).fetchone()[0]

            scoped_supersessions = db.execute(
                "SELECT count(*) FROM supersessions WHERE prior_id IN (SELECT race_id FROM races "
                "WHERE racing_day_id=?) OR replacement_id IN (SELECT race_id FROM races "
                "WHERE racing_day_id=?)",
                (racing_day_id, racing_day_id),
            ).fetchone()[0]
            metrics = {
                "expected": len(expected),
                "discovered": len(races),
                "runner_sets": count("field_evidence", "field_name='runner_set'"),
                "boxes": count("field_evidence", "field_name='box'"),
                "odds_attempts": count("odds_attempts"),
                "odds_successes": count("odds_attempts", "status='succeeded'"),
                "final_odds": count("sealed_evidence"),
                "seals": count("sealed_evidence"),
                "prediction_commits": count("deferred_predictions"),
                "prediction_quarantines": count("prediction_quarantines"),
                "result_collected": count("result_attempts", "status='collected'"),
                "training_joins": count("training_examples"),
                "retries": count("result_attempts", "attempt_number>1"),
                "supersessions": scoped_supersessions,
                "failures": count("odds_attempts", "status='failed'")
                + count("result_attempts", "status='failed'"),
            }
            terminal = {"training_example_ready", "evaluation_ineligible", "result_quarantined"}
            mismatches: list[str] = []

            def add(code: str, race_id: str | None = None) -> None:
                mismatches.append(f"{race_id}:{code}" if race_id else code)

            if not expected:
                add("expected_inventory_empty")
            expected_ids = {row["race_id"] for row in expected}
            for expected_race in expected:
                try:
                    self.artifacts.verify(ArtifactChecksum(expected_race["programme_checksum"]))
                except ArtifactStoreError:
                    add("programme_artifact_missing_or_corrupt", expected_race["race_id"])
            programme_ids = {
                row[0]
                for row in db.execute(
                    "SELECT DISTINCT o.race_id FROM programme_race_observations o "
                    "JOIN races r USING(race_id) WHERE r.racing_day_id=? AND o.collision=0",
                    (racing_day_id,),
                )
            }
            for race_id in sorted(set(ids) - expected_ids):
                add("discovered_without_expected", race_id)
            for race_id in sorted(expected_ids - programme_ids):
                add("expected_missing_programme_observation", race_id)
            for race in races:
                race_id = race["race_id"]
                collection_quarantine = self._terminal_collection_quarantine(db, race_id)
                if collection_quarantine is not None:
                    continue
                if race["state"] not in terminal:
                    add(f"non_terminal:{race['state']}", race_id)
                resolved = {
                    row["field_name"]: row
                    for row in db.execute(
                        "SELECT * FROM phase6_resolved_field_evidence WHERE race_id=?",
                        (race_id,),
                    )
                }
                try:
                    runners = json.loads(resolved["runner_set"]["value_json"])
                    boxes = json.loads(resolved["box"]["value_json"])
                    if not isinstance(runners, list) or not isinstance(boxes, dict):
                        raise ValueError
                    if set(boxes) != set(runners) or len(set(boxes.values())) != len(runners):
                        raise ValueError
                    if any(
                        resolved[field]["distinct_top_values"] != 1
                        for field in ("runner_set", "box")
                    ):
                        add("runner_box_authority_conflict", race_id)
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    add("runner_box_incomplete", race_id)
                odds = db.execute(
                    "SELECT * FROM odds_attempts WHERE race_id=? "
                    "ORDER BY scheduled_due_at,attempted_at",
                    (race_id,),
                ).fetchall()
                successes = [row for row in odds if row["status"] == "succeeded"]
                seal = db.execute(
                    "SELECT * FROM sealed_evidence WHERE race_id=?", (race_id,)
                ).fetchone()
                if not odds:
                    add("odds_attempt_missing", race_id)
                if not successes:
                    add("odds_success_missing", race_id)
                for attempt in successes:
                    for column in ("artifact_checksum", "runner_mapping_checksum"):
                        try:
                            self.artifacts.verify(ArtifactChecksum(attempt[column]))
                        except ArtifactStoreError:
                            add(f"odds_{column}_missing_or_corrupt", race_id)
                expected_race = db.execute(
                    "SELECT scheduled_jump FROM expected_races WHERE race_id=?", (race_id,)
                ).fetchone()
                if expected_race is None:
                    add("scheduled_jump_authority_missing", race_id)
                else:
                    scheduled_jump = datetime.fromisoformat(expected_race[0])
                    if not _adaptive_odds_history_complete(
                        odds,
                        discovery_at=datetime.fromisoformat(race["discovered_at"]),
                        scheduled_jump=scheduled_jump,
                        cutoff=(
                            datetime.fromisoformat(seal["frozen_at"])
                            if seal is not None
                            else scheduled_jump
                        ),
                    ):
                        add("odds_cadence_gap", race_id)
                for mismatch in _odds_snapshot_mismatches(successes, seal, self.artifacts):
                    add(mismatch, race_id)
                prediction = db.execute(
                    "SELECT * FROM deferred_predictions WHERE race_id=?", (race_id,)
                ).fetchone()
                quarantine = db.execute(
                    "SELECT 1 FROM prediction_quarantines WHERE race_id=?", (race_id,)
                ).fetchone()
                if (prediction is None) == (quarantine is None):
                    add("prediction_outcome_not_exact", race_id)
                result = db.execute(
                    "SELECT * FROM result_attempts WHERE race_id=? AND status='collected'",
                    (race_id,),
                ).fetchone()
                if prediction is not None and result is not None:
                    if result["attempted_at"] <= prediction["computed_at"]:
                        add("result_before_prediction", race_id)
                    try:
                        if json.loads(
                            self.artifacts.read(ArtifactChecksum(result["artifact_checksum"]))
                        ) != json.loads(result["outcome_json"]):
                            add("result_provenance_disagrees", race_id)
                    except (ArtifactStoreError, json.JSONDecodeError, UnicodeDecodeError):
                        add("result_artifact_missing_or_corrupt", race_id)
                if (
                    result is not None
                    and db.execute(
                        "SELECT 1 FROM training_examples WHERE race_id=?", (race_id,)
                    ).fetchone()
                    is None
                ):
                    add("training_join_missing", race_id)
            manifest = self._verified_release(db, release_id)
            relevant_ids = {
                race_id
                for race_id in expected_ids
                if self._terminal_collection_quarantine(db, race_id) is None
            }
            assignment = db.execute(
                "SELECT d.bundle_id,d.bundle_checksum FROM canonical_day_assignments d "
                "WHERE d.racing_day_id=?",
                (racing_day_id,),
            ).fetchone()
            if relevant_ids and assignment is None:
                add("day_champion_assignment_missing")
            cohort_rows = db.execute(
                "SELECT c.assignment_id,m.role,m.bundle_id,m.bundle_checksum,"
                "m.service_run_id FROM phase7_day_forecast_cohorts c "
                "JOIN phase7_day_forecast_cohort_members m USING(racing_day_id) "
                "WHERE c.racing_day_id=? ORDER BY m.bundle_id",
                (racing_day_id,),
            ).fetchall()
            authorized_cohort = {(row["bundle_id"], row["bundle_checksum"]) for row in cohort_rows}
            if relevant_ids and not cohort_rows:
                add("day_forecast_cohort_missing")
            elif (
                relevant_ids
                and assignment is not None
                and (
                    len([row for row in cohort_rows if row["role"] == "champion"]) != 1
                    or (
                        assignment["bundle_id"],
                        assignment["bundle_checksum"],
                    )
                    not in authorized_cohort
                )
            ):
                add("day_forecast_cohort_assignment_mismatch")
            elif relevant_ids and not any(row["role"] == "challenger" for row in cohort_rows):
                add("challenger_cohort_missing")
            for race_id in sorted(relevant_ids):
                forecasts = db.execute(
                    "SELECT f.bundle_id,f.bundle_checksum,f.forecast_checksum,"
                    "f.artifact_checksum,f.service_run_id,f.generated_at "
                    "FROM phase6_forecast_service_artifacts f "
                    "JOIN phase7_day_forecast_commands c "
                    "ON c.race_id=f.race_id AND c.bundle_id=f.bundle_id "
                    "AND c.operation_id=f.operation_id "
                    "WHERE c.racing_day_id=? AND f.race_id=? ORDER BY f.bundle_id",
                    (racing_day_id, race_id),
                ).fetchall()
                unauthorized = db.execute(
                    "SELECT f.bundle_id FROM phase6_forecast_service_artifacts f "
                    "WHERE f.race_id=? AND NOT EXISTS ("
                    " SELECT 1 FROM phase7_day_forecast_commands c "
                    " WHERE c.racing_day_id=? AND c.race_id=f.race_id "
                    " AND c.bundle_id=f.bundle_id AND c.operation_id=f.operation_id)",
                    (race_id, racing_day_id),
                ).fetchall()
                coverage = {(row["bundle_id"], row["bundle_checksum"]) for row in forecasts}
                if unauthorized:
                    add("unauthorized_forecast_cohort_member", race_id)
                if coverage != authorized_cohort:
                    add("day_forecast_cohort_coverage_mismatch", race_id)
                if (
                    assignment is not None
                    and (assignment["bundle_id"], assignment["bundle_checksum"]) not in coverage
                ):
                    add("assigned_champion_forecast_missing", race_id)
                challengers = (
                    coverage
                    if assignment is None
                    else coverage - {(assignment["bundle_id"], assignment["bundle_checksum"])}
                )
                if not challengers:
                    add("challenger_cohort_missing", race_id)
                for forecast in forecasts:
                    bundle = db.execute(
                        "SELECT * FROM canonical_model_bundles WHERE bundle_id=? "
                        "AND bundle_checksum=?",
                        (forecast["bundle_id"], forecast["bundle_checksum"]),
                    ).fetchone()
                    if bundle is None:
                        add(f"forecast_bundle_unregistered:{forecast['bundle_id']}", race_id)
                        continue
                    if bundle["forecast_contract_version"] not in json.loads(
                        manifest["bundle_versions_json"]
                    ):
                        add(f"forecast_bundle_contract_stale:{forecast['bundle_id']}", race_id)
                    try:
                        self.artifacts.verify(ArtifactChecksum(forecast["artifact_checksum"]))
                        self.artifacts.verify(ArtifactChecksum(bundle["bundle_checksum"]))
                        for component in db.execute(
                            "SELECT artifact_checksum FROM "
                            "phase7_day_forecast_cohort_components "
                            "WHERE racing_day_id=? AND bundle_id=?",
                            (racing_day_id, forecast["bundle_id"]),
                        ):
                            self.artifacts.verify(ArtifactChecksum(component[0]))
                    except ArtifactStoreError:
                        add(f"forecast_bundle_artifact_missing:{forecast['bundle_id']}", race_id)
            mismatches.sort()
            report = {
                "schema_version": "phase7-reconciliation-v1",
                "racing_day_id": racing_day_id,
                "metrics": metrics,
                "mismatches": mismatches,
                "versions": (
                    None
                    if manifest is None
                    else {
                        "code": manifest["code_commit"],
                        "config": manifest["config_checksum"],
                        "schema": manifest["schema_version"],
                        "artifact_contract": manifest["artifact_contract"],
                        "policy": manifest["policy_version"],
                        "bundles": json.loads(manifest["bundle_versions_json"]),
                    }
                ),
            }
            artifact = self.artifacts.put(
                _canonical(report), media_type="application/vnd.race-reconciliation+json"
            )
            for checksum in (restart_checksum, ordering_checksum, determinism_checksum):
                self.artifacts.verify(checksum)
            required = (
                (restart_checksum, "restart"),
                (ordering_checksum, "ordering"),
                (determinism_checksum, "determinism"),
            )
            for checksum, kind in required:
                self._verify_operational_evidence(
                    db,
                    racing_day_id=racing_day_id,
                    release_id=release_id,
                    evidence_kind=kind,
                    checksum=checksum,
                    require_complete_ordering=kind != "ordering",
                )
            complete = not mismatches
            evidence_manifest = {
                "schema_version": "phase7-operational-evidence-v1",
                "evidence_kind": "reconciliation",
                "racing_day_id": racing_day_id,
                "release_id": release_id,
                "artifact_checksum": str(artifact.checksum),
                "checks": {"mismatch_count": len(mismatches), "complete": complete},
                "versions": report["versions"],
            }
            manifest_artifact = self.artifacts.put(
                _canonical(evidence_manifest),
                media_type="application/vnd.race-operational-evidence+json",
            )
            db.execute(
                "INSERT INTO phase7_operational_evidence VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    str(artifact.checksum),
                    "reconciliation",
                    racing_day_id,
                    release_id,
                    str(manifest_artifact.checksum),
                    _canonical(evidence_manifest).decode(),
                    len(_canonical(report)),
                    iso_timestamp(at),
                    str(operation_id),
                ),
            )
            db.execute(
                "INSERT INTO phase7_day_evidence VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    racing_day_id,
                    str(artifact.checksum),
                    str(restart_checksum),
                    str(ordering_checksum),
                    str(determinism_checksum),
                    int(complete),
                    0,
                    release_id,
                    iso_timestamp(at),
                    str(operation_id),
                ),
            )
            db.execute(
                "INSERT INTO phase7_reconciliation VALUES(?,?,?,?,?)",
                (
                    racing_day_id,
                    str(artifact.checksum),
                    len(mismatches),
                    json.dumps(report, sort_keys=True),
                    iso_timestamp(at),
                ),
            )
            if mismatches:
                db.execute(
                    "INSERT INTO phase7_pauses VALUES('training_requests',1,?,?,?) "
                    "ON CONFLICT(scope) DO UPDATE SET paused=1,reason=excluded.reason,"
                    "changed_at=excluded.changed_at,operation_id=excluded.operation_id",
                    (
                        f"incomplete_reconciliation:{racing_day_id}",
                        iso_timestamp(at),
                        str(operation_id),
                    ),
                )
            return report

    def admin_pause(
        self,
        operation_id: OperationId,
        *,
        scope: str,
        paused: bool,
        actor: str,
        reason: str,
        at: datetime,
    ) -> bool:
        require_aware(at, "at")
        if not actor.strip() or not reason.strip():
            raise ValueError("actor and reason are mandatory")
        payload = {
            "scope": scope,
            "paused": paused,
            "actor": actor,
            "reason": reason,
            "at": iso_timestamp(at),
        }
        with self.store._operation(operation_id, "phase7_admin_pause", payload) as (db, replay):
            if replay:
                return False
            prior = db.execute(
                "SELECT paused,reason FROM phase7_pauses WHERE scope=?", (scope,)
            ).fetchone()
            if not paused:
                unresolved = db.execute(
                    "SELECT category FROM phase7_alerts WHERE resolved_at IS NULL"
                ).fetchall()
                if any(self.ALERT_SCOPES[row[0]] == scope for row in unresolved):
                    raise OperationalRejected("scope cannot resume while its alert is unresolved")
            before = {} if prior is None else dict(prior)
            db.execute(
                "INSERT INTO phase7_pauses VALUES(?,?,?,?,?) ON CONFLICT(scope) DO UPDATE SET paused=excluded.paused,reason=excluded.reason,changed_at=excluded.changed_at,operation_id=excluded.operation_id",
                (scope, int(paused), reason, iso_timestamp(at), str(operation_id)),
            )
            db.execute(
                "INSERT INTO phase7_admin_audit(actor,reason,command,scope,before_json,after_json,occurred_at,operation_id) VALUES(?,?,?,?,?,?,?,?)",
                (
                    actor,
                    reason,
                    "pause" if paused else "resume",
                    scope,
                    json.dumps(before),
                    json.dumps({"paused": paused, "reason": reason}),
                    iso_timestamp(at),
                    str(operation_id),
                ),
            )
        return True

    def resolve_alert(
        self,
        operation_id: OperationId,
        *,
        alert_id: str,
        actor: str,
        reason: str,
        at: datetime,
    ) -> bool:
        """Resolve an alert through the shared audited administrative command surface."""
        require_aware(at, "at")
        if not actor.strip() or not reason.strip():
            raise ValueError("actor and reason are mandatory")
        payload = {
            "alert": alert_id,
            "actor": actor,
            "reason": reason,
            "at": iso_timestamp(at),
        }
        with self.store._operation(operation_id, "phase7_resolve_alert", payload) as (db, replay):
            if replay:
                return False
            alert = db.execute(
                "SELECT * FROM phase7_alerts WHERE alert_id=?", (alert_id,)
            ).fetchone()
            if alert is None or alert["resolved_at"] is not None:
                raise OperationalRejected("alert is missing or already resolved")
            scope = self.ALERT_SCOPES[alert["category"]]
            db.execute(
                "INSERT INTO phase7_alert_resolutions VALUES(?,?,?)",
                (alert_id, iso_timestamp(at), str(operation_id)),
            )
            db.execute(
                "UPDATE phase7_alerts SET resolved_at=? WHERE alert_id=?",
                (iso_timestamp(at), alert_id),
            )
            db.execute(
                "INSERT INTO phase7_admin_audit(actor,reason,command,scope,before_json,"
                "after_json,occurred_at,operation_id) VALUES(?,?,?,?,?,?,?,?)",
                (
                    actor,
                    reason,
                    "resolve_alert",
                    scope,
                    json.dumps(dict(alert)),
                    json.dumps({"alert_id": alert_id, "resolved_at": iso_timestamp(at)}),
                    iso_timestamp(at),
                    str(operation_id),
                ),
            )
        return True

    def raise_alert(
        self,
        operation_id: OperationId,
        *,
        alert_id: str,
        category: str,
        details: str,
        racing_day_id: str | None,
        at: datetime,
    ) -> bool:
        require_aware(at, "at")
        payload = {
            "alert": alert_id,
            "category": category,
            "details": details,
            "day": racing_day_id,
            "at": iso_timestamp(at),
        }
        with self.store._operation(operation_id, "phase7_raise_alert", payload) as (db, replay):
            if replay:
                return False
            if category not in self.ALERT_SCOPES or not details.strip():
                raise ValueError("unsupported alert or empty details")
            db.execute(
                "INSERT INTO phase7_alerts VALUES(?,?,?,?,?,NULL,?)",
                (alert_id, category, racing_day_id, details, iso_timestamp(at), str(operation_id)),
            )
            scope = self.ALERT_SCOPES[category]
            db.execute(
                "INSERT INTO phase7_pauses VALUES(?,1,?,?,?) "
                "ON CONFLICT(scope) DO UPDATE SET paused=1,reason=excluded.reason,"
                "changed_at=excluded.changed_at,operation_id=excluded.operation_id",
                (scope, f"alert:{category}", iso_timestamp(at), str(operation_id)),
            )
            # Alerts cannot stop safe collection: only their affected downstream scope.
            if category in {
                "checksum_failure",
                "post_freeze_contamination",
                "result_before_prediction",
                "champion_failure",
            }:
                state = db.execute(
                    "SELECT * FROM phase7_probation_control WHERE singleton=1"
                ).fetchone()
                generation = 1 if state is None else state["generation"]
                db.execute(
                    "INSERT INTO phase7_probation_control VALUES(1,'paused',?,?,?,?) "
                    "ON CONFLICT(singleton) DO UPDATE SET state='paused',reason=excluded.reason,"
                    "changed_at=excluded.changed_at,operation_id=excluded.operation_id",
                    (category, generation, iso_timestamp(at), str(operation_id)),
                )
        return True

    def reset_probation(
        self, operation_id: OperationId, *, actor: str, reason: str, at: datetime
    ) -> int:
        require_aware(at, "at")
        if not actor.strip() or not reason.strip():
            raise ValueError("audited reset requires actor and reason")
        payload = {"actor": actor, "reason": reason, "at": iso_timestamp(at)}
        with self.store._operation(operation_id, "phase7_reset_probation", payload) as (db, replay):
            if replay:
                return db.execute("SELECT generation FROM phase7_probation_control").fetchone()[0]
            state = db.execute(
                "SELECT * FROM phase7_probation_control WHERE singleton=1"
            ).fetchone()
            if db.execute("SELECT 1 FROM phase7_probation_seals LIMIT 1").fetchone() is not None:
                raise OperationalRejected(
                    "sealed probation authority is immutable and cannot be reset"
                )
            generation = 1 if state is None else state["generation"] + 1
            before = {} if state is None else dict(state)
            db.execute(
                "INSERT INTO phase7_probation_control VALUES(1,'reset',?,?,?,?) "
                "ON CONFLICT(singleton) DO UPDATE SET state='reset',reason=excluded.reason,"
                "generation=excluded.generation,changed_at=excluded.changed_at,operation_id=excluded.operation_id",
                (reason, generation, iso_timestamp(at), str(operation_id)),
            )
            db.execute(
                "INSERT INTO phase7_admin_audit(actor,reason,command,scope,before_json,after_json,occurred_at,operation_id) VALUES(?,?,?,?,?,?,?,?)",
                (
                    actor,
                    reason,
                    "reset",
                    "promotion",
                    json.dumps(before),
                    json.dumps({"state": "reset", "generation": generation}),
                    iso_timestamp(at),
                    str(operation_id),
                ),
            )
            return generation

    def cutover_eligibility(
        self, operation_id: OperationId, *, candidate_release_id: str, at: datetime
    ) -> bool:
        require_aware(at, "at")
        resolved: dict[str, str] = {}

        def payload(db: sqlite3.Connection) -> Mapping[str, Any]:
            durable = db.execute(
                "SELECT eligible_at FROM phase7_cutover_eligibility "
                "WHERE operation_id=? AND candidate_release_id=?",
                (str(operation_id), candidate_release_id),
            ).fetchone()
            if durable is None:
                authority_now = self.__clock()
                require_aware(authority_now, "trusted cutover clock")
                eligible_at = iso_timestamp(authority_now)
            else:
                eligible_at = durable["eligible_at"]
            resolved["at"] = eligible_at
            return {"candidate": candidate_release_id, "at": eligible_at}

        with self.store._operation(operation_id, "phase7_cutover_eligibility", payload) as (
            db,
            replay,
        ):
            observation = db.execute(
                "SELECT candidate_release_id,occurred_at FROM "
                "phase7_observation_authority_events WHERE event_id=("
                "SELECT max(event_id) FROM phase7_observation_authority_events)"
            ).fetchone()
            if (
                observation is None
                or observation["candidate_release_id"] != candidate_release_id
                or db.execute(
                    "SELECT action FROM phase7_observation_authority_events "
                    "ORDER BY event_id DESC LIMIT 1"
                ).fetchone()[0]
                != "authorize"
            ):
                raise OperationalRejected("candidate lacks current observation authority")
            rows = db.execute(
                "SELECT e.racing_day_id,d.local_date,e.reconciliation_checksum,"
                "e.restart_checksum,e.ordering_checksum,e.determinism_checksum,"
                "e.recorded_at FROM phase7_day_evidence e "
                "JOIN racing_days d USING(racing_day_id) WHERE e.release_id=? "
                "AND e.complete=1 AND e.critical_failure=0 "
                "ORDER BY d.local_date DESC LIMIT 2",
                (candidate_release_id,),
            ).fetchall()
            predecessor = (
                None
                if len(rows) != 2
                else db.execute(
                    "SELECT predecessor_racing_day_id FROM phase6_racing_day_schedule "
                    "WHERE racing_day_id=?",
                    (rows[0]["racing_day_id"],),
                ).fetchone()
            )
            if len(rows) != 2 or predecessor is None or predecessor[0] != rows[1]["racing_day_id"]:
                raise OperationalRejected("two consecutive complete Racing Days are required")
            eligible_at = resolved["at"]
            if any(row["recorded_at"] > eligible_at for row in rows):
                raise OperationalRejected("cutover eligibility predates its supporting evidence")
            if any(row["recorded_at"] < observation["occurred_at"] for row in rows):
                raise OperationalRejected("cutover evidence predates observation authorization")
            evidence_rows = [
                {key: row[key] for key in row.keys() if key != "recorded_at"}
                for row in reversed(rows)
            ]
            evidence = _checksum(evidence_rows)
            content = _canonical(evidence_rows)
            if replay:
                durable = db.execute(
                    "SELECT * FROM phase7_cutover_eligibility WHERE operation_id=?",
                    (str(operation_id),),
                ).fetchone()
                if (
                    durable is None
                    or durable["candidate_release_id"] != candidate_release_id
                    or durable["first_racing_day_id"] != rows[1]["racing_day_id"]
                    or durable["second_racing_day_id"] != rows[0]["racing_day_id"]
                    or durable["evidence_checksum"] != str(evidence)
                    or durable["eligible_at"] != eligible_at
                ):
                    raise OperationalRejected("cutover eligibility replay evidence changed")
                return False
            self.artifacts.put(content, media_type="application/json", expected_checksum=evidence)
            evidence_manifest = {
                "schema_version": "phase7-operational-evidence-v1",
                "evidence_kind": "cutover",
                "racing_day_id": rows[0]["racing_day_id"],
                "release_id": candidate_release_id,
                "artifact_checksum": str(evidence),
                "checks": {"first": rows[1]["racing_day_id"], "second": rows[0]["racing_day_id"]},
            }
            manifest_artifact = self.artifacts.put(
                _canonical(evidence_manifest),
                media_type="application/vnd.race-operational-evidence+json",
            )
            db.execute(
                "INSERT INTO phase7_operational_evidence VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    str(evidence),
                    "cutover",
                    rows[0]["racing_day_id"],
                    candidate_release_id,
                    str(manifest_artifact.checksum),
                    _canonical(evidence_manifest).decode(),
                    len(content),
                    eligible_at,
                    str(operation_id),
                ),
            )
            db.execute(
                "INSERT INTO phase7_cutover_eligibility VALUES(?,?,?,?,?,?)",
                (
                    candidate_release_id,
                    rows[1]["racing_day_id"],
                    rows[0]["racing_day_id"],
                    str(evidence),
                    eligible_at,
                    str(operation_id),
                ),
            )
        return True

    def authorize_observation(
        self,
        operation_id: OperationId,
        *,
        candidate_release_id: str,
        actor: str,
        reason: str,
        at: datetime,
    ) -> bool:
        require_aware(at, "at")
        _strict_nonempty(actor, "actor")
        _strict_nonempty(reason, "reason")
        resolved: dict[str, str] = {}

        def payload(db: sqlite3.Connection) -> Mapping[str, Any]:
            durable = db.execute(
                "SELECT occurred_at FROM phase7_observation_authority_events "
                "WHERE operation_id=? AND action='authorize'",
                (str(operation_id),),
            ).fetchone()
            if durable is None:
                now = self.__clock()
                require_aware(now, "trusted observation clock")
                occurred_at = iso_timestamp(now)
            else:
                occurred_at = durable["occurred_at"]
            resolved["at"] = occurred_at
            return {
                "candidate": candidate_release_id,
                "actor": actor,
                "reason": reason,
                "at": occurred_at,
            }

        with self.store._operation(operation_id, "phase7_authorize_observation", payload) as (
            db,
            replay,
        ):
            self._verified_release(db, candidate_release_id)
            pointer = db.execute(
                "SELECT release_id,authority,legacy_preserved "
                "FROM phase7_release_pointer WHERE singleton=1"
            ).fetchone()
            if (
                pointer is None
                or pointer["authority"] != "legacy"
                or not pointer["legacy_preserved"]
                or pointer["release_id"] == candidate_release_id
            ):
                raise OperationalRejected(
                    "observation requires a distinct candidate and intact legacy authority"
                )
            if replay:
                event = db.execute(
                    "SELECT candidate_release_id,actor,reason,occurred_at "
                    "FROM phase7_observation_authority_events WHERE operation_id=?",
                    (str(operation_id),),
                ).fetchone()
                if (
                    event is None
                    or event["candidate_release_id"] != candidate_release_id
                    or event["actor"] != actor
                    or event["reason"] != reason
                    or event["occurred_at"] != resolved["at"]
                ):
                    raise OperationalRejected("observation replay authority changed")
                return False
            db.execute(
                "INSERT INTO phase7_observation_authority_events("
                "candidate_release_id,action,actor,reason,occurred_at,operation_id) "
                "VALUES(?,'authorize',?,?,?,?)",
                (
                    candidate_release_id,
                    actor,
                    reason,
                    resolved["at"],
                    str(operation_id),
                ),
            )
        return True

    def revoke_observation(
        self,
        operation_id: OperationId,
        *,
        candidate_release_id: str,
        actor: str,
        reason: str,
        at: datetime,
    ) -> bool:
        require_aware(at, "at")
        _strict_nonempty(actor, "actor")
        _strict_nonempty(reason, "reason")
        resolved: dict[str, str] = {}

        def payload(db: sqlite3.Connection) -> Mapping[str, Any]:
            durable = db.execute(
                "SELECT occurred_at FROM phase7_observation_authority_events "
                "WHERE operation_id=? AND action='revoke'",
                (str(operation_id),),
            ).fetchone()
            if durable is None:
                now = self.__clock()
                require_aware(now, "trusted observation clock")
                occurred_at = iso_timestamp(now)
            else:
                occurred_at = durable["occurred_at"]
            resolved["at"] = occurred_at
            return {
                "candidate": candidate_release_id,
                "actor": actor,
                "reason": reason,
                "at": occurred_at,
            }

        with self.store._operation(operation_id, "phase7_revoke_observation", payload) as (
            db,
            replay,
        ):
            if replay:
                return False
            db.execute(
                "INSERT INTO phase7_observation_authority_events("
                "candidate_release_id,action,actor,reason,occurred_at,operation_id) "
                "VALUES(?,'revoke',?,?,?,?)",
                (
                    candidate_release_id,
                    actor,
                    reason,
                    resolved["at"],
                    str(operation_id),
                ),
            )
        return True

    def initialize_legacy_authority(
        self, operation_id: OperationId, *, release_id: str, actor: str, reason: str, at: datetime
    ) -> bool:
        """Record—not deploy—the intact legacy rollback target exactly once."""
        require_aware(at, "at")
        payload = {"release": release_id, "actor": actor, "reason": reason, "at": iso_timestamp(at)}
        with self.store._operation(operation_id, "phase7_initialize_legacy", payload) as (
            db,
            replay,
        ):
            if replay:
                return False
            if not actor.strip() or not reason.strip():
                raise ValueError("actor and reason are mandatory")
            db.execute(
                "INSERT INTO phase7_release_pointer VALUES(1,?,'legacy',1,NULL,?,?)",
                (release_id, iso_timestamp(at), str(operation_id)),
            )
            db.execute(
                "INSERT INTO phase7_release_history(release_id,authority,prior_release_id,prior_authority,effective_racing_day_id,action,changed_at,operation_id) VALUES(?,'legacy',NULL,NULL,NULL,'initial',?,?)",
                (release_id, iso_timestamp(at), str(operation_id)),
            )
            db.execute(
                "INSERT INTO phase7_admin_audit(actor,reason,command,scope,before_json,"
                "after_json,occurred_at,operation_id) VALUES(?,?,?,?,?,?,?,?)",
                (
                    actor,
                    reason,
                    "initialize_legacy",
                    "cutover",
                    "{}",
                    json.dumps({"release_id": release_id, "authority": "legacy"}),
                    iso_timestamp(at),
                    str(operation_id),
                ),
            )
        return True

    def activate(
        self,
        operation_id: OperationId,
        *,
        release_id: str,
        boundary_day_id: str,
        actor: str,
        reason: str,
        at: datetime,
    ) -> bool:
        require_aware(at, "at")
        if not actor.strip() or not reason.strip():
            raise ValueError("actor and reason are mandatory")
        resolved: dict[str, str] = {}

        def payload(db: sqlite3.Connection) -> Mapping[str, Any]:
            durable = db.execute(
                "SELECT changed_at FROM phase7_release_history "
                "WHERE operation_id=? AND action='activate'",
                (str(operation_id),),
            ).fetchone()
            if durable is None:
                authority_now = self.__clock()
                require_aware(authority_now, "trusted cutover clock")
                changed_at = iso_timestamp(authority_now)
            else:
                changed_at = durable["changed_at"]
            resolved["at"] = changed_at
            return {
                "release": release_id,
                "boundary": boundary_day_id,
                "actor": actor,
                "reason": reason,
                "at": changed_at,
            }

        with self.store._operation(operation_id, "phase7_activate_release", payload) as (
            db,
            replay,
        ):
            changed_at = resolved["at"]
            day = db.execute(
                "SELECT local_date,timezone FROM racing_days WHERE racing_day_id=?",
                (boundary_day_id,),
            ).fetchone()
            eligible = db.execute(
                "SELECT * FROM phase7_cutover_eligibility WHERE candidate_release_id=?",
                (release_id,),
            ).fetchone()
            schedule = db.execute(
                "SELECT predecessor_racing_day_id FROM phase6_racing_day_schedule "
                "WHERE racing_day_id=?",
                (boundary_day_id,),
            ).fetchone()
            prior_schedule = (
                None
                if eligible is None
                else db.execute(
                    "SELECT predecessor_racing_day_id FROM phase6_racing_day_schedule "
                    "WHERE racing_day_id=?",
                    (eligible["second_racing_day_id"],),
                ).fetchone()
            )
            paused = db.execute("SELECT paused FROM phase7_pauses WHERE scope='cutover'").fetchone()
            try:
                changed_at_instant = datetime.fromisoformat(changed_at)
                boundary_instant = (
                    datetime.combine(date.fromisoformat(day["local_date"]), datetime.min.time())
                    .replace(tzinfo=ZoneInfo(day["timezone"]))
                    .astimezone(changed_at_instant.tzinfo)
                    if day is not None
                    else None
                )
            except (ValueError, ZoneInfoNotFoundError):
                boundary_instant = None
            if (
                day is None
                or boundary_instant is None
                or changed_at_instant >= boundary_instant
                or eligible is None
                or eligible["eligible_at"] > changed_at
                or schedule is None
                or schedule[0] != eligible["second_racing_day_id"]
                or prior_schedule is None
                or prior_schedule[0] != eligible["first_racing_day_id"]
                or (paused and paused[0])
            ):
                raise OperationalRejected("prospective boundary cutover is not eligible")
            self._verified_release(db, release_id)
            if replay:
                history = db.execute(
                    "SELECT release_id,effective_racing_day_id,changed_at "
                    "FROM phase7_release_history WHERE operation_id=? AND action='activate'",
                    (str(operation_id),),
                ).fetchone()
                if (
                    history is None
                    or history["release_id"] != release_id
                    or history["effective_racing_day_id"] != boundary_day_id
                    or history["changed_at"] != changed_at
                ):
                    raise OperationalRejected("activation replay release identity changed")
                return False
            prior = db.execute("SELECT * FROM phase7_release_pointer WHERE singleton=1").fetchone()
            if prior is None or prior["authority"] != "legacy":
                raise OperationalRejected("intact legacy authority must be active before cutover")
            observation = db.execute(
                "SELECT candidate_release_id,action FROM "
                "phase7_observation_authority_events ORDER BY event_id DESC LIMIT 1"
            ).fetchone()
            if (
                observation is None
                or observation["candidate_release_id"] != release_id
                or observation["action"] != "authorize"
            ):
                raise OperationalRejected("activation lacks current observation authority")
            db.execute(
                "INSERT INTO phase7_observation_authority_events("
                "candidate_release_id,action,actor,reason,occurred_at,operation_id) "
                "VALUES(?,'revoke',?,?,?,?)",
                (release_id, actor, reason, changed_at, str(operation_id)),
            )
            db.execute(
                "INSERT INTO phase7_release_history(release_id,authority,prior_release_id,prior_authority,effective_racing_day_id,action,changed_at,operation_id,prior_effective_racing_day_id) VALUES(?,?,?,?,?,'activate',?,?,?)",
                (
                    release_id,
                    "race_collection_service",
                    None if prior is None else prior["release_id"],
                    None if prior is None else prior["authority"],
                    boundary_day_id,
                    changed_at,
                    str(operation_id),
                    None if prior is None else prior["effective_racing_day_id"],
                ),
            )
            db.execute(
                "INSERT INTO phase7_release_pointer VALUES(1,?,'race_collection_service',1,?,?,?) ON CONFLICT(singleton) DO UPDATE SET release_id=excluded.release_id,authority=excluded.authority,effective_racing_day_id=excluded.effective_racing_day_id,changed_at=excluded.changed_at,operation_id=excluded.operation_id",
                (release_id, boundary_day_id, changed_at, str(operation_id)),
            )
            db.execute(
                "INSERT INTO phase7_admin_audit(actor,reason,command,scope,before_json,"
                "after_json,occurred_at,operation_id) VALUES(?,?,?,?,?,?,?,?)",
                (
                    actor,
                    reason,
                    "activate",
                    "cutover",
                    json.dumps(dict(prior)),
                    json.dumps(
                        {
                            "release_id": release_id,
                            "authority": "race_collection_service",
                            "effective_racing_day_id": boundary_day_id,
                        }
                    ),
                    changed_at,
                    str(operation_id),
                ),
            )
        return True

    def rollback(self, operation_id: OperationId, *, actor: str, reason: str, at: datetime) -> bool:
        require_aware(at, "at")
        if not actor.strip() or not reason.strip():
            raise ValueError("actor and reason are mandatory")
        payload = {"actor": actor, "reason": reason, "at": iso_timestamp(at)}
        with self.store._operation(operation_id, "phase7_rollback_release", payload) as (
            db,
            replay,
        ):
            if replay:
                return False
            active = db.execute("SELECT * FROM phase7_release_pointer WHERE singleton=1").fetchone()
            history = db.execute(
                "SELECT * FROM phase7_release_history WHERE action='activate' AND release_id=? ORDER BY history_id DESC LIMIT 1",
                (active["release_id"] if active else "",),
            ).fetchone()
            if active is None or history is None or history["prior_release_id"] is None:
                raise OperationalRejected("no exact rollback target")
            db.execute(
                "INSERT INTO phase7_release_history(release_id,authority,prior_release_id,prior_authority,effective_racing_day_id,action,changed_at,operation_id,prior_effective_racing_day_id) VALUES(?,?,?,?,?,'rollback',?,?,?)",
                (
                    history["prior_release_id"],
                    history["prior_authority"],
                    active["release_id"],
                    active["authority"],
                    history["prior_effective_racing_day_id"],
                    iso_timestamp(at),
                    str(operation_id),
                    active["effective_racing_day_id"],
                ),
            )
            db.execute(
                "INSERT INTO phase7_admin_audit(actor,reason,command,scope,before_json,"
                "after_json,occurred_at,operation_id) VALUES(?,?,?,?,?,?,?,?)",
                (
                    actor,
                    reason,
                    "rollback",
                    "cutover",
                    json.dumps(dict(active)),
                    json.dumps(
                        {
                            "release_id": history["prior_release_id"],
                            "authority": history["prior_authority"],
                            "effective_racing_day_id": history["prior_effective_racing_day_id"],
                        }
                    ),
                    iso_timestamp(at),
                    str(operation_id),
                ),
            )
            db.execute(
                "UPDATE phase7_release_pointer SET release_id=?,authority=?,effective_racing_day_id=?,changed_at=?,operation_id=? WHERE singleton=1",
                (
                    history["prior_release_id"],
                    history["prior_authority"],
                    history["prior_effective_racing_day_id"],
                    iso_timestamp(at),
                    str(operation_id),
                ),
            )
        return True

    def record_probation_day(
        self, operation_id: OperationId, *, racing_day_id: str, at: datetime
    ) -> int:
        require_aware(at, "at")
        with self.store._connect() as read:
            prior_operation = read.execute(
                "SELECT kind FROM operations WHERE operation_id=?", (str(operation_id),)
            ).fetchone()
            prior_receipt = read.execute(
                "SELECT generation,local_date,racing_day_id "
                "FROM phase7_probation_acceptances WHERE operation_id=?",
                (str(operation_id),),
            ).fetchone()
            duplicate = read.execute(
                "SELECT generation FROM phase7_probation_acceptances WHERE racing_day_id=?",
                (racing_day_id,),
            ).fetchone()
            if prior_operation is not None:
                if (
                    prior_operation["kind"] != "phase7_accept_probation_day"
                    or prior_receipt is None
                    or prior_receipt["racing_day_id"] != racing_day_id
                ):
                    raise ConflictingOperation("probation operation replay has different intent")
                return read.execute(
                    "SELECT count(*) FROM phase7_probation_acceptances WHERE generation=? "
                    "AND local_date<=?",
                    (prior_receipt["generation"], prior_receipt["local_date"]),
                ).fetchone()[0]
            if duplicate is not None:
                raise OperationalRejected("Racing Day is already accepted and cannot count twice")
            state = read.execute(
                "SELECT * FROM phase7_probation_control WHERE singleton=1"
            ).fetchone()
            generation = 1 if state is None else state["generation"]
            evidence = read.execute(
                "SELECT complete,critical_failure FROM phase7_day_evidence "
                "WHERE racing_day_id=?",
                (racing_day_id,),
            ).fetchone()
            day_exists = read.execute(
                "SELECT 1 FROM racing_days WHERE racing_day_id=?", (racing_day_id,)
            ).fetchone()
            pointer = read.execute(
                "SELECT effective_racing_day_id FROM phase7_release_pointer "
                "WHERE singleton=1 AND authority='race_collection_service'"
            ).fetchone()
            last = read.execute(
                "SELECT racing_day_id FROM phase7_probation_acceptances WHERE generation=? "
                "ORDER BY local_date DESC LIMIT 1",
                (generation,),
            ).fetchone()
            schedule = read.execute(
                "SELECT predecessor_racing_day_id,programme_checksum "
                "FROM phase6_racing_day_schedule "
                "WHERE racing_day_id=?",
                (racing_day_id,),
            ).fetchone()
            programme = read.execute(
                "SELECT count(*) race_count,count(DISTINCT e.programme_checksum) checksum_count,"
                "min(e.programme_checksum) checksum FROM expected_races e "
                "JOIN races r USING(race_id) WHERE r.racing_day_id=?",
                (racing_day_id,),
            ).fetchone()
            critical_alert = read.execute(
                "SELECT category FROM phase7_alerts WHERE resolved_at IS NULL "
                "AND category IN ('checksum_failure','post_freeze_contamination',"
                "'result_before_prediction','champion_failure') "
                "ORDER BY raised_at LIMIT 1"
            ).fetchone()
        expected_predecessor = None if pointer is None else pointer[0] if last is None else last[0]
        failure = None
        if day_exists is not None and (evidence is None or schedule is None):
            failure = "incomplete_day_evidence"
        elif evidence is not None and (not evidence["complete"] or evidence["critical_failure"]):
            failure = "critical_or_incomplete_day"
        elif critical_alert is not None:
            failure = f"unresolved_critical_alert:{critical_alert['category']}"
        elif (
            evidence is not None
            and expected_predecessor is not None
            and (schedule is None or schedule[0] != expected_predecessor)
        ):
            failure = "probation_schedule_gap"
        elif evidence is not None and (
            programme["race_count"] == 0
            or programme["checksum_count"] != 1
            or schedule is None
            or programme["checksum"] != schedule["programme_checksum"]
        ):
            failure = "probation_programme_mismatch"
        if failure is not None:
            payload = {"day": racing_day_id, "failure": failure, "at": iso_timestamp(at)}
            with self.store._operation(operation_id, "phase7_pause_probation_failure", payload) as (
                db,
                replay,
            ):
                if not replay:
                    db.execute(
                        "INSERT INTO phase7_probation_control VALUES(1,'paused',?,?,?,?) "
                        "ON CONFLICT(singleton) DO UPDATE SET state='paused',"
                        "reason=excluded.reason,changed_at=excluded.changed_at,"
                        "operation_id=excluded.operation_id",
                        (failure, generation, iso_timestamp(at), str(operation_id)),
                    )
            raise OperationalRejected("probation failure durably paused the generation")
        payload = {"day": racing_day_id, "at": iso_timestamp(at)}
        with self.store._operation(operation_id, "phase7_accept_probation_day", payload) as (
            db,
            replay,
        ):
            if replay:
                receipt = db.execute(
                    "SELECT generation,local_date FROM phase7_probation_acceptances "
                    "WHERE operation_id=?",
                    (str(operation_id),),
                ).fetchone()
                if receipt is None:
                    raise OperationalRejected("probation replay lacks its exact acceptance")
                return db.execute(
                    "SELECT count(*) FROM phase7_probation_acceptances WHERE generation=? "
                    "AND local_date<=?",
                    (receipt["generation"], receipt["local_date"]),
                ).fetchone()[0]
            state = db.execute(
                "SELECT * FROM phase7_probation_control WHERE singleton=1"
            ).fetchone()
            generation = 1 if state is None else state["generation"]
            if state is not None and state["state"] == "paused":
                raise OperationalRejected("probation is paused after critical failure")
            if state is None:
                db.execute(
                    "INSERT INTO phase7_probation_control VALUES(1,'running',?,?,?,?)",
                    (
                        "probation generation initialized",
                        generation,
                        iso_timestamp(at),
                        str(operation_id),
                    ),
                )
            row = db.execute(
                "SELECT e.*,d.local_date,d.closed_at,rn.reconciled_at,"
                "s.programme_checksum programme "
                "FROM phase7_day_evidence e JOIN racing_days d USING(racing_day_id) "
                "JOIN phase7_reconciliation rn USING(racing_day_id) "
                "JOIN phase6_racing_day_schedule s USING(racing_day_id) "
                "WHERE e.racing_day_id=? AND EXISTS ("
                "SELECT 1 FROM expected_races x JOIN races r USING(race_id) "
                "WHERE r.racing_day_id=e.racing_day_id "
                "GROUP BY r.racing_day_id HAVING count(DISTINCT x.programme_checksum)=1 "
                "AND min(x.programme_checksum)=s.programme_checksum)",
                (racing_day_id,),
            ).fetchone()
            if row is None or not row["complete"] or row["critical_failure"]:
                raise OperationalRejected("day is not successful probation evidence")
            accepted = iso_timestamp(at)
            if (
                date.fromisoformat(row["local_date"]) > at.date()
                or row["closed_at"] is None
                or accepted < row["closed_at"]
                or accepted < row["recorded_at"]
                or accepted < row["reconciled_at"]
            ):
                raise OperationalRejected("probation acceptance predates durable day evidence")
            pointer = db.execute(
                "SELECT p.*,d.local_date effective_date FROM phase7_release_pointer p JOIN racing_days d ON d.racing_day_id=p.effective_racing_day_id WHERE p.singleton=1 AND p.authority='race_collection_service'"
            ).fetchone()
            if (
                pointer is None
                or pointer["release_id"] != row["release_id"]
                or date.fromisoformat(row["local_date"])
                <= date.fromisoformat(pointer["effective_date"])
            ):
                raise OperationalRejected("probation must follow active candidate cutover")
            self._verified_release(db, row["release_id"])
            for kind, column in (
                ("restart", "restart_checksum"),
                ("ordering", "ordering_checksum"),
                ("determinism", "determinism_checksum"),
            ):
                self._verify_operational_evidence(
                    db,
                    racing_day_id=racing_day_id,
                    release_id=row["release_id"],
                    evidence_kind=kind,
                    checksum=ArtifactChecksum(row[column]),
                )
            unresolved = db.execute(
                "SELECT 1 FROM phase7_alerts WHERE resolved_at IS NULL AND category IN ('checksum_failure','post_freeze_contamination','result_before_prediction','champion_failure') LIMIT 1"
            ).fetchone()
            if unresolved is not None:
                raise OperationalRejected("unresolved critical alert pauses probation")
            last = db.execute(
                "SELECT racing_day_id,local_date FROM phase7_probation_acceptances "
                "WHERE generation=? ORDER BY local_date DESC LIMIT 1",
                (generation,),
            ).fetchone()
            schedule = db.execute(
                "SELECT predecessor_racing_day_id,programme_checksum "
                "FROM phase6_racing_day_schedule "
                "WHERE racing_day_id=?",
                (racing_day_id,),
            ).fetchone()
            expected_predecessor = pointer["effective_racing_day_id"] if last is None else last[0]
            if (
                schedule is None
                or schedule["predecessor_racing_day_id"] != expected_predecessor
                or schedule["programme_checksum"] != row["programme"]
            ):
                raise OperationalRejected("probation must follow the authentic schedule chain")
            current_count = db.execute(
                "SELECT count(*) FROM phase7_probation_acceptances WHERE generation=?",
                (generation,),
            ).fetchone()[0]
            if current_count >= 14:
                raise OperationalRejected("probation cannot accept a fifteenth day")
            db.execute(
                "INSERT INTO phase7_probation_acceptances VALUES(?,?,?,?,?,?)",
                (
                    generation,
                    racing_day_id,
                    row["local_date"],
                    row["programme"],
                    iso_timestamp(at),
                    str(operation_id),
                ),
            )
            count = db.execute(
                "SELECT count(*) FROM phase7_probation_acceptances WHERE generation=?",
                (generation,),
            ).fetchone()[0]
            if count > 14:
                raise OperationalRejected("probation cannot accept a fifteenth day")
            status = "complete" if count == 14 else "running"
            db.execute(
                "INSERT INTO phase7_probation_control VALUES(1,?,?,?, ?,?) ON CONFLICT(singleton) DO UPDATE SET state=excluded.state,reason=excluded.reason,changed_at=excluded.changed_at,operation_id=excluded.operation_id",
                (
                    status,
                    "durable successful Racing Day",
                    generation,
                    iso_timestamp(at),
                    str(operation_id),
                ),
            )
            return count

    def seal_probation(
        self, operation_id: OperationId, *, probation_id: str, at: datetime
    ) -> ArtifactChecksum:
        require_aware(at, "at")
        payload = {"probation": probation_id, "at": iso_timestamp(at)}
        with self.store._operation(operation_id, "phase7_seal_probation", payload) as (db, replay):
            if replay:
                return ArtifactChecksum(
                    db.execute(
                        "SELECT state_checksum FROM phase6_probation_states WHERE operation_id=?",
                        (str(operation_id),),
                    ).fetchone()[0]
                )
            state = db.execute(
                "SELECT * FROM phase7_probation_control WHERE singleton=1 AND state='complete'"
            ).fetchone()
            if (
                state is not None
                and db.execute(
                    "SELECT 1 FROM phase7_probation_seals WHERE generation=?",
                    (state["generation"],),
                ).fetchone()
            ):
                raise OperationalRejected("probation generation is already sealed")
            rows = (
                []
                if state is None
                else db.execute(
                    "SELECT a.*,e.* FROM phase7_probation_acceptances a "
                    "JOIN phase7_day_evidence e USING(racing_day_id) "
                    "WHERE a.generation=? ORDER BY a.local_date",
                    (state["generation"],),
                ).fetchall()
            )
            if len(rows) != 14:
                raise OperationalRejected("exactly fourteen durable days are required")
            pointer = db.execute(
                "SELECT * FROM phase7_release_pointer WHERE singleton=1 "
                "AND authority='race_collection_service'"
            ).fetchone()
            unresolved = db.execute(
                "SELECT 1 FROM phase7_alerts WHERE resolved_at IS NULL AND category IN "
                "('checksum_failure','post_freeze_contamination','result_before_prediction',"
                "'champion_failure') LIMIT 1"
            ).fetchone()
            if (
                pointer is None
                or unresolved is not None
                or any(row["release_id"] != pointer["release_id"] for row in rows)
            ):
                raise OperationalRejected("probation release or critical-alert state changed")
            self._verified_release(db, pointer["release_id"])
            expected_predecessor = pointer["effective_racing_day_id"]
            for row in rows:
                schedule = db.execute(
                    "SELECT predecessor_racing_day_id FROM phase6_racing_day_schedule "
                    "WHERE racing_day_id=?",
                    (row["racing_day_id"],),
                ).fetchone()
                if schedule is None or schedule[0] != expected_predecessor:
                    raise OperationalRejected("probation schedule chain changed")
                day_times = db.execute(
                    "SELECT d.closed_at,r.reconciled_at FROM racing_days d "
                    "JOIN phase7_reconciliation r USING(racing_day_id) "
                    "WHERE d.racing_day_id=?",
                    (row["racing_day_id"],),
                ).fetchone()
                if (
                    day_times is None
                    or row["accepted_at"] < row["recorded_at"]
                    or row["accepted_at"] < day_times["closed_at"]
                    or row["accepted_at"] < day_times["reconciled_at"]
                ):
                    raise OperationalRejected("probation acceptance time is invalid")
                self.artifacts.verify(ArtifactChecksum(row["reconciliation_checksum"]))
                reconciliation = db.execute(
                    "SELECT report_checksum FROM phase7_reconciliation " "WHERE racing_day_id=?",
                    (row["racing_day_id"],),
                ).fetchone()
                if reconciliation is None or reconciliation[0] != row["reconciliation_checksum"]:
                    raise OperationalRejected("probation reconciliation identity changed")
                for kind, column in (
                    ("restart", "restart_checksum"),
                    ("ordering", "ordering_checksum"),
                    ("determinism", "determinism_checksum"),
                ):
                    self._verify_operational_evidence(
                        db,
                        racing_day_id=row["racing_day_id"],
                        release_id=row["release_id"],
                        evidence_kind=kind,
                        checksum=ArtifactChecksum(row[column]),
                    )
                expected_predecessor = row["racing_day_id"]
            days = [
                {
                    "racing_day": r["local_date"],
                    "reconciliation_checksum": r["reconciliation_checksum"],
                    "restart_checksum": r["restart_checksum"],
                    "ordering_checksum": r["ordering_checksum"],
                    "determinism_checksum": r["determinism_checksum"],
                }
                for r in rows
            ]
            manifest = {
                "schema_version": "phase7-probation-v1",
                "probation_id": probation_id,
                "through_racing_day": rows[-1]["local_date"],
                "days": days,
            }
            artifact = self.artifacts.put(
                _canonical(manifest), media_type="application/vnd.race-probation+json"
            )
            cutover = db.execute(
                "SELECT operation_id FROM phase7_release_history WHERE action='activate' "
                "AND release_id=? AND effective_racing_day_id=? ORDER BY history_id DESC LIMIT 1",
                (pointer["release_id"], pointer["effective_racing_day_id"]),
            ).fetchone()
            if cutover is None:
                raise OperationalRejected("probation has no exact active cutover generation")
            db.execute(
                "INSERT INTO phase7_phase6_probation_authority VALUES(?,?,?,?,?,?)",
                (
                    probation_id,
                    state["generation"],
                    pointer["release_id"],
                    cutover[0],
                    str(operation_id),
                    str(artifact.checksum),
                ),
            )
            db.execute(
                "INSERT INTO phase6_probation_states VALUES(?,?,?,?,?)",
                (
                    probation_id,
                    rows[-1]["local_date"],
                    str(artifact.checksum),
                    iso_timestamp(at),
                    str(operation_id),
                ),
            )
            db.executemany(
                "INSERT INTO phase6_probation_days VALUES(?,?,?,?,?,?,1)",
                [
                    (
                        probation_id,
                        d["racing_day"],
                        d["reconciliation_checksum"],
                        d["restart_checksum"],
                        d["ordering_checksum"],
                        d["determinism_checksum"],
                    )
                    for d in days
                ],
            )
            db.executemany(
                "INSERT INTO phase6_probation_day_auth VALUES(?,?,?)",
                [
                    (
                        probation_id,
                        r["racing_day_id"],
                        db.execute(
                            "SELECT programme_checksum FROM phase7_probation_acceptances WHERE racing_day_id=?",
                            (r["racing_day_id"],),
                        ).fetchone()[0],
                    )
                    for r in rows
                ],
            )
            db.execute(
                "INSERT INTO phase7_probation_seals VALUES(?,?,?,?,?,?,?)",
                (
                    state["generation"],
                    probation_id,
                    pointer["release_id"],
                    cutover[0],
                    str(artifact.checksum),
                    iso_timestamp(at),
                    str(operation_id),
                ),
            )
            return artifact.checksum

    def record_legacy_retirement_eligibility(
        self,
        operation_id: OperationId,
        *,
        eligibility_id: str,
        probation_id: str,
        actor: str,
        reason: str,
        at: datetime,
    ) -> Mapping[str, str]:
        """Record a non-destructive gate; this command cannot retire the legacy service."""
        require_aware(at, "at")
        for value, field in (
            (eligibility_id, "eligibility identity"),
            (probation_id, "probation identity"),
            (actor, "actor"),
            (reason, "reason"),
        ):
            _strict_nonempty(value, field)
        payload = {
            "eligibility_id": eligibility_id,
            "probation_id": probation_id,
            "actor": actor,
            "reason": reason,
            "at": iso_timestamp(at),
        }
        with self.store._operation(
            operation_id, "phase7_record_legacy_retirement_eligibility", payload
        ) as (db, replay):
            if replay:
                row = db.execute(
                    "SELECT * FROM phase7_legacy_retirement_eligibility " "WHERE operation_id=?",
                    (str(operation_id),),
                ).fetchone()
                if row is None:
                    raise OperationalRejected("retirement eligibility replay is missing")
                return dict(row)
            seal = db.execute(
                "SELECT * FROM phase7_probation_seals WHERE probation_id=?",
                (probation_id,),
            ).fetchone()
            pointer = db.execute(
                "SELECT * FROM phase7_release_pointer WHERE singleton=1 "
                "AND authority='race_collection_service' AND legacy_preserved=1"
            ).fetchone()
            activation = (
                None
                if seal is None
                else db.execute(
                    "SELECT * FROM phase7_release_history WHERE operation_id=? "
                    "AND action='activate'",
                    (seal["cutover_operation_id"],),
                ).fetchone()
            )
            if (
                seal is None
                or pointer is None
                or activation is None
                or pointer["release_id"] != seal["release_id"]
                or activation["prior_authority"] != "legacy"
                or not activation["prior_release_id"]
            ):
                raise OperationalRejected(
                    "exact sealed probation and preserved rollback target are required"
                )
            row = {
                "eligibility_id": eligibility_id,
                "probation_id": probation_id,
                "probation_generation": seal["generation"],
                "candidate_release_id": seal["release_id"],
                "legacy_release_id": activation["prior_release_id"],
                "activation_operation_id": seal["cutover_operation_id"],
                "probation_seal_operation_id": seal["operation_id"],
                "probation_state_checksum": seal["state_checksum"],
                "recorded_at": iso_timestamp(at),
                "operation_id": str(operation_id),
            }
            db.execute(
                "INSERT INTO phase7_legacy_retirement_eligibility VALUES" "(?,?,?,?,?,?,?,?,?,?)",
                tuple(row.values()),
            )
            db.execute(
                "INSERT INTO phase7_admin_audit(actor,reason,command,scope,before_json,"
                "after_json,occurred_at,operation_id) VALUES(?,?,?,?,?,?,?,?)",
                (
                    actor,
                    reason,
                    "record_legacy_retirement_eligibility",
                    "retirement",
                    "{}",
                    json.dumps(row, sort_keys=True),
                    iso_timestamp(at),
                    str(operation_id),
                ),
            )
            return row


class RaceCollectionService:
    """The only unattended dispatcher; adapters invoke these same commands."""

    ORDER = (
        "discover_programme",
        "collect_cards_and_form",
        "collect_adaptive_odds",
        "close_and_seal",
        "deferred_prediction",
        "collect_results",
        "join_training_examples",
        "reconcile",
        "request_training",
    )

    def __init__(self, authority: OperationalAuthority, *, token: str, generation: int):
        self.authority, self.token, self.generation = authority, token, generation

    def dispatch(self, command: ApplicationCommand, *, now: datetime) -> Any:
        """Fence every application command with current durable ownership."""
        phase = command.phase
        self.authority.assert_lease(self.token, self.generation, now)
        with self.authority.store._connect() as db:
            scope = {
                "collect_results": "results",
                "join_training_examples": "joins",
                "request_training": "training_requests",
            }.get(phase)
            paused = (
                None
                if scope is None
                else db.execute(
                    "SELECT paused FROM phase7_pauses WHERE scope=?", (scope,)
                ).fetchone()
            )
        if paused is not None and paused[0]:
            raise OperationalRejected(f"{scope} is administratively paused")
        # Collection is deliberately callable while downstream scopes are paused.
        return self.authority.execute_application_command(
            command,
            token=self.token,
            generation=self.generation,
            at=now,
        )

    def advance(
        self,
        operation_id: OperationId,
        *,
        racing_day_id: str,
        phase: str,
        now: datetime,
        command: ApplicationCommand,
    ) -> Any:
        """Run one idempotent application command and durably cross its ordered barrier."""
        ordinal = self.ORDER.index(phase) + 1
        trusted_now = self.authority.trusted_time()
        self.authority.assert_lease(self.token, self.generation, trusted_now)
        payload = {
            "day": racing_day_id,
            "phase": phase,
            "ordinal": ordinal,
            "lease_generation": self.generation,
            "command_operation_id": str(command.operation_id),
            "at": iso_timestamp(now),
        }
        # Exact advancement replay returns its persisted result without touching
        # the application command. Crash recovery reuses command.operation_id,
        # whose domain API is independently idempotent.
        with self.authority.store._connect() as read:
            prior = read.execute(
                "SELECT * FROM phase7_scheduler_progress WHERE operation_id=?",
                (str(operation_id),),
            ).fetchone()
            if prior is not None:
                receipt_identity = read.execute(
                    "SELECT command_payload_sha256 "
                    "FROM phase7_application_command_receipts "
                    "WHERE command_operation_id=?",
                    (str(command.operation_id),),
                ).fetchone()
                lease_history = read.execute(
                    "SELECT lease_token FROM phase7_scheduler_history WHERE generation=?",
                    (prior["lease_generation"],),
                ).fetchone()
                adoption = read.execute(
                    "SELECT 1 FROM phase7_day_plan_adoptions WHERE racing_day_id=? "
                    "AND lease_generation=? AND lease_token=?",
                    (racing_day_id, self.generation, self.token),
                ).fetchone()
                if (
                    prior["racing_day_id"] != racing_day_id
                    or prior["phase_ordinal"] != ordinal
                    or prior["phase_name"] != phase
                    or prior["command_operation_id"] != str(command.operation_id)
                    or (prior["lease_generation"] != self.generation and adoption is None)
                    or receipt_identity is None
                    or receipt_identity["command_payload_sha256"] != command.payload_sha256()
                    or lease_history is None
                    or (
                        prior["lease_generation"] == self.generation
                        and lease_history["lease_token"] != self.token
                    )
                ):
                    raise ConflictingOperation(
                        f"scheduler advancement {operation_id} has different intent"
                    )
                if prior["lease_generation"] != self.generation:
                    return json.loads(prior["result_json"])
                with self.authority.store._operation(
                    operation_id, "phase7_advance_phase", payload
                ) as (db, replay):
                    if not replay:
                        raise OperationalRejected(
                            "scheduler progress exists without its operation authority"
                        )
                    row = db.execute(
                        "SELECT result_json FROM phase7_scheduler_progress WHERE operation_id=?",
                        (str(operation_id),),
                    ).fetchone()
                    if row is None:
                        raise OperationalRejected("phase replay has no durable outcome")
                    return json.loads(row[0])
            if command.phase != phase or command.racing_day_id != racing_day_id:
                raise ConflictingOperation("application command disagrees with advancement intent")
            planned = read.execute(
                "SELECT p.command_operation_id,"
                "(p.lease_generation=? OR EXISTS ("
                " SELECT 1 FROM phase7_day_plan_adoptions adoption"
                " WHERE adoption.racing_day_id=p.racing_day_id"
                " AND adoption.lease_generation=?"
                " AND adoption.lease_token=?)) AS lease_authorized "
                "FROM phase7_day_command_plan p "
                "WHERE p.racing_day_id=? AND p.phase_ordinal=? AND p.phase_name=?",
                (
                    self.generation,
                    self.generation,
                    self.token,
                    racing_day_id,
                    ordinal,
                    phase,
                ),
            ).fetchone()
            if (
                planned is None
                or planned["command_operation_id"] != str(command.operation_id)
                or not planned["lease_authorized"]
            ):
                raise OperationalRejected("advancement is absent from the exact Racing Day plan")
            expected = read.execute(
                "SELECT count(*)+1 FROM phase7_scheduler_progress WHERE racing_day_id=?",
                (racing_day_id,),
            ).fetchone()[0]
            receipt = read.execute(
                "SELECT result_json,result_checksum,committed_at,command_payload_sha256 "
                "FROM phase7_application_command_receipts "
                "WHERE command_operation_id=? AND racing_day_id=? AND phase_name=?",
                (str(command.operation_id), racing_day_id, phase),
            ).fetchone()
        if ordinal != expected:
            raise OperationalRejected("phase advancement is out of order")
        self.authority.assert_lease(self.token, self.generation, now)
        if receipt is None:
            self.dispatch(command, now=now)
            with self.authority.store._connect() as read:
                receipt = read.execute(
                    "SELECT result_json,result_checksum,committed_at,command_payload_sha256 "
                    "FROM phase7_application_command_receipts "
                    "WHERE command_operation_id=? AND racing_day_id=? AND phase_name=?",
                    (str(command.operation_id), racing_day_id, phase),
                ).fetchone()
        if receipt is None:
            raise OperationalRejected("application command did not commit an exact result receipt")
        if receipt["command_payload_sha256"] != command.payload_sha256():
            raise ConflictingOperation("application command receipt has different intent")
        result_json = receipt["result_json"]
        result_checksum = receipt["result_checksum"]
        result = json.loads(result_json)
        with self.authority.store._operation(operation_id, "phase7_advance_phase", payload) as (
            db,
            replay,
        ):
            if replay:
                row = db.execute(
                    "SELECT result_json FROM phase7_scheduler_progress WHERE operation_id=?",
                    (str(operation_id),),
                ).fetchone()
                if row is None:
                    raise OperationalRejected("phase replay has no durable outcome")
                return json.loads(row[0])
            authority_now = self.authority.trusted_time()
            lease = db.execute(
                "SELECT 1 FROM phase7_scheduler_lease WHERE singleton=1 AND lease_token=? "
                "AND generation=? AND acquired_at<=? AND expires_at>?",
                (
                    self.token,
                    self.generation,
                    iso_timestamp(authority_now),
                    iso_timestamp(authority_now),
                ),
            ).fetchone()
            if lease is None:
                raise OperationalRejected("lease changed before phase barrier commit")
            db.execute(
                "INSERT INTO phase7_scheduler_progress VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    racing_day_id,
                    ordinal,
                    phase,
                    self.generation,
                    str(command.operation_id),
                    result_json,
                    result_checksum,
                    receipt["committed_at"],
                    str(operation_id),
                ),
            )
        return result
