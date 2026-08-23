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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Protocol, Sequence

from .artifacts import ArtifactStoreError, LocalArtifactStore
from .domain import (
    ArtifactChecksum,
    OperationId,
    ProgrammeRaceCandidate,
    RaceId,
    RacingDay,
    RacingDayId,
)
from .forecasting import ForecastingAuthority
from .forward_sealed_corpus import ForwardCorpusRejected, ForwardSealedCorpus, canonical_json
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
from .operations import BarrierNotSatisfied, OperationsStoreError, SQLiteOperationsStore


class ServiceUnavailable(RuntimeError):
    """The release cannot form one complete, trusted runtime composition."""


@dataclass(frozen=True, slots=True)
class ForwardBaselineCaptureConfiguration:
    """Fixed production policy for one bounded forward-baseline cohort."""

    cohort_id: str
    corpus_root: Path
    current_index_max_age: timedelta
    feature_cutoff: timedelta
    timezone: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.cohort_id, str)
            or not self.cohort_id.strip()
            or not isinstance(self.corpus_root, Path)
            or self.current_index_max_age <= timedelta(0)
            or self.feature_cutoff <= timedelta(0)
            or not isinstance(self.timezone, str)
            or not self.timezone.strip()
        ):
            raise ValueError("forward baseline capture configuration is invalid")


@dataclass(frozen=True, slots=True)
class _ForwardCohortCandidate:
    race_id: str
    racing_date: str
    venue: str
    race_number: int
    jump: datetime
    source_native_race_id: str
    source_native_runner_ids: tuple[str, ...]


class ForwardBaselineCaptureService:
    """Preflight and schedule one immutable 20-race forward-baseline cohort."""

    REPORT_SCHEMA = "forward-baseline-capture-service-report-v1"

    def __init__(
        self,
        store: SQLiteOperationsStore,
        configuration: ForwardBaselineCaptureConfiguration,
    ) -> None:
        self.store = store
        self.configuration = configuration
        self.forecasting = ForecastingAuthority(store)

    def open_results(
        self,
        operation_id: OperationId,
        race_id: RaceId,
        at: datetime,
        *,
        cohort_bytes: bytes,
    ) -> bool:
        """Delegate result access through the frozen cohort's 20-terminal barrier."""
        return self.forecasting.open_baseline_results(
            operation_id,
            race_id,
            at,
            cohort_bytes=cohort_bytes,
        )

    def capture_scheduled(
        self,
        *,
        protocol: Any,
        evidence_root: Path,
        collector_run_id: str,
        plan_item: Mapping[str, Any],
        verified_index: Any,
        emitted_at: datetime,
        attempt: Mapping[str, Any] | None = None,
        receipt_publish: Mapping[str, Any] | None = None,
        scheduled_admitter: Callable[..., Mapping[str, Any]] | None = None,
    ) -> Mapping[str, Any]:
        """Own scheduled corpus admission after durable cohort authority exists."""
        from .synchronous_manual_capture import VerifiedCurrentRaceIndex

        if (
            not isinstance(verified_index, VerifiedCurrentRaceIndex)
            or verified_index.schema_version != "collector_current_race_index_v2"
            or type(verified_index.packet_bytes) is not bytes
            or hashlib.sha256(verified_index.packet_bytes).hexdigest()
            != verified_index.packet_sha256
            or emitted_at.tzinfo is None
            or emitted_at.utcoffset() is None
        ):
            raise ServiceUnavailable("verified collector current-race index is required")
        try:
            source_generated_at = datetime.fromisoformat(
                verified_index.source_generated_at
            )
        except (TypeError, ValueError) as error:
            raise ServiceUnavailable("verified current-index timestamp is invalid") from error
        if (
            source_generated_at.tzinfo is None
            or source_generated_at.utcoffset() is None
            or not timedelta(0)
            <= emitted_at.astimezone(timezone.utc)
            - source_generated_at.astimezone(timezone.utc)
            <= self.configuration.current_index_max_age
        ):
            raise ServiceUnavailable("verified current-race index is not READY_NOW")
        durable = self.store.forward_baseline_cohort(self.configuration.cohort_id)
        if durable is None:
            raise ServiceUnavailable("forward baseline cohort authority is unavailable")
        corpus = ForwardSealedCorpus(self.configuration.corpus_root)
        try:
            cohort_bytes = corpus.load_prediction_cohort(self.configuration.cohort_id)
            cohort = json.loads(cohort_bytes) if cohort_bytes is not None else None
            cohort_checksum = LocalArtifactStore.checksum(cohort_bytes or b"")
        except (ArtifactStoreError, ForwardCorpusRejected, json.JSONDecodeError) as error:
            raise ServiceUnavailable("forward baseline cohort authority is invalid") from error
        if (
            cohort is None
            or str(cohort_checksum) != durable["artifact_checksum"]
            or cohort.get("frozen_at") != durable["frozen_at"]
            or cohort.get("race_count") != durable["race_count"]
            or cohort.get("members") != durable["members"]
        ):
            raise ServiceUnavailable("forward baseline cohort authority disagrees")
        if scheduled_admitter is None:
            from .scheduled_forward_corpus import admit_scheduled_capture

            scheduled_admitter = admit_scheduled_capture
        return scheduled_admitter(
            protocol=protocol,
            evidence_root=evidence_root,
            corpus_root=self.configuration.corpus_root,
            collector_run_id=collector_run_id,
            plan_item=plan_item,
            verified_index=verified_index,
            emitted_at=emitted_at,
            cohort_id=self.configuration.cohort_id,
            cohort_checksum=cohort_checksum,
            attempt=attempt,
            receipt_publish=receipt_publish,
        )

    def run(self, verified_index: Any, *, now: datetime) -> Mapping[str, Any]:
        """Return readiness before any corpus or lifecycle mutation."""
        from .synchronous_manual_capture import VerifiedCurrentRaceIndex

        if (
            not isinstance(verified_index, VerifiedCurrentRaceIndex)
            or verified_index.schema_version != "collector_current_race_index_v2"
            or type(verified_index.packet_bytes) is not bytes
            or hashlib.sha256(verified_index.packet_bytes).hexdigest()
            != verified_index.packet_sha256
            or now.tzinfo is None
            or now.utcoffset() is None
        ):
            raise ServiceUnavailable("verified collector current-race index is required")
        try:
            frozen_at = datetime.fromisoformat(verified_index.source_generated_at)
        except (TypeError, ValueError) as error:
            raise ServiceUnavailable("verified current-index timestamp is invalid") from error
        if (
            frozen_at.tzinfo is None
            or frozen_at.utcoffset() is None
            or not timedelta(0)
            <= now.astimezone(timezone.utc) - frozen_at.astimezone(timezone.utc)
            <= self.configuration.current_index_max_age
        ):
            return {
                "schema_version": self.REPORT_SCHEMA,
                "status": "AWAITING_COHORT_CANDIDATES",
                "reason": "CURRENT_RACE_INDEX_NOT_READY_NOW",
            }
        races = tuple(verified_index.races)
        venues = {
            race.get("venue")
            for race in races
            if isinstance(race, Mapping) and isinstance(race.get("venue"), str)
        }
        dates = {
            race.get("date")
            for race in races
            if isinstance(race, Mapping) and isinstance(race.get("date"), str)
        }
        if len(races) < 20 or len(venues) < 3 or len(dates) < 2:
            return {
                "schema_version": self.REPORT_SCHEMA,
                "status": "AWAITING_COHORT_CANDIDATES",
                "candidate_race_count": len(races),
                "candidate_venue_count": len(venues),
                "candidate_race_date_count": len(dates),
                "required_race_count": 20,
                "required_venue_count": 3,
                "required_race_date_count": 2,
            }
        selected = self._select_cohort(races)
        prepared: list[_ForwardCohortCandidate] = []
        for race in selected:
            runners = race.get("runners") if isinstance(race, Mapping) else None
            native_ids = (
                [runner.get("source_native_runner_id") for runner in runners]
                if isinstance(runners, list)
                and all(isinstance(runner, Mapping) for runner in runners)
                else []
            )
            if (
                len(native_ids) < 2
                or any(
                    type(native_id) is not str
                    or not native_id.isascii()
                    or not native_id.isdecimal()
                    for native_id in native_ids
                )
                or len(native_ids) != len(set(native_ids))
            ):
                return {
                    "schema_version": self.REPORT_SCHEMA,
                    "status": "INTEGRITY_FAILED",
                    "reason": "NUMERIC_SOURCE_NATIVE_RUNNER_IDS_REQUIRED",
                    "race_id": race.get("race_id") if isinstance(race, Mapping) else None,
                }
        for race, native_ids in zip(
            selected,
            (
                tuple(
                    runner["source_native_runner_id"]
                    for runner in race["runners"]
                )
                for race in selected
            ),
            strict=True,
        ):
            try:
                race_id = race["race_id"]
                racing_date = race["date"]
                venue = race["venue"]
                race_number = race["race_number"]
                native_race_id = race["source_native_race_id"]
                jump = datetime.fromisoformat(race["jump_datetime"])
                parsed_date = date.fromisoformat(racing_date)
                valid = (
                    type(race_id) is str
                    and bool(race_id.strip())
                    and type(racing_date) is str
                    and type(venue) is str
                    and bool(venue.strip())
                    and type(race_number) is int
                    and race_number > 0
                    and jump.tzinfo is not None
                    and jump.utcoffset() is not None
                    and jump.date() == parsed_date
                    and frozen_at
                    < jump - self.configuration.feature_cutoff
                    < jump
                )
            except (KeyError, TypeError, ValueError):
                valid = False
            if not valid:
                return {
                    "schema_version": self.REPORT_SCHEMA,
                    "status": "INTEGRITY_FAILED",
                    "reason": "COHORT_CANDIDATE_INVALID",
                    "race_id": race.get("race_id") if isinstance(race, Mapping) else None,
                }
            if (
                type(native_race_id) is not str
                or not native_race_id.isascii()
                or not native_race_id.isdecimal()
            ):
                return {
                    "schema_version": self.REPORT_SCHEMA,
                    "status": "INTEGRITY_FAILED",
                    "reason": "NUMERIC_SOURCE_NATIVE_RACE_IDS_REQUIRED",
                    "race_id": race_id,
                }
            prepared.append(
                _ForwardCohortCandidate(
                    race_id,
                    racing_date,
                    venue,
                    race_number,
                    jump,
                    native_race_id,
                    native_ids,
                )
            )
        native_race_ids = [race.source_native_race_id for race in prepared]
        if len(native_race_ids) != len(set(native_race_ids)):
            return {
                "schema_version": self.REPORT_SCHEMA,
                "status": "INTEGRITY_FAILED",
                "reason": "NUMERIC_SOURCE_NATIVE_RACE_IDS_REQUIRED",
                "race_id": None,
            }
        corpus = ForwardSealedCorpus(
            self.configuration.corpus_root,
            clock=lambda: now,
        )
        try:
            cohort_bytes = corpus.load_prediction_cohort(self.configuration.cohort_id)
        except (ArtifactStoreError, ForwardCorpusRejected):
            return self._integrity("FROZEN_COHORT_BINDING_MISMATCH")
        if cohort_bytes is not None:
            try:
                existing = json.loads(cohort_bytes)
                existing_frozen_at = datetime.fromisoformat(existing["frozen_at"])
                members = existing["members"]
                existing_projection = sorted(
                    (
                        member["racing_date"],
                        member["venue"],
                        member["race_number"],
                        member["source_native_race_id"],
                        tuple(member["source_native_runner_ids"]),
                        datetime.fromisoformat(member["feature_cutoff_at"]),
                        datetime.fromisoformat(member["scheduled_jump_at"]),
                    )
                    for member in members
                )
                prepared_projection = sorted(
                    (
                        race.racing_date,
                        race.venue,
                        race.race_number,
                        race.source_native_race_id,
                        race.source_native_runner_ids,
                        race.jump - self.configuration.feature_cutoff,
                        race.jump,
                    )
                    for race in prepared
                )
                binding_matches = (
                    type(existing) is dict
                    and canonical_json(existing) == cohort_bytes
                    and existing.get("schema_version") == "forward-baseline-cohort-v1"
                    and existing.get("cohort_id") == self.configuration.cohort_id
                    and existing.get("race_count") == 20
                    and type(members) is list
                    and len(members) == 20
                    and existing_frozen_at <= frozen_at
                    and existing_projection == prepared_projection
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                binding_matches = False
            if not binding_matches:
                return self._integrity("FROZEN_COHORT_BINDING_MISMATCH")
            self._register_cohort(cohort_bytes)
            return self._cohort_report(cohort_bytes)
        programme_checksum = ArtifactChecksum(f"sha256:{verified_index.packet_sha256}")
        days: dict[str, RacingDay] = {}
        for racing_date in sorted({race.racing_date for race in prepared}):
            day = RacingDay(
                RacingDayId(self._identity("day", f"{self.configuration.cohort_id}:{racing_date}")),
                date.fromisoformat(racing_date),
                self.configuration.timezone,
                frozen_at,
            )
            self.store.create_racing_day(
                OperationId(
                    self._identity("op", f"{self.configuration.cohort_id}:day:{racing_date}")
                ),
                day,
            )
            days[racing_date] = day
        members = []
        for race in prepared:
            jump = race.jump
            cutoff = jump - self.configuration.feature_cutoff
            native_race_id = race.source_native_race_id
            race_id = self.store.record_expected_race(
                OperationId(
                    self._identity(
                        "op", f"{self.configuration.cohort_id}:race:{native_race_id}"
                    )
                ),
                days[race.racing_date],
                ProgrammeRaceCandidate(
                    "collector-current-race-index-v2",
                    native_race_id,
                    race.venue,
                    race.race_number,
                    jump,
                ),
                programme_checksum,
                frozen_at,
            )
            members.append(
                {
                    "race_id": str(race_id),
                    "racing_date": race.racing_date,
                    "venue": race.venue,
                    "race_number": race.race_number,
                    "source_native_race_id": native_race_id,
                    "source_native_runner_ids": list(race.source_native_runner_ids),
                    "feature_cutoff_at": cutoff.isoformat(),
                    "scheduled_jump_at": jump.isoformat(),
                }
            )
        cohort = corpus.freeze_prediction_cohort(
            cohort_id=self.configuration.cohort_id,
            races=members,
            frozen_at=frozen_at.isoformat(),
        )
        cohort_bytes = canonical_json(
            {key: value for key, value in cohort.items() if key != "checksum"}
        )
        self._register_cohort(cohort_bytes)
        return self._cohort_report(cohort_bytes)

    def _register_cohort(self, cohort_bytes: bytes) -> None:
        cohort = json.loads(cohort_bytes)
        frozen_at = datetime.fromisoformat(cohort["frozen_at"])
        self.store.register_forward_baseline_cohort(
            OperationId(
                self._identity("op", f"{self.configuration.cohort_id}:cohort-authority")
            ),
            cohort_id=self.configuration.cohort_id,
            artifact_checksum=ArtifactChecksum(
                "sha256:" + hashlib.sha256(cohort_bytes).hexdigest()
            ),
            frozen_at=frozen_at,
            frozen_at_text=cohort["frozen_at"],
            members=cohort["members"],
            registered_at=frozen_at,
        )

    def _cohort_report(self, cohort_bytes: bytes) -> Mapping[str, Any]:
        cohort = json.loads(cohort_bytes)
        report = {
            "schema_version": self.REPORT_SCHEMA,
            "status": "COHORT_FROZEN_AWAITING_SCHEDULED_CAPTURE",
            "cohort_id": cohort["cohort_id"],
            "cohort_checksum": "sha256:" + hashlib.sha256(cohort_bytes).hexdigest(),
            "race_count": 20,
            "terminal_count": 0,
        }
        try:
            terminal = self.forecasting.baseline_cohort_terminal_records(cohort_bytes)
        except BarrierNotSatisfied:
            return report
        except OperationsStoreError:
            return self._integrity("FROZEN_COHORT_LIFECYCLE_MISMATCH")
        return {
            **report,
            "status": "COHORT_TERMINAL",
            "terminal_count": terminal["terminal_count"],
            "terminal_records": terminal["records"],
        }

    def _integrity(self, reason: str) -> Mapping[str, Any]:
        return {
            "schema_version": self.REPORT_SCHEMA,
            "status": "INTEGRITY_FAILED",
            "reason": reason,
        }

    @staticmethod
    def _identity(prefix: str, value: str) -> str:
        return f"{prefix}_{hashlib.sha256(value.encode()).hexdigest()[:32]}"

    @staticmethod
    def _select_cohort(races: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
        ordered = sorted(
            races,
            key=lambda race: (
                race.get("jump_datetime", ""),
                race.get("venue", ""),
                race.get("race_number", 0),
                race.get("race_id", ""),
            ),
        )
        selected: list[Mapping[str, Any]] = []
        for field, required in (("date", 2), ("venue", 3)):
            seen = {race[field] for race in selected}
            for race in ordered:
                if len(seen) >= required:
                    break
                if race[field] not in seen:
                    selected.append(race)
                    seen.add(race[field])
        for race in ordered:
            if len(selected) == 20:
                break
            if race not in selected:
                selected.append(race)
        return tuple(selected)


@dataclass(frozen=True, slots=True)
class ForwardBaselineCaptureBinding:
    """Validated runtime paths bound to one cohort-aware capture module."""

    service: ForwardBaselineCaptureService
    evidence_root: Path
    current_index_path: Path
    current_index_max_age_seconds: int
    current_index_timeout_seconds: float


def load_forward_baseline_capture_binding(
    config_path: Path,
) -> ForwardBaselineCaptureBinding:
    """Load one closed configuration for preflight and scheduled admission callers."""
    try:
        content = config_path.read_bytes()
        document = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ServiceUnavailable("forward baseline configuration is unavailable") from error
    expected = {
        "schema_version",
        "cohort_id",
        "operations_database",
        "corpus_root",
        "evidence_root",
        "current_race_index_path",
        "current_index_max_age_seconds",
        "current_index_timeout_seconds",
        "feature_cutoff_seconds",
        "timezone",
    }
    if (
        type(document) is not dict
        or set(document) != expected
        or document.get("schema_version")
        != "forward-baseline-capture-service-config-v1"
        or content != json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ):
        raise ServiceUnavailable("forward baseline configuration is malformed")
    try:
        database = Path(document["operations_database"])
        corpus_root = Path(document["corpus_root"])
        evidence_root = Path(document["evidence_root"])
        index_path = Path(document["current_race_index_path"])
        max_age = document["current_index_max_age_seconds"]
        timeout = document["current_index_timeout_seconds"]
        cutoff = document["feature_cutoff_seconds"]
        if (
            any(
                not path.is_absolute()
                for path in (database, corpus_root, evidence_root, index_path)
            )
            or not database.is_file()
            or database.is_symlink()
            or not evidence_root.is_dir()
            or evidence_root.is_symlink()
            or not index_path.is_relative_to(evidence_root)
            or isinstance(max_age, bool)
            or type(max_age) is not int
            or max_age <= 0
            or isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
            or isinstance(cutoff, bool)
            or type(cutoff) is not int
            or cutoff <= 0
        ):
            raise ValueError
        configuration = ForwardBaselineCaptureConfiguration(
            cohort_id=document["cohort_id"],
            corpus_root=corpus_root,
            current_index_max_age=timedelta(seconds=max_age),
            feature_cutoff=timedelta(seconds=cutoff),
            timezone=document["timezone"],
        )
    except (TypeError, ValueError) as error:
        raise ServiceUnavailable("forward baseline configuration is malformed") from error
    return ForwardBaselineCaptureBinding(
        service=ForwardBaselineCaptureService(
            SQLiteOperationsStore(database), configuration
        ),
        evidence_root=evidence_root,
        current_index_path=index_path,
        current_index_max_age_seconds=max_age,
        current_index_timeout_seconds=float(timeout),
    )


def run_forward_baseline_capture(
    config_path: Path,
    *,
    now: datetime | None = None,
    current_index_reader: Callable[..., Any] | None = None,
) -> Mapping[str, Any]:
    """Checked-in production entrypoint for one bounded capture preflight/schedule pass."""
    binding = load_forward_baseline_capture_binding(config_path)
    if current_index_reader is None:
        from .synchronous_manual_capture import bounded_current_race_index

        current_index_reader = bounded_current_race_index
    observed_at = now or datetime.now(timezone.utc)
    verified = current_index_reader(
        current_time=observed_at,
        timeout_seconds=binding.current_index_timeout_seconds,
        index_path=binding.current_index_path,
        evidence_root=binding.evidence_root,
        max_age_seconds=binding.current_index_max_age_seconds,
        return_verified_view=True,
    )
    return binding.service.run(verified, now=observed_at)


@dataclass(frozen=True, slots=True)
class RacingDayCycle:
    """One immutable Racing Day plan with an explicitly bounded execution prefix."""

    racing_day_id: str
    commands: tuple[ApplicationCommand, ...]
    plan_operation_id: OperationId
    advancement_operation_ids: tuple[OperationId, ...]
    at: datetime
    terminal_phase: str = "request_training"
    plan_commands: tuple[ApplicationCommand, ...] | None = None

    def __post_init__(self) -> None:
        planned_commands = self.planned_commands
        planned_phases = tuple(command.phase for command in planned_commands)
        try:
            terminal_ordinal = RaceCollectionService.ORDER.index(self.terminal_phase) + 1
        except ValueError:
            terminal_ordinal = 0
        if (
            not isinstance(self.racing_day_id, str)
            or not self.racing_day_id.strip()
            or self.terminal_phase not in {"deferred_prediction", "request_training"}
            or planned_phases != RaceCollectionService.ORDER
            or self.commands != planned_commands[:terminal_ordinal]
            or len(self.advancement_operation_ids) != len(self.commands)
            or any(command.racing_day_id != self.racing_day_id for command in planned_commands)
            or len(
                {
                    self.plan_operation_id,
                    *self.advancement_operation_ids,
                    *(command.operation_id for command in planned_commands),
                }
            )
            != 1 + len(self.advancement_operation_ids) + len(planned_commands)
            or self.at.tzinfo is None
            or self.at.utcoffset() is None
        ):
            raise ValueError("runtime cycle is not one exact ordered Racing Day plan")

    @property
    def terminal_ordinal(self) -> int:
        """Return the last phase this immutable input is authorized to execute."""
        return RaceCollectionService.ORDER.index(self.terminal_phase) + 1

    @property
    def planned_commands(self) -> tuple[ApplicationCommand, ...]:
        """Return the complete authenticated plan retained for recovery."""
        return self.commands if self.plan_commands is None else self.plan_commands


class RuntimeAdapter(Protocol):
    """Live capability plugin; it supplies bindings and plans, never a scheduler."""

    def registrations(self) -> Sequence[PhaseHandlerRegistration]:
        """Return exactly one trusted handler for every closed command type."""

    def next_cycle(self, *, now: datetime) -> RacingDayCycle | None:
        """Return the next plan, rehydrating any durable command IDs for a partial day."""

    def bind_release_authority(self, mode: str) -> None:
        """Bind every immutable cycle mode and terminal phase to release authority."""

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
        if self.release_id is None and self.mode is None:
            return
        if (
            not isinstance(self.release_id, str)
            or not self.release_id.strip()
            or self.mode not in {"active", "observation"}
        ):
            raise OperationalRejected("service release authority mode is incomplete or unsupported")
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

    def _assert_cycle_authority(self, cycle: RacingDayCycle) -> None:
        try:
            cycle.__post_init__()
        except ValueError as error:
            raise OperationalRejected(
                "runtime cycle authority contract is internally inconsistent"
            ) from error
        if self.mode == "observation" and (
            cycle.terminal_phase != "deferred_prediction"
            or tuple(command.phase for command in cycle.commands) != RaceCollectionService.ORDER[:5]
        ):
            raise OperationalRejected(
                "observation authority cannot execute beyond deferred prediction"
            )

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
        self._assert_cycle_authority(cycle)
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
            commands=cycle.planned_commands,
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
        if any(ordinal > cycle.terminal_ordinal for ordinal in completed):
            raise OperationalRejected(
                "runtime cycle has durable progress beyond its authorized terminal phase"
            )
        results = []
        for ordinal, (command, advancement_id) in enumerate(
            zip(
                cycle.commands,
                cycle.advancement_operation_ids,
                strict=True,
            ),
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
        binder = getattr(adapter, "bind_release_authority", None)
        if not callable(binder):
            raise ServiceUnavailable("runtime adapter release authority contract is unavailable")
        binder(mode)
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
    configuration = parser.add_mutually_exclusive_group(required=True)
    configuration.add_argument("--config", type=Path)
    configuration.add_argument("--forward-baseline-config", type=Path)
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
    forward_baseline_runner: Callable[[Path], Mapping[str, Any]] = (
        run_forward_baseline_capture
    ),
) -> int:
    args = _parser().parse_args(argv)
    if args.lease_seconds < 1 or args.poll_seconds <= 0:
        _parser().error("lease and poll intervals must be positive")
    if args.forward_baseline_config is not None:
        if args.once or args.continuous:
            _parser().error("forward baseline capture does not accept service cycle modes")
        try:
            report = forward_baseline_runner(args.forward_baseline_config)
            print(
                json.dumps(
                    report,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
            return 65 if report.get("status") == "INTEGRITY_FAILED" else 0
        except Exception:
            print("race-collection-service unavailable", file=sys.stderr)
            return 69
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
