"""Checked-in immutable-input adapter for the authoritative Race Collection Service."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta
from typing import Any, Mapping, Sequence

from .artifacts import ArtifactStoreError, LocalArtifactStore
from .collection import CollectionRepository
from .domain import (
    ArtifactChecksum,
    EvidenceAuthority,
    EvidenceField,
    FieldEvidence,
    OddsAttemptStatus,
    OddsObservation,
    OperationId,
    RaceId,
    RaceState,
    RacingDay,
    RacingDayId,
    RunObservation,
)
from .evaluation import EvaluationAuthority
from .forecast_service import CanonicalForecastService, ForecastRequest
from .forecasting import ForecastingAuthority
from .identity import resolve_dog_identity
from .inventory import JsonProgrammeAdapter
from .model_bundle import ChampionLoader
from .operational import (
    COMMAND_PHASES,
    ApplicationCommand,
    CloseAndSeal,
    CollectAdaptiveOdds,
    CollectCardsAndForm,
    CollectResults,
    CommitDeferredPrediction,
    DayForecastCohortMember,
    DiscoverProgramme,
    JoinTrainingExamples,
    OperationalAuthority,
    OperationalRejected,
    PhaseHandlerRegistration,
    ReconcileRacingDay,
    ReleaseConfiguration,
    RequestTraining,
    _adaptive_odds_history_complete,
)
from .operations import SQLiteOperationsStore
from .sealing import EvidenceSealer, FieldObservation
from .service import RacingDayCycle, ServiceUnavailable
from .training import LinearStrengthModel, TrainingCorpusAuthority

_ROOT_KEYS = frozenset({"schema_version", "release_id", "cycles"})
_CYCLE_KEYS = frozenset(
    {
        "racing_day_id",
        "local_date",
        "timezone",
        "opened_at",
        "at",
        "plan_operation_id",
        "command_operation_ids",
        "advancement_operation_ids",
        "programme",
        "races",
        "champion",
        "forecast_cohort",
        "determinism_input_checksum",
        "training_request",
    }
)
_PROGRAMME_KEYS = frozenset({"source", "checksum"})
_RACE_KEYS = frozenset(
    {
        "source_race_id",
        "identities",
        "run_observations",
        "observations",
        "odds_attempts",
        "seal",
        "prediction",
    }
)
_FULL_RACE_KEYS = _RACE_KEYS | {"result", "training_example"}
_RESULT_BLIND_MODE = "result-blind-observation-v1"
_COMPLETE_MODE = "complete-v1"
_IDENTITY_KEYS = frozenset(
    {
        "operation_id",
        "source",
        "source_alias",
        "name",
        "registration_authority",
        "registration_id",
        "decided_at",
    }
)
_RUN_KEYS = frozenset(
    {
        "operation_id",
        "identity_source_alias",
        "local_racing_date",
        "source",
        "checksum",
        "observed_at",
        "starts",
        "wins",
        "authoritative",
    }
)
_OBSERVATION_KEYS = frozenset(
    {"operation_id", "field", "authority", "value", "source", "checksum", "observed_at"}
)
_ODDS_KEYS = frozenset(
    {
        "operation_id",
        "source",
        "scheduled_due_at",
        "attempted_at",
        "timing_policy",
        "status",
        "artifact_checksum",
        "runner_mapping_checksum",
        "error",
    }
)
_SEAL_KEYS = frozenset(
    {
        "operation_id",
        "buffer_seconds",
        "schema_version",
        "normalization_version",
        "sealed_at",
    }
)
_PREDICTION_KEYS = frozenset({"begin_operation_id", "operation_id", "prediction_id"})
_RESULT_KEYS = frozenset(
    {
        "open_operation_id",
        "operation_id",
        "attempt_id",
        "attempted_at",
        "timing_policy",
        "deadline",
        "max_attempts",
        "source",
        "source_checksum",
    }
)

OFFICIAL_RESULT_TIMING_POLICY = "official-result-timing-v1"
OFFICIAL_RESULT_MAX_LATENCY = timedelta(minutes=5)
_EXAMPLE_KEYS = frozenset(
    {
        "join_operation_id",
        "phase3_example_id",
        "build_operation_id",
        "canonical_example_id",
        "joined_at",
    }
)
_CHAMPION_KEYS = frozenset({"bundle_id", "bundle_checksum"})
_COHORT_KEYS = frozenset({"assignment_id", "authorization_operation_id", "members"})
_COHORT_MEMBER_KEYS = frozenset(
    {
        "role",
        "bundle_id",
        "bundle_checksum",
        "service_run_id",
        "forecast_operations",
    }
)
_FORECAST_OPERATION_KEYS = frozenset({"source_race_id", "operation_id"})
_TRAINING_KEYS = frozenset(
    {
        "request_id",
        "request_operation_id",
        "authorization_operation_id",
        "binding_operation_id",
        "service_run_id",
    }
)


def _operation(label: str) -> OperationId:
    return OperationId(f"op_{hashlib.sha256(label.encode()).hexdigest()[:32]}")


def _strict_object(value: Any, keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ServiceUnavailable(f"runtime {label} has unknown or missing keys")
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _datetime(value: Any, label: str) -> datetime:
    if type(value) is not str:
        raise ServiceUnavailable(f"runtime {label} must be an ISO timestamp")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ServiceUnavailable(f"runtime {label} must be timezone-aware")
    return parsed


def _official_result_timeline_valid(
    published_at: Any,
    observed_at: Any,
    attempted_at: Any,
    trusted_command_at: Any,
    timing_policy: Any,
) -> bool:
    """Validate one versioned ordered official-result acquisition timeline."""
    moments = (published_at, observed_at, attempted_at, trusted_command_at)
    if (
        timing_policy != OFFICIAL_RESULT_TIMING_POLICY
        or any(type(moment) is not datetime for moment in moments)
        or any(moment.tzinfo is None or moment.utcoffset() is None for moment in moments)
    ):
        return False
    return (
        published_at <= observed_at <= attempted_at <= trusted_command_at
        and attempted_at - published_at <= OFFICIAL_RESULT_MAX_LATENCY
    )


def _identities(values: Any, count: int, label: str) -> tuple[OperationId, ...]:
    if type(values) is not list or len(values) != count:
        raise ServiceUnavailable(f"runtime {label} must contain exactly {count} identities")
    result = tuple(OperationId(value) for value in values)
    if len(set(result)) != len(result):
        raise ServiceUnavailable(f"runtime {label} contains duplicate identities")
    return result


class _ExactRegisteredDeferredPredictor:
    """Phase-3 predictor pinned to the manifest's exact registered champion."""

    def __init__(
        self,
        service: CanonicalForecastService,
        artifacts: LocalArtifactStore,
        champion: Any,
        computed_at: datetime,
    ):
        self.service = service
        self.artifacts = artifacts
        self.champion = champion
        self.computed_at = computed_at

    def predict(self, request: Any) -> ArtifactChecksum:
        result = self.service.forecast_with_champion(
            self.champion,
            ForecastRequest(request.evidence_checksum, request.seal_id, str(request.race_id)),
            computed_at=self.computed_at,
        )
        return self.artifacts.put(
            _canonical(result),
            media_type="application/vnd.canonical-race-forecast+json",
        ).checksum

    def authenticate(self, checksum: ArtifactChecksum, expected_computed_at: datetime) -> None:
        document = json.loads(self.artifacts.read(checksum))
        if document.get("provenance", {}).get(
            "prediction_computed_at"
        ) != expected_computed_at.isoformat(timespec="microseconds"):
            raise ServiceUnavailable("prediction computation time disagrees")


class ImmutableInputRuntimeAdapter:
    """Execute one closed plan, including a result-blind observation prefix."""

    def __init__(
        self,
        configuration: ReleaseConfiguration,
        store: SQLiteOperationsStore,
        artifacts: LocalArtifactStore,
    ):
        self._store = store
        self._artifacts = artifacts
        self._repository = CollectionRepository(store)
        self._authority = OperationalAuthority(store, artifacts)
        self._closed = False
        try:
            content = artifacts.read(configuration.runtime_input_checksum)
            document = json.loads(content)
            _strict_object(document, _ROOT_KEYS, "input")
            if content != _canonical(document):
                raise ServiceUnavailable("runtime input is not canonical JSON")
            if document["schema_version"] != "phase7-runtime-input-v1":
                raise ServiceUnavailable("runtime input schema is unsupported")
            if type(document["release_id"]) is not str or not document["release_id"].strip():
                raise ServiceUnavailable("runtime release identity is malformed")
            if type(document["cycles"]) is not list or not document["cycles"]:
                raise ServiceUnavailable("runtime input must contain cycles")
            self._release_id = document["release_id"]
            parsed = tuple(self._parse_cycle(item) for item in document["cycles"])
        except ServiceUnavailable:
            raise
        except (
            ArtifactStoreError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise ServiceUnavailable("immutable runtime input contract is malformed") from error
        day_ids = [cycle.racing_day_id for cycle, _ in parsed]
        all_operations = [
            str(identity)
            for cycle, _ in parsed
            for identity in (
                cycle.plan_operation_id,
                *cycle.advancement_operation_ids,
                *(command.operation_id for command in cycle.commands),
            )
        ]
        if len(day_ids) != len(set(day_ids)) or len(all_operations) != len(set(all_operations)):
            raise ServiceUnavailable("runtime input contains duplicate immutable identities")
        self._cycles = tuple(cycle for cycle, _ in parsed)
        self._documents = {cycle.racing_day_id: item for cycle, item in parsed}
        self._verify_release_and_sources(configuration, document)
        self._verify_registered_forecast_cohorts(document)
        self._verify_external_artifacts(document)

    def _verify_release_and_sources(
        self,
        configuration: ReleaseConfiguration,
        document: Mapping[str, Any],
    ) -> None:
        config_checksum = (
            "sha256:" + hashlib.sha256(_canonical(configuration.document())).hexdigest()
        )
        with self._store._connect() as db:
            manifest = db.execute(
                "SELECT 1 FROM phase7_release_manifests "
                "WHERE release_id=? AND config_checksum=?",
                (self._release_id, config_checksum),
            ).fetchone()
            pointer = db.execute(
                "SELECT release_id,authority,legacy_preserved "
                "FROM phase7_release_pointer WHERE singleton=1"
            ).fetchone()
            observation = db.execute(
                "SELECT candidate_release_id,action FROM "
                "phase7_observation_authority_events "
                "ORDER BY event_id DESC LIMIT 1"
            ).fetchone()
        authorized = (
            pointer is not None
            and pointer["authority"] == "race_collection_service"
            and pointer["release_id"] == self._release_id
        ) or (
            pointer is not None
            and pointer["authority"] == "legacy"
            and pointer["legacy_preserved"]
            and observation is not None
            and observation["action"] == "authorize"
            and observation["candidate_release_id"] == self._release_id
        )
        if manifest is None or not authorized:
            raise ServiceUnavailable(
                "runtime input release is not the configured authorized release"
            )
        configured_sources = set(configuration.sources)
        supplied_sources = {cycle["programme"]["source"] for cycle in document["cycles"]}
        for cycle in document["cycles"]:
            for race in cycle["races"]:
                supplied_sources.update(identity["source"] for identity in race["identities"])
                supplied_sources.update(
                    observation["source"] for observation in race["run_observations"]
                )
                supplied_sources.update(
                    observation["source"] for observation in race["observations"]
                )
                supplied_sources.update(attempt["source"] for attempt in race["odds_attempts"])
                if cycle.get("mode", _COMPLETE_MODE) == _COMPLETE_MODE:
                    supplied_sources.add(race["result"]["source"])
        if not supplied_sources <= configured_sources:
            raise ServiceUnavailable("runtime input uses a source outside the configured release")

    def _parse_cycle(self, value: Any) -> tuple[RacingDayCycle, Mapping[str, Any]]:
        if type(value) is not dict:
            raise ServiceUnavailable("runtime cycle has unknown or missing keys")
        mode = value.get("mode", _COMPLETE_MODE)
        expected_cycle_keys = _CYCLE_KEYS | ({"mode"} if "mode" in value else set())
        item = _strict_object(value, frozenset(expected_cycle_keys), "cycle")
        if mode not in {_COMPLETE_MODE, _RESULT_BLIND_MODE}:
            raise ServiceUnavailable("runtime cycle mode is unsupported")
        programme = _strict_object(item["programme"], _PROGRAMME_KEYS, "programme")
        champion = _strict_object(item["champion"], _CHAMPION_KEYS, "champion")
        cohort = _strict_object(item["forecast_cohort"], _COHORT_KEYS, "day forecast cohort")
        training = _strict_object(item["training_request"], _TRAINING_KEYS, "training request")
        if type(item["races"]) is not list or not item["races"]:
            raise ServiceUnavailable("runtime cycle must contain race inputs")
        source_ids: list[str] = []
        nested_operations: list[OperationId] = []
        for race_value in item["races"]:
            race_keys = _RACE_KEYS if mode == _RESULT_BLIND_MODE else _FULL_RACE_KEYS
            race = _strict_object(race_value, race_keys, "race")
            if type(race["source_race_id"]) is not str or not race["source_race_id"].strip():
                raise ServiceUnavailable("runtime source race identity is malformed")
            source_ids.append(race["source_race_id"])
            if type(race["identities"]) is not list or not race["identities"]:
                raise ServiceUnavailable("runtime race identities are missing")
            aliases: list[str] = []
            for identity in race["identities"]:
                _strict_object(identity, _IDENTITY_KEYS, "runner identity")
                aliases.append(identity["source_alias"])
                nested_operations.append(OperationId(identity["operation_id"]))
            if len(aliases) != len(set(aliases)):
                raise ServiceUnavailable("runtime race contains duplicate runner identities")
            if type(race["run_observations"]) is not list:
                raise ServiceUnavailable("runtime form observations are malformed")
            for run in race["run_observations"]:
                _strict_object(run, _RUN_KEYS, "run observation")
                if run["identity_source_alias"] not in aliases:
                    raise ServiceUnavailable("runtime form observation identity is unavailable")
                nested_operations.append(OperationId(run["operation_id"]))
            if type(race["observations"]) is not list or not race["observations"]:
                raise ServiceUnavailable("runtime race observations are missing")
            for observation in race["observations"]:
                _strict_object(observation, _OBSERVATION_KEYS, "observation")
                nested_operations.append(OperationId(observation["operation_id"]))
            if type(race["odds_attempts"]) is not list or not race["odds_attempts"]:
                raise ServiceUnavailable("runtime odds history is missing")
            for attempt in race["odds_attempts"]:
                _strict_object(attempt, _ODDS_KEYS, "odds attempt")
                nested_operations.append(OperationId(attempt["operation_id"]))
            nested_contracts = [
                ("seal", _SEAL_KEYS),
                ("prediction", _PREDICTION_KEYS),
            ]
            if mode == _COMPLETE_MODE:
                nested_contracts.extend(
                    (
                        ("result", _RESULT_KEYS),
                        ("training_example", _EXAMPLE_KEYS),
                    )
                )
            for name, keys in nested_contracts:
                nested = _strict_object(race[name], keys, name)
                nested_operations.extend(
                    OperationId(value)
                    for key, value in nested.items()
                    if key.endswith("operation_id")
                )
            if mode == _COMPLETE_MODE:
                ArtifactChecksum(race["result"]["source_checksum"])
        if len(source_ids) != len(set(source_ids)):
            raise ServiceUnavailable("runtime cycle contains duplicate source race identities")
        if type(cohort["members"]) is not list or len(cohort["members"]) < 2:
            raise ServiceUnavailable("runtime day forecast cohort requires champion and challenger")
        cohort_operations = [OperationId(cohort["authorization_operation_id"])]
        cohort_runs: list[OperationId] = []
        for member_value in cohort["members"]:
            member = _strict_object(member_value, _COHORT_MEMBER_KEYS, "day forecast cohort member")
            if member["role"] not in {"champion", "challenger"}:
                raise ServiceUnavailable("runtime day forecast cohort role is invalid")
            ArtifactChecksum(member["bundle_checksum"])
            cohort_runs.append(OperationId(member["service_run_id"]))
            bindings = member["forecast_operations"]
            if type(bindings) is not list or len(bindings) != len(source_ids):
                raise ServiceUnavailable(
                    "runtime day forecast operations must cover every source race"
                )
            checked_bindings = [
                _strict_object(binding, _FORECAST_OPERATION_KEYS, "day forecast operation")
                for binding in bindings
            ]
            if {binding["source_race_id"] for binding in checked_bindings} != set(
                source_ids
            ) or any(
                type(binding["source_race_id"]) is not str or not binding["source_race_id"].strip()
                for binding in checked_bindings
            ):
                raise ServiceUnavailable(
                    "runtime day forecast operations have ambiguous source-race coverage"
                )
            cohort_operations.extend(
                _identities(
                    [binding["operation_id"] for binding in checked_bindings],
                    len(source_ids),
                    "day forecast operation identities",
                )
            )
        if (
            sum(member["role"] == "champion" for member in cohort["members"]) != 1
            or not any(member["role"] == "challenger" for member in cohort["members"])
            or len(cohort_runs) != len(set(cohort_runs))
        ):
            raise ServiceUnavailable("runtime day forecast cohort is ambiguous")
        payloads = (
            DiscoverProgramme(programme["source"], ArtifactChecksum(programme["checksum"])),
            CollectCardsAndForm(),
            CollectAdaptiveOdds(),
            CloseAndSeal(),
            CommitDeferredPrediction(),
            CollectResults(),
            JoinTrainingExamples(),
            ReconcileRacingDay(),
            RequestTraining(
                training["request_id"],
                OperationId(training["request_operation_id"]),
                OperationId(training["authorization_operation_id"]),
                OperationId(training["binding_operation_id"]),
            ),
        )
        command_ids = _identities(item["command_operation_ids"], 9, "command identities")
        advancement_ids = _identities(
            item["advancement_operation_ids"], 9, "advancement identities"
        )
        commands = tuple(
            ApplicationCommand(identity, item["racing_day_id"], payload)
            for identity, payload in zip(command_ids, payloads, strict=True)
        )
        immutable_ids = (
            OperationId(item["plan_operation_id"]),
            *command_ids,
            *advancement_ids,
            *nested_operations,
            OperationId(training["request_operation_id"]),
            OperationId(training["authorization_operation_id"]),
            OperationId(training["binding_operation_id"]),
            OperationId(training["service_run_id"]),
            *cohort_operations,
            *cohort_runs,
        )
        if len(immutable_ids) != len(set(immutable_ids)):
            raise ServiceUnavailable("runtime cycle contains duplicate operation identities")
        ArtifactChecksum(champion["bundle_checksum"])
        ArtifactChecksum(item["determinism_input_checksum"])
        cycle = RacingDayCycle(
            item["racing_day_id"],
            commands,
            OperationId(item["plan_operation_id"]),
            advancement_ids,
            _datetime(item["at"], "cycle time"),
            "deferred_prediction" if mode == _RESULT_BLIND_MODE else "request_training",
        )
        date.fromisoformat(item["local_date"])
        _datetime(item["opened_at"], "opened_at")
        return cycle, item

    def _verify_registered_forecast_cohorts(self, document: Mapping[str, Any]) -> None:
        """Authenticate champion and challenger registry identities before receipt one."""
        try:
            with self._store._connect() as db:
                for cycle in document["cycles"]:
                    champion = cycle["champion"]
                    cohort = cycle["forecast_cohort"]
                    champion_members = [
                        member for member in cohort["members"] if member["role"] == "champion"
                    ]
                    assignment = db.execute(
                        "SELECT assignment_id,bundle_id,bundle_checksum "
                        "FROM canonical_day_assignments WHERE racing_day_id=?",
                        (cycle["racing_day_id"],),
                    ).fetchone()
                    if (
                        len(champion_members) != 1
                        or champion_members[0]["bundle_id"] != champion["bundle_id"]
                        or champion_members[0]["bundle_checksum"] != champion["bundle_checksum"]
                        or assignment is None
                        or assignment["assignment_id"] != cohort["assignment_id"]
                        or assignment["bundle_id"] != champion["bundle_id"]
                        or assignment["bundle_checksum"] != champion["bundle_checksum"]
                    ):
                        raise ServiceUnavailable(
                            "runtime champion is not the exact registered Racing Day assignment"
                        )
                    for member in cohort["members"]:
                        bundle = db.execute(
                            "SELECT created_at FROM canonical_model_bundles "
                            "WHERE bundle_id=? AND bundle_checksum=?",
                            (member["bundle_id"], member["bundle_checksum"]),
                        ).fetchone()
                        components = db.execute(
                            "SELECT artifact_checksum FROM canonical_bundle_components "
                            "WHERE bundle_id=? ORDER BY component_kind",
                            (member["bundle_id"],),
                        ).fetchall()
                        if (
                            bundle is None
                            or len(components) != 9
                            or _datetime(bundle["created_at"], "bundle registration time")
                            >= _datetime(cycle["at"], "cycle time")
                        ):
                            raise ServiceUnavailable(
                                f"runtime {member['role']} is not exactly registered"
                            )
                        self._artifacts.verify(ArtifactChecksum(member["bundle_checksum"]))
                        for component in components:
                            self._artifacts.verify(ArtifactChecksum(component["artifact_checksum"]))
                    bundle_identities = [
                        (member["bundle_id"], member["bundle_checksum"])
                        for member in cohort["members"]
                    ]
                    if len(bundle_identities) != len(set(bundle_identities)):
                        raise ServiceUnavailable(
                            "runtime champion and challenger registrations are not unique"
                        )
        except ServiceUnavailable:
            raise
        except (ArtifactStoreError, KeyError, TypeError, ValueError) as error:
            raise ServiceUnavailable(
                "runtime champion or challenger registration is unavailable"
            ) from error

    def _verify_external_artifacts(self, document: Mapping[str, Any]) -> None:
        """Read every pre-result source object before accepting a plan."""
        checksums: set[ArtifactChecksum] = set()
        for cycle in document["cycles"]:
            checksums.add(ArtifactChecksum(cycle["programme"]["checksum"]))
            checksums.add(ArtifactChecksum(cycle["determinism_input_checksum"]))
            for race in cycle["races"]:
                checksums.update(
                    ArtifactChecksum(item["checksum"]) for item in race["observations"]
                )
                checksums.update(
                    ArtifactChecksum(item["checksum"]) for item in race["run_observations"]
                )
                for attempt in race["odds_attempts"]:
                    if attempt["artifact_checksum"] is not None:
                        checksums.add(ArtifactChecksum(attempt["artifact_checksum"]))
                    if attempt["runner_mapping_checksum"] is not None:
                        checksums.add(ArtifactChecksum(attempt["runner_mapping_checksum"]))
        try:
            for checksum in checksums:
                self._artifacts.verify(checksum)
        except ArtifactStoreError as error:
            raise ServiceUnavailable("runtime source artifact is unreadable or mutable") from error

    def registrations(self) -> Sequence[PhaseHandlerRegistration]:
        handlers = {
            DiscoverProgramme: self._discover,
            CollectCardsAndForm: self._cards,
            CollectAdaptiveOdds: self._odds,
            CloseAndSeal: self._close_and_seal,
            CommitDeferredPrediction: self._predict,
            CollectResults: self._results,
            JoinTrainingExamples: self._join,
            ReconcileRacingDay: self._reconcile,
            RequestTraining: self._request_training,
        }
        return tuple(
            PhaseHandlerRegistration(payload_type, handlers[payload_type])
            for payload_type in COMMAND_PHASES
        )

    def next_cycle(self, *, now: datetime) -> RacingDayCycle | None:
        if self._closed:
            raise ServiceUnavailable("runtime adapter is closed")
        with self._store._connect() as db:
            for cycle in self._cycles:
                day = db.execute(
                    "SELECT local_date,timezone,opened_at FROM racing_days WHERE racing_day_id=?",
                    (cycle.racing_day_id,),
                ).fetchone()
                if day is None:
                    raise ServiceUnavailable("configured Racing Day identity is unavailable")
                item = self._documents[cycle.racing_day_id]
                if (
                    day["local_date"] != item["local_date"]
                    or day["timezone"] != item["timezone"]
                    or day["opened_at"] != item["opened_at"]
                ):
                    raise ServiceUnavailable("configured Racing Day identity changed")
                completed = [
                    row[0]
                    for row in db.execute(
                        "SELECT phase_ordinal FROM phase7_scheduler_progress "
                        "WHERE racing_day_id=? ORDER BY phase_ordinal",
                        (cycle.racing_day_id,),
                    )
                ]
                if completed != list(range(1, len(completed) + 1)):
                    raise ServiceUnavailable(
                        "runtime cycle progress is not a contiguous receipt prefix"
                    )
                if any(ordinal > cycle.terminal_ordinal for ordinal in completed):
                    raise ServiceUnavailable(
                        "runtime cycle has progress beyond its authorized terminal phase"
                    )
                if len(completed) < cycle.terminal_ordinal and cycle.at <= now:
                    return cycle
        return None

    def close(self) -> None:
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def _document(self, command: ApplicationCommand) -> Mapping[str, Any]:
        try:
            return self._documents[command.racing_day_id]
        except KeyError as error:
            raise ServiceUnavailable("command Racing Day is outside immutable input") from error

    def _day(self, item: Mapping[str, Any]) -> RacingDay:
        return RacingDay(
            RacingDayId(item["racing_day_id"]),
            date.fromisoformat(item["local_date"]),
            item["timezone"],
            _datetime(item["opened_at"], "opened_at"),
        )

    def _race_inputs(
        self, command: ApplicationCommand
    ) -> tuple[tuple[RaceId, Mapping[str, Any], Mapping[str, Any]], ...]:
        item = self._document(command)
        supplied = {race["source_race_id"]: race for race in item["races"]}
        with self._store._connect() as db:
            rows = db.execute(
                "SELECT e.race_id,e.source_race_id,e.scheduled_jump FROM expected_races e "
                "JOIN races r USING(race_id) WHERE r.racing_day_id=? ORDER BY e.race_id",
                (command.racing_day_id,),
            ).fetchall()
        if set(supplied) != {row["source_race_id"] for row in rows}:
            raise ServiceUnavailable("runtime race inputs disagree with official programme")
        return tuple((RaceId(row["race_id"]), supplied[row["source_race_id"]], row) for row in rows)

    def _discover(self, command: ApplicationCommand, at: datetime) -> None:
        del at
        item = self._document(command)
        programme = item["programme"]
        content = self._artifacts.read(command.payload.programme_checksum)
        candidates = JsonProgrammeAdapter(programme["source"]).parse(content)
        if {candidate.source_race_id for candidate in candidates} != {
            race["source_race_id"] for race in item["races"]
        }:
            raise ServiceUnavailable("runtime race inputs disagree with programme bytes")
        day = self._day(item)
        for index, candidate in enumerate(candidates):
            self._repository.record_expected_race(
                _operation(f"{command.operation_id}:race:{index}"),
                day,
                candidate,
                command.payload.programme_checksum,
                min(
                    _datetime(
                        race["odds_attempts"][0]["scheduled_due_at"],
                        "programme discovery time",
                    )
                    for race in item["races"]
                ),
            )

    @staticmethod
    def _field_value(field: EvidenceField, value: Any) -> Any:
        if field in {
            EvidenceField.SCHEDULED_JUMP,
            EvidenceField.ACTUAL_JUMP,
            EvidenceField.JUMP_TIME,
        }:
            return _datetime(value, field.value)
        return value

    def _cards(self, command: ApplicationCommand, at: datetime) -> None:
        del at
        for race_id, item, _ in self._race_inputs(command):
            card_at = max(
                *(
                    _datetime(identity["decided_at"], "identity decision time")
                    for identity in item["identities"]
                ),
                *(
                    _datetime(run["observed_at"], "run observation time")
                    for run in item["run_observations"]
                ),
                *(
                    _datetime(observation["observed_at"], "observation time")
                    for observation in item["observations"]
                ),
            )
            decisions = {}
            for identity in item["identities"]:
                decision = resolve_dog_identity(
                    source=identity["source"],
                    registration_authority=identity["registration_authority"],
                    registration_id=identity["registration_id"],
                    name=identity["name"],
                )
                self._repository.record_identity_decision(
                    OperationId(identity["operation_id"]),
                    source=identity["source"],
                    source_alias=identity["source_alias"],
                    name=identity["name"],
                    decision=decision,
                    at=_datetime(identity["decided_at"], "identity decision time"),
                )
                if decision.dog_id is None:
                    raise ServiceUnavailable("runtime runner identity is ambiguous")
                decisions[identity["source_alias"]] = decision.dog_id
            for run in item["run_observations"]:
                checksum = ArtifactChecksum(run["checksum"])
                self._artifacts.verify(checksum)
                self._repository.ingest_run(
                    RunObservation(
                        OperationId(run["operation_id"]),
                        decisions[run["identity_source_alias"]],
                        date.fromisoformat(run["local_racing_date"]),
                        run["source"],
                        checksum,
                        _datetime(run["observed_at"], "run observation time"),
                        run["starts"],
                        run["wins"],
                    ),
                    authoritative=run["authoritative"],
                )
            for observation in item["observations"]:
                checksum = ArtifactChecksum(observation["checksum"])
                self._artifacts.verify(checksum)
                field = EvidenceField(observation["field"])
                self._repository.record_field_evidence(
                    FieldEvidence(
                        OperationId(observation["operation_id"]),
                        race_id,
                        field,
                        EvidenceAuthority(observation["authority"]),
                        self._field_value(field, observation["value"]),
                        observation["source"],
                        checksum,
                        _datetime(observation["observed_at"], "observation time"),
                    )
                )
            self._store.advance_race(
                _operation(f"{command.operation_id}:{race_id}:card"),
                race_id,
                RaceState.CARD_COLLECTED,
                card_at,
            )

    def _odds(self, command: ApplicationCommand, at: datetime) -> None:
        del at
        inputs = self._race_inputs(command)
        prepared: list[tuple[RaceId, tuple[OddsObservation, ...]]] = []
        for race_id, item, row in inputs:
            scheduled_jump = _datetime(row["scheduled_jump"], "scheduled jump")
            try:
                discovery_at = _datetime(
                    item["odds_attempts"][0]["scheduled_due_at"],
                    "odds discovery due time",
                )
            except (IndexError, KeyError, TypeError) as error:
                raise ServiceUnavailable("runtime adaptive odds history is incomplete") from error
            if not _adaptive_odds_history_complete(
                item["odds_attempts"],
                discovery_at=discovery_at,
                scheduled_jump=scheduled_jump,
                cutoff=scheduled_jump,
            ):
                raise ServiceUnavailable(
                    "runtime adaptive odds history violates the versioned bounded timing policy"
                )
            observations: list[OddsObservation] = []
            try:
                for attempt in item["odds_attempts"]:
                    artifact = (
                        ArtifactChecksum(attempt["artifact_checksum"])
                        if attempt["artifact_checksum"] is not None
                        else None
                    )
                    mapping = (
                        ArtifactChecksum(attempt["runner_mapping_checksum"])
                        if attempt["runner_mapping_checksum"] is not None
                        else None
                    )
                    if artifact is not None:
                        self._artifacts.verify(artifact)
                    if mapping is not None:
                        self._artifacts.verify(mapping)
                    observations.append(
                        OddsObservation(
                            operation_id=OperationId(attempt["operation_id"]),
                            race_id=race_id,
                            source=attempt["source"],
                            scheduled_due_at=_datetime(
                                attempt["scheduled_due_at"], "odds scheduled_due_at"
                            ),
                            attempted_at=_datetime(attempt["attempted_at"], "odds attempted_at"),
                            timing_policy=attempt["timing_policy"],
                            status=OddsAttemptStatus(attempt["status"]),
                            artifact_checksum=artifact,
                            runner_mapping_checksum=mapping,
                            error=attempt["error"],
                        )
                    )
            except (ArtifactStoreError, KeyError, TypeError, ValueError) as error:
                raise ServiceUnavailable(
                    "runtime adaptive odds attempt is malformed or unauthenticated"
                ) from error
            prepared.append((race_id, tuple(observations)))
        for race_id, observations in prepared:
            self._store.advance_race(
                _operation(f"{command.operation_id}:{race_id}:odds"),
                race_id,
                RaceState.COLLECTING_ODDS,
                observations[0].attempted_at,
            )
            for observation in observations:
                self._repository.record_odds_attempt(observation)

    def _close_and_seal(self, command: ApplicationCommand, at: datetime) -> None:
        del at
        item = self._document(command)
        closed_at = max(_datetime(race["seal"]["sealed_at"], "sealed_at") for race in item["races"])
        for race_id, item, row in self._race_inputs(command):
            seal = item["seal"]
            observations = tuple(
                FieldObservation(
                    EvidenceField(observation["field"]),
                    self._field_value(EvidenceField(observation["field"]), observation["value"]),
                    EvidenceAuthority(observation["authority"]),
                    EvidenceField(observation["field"]).critical,
                    observation["source"],
                    ArtifactChecksum(observation["checksum"]),
                )
                for observation in item["observations"]
            )
            sources = {value.source: value.artifact_checksum for value in observations}
            EvidenceSealer(self._store, self._artifacts).seal(
                operation_id=OperationId(seal["operation_id"]),
                race_id=race_id,
                source_artifacts=sources,
                observations=observations,
                scheduled_jump=_datetime(row["scheduled_jump"], "scheduled jump"),
                buffer=timedelta(seconds=seal["buffer_seconds"]),
                schema_version=seal["schema_version"],
                normalization_version=seal["normalization_version"],
                sealed_at=_datetime(seal["sealed_at"], "sealed_at"),
            )
            self._store.advance_race(
                _operation(f"{command.operation_id}:{race_id}:await-close"),
                race_id,
                RaceState.AWAITING_DAY_CLOSE,
                closed_at,
            )
        self._store.close_racing_day(
            _operation(f"{command.operation_id}:close"),
            self._day(self._document(command)),
            closed_at,
        )

    def _champion_loader(self) -> ChampionLoader:
        return ChampionLoader(
            self._store,
            self._artifacts,
            deserializer=LinearStrengthModel.from_bytes,
        )

    def _predict(self, command: ApplicationCommand, at: datetime) -> None:
        item = self._document(command)
        champion = item["champion"]
        loaded = self._champion_loader().load_registered(
            champion["bundle_id"], ArtifactChecksum(champion["bundle_checksum"])
        )
        authority = ForecastingAuthority(self._store)
        forecast_service = CanonicalForecastService(self._champion_loader(), self._artifacts)
        self._authorize_day_cohort(command, at)
        for race_id, race, _ in self._race_inputs(command):
            prediction = race["prediction"]
            with self._store._connect() as db:
                begun = db.execute(
                    "SELECT begun_at FROM prediction_begins " "WHERE race_id=? AND operation_id=?",
                    (str(race_id), prediction["begin_operation_id"]),
                ).fetchone()
            prediction_at = (
                at
                if begun is None
                else _datetime(begun["begun_at"], "durable prediction begin time")
            )
            predictor = _ExactRegisteredDeferredPredictor(
                forecast_service,
                self._artifacts,
                loaded,
                prediction_at,
            )
            authority.begin_prediction(
                OperationId(prediction["begin_operation_id"]),
                race_id,
                prediction_at,
            )
            outcome = authority.predict(
                OperationId(prediction["operation_id"]),
                race_id,
                prediction["prediction_id"],
                predictor,
                prediction_at,
            )
            if outcome.status != "committed":
                raise ServiceUnavailable("canonical champion prediction did not commit")
            with self._store._connect() as db:
                committed = db.execute(
                    "SELECT artifact_checksum FROM deferred_predictions "
                    "WHERE race_id=? AND prediction_id=?",
                    (str(race_id), prediction["prediction_id"]),
                ).fetchone()
            if committed is None:
                raise ServiceUnavailable("canonical prediction authority is unavailable")
            prediction_document = json.loads(
                self._artifacts.read(ArtifactChecksum(committed["artifact_checksum"]))
            )
            if (
                prediction_document.get("provenance", {}).get("artifact_checksum")
                != champion["bundle_checksum"]
            ):
                raise ServiceUnavailable("prediction used a different champion")
            self._record_day_cohort_forecasts(
                command,
                race_id,
                outcome,
                forecast_service,
            )
        if loaded.bundle.bundle_id != champion[
            "bundle_id"
        ] or loaded.bundle.bundle_checksum != ArtifactChecksum(champion["bundle_checksum"]):
            raise ServiceUnavailable("prediction champion identity changed")

    def _authorize_day_cohort(
        self,
        command: ApplicationCommand,
        at: datetime,
    ) -> None:
        cohort = self._document(command)["forecast_cohort"]
        race_inputs = self._race_inputs(command)
        race_by_source = {
            race["source_race_id"]: str(identity) for identity, race, _ in race_inputs
        }
        members = tuple(
            DayForecastCohortMember(
                member["role"],
                member["bundle_id"],
                ArtifactChecksum(member["bundle_checksum"]),
                OperationId(member["service_run_id"]),
                tuple(
                    (
                        race_by_source[binding["source_race_id"]],
                        OperationId(binding["operation_id"]),
                    )
                    for binding in member["forecast_operations"]
                ),
            )
            for member in cohort["members"]
        )
        with self._store._connect() as db:
            durable = db.execute(
                "SELECT authorized_at FROM phase7_day_forecast_cohorts " "WHERE operation_id=?",
                (cohort["authorization_operation_id"],),
            ).fetchone()
        authorized_at = (
            at
            if durable is None
            else _datetime(
                durable["authorized_at"],
                "durable cohort authorization time",
            )
        )
        self._authority.authorize_day_forecast_cohort(
            OperationId(cohort["authorization_operation_id"]),
            racing_day_id=command.racing_day_id,
            assignment_id=cohort["assignment_id"],
            members=members,
            at=authorized_at,
        )
        evaluation_authority = EvaluationAuthority(self._store, self._artifacts)
        for member in cohort["members"]:
            evaluation_authority.begin_run(
                OperationId(member["service_run_id"]),
                run_kind="forecast_service",
                started_at=authorized_at,
            )

    def _record_day_cohort_forecasts(
        self,
        command: ApplicationCommand,
        race_id: RaceId,
        outcome: Any,
        service: CanonicalForecastService,
    ) -> None:
        if outcome.artifact_checksum is None:
            raise ServiceUnavailable("prediction artifact is unavailable")
        item = self._document(command)
        cohort = item["forecast_cohort"]
        with self._store._connect() as db:
            prediction = db.execute(
                "SELECT evidence_checksum,computed_at FROM deferred_predictions " "WHERE race_id=?",
                (str(race_id),),
            ).fetchone()
        if prediction is None:
            raise ServiceUnavailable("day cohort forecast lacks deferred prediction")
        race_inputs = self._race_inputs(command)
        for member in cohort["members"]:
            bundle_id = member["bundle_id"]
            bundle_checksum = ArtifactChecksum(member["bundle_checksum"])
            service_run_id = OperationId(member["service_run_id"])
            source_race_id = next(
                race["source_race_id"] for identity, race, _ in race_inputs if identity == race_id
            )
            forecast_operation = OperationId(
                next(
                    binding["operation_id"]
                    for binding in member["forecast_operations"]
                    if binding["source_race_id"] == source_race_id
                )
            )
            service.persist_evaluation_forecast(
                forecast_operation,
                service_run_id=service_run_id,
                race_id=str(race_id),
                bundle_id=bundle_id,
                bundle_checksum=bundle_checksum,
                evidence_checksum=ArtifactChecksum(prediction["evidence_checksum"]),
                computed_at=datetime.fromisoformat(prediction["computed_at"]),
                computation_id=(f"phase7:{command.operation_id}:{race_id}:{member['role']}"),
            )

    def _results(self, command: ApplicationCommand, at: datetime) -> None:
        authority = ForecastingAuthority(self._store)
        prepared: list[
            tuple[
                RaceId,
                Mapping[str, Any],
                datetime,
                datetime,
                ArtifactChecksum,
                Mapping[str, Any],
            ]
        ] = []
        for race_id, race, _ in self._race_inputs(command):
            result = race["result"]
            attempted_at = _datetime(result["attempted_at"], "result attempted_at")
            if attempted_at > at:
                raise ServiceUnavailable(
                    "official result attempt is later than trusted command time"
                )
            checksum = ArtifactChecksum(result["source_checksum"])
            content = self._artifacts.read(checksum)
            try:
                outcome = json.loads(content)
            except json.JSONDecodeError as error:
                raise ServiceUnavailable("official result source is malformed") from error
            if (
                type(outcome) is not dict
                or type(result["source"]) is not str
                or not result["source"].strip()
                or outcome.get("source") != result["source"]
            ):
                raise ServiceUnavailable("official result source provenance disagrees")
            observed_at = _datetime(
                outcome.get("provenance", {}).get("observed_at"),
                "result observed_at",
            )
            published_at = _datetime(outcome.get("published_at"), "result published_at")
            if not _official_result_timeline_valid(
                published_at,
                observed_at,
                attempted_at,
                at,
                result["timing_policy"],
            ):
                raise ServiceUnavailable(
                    "official result timeline violates the versioned bounded ordering policy"
                )
            deadline = _datetime(result["deadline"], "result deadline")
            order = outcome.get("order")
            if (
                type(result["attempt_id"]) is not str
                or not result["attempt_id"].strip()
                or type(result["max_attempts"]) is not int
                or isinstance(result["max_attempts"], bool)
                or result["max_attempts"] <= 0
                or not isinstance(order, list)
                or not order
                or any(type(box) is not int or box <= 0 for box in order)
                or len(set(order)) != len(order)
            ):
                raise ServiceUnavailable("official result attempt contract is malformed")
            try:
                encoded = _canonical(outcome)
            except (TypeError, ValueError) as error:
                raise ServiceUnavailable("official result source is not exact JSON") from error
            if json.loads(encoded) != outcome:
                raise ServiceUnavailable("official result source is not exact JSON")
            prepared.append((race_id, result, attempted_at, deadline, checksum, outcome))
        for race_id, result, attempted_at, deadline, checksum, outcome in prepared:
            authority.open_results(OperationId(result["open_operation_id"]), race_id, attempted_at)
            status = authority.record_result_attempt(
                OperationId(result["operation_id"]),
                race_id,
                result["attempt_id"],
                at=attempted_at,
                max_attempts=result["max_attempts"],
                deadline=deadline,
                artifact_checksum=checksum,
                outcome=outcome,
            )
            if status != "collected":
                raise ServiceUnavailable("official result did not collect")

    def _join(self, command: ApplicationCommand, at: datetime) -> None:
        del at
        authority = ForecastingAuthority(self._store)
        loader = self._champion_loader()
        champion = self._document(command)["champion"]
        loaded = loader.load_registered(
            champion["bundle_id"], ArtifactChecksum(champion["bundle_checksum"])
        )
        schema = loaded.bundle.component("feature_schema")
        missingness = loaded.bundle.component("missingness_policy")
        schema_bytes = self._artifacts.read(schema.checksum)
        missingness_bytes = self._artifacts.read(missingness.checksum)
        for race_id, race, _ in self._race_inputs(command):
            example = race["training_example"]
            result_checksum = ArtifactChecksum(race["result"]["source_checksum"])
            authority.join_training_example(
                OperationId(example["join_operation_id"]),
                race_id,
                example["phase3_example_id"],
                result_checksum,
                eligible=True,
                reason=None,
                at=_datetime(example["joined_at"], "joined_at"),
            )
            TrainingCorpusAuthority(self._store, self._artifacts).build_forward_example(
                OperationId(example["build_operation_id"]),
                phase3_example_id=example["phase3_example_id"],
                example_id=example["canonical_example_id"],
                schema_bytes=schema_bytes,
                schema_checksum=schema.checksum,
                missingness_bytes=missingness_bytes,
                missingness_checksum=missingness.checksum,
                joined_at=_datetime(example["joined_at"], "joined_at"),
            )

    def _reconcile(self, command: ApplicationCommand, at: datetime) -> None:
        item = self._document(command)
        at = max(
            _datetime(
                race["training_example"]["joined_at"],
                "training example joined_at",
            )
            for race in item["races"]
        ) + timedelta(microseconds=1)
        with self._store._connect() as db:
            progress = [
                dict(row)
                for row in db.execute(
                    "SELECT phase_ordinal,phase_name,lease_generation,"
                    "command_operation_id,result_checksum FROM phase7_scheduler_progress "
                    "WHERE racing_day_id=? ORDER BY phase_ordinal",
                    (command.racing_day_id,),
                )
            ]
            planned = [
                row[0]
                for row in db.execute(
                    "SELECT command_operation_id FROM phase7_day_command_plan "
                    "WHERE racing_day_id=? ORDER BY phase_ordinal",
                    (command.racing_day_id,),
                )
            ]
            rejected = [
                row[0]
                for row in db.execute(
                    "SELECT operation_id FROM phase7_rejected_result_commands "
                    "WHERE racing_day_id=? ORDER BY rejected_at,operation_id",
                    (command.racing_day_id,),
                )
            ]
            adoption = db.execute(
                "SELECT operation_id,lease_generation FROM phase7_day_plan_adoptions "
                "WHERE racing_day_id=? ORDER BY lease_generation DESC LIMIT 1",
                (command.racing_day_id,),
            ).fetchone()
        if adoption is None:
            raise OperationalRejected("restart evidence requires a durable plan adoption")
        recovered = [
            row for row in progress if row["lease_generation"] < adoption["lease_generation"]
        ]
        input_checksum = ArtifactChecksum(item["determinism_input_checksum"])
        first = _operation(f"{command.operation_id}:determinism:first")
        replay = _operation(f"{command.operation_id}:determinism:replay")
        first_output = self._authority.record_determinism_execution(
            first,
            racing_day_id=command.racing_day_id,
            release_id=self._release_id,
            input_checksum=input_checksum,
            at=at,
        )
        replay_output = self._authority.record_determinism_execution(
            replay,
            racing_day_id=command.racing_day_id,
            release_id=self._release_id,
            input_checksum=input_checksum,
            at=at,
        )
        checks = {
            "restart": {
                "adoption_operation_id": adoption["operation_id"],
                "lease_generation": adoption["lease_generation"],
                "recovered_phase": len(recovered),
                "state_checksum": str(
                    ArtifactChecksum("sha256:" + hashlib.sha256(_canonical(recovered)).hexdigest())
                ),
            },
            "ordering": {
                "phase_operations": planned,
                "result_before_prediction_rejection_operations": rejected,
            },
            "determinism": {
                "input_checksum": str(input_checksum),
                "first_execution_operation_id": str(first),
                "output_checksum": str(first_output),
                "replay_execution_operation_id": str(replay),
                "replay_output_checksum": str(replay_output),
            },
        }
        evidence: dict[str, ArtifactChecksum] = {}
        for kind in ("restart", "ordering", "determinism"):
            proof = {
                "schema_version": "phase7-check-proof-v1",
                "evidence_kind": kind,
                "racing_day_id": command.racing_day_id,
                "release_id": self._release_id,
                "checks": checks[kind],
            }
            checksum = self._artifacts.put(
                _canonical(proof), media_type="application/json"
            ).checksum
            evidence[kind] = checksum
            self._authority.record_operational_evidence(
                _operation(f"{command.operation_id}:evidence:{kind}"),
                checksum=checksum,
                evidence_kind=kind,
                racing_day_id=command.racing_day_id,
                release_id=self._release_id,
                checks=checks[kind],
                at=at,
            )
        self._authority.reconcile_day(
            _operation(f"{command.operation_id}:reconcile"),
            racing_day_id=command.racing_day_id,
            release_id=self._release_id,
            restart_checksum=evidence["restart"],
            ordering_checksum=evidence["ordering"],
            determinism_checksum=evidence["determinism"],
            at=at,
        )

    def _request_training(self, command: ApplicationCommand, at: datetime) -> None:
        payload = command.payload
        if not isinstance(payload, RequestTraining):
            raise ServiceUnavailable("training command payload is invalid")
        training = self._document(command)["training_request"]
        at = max(
            _datetime(
                race["training_example"]["joined_at"],
                "training example joined_at",
            )
            for race in self._document(command)["races"]
        ) + timedelta(microseconds=2)
        run_id = OperationId(training["service_run_id"])
        EvaluationAuthority(self._store, self._artifacts).begin_run(
            run_id, run_kind="forecast_service", started_at=at
        )
        self._authority.authorize_training_request(
            payload.authorization_operation_id,
            racing_day_id=command.racing_day_id,
            request_id=payload.request_id,
            request_operation_id=payload.request_operation_id,
            at=at,
        )
        CanonicalForecastService(self._champion_loader(), self._artifacts).emit_training_request(
            payload.request_operation_id,
            request_id=payload.request_id,
            reason="complete reconciled Racing Day",
            requested_at=at,
            service_run_id=run_id,
        )
        self._authority.bind_training_request(
            payload.binding_operation_id,
            racing_day_id=command.racing_day_id,
            request_id=payload.request_id,
            request_operation_id=payload.request_operation_id,
            at=at,
        )


def checked_in(
    configuration: ReleaseConfiguration,
    store: SQLiteOperationsStore,
    artifacts: LocalArtifactStore,
) -> ImmutableInputRuntimeAdapter:
    """Build the sole production adapter from one explicit immutable input manifest."""
    return ImmutableInputRuntimeAdapter(configuration, store, artifacts)


def unavailable(*_args: object, **_kwargs: object) -> None:
    """Compatibility binding used only to prove startup fails closed."""
    raise RuntimeError("release runtime adapter is deliberately unavailable")
