from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
import threading
import unittest
from contextlib import redirect_stderr
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from race_collection.artifacts import ArtifactStoreError, LocalArtifactStore
from race_collection.collection import CollectionRepository
from race_collection.domain import (
    ArtifactChecksum,
    OperationId,
    ProgrammeRaceCandidate,
    RaceId,
    RacingDay,
    RacingDayId,
    next_odds_attempt_at,
)
from race_collection.evaluation import (
    EvaluationAuthority,
    PromotionAuthority,
    PromotionPolicy,
    PromotionRejected,
)
from race_collection.forecast_service import CanonicalForecastService
from race_collection.model_bundle import ChampionLoader
from race_collection.operational import (
    COMMAND_PHASES,
    ApplicationCommand,
    CloseAndSeal,
    ClosedCommandDispatcher,
    CollectAdaptiveOdds,
    CollectCardsAndForm,
    CollectResults,
    CommitDeferredPrediction,
    DiscoverProgramme,
    JoinTrainingExamples,
    OperationalAuthority,
    OperationalRejected,
    PhaseHandlerRegistration,
    RaceCollectionService,
    ReconcileRacingDay,
    ReleaseConfiguration,
    ReleaseManifest,
    RequestTraining,
    _adaptive_odds_history_complete,
    _checksum,
    _derived_operation_id,
    _odds_snapshot_mismatches,
    _safe_operational_path,
)
from race_collection.operations import (
    BarrierNotSatisfied,
    ConflictingOperation,
    SQLiteOperationsStore,
    iso_timestamp,
)
from race_collection.recovery import (
    RELATIONAL_ARTIFACT_REFERENCES,
    RecoveryAuthority,
    RecoveryRejected,
    artifact_inventory,
)
from race_collection.service import (
    RacingDayCycle,
    ServiceComposition,
    ServiceUnavailable,
    compose,
    main,
)

NOW = datetime(2026, 7, 22, tzinfo=timezone.utc)
PROGRAMME = ArtifactChecksum("sha256:" + "1" * 64)


def operation(number: int) -> OperationId:
    return OperationId(f"op_{number:032x}")


def command_payload(phase: str):
    return {
        "discover_programme": DiscoverProgramme("official", PROGRAMME),
        "collect_cards_and_form": CollectCardsAndForm(),
        "collect_adaptive_odds": CollectAdaptiveOdds(),
        "close_and_seal": CloseAndSeal(),
        "deferred_prediction": CommitDeferredPrediction(),
        "collect_results": CollectResults(),
        "join_training_examples": JoinTrainingExamples(),
        "reconcile": ReconcileRacingDay(),
        "request_training": RequestTraining(
            "request-day", operation(880001), operation(880002), operation(880003)
        ),
    }[phase]


class QuarantinedDayHandlers:
    """Real public-API producer fixture for one explicitly excluded expected race."""

    def __init__(
        self,
        authority: OperationalAuthority,
        *,
        day: RacingDay,
        programme_checksum: ArtifactChecksum,
        release_id: str,
        request: RequestTraining,
        base: int,
    ):
        self.authority = authority
        self.store = authority.store
        self.artifacts = authority.artifacts
        self.repository = CollectionRepository(self.store)
        self.day = day
        self.programme_checksum = programme_checksum
        self.release_id = release_id
        self.request = request
        self.base = base
        self.at = NOW
        self.race_id: RaceId | None = None
        self.executions: dict[str, int] = {}

    def _record(self, command: ApplicationCommand) -> None:
        self.executions[command.phase] = self.executions.get(command.phase, 0) + 1

    def discover(self, command: ApplicationCommand, at: datetime) -> None:
        self._record(command)
        self.race_id = self.repository.record_expected_race(
            operation(self.base + 1),
            self.day,
            ProgrammeRaceCandidate(
                "official",
                f"fixture-{self.base}",
                "Synthetic",
                1,
                at + timedelta(seconds=30),
            ),
            self.programme_checksum,
            at,
        )

    def cards(self, command: ApplicationCommand, at: datetime) -> None:
        self._record(command)
        assert self.race_id is not None
        self.repository.quarantine_collection(
            operation(self.base + 2),
            self.race_id,
            stage="identity",
            code="terminal_identity_exclusion",
            details="realistic synthetic source ambiguity",
            at=at,
        )

    def excluded(self, command: ApplicationCommand, at: datetime) -> None:
        self._record(command)

    def close(self, command: ApplicationCommand, at: datetime) -> None:
        self._record(command)
        self.store.close_racing_day(operation(self.base + 3), self.day, at)

    def reconcile(self, command: ApplicationCommand, at: datetime) -> None:
        self._record(command)
        with self.store._connect() as db:
            progress = [
                dict(row)
                for row in db.execute(
                    "SELECT phase_ordinal,phase_name,lease_generation,"
                    "command_operation_id,result_checksum "
                    "FROM phase7_scheduler_progress WHERE racing_day_id=? "
                    "ORDER BY phase_ordinal",
                    (str(self.day.id),),
                )
            ]
            planned = [
                row[0]
                for row in db.execute(
                    "SELECT command_operation_id FROM phase7_day_command_plan "
                    "WHERE racing_day_id=? ORDER BY phase_ordinal",
                    (str(self.day.id),),
                )
            ]
            rejection_operations = [
                row[0]
                for row in db.execute(
                    "SELECT operation_id FROM phase7_rejected_result_commands "
                    "WHERE racing_day_id=? ORDER BY rejected_at,operation_id",
                    (str(self.day.id),),
                )
            ]
        input_artifact = self.artifacts.put(b"input", media_type="application/json")
        first_execution = operation(self.base + 80)
        replay_execution = operation(self.base + 81)
        output_checksum = self.authority.record_determinism_execution(
            first_execution,
            racing_day_id=str(self.day.id),
            release_id=self.release_id,
            input_checksum=input_artifact.checksum,
            at=at,
        )
        replay_output_checksum = self.authority.record_determinism_execution(
            replay_execution,
            racing_day_id=str(self.day.id),
            release_id=self.release_id,
            input_checksum=input_artifact.checksum,
            at=at,
        )
        with self.store._connect() as db:
            adoption = db.execute(
                "SELECT operation_id,lease_generation FROM phase7_day_plan_adoptions "
                "WHERE racing_day_id=? ORDER BY lease_generation DESC LIMIT 1",
                (str(self.day.id),),
            ).fetchone()
        if adoption is None:
            raise OperationalRejected("reconciliation fixture requires a durable restart adoption")
        recovered = [
            row for row in progress if row["lease_generation"] < adoption["lease_generation"]
        ]
        checks = {
            "restart": {
                "adoption_operation_id": adoption["operation_id"],
                "lease_generation": adoption["lease_generation"],
                "recovered_phase": len(recovered),
                "state_checksum": str(_checksum(recovered)),
            },
            "ordering": {
                "phase_operations": planned,
                "result_before_prediction_rejection_operations": rejection_operations,
            },
            "determinism": {
                "input_checksum": str(input_artifact.checksum),
                "first_execution_operation_id": str(first_execution),
                "output_checksum": str(output_checksum),
                "replay_execution_operation_id": str(replay_execution),
                "replay_output_checksum": str(replay_output_checksum),
            },
        }
        evidence = {}
        for offset, kind in enumerate(("restart", "ordering", "determinism"), 10):
            proof = {
                "schema_version": "phase7-check-proof-v1",
                "evidence_kind": kind,
                "racing_day_id": str(self.day.id),
                "release_id": self.release_id,
                "checks": checks[kind],
            }
            evidence[kind] = self.artifacts.put(
                json.dumps(proof, sort_keys=True, separators=(",", ":")).encode(),
                media_type="application/json",
            ).checksum
            self.authority.record_operational_evidence(
                operation(self.base + offset),
                checksum=evidence[kind],
                evidence_kind=kind,
                racing_day_id=str(self.day.id),
                release_id=self.release_id,
                checks=checks[kind],
                at=at,
            )
        self.authority.reconcile_day(
            operation(self.base + 20),
            racing_day_id=str(self.day.id),
            release_id=self.release_id,
            restart_checksum=evidence["restart"],
            ordering_checksum=evidence["ordering"],
            determinism_checksum=evidence["determinism"],
            at=at,
        )

    def request_training(self, command: ApplicationCommand, at: datetime) -> None:
        self._record(command)
        run_id = operation(self.base + 30)
        EvaluationAuthority(self.store, self.artifacts).begin_run(
            run_id, run_kind="forecast_service", started_at=at
        )
        self.authority.authorize_training_request(
            self.request.authorization_operation_id,
            racing_day_id=str(self.day.id),
            request_id=self.request.request_id,
            request_operation_id=self.request.request_operation_id,
            at=at,
        )
        service = CanonicalForecastService(
            ChampionLoader(self.store, self.artifacts, deserializer=lambda content: content),
            self.artifacts,
        )
        service.emit_training_request(
            self.request.request_operation_id,
            request_id=self.request.request_id,
            reason="complete reconciled synthetic Racing Day",
            requested_at=at,
            service_run_id=run_id,
        )
        self.authority.bind_training_request(
            self.request.binding_operation_id,
            racing_day_id=str(self.day.id),
            request_id=self.request.request_id,
            request_operation_id=self.request.request_operation_id,
            at=at,
        )

    def registrations(self):
        return (
            PhaseHandlerRegistration(DiscoverProgramme, self.discover),
            PhaseHandlerRegistration(CollectCardsAndForm, self.cards),
            PhaseHandlerRegistration(CollectAdaptiveOdds, self.excluded),
            PhaseHandlerRegistration(CloseAndSeal, self.close),
            PhaseHandlerRegistration(CommitDeferredPrediction, self.excluded),
            PhaseHandlerRegistration(CollectResults, self.excluded),
            PhaseHandlerRegistration(JoinTrainingExamples, self.excluded),
            PhaseHandlerRegistration(ReconcileRacingDay, self.reconcile),
            PhaseHandlerRegistration(RequestTraining, self.request_training),
        )

    def dispatcher(self) -> ClosedCommandDispatcher:
        return ClosedCommandDispatcher(self.registrations())


class Phase7OperationalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        root = Path(self.temporary.name)
        self.store = SQLiteOperationsStore(root / "operations.sqlite3")
        self.store.migrate()
        self.artifacts = LocalArtifactStore(root / "artifacts")
        self.authority = OperationalAuthority(self.store, self.artifacts, clock=lambda: NOW)
        EvaluationAuthority(self.store, self.artifacts).register_policy(
            operation(990000), PromotionPolicy(), NOW - timedelta(days=1)
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def manifest(self, release_id: str = "candidate") -> ReleaseManifest:
        content = json.dumps(
            self.configuration().document(), sort_keys=True, separators=(",", ":")
        ).encode()
        return ReleaseManifest(
            "phase7-release-v1",
            release_id,
            "6cd5dacfe83719cbbc376a265829e84593eafb68",
            ArtifactChecksum("sha256:" + hashlib.sha256(content).hexdigest()),
            28,
            "canonical-artifacts-v1",
            "phase6-promotion-v1",
            ("runner-win-probability-v1",),
            "/opt/race-collection/current",
        )

    def configuration(self) -> ReleaseConfiguration:
        return ReleaseConfiguration(
            "phase7-config-v1",
            "/opt/race-collection/current",
            "/var/lib/race-collection/artifacts",
            "/var/lib/race-collection/operations.sqlite3",
            ("official",),
            "adaptive-odds-v1",
            "phase6-promotion-v1",
            ("runner-win-probability-v1",),
            "race_collection.runtime_adapters:unavailable",
            ArtifactChecksum("sha256:" + "9" * 64),
        )

    def register_config(self, number: int) -> None:
        self.authority.register_configuration(operation(number), self.configuration(), NOW)

    def trusted_scheduler_fixture(
        self,
        base: int = 800000,
        *,
        release_id: str = "scheduler-release",
        local_date: date | None = None,
        predecessor: str | None = None,
        planning_clock=None,
        recovered_after_discovery: bool = False,
    ):
        programme = self.artifacts.put(
            b'{"programme":"synthetic"}', media_type="application/json"
        ).checksum
        with self.store._connect() as db:
            registered = db.execute(
                "SELECT 1 FROM phase7_release_manifests WHERE release_id=?",
                (release_id,),
            ).fetchone()
        if registered is None:
            with self.store._connect() as db:
                configured = db.execute(
                    "SELECT 1 FROM phase7_release_configurations LIMIT 1"
                ).fetchone()
            if configured is None:
                self.register_config(base)
            self.authority.register_release(operation(base + 1), self.manifest(release_id), NOW)
        day = RacingDay(
            RacingDayId(f"day_{base:032x}"),
            NOW.date() if local_date is None else local_date,
            "UTC",
            NOW,
        )
        self.store.create_racing_day(operation(base + 2), day)
        if local_date is not None or predecessor is not None:
            with self.store._operation(
                operation(base + 5),
                "synthetic_schedule_primitive",
                {"day": str(day.id), "predecessor": predecessor},
            ) as (db, _):
                db.execute(
                    "INSERT INTO phase6_racing_day_schedule VALUES(?,?,?,?,?)",
                    (
                        str(day.id),
                        predecessor,
                        str(programme),
                        NOW.isoformat(),
                        str(operation(base + 5)),
                    ),
                )
        request = RequestTraining(
            f"request-{base}",
            operation(base + 39),
            operation(base + 40),
            operation(base + 41),
        )
        commands = (
            ApplicationCommand(
                operation(base + 50),
                str(day.id),
                DiscoverProgramme("official", programme),
            ),
            ApplicationCommand(operation(base + 51), str(day.id), CollectCardsAndForm()),
            ApplicationCommand(operation(base + 52), str(day.id), CollectAdaptiveOdds()),
            ApplicationCommand(operation(base + 53), str(day.id), CloseAndSeal()),
            ApplicationCommand(operation(base + 54), str(day.id), CommitDeferredPrediction()),
            ApplicationCommand(operation(base + 55), str(day.id), CollectResults()),
            ApplicationCommand(operation(base + 56), str(day.id), JoinTrainingExamples()),
            ApplicationCommand(operation(base + 57), str(day.id), ReconcileRacingDay()),
            ApplicationCommand(operation(base + 58), str(day.id), request),
        )
        run_at = NOW - timedelta(days=1) + timedelta(microseconds=base)
        generation = self.authority.acquire_lease(
            operation(base + 3),
            owner="synthetic-service",
            token=f"lease-{base}",
            now=run_at,
            ttl=timedelta(microseconds=1),
        )
        OperationalAuthority(
            self.store,
            self.artifacts,
            clock=(lambda: run_at) if planning_clock is None else planning_clock,
        ).plan_racing_day(
            operation(base + 4),
            racing_day_id=str(day.id),
            lease_token=f"lease-{base}",
            lease_generation=generation,
            commands=commands,
            at=run_at,
        )
        handlers = QuarantinedDayHandlers(
            self.authority,
            day=day,
            programme_checksum=programme,
            release_id=release_id,
            request=request,
            base=base + 100,
        )
        handlers.at = run_at
        first_authority = OperationalAuthority(
            self.store,
            self.artifacts,
            command_executor=handlers.dispatcher(),
            clock=lambda: handlers.at,
        )
        if not recovered_after_discovery:
            return (
                RaceCollectionService(
                    first_authority, token=f"lease-{base}", generation=generation
                ),
                handlers,
                commands,
            )
        first_service = RaceCollectionService(
            first_authority, token=f"lease-{base}", generation=generation
        )
        first_service.advance(
            operation(base + 71),
            racing_day_id=str(day.id),
            phase=commands[0].phase,
            now=handlers.at,
            command=commands[0],
        )
        handlers.at = run_at + timedelta(microseconds=2)
        recovered_generation = self.authority.acquire_lease(
            operation(base + 6),
            owner="synthetic-recovered-service",
            token=f"recovered-lease-{base}",
            now=handlers.at,
            ttl=timedelta(microseconds=1),
        )
        OperationalAuthority(self.store, self.artifacts, clock=lambda: handlers.at).plan_racing_day(
            operation(base + 7),
            racing_day_id=str(day.id),
            lease_token=f"recovered-lease-{base}",
            lease_generation=recovered_generation,
            commands=commands,
            at=handlers.at,
        )
        authority = OperationalAuthority(
            self.store,
            self.artifacts,
            command_executor=handlers.dispatcher(),
            clock=lambda: handlers.at,
        )
        service = RaceCollectionService(
            authority,
            token=f"recovered-lease-{base}",
            generation=recovered_generation,
        )
        return service, handlers, commands

    def test_adaptive_odds_history_allows_early_cadence_and_rejects_missing_due_attempt(
        self,
    ):
        jump = NOW + timedelta(hours=4)
        cutoff = jump - timedelta(seconds=1)
        legitimate = []
        attempt_at = NOW
        failures = 0
        while attempt_at < cutoff:
            status = "failed" if len(legitimate) == 2 else "succeeded"
            legitimate.append({"attempted_at": attempt_at.isoformat(), "status": status})
            failures = failures + 1 if status == "failed" else 0
            following = next_odds_attempt_at(
                now=attempt_at,
                scheduled_jump=jump,
                last_attempt_at=attempt_at,
                consecutive_failures=failures,
            )
            if following is None or following >= cutoff:
                break
            attempt_at = following
        deltas = {
            datetime.fromisoformat(current["attempted_at"])
            - datetime.fromisoformat(prior["attempted_at"])
            for prior, current in zip(legitimate, legitimate[1:])
        }
        self.assertTrue(
            {
                timedelta(minutes=30),
                timedelta(seconds=30),
                timedelta(minutes=10),
                timedelta(minutes=5),
                timedelta(minutes=1),
            }
            <= deltas
        )
        self.assertTrue(
            _adaptive_odds_history_complete(
                legitimate, discovery_at=NOW, scheduled_jump=jump, cutoff=cutoff
            )
        )
        delayed_first = [dict(row) for row in legitimate]
        delayed_first[0]["attempted_at"] = (NOW + timedelta(seconds=1)).isoformat()
        self.assertFalse(
            _adaptive_odds_history_complete(
                delayed_first, discovery_at=NOW, scheduled_jump=jump, cutoff=cutoff
            )
        )
        missing_later = legitimate[:4] + legitimate[5:]
        premature = [dict(row) for row in legitimate]
        premature.insert(
            1,
            {
                "attempted_at": (NOW + timedelta(minutes=1)).isoformat(),
                "status": "succeeded",
            },
        )
        duplicate = [dict(row) for row in legitimate]
        duplicate.insert(1, dict(duplicate[0]))
        non_monotonic = [dict(row) for row in legitimate]
        non_monotonic[1], non_monotonic[2] = non_monotonic[2], non_monotonic[1]
        post_cutoff = [
            *legitimate,
            {"attempted_at": cutoff.isoformat(), "status": "succeeded"},
        ]
        for invalid in (
            missing_later,
            premature,
            duplicate,
            non_monotonic,
            post_cutoff,
        ):
            with self.subTest(invalid=invalid):
                self.assertFalse(
                    _adaptive_odds_history_complete(
                        invalid,
                        discovery_at=NOW,
                        scheduled_jump=jump,
                        cutoff=cutoff,
                    )
                )

    def test_reconciliation_seal_uses_exact_final_pre_freeze_odds_snapshot(
        self,
    ) -> None:
        raw = self.artifacts.put(b"raw", media_type="application/json").checksum
        normalized = self.artifacts.put(b"normalized", media_type="application/json").checksum
        first = self.artifacts.put(b"first", media_type="application/json").checksum
        final = self.artifacts.put(b"final", media_type="application/json").checksum
        later = self.artifacts.put(b"later", media_type="application/json").checksum
        successes = [
            {"attempted_at": NOW.isoformat(), "artifact_checksum": str(first)},
            {
                "attempted_at": (NOW + timedelta(minutes=1)).isoformat(),
                "artifact_checksum": str(final),
            },
        ]
        seal = {
            "raw_manifest_checksum": str(raw),
            "normalized_checksum": str(normalized),
            "odds_checksum": str(final),
            "frozen_at": (NOW + timedelta(minutes=1)).isoformat(),
        }
        self.assertEqual(_odds_snapshot_mismatches(successes, seal, self.artifacts), ())
        wrong = dict(seal, odds_checksum=str(first))
        self.assertEqual(
            _odds_snapshot_mismatches(successes, wrong, self.artifacts),
            ("final_pre_jump_odds_checksum_mismatch",),
        )
        contaminated = successes + [
            {
                "attempted_at": (NOW + timedelta(minutes=2)).isoformat(),
                "artifact_checksum": str(later),
            }
        ]
        self.assertEqual(
            _odds_snapshot_mismatches(contaminated, seal, self.artifacts),
            ("post_freeze_odds_contamination",),
        )
        self.artifacts.path_for(final).write_bytes(b"corrupt")
        self.assertEqual(
            _odds_snapshot_mismatches(successes, seal, self.artifacts),
            ("odds_checksum_missing_or_corrupt",),
        )
        self.assertEqual(
            _odds_snapshot_mismatches([], seal, self.artifacts),
            ("odds_checksum_missing_or_corrupt", "final_pre_jump_odds_missing"),
        )

    @staticmethod
    def checks(kind: str, index: int):
        checksum = "sha256:" + f"{index:064x}"
        return {
            "restart": {
                "adoption_operation_id": f"op-{index}-adoption",
                "lease_generation": 2,
                "recovered_phase": 1,
                "state_checksum": checksum,
            },
            "ordering": {
                "phase_operations": [f"op-{index}"],
                "result_before_prediction_rejection_operations": [f"op-{index}"],
            },
            "determinism": {
                "input_checksum": checksum,
                "first_execution_operation_id": f"op-{index}-first",
                "output_checksum": checksum,
                "replay_execution_operation_id": f"op-{index}-replay",
                "replay_output_checksum": checksum,
            },
        }[kind]

    def seed_complete_day(
        self,
        index: int,
        local_date: date,
        *,
        release: str,
        predecessor: str | None = None,
    ) -> str:
        base = 100000 + index * 200
        service, handlers, commands = self.trusted_scheduler_fixture(
            base,
            release_id=release,
            local_date=local_date,
            predecessor=predecessor,
            recovered_after_discovery=True,
        )
        for ordinal, command in enumerate(commands, 1):
            service.advance(
                operation(base + 70 + ordinal),
                racing_day_id=command.racing_day_id,
                phase=command.phase,
                now=handlers.at,
                command=command,
            )
        return commands[0].racing_day_id

    def establish_cutover(
        self,
        base: int,
        observation_index: int,
        boundary_date: date,
        *,
        boundary_timezone: str = "Australia/Melbourne",
    ) -> str:
        self.register_config(base)
        self.authority.register_release(operation(base + 1), self.manifest("legacy"), NOW)
        self.authority.register_release(operation(base + 2), self.manifest("candidate"), NOW)
        self.authority.initialize_legacy_authority(
            operation(base + 3),
            release_id="legacy",
            actor="owner",
            reason="baseline",
            at=NOW,
        )
        OperationalAuthority(
            self.store, self.artifacts, clock=lambda: NOW - timedelta(days=2)
        ).authorize_observation(
            operation(base + 7),
            candidate_release_id="candidate",
            actor="owner",
            reason="candidate observation",
            at=NOW,
        )
        first = self.seed_complete_day(
            observation_index, boundary_date - timedelta(days=2), release="candidate"
        )
        second = self.seed_complete_day(
            observation_index + 1,
            boundary_date - timedelta(days=1),
            release="candidate",
            predecessor=first,
        )
        cutover_at = datetime.combine(
            boundary_date - timedelta(days=1), datetime.min.time(), timezone.utc
        )
        self.authority.cutover_eligibility(
            operation(base + 4), candidate_release_id="candidate", at=cutover_at
        )
        boundary = f"day_{observation_index + 2:032x}"
        programme = ArtifactChecksum(f"sha256:{observation_index + 2:064x}")
        with self.store._operation(operation(base + 5), "synthetic_boundary", {}) as (
            db,
            _,
        ):
            db.execute(
                "INSERT INTO racing_days VALUES(?,?,?,?,NULL)",
                (
                    boundary,
                    boundary_date.isoformat(),
                    boundary_timezone,
                    max(
                        cutover_at + timedelta(hours=12),
                        NOW + timedelta(days=1),
                    ).isoformat(),
                ),
            )
            db.execute(
                "INSERT INTO phase6_racing_day_schedule VALUES(?,?,?,?,?)",
                (
                    boundary,
                    second,
                    str(programme),
                    cutover_at.isoformat(),
                    str(operation(base + 5)),
                ),
            )
        self.authority.activate(
            operation(base + 6),
            release_id="candidate",
            boundary_day_id=boundary,
            actor="owner",
            reason="two-day evidence",
            at=cutover_at,
        )
        return boundary

    def test_single_scheduler_competing_lease_restart_and_stale_fencing(self) -> None:
        self.assertEqual(
            self.authority.acquire_lease(
                operation(1),
                owner="one",
                token="token-one",
                now=NOW,
                ttl=timedelta(minutes=1),
            ),
            1,
        )
        self.assertEqual(
            self.authority.acquire_lease(
                operation(1),
                owner="one",
                token="token-one",
                now=NOW,
                ttl=timedelta(minutes=1),
            ),
            1,
        )
        with self.assertRaises(OperationalRejected):
            self.authority.acquire_lease(
                operation(2),
                owner="two",
                token="token-two",
                now=NOW,
                ttl=timedelta(minutes=1),
            )
        self.assertEqual(
            self.authority.acquire_lease(
                operation(3),
                owner="two",
                token="token-two",
                now=NOW + timedelta(minutes=2),
                ttl=timedelta(minutes=1),
            ),
            2,
        )
        with self.assertRaises(OperationalRejected):
            self.authority.assert_lease("token-one", 1, NOW + timedelta(minutes=2))

    def test_public_command_boundary_rejects_expired_stale_and_takeover_generations(
        self,
    ) -> None:
        service, handlers, commands = self.trusted_scheduler_fixture(39000)
        command = commands[0]
        takeover_at = handlers.at + timedelta(microseconds=1)
        with self.assertRaisesRegex(OperationalRejected, "scheduler ownership"):
            service.authority.execute_application_command(
                command,
                token=service.token,
                generation=service.generation,
                at=takeover_at,
            )
        takeover = service.authority.acquire_lease(
            operation(39200),
            owner="takeover",
            token="takeover-token",
            now=takeover_at,
            ttl=timedelta(minutes=1),
        )
        with self.assertRaisesRegex(OperationalRejected, "scheduler ownership"):
            service.authority.execute_application_command(
                command,
                token=service.token,
                generation=service.generation,
                at=takeover_at,
            )
        with self.assertRaisesRegex(OperationalRejected, "ordered Racing Day plan"):
            service.authority.execute_application_command(
                command,
                token="takeover-token",
                generation=takeover,
                at=takeover_at,
            )

    def test_application_command_without_trusted_executor_cannot_mint_receipt(
        self,
    ) -> None:
        day_id = "day_" + "4" * 32
        with self.store._operation(operation(3990), "seed_command_day", {}) as (db, _):
            db.execute(
                "INSERT INTO racing_days VALUES(?,?,?,?,NULL)",
                (day_id, "2026-08-20", "UTC", NOW.isoformat()),
            )
        command_id = operation(3991)
        generation = self.authority.acquire_lease(
            operation(3992),
            owner="no-executor",
            token="no-executor-token",
            now=NOW,
            ttl=timedelta(hours=1),
        )
        with self.assertRaisesRegex(OperationalRejected, "trusted executor"):
            self.authority.execute_application_command(
                ApplicationCommand(
                    command_id,
                    day_id,
                    DiscoverProgramme("official", PROGRAMME),
                ),
                token="no-executor-token",
                generation=generation,
                at=NOW,
            )
        with self.store._connect() as db:
            self.assertIsNone(
                db.execute(
                    "SELECT 1 FROM operations WHERE operation_id=?", (str(command_id),)
                ).fetchone()
            )

    def test_executor_exception_cannot_mint_operation_or_receipt(self) -> None:
        day_id = "day_" + "3" * 32
        self.store.create_racing_day(
            operation(3980),
            RacingDay(RacingDayId(day_id), NOW.date(), "UTC", NOW),
        )

        secret = "https://user:credential@example.invalid/private?token=secret"

        def fail(command, at):
            raise RuntimeError(secret)

        noop = lambda command, at: None
        dispatcher = ClosedCommandDispatcher(
            (
                PhaseHandlerRegistration(DiscoverProgramme, fail),
                PhaseHandlerRegistration(CollectCardsAndForm, noop),
                PhaseHandlerRegistration(CollectAdaptiveOdds, noop),
                PhaseHandlerRegistration(CloseAndSeal, noop),
                PhaseHandlerRegistration(CommitDeferredPrediction, noop),
                PhaseHandlerRegistration(CollectResults, noop),
                PhaseHandlerRegistration(JoinTrainingExamples, noop),
                PhaseHandlerRegistration(ReconcileRacingDay, noop),
                PhaseHandlerRegistration(RequestTraining, noop),
            )
        )
        authority = OperationalAuthority(
            self.store, self.artifacts, command_executor=dispatcher, clock=lambda: NOW
        )
        command = ApplicationCommand(
            operation(3981),
            day_id,
            DiscoverProgramme("official", PROGRAMME),
        )
        commands = (
            command,
            ApplicationCommand(operation(3982), day_id, CollectCardsAndForm()),
            ApplicationCommand(operation(3983), day_id, CollectAdaptiveOdds()),
            ApplicationCommand(operation(3984), day_id, CloseAndSeal()),
            ApplicationCommand(operation(3985), day_id, CommitDeferredPrediction()),
            ApplicationCommand(operation(3986), day_id, CollectResults()),
            ApplicationCommand(operation(3987), day_id, JoinTrainingExamples()),
            ApplicationCommand(operation(3988), day_id, ReconcileRacingDay()),
            ApplicationCommand(
                operation(3989),
                day_id,
                RequestTraining(
                    "failed-request",
                    operation(3969),
                    operation(3970),
                    operation(3971),
                ),
            ),
        )
        generation = self.authority.acquire_lease(
            operation(3972),
            owner="failure-fixture",
            token="failure",
            now=NOW,
            ttl=timedelta(hours=1),
        )
        self.authority.plan_racing_day(
            operation(3973),
            racing_day_id=day_id,
            lease_token="failure",
            lease_generation=generation,
            commands=commands,
            at=NOW,
        )
        with self.assertRaisesRegex(RuntimeError, "credential"):
            authority.execute_application_command(
                command,
                token="failure",
                generation=generation,
                at=NOW,
            )
        with self.store._connect() as db:
            self.assertIsNone(
                db.execute(
                    "SELECT 1 FROM operations WHERE operation_id=?",
                    (str(command.operation_id),),
                ).fetchone()
            )
            self.assertIsNone(
                db.execute(
                    "SELECT 1 FROM phase7_application_command_receipts "
                    "WHERE command_operation_id=?",
                    (str(command.operation_id),),
                ).fetchone()
            )
            self.assertIsNotNone(
                db.execute(
                    "SELECT 1 FROM phase7_application_command_claims "
                    "WHERE command_operation_id=?",
                    (str(command.operation_id),),
                ).fetchone()
            )
            details = db.execute(
                "SELECT details FROM phase7_application_command_attempts "
                "WHERE command_operation_id=? AND state='handler_failed'",
                (str(command.operation_id),),
            ).fetchone()[0]
            self.assertEqual(details, "command_failure:RuntimeError")
            self.assertNotIn(secret, details)
        with self.assertRaisesRegex(OperationalRejected, "indeterminate"):
            authority.execute_application_command(
                command,
                token="failure",
                generation=generation,
                at=NOW,
            )

    def test_concurrent_public_execution_claim_invokes_handler_only_once(self) -> None:
        service, handlers, commands = self.trusted_scheduler_fixture(39700)
        entered = threading.Event()
        release = threading.Event()
        calls = []

        def blocking_discover(command, at):
            calls.append(command.operation_id)
            entered.set()
            self.assertTrue(release.wait(5))
            handlers.discover(command, at)

        registrations = list(handlers.registrations())
        registrations[0] = PhaseHandlerRegistration(DiscoverProgramme, blocking_discover)
        authority = OperationalAuthority(
            self.store,
            self.artifacts,
            command_executor=ClosedCommandDispatcher(registrations),
            clock=lambda: handlers.at,
        )
        command = commands[0]
        outcomes = []

        def execute() -> None:
            try:
                outcomes.append(
                    authority.execute_application_command(
                        command,
                        token=service.token,
                        generation=service.generation,
                        at=handlers.at,
                    )
                )
            except Exception as error:  # recorded and asserted on the main thread
                outcomes.append(error)

        first = threading.Thread(target=execute)
        second = threading.Thread(target=execute)
        first.start()
        self.assertTrue(entered.wait(5))
        second.start()
        second.join(5)
        release.set()
        first.join(5)
        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertEqual(calls, [command.operation_id])
        self.assertEqual(len([item for item in outcomes if isinstance(item, dict)]), 1)
        self.assertEqual(
            len(
                [
                    item
                    for item in outcomes
                    if isinstance(item, OperationalRejected) and "indeterminate" in str(item)
                ]
            ),
            1,
        )

    def test_completed_command_replay_binds_closed_payload_identity_before_handler(
        self,
    ) -> None:
        service, handlers, commands = self.trusted_scheduler_fixture(39720)
        command = commands[0]
        service.advance(
            operation(39790),
            racing_day_id=command.racing_day_id,
            phase=command.phase,
            now=handlers.at,
            command=command,
        )
        variants = (
            ApplicationCommand(
                command.operation_id,
                command.racing_day_id,
                DiscoverProgramme("changed-source", command.payload.programme_checksum),
            ),
            ApplicationCommand(
                command.operation_id,
                command.racing_day_id,
                CollectCardsAndForm(),
            ),
        )
        for variant in variants:
            with self.subTest(payload=type(variant.payload).__name__):
                with self.assertRaises(ConflictingOperation):
                    service.authority.execute_application_command(
                        variant,
                        token=service.token,
                        generation=service.generation,
                        at=handlers.at,
                    )
        self.assertEqual(handlers.executions["discover_programme"], 1)
        self.assertNotEqual(
            ApplicationCommand(
                operation(39791), command.racing_day_id, CollectCardsAndForm()
            ).payload_sha256(),
            ApplicationCommand(
                operation(39792), command.racing_day_id, CollectAdaptiveOdds()
            ).payload_sha256(),
        )

    def test_stale_claim_recovery_rejects_changed_payload_before_handler(self) -> None:
        service, handlers, commands = self.trusted_scheduler_fixture(39730)
        command = commands[0]

        def crash(_command, _at):
            raise RuntimeError("crash after durable claim")

        registrations = list(handlers.registrations())
        registrations[0] = PhaseHandlerRegistration(DiscoverProgramme, crash)
        crashed = OperationalAuthority(
            self.store,
            self.artifacts,
            command_executor=ClosedCommandDispatcher(registrations),
            clock=lambda: handlers.at,
        )
        with self.assertRaises(RuntimeError):
            crashed.execute_application_command(
                command,
                token=service.token,
                generation=service.generation,
                at=handlers.at,
            )

        handlers.at += timedelta(microseconds=1)
        generation = self.authority.acquire_lease(
            operation(39799),
            owner="claim-recovery",
            token="claim-recovery",
            now=handlers.at,
            ttl=timedelta(hours=1),
        )
        recovery_calls = []

        def should_not_run(_command, _at):
            recovery_calls.append(True)

        registrations[0] = PhaseHandlerRegistration(DiscoverProgramme, should_not_run)
        recovery = OperationalAuthority(
            self.store,
            self.artifacts,
            command_executor=ClosedCommandDispatcher(registrations),
            clock=lambda: handlers.at,
        )
        recovery.plan_racing_day(
            operation(39798),
            racing_day_id=command.racing_day_id,
            lease_token="claim-recovery",
            lease_generation=generation,
            commands=commands,
            at=handlers.at,
        )
        changed = ApplicationCommand(
            command.operation_id,
            command.racing_day_id,
            DiscoverProgramme("changed-source", command.payload.programme_checksum),
        )
        with self.assertRaises(ConflictingOperation):
            recovery.execute_application_command(
                changed,
                token="claim-recovery",
                generation=generation,
                at=handlers.at,
            )
        self.assertEqual(recovery_calls, [])

    def test_closed_dispatcher_requires_exactly_one_handler_for_every_typed_command(
        self,
    ) -> None:
        handler = lambda command, at: None
        complete = [
            PhaseHandlerRegistration(payload_type, handler)
            for payload_type in (
                DiscoverProgramme,
                CollectCardsAndForm,
                CollectAdaptiveOdds,
                CloseAndSeal,
                CommitDeferredPrediction,
                CollectResults,
                JoinTrainingExamples,
                ReconcileRacingDay,
                RequestTraining,
            )
        ]
        dispatcher = ClosedCommandDispatcher(complete)
        dispatcher.execute(
            ApplicationCommand(
                operation(3992),
                "day_" + "4" * 32,
                DiscoverProgramme("official", PROGRAMME),
            ),
            at=NOW,
        )
        with self.assertRaisesRegex(ValueError, "missing"):
            ClosedCommandDispatcher(complete[:-1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ClosedCommandDispatcher([*complete, complete[0]])
        with self.assertRaisesRegex(ValueError, "unknown"):
            ClosedCommandDispatcher([*complete[:-1], PhaseHandlerRegistration(str, handler)])
        invalid = list(complete)
        invalid[0] = PhaseHandlerRegistration(DiscoverProgramme, None)
        with self.assertRaisesRegex(ValueError, "callable"):
            ClosedCommandDispatcher(invalid)

    def test_stale_generation_cannot_commit_receipt_after_handler_takeover(
        self,
    ) -> None:
        service, handlers, commands = self.trusted_scheduler_fixture(39750)
        command = commands[0]
        current = [handlers.at]
        entered, release = threading.Event(), threading.Event()

        def blocked(command, at):
            entered.set()
            self.assertTrue(release.wait(5))
            handlers.discover(command, at)

        registrations = list(handlers.registrations())
        registrations[0] = PhaseHandlerRegistration(DiscoverProgramme, blocked)
        authority = OperationalAuthority(
            self.store,
            self.artifacts,
            command_executor=ClosedCommandDispatcher(registrations),
            clock=lambda: current[0],
        )
        outcome = []
        worker = threading.Thread(
            target=lambda: outcome.append(
                self._capture(
                    lambda: authority.execute_application_command(
                        command,
                        token=service.token,
                        generation=service.generation,
                        at=handlers.at,
                    )
                )
            )
        )
        worker.start()
        self.assertTrue(entered.wait(5))
        current[0] = handlers.at + timedelta(microseconds=1)
        takeover_generation = self.authority.acquire_lease(
            operation(39759),
            owner="takeover",
            token="takeover",
            now=current[0],
            ttl=timedelta(hours=1),
        )
        with self.store._connect() as db:
            db.execute(
                "INSERT INTO phase7_application_command_attempts("
                "command_operation_id,lease_generation,lease_token,state,recorded_at,details"
                ") VALUES(?,?,?,?,?,?)",
                (
                    str(command.operation_id),
                    takeover_generation,
                    "takeover",
                    "recovering",
                    current[0].isoformat(),
                    "newer authoritative recovery attempt",
                ),
            )
        release.set()
        worker.join(5)
        self.assertIsInstance(outcome[0], OperationalRejected)
        with self.store._connect() as db:
            self.assertIsNone(
                db.execute(
                    "SELECT 1 FROM phase7_application_command_receipts "
                    "WHERE command_operation_id=?",
                    (str(command.operation_id),),
                ).fetchone()
            )
            self.assertEqual(
                db.execute(
                    "SELECT state FROM phase7_application_command_attempts "
                    "WHERE command_operation_id=? ORDER BY attempt_id DESC LIMIT 1",
                    (str(command.operation_id),),
                ).fetchone()["state"],
                "fenced",
            )

    def test_fresh_generation_recovers_completed_postcondition_after_handler_crash(
        self,
    ) -> None:
        service, handlers, commands = self.trusted_scheduler_fixture(39760)
        command = commands[0]
        current = [handlers.at]

        def complete_then_crash(command, at):
            handlers.discover(command, at)
            raise RuntimeError("crash after durable producer mutation")

        registrations = list(handlers.registrations())
        registrations[0] = PhaseHandlerRegistration(DiscoverProgramme, complete_then_crash)
        authority = OperationalAuthority(
            self.store,
            self.artifacts,
            command_executor=ClosedCommandDispatcher(registrations),
            clock=lambda: current[0],
        )
        with self.assertRaisesRegex(RuntimeError, "durable producer"):
            authority.execute_application_command(
                command,
                token=service.token,
                generation=service.generation,
                at=handlers.at,
            )
        current[0] += timedelta(microseconds=1)
        generation = self.authority.acquire_lease(
            operation(39769),
            owner="recovery",
            token="recovery",
            now=current[0],
            ttl=timedelta(hours=1),
        )
        authority.plan_racing_day(
            operation(39768),
            racing_day_id=command.racing_day_id,
            lease_token="recovery",
            lease_generation=generation,
            commands=commands,
            at=current[0],
        )
        result = authority.execute_application_command(
            command,
            token="recovery",
            generation=generation,
            at=current[0],
        )
        self.assertEqual(result["source"], "official")
        self.assertEqual(handlers.executions["discover_programme"], 1)
        with self.store._connect() as db:
            self.assertEqual(
                [
                    row["state"]
                    for row in db.execute(
                        "SELECT state FROM phase7_application_command_attempts "
                        "WHERE command_operation_id=? ORDER BY attempt_id",
                        (str(command.operation_id),),
                    )
                ],
                ["claimed", "handler_failed", "recovering", "completed"],
            )

    def test_recovery_rechecks_changed_postcondition_inside_receipt_transaction(
        self,
    ) -> None:
        service, handlers, commands = self.trusted_scheduler_fixture(39770)
        command = commands[0]
        current = [handlers.at]

        def complete_then_crash(command, at):
            handlers.discover(command, at)
            raise RuntimeError("crash after mutation")

        initial = list(handlers.registrations())
        initial[0] = PhaseHandlerRegistration(DiscoverProgramme, complete_then_crash)
        crashed = OperationalAuthority(
            self.store,
            self.artifacts,
            command_executor=ClosedCommandDispatcher(initial),
            clock=lambda: current[0],
        )
        with self.assertRaises(RuntimeError):
            crashed.execute_application_command(
                command,
                token=service.token,
                generation=service.generation,
                at=current[0],
            )
        current[0] += timedelta(microseconds=1)
        generation = self.authority.acquire_lease(
            operation(39779),
            owner="recovery",
            token="recovery-changed",
            now=current[0],
            ttl=timedelta(hours=1),
        )

        class ChangingPostconditionAuthority(OperationalAuthority):
            reads = 0

            def _phase_postcondition(self, db, recovered_command):
                result = super()._phase_postcondition(db, recovered_command)
                self.reads += 1
                return result if self.reads == 1 else {**result, "source": "changed"}

        recovery = ChangingPostconditionAuthority(
            self.store,
            self.artifacts,
            command_executor=handlers.dispatcher(),
            clock=lambda: current[0],
        )
        recovery.plan_racing_day(
            operation(39778),
            racing_day_id=command.racing_day_id,
            lease_token="recovery-changed",
            lease_generation=generation,
            commands=commands,
            at=current[0],
        )
        with self.assertRaisesRegex(OperationalRejected, "postcondition changed"):
            recovery.execute_application_command(
                command,
                token="recovery-changed",
                generation=generation,
                at=current[0],
            )
        with self.store._connect() as db:
            self.assertIsNone(
                db.execute(
                    "SELECT 1 FROM phase7_application_command_receipts "
                    "WHERE command_operation_id=?",
                    (str(command.operation_id),),
                ).fetchone()
            )
            self.assertEqual(
                db.execute(
                    "SELECT state FROM phase7_application_command_attempts "
                    "WHERE command_operation_id=? ORDER BY attempt_id DESC LIMIT 1",
                    (str(command.operation_id),),
                ).fetchone()["state"],
                "postcondition_failed",
            )

    def test_handler_postcondition_failure_is_audited_without_a_receipt(self) -> None:
        service, handlers, commands = self.trusted_scheduler_fixture(39780)
        command = commands[0]

        class FailingPostconditionAuthority(OperationalAuthority):
            def _phase_postcondition(self, db, completed_command):
                raise RuntimeError("injected postcondition boundary failure")

        authority = FailingPostconditionAuthority(
            self.store,
            self.artifacts,
            command_executor=handlers.dispatcher(),
            clock=lambda: handlers.at,
        )
        with self.assertRaisesRegex(RuntimeError, "postcondition boundary"):
            authority.execute_application_command(
                command,
                token=service.token,
                generation=service.generation,
                at=handlers.at,
            )
        with self.store._connect() as db:
            self.assertIsNone(
                db.execute(
                    "SELECT 1 FROM phase7_application_command_receipts "
                    "WHERE command_operation_id=?",
                    (str(command.operation_id),),
                ).fetchone()
            )
            self.assertEqual(
                db.execute(
                    "SELECT state FROM phase7_application_command_attempts "
                    "WHERE command_operation_id=? ORDER BY attempt_id DESC LIMIT 1",
                    (str(command.operation_id),),
                ).fetchone()["state"],
                "postcondition_failed",
            )

    @staticmethod
    def _capture(callback):
        try:
            return callback()
        except Exception as error:
            return error

    def test_renewal_must_advance_time_and_extend_expiry_in_python_and_sql(
        self,
    ) -> None:
        generation = self.authority.acquire_lease(
            operation(39940),
            owner="renewal",
            token="renewal",
            now=NOW,
            ttl=timedelta(minutes=5),
        )
        with self.assertRaisesRegex(OperationalRejected, "stale or fenced"):
            self.authority.renew_lease(
                operation(39941),
                token="renewal",
                generation=generation,
                now=NOW + timedelta(minutes=1),
                ttl=timedelta(minutes=1),
            )
        with self.assertRaises(sqlite3.IntegrityError):
            with self.store._operation(operation(39942), "phase7_renew_scheduler_lease", {}) as (
                db,
                _,
            ):
                db.execute(
                    "UPDATE phase7_scheduler_lease SET expires_at=? WHERE singleton=1",
                    ((NOW + timedelta(minutes=5)).isoformat(),),
                )
                db.execute(
                    "INSERT INTO phase7_scheduler_renewals VALUES(?,?,?,?,?)",
                    (
                        str(operation(39942)),
                        generation,
                        "renewal",
                        (NOW - timedelta(seconds=1)).isoformat(),
                        (NOW + timedelta(minutes=5)).isoformat(),
                    ),
                )

    def test_public_service_advancement_is_ordered_and_lease_fenced(self) -> None:
        service, handlers, commands = self.trusted_scheduler_fixture(810000)
        day_id = commands[0].racing_day_id
        with self.assertRaises(OperationalRejected):
            service.advance(
                operation(810090),
                racing_day_id=day_id,
                phase="collect_adaptive_odds",
                now=handlers.at,
                command=commands[2],
            )
        self.authority.acquire_lease(
            operation(810091),
            owner="replacement",
            token="new",
            now=handlers.at + timedelta(seconds=2),
            ttl=timedelta(hours=1),
        )
        with self.assertRaises(OperationalRejected):
            service.advance(
                operation(810092),
                racing_day_id=day_id,
                phase="discover_programme",
                now=handlers.at + timedelta(seconds=2),
                command=commands[0],
            )

    def test_crash_recovery_and_exact_replay_at_all_nine_barriers(self) -> None:
        base = 820000
        service, handlers, commands = self.trusted_scheduler_fixture(base)
        day_id = commands[0].racing_day_id
        for ordinal, command in enumerate(commands, 1):
            if ordinal == 8:
                handlers.at += timedelta(microseconds=2)
                generation = self.authority.acquire_lease(
                    operation(base + 60),
                    owner="barrier-recovery",
                    token="barrier-recovery-token",
                    now=handlers.at,
                    ttl=timedelta(hours=1),
                )
                OperationalAuthority(
                    self.store, self.artifacts, clock=lambda: handlers.at
                ).plan_racing_day(
                    operation(base + 61),
                    racing_day_id=day_id,
                    lease_token="barrier-recovery-token",
                    lease_generation=generation,
                    commands=commands,
                    at=handlers.at,
                )
                service = RaceCollectionService(
                    OperationalAuthority(
                        self.store,
                        self.artifacts,
                        command_executor=handlers.dispatcher(),
                        clock=lambda: handlers.at,
                    ),
                    token="barrier-recovery-token",
                    generation=generation,
                )
            advance_id = operation(base + 70 + ordinal)
            expected = service.dispatch(command, now=handlers.at)
            with self.store._connect() as db:
                self.assertIsNone(
                    db.execute(
                        "SELECT 1 FROM phase7_scheduler_progress WHERE operation_id=?",
                        (str(advance_id),),
                    ).fetchone()
                )
                receipt = db.execute(
                    "SELECT result_json,result_checksum FROM "
                    "phase7_application_command_receipts WHERE command_operation_id=?",
                    (str(command.operation_id),),
                ).fetchone()
            self.assertEqual(json.loads(receipt["result_json"]), expected)
            self.assertEqual(str(_checksum(expected)), receipt["result_checksum"])
            self.assertEqual(
                service.advance(
                    advance_id,
                    racing_day_id=day_id,
                    phase=command.phase,
                    now=handlers.at,
                    command=command,
                ),
                expected,
            )

            self.assertEqual(
                service.advance(
                    advance_id,
                    racing_day_id=day_id,
                    phase=command.phase,
                    now=handlers.at,
                    command=command,
                ),
                expected,
            )
            self.assertEqual(handlers.executions[command.phase], 1)

    def test_normal_plan_exact_replay_uses_immutable_trusted_planning_time(
        self,
    ) -> None:
        base = 855000
        sampled_under_operation_lock = []

        def planning_clock():
            probe = self.store._connect()
            probe.execute("PRAGMA busy_timeout = 0")
            try:
                with self.assertRaises(sqlite3.OperationalError):
                    probe.execute("BEGIN IMMEDIATE")
                sampled_under_operation_lock.append(True)
            finally:
                probe.close()
            return NOW - timedelta(days=1) + timedelta(microseconds=base)

        _service, handlers, commands = self.trusted_scheduler_fixture(
            base, planning_clock=planning_clock
        )
        self.assertEqual(sampled_under_operation_lock, [True])

        self.assertFalse(
            OperationalAuthority(
                self.store, self.artifacts, clock=lambda: handlers.at
            ).plan_racing_day(
                operation(base + 4),
                racing_day_id=commands[0].racing_day_id,
                lease_token=f"lease-{base}",
                lease_generation=1,
                commands=commands,
                at=handlers.at + timedelta(days=1),
            )
        )
        with self.store._connect() as db:
            authority_at = db.execute(
                "SELECT planned_at FROM phase7_day_plan_authorities WHERE operation_id=?",
                (str(operation(base + 4)),),
            ).fetchone()[0]
            plan_times = {
                row[0]
                for row in db.execute(
                    "SELECT planned_at FROM phase7_day_command_plan " "WHERE racing_day_id=?",
                    (commands[0].racing_day_id,),
                )
            }
        self.assertEqual(plan_times, {authority_at})
        self.assertEqual(sampled_under_operation_lock, [True])

    def _assert_ordinary_plan_is_adopted_after_restart(self, *, base: int, prefix: int) -> None:
        service, handlers, commands = self.trusted_scheduler_fixture(base)
        if prefix:
            service.advance(
                operation(base + 70),
                racing_day_id=commands[0].racing_day_id,
                phase=commands[0].phase,
                now=handlers.at,
                command=commands[0],
            )
        takeover_at = handlers.at + timedelta(microseconds=1)
        handlers.at = takeover_at
        sampled_under_operation_lock = []

        def adoption_clock():
            probe = self.store._connect()
            probe.execute("PRAGMA busy_timeout = 0")
            try:
                try:
                    probe.execute("BEGIN IMMEDIATE")
                except sqlite3.OperationalError:
                    sampled_under_operation_lock.append(True)
                else:
                    probe.rollback()
            finally:
                probe.close()
            return takeover_at

        class Adapter:
            def registrations(self):
                return handlers.registrations()

            def next_cycle(self, *, now):
                return None

            def close(self):
                pass

        composition = ServiceComposition(
            self.configuration(),
            self.store,
            self.artifacts,
            Adapter(),
            owner=f"restart-{base}",
            token=f"restart-token-{base}",
            lease_ttl=timedelta(hours=1),
            clock=adoption_clock,
        )
        cycle = RacingDayCycle(
            commands[0].racing_day_id,
            commands,
            operation(base + 4),
            tuple(operation(base + 70 + ordinal) for ordinal in range(9)),
            takeover_at,
        )
        if prefix:
            self.assertEqual(len(composition.run_cycle(cycle)), 9)
        else:
            with self.assertRaisesRegex(OperationalRejected, "durable recovered prefix"):
                composition.run_cycle(cycle)
        self.assertTrue(sampled_under_operation_lock)
        self.assertEqual(handlers.executions["discover_programme"], 1)
        with self.store._connect() as db:
            adoption = db.execute(
                "SELECT lease_generation,lease_token,plan_kind,operation_id "
                "FROM phase7_day_plan_adoptions WHERE racing_day_id=?",
                (commands[0].racing_day_id,),
            ).fetchone()
            self.assertEqual(
                tuple(adoption)[:3],
                (2, f"restart-token-{base}", "phase7_plan_racing_day"),
            )
        samples_before_replay = len(sampled_under_operation_lock)
        self.assertFalse(
            composition.authority.plan_racing_day(
                OperationId(adoption["operation_id"]),
                racing_day_id=commands[0].racing_day_id,
                lease_token=f"restart-token-{base}",
                lease_generation=2,
                commands=commands,
                at=takeover_at + timedelta(days=1),
            )
        )
        self.assertEqual(len(sampled_under_operation_lock), samples_before_replay + 1)

    def test_ordinary_plan_is_adopted_after_restart_with_empty_prefix(self) -> None:
        self._assert_ordinary_plan_is_adopted_after_restart(base=856000, prefix=0)

    def test_ordinary_plan_is_adopted_after_restart_with_partial_prefix(self) -> None:
        self._assert_ordinary_plan_is_adopted_after_restart(base=857000, prefix=1)

    def test_receipt_cannot_cross_expired_trusted_time_into_progress(self) -> None:
        service, handlers, commands = self.trusted_scheduler_fixture(858000)
        command = commands[0]
        service.dispatch(command, now=handlers.at)
        caller_now = handlers.at
        handlers.at += timedelta(microseconds=1)
        with self.assertRaisesRegex(OperationalRejected, "stale or absent scheduler ownership"):
            service.advance(
                operation(858070),
                racing_day_id=command.racing_day_id,
                phase=command.phase,
                now=caller_now,
                command=command,
            )
        with self.store._connect() as db:
            self.assertEqual(
                db.execute(
                    "SELECT count(*) FROM phase7_scheduler_progress WHERE racing_day_id=?",
                    (command.racing_day_id,),
                ).fetchone()[0],
                0,
            )

    def test_planning_requires_exact_token_and_fresh_trusted_time(self) -> None:
        service, handlers, commands = self.trusted_scheduler_fixture(859000)
        with self.store._connect() as db:
            self.assertEqual(
                db.execute(
                    "SELECT count(*) FROM phase7_day_command_plan WHERE racing_day_id=?",
                    (commands[0].racing_day_id,),
                ).fetchone()[0],
                9,
            )
        rejected_operations = []
        for operation_id, token, trusted_at in (
            (operation(859090), "wrong-token", handlers.at),
            (
                operation(859091),
                service.token,
                handlers.at + timedelta(microseconds=1),
            ),
        ):
            rejected_operations.append(str(operation_id))
            with self.assertRaisesRegex(OperationalRejected, "live scheduler lease"):
                OperationalAuthority(
                    self.store, self.artifacts, clock=lambda at=trusted_at: at
                ).plan_racing_day(
                    operation_id,
                    racing_day_id=commands[0].racing_day_id,
                    lease_token=token,
                    lease_generation=service.generation,
                    commands=commands,
                    at=handlers.at,
                )
        with self.store._connect() as db:
            self.assertEqual(
                db.execute(
                    "SELECT count(*) FROM operations WHERE operation_id IN (?,?)",
                    rejected_operations,
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                db.execute(
                    "SELECT count(*) FROM phase7_day_plan_authorities " "WHERE racing_day_id=?",
                    (commands[0].racing_day_id,),
                ).fetchone()[0],
                1,
            )
            self.assertEqual(
                db.execute(
                    "SELECT count(*) FROM phase7_day_plan_adoptions " "WHERE racing_day_id=?",
                    (commands[0].racing_day_id,),
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                db.execute(
                    "SELECT count(*) FROM phase7_day_command_plan " "WHERE racing_day_id=?",
                    (commands[0].racing_day_id,),
                ).fetchone()[0],
                9,
            )

    def test_main_sanitizes_arbitrary_exception_text(self) -> None:
        secret = "postgres://admin:password@example.invalid/database"

        def fail_loader(*_args, **_kwargs):
            raise RuntimeError(secret)

        stderr = StringIO()
        with redirect_stderr(stderr):
            exit_code = main(
                ("--config", "/unused/synthetic.json", "--once"),
                composition_loader=fail_loader,
            )
        self.assertEqual(exit_code, 69)
        self.assertEqual(stderr.getvalue(), "race-collection-service unavailable\n")
        self.assertNotIn(secret, stderr.getvalue())

    def test_unrelated_committed_operation_cannot_advance_a_phase(self) -> None:
        day_id = "day_" + "7" * 32
        with self.store._operation(operation(150), "seed_day", {}) as (db, _):
            db.execute(
                "INSERT INTO racing_days VALUES(?,?,?,?,NULL)",
                (day_id, "2026-08-02", "UTC", NOW.isoformat()),
            )
        generation = self.authority.acquire_lease(
            operation(151),
            owner="service",
            token="receipt",
            now=NOW,
            ttl=timedelta(days=1),
        )
        unrelated = operation(152)
        with self.store._operation(unrelated, "record_odds_attempt", {"unrelated": True}):
            pass
        with self.store._connect() as db, self.assertRaises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO phase7_application_command_receipts VALUES(?,?,?,?,?,?,?)",
                (
                    str(unrelated),
                    day_id,
                    "discover_programme",
                    "{}",
                    "sha256:" + "a" * 64,
                    NOW.isoformat(),
                    "0" * 64,
                ),
            )
        service = RaceCollectionService(self.authority, token="receipt", generation=generation)
        with self.assertRaises(OperationalRejected):
            service.advance(
                operation(153),
                racing_day_id=day_id,
                phase="discover_programme",
                now=NOW,
                command=ApplicationCommand(
                    unrelated, day_id, DiscoverProgramme("official", PROGRAMME)
                ),
            )

    def test_direct_sql_probation_rejects_nonhex_sha256_shape(self) -> None:
        checksum = "sha256:" + "a" * 64
        parent_operation = operation(159)
        with self.store._operation(parent_operation, "seed_probation_parent", {}) as (
            db,
            _,
        ):
            with self.assertRaisesRegex(
                sqlite3.IntegrityError,
                "probation state was not issued by exact Phase 7 authority",
            ):
                db.execute(
                    "INSERT INTO phase6_probation_states VALUES(?,?,?,?,?)",
                    (
                        "forged-nonhex",
                        "2026-07-02",
                        checksum,
                        NOW.isoformat(),
                        str(parent_operation),
                    ),
                )

    def test_advancement_operation_replay_rejects_changed_intent(self) -> None:
        base = 830000
        service, handlers, commands = self.trusted_scheduler_fixture(base)
        command = commands[0]
        day_id = command.racing_day_id
        advance_id = operation(base + 70)
        expected = service.advance(
            advance_id,
            racing_day_id=day_id,
            phase=command.phase,
            now=handlers.at,
            command=command,
        )
        self.assertEqual(expected["programme_checksum"], str(command.payload.programme_checksum))

        conflicts = (
            (service, "day_" + "5" * 32, "discover_programme", handlers.at, command),
            (service, day_id, "collect_cards_and_form", handlers.at, command),
            (
                service,
                day_id,
                "discover_programme",
                handlers.at + timedelta(seconds=1),
                command,
            ),
            (
                service,
                day_id,
                "discover_programme",
                handlers.at,
                ApplicationCommand(
                    operation(base + 99),
                    day_id,
                    DiscoverProgramme("other", command.payload.programme_checksum),
                ),
            ),
        )
        for (
            replay_service,
            replay_day,
            replay_phase,
            replay_at,
            replay_command,
        ) in conflicts:
            with self.subTest(day=replay_day, phase=replay_phase, at=replay_at):
                with self.assertRaises(ConflictingOperation):
                    replay_service.advance(
                        advance_id,
                        racing_day_id=replay_day,
                        phase=replay_phase,
                        now=replay_at,
                        command=replay_command,
                    )

    def test_completed_advancement_replay_requires_current_live_lease(self) -> None:
        base = 830500
        service, handlers, commands = self.trusted_scheduler_fixture(base)
        command = commands[0]
        advance_id = operation(base + 70)
        expected = service.advance(
            advance_id,
            racing_day_id=command.racing_day_id,
            phase=command.phase,
            now=handlers.at,
            command=command,
        )
        handlers.at += timedelta(microseconds=1)
        with self.assertRaisesRegex(OperationalRejected, "scheduler ownership"):
            service.advance(
                advance_id,
                racing_day_id=command.racing_day_id,
                phase=command.phase,
                now=handlers.at - timedelta(microseconds=1),
                command=command,
            )
        with self.store._connect() as db:
            self.assertEqual(
                db.execute(
                    "SELECT count(*) FROM phase7_scheduler_progress WHERE operation_id=?",
                    (str(advance_id),),
                ).fetchone()[0],
                1,
            )
        handlers.at -= timedelta(microseconds=1)
        self.assertEqual(
            service.advance(
                advance_id,
                racing_day_id=command.racing_day_id,
                phase=command.phase,
                now=handlers.at,
                command=command,
            ),
            expected,
        )

    def test_admin_replay_conflict_scoped_pause_and_safe_collection(self) -> None:
        service, handlers, commands = self.trusted_scheduler_fixture(840000)
        safe_receipt = service.dispatch(commands[0], now=handlers.at)
        self.assertTrue(
            self.authority.admin_pause(
                operation(11),
                scope="results",
                paused=True,
                actor="operator",
                reason="result barrier incident",
                at=NOW,
            )
        )
        self.assertFalse(
            self.authority.admin_pause(
                operation(11),
                scope="results",
                paused=True,
                actor="operator",
                reason="result barrier incident",
                at=NOW,
            )
        )
        with self.assertRaises(ConflictingOperation):
            self.authority.admin_pause(
                operation(11),
                scope="joins",
                paused=True,
                actor="operator",
                reason="changed",
                at=NOW,
            )
        self.assertEqual(
            safe_receipt["programme_checksum"],
            str(commands[0].payload.programme_checksum),
        )
        with self.assertRaises(OperationalRejected):
            service.dispatch(commands[5], now=handlers.at)
        self.assertTrue(
            self.authority.admin_pause(
                operation(12),
                scope="results",
                paused=False,
                actor="operator",
                reason="evidence repaired by command",
                at=NOW,
            )
        )
        with self.assertRaises(TypeError):
            ApplicationCommand(operation(12002), lambda: None)  # type: ignore[call-arg]

    def test_release_manifest_and_generic_unit_validation(self) -> None:
        configuration_values = {
            "schema_version": "phase7-config-v1",
            "service_root": "/opt/race-collection/current",
            "artifact_root": "/var/lib/race-collection/artifacts",
            "operations_database": "/var/lib/race-collection/operations.sqlite3",
            "sources": ("official",),
            "schedule_policy": "adaptive-odds-v1",
            "promotion_policy": "phase6-promotion-v1",
            "bundle_versions": ("runner-win-probability-v1",),
            "runtime_adapter": "race_collection.runtime_adapters:unavailable",
            "runtime_input_checksum": ArtifactChecksum("sha256:" + "9" * 64),
        }
        for field, invalid in (
            ("sources", (" ",)),
            ("sources", ("official", "official")),
            ("bundle_versions", (" ",)),
            (
                "bundle_versions",
                ("runner-win-probability-v1", "runner-win-probability-v1"),
            ),
            ("schedule_policy", " "),
            ("promotion_policy", " "),
        ):
            with (
                self.subTest(field=field, invalid=invalid),
                self.assertRaises(ValueError),
            ):
                ReleaseConfiguration(**{**configuration_values, field: invalid})
        with self.assertRaises(ValueError):
            DiscoverProgramme(None, PROGRAMME)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ApplicationCommand(operation(18), None, CollectCardsAndForm())  # type: ignore[arg-type]
        self.register_config(19)
        checksum = self.authority.register_release(operation(20), self.manifest(), NOW)
        self.assertEqual(self.artifacts.verify(checksum).checksum, checksum)
        units = self.authority.generate_units(
            self.manifest(),
            self.configuration(),
            config_path="/etc/race-collection/release.json",
        )
        self.assertEqual(set(units), {"race-collection.service"})
        self.assertNotIn("20260722", units["race-collection.service"])
        self.assertNotIn("timer", units["race-collection.service"])
        self.assertIn("/bin/race-collection-service", units["race-collection.service"])
        self.assertIn("--continuous", units["race-collection.service"])
        executable = Path(__file__).parents[2] / "bin" / "race-collection-service"
        self.assertTrue(executable.is_file())
        self.assertTrue(executable.stat().st_mode & 0o111)
        help_result = subprocess.run(
            (str(executable), "--help"),
            cwd=Path(__file__).parents[2],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--config", help_result.stdout)
        with self.assertRaises(ValueError):
            ReleaseManifest(
                "phase7-release-v1",
                "bad",
                "x" * 40,
                ArtifactChecksum("sha256:" + "2" * 64),
                18,
                "a",
                "p",
                ("v1",),
                "/tmp/20260722T083110Z-run/source",
            )

        with self.assertRaises(ValueError):
            ReleaseManifest(
                "phase7-release-v1",
                "schema-mismatch",
                "a" * 40,
                self.manifest().config_checksum,
                29,
                "canonical-artifacts-v1",
                "phase6-promotion-v1",
                ("runner-win-probability-v1",),
                "/opt/race-collection/current",
            )
        unsupported = ReleaseConfiguration(
            "phase7-config-v1",
            "/opt/race-collection/current",
            "/var/lib/race-collection/artifacts",
            "/var/lib/race-collection/operations.sqlite3",
            ("official",),
            "adaptive-odds-v1",
            "phase6-promotion-v1",
            ("unsupported-forecast-v99",),
            "race_collection.runtime_adapters:unavailable",
            ArtifactChecksum("sha256:" + "9" * 64),
        )
        unsupported_checksum = self.authority.register_configuration(
            operation(22), unsupported, NOW
        )
        with self.assertRaisesRegex(OperationalRejected, "typed configuration contract"):
            self.authority.register_release(
                operation(23),
                ReleaseManifest(
                    "phase7-release-v1",
                    "unsupported-bundle-contract",
                    "a" * 40,
                    unsupported_checksum,
                    28,
                    "canonical-artifacts-v1",
                    "phase6-promotion-v1",
                    ("unsupported-forecast-v99",),
                    "/opt/race-collection/current",
                ),
                NOW,
            )

    def test_operational_path_rejects_symlink_run_escape_and_loop(self) -> None:
        root = Path(self.temporary.name)
        disposable = root / "codex-x-pilot" / ".state" / "runs" / "run-1" / "service"
        disposable.mkdir(parents=True)
        escaped = root / "stable-service"
        escaped.symlink_to(disposable, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "outside run worktrees"):
            _safe_operational_path(str(escaped))

        first = root / "first-link"
        second = root / "second-link"
        first.symlink_to(second)
        second.symlink_to(first)
        with self.assertRaisesRegex(ValueError, "symlink ambiguity"):
            _safe_operational_path(str(first))

    def test_real_service_entrypoint_uses_one_closed_dispatcher_and_durable_scheduler(
        self,
    ) -> None:
        self.register_config(960000)
        self.authority.register_release(operation(960001), self.manifest(), NOW)
        programme = self.artifacts.put(
            b'{"programme":"service-entrypoint"}',
            media_type="application/json",
        ).checksum
        day = RacingDay(
            RacingDayId("day_" + f"{960002:032x}"),
            NOW.date(),
            "UTC",
            NOW,
        )
        self.store.create_racing_day(operation(960002), day)
        request = RequestTraining(
            "service-request",
            operation(960050),
            operation(960051),
            operation(960052),
        )
        commands = tuple(
            ApplicationCommand(
                operation(960020 + ordinal),
                str(day.id),
                payload,
            )
            for ordinal, payload in enumerate(
                (
                    DiscoverProgramme("official", programme),
                    CollectCardsAndForm(),
                    CollectAdaptiveOdds(),
                    CloseAndSeal(),
                    CommitDeferredPrediction(),
                    CollectResults(),
                    JoinTrainingExamples(),
                    ReconcileRacingDay(),
                    request,
                )
            )
        )
        cycle = RacingDayCycle(
            str(day.id),
            commands,
            operation(960010),
            tuple(operation(960030 + ordinal) for ordinal in range(9)),
            NOW,
        )
        handlers = QuarantinedDayHandlers(
            self.authority,
            day=day,
            programme_checksum=programme,
            release_id="candidate",
            request=request,
            base=960100,
        )

        class Adapter:
            def __init__(self, registrations, next_cycle=cycle):
                self._registrations = registrations
                self._next_cycle = next_cycle

            def registrations(self):
                return self._registrations

            def next_cycle(self, *, now):
                result, self._next_cycle = self._next_cycle, None
                return result

            def close(self):
                pass

        complete = handlers.registrations()
        operation_count = self.store.count("operations")
        for invalid in (
            complete[:-1],
            (*complete, complete[0]),
            (*complete[:-1], PhaseHandlerRegistration(str, handlers.excluded)),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ServiceUnavailable):
                    ServiceComposition(
                        self.configuration(),
                        self.store,
                        self.artifacts,
                        Adapter(invalid),
                        owner="invalid",
                        token="invalid",
                        lease_ttl=timedelta(minutes=5),
                    )
                self.assertEqual(self.store.count("operations"), operation_count)

        handlers.at = NOW - timedelta(seconds=2)
        initial_generation = self.authority.acquire_lease(
            operation(960003),
            owner="service-before-restart",
            token="service-before-restart-token",
            now=handlers.at,
            ttl=timedelta(seconds=1),
        )
        initial_authority = OperationalAuthority(
            self.store,
            self.artifacts,
            command_executor=handlers.dispatcher(),
            clock=lambda: handlers.at,
        )
        initial_authority.plan_racing_day(
            cycle.plan_operation_id,
            racing_day_id=str(day.id),
            lease_token="service-before-restart-token",
            lease_generation=initial_generation,
            commands=commands,
            at=handlers.at,
        )
        RaceCollectionService(
            initial_authority,
            token="service-before-restart-token",
            generation=initial_generation,
        ).advance(
            cycle.advancement_operation_ids[0],
            racing_day_id=str(day.id),
            phase=commands[0].phase,
            now=handlers.at,
            command=commands[0],
        )
        handlers.at = NOW
        composition = ServiceComposition(
            self.configuration(),
            self.store,
            self.artifacts,
            Adapter(complete),
            owner="service",
            token="service-token",
            lease_ttl=timedelta(minutes=5),
        )
        exit_code = main(
            ("--config", "/unused/synthetic.json", "--once"),
            composition_loader=lambda *_args, **_kwargs: composition,
            token_factory=lambda: "service-token",
        )
        self.assertEqual(exit_code, 0)
        with self.store._connect() as db:
            self.assertEqual(
                db.execute(
                    "SELECT count(*) FROM phase7_scheduler_progress WHERE racing_day_id=?",
                    (str(day.id),),
                ).fetchone()[0],
                9,
            )
            self.assertEqual(
                db.execute(
                    "SELECT count(*) FROM phase7_application_command_receipts "
                    "WHERE racing_day_id=?",
                    (str(day.id),),
                ).fetchone()[0],
                9,
            )
            self.assertIsNotNone(
                db.execute(
                    "SELECT 1 FROM phase7_day_training_requests WHERE racing_day_id=?",
                    (str(day.id),),
                ).fetchone()
            )
            self.assertEqual(
                db.execute("SELECT count(*) FROM phase6_promotion_records").fetchone()[0],
                0,
            )
            self.assertEqual(
                db.execute(
                    "SELECT count(*) FROM phase6_runs WHERE run_kind IN ('training','tuning')"
                ).fetchone()[0],
                0,
            )
        self.assertEqual(len(composition.run_cycle(cycle)), 9)
        competing = ServiceComposition(
            self.configuration(),
            self.store,
            self.artifacts,
            Adapter(complete, cycle),
            owner="competitor",
            token="competitor-token",
            lease_ttl=timedelta(minutes=5),
        )
        with self.assertRaises(OperationalRejected):
            competing.run_cycle(cycle)

    def test_restart_prefix_rejects_receipt_timing_corruption(self) -> None:
        base = 961000
        service, handlers, commands = self.trusted_scheduler_fixture(base)
        command = commands[0]
        service.advance(
            operation(base + 70),
            racing_day_id=command.racing_day_id,
            phase=command.phase,
            now=handlers.at,
            command=command,
        )

        class Adapter:
            def registrations(self):
                return handlers.registrations()

            def next_cycle(self, *, now):
                return None

            def close(self):
                pass

        composition = ServiceComposition(
            self.configuration(),
            self.store,
            self.artifacts,
            Adapter(),
            owner="prefix",
            token=service.token,
            lease_ttl=timedelta(minutes=5),
            clock=lambda: handlers.at,
        )
        composition.generation = service.generation
        cycle = RacingDayCycle(
            command.racing_day_id,
            commands,
            operation(base + 4),
            tuple(operation(base + 70 + ordinal) for ordinal in range(9)),
            handlers.at,
        )
        with self.store._connect() as db:
            db.execute("DROP TRIGGER phase7_receipt_append_only_update")
            db.execute(
                "UPDATE phase7_application_command_receipts SET committed_at=? "
                "WHERE command_operation_id=?",
                (
                    (handlers.at + timedelta(seconds=1)).isoformat(),
                    str(command.operation_id),
                ),
            )
        with self.assertRaisesRegex(OperationalRejected, "completed prefix"):
            composition.run_cycle(cycle)

    def test_trusted_clock_rejects_future_schedule_and_monotonizes_regression(
        self,
    ) -> None:
        handler = lambda command, at: None

        class Adapter:
            def registrations(self):
                return tuple(
                    PhaseHandlerRegistration(payload_type, handler)
                    for payload_type in (
                        DiscoverProgramme,
                        CollectCardsAndForm,
                        CollectAdaptiveOdds,
                        CloseAndSeal,
                        CommitDeferredPrediction,
                        CollectResults,
                        JoinTrainingExamples,
                        ReconcileRacingDay,
                        RequestTraining,
                    )
                )

            def next_cycle(self, *, now):
                return None

            def close(self):
                pass

        clock_values = iter((NOW, NOW - timedelta(minutes=1), NOW))
        composition = ServiceComposition(
            self.configuration(),
            self.store,
            self.artifacts,
            Adapter(),
            owner="clock",
            token="clock-token",
            lease_ttl=timedelta(minutes=5),
            clock=lambda: next(clock_values),
        )
        first = composition.trusted_timestamp()
        second = composition.trusted_timestamp()
        self.assertEqual(second, first + timedelta(microseconds=1))
        day_id = "day_" + "c" * 32
        commands = tuple(
            ApplicationCommand(operation(962000 + index), day_id, command_payload(phase))
            for index, phase in enumerate(RaceCollectionService.ORDER)
        )
        cycle = RacingDayCycle(
            day_id,
            commands,
            operation(962020),
            tuple(operation(962030 + index) for index in range(9)),
            NOW + timedelta(minutes=1),
        )
        with self.assertRaisesRegex(OperationalRejected, "schedule time is in the future"):
            composition.run_cycle(cycle)
        with self.store._connect() as db:
            self.assertEqual(
                db.execute("SELECT count(*) FROM phase7_scheduler_history").fetchone()[0],
                0,
            )

    def test_restart_proof_requires_later_generation_adoption(self) -> None:
        service, handlers, commands = self.trusted_scheduler_fixture(
            962500, recovered_after_discovery=True
        )
        with self.store._connect() as db:
            progress = [
                dict(row)
                for row in db.execute(
                    "SELECT phase_ordinal,phase_name,lease_generation,"
                    "command_operation_id,result_checksum "
                    "FROM phase7_scheduler_progress WHERE racing_day_id=? "
                    "ORDER BY phase_ordinal",
                    (commands[0].racing_day_id,),
                )
            ]
        for evidence_operation_id, adoption_id, generation, reason in (
            (
                operation(962598),
                str(operation(962599)),
                service.generation,
                "no durable recovered prefix",
            ),
            (
                operation(962600),
                str(operation(962507)),
                progress[0]["lease_generation"],
                "disagrees with scheduler state",
            ),
        ):
            checks = {
                "adoption_operation_id": adoption_id,
                "lease_generation": generation,
                "recovered_phase": 1,
                "state_checksum": str(_checksum(progress)),
            }
            proof = self.artifacts.put(
                json.dumps(
                    {
                        "schema_version": "phase7-check-proof-v1",
                        "evidence_kind": "restart",
                        "racing_day_id": commands[0].racing_day_id,
                        "release_id": handlers.release_id,
                        "checks": checks,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode(),
                media_type="application/json",
            )
            with self.assertRaisesRegex(OperationalRejected, reason):
                self.authority.record_operational_evidence(
                    evidence_operation_id,
                    checksum=proof.checksum,
                    evidence_kind="restart",
                    racing_day_id=commands[0].racing_day_id,
                    release_id=handlers.release_id,
                    checks=checks,
                    at=handlers.at,
                )

    def test_result_rejection_resumes_the_unchanged_planned_command(self) -> None:
        service, handlers, commands = self.trusted_scheduler_fixture(
            962300, recovered_after_discovery=True
        )
        service.advance(
            operation(962371),
            racing_day_id=commands[0].racing_day_id,
            phase=commands[0].phase,
            now=handlers.at,
            command=commands[0],
        )
        self.assertIsNotNone(handlers.race_id)
        planned_result = commands[5]
        rejection = _derived_operation_id(
            "result-before-prediction-attempt-v1",
            str(planned_result.operation_id),
            planned_result.racing_day_id,
            str(handlers.race_id),
        )
        with self.assertRaises(BarrierNotSatisfied):
            self.authority.reject_result_before_prediction(
                rejection,
                racing_day_id=planned_result.racing_day_id,
                race_id=str(handlers.race_id),
                at=handlers.at,
            )
        with self.store._connect() as db:
            self.assertIsNone(
                db.execute(
                    "SELECT 1 FROM operations WHERE operation_id=?",
                    (str(planned_result.operation_id),),
                ).fetchone()
            )
        self.authority.resolve_alert(
            operation(962390),
            alert_id=f"result-rejection:{rejection}",
            actor="operator",
            reason="prediction barrier restored",
            at=handlers.at,
        )
        self.authority.admin_pause(
            operation(962391),
            scope="results",
            paused=False,
            actor="operator",
            reason="resume unchanged plan",
            at=handlers.at,
        )
        for ordinal, command in enumerate(commands[1:], 2):
            service.advance(
                operation(963300 + ordinal),
                racing_day_id=command.racing_day_id,
                phase=command.phase,
                now=handlers.at,
                command=command,
            )
        with self.store._connect() as db:
            self.assertEqual(
                db.execute(
                    "SELECT kind FROM operations WHERE operation_id=?",
                    (str(planned_result.operation_id),),
                ).fetchone()[0],
                "phase7_command_collect_results",
            )
            self.assertEqual(
                db.execute(
                    "SELECT command_operation_id FROM phase7_day_command_plan "
                    "WHERE racing_day_id=? AND phase_name='collect_results'",
                    (planned_result.racing_day_id,),
                ).fetchone()[0],
                str(planned_result.operation_id),
            )

    def test_determinism_proof_rejects_one_execution_submitted_twice(self) -> None:
        _service, handlers, commands = self.trusted_scheduler_fixture(962700)
        input_artifact = self.artifacts.put(b"distinct-input", media_type="application/json")
        execution = operation(962790)
        output_checksum = self.authority.record_determinism_execution(
            execution,
            racing_day_id=commands[0].racing_day_id,
            release_id=handlers.release_id,
            input_checksum=input_artifact.checksum,
            at=handlers.at,
        )
        checks = {
            "input_checksum": str(input_artifact.checksum),
            "first_execution_operation_id": str(execution),
            "output_checksum": str(output_checksum),
            "replay_execution_operation_id": str(execution),
            "replay_output_checksum": str(output_checksum),
        }
        proof = self.artifacts.put(
            json.dumps(
                {
                    "schema_version": "phase7-check-proof-v1",
                    "evidence_kind": "determinism",
                    "racing_day_id": commands[0].racing_day_id,
                    "release_id": handlers.release_id,
                    "checks": checks,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
            media_type="application/json",
        )
        with self.assertRaises(OperationalRejected):
            self.authority.record_operational_evidence(
                operation(962791),
                checksum=proof.checksum,
                evidence_kind="determinism",
                racing_day_id=commands[0].racing_day_id,
                release_id=handlers.release_id,
                checks=checks,
                at=handlers.at,
            )

    def test_phase7_release_manifest_rejects_future_database_schema(self) -> None:
        with self.assertRaisesRegex(ValueError, "release schema"):
            ReleaseManifest(
                "phase7-release-v1",
                "future-schema",
                "6cd5dacfe83719cbbc376a265829e84593eafb68",
                PROGRAMME,
                29,
                "canonical-artifacts-v1",
                "phase6-promotion-v1",
                ("runner-win-probability-v1",),
                "/opt/race-collection/current",
            )

    def test_lost_lease_requires_fresh_composition_token_and_authoritative_expiry(
        self,
    ) -> None:
        class Adapter:
            def registrations(self):
                def no_op(_command, _at):
                    return None

                return tuple(
                    PhaseHandlerRegistration(payload_type, no_op)
                    for payload_type in (
                        DiscoverProgramme,
                        CollectCardsAndForm,
                        CollectAdaptiveOdds,
                        CloseAndSeal,
                        CommitDeferredPrediction,
                        CollectResults,
                        JoinTrainingExamples,
                        ReconcileRacingDay,
                        RequestTraining,
                    )
                )

            def next_cycle(self, *, now):
                return None

            def close(self):
                pass

        def composition(owner, token):
            return ServiceComposition(
                self.configuration(),
                self.store,
                self.artifacts,
                Adapter(),
                owner=owner,
                token=token,
                lease_ttl=timedelta(seconds=5),
            )

        old = composition("old-process", "old-process-token")
        self.assertEqual(old.maintain_lease(NOW), 1)
        fresh = composition("fresh-process", "fresh-process-token")
        with self.assertRaisesRegex(OperationalRejected, "another live generation"):
            fresh.maintain_lease(NOW + timedelta(seconds=4))

        self.assertEqual(fresh.maintain_lease(NOW + timedelta(seconds=5)), 2)
        with self.assertRaisesRegex(
            OperationalRejected,
            "scheduler lease renewal is stale or fenced",
        ):
            old.maintain_lease(NOW + timedelta(seconds=5, microseconds=1))
        self.assertEqual(old.generation, 1)
        with self.store._connect() as db:
            self.assertEqual(
                db.execute(
                    "SELECT count(*) FROM phase7_scheduler_history "
                    "WHERE lease_token='old-process-token'"
                ).fetchone()[0],
                1,
            )
            self.assertEqual(
                db.execute(
                    "SELECT lease_token FROM phase7_scheduler_lease WHERE singleton=1"
                ).fetchone()[0],
                "fresh-process-token",
            )

        with self.assertRaises(ValueError):
            composition(123, "typed-token")
        with self.assertRaises(ValueError):
            composition("typed-owner", 123)

    def test_migrated_partial_day_is_adopted_and_continued_through_service_boundary(
        self,
    ) -> None:
        root = Path(self.temporary.name) / "migrated-public-continuation"
        store = SQLiteOperationsStore(root / "operations.sqlite3")
        scripts = store._migration_scripts()
        store._migration_scripts = lambda: tuple(item for item in scripts if item[0] <= 27)
        store.migrate()
        artifacts = LocalArtifactStore(root / "artifacts")
        authority = OperationalAuthority(store, artifacts)
        EvaluationAuthority(store, artifacts).register_policy(
            operation(970000), PromotionPolicy(), NOW - timedelta(days=1)
        )
        configuration = self.configuration()
        config_checksum = authority.register_configuration(operation(970001), configuration, NOW)
        programme = artifacts.put(
            b'{"programme":"migrated-partial-day"}',
            media_type="application/json",
        ).checksum
        day = RacingDay(RacingDayId("day_" + f"{970003:032x}"), NOW.date(), "UTC", NOW)
        store.create_racing_day(operation(970003), day)
        request = RequestTraining(
            "migrated-request",
            operation(970090),
            operation(970091),
            operation(970092),
        )
        handlers = QuarantinedDayHandlers(
            authority,
            day=day,
            programme_checksum=programme,
            release_id="candidate",
            request=request,
            base=970100,
        )
        generation = authority.acquire_lease(
            operation(970004),
            owner="v27-process",
            token="v27-token",
            now=NOW,
            ttl=timedelta(hours=1),
        )
        discover = ApplicationCommand(
            operation(970005),
            str(day.id),
            DiscoverProgramme("official", programme),
        )
        handlers.discover(discover, NOW)
        with store._connect() as db:
            result = authority._phase_postcondition(db, discover)
        result_json = json.dumps(result, sort_keys=True, separators=(",", ":"))
        result_checksum = str(_checksum(result))
        with store._operation(discover.operation_id, "phase7_command_discover_programme", {}) as (
            db,
            _,
        ):
            db.execute(
                "INSERT INTO phase7_application_command_receipts VALUES(?,?,?,?,?,?)",
                (
                    str(discover.operation_id),
                    str(day.id),
                    discover.phase,
                    result_json,
                    result_checksum,
                    NOW.isoformat(timespec="microseconds"),
                ),
            )
        with store._operation(operation(970006), "phase7_advance_phase", {}) as (db, _):
            db.execute(
                "INSERT INTO phase7_scheduler_progress VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    str(day.id),
                    1,
                    discover.phase,
                    generation,
                    str(discover.operation_id),
                    result_json,
                    result_checksum,
                    NOW.isoformat(timespec="microseconds"),
                    str(operation(970006)),
                ),
            )

        del store._migration_scripts
        store.migrate()
        authority.register_release(
            operation(970002),
            ReleaseManifest(
                "phase7-release-v1",
                "candidate",
                "6cd5dacfe83719cbbc376a265829e84593eafb68",
                config_checksum,
                28,
                "canonical-artifacts-v1",
                "phase6-promotion-v1",
                ("runner-win-probability-v1",),
                configuration.service_root,
            ),
            NOW,
        )
        with store._connect() as db:
            plan = db.execute(
                "SELECT phase_ordinal,phase_name,command_operation_id,operation_id "
                "FROM phase7_day_command_plan WHERE racing_day_id=? ORDER BY phase_ordinal",
                (str(day.id),),
            ).fetchall()
        payloads = (
            discover.payload,
            CollectCardsAndForm(),
            CollectAdaptiveOdds(),
            CloseAndSeal(),
            CommitDeferredPrediction(),
            CollectResults(),
            JoinTrainingExamples(),
            ReconcileRacingDay(),
            request,
        )
        commands = tuple(
            ApplicationCommand(OperationId(row["command_operation_id"]), str(day.id), payload)
            for row, payload in zip(plan, payloads, strict=True)
        )
        takeover_at = NOW + timedelta(hours=1)
        handlers.at = takeover_at

        class Adapter:
            def registrations(self):
                return handlers.registrations()

            def next_cycle(self, *, now):
                return None

            def close(self):
                pass

        sampled_under_operation_lock = []

        def adoption_clock():
            probe = store._connect()
            probe.execute("PRAGMA busy_timeout = 0")
            try:
                try:
                    probe.execute("BEGIN IMMEDIATE")
                except sqlite3.OperationalError:
                    sampled_under_operation_lock.append(True)
                else:
                    probe.rollback()
            finally:
                probe.close()
            return takeover_at

        composition = ServiceComposition(
            configuration,
            store,
            artifacts,
            Adapter(),
            owner="v28-process",
            token="v28-token",
            lease_ttl=timedelta(hours=1),
            clock=adoption_clock,
        )
        mismatch = list(commands)
        mismatch[4] = ApplicationCommand(operation(970500), str(day.id), CommitDeferredPrediction())
        mismatched_cycle = RacingDayCycle(
            str(day.id),
            tuple(mismatch),
            operation(970501),
            tuple(operation(970510 + ordinal) for ordinal in range(9)),
            takeover_at,
        )
        with self.assertRaisesRegex(OperationalRejected, "different command identities"):
            composition.run_cycle(mismatched_cycle)
        self.assertEqual(handlers.executions, {"discover_programme": 1})

        cycle = RacingDayCycle(
            str(day.id),
            commands,
            OperationId(plan[0]["operation_id"]),
            tuple(operation(970610 + ordinal) for ordinal in range(9)),
            takeover_at,
        )
        results = composition.run_cycle(cycle)
        self.assertTrue(sampled_under_operation_lock)
        self.assertEqual(len(results), 9)
        self.assertEqual(handlers.executions["discover_programme"], 1)
        self.assertEqual(set(handlers.executions), set(RaceCollectionService.ORDER))
        with store._connect() as db:
            progress = db.execute(
                "SELECT phase_ordinal,lease_generation,command_operation_id "
                "FROM phase7_scheduler_progress WHERE racing_day_id=? ORDER BY phase_ordinal",
                (str(day.id),),
            ).fetchall()
            self.assertEqual([row["phase_ordinal"] for row in progress], list(range(1, 10)))
            self.assertEqual(progress[0]["lease_generation"], 1)
            self.assertTrue(all(row["lease_generation"] == 2 for row in progress[1:]))
            self.assertEqual(
                [row["command_operation_id"] for row in progress],
                [str(command.operation_id) for command in commands],
            )
            claims = db.execute(
                "SELECT lease_generation,lease_token FROM phase7_application_command_claims "
                "WHERE racing_day_id=? ORDER BY claimed_at",
                (str(day.id),),
            ).fetchall()
            self.assertEqual(len(claims), 8)
            self.assertTrue(
                all(
                    row["lease_generation"] == 2 and row["lease_token"] == "v28-token"
                    for row in claims
                )
            )
            self.assertEqual(
                db.execute(
                    "SELECT count(*) FROM operations WHERE kind='phase7_plan_racing_day'"
                ).fetchone()[0],
                0,
            )
            self.assertIsNotNone(
                db.execute(
                    "SELECT 1 FROM operations "
                    "WHERE kind='phase7_adopt_migrated_day_command_plan'",
                ).fetchone()
            )
            adoption_operation_id = db.execute(
                "SELECT operation_id FROM phase7_day_plan_adoptions "
                "WHERE racing_day_id=? AND lease_generation=2",
                (str(day.id),),
            ).fetchone()[0]
            self.assertNotEqual(adoption_operation_id, plan[0]["operation_id"])
        samples_before_replay = len(sampled_under_operation_lock)
        self.assertFalse(
            composition.authority.plan_racing_day(
                OperationId(adoption_operation_id),
                racing_day_id=str(day.id),
                lease_token="v28-token",
                lease_generation=2,
                commands=commands,
                at=takeover_at + timedelta(days=1),
            )
        )
        self.assertEqual(len(sampled_under_operation_lock), samples_before_replay + 1)

    def test_direct_sql_probation_forgery_is_rejected(self) -> None:
        checksum = "sha256:" + "a" * 64
        with (
            self.store._connect() as db,
            self.assertRaises(
                sqlite3.IntegrityError,
                msg="forged Phase 6 input must lack Phase 7 authority",
            ),
        ):
            db.execute("PRAGMA foreign_keys=OFF")
            db.execute(
                "INSERT INTO phase6_probation_days VALUES(?,?,?,?,?,?,1)",
                ("forged", "2026-07-01", checksum, checksum, checksum, checksum),
            )

    def test_direct_sql_day_evidence_checksum_shapes_are_not_authority(self) -> None:
        checksum = "sha256:" + "b" * 64
        with self.store._operation(operation(32), "forged_source", {}) as (db, _):
            db.execute(
                "INSERT INTO racing_days VALUES(?,?,?,?,?)",
                (
                    "day_" + "3" * 32,
                    "2026-07-20",
                    "UTC",
                    NOW.isoformat(),
                    NOW.isoformat(),
                ),
            )
            with self.assertRaises(sqlite3.IntegrityError, msg="shape alone must not authenticate"):
                db.execute(
                    "INSERT INTO phase7_day_evidence VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        "day_" + "3" * 32,
                        checksum,
                        checksum,
                        checksum,
                        checksum,
                        1,
                        0,
                        "missing-release",
                        NOW.isoformat(),
                        str(operation(32)),
                    ),
                )

    def test_critical_alert_pauses_only_affected_scope_and_probation(self) -> None:
        self.authority.raise_alert(
            operation(30),
            alert_id="checksum-1",
            category="checksum_failure",
            details="synthetic corrupt artifact",
            racing_day_id=None,
            at=NOW,
        )
        with self.store._connect() as db:
            self.assertEqual(
                db.execute("SELECT paused FROM phase7_pauses WHERE scope='promotion'").fetchone()[
                    0
                ],
                1,
            )
            self.assertEqual(
                db.execute("SELECT state FROM phase7_probation_control").fetchone()[0],
                "paused",
            )
            self.assertIsNone(
                db.execute("SELECT paused FROM phase7_pauses WHERE scope='results'").fetchone()
            )
        with self.assertRaisesRegex(OperationalRejected, "unresolved"):
            self.authority.admin_pause(
                operation(31000),
                scope="promotion",
                paused=False,
                actor="operator",
                reason="unsafe early resume",
                at=NOW,
            )
        with self.store._connect() as db, self.assertRaises(sqlite3.IntegrityError):
            db.execute(
                "UPDATE phase7_alerts SET resolved_at=? WHERE alert_id='checksum-1'",
                (NOW.isoformat(),),
            )
        self.assertTrue(
            self.authority.resolve_alert(
                operation(31001),
                alert_id="checksum-1",
                actor="operator",
                reason="artifact restored and verified",
                at=NOW,
            )
        )
        self.assertTrue(
            self.authority.admin_pause(
                operation(31002),
                scope="promotion",
                paused=False,
                actor="operator",
                reason="resolved alert permits explicit resume",
                at=NOW,
            )
        )
        with self.store._connect() as db:
            audit = db.execute(
                "SELECT command FROM phase7_admin_audit WHERE operation_id=?",
                (str(operation(31001)),),
            ).fetchone()
            self.assertEqual(audit[0], "resolve_alert")
        self.assertEqual(
            self.authority.reset_probation(
                operation(31),
                actor="operator",
                reason="discard failed generation",
                at=NOW,
            ),
            2,
        )

    def test_backup_and_restore_require_integrity_not_command_success(self) -> None:
        service, handlers, commands = self.trusted_scheduler_fixture(
            860000, recovered_after_discovery=True
        )
        for ordinal, command in enumerate(commands, 1):
            service.advance(
                operation(860070 + ordinal),
                racing_day_id=command.racing_day_id,
                phase=command.phase,
                now=handlers.at,
                command=command,
            )
        day_id = commands[0].racing_day_id
        with self.store._connect() as db:
            evidence = [
                self.artifacts.verify(ArtifactChecksum(value))
                for value in db.execute(
                    "SELECT reconciliation_checksum,restart_checksum,ordering_checksum,"
                    "determinism_checksum FROM phase7_day_evidence "
                    "WHERE racing_day_id=?",
                    (day_id,),
                ).fetchone()
            ]
        root = Path(self.temporary.name)
        recovery = RecoveryAuthority(self.store, self.artifacts)
        replica = LocalArtifactStore(root / "replica")
        snapshot = root / "isolated" / "snapshot.sqlite3"
        checksum = recovery.backup(
            operation(42),
            backup_id="backup-1",
            racing_day_id=day_id,
            snapshot_path=snapshot,
            replica=replica,
            at=NOW,
        )
        self.assertEqual(
            checksum,
            ArtifactChecksum(
                "sha256:" + __import__("hashlib").sha256(snapshot.read_bytes()).hexdigest()
            ),
        )
        self.assertTrue(
            recovery.restore_drill(
                operation(43),
                drill_id="drill-good",
                backup_id="backup-1",
                snapshot_path=snapshot,
                replica=replica,
                at=NOW,
            )
        )
        with self.store._connect() as db:
            first_backup = db.execute(
                "SELECT * FROM phase7_backups WHERE backup_id='backup-1'"
            ).fetchone()
        inventory = json.loads(
            replica.read(ArtifactChecksum(first_backup["artifact_inventory_checksum"]))
        )
        inventory_checksum = ArtifactChecksum(first_backup["artifact_inventory_checksum"])

        class HostileInventoryReplica:
            """Serve hostile inventory bytes without weakening artifact verification."""

            def __init__(self, payload):
                self.payload = payload

            def read(self, checksum):
                if checksum == inventory_checksum:
                    return self.payload
                return replica.read(checksum)

            def verify(self, checksum):
                return replica.verify(checksum)

        hostile_inventories = {
            "wrong-top-level": b"{}",
            "non-string-member": b"[1]",
            "invalid-checksum": b'["not-a-checksum"]',
            "duplicate": json.dumps([inventory[0], inventory[0]], separators=(",", ":")).encode(),
            "unordered": json.dumps(list(reversed(inventory)), separators=(",", ":")).encode(),
            "missing-reference": json.dumps(inventory[:-1], separators=(",", ":")).encode(),
            "extra-reference": json.dumps(
                sorted([*inventory, "sha256:" + "f" * 64]), separators=(",", ":")
            ).encode(),
        }
        for offset, (attack, payload) in enumerate(hostile_inventories.items(), 43000):
            with self.subTest(recorded_inventory_attack=attack):
                self.assertFalse(
                    recovery.restore_drill(
                        operation(offset),
                        drill_id=f"drill-inventory-{attack}",
                        backup_id="backup-1",
                        snapshot_path=snapshot,
                        replica=HostileInventoryReplica(payload),
                        at=NOW,
                    )
                )
                with self.store._connect() as db:
                    drill = db.execute(
                        "SELECT database_verified,artifacts_verified,"
                        "application_readable,successful "
                        "FROM phase7_restore_drills WHERE drill_id=?",
                        (f"drill-inventory-{attack}",),
                    ).fetchone()
                self.assertEqual(tuple(drill), (1, 0, 0, 0))
                # A leaked read-only recovered connection would prevent an
                # immediate exclusive lock on platforms used by the service.
                probe = sqlite3.connect(snapshot, timeout=0)
                try:
                    probe.execute("BEGIN EXCLUSIVE")
                    probe.rollback()
                finally:
                    probe.close()
        self.assertNotIn(
            ("phase6_evaluation_evidence", "population_checksum"),
            RELATIONAL_ARTIFACT_REFERENCES,
        )
        self.assertIn(
            ("training_examples", "artifact_checksum"),
            RELATIONAL_ARTIFACT_REFERENCES,
        )
        self.assertNotIn(
            ("canonical_training_examples", "feature_matrix_checksum"),
            RELATIONAL_ARTIFACT_REFERENCES,
        )
        with sqlite3.connect(snapshot) as snapshot_db:
            expected = set(artifact_inventory(snapshot_db))
        self.assertEqual(set(inventory), expected)
        for value in expected:
            metadata = replica.verify(ArtifactChecksum(value))
            self.assertEqual(str(metadata.checksum), value)
        self.assertNotIn(first_backup["database_checksum"], inventory)
        self.assertNotIn(first_backup["artifact_inventory_checksum"], inventory)

        second_snapshot = root / "isolated" / "snapshot-second.sqlite3"
        recovery.backup(
            operation(71),
            backup_id="backup-2",
            racing_day_id=day_id,
            snapshot_path=second_snapshot,
            replica=replica,
            at=NOW,
        )
        with self.store._connect() as db:
            second_backup = db.execute(
                "SELECT * FROM phase7_backups WHERE backup_id='backup-2'"
            ).fetchone()
        second_inventory = json.loads(
            replica.read(ArtifactChecksum(second_backup["artifact_inventory_checksum"]))
        )
        self.assertNotIn(first_backup["database_checksum"], second_inventory)
        self.assertNotIn(first_backup["artifact_inventory_checksum"], second_inventory)

        missing = evidence[0].checksum
        missing_content = replica.read(missing)
        replica.path_for(missing).unlink()
        self.assertFalse(
            recovery.restore_drill(
                operation(72),
                drill_id="drill-missing",
                backup_id="backup-1",
                snapshot_path=snapshot,
                replica=replica,
                at=NOW,
            )
        )

        replica.put(missing_content, media_type="application/json", expected_checksum=missing)
        replica.path_for(missing).write_bytes(b"corrupt")
        self.assertFalse(
            recovery.restore_drill(
                operation(73),
                drill_id="drill-corrupt-artifact",
                backup_id="backup-1",
                snapshot_path=snapshot,
                replica=replica,
                at=NOW,
            )
        )
        replica.path_for(missing).write_bytes(missing_content)

        primary_missing = evidence[1].checksum
        primary_content = self.artifacts.read(primary_missing)
        self.artifacts.path_for(primary_missing).unlink()
        with self.assertRaises(ArtifactStoreError):
            recovery.backup(
                operation(74),
                backup_id="backup-primary-missing",
                racing_day_id=day_id,
                snapshot_path=root / "isolated" / "primary-missing.sqlite3",
                replica=replica,
                at=NOW,
            )
        with self.store._connect() as db:
            self.assertIsNone(
                db.execute(
                    "SELECT 1 FROM phase7_backups WHERE backup_id='backup-primary-missing'"
                ).fetchone()
            )
        self.artifacts.put(
            primary_content,
            media_type="application/json",
            expected_checksum=primary_missing,
        )

        self.artifacts.path_for(primary_missing).write_bytes(b"corrupt-primary")
        with self.assertRaises(ArtifactStoreError):
            recovery.backup(
                operation(75),
                backup_id="backup-primary-corrupt",
                racing_day_id=day_id,
                snapshot_path=root / "isolated" / "primary-corrupt.sqlite3",
                replica=replica,
                at=NOW,
            )
        with self.store._connect() as db:
            self.assertIsNone(
                db.execute(
                    "SELECT 1 FROM phase7_backups WHERE backup_id='backup-primary-corrupt'"
                ).fetchone()
            )
        self.artifacts.path_for(primary_missing).write_bytes(primary_content)
        snapshot.write_bytes(snapshot.read_bytes() + b"corrupt")
        self.assertFalse(
            recovery.restore_drill(
                operation(44),
                drill_id="drill-bad",
                backup_id="backup-1",
                snapshot_path=snapshot,
                replica=replica,
                at=NOW,
            )
        )

    def test_cross_phase_artifact_contract_uses_readable_producer_objects(self) -> None:
        test_root = str(Path(__file__).parent)
        sys.path.insert(0, test_root)
        try:
            from test_phase6_evaluation_promotion import (
                _build_authentic_promotion_template,
            )
        finally:
            sys.path.remove(test_root)
        root = Path(self.temporary.name) / "cross-phase"
        store, artifacts = _build_authentic_promotion_template(root, count=1, artifact_backed=True)
        EvaluationAuthority(store, artifacts).register_policy(
            operation(89999), PromotionPolicy(), NOW - timedelta(days=1)
        )
        authority = OperationalAuthority(store, artifacts)
        config_checksum = authority.register_configuration(
            operation(90000), self.configuration(), NOW
        )
        authority.register_release(
            operation(90001),
            ReleaseManifest(
                "phase7-release-v1",
                "cross-phase-release",
                "6cd5dacfe83719cbbc376a265829e84593eafb68",
                config_checksum,
                28,
                "canonical-artifacts-v1",
                "phase6-promotion-v1",
                ("runner-win-probability-v1",),
                "/opt/race-collection/current",
            ),
            NOW,
        )
        with store._connect() as db:
            inventory = set(artifact_inventory(db))
            required = {
                row[0]
                for row in db.execute(
                    "SELECT bundle_checksum FROM canonical_model_bundles "
                    "UNION SELECT artifact_checksum FROM canonical_bundle_components"
                )
            }
        self.assertTrue(required <= inventory)
        for value in inventory:
            self.assertTrue(artifacts.read(ArtifactChecksum(value)))
        self.assertNotIn("sha256:" + "9" * 64, inventory)
        self.assertNotIn(
            ("phase6_evaluation_evidence", "population_checksum"),
            RELATIONAL_ARTIFACT_REFERENCES,
        )
        self.assertNotIn(
            ("phase7_scheduler_progress", "result_checksum"),
            RELATIONAL_ARTIFACT_REFERENCES,
        )
        self.assertNotIn(
            ("phase7_backups", "database_checksum"),
            RELATIONAL_ARTIFACT_REFERENCES,
        )

    def test_artifact_inventory_fails_closed_for_unknown_future_schema(self) -> None:
        with self.store._connect() as db:
            db.execute(
                "INSERT INTO schema_migrations VALUES(29,?,?)",
                ("future-contract-unknown", NOW.isoformat()),
            )
            with self.assertRaisesRegex(RecoveryRejected, "does not cover"):
                artifact_inventory(db)

    def test_real_scheduler_receipts_reauthenticate_operational_evidence(self) -> None:
        service, handlers, commands = self.trusted_scheduler_fixture(
            850000, recovered_after_discovery=True
        )
        for ordinal, command in enumerate(commands, 1):
            service.advance(
                operation(850070 + ordinal),
                racing_day_id=command.racing_day_id,
                phase=command.phase,
                now=handlers.at,
                command=command,
            )
        with self.store._connect() as db:
            evidence = db.execute(
                "SELECT release_id,restart_checksum,ordering_checksum,determinism_checksum "
                "FROM phase7_day_evidence WHERE racing_day_id=?",
                (commands[0].racing_day_id,),
            ).fetchone()
            for kind in ("restart", "ordering", "determinism"):
                self.authority.verify_operational_evidence(
                    db,
                    racing_day_id=commands[0].racing_day_id,
                    release_id=evidence["release_id"],
                    evidence_kind=kind,
                    checksum=ArtifactChecksum(evidence[f"{kind}_checksum"]),
                )

    def test_training_request_receipt_rejects_foreign_duplicate_and_replay_conflict(
        self,
    ) -> None:
        service, handlers, commands = self.trusted_scheduler_fixture(
            851000, recovered_after_discovery=True
        )
        for ordinal, command in enumerate(commands, 1):
            service.advance(
                operation(851070 + ordinal),
                racing_day_id=command.racing_day_id,
                phase=command.phase,
                now=handlers.at,
                command=command,
            )
        request_command = commands[-1]
        with self.assertRaises(ConflictingOperation):
            service.authority.execute_application_command(
                ApplicationCommand(
                    request_command.operation_id,
                    request_command.racing_day_id,
                    RequestTraining(
                        "foreign-request",
                        request_command.payload.request_operation_id,
                        request_command.payload.authorization_operation_id,
                        request_command.payload.binding_operation_id,
                    ),
                ),
                token=service.token,
                generation=service.generation,
                at=handlers.at,
            )
        self.assertEqual(handlers.executions["request_training"], 1)

        foreign_run = operation(851200)
        foreign_request_operation = operation(851201)
        foreign_binding_operation = operation(851202)
        EvaluationAuthority(self.store, self.artifacts).begin_run(
            foreign_run,
            run_kind="forecast_service",
            started_at=handlers.at,
        )
        CanonicalForecastService(
            ChampionLoader(self.store, self.artifacts, deserializer=lambda content: content),
            self.artifacts,
        ).emit_training_request(
            foreign_request_operation,
            request_id="foreign-request",
            reason="unrelated request",
            requested_at=handlers.at,
            service_run_id=foreign_run,
        )
        with self.assertRaises(BarrierNotSatisfied):
            self.authority.bind_training_request(
                foreign_binding_operation,
                racing_day_id=request_command.racing_day_id,
                request_id="foreign-request",
                request_operation_id=foreign_request_operation,
                at=handlers.at,
            )

    def test_training_binding_rejects_stale_time_in_api_and_database_trigger(
        self,
    ) -> None:
        service, handlers, commands = self.trusted_scheduler_fixture(
            852000, recovered_after_discovery=True
        )
        for ordinal, command in enumerate(commands[:8], 1):
            service.advance(
                operation(852070 + ordinal),
                racing_day_id=command.racing_day_id,
                phase=command.phase,
                now=handlers.at,
                command=command,
            )
        run_id = operation(852200)
        request_operation = operation(852201)
        authorization_operation = operation(852204)
        EvaluationAuthority(self.store, self.artifacts).begin_run(
            run_id,
            run_kind="forecast_service",
            started_at=handlers.at,
        )
        CanonicalForecastService(
            ChampionLoader(self.store, self.artifacts, deserializer=lambda content: content),
            self.artifacts,
        ).emit_training_request(
            request_operation,
            request_id="stale-request",
            reason="stale binding adversarial fixture",
            requested_at=handlers.at,
            service_run_id=run_id,
        )
        self.authority.authorize_training_request(
            authorization_operation,
            racing_day_id=commands[0].racing_day_id,
            request_id="stale-request",
            request_operation_id=request_operation,
            at=handlers.at,
        )
        stale_at = handlers.at - timedelta(microseconds=1)
        with self.assertRaises(BarrierNotSatisfied):
            self.authority.bind_training_request(
                operation(852202),
                racing_day_id=commands[0].racing_day_id,
                request_id="stale-request",
                request_operation_id=request_operation,
                at=stale_at,
            )
        with self.assertRaisesRegex(sqlite3.IntegrityError, "exact Racing Day authority"):
            with self.store._operation(
                operation(852203),
                "phase7_bind_training_request",
                {"fully_parented_stale_forgery": True},
            ) as (db, _):
                db.execute(
                    "INSERT INTO phase7_day_training_requests VALUES(?,?,?,?,?)",
                    (
                        commands[0].racing_day_id,
                        "stale-request",
                        str(request_operation),
                        stale_at.isoformat(),
                        str(operation(852203)),
                    ),
                )

    def test_training_binding_trigger_rejects_foreign_fully_parented_day(self) -> None:
        first_service, first_handlers, first_commands = self.trusted_scheduler_fixture(
            853000,
            local_date=date(2026, 7, 20),
            recovered_after_discovery=True,
        )
        for ordinal, command in enumerate(first_commands[:8], 1):
            first_service.advance(
                operation(853070 + ordinal),
                racing_day_id=command.racing_day_id,
                phase=command.phase,
                now=first_handlers.at,
                command=command,
            )
        second_service, second_handlers, second_commands = self.trusted_scheduler_fixture(
            854000,
            local_date=date(2026, 7, 21),
            predecessor=first_commands[0].racing_day_id,
            recovered_after_discovery=True,
        )
        for ordinal, command in enumerate(second_commands[:8], 1):
            second_service.advance(
                operation(854070 + ordinal),
                racing_day_id=command.racing_day_id,
                phase=command.phase,
                now=second_handlers.at,
                command=command,
            )
        run_id = operation(853200)
        request_operation = operation(853201)
        authorization_operation = operation(853202)
        EvaluationAuthority(self.store, self.artifacts).begin_run(
            run_id,
            run_kind="forecast_service",
            started_at=second_handlers.at,
        )
        self.authority.authorize_training_request(
            authorization_operation,
            racing_day_id=first_commands[0].racing_day_id,
            request_id="first-day-request",
            request_operation_id=request_operation,
            at=second_handlers.at,
        )
        CanonicalForecastService(
            ChampionLoader(self.store, self.artifacts, deserializer=lambda content: content),
            self.artifacts,
        ).emit_training_request(
            request_operation,
            request_id="first-day-request",
            reason="foreign-day trigger adversarial fixture",
            requested_at=second_handlers.at,
            service_run_id=run_id,
        )
        with self.assertRaisesRegex(sqlite3.IntegrityError, "exact Racing Day authority"):
            with self.store._operation(
                operation(853203),
                "phase7_bind_training_request",
                {"fully_parented_foreign_day": True},
            ) as (db, _):
                db.execute(
                    "INSERT INTO phase7_day_training_requests VALUES(?,?,?,?,?)",
                    (
                        second_commands[0].racing_day_id,
                        "first-day-request",
                        str(request_operation),
                        second_handlers.at.isoformat(),
                        str(operation(853203)),
                    ),
                )

    def test_command_plan_trigger_rejects_name_lease_and_operation_forgeries(
        self,
    ) -> None:
        day = RacingDay(RacingDayId("day_" + "2" * 32), NOW.date(), "UTC", NOW)
        self.store.create_racing_day(operation(855000), day)
        generation = self.authority.acquire_lease(
            operation(855001),
            owner="plan-trigger",
            token="plan-trigger",
            now=NOW,
            ttl=timedelta(hours=1),
        )
        cases = (
            (
                "wrong-name",
                generation,
                "collect_results",
                "phase7_plan_racing_day",
                NOW,
            ),
            (
                "wrong-lease",
                generation + 1,
                "discover_programme",
                "phase7_plan_racing_day",
                NOW,
            ),
            (
                "wrong-operation",
                generation,
                "discover_programme",
                "unrelated_operation",
                NOW,
            ),
            (
                "closed-migration-authority",
                generation,
                "discover_programme",
                "phase7_migrate_v27_day_command_plan",
                NOW,
            ),
            (
                "expired-lease",
                generation,
                "discover_programme",
                "phase7_plan_racing_day",
                NOW + timedelta(hours=2),
            ),
        )
        for offset, (label, lease, phase, kind, planned_at) in enumerate(cases, 1):
            operation_id = operation(855010 + offset)
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    sqlite3.IntegrityError, "day command plan lacks scheduler authority"
                ):
                    with self.store._operation(operation_id, kind, {"case": label}) as (
                        db,
                        _,
                    ):
                        db.execute(
                            "INSERT INTO phase7_day_command_plan VALUES(?,?,?,?,?,?,?)",
                            (
                                str(day.id),
                                1,
                                phase,
                                str(operation(855100 + offset)),
                                lease,
                                planned_at.isoformat(),
                                str(operation_id),
                            ),
                        )

    def test_command_plan_trigger_rejects_known_generation_without_token_authority(
        self,
    ) -> None:
        day = RacingDay(RacingDayId("day_" + "3" * 32), NOW.date(), "UTC", NOW)
        self.store.create_racing_day(operation(855200), day)
        generation = self.authority.acquire_lease(
            operation(855201),
            owner="plan-trigger",
            token="unavailable-to-sql-attacker",
            now=NOW,
            ttl=timedelta(hours=1),
        )
        plan_operation = operation(855202)
        with self.assertRaisesRegex(
            sqlite3.IntegrityError, "day command plan lacks scheduler authority"
        ):
            with self.store._operation(
                plan_operation, "phase7_plan_racing_day", {"generation": generation}
            ) as (db, _):
                db.execute(
                    "INSERT INTO phase7_day_command_plan VALUES(?,?,?,?,?,?,?)",
                    (
                        str(day.id),
                        1,
                        "discover_programme",
                        str(operation(855203)),
                        generation,
                        NOW.isoformat(),
                        str(plan_operation),
                    ),
                )
        with self.store._connect() as db:
            self.assertEqual(
                db.execute(
                    "SELECT count(*) FROM phase7_day_plan_authorities " "WHERE racing_day_id=?",
                    (str(day.id),),
                ).fetchone()[0],
                0,
            )

    def test_one_vs_two_complete_days_prospective_cutover_and_exact_rollback(
        self,
    ) -> None:
        self.register_config(49)
        self.authority.register_release(operation(50), self.manifest("legacy"), NOW)
        self.authority.register_release(operation(51), self.manifest("candidate"), NOW)
        self.authority.register_release(operation(59), self.manifest("candidate-2"), NOW)
        self.authority.initialize_legacy_authority(
            operation(52),
            release_id="legacy",
            actor="owner",
            reason="record intact authority",
            at=NOW,
        )
        OperationalAuthority(
            self.store, self.artifacts, clock=lambda: NOW - timedelta(days=2)
        ).authorize_observation(
            operation(57),
            candidate_release_id="candidate",
            actor="owner",
            reason="candidate observation",
            at=NOW,
        )

        class ObservationAdapter:
            @staticmethod
            def registrations():
                def noop(_command, _at):
                    return None

                return tuple(
                    PhaseHandlerRegistration(payload_type, noop)
                    for payload_type in (
                        DiscoverProgramme,
                        CollectCardsAndForm,
                        CollectAdaptiveOdds,
                        CloseAndSeal,
                        CommitDeferredPrediction,
                        CollectResults,
                        JoinTrainingExamples,
                        ReconcileRacingDay,
                        RequestTraining,
                    )
                )

            def close(self):
                pass

        observation_composition = ServiceComposition(
            self.configuration(),
            self.store,
            self.artifacts,
            ObservationAdapter(),
            owner="observation",
            token="observation",
            lease_ttl=timedelta(minutes=5),
            release_id="candidate",
            mode="observation",
        )
        active_composition = ServiceComposition(
            self.configuration(),
            self.store,
            self.artifacts,
            ObservationAdapter(),
            owner="active",
            token="active",
            lease_ttl=timedelta(minutes=5),
            release_id="candidate",
            mode="active",
        )
        observation_composition._revalidate_release_mode()
        with self.assertRaisesRegex(OperationalRejected, "no longer authorized"):
            active_composition._revalidate_release_mode()
        with self.store._connect() as db:
            self.assertEqual(
                tuple(
                    db.execute("SELECT release_id,authority FROM phase7_release_pointer").fetchone()
                ),
                ("legacy", "legacy"),
            )
        with self.assertRaises(sqlite3.IntegrityError):
            self.authority.authorize_observation(
                operation(58),
                candidate_release_id="candidate-2",
                actor="competitor",
                reason="competing candidate",
                at=NOW,
            )
        first = self.seed_complete_day(101, date(2026, 7, 20), release="candidate")
        with self.assertRaises(OperationalRejected, msg="one day cannot activate cutover"):
            self.authority.cutover_eligibility(
                operation(53),
                candidate_release_id="candidate",
                at=NOW - timedelta(days=30),
            )
        second = self.seed_complete_day(
            102, date(2026, 7, 21), release="candidate", predecessor=first
        )
        backdated_authority = OperationalAuthority(
            self.store,
            self.artifacts,
            clock=lambda: NOW - timedelta(days=30),
        )
        with self.assertRaisesRegex(OperationalRejected, "predates"):
            backdated_authority.cutover_eligibility(
                operation(530),
                candidate_release_id="candidate",
                at=NOW + timedelta(days=300),
            )
        with self.store._connect() as db:
            self.assertIsNone(
                db.execute(
                    "SELECT 1 FROM operations WHERE operation_id=?",
                    (str(operation(530)),),
                ).fetchone()
            )
        self.assertTrue(
            self.authority.cutover_eligibility(
                operation(53),
                candidate_release_id="candidate",
                at=NOW - timedelta(days=30),
            )
        )
        boundary = "day_" + "f" * 32
        with self.store._operation(operation(54), "synthetic_boundary", {}) as (db, _):
            db.execute(
                "INSERT INTO racing_days VALUES(?,?,?,?,NULL)",
                (
                    boundary,
                    "2026-07-23",
                    "Australia/Melbourne",
                    (NOW + timedelta(days=1)).isoformat(),
                ),
            )
            db.execute(
                "INSERT INTO phase6_racing_day_schedule VALUES(?,?,?,?,?)",
                (
                    boundary,
                    second,
                    "sha256:" + "f" * 64,
                    NOW.isoformat(),
                    str(operation(54)),
                ),
            )
        with self.assertRaisesRegex(OperationalRejected, "prospective boundary"):
            OperationalAuthority(
                self.store,
                self.artifacts,
                clock=lambda: NOW - timedelta(microseconds=1),
            ).activate(
                operation(550),
                release_id="candidate",
                boundary_day_id=boundary,
                actor="owner",
                reason="backdated authority",
                at=NOW + timedelta(days=300),
            )
        with self.store._connect() as db:
            self.assertIsNone(
                db.execute(
                    "SELECT 1 FROM operations WHERE operation_id=?",
                    (str(operation(550)),),
                ).fetchone()
            )
        self.assertTrue(
            self.authority.activate(
                operation(55),
                release_id="candidate",
                boundary_day_id=boundary,
                actor="owner",
                reason="two-day observation accepted",
                at=NOW - timedelta(days=30),
            )
        )
        with self.assertRaisesRegex(OperationalRejected, "no longer authorized"):
            observation_composition._revalidate_release_mode()
        active_composition._revalidate_release_mode()
        with self.store._connect() as db:
            self.assertEqual(
                db.execute(
                    "SELECT eligible_at FROM phase7_cutover_eligibility "
                    "WHERE candidate_release_id='candidate'"
                ).fetchone()[0],
                iso_timestamp(NOW),
            )
            self.assertEqual(
                db.execute(
                    "SELECT changed_at FROM phase7_release_history " "WHERE operation_id=?",
                    (str(operation(55)),),
                ).fetchone()[0],
                iso_timestamp(NOW),
            )
            self.assertEqual(
                tuple(
                    db.execute(
                        "SELECT release_id,authority,legacy_preserved FROM phase7_release_pointer"
                    ).fetchone()
                ),
                ("candidate", "race_collection_service", 1),
            )
            self.assertEqual(
                db.execute(
                    "SELECT command FROM phase7_admin_audit WHERE operation_id=?",
                    (str(operation(55)),),
                ).fetchone()[0],
                "activate",
            )
        self.assertFalse(
            self.authority.activate(
                operation(55),
                release_id="candidate",
                boundary_day_id=boundary,
                actor="owner",
                reason="two-day observation accepted",
                at=NOW + timedelta(days=300),
            )
        )
        self.assertTrue(
            self.authority.rollback(
                operation(56), actor="owner", reason="synthetic rollback drill", at=NOW
            )
        )
        with self.assertRaisesRegex(OperationalRejected, "no longer authorized"):
            observation_composition._revalidate_release_mode()
        with self.assertRaisesRegex(OperationalRejected, "no longer authorized"):
            active_composition._revalidate_release_mode()
        with self.store._connect() as db:
            self.assertEqual(
                tuple(
                    db.execute(
                        "SELECT release_id,authority,legacy_preserved FROM phase7_release_pointer"
                    ).fetchone()
                ),
                ("legacy", "legacy", 1),
            )

    def test_compose_requires_exact_active_or_observation_release_authority(
        self,
    ) -> None:
        import race_collection.operational as operational_module

        original_safe_path = operational_module._safe_operational_path
        operational_module._safe_operational_path = lambda value: Path(value)
        self.addCleanup(
            setattr,
            operational_module,
            "_safe_operational_path",
            original_safe_path,
        )
        root = Path(self.temporary.name) / "observation-compose"
        store = SQLiteOperationsStore(root / "operations.sqlite3")
        store.migrate()
        artifacts = LocalArtifactStore(root / "artifacts")
        authority = OperationalAuthority(store, artifacts, clock=lambda: NOW)
        EvaluationAuthority(store, artifacts).register_policy(
            operation(5350), PromotionPolicy(), NOW - timedelta(days=1)
        )
        configuration = ReleaseConfiguration(
            "phase7-config-v1",
            str(root / "service"),
            str(root / "artifacts"),
            str(root / "operations.sqlite3"),
            ("official",),
            "adaptive-odds-v1",
            "phase6-promotion-v1",
            ("runner-win-probability-v1",),
            "race_collection.runtime_adapters:unavailable",
            ArtifactChecksum("sha256:" + "9" * 64),
        )
        checksum = authority.register_configuration(operation(5351), configuration, NOW)

        def manifest(release_id):
            return ReleaseManifest(
                "phase7-release-v1",
                release_id,
                "6cd5dacfe83719cbbc376a265829e84593eafb68",
                checksum,
                28,
                "canonical-artifacts-v1",
                "phase6-promotion-v1",
                ("runner-win-probability-v1",),
                configuration.service_root,
            )

        authority.register_release(operation(5352), manifest("legacy"), NOW)
        authority.register_release(operation(5353), manifest("candidate"), NOW)
        authority.initialize_legacy_authority(
            operation(5354),
            release_id="legacy",
            actor="owner",
            reason="baseline",
            at=NOW,
        )
        config_path = root / "release.json"
        config_path.write_bytes(
            json.dumps(configuration.document(), sort_keys=True, separators=(",", ":")).encode()
        )
        with self.assertRaisesRegex(ServiceUnavailable, "lacks active or observation"):
            compose(
                config_path,
                owner="compose",
                token="compose",
                lease_ttl=timedelta(minutes=5),
            )
        authority.authorize_observation(
            operation(5355),
            candidate_release_id="candidate",
            actor="owner",
            reason="observe candidate",
            at=NOW - timedelta(days=30),
        )

        class Adapter:
            @staticmethod
            def registrations():
                def noop(_command, _at):
                    return None

                return tuple(
                    PhaseHandlerRegistration(payload_type, noop) for payload_type in COMMAND_PHASES
                )

            def close(self):
                pass

        import race_collection.runtime_adapters as runtime_adapters

        original = runtime_adapters.unavailable
        runtime_adapters.unavailable = lambda *_args: Adapter()
        try:
            composition = compose(
                config_path,
                owner="compose",
                token="compose",
                lease_ttl=timedelta(minutes=5),
            )
        finally:
            runtime_adapters.unavailable = original
        self.assertEqual((composition.release_id, composition.mode), ("candidate", "observation"))
        with store._connect() as db:
            self.assertEqual(
                tuple(
                    db.execute("SELECT release_id,authority FROM phase7_release_pointer").fetchone()
                ),
                ("legacy", "legacy"),
            )

    def test_probation_gap_duplicate_and_exact_thirteen_fourteen_boundary(self) -> None:
        probation_at = datetime(2026, 8, 31, 23, tzinfo=timezone.utc)
        predecessor = self.establish_cutover(70, 180, date(2026, 8, 1))
        days = []
        for index in range(201, 215):
            day_id = self.seed_complete_day(
                index,
                date(2026, 8, 2) + timedelta(days=index - 201),
                release="candidate",
                predecessor=predecessor,
            )
            predecessor = day_id
            days.append(day_id)
        with self.store._connect() as db:
            self.assertIsNone(
                db.execute("SELECT 1 FROM phase7_probation_control WHERE singleton=1").fetchone()
            )
        self.assertEqual(
            self.authority.record_probation_day(
                operation(20000), racing_day_id=days[0], at=probation_at
            ),
            1,
        )
        with self.store._connect() as db:
            self.assertEqual(
                tuple(
                    db.execute(
                        "SELECT state,generation,operation_id "
                        "FROM phase7_probation_control WHERE singleton=1"
                    ).fetchone()
                ),
                ("running", 1, str(operation(20000))),
            )
        for offset, day_id in enumerate(days[1:13], 1):
            self.assertEqual(
                self.authority.record_probation_day(
                    operation(20000 + offset), racing_day_id=day_id, at=probation_at
                ),
                offset + 1,
            )
        with self.assertRaises(OperationalRejected, msg="13 is not 14"):
            self.authority.seal_probation(
                operation(20100), probation_id="probation", at=probation_at
            )
        with self.assertRaises(OperationalRejected, msg="duplicate day is not new evidence"):
            self.authority.record_probation_day(
                operation(20101), racing_day_id=days[12], at=probation_at
            )
        self.assertEqual(
            self.authority.record_probation_day(
                operation(20102), racing_day_id=days[13], at=probation_at
            ),
            14,
        )
        checksum = self.authority.seal_probation(
            operation(20103), probation_id="probation", at=probation_at
        )
        self.assertEqual(self.artifacts.verify(checksum).checksum, checksum)
        eligibility = self.authority.record_legacy_retirement_eligibility(
            operation(20109),
            eligibility_id="legacy-retirement-eligibility",
            probation_id="probation",
            actor="owner",
            reason="sealed fourteen-day generation",
            at=probation_at,
        )
        self.assertEqual(eligibility["candidate_release_id"], "candidate")
        self.assertEqual(eligibility["legacy_release_id"], "legacy")
        self.assertEqual(
            self.authority.record_legacy_retirement_eligibility(
                operation(20109),
                eligibility_id="legacy-retirement-eligibility",
                probation_id="probation",
                actor="owner",
                reason="sealed fourteen-day generation",
                at=probation_at,
            ),
            eligibility,
        )
        with self.store._connect() as db:
            self.assertEqual(
                db.execute("SELECT count(*) FROM phase7_legacy_retirement_eligibility").fetchone()[
                    0
                ],
                1,
            )
            with self.assertRaisesRegex(sqlite3.IntegrityError, "sealed probation authority"):
                db.execute(
                    "INSERT INTO phase7_legacy_retirement_eligibility VALUES"
                    "(?,?,?,?,?,?,?,?,?,?)",
                    (
                        "forged",
                        "probation",
                        1,
                        "candidate",
                        "legacy",
                        str(operation(70 + 6)),
                        str(operation(20103)),
                        str(checksum),
                        probation_at.isoformat(),
                        str(operation(29999)),
                    ),
                )
        with self.assertRaisesRegex(OperationalRejected, "already sealed"):
            self.authority.seal_probation(
                operation(20104), probation_id="forged-reuse", at=probation_at
            )
        fifteenth = self.seed_complete_day(
            215,
            date(2026, 8, 16),
            release="candidate",
            predecessor=days[-1],
        )
        with self.assertRaisesRegex(OperationalRejected, "fifteenth"):
            self.authority.record_probation_day(
                operation(20105), racing_day_id=fifteenth, at=probation_at
            )
        with self.store._connect() as read:
            programme = read.execute(
                "SELECT programme_checksum FROM phase6_racing_day_schedule "
                "WHERE racing_day_id=?",
                (fifteenth,),
            ).fetchone()[0]

        def direct_accept(number: int) -> None:
            with self.store._operation(
                operation(number), "phase7_accept_probation_day", {"direct": number}
            ) as (db, _):
                db.execute(
                    "INSERT INTO phase7_probation_acceptances VALUES(?,?,?,?,?,?)",
                    (
                        1,
                        fifteenth,
                        "2026-08-16",
                        programme,
                        NOW.isoformat(),
                        str(operation(number)),
                    ),
                )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "schedule authority"):
            direct_accept(20106)  # complete is not an acceptance state
        with self.store._connect() as db:
            db.execute("UPDATE phase7_probation_control SET state='reset' WHERE singleton=1")
        with self.assertRaisesRegex(sqlite3.IntegrityError, "schedule authority"):
            direct_accept(20107)  # reset is not an acceptance state
        with self.store._connect() as db:
            db.execute("UPDATE phase7_probation_control SET state='running' WHERE singleton=1")
        with self.assertRaisesRegex(sqlite3.IntegrityError, "schedule authority"):
            direct_accept(20108)  # the database itself enforces the fourteen-day cap

        other = Phase7OperationalTests(methodName="runTest")
        other.setUp()
        try:
            boundary = other.establish_cutover(80, 280, date(2026, 8, 1))
            first = other.seed_complete_day(
                301, date(2026, 8, 2), release="candidate", predecessor=boundary
            )
            omitted = "day_" + "9" * 32
            with other.store._operation(operation(20990), "omitted_scheduled_day", {}) as (db, _):
                db.execute(
                    "INSERT INTO racing_days VALUES(?,?,?,?,?)",
                    (
                        omitted,
                        "2026-08-03",
                        "Australia/Melbourne",
                        NOW.isoformat(),
                        NOW.isoformat(),
                    ),
                )
                db.execute(
                    "INSERT INTO phase6_racing_day_schedule VALUES(?,?,?,?,?)",
                    (
                        omitted,
                        first,
                        "sha256:" + "9" * 64,
                        NOW.isoformat(),
                        str(operation(20990)),
                    ),
                )
            gap = other.seed_complete_day(
                302, date(2026, 8, 4), release="candidate", predecessor=omitted
            )
            other.authority.record_probation_day(
                operation(21000), racing_day_id=first, at=probation_at
            )
            with self.assertRaises(OperationalRejected, msg="a gap breaks consecutiveness"):
                other.authority.record_probation_day(
                    operation(21001), racing_day_id=gap, at=probation_at
                )
        finally:
            other.tearDown()

    def test_probation_database_triggers_reject_fully_parented_chain_forgeries(
        self,
    ) -> None:
        boundary = self.establish_cutover(870000, 600, date(2026, 8, 1))
        predecessor = boundary
        days = []
        for offset in range(14):
            day_id = self.seed_complete_day(
                610 + offset,
                date(2026, 8, 2) + timedelta(days=offset),
                release="candidate",
                predecessor=predecessor,
            )
            predecessor = day_id
            days.append(day_id)
            self.authority.record_probation_day(
                operation(871000 + offset),
                racing_day_id=day_id,
                at=datetime(2026, 8, 2 + offset, 23, tzinfo=timezone.utc),
            )
        with self.store._connect() as db:
            with self.assertRaisesRegex(sqlite3.IntegrityError, "schedule is append-only"):
                db.execute(
                    "DELETE FROM phase6_racing_day_schedule WHERE racing_day_id=?",
                    (days[-1],),
                )
            with self.assertRaisesRegex(sqlite3.IntegrityError, "schedule is append-only"):
                db.execute(
                    "UPDATE phase6_racing_day_schedule SET predecessor_racing_day_id=? "
                    "WHERE racing_day_id=?",
                    (days[-3], days[-1]),
                )

        def isolated_case(case: str, number: int) -> None:
            fixture = Phase7OperationalTests(methodName="runTest")
            fixture.setUp()
            try:
                boundary = fixture.establish_cutover(
                    number,
                    number + 10,
                    date(2026, 8, 1),
                )
                first = fixture.seed_complete_day(
                    number + 20,
                    date(2026, 8, 2),
                    release="candidate",
                    predecessor=boundary,
                )
                fixture.authority.record_probation_day(
                    operation(number + 30),
                    racing_day_id=first,
                    at=datetime(2026, 8, 2, 23, tzinfo=timezone.utc),
                )
                operation_kind = "phase7_accept_probation_day"
                if case == "missing_schedule":
                    # The accepted first probation day uses seed_complete_day,
                    # whose scheduler base is derived from its day index. Start
                    # strictly after that microsecond lease has expired. Reserve
                    # a separate 10,000-operation block beyond seed_complete_day's
                    # base+handler range, so lease and operation identities cannot
                    # overlap any established fixture operation.
                    missing_schedule_base = 100000 + (number + 20) * 200 + 10_000
                    service, handlers, commands = fixture.trusted_scheduler_fixture(
                        missing_schedule_base,
                        release_id="candidate",
                        recovered_after_discovery=True,
                    )
                    for ordinal, command in enumerate(commands[1:], 2):
                        service.advance(
                            operation(number + 80 + ordinal),
                            racing_day_id=command.racing_day_id,
                            phase=command.phase,
                            now=handlers.at,
                            command=command,
                        )
                    target = commands[0].racing_day_id
                elif case == "non_immediate":
                    omitted = f"day_{number + 40:032x}"
                    with fixture.store._operation(
                        operation(number + 41),
                        "synthetic_omitted_schedule",
                        {},
                    ) as (db, _):
                        db.execute(
                            "INSERT INTO racing_days VALUES(?,?,?,?,?)",
                            (
                                omitted,
                                "2026-06-03",
                                "Australia/Melbourne",
                                NOW.isoformat(),
                                NOW.isoformat(),
                            ),
                        )
                        db.execute(
                            "INSERT INTO phase6_racing_day_schedule VALUES(?,?,?,?,?)",
                            (
                                omitted,
                                first,
                                "sha256:" + "e" * 64,
                                NOW.isoformat(),
                                str(operation(number + 41)),
                            ),
                        )
                    target = fixture.seed_complete_day(
                        number + 42,
                        date(2026, 6, 4),
                        release="candidate",
                        predecessor=omitted,
                    )
                else:
                    target = fixture.seed_complete_day(
                        number + 42,
                        date(2026, 6, 3),
                        release="candidate",
                        predecessor=first,
                    )
                    if case == "wrong_operation_kind":
                        operation_kind = "unrelated_operation"
                with fixture.store._connect() as db:
                    target_row = db.execute(
                        "SELECT d.local_date,e.programme_checksum FROM racing_days d "
                        "JOIN races r USING(racing_day_id) "
                        "JOIN expected_races e USING(race_id) WHERE d.racing_day_id=? "
                        "LIMIT 1",
                        (target,),
                    ).fetchone()
                    generation = db.execute(
                        "SELECT generation FROM phase7_probation_control WHERE singleton=1"
                    ).fetchone()[0]
                programme = (
                    "sha256:" + "f" * 64
                    if case == "programme_mismatch"
                    else target_row["programme_checksum"]
                )
                with self.assertRaisesRegex(
                    sqlite3.IntegrityError,
                    "probation .* authority",
                ):
                    with fixture.store._operation(
                        operation(number + 90),
                        operation_kind,
                        {"direct_sql_case": case},
                    ) as (db, _):
                        db.execute(
                            "INSERT INTO phase7_probation_acceptances VALUES(?,?,?,?,?,?)",
                            (
                                generation,
                                target,
                                target_row["local_date"],
                                programme,
                                NOW.isoformat(),
                                str(operation(number + 90)),
                            ),
                        )
            finally:
                fixture.tearDown()

        for offset, case in enumerate(
            (
                "missing_schedule",
                "non_immediate",
                "programme_mismatch",
                "wrong_operation_kind",
            )
        ):
            with self.subTest(case=case):
                isolated_case(case, 880000 + offset * 1000)

    def test_real_phase6_promotion_authority_requires_exact_phase7_fourteen_day_seal(
        self,
    ):
        test_root = str(Path(__file__).parent)
        sys.path.insert(0, test_root)
        try:
            from test_phase6_evaluation_promotion import (
                _build_authentic_promotion_template,
                registered_report,
            )
        finally:
            sys.path.remove(test_root)
        root = Path(self.temporary.name) / "promotion-integration"
        store, artifacts = _build_authentic_promotion_template(
            root, count=500, artifact_backed=True
        )
        report = registered_report(store, artifacts)
        fixture = Phase7OperationalTests(methodName="runTest")
        fixture.store = store
        fixture.artifacts = artifacts
        fixture.authority = OperationalAuthority(store, artifacts)
        boundary = fixture.establish_cutover(50000, 400, date(2026, 8, 1))
        predecessor = boundary
        probation_days = []
        for offset in range(14):
            racing_day_id = fixture.seed_complete_day(
                410 + offset,
                date(2026, 8, 2) + timedelta(days=offset),
                release="candidate",
                predecessor=predecessor,
            )
            predecessor = racing_day_id
            probation_days.append(racing_day_id)
        for offset, racing_day_id in enumerate(probation_days[:13]):
            fixture.authority.record_probation_day(
                operation(700000 + offset),
                racing_day_id=racing_day_id,
                at=datetime(2026, 8, 2 + offset, 23, tzinfo=timezone.utc),
            )
        evaluation = EvaluationAuthority(store, artifacts)
        promotion_run = operation(700100)
        evaluation.begin_run(
            promotion_run,
            run_kind="promotion",
            started_at=datetime(2026, 8, 15, 22, tzinfo=timezone.utc),
        )
        promote = PromotionAuthority(store, artifacts)
        arguments = {
            "evidence_id": "evidence",
            "report": report,
            "challenger_bundle_id": "challenger",
            "assignment_id": "phase7-assignment",
            "promotion_record_id": "phase7-promotion",
            "approved_at": datetime(2026, 8, 15, 23, 30, tzinfo=timezone.utc),
            "effective_racing_day": "2026-08-16",
            "approver": "automatic-policy",
            "reason": "synthetic end-to-end contract proof",
            "probation_id": "phase7-probation",
            "promotion_run_id": promotion_run,
            "approval_racing_day_id": probation_days[-1],
        }
        effective_day_id = "day_" + "8" * 32
        with store._operation(operation(700104), "synthetic_effective_day", {}) as (
            db,
            _,
        ):
            db.execute(
                "INSERT INTO racing_days VALUES(?,?,?,?,NULL)",
                (
                    effective_day_id,
                    "2026-08-16",
                    "Australia/Melbourne",
                    "2026-08-16T00:00:00+00:00",
                ),
            )
            db.execute(
                "INSERT INTO phase6_racing_day_schedule VALUES(?,?,?,?,?)",
                (
                    effective_day_id,
                    probation_days[-1],
                    "sha256:" + "8" * 64,
                    "2026-08-15T00:00:00+00:00",
                    str(operation(700104)),
                ),
            )
        with self.assertRaisesRegex(PromotionRejected, "probation"):
            promote.promote(operation(700101), **arguments)
        fixture.authority.record_probation_day(
            operation(700102),
            racing_day_id=probation_days[-1],
            at=datetime(2026, 8, 15, 23, tzinfo=timezone.utc),
        )
        fixture.authority.seal_probation(
            operation(700103),
            probation_id="phase7-probation",
            at=datetime(2026, 8, 15, 23, 1, tzinfo=timezone.utc),
        )
        with self.assertRaisesRegex(OperationalRejected, "immutable"):
            fixture.authority.reset_probation(
                operation(700105),
                actor="operator",
                reason="attempt to invalidate sealed promotion input",
                at=datetime(2026, 8, 15, 23, 2, tzinfo=timezone.utc),
            )
        with store._connect() as db:
            seal_checksum = ArtifactChecksum(
                db.execute(
                    "SELECT state_checksum FROM phase7_probation_seals "
                    "WHERE probation_id='phase7-probation'"
                ).fetchone()[0]
            )
            self.assertIn(str(seal_checksum), artifact_inventory(db))
            determinism_artifacts = {
                value
                for row in db.execute(
                    "SELECT input_checksum,output_checksum " "FROM phase7_determinism_executions"
                )
                for value in row
            }
        replica = LocalArtifactStore(root / "recorded-inventory-replica")
        snapshot = root / "recorded-inventory-snapshot.sqlite3"
        recovery = RecoveryAuthority(store, artifacts)
        recovery.backup(
            operation(700108),
            backup_id="probation-inventory-backup",
            racing_day_id=probation_days[-1],
            snapshot_path=snapshot,
            replica=replica,
            at=datetime(2026, 8, 15, 23, 2, tzinfo=timezone.utc),
        )
        with store._connect() as db:
            backup = db.execute(
                "SELECT artifact_inventory_checksum FROM phase7_backups "
                "WHERE backup_id='probation-inventory-backup'"
            ).fetchone()
        recorded_inventory = set(json.loads(replica.read(ArtifactChecksum(backup[0]))))
        self.assertIn(str(seal_checksum), recorded_inventory)
        self.assertTrue(determinism_artifacts)
        self.assertTrue(determinism_artifacts <= recorded_inventory)
        for checksum in {str(seal_checksum), *determinism_artifacts}:
            self.assertEqual(
                replica.verify(ArtifactChecksum(checksum)).checksum,
                ArtifactChecksum(checksum),
            )
        self.assertTrue(
            recovery.restore_drill(
                operation(700109),
                drill_id="probation-inventory-drill",
                backup_id="probation-inventory-backup",
                snapshot_path=snapshot,
                replica=replica,
                at=datetime(2026, 8, 15, 23, 3, tzinfo=timezone.utc),
            )
        )
        recovery = RecoveryAuthority(store, artifacts)
        recovery.backup(
            operation(700106),
            backup_id="probation-backup",
            racing_day_id=probation_days[-1],
            snapshot_path=root / "probation-snapshot.sqlite3",
            replica=LocalArtifactStore(root / "probation-replica"),
            at=datetime(2026, 8, 16, 0, tzinfo=timezone.utc),
        )
        seal_content = artifacts.read(seal_checksum)
        artifacts.path_for(seal_checksum).write_bytes(b"corrupt-seal")
        with self.assertRaises(ArtifactStoreError):
            recovery.backup(
                operation(700107),
                backup_id="corrupt-probation-backup",
                racing_day_id=probation_days[-1],
                snapshot_path=root / "corrupt-probation-snapshot.sqlite3",
                replica=LocalArtifactStore(root / "corrupt-probation-replica"),
                at=datetime(2026, 8, 16, 0, 1, tzinfo=timezone.utc),
            )
        with store._connect() as db:
            self.assertIsNone(
                db.execute(
                    "SELECT 1 FROM phase7_backups " "WHERE backup_id='corrupt-probation-backup'"
                ).fetchone()
            )
        artifacts.path_for(seal_checksum).write_bytes(seal_content)
        assignment = promote.promote(operation(700105), **arguments)
        self.assertEqual(assignment.bundle_id, "challenger")


if __name__ == "__main__":
    unittest.main()
