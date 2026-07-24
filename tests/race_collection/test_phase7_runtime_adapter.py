import json
import hashlib
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from race_collection.artifacts import LocalArtifactStore
from race_collection.domain import (
    ArtifactChecksum,
    OperationId,
    RacingDay,
    RacingDayId,
)
from race_collection.evaluation import EvaluationAuthority, PromotionPolicy
from race_collection.forecasting import ForecastingAuthority, LegacyBundle, ModelRelease
from race_collection.model_bundle import (
    BundleComponent,
    CanonicalBundle,
    ModelBundleAuthority,
    ServingAssignment,
)
from race_collection.ordered_finish import ORDERED_FINISH_CONTRACT
from race_collection.operational import (
    DayForecastCohortMember,
    OperationalAuthority,
    OperationalRejected,
    RaceCollectionService,
    ReleaseConfiguration,
    ReleaseManifest,
)
from race_collection.operations import BarrierNotSatisfied, SQLiteOperationsStore
from race_collection.service import compose, main
from race_collection.training import LinearStrengthModel


NOW = datetime.now(timezone.utc).replace(microsecond=0)
LOCAL_DATE = NOW.astimezone(ZoneInfo("Australia/Melbourne")).date()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def operation(number):
    return OperationId(f"op_{number:032x}")


def _component_documents(bundle_id):
    corpus_entries = [
        {
            "training_example_id": "fixture-history",
            "artifact_checksum": "sha256:" + "1" * 64,
            "evidence_checksum": "sha256:" + "2" * 64,
            "result_checksum": "sha256:" + "3" * 64,
            "feature_matrix_checksum": "sha256:" + "4" * 64,
            "racing_date": "2026-07-20",
        }
    ]
    corpus_id = "sha256:" + __import__("hashlib").sha256(canonical(corpus_entries)).hexdigest()
    return {
        "model": LinearStrengthModel(
            (0.5, -0.25) if bundle_id == "runtime-legacy-canonical" else (-0.25, 0.5)
        ).to_bytes(),
        "feature_schema": canonical(
            {
                "bundle_id": bundle_id,
                "contract_version": "sealed-race-features-v1",
                "evidence_schema_version": "race-evidence-v1",
                "normalization_version": "normalizer-v1",
                "fields": [
                    {
                        "name": "speed",
                        "source_field": "runner_features",
                        "semantics": "identity-critical",
                    },
                    {
                        "name": "form",
                        "source_field": "runner_features",
                        "semantics": "forecast-required",
                    },
                ],
            }
        ),
        "missingness_policy": canonical(
            {
                "bundle_id": bundle_id,
                "feature_contract_version": "sealed-race-features-v1",
                "imputation": {},
            }
        ),
        "training_configuration": canonical(
            {
                "model_id": "runtime-linear",
                "feature_contract_version": "sealed-race-features-v1",
                "forecast_contract_version": ORDERED_FINISH_CONTRACT,
                "algorithm": "full-batch-plackett-luce-linear-v1",
                "optimizer": "deterministic-full-batch-gradient-ascent",
                "seed": 7,
                "epochs": 1,
                "learning_rate": 0.01,
            }
        ),
        "dependency_manifest": canonical({"model_id": "runtime-linear", "packages": {}}),
        "training_corpus": canonical(
            {
                "model_id": "runtime-linear",
                "training_example_ids": ["fixture-history"],
                "training_examples": corpus_entries,
                "corpus_id": corpus_id,
            }
        ),
        "calibration": canonical(
            {
                "model_id": "runtime-linear",
                "forecast_contract_version": ORDERED_FINISH_CONTRACT,
                "method": "identity",
                "status": "calibrated",
            }
        ),
        "evaluation": canonical(
            {
                "model_id": "runtime-linear",
                "forecast_contract_version": ORDERED_FINISH_CONTRACT,
                "population": "synthetic",
                "log_loss": 0.2,
            }
        ),
        "runtime_requirements": canonical(
            {
                "model_id": "runtime-linear",
                "python_implementation": __import__("platform").python_implementation(),
                "python_major_minor": ".".join(__import__("platform").python_version_tuple()[:2]),
            }
        ),
    }


def _register_serving_authority(store, artifacts, day_id):
    registered_at = NOW - timedelta(days=4)
    bundle_id = "runtime-legacy-canonical"
    documents = _component_documents(bundle_id)
    components = []
    for kind, content in documents.items():
        artifact = artifacts.put(content, media_type="application/octet-stream")
        components.append(BundleComponent(f"{kind}.json", kind, artifact.checksum, len(content)))
    model = next(item for item in components if item.kind == "model")
    schema = next(item for item in components if item.kind == "feature_schema")
    legacy = LegacyBundle(
        "runtime-phase3-legacy",
        "runtime-linear",
        model.checksum,
        model.byte_size,
        schema.checksum,
        None,
        "raw_registry_model",
        {"source_proven": True},
    )
    phase3 = ForecastingAuthority(store)
    phase3.register_bundle(operation(10), legacy, registered_at)
    provisional = CanonicalBundle(
        bundle_id,
        "runtime-linear",
        "legacy-origin",
        ArtifactChecksum("sha256:" + "0" * 64),
        "sealed-race-features-v1",
        ORDERED_FINISH_CONTRACT,
        tuple(components),
        "2026-07-20",
        legacy.bundle_id,
    )
    manifest = artifacts.put(canonical(provisional.manifest()), media_type="application/json")
    bundle = CanonicalBundle(
        provisional.bundle_id,
        provisional.model_id,
        provisional.origin,
        manifest.checksum,
        provisional.feature_contract_version,
        provisional.forecast_contract_version,
        provisional.components,
        provisional.trained_through,
        provisional.legacy_model_bundle_id,
    )
    bundles = ModelBundleAuthority(store)
    bundles.register(operation(11), bundle, registered_at)
    release = ModelRelease(
        "runtime-phase3-release", legacy.bundle_id, "runtime-policy", {"fixture": True}
    )
    phase3.register_release(operation(12), release, registered_at)
    phase3.pin_day(operation(13), RacingDayId(day_id), release, registered_at)
    assignment = ServingAssignment(
        "runtime-assignment",
        bundle.bundle_id,
        bundle.bundle_checksum,
        registered_at.isoformat(),
        LOCAL_DATE.isoformat(),
        "runtime-promotion-record",
    )
    bundles.register_assignment(operation(14), assignment, registered_at)
    bundles.bootstrap_champion(operation(15), assignment, registered_at)
    bundles.bind_day_assignment(operation(16), day_id, assignment, registered_at)
    challenger_id = "runtime-challenger"
    challenger_documents = _component_documents(challenger_id)
    challenger_components = []
    for kind, content in challenger_documents.items():
        artifact = artifacts.put(content, media_type="application/octet-stream")
        challenger_components.append(
            BundleComponent(f"{kind}.json", kind, artifact.checksum, len(content))
        )
    challenger_provisional = CanonicalBundle(
        challenger_id,
        "runtime-linear",
        "canonical",
        ArtifactChecksum("sha256:" + "0" * 64),
        "sealed-race-features-v1",
        ORDERED_FINISH_CONTRACT,
        tuple(challenger_components),
        "2026-07-20",
    )
    challenger_manifest = artifacts.put(
        canonical(challenger_provisional.manifest()), media_type="application/json"
    )
    challenger = CanonicalBundle(
        challenger_provisional.bundle_id,
        challenger_provisional.model_id,
        challenger_provisional.origin,
        challenger_manifest.checksum,
        challenger_provisional.feature_contract_version,
        challenger_provisional.forecast_contract_version,
        challenger_provisional.components,
        challenger_provisional.trained_through,
    )
    bundles.register(operation(17), challenger, registered_at)
    return bundle, challenger, assignment


def _runtime_fixture(tmp_path):
    store = SQLiteOperationsStore(tmp_path / "operations.sqlite3")
    store.migrate()
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    day_id = "day_" + "7" * 32
    day = RacingDay(
        RacingDayId(day_id),
        LOCAL_DATE,
        "Australia/Melbourne",
        NOW - timedelta(hours=1),
    )
    store.create_racing_day(operation(1), day)
    bundle, challenger, assignment = _register_serving_authority(store, artifacts, day_id)

    jump = datetime.now(timezone.utc) - timedelta(seconds=2)
    discovery = jump - timedelta(seconds=10)
    result_at = datetime.now(timezone.utc) + timedelta(seconds=1)
    card_bytes = canonical({"source": "official-card", "race": "R-1"})
    card = artifacts.put(card_bytes, media_type="application/json")
    form_a = artifacts.put(
        canonical({"source": "official-form", "runner": "dog-a"}),
        media_type="application/json",
    )
    form_b = artifacts.put(
        canonical({"source": "official-form", "runner": "dog-b"}),
        media_type="application/json",
    )
    odds = artifacts.put(canonical({"dog-a": 2.0, "dog-b": 3.0}), media_type="application/json")
    mapping = artifacts.put(canonical({"dog-a": 1, "dog-b": 2}), media_type="application/json")
    result_content = canonical(
        {
            "source": "official-results",
            "official": True,
            "exclusions": [],
            "order": [1, 2],
            "provenance": {
                "source": "official-results",
                "source_record_id": "R-1-result",
                "observed_at": result_at.isoformat(),
            },
            "published_at": result_at.isoformat(),
        }
    )
    result_checksum = ArtifactChecksum("sha256:" + hashlib.sha256(result_content).hexdigest())
    programme = artifacts.put(
        canonical(
            {
                "races": [
                    {
                        "source_race_id": "R-1",
                        "venue": "Ballarat",
                        "race_number": 1,
                        "scheduled_jump": jump.isoformat(),
                    }
                ]
            }
        ),
        media_type="application/json",
    )
    determinism_input = artifacts.put(
        canonical({"programme_checksum": str(programme.checksum), "race": "R-1"}),
        media_type="application/json",
    )
    identities = [
        {
            "operation_id": str(operation(100)),
            "source": "official-card",
            "source_alias": "dog-a",
            "name": "Dog A",
            "registration_authority": "GRV",
            "registration_id": "A-1",
            "decided_at": discovery.isoformat(),
        },
        {
            "operation_id": str(operation(101)),
            "source": "official-card",
            "source_alias": "dog-b",
            "name": "Dog B",
            "registration_authority": "GRV",
            "registration_id": "B-1",
            "decided_at": discovery.isoformat(),
        },
    ]
    runs = [
        {
            "operation_id": str(operation(102 + index)),
            "identity_source_alias": alias,
            "local_racing_date": "2026-07-20",
            "source": "official-form",
            "checksum": str(artifact.checksum),
            "observed_at": discovery.isoformat(),
            "starts": 10,
            "wins": 2 + index,
            "authoritative": True,
        }
        for index, (alias, artifact) in enumerate((("dog-a", form_a), ("dog-b", form_b)))
    ]
    fields = {
        "runner_set": ["dog-a", "dog-b"],
        "runner_identity": {
            "dog-a": "authoritative",
            "dog-b": "authoritative",
        },
        "runner_features": {
            "dog-a": {"speed": 8, "form": 3},
            "dog-b": {"speed": 2, "form": 6},
        },
        "box": {"dog-a": 1, "dog-b": 2},
        "actual_jump": jump.isoformat(),
    }
    observations = [
        {
            "operation_id": str(operation(110 + index)),
            "field": name,
            "authority": "official_jump" if name == "actual_jump" else "official_card",
            "value": value,
            "source": "official-card",
            "checksum": str(card.checksum),
            "observed_at": discovery.isoformat(),
        }
        for index, (name, value) in enumerate(fields.items())
    ]
    cycle = {
        "racing_day_id": day_id,
        "local_date": day.local_date.isoformat(),
        "timezone": day.timezone,
        "opened_at": day.opened_at.isoformat(timespec="microseconds"),
        "at": (NOW - timedelta(minutes=1)).isoformat(),
        "plan_operation_id": str(operation(200)),
        "command_operation_ids": [str(operation(210 + index)) for index in range(9)],
        "advancement_operation_ids": [str(operation(220 + index)) for index in range(9)],
        "programme": {"source": "official", "checksum": str(programme.checksum)},
        "races": [
            {
                "source_race_id": "R-1",
                "identities": identities,
                "run_observations": runs,
                "observations": observations,
                "odds_attempts": [
                    {
                        "operation_id": str(operation(120)),
                        "source": "market",
                        "attempted_at": discovery.isoformat(),
                        "status": "succeeded",
                        "artifact_checksum": str(odds.checksum),
                        "runner_mapping_checksum": str(mapping.checksum),
                        "error": None,
                    }
                ],
                "seal": {
                    "operation_id": str(operation(121)),
                    "buffer_seconds": 120,
                    "schema_version": "race-evidence-v1",
                    "normalization_version": "normalizer-v1",
                    "sealed_at": (jump + timedelta(seconds=1)).isoformat(),
                },
                "prediction": {
                    "begin_operation_id": str(operation(122)),
                    "operation_id": str(operation(123)),
                    "prediction_id": "runtime-prediction",
                },
                "result": {
                    "open_operation_id": str(operation(124)),
                    "operation_id": str(operation(125)),
                    "attempt_id": "runtime-result-attempt",
                    "attempted_at": result_at.isoformat(),
                    "deadline": (jump + timedelta(hours=1)).isoformat(),
                    "max_attempts": 1,
                    "source": "official-results",
                    "source_checksum": str(result_checksum),
                },
                "training_example": {
                    "join_operation_id": str(operation(126)),
                    "phase3_example_id": "runtime-phase3-example",
                    "build_operation_id": str(operation(127)),
                    "canonical_example_id": "runtime-canonical-example",
                    "joined_at": result_at.isoformat(),
                },
            }
        ],
        "champion": {
            "bundle_id": bundle.bundle_id,
            "bundle_checksum": str(bundle.bundle_checksum),
        },
        "forecast_cohort": {
            "assignment_id": assignment.assignment_id,
            "authorization_operation_id": str(operation(310)),
            "members": [
                {
                    "role": "champion",
                    "bundle_id": bundle.bundle_id,
                    "bundle_checksum": str(bundle.bundle_checksum),
                    "service_run_id": str(operation(311)),
                    "forecast_operations": [
                        {
                            "source_race_id": "R-1",
                            "operation_id": str(operation(313)),
                        }
                    ],
                },
                {
                    "role": "challenger",
                    "bundle_id": challenger.bundle_id,
                    "bundle_checksum": str(challenger.bundle_checksum),
                    "service_run_id": str(operation(312)),
                    "forecast_operations": [
                        {
                            "source_race_id": "R-1",
                            "operation_id": str(operation(314)),
                        }
                    ],
                },
            ],
        },
        "determinism_input_checksum": str(determinism_input.checksum),
        "training_request": {
            "request_id": "runtime-training-request",
            "request_operation_id": str(operation(230)),
            "authorization_operation_id": str(operation(231)),
            "binding_operation_id": str(operation(232)),
            "service_run_id": str(operation(233)),
        },
    }
    runtime_document = {
        "schema_version": "phase7-runtime-input-v1",
        "release_id": "runtime-candidate",
        "cycles": [cycle],
    }
    runtime_artifact = artifacts.put(canonical(runtime_document), media_type="application/json")
    configuration = ReleaseConfiguration(
        "phase7-config-v1",
        str(tmp_path / "service"),
        str(artifacts.root),
        str(store.path),
        ("official", "official-card", "official-form", "market", "official-results"),
        "adaptive-odds-v1",
        "phase6-promotion-v1",
        (ORDERED_FINISH_CONTRACT,),
        "race_collection.runtime_adapters:checked_in",
        runtime_artifact.checksum,
    )
    authority = OperationalAuthority(store, artifacts, clock=lambda: NOW)
    EvaluationAuthority(store, artifacts).register_policy(
        operation(300), PromotionPolicy(), NOW - timedelta(days=4)
    )
    configuration_at = NOW - timedelta(days=3)
    config_checksum = authority.register_configuration(
        operation(301), configuration, configuration_at
    )

    def release(release_id):
        return ReleaseManifest(
            "phase7-release-v1",
            release_id,
            "7" * 40,
            config_checksum,
            28,
            "canonical-artifacts-v1",
            "phase6-promotion-v1",
            (ORDERED_FINISH_CONTRACT,),
            configuration.service_root,
        )

    authority.register_release(operation(302), release("runtime-legacy"), configuration_at)
    authority.register_release(operation(303), release("runtime-candidate"), configuration_at)
    authority.initialize_legacy_authority(
        operation(304),
        release_id="runtime-legacy",
        actor="fixture",
        reason="synthetic baseline",
        at=NOW - timedelta(days=2),
    )
    authority.authorize_observation(
        operation(305),
        candidate_release_id="runtime-candidate",
        actor="fixture",
        reason="exercise checked-in adapter",
        at=NOW - timedelta(days=1),
    )
    config_path = tmp_path / "release.json"
    config_path.write_bytes(canonical(configuration.document()))
    return store, artifacts, config_path, cycle, result_content


def test_checked_in_adapter_resumes_real_prefix_through_main_once(tmp_path, monkeypatch):
    store, artifacts, config_path, cycle, result_content = _runtime_fixture(tmp_path)
    trusted_now = [datetime.now(timezone.utc)]
    result_checksum = cycle["races"][0]["result"]["source_checksum"]
    result_reads = []
    original_read = LocalArtifactStore.read

    def audited_read(instance, checksum):
        if str(checksum) == result_checksum:
            with store._connect() as db:
                result_reads.append(
                    db.execute("SELECT count(*) FROM phase6_forecast_service_artifacts").fetchone()[
                        0
                    ]
                )
        return original_read(instance, checksum)

    monkeypatch.setattr(LocalArtifactStore, "read", audited_read)
    first = compose(
        config_path,
        owner="runtime-prefix",
        token="runtime-prefix",
        lease_ttl=timedelta(seconds=30),
    )
    first.clock = lambda: trusted_now[0]
    runtime_cycle = first.adapter.next_cycle(now=first.trusted_timestamp())
    generation = first.maintain_lease(first.trusted_timestamp())
    first.authority.plan_racing_day(
        runtime_cycle.plan_operation_id,
        racing_day_id=runtime_cycle.racing_day_id,
        lease_token=first.token,
        lease_generation=generation,
        commands=runtime_cycle.commands,
        at=first.trusted_timestamp(),
    )
    service = RaceCollectionService(first.authority, token=first.token, generation=generation)
    prediction_floor = None
    for ordinal in range(5):
        command_at = first.trusted_timestamp()
        if ordinal == 4:
            prediction_floor = command_at
        service.advance(
            runtime_cycle.advancement_operation_ids[ordinal],
            racing_day_id=runtime_cycle.racing_day_id,
            phase=runtime_cycle.commands[ordinal].phase,
            now=command_at,
            command=runtime_cycle.commands[ordinal],
        )
    assert prediction_floor is not None
    with store._connect() as db:
        assert (
            db.execute(
                "SELECT count(*) FROM phase7_application_command_receipts "
                "WHERE phase_name='deferred_prediction'"
            ).fetchone()[0]
            == 1
        )
        assert (
            db.execute("SELECT count(*) FROM phase6_forecast_service_artifacts").fetchone()[0] == 2
        )
        computed_at = datetime.fromisoformat(
            db.execute("SELECT computed_at FROM deferred_predictions").fetchone()[0]
        )
        authorized_at = datetime.fromisoformat(
            db.execute("SELECT authorized_at FROM phase7_day_forecast_cohorts").fetchone()[0]
        )
        assert computed_at >= prediction_floor
        assert authorized_at >= prediction_floor
    assert result_reads == []
    first.close()
    assert first.adapter.closed

    publication_at = datetime.fromisoformat(json.loads(result_content)["published_at"])
    with store._connect() as db:
        lease_expires_at = datetime.fromisoformat(
            db.execute(
                "SELECT expires_at FROM phase7_scheduler_lease WHERE singleton=1"
            ).fetchone()[0]
        )
    ready_at = max(
        publication_at,
        lease_expires_at + timedelta(milliseconds=10),
    )
    trusted_now[0] = ready_at
    artifacts.put(result_content, media_type="application/json")

    def resume_composition(path, *, owner, token, lease_ttl):
        composition = compose(
            path,
            owner=owner,
            token=token,
            lease_ttl=lease_ttl,
        )
        composition.clock = lambda: trusted_now[0]
        return composition

    assert (
        main(
            [
                "--config",
                str(config_path),
                "--once",
                "--owner",
                "runtime-resume",
                "--lease-seconds",
                "30",
            ],
            composition_loader=resume_composition,
            token_factory=lambda: "runtime-resume-token",
        )
        == 0
    )
    assert result_reads and min(result_reads) == 2
    with store._connect() as db:
        progress = db.execute(
            "SELECT phase_ordinal,command_operation_id FROM phase7_scheduler_progress "
            "WHERE racing_day_id=? ORDER BY phase_ordinal",
            (cycle["racing_day_id"],),
        ).fetchall()
        receipts = db.execute(
            "SELECT phase_name FROM phase7_application_command_receipts "
            "WHERE racing_day_id=? ORDER BY committed_at",
            (cycle["racing_day_id"],),
        ).fetchall()
        plan_count = db.execute(
            "SELECT count(DISTINCT operation_id) FROM phase7_day_command_plan "
            "WHERE racing_day_id=?",
            (cycle["racing_day_id"],),
        ).fetchone()[0]
        durable_counts = {
            table: db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in (
                "field_evidence",
                "odds_attempts",
                "sealed_evidence",
                "deferred_predictions",
                "result_attempts",
                "training_examples",
                "canonical_training_examples",
                "phase7_determinism_executions",
                "phase7_reconciliation",
                "phase7_day_training_requests",
            )
        }
        prediction_checksum = ArtifactChecksum(
            db.execute("SELECT artifact_checksum FROM deferred_predictions").fetchone()[0]
        )
        training_request = db.execute(
            "SELECT training_request_id,request_operation_id " "FROM phase7_day_training_requests"
        ).fetchone()
        service_forecasts = db.execute(
            "SELECT bundle_id,bundle_checksum,forecast_checksum,operation_id "
            "FROM phase6_forecast_service_artifacts ORDER BY bundle_id"
        ).fetchall()
        assert len(service_forecasts) == 2
        assert len({row["forecast_checksum"] for row in service_forecasts}) == 2
        expected_members = {
            (
                member["bundle_id"],
                member["bundle_checksum"],
                member["forecast_operations"][0]["operation_id"],
            )
            for member in cycle["forecast_cohort"]["members"]
        }
        assert {
            (row["bundle_id"], row["bundle_checksum"], row["operation_id"])
            for row in service_forecasts
        } == expected_members
        assert db.execute("SELECT count(*) FROM phase6_evaluation_evidence").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM phase6_trusted_evaluations").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM phase6_promotion_records").fetchone()[0] == 0
        assert (
            db.execute(
                "SELECT count(*) FROM phase6_runs " "WHERE run_kind IN ('training','promotion')"
            ).fetchone()[0]
            == 0
        )
    assert [row["phase_ordinal"] for row in progress] == list(range(1, 10))
    assert len(receipts) == 9
    assert plan_count == 1
    assert all(value > 0 for value in durable_counts.values())
    assert durable_counts["phase7_determinism_executions"] == 2
    prediction = json.loads(artifacts.read(prediction_checksum))
    assert prediction["provenance"]["artifact_checksum"] == cycle["champion"]["bundle_checksum"]
    assert tuple(training_request) == (
        cycle["training_request"]["request_id"],
        cycle["training_request"]["request_operation_id"],
    )

    assert (
        main(
            [
                "--config",
                str(config_path),
                "--once",
                "--owner",
                "runtime-replay",
            ]
        )
        == 0
    )
    with store._connect() as db:
        assert (
            db.execute(
                "SELECT count(*) FROM phase7_scheduler_progress WHERE racing_day_id=?",
                (cycle["racing_day_id"],),
            ).fetchone()[0]
            == 9
        )


def test_day_forecast_cohort_authority_is_exact_idempotent_and_immutable(tmp_path):
    store, _, config_path, cycle, _ = _runtime_fixture(tmp_path)
    composition = compose(
        config_path,
        owner="cohort-authority",
        token="cohort-authority",
        lease_ttl=timedelta(seconds=30),
    )
    runtime_cycle = composition.adapter.next_cycle(now=datetime.now(timezone.utc))
    generation = composition.maintain_lease(composition.trusted_timestamp())
    composition.authority.plan_racing_day(
        runtime_cycle.plan_operation_id,
        racing_day_id=runtime_cycle.racing_day_id,
        lease_token=composition.token,
        lease_generation=generation,
        commands=runtime_cycle.commands,
        at=composition.trusted_timestamp(),
    )
    service = RaceCollectionService(
        composition.authority,
        token=composition.token,
        generation=generation,
    )
    for ordinal in range(4):
        service.advance(
            runtime_cycle.advancement_operation_ids[ordinal],
            racing_day_id=runtime_cycle.racing_day_id,
            phase=runtime_cycle.commands[ordinal].phase,
            now=composition.trusted_timestamp(),
            command=runtime_cycle.commands[ordinal],
        )
    with store._connect() as db:
        race_id = db.execute(
            "SELECT race_id FROM races WHERE racing_day_id=?",
            (runtime_cycle.racing_day_id,),
        ).fetchone()[0]
    members = tuple(
        DayForecastCohortMember(
            member["role"],
            member["bundle_id"],
            ArtifactChecksum(member["bundle_checksum"]),
            OperationId(member["service_run_id"]),
            tuple(
                (race_id, OperationId(binding["operation_id"]))
                for binding in member["forecast_operations"]
            ),
        )
        for member in cycle["forecast_cohort"]["members"]
    )
    cohort_at = datetime.fromisoformat(cycle["races"][0]["seal"]["sealed_at"]) + timedelta(
        microseconds=1
    )

    with pytest.raises(OperationalRejected, match="distinct across"):
        composition.authority.authorize_day_forecast_cohort(
            operation(320),
            racing_day_id=runtime_cycle.racing_day_id,
            assignment_id=cycle["forecast_cohort"]["assignment_id"],
            members=(
                members[0],
                replace(
                    members[1],
                    forecast_operations=members[0].forecast_operations,
                ),
            ),
            at=cohort_at,
        )
    with pytest.raises(OperationalRejected, match="unknown, incomplete, stale"):
        composition.authority.authorize_day_forecast_cohort(
            operation(321),
            racing_day_id=runtime_cycle.racing_day_id,
            assignment_id=cycle["forecast_cohort"]["assignment_id"],
            members=(
                members[0],
                replace(
                    members[1],
                    forecast_operations=(("race_" + "8" * 32, operation(322)),),
                ),
            ),
            at=cohort_at,
        )
    with pytest.raises(OperationalRejected, match="unknown, incomplete, or stale"):
        composition.authority.authorize_day_forecast_cohort(
            operation(323),
            racing_day_id=runtime_cycle.racing_day_id,
            assignment_id=cycle["forecast_cohort"]["assignment_id"],
            members=(
                members[0],
                DayForecastCohortMember(
                    "challenger",
                    "unknown-bundle",
                    ArtifactChecksum("sha256:" + "9" * 64),
                    operation(324),
                    ((race_id, operation(325)),),
                ),
            ),
            at=cohort_at,
        )
    with pytest.raises(OperationalRejected, match="assignment-mismatched"):
        composition.authority.authorize_day_forecast_cohort(
            operation(326),
            racing_day_id=runtime_cycle.racing_day_id,
            assignment_id="stale-assignment",
            members=members,
            at=cohort_at,
        )

    authorization = OperationId(cycle["forecast_cohort"]["authorization_operation_id"])
    assert composition.authority.authorize_day_forecast_cohort(
        authorization,
        racing_day_id=runtime_cycle.racing_day_id,
        assignment_id=cycle["forecast_cohort"]["assignment_id"],
        members=members,
        at=cohort_at,
    )
    assert not composition.authority.authorize_day_forecast_cohort(
        authorization,
        racing_day_id=runtime_cycle.racing_day_id,
        assignment_id=cycle["forecast_cohort"]["assignment_id"],
        members=members,
        at=cohort_at,
    )
    service.advance(
        runtime_cycle.advancement_operation_ids[4],
        racing_day_id=runtime_cycle.racing_day_id,
        phase=runtime_cycle.commands[4].phase,
        now=composition.trusted_timestamp(),
        command=runtime_cycle.commands[4],
    )
    with pytest.raises(OperationalRejected, match="immutable after forecast work"):
        composition.authority.authorize_day_forecast_cohort(
            operation(327),
            racing_day_id=runtime_cycle.racing_day_id,
            assignment_id=cycle["forecast_cohort"]["assignment_id"],
            members=members,
            at=cohort_at + timedelta(microseconds=1),
        )
    composition.close()


def test_main_sanitizes_failure_and_closes_real_checked_in_adapter(tmp_path):
    _, artifacts, config_path, cycle, _ = _runtime_fixture(tmp_path)
    captured = {}

    def real_loader(path, *, owner, token, lease_ttl):
        composition = compose(
            path,
            owner=owner,
            token=token,
            lease_ttl=lease_ttl,
        )
        captured["composition"] = composition
        artifacts.path_for(ArtifactChecksum(cycle["programme"]["checksum"])).write_bytes(
            b"corrupt-after-compose"
        )
        return composition

    assert (
        main(
            ["--config", str(config_path), "--once", "--owner", "runtime-failure"],
            composition_loader=real_loader,
        )
        == 69
    )
    assert captured["composition"].adapter.closed
