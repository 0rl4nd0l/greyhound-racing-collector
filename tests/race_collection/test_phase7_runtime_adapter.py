import hashlib
import json
import subprocess
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

import race_collection.runtime_adapters as runtime_adapters
from race_collection.artifacts import LocalArtifactStore
from race_collection.domain import (
    ADAPTIVE_ODDS_TIMING_POLICY,
    ArtifactChecksum,
    OperationId,
    RaceId,
    RacingDay,
    RacingDayId,
)
from race_collection.evaluation import EvaluationAuthority, PromotionPolicy
from race_collection.forecasting import ForecastingAuthority, LegacyBundle, ModelRelease
from race_collection.model_bundle import (
    SUPPORTED_FORECAST_CONTRACT,
    BundleComponent,
    CanonicalBundle,
    ModelBundleAuthority,
    ServingAssignment,
)
from race_collection.operational import (
    DayForecastCohortMember,
    OperationalAuthority,
    OperationalRejected,
    RaceCollectionService,
    ReleaseConfiguration,
    ReleaseManifest,
)
from race_collection.operations import BarrierNotSatisfied, SQLiteOperationsStore
from race_collection.ordered_finish import ORDERED_FINISH_CONTRACT
from race_collection.runtime_adapters import (
    OFFICIAL_RESULT_MAX_LATENCY,
    OFFICIAL_RESULT_TIMING_POLICY,
    ImmutableInputRuntimeAdapter,
    _official_result_timeline_valid,
)
from race_collection.service import RacingDayCycle, ServiceUnavailable, compose, main
from race_collection.training import LinearStrengthModel

NOW = datetime.now(timezone.utc).replace(microsecond=0)
LOCAL_DATE = NOW.astimezone(ZoneInfo("Australia/Melbourne")).date()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def operation(number):
    return OperationId(f"op_{number:032x}")


def test_official_result_timeline_accepts_positive_latency_and_exact_boundary():
    published = NOW
    observed = published + timedelta(seconds=30)
    attempted = observed + timedelta(seconds=30)
    trusted = published + OFFICIAL_RESULT_MAX_LATENCY
    assert _official_result_timeline_valid(
        published, observed, attempted, trusted, OFFICIAL_RESULT_TIMING_POLICY
    )
    assert _official_result_timeline_valid(
        published,
        observed,
        published + OFFICIAL_RESULT_MAX_LATENCY,
        trusted,
        OFFICIAL_RESULT_TIMING_POLICY,
    )


@pytest.mark.parametrize(
    ("published", "observed", "attempted", "trusted", "policy"),
    [
        (None, NOW, NOW, NOW, OFFICIAL_RESULT_TIMING_POLICY),
        (NOW, None, NOW, NOW, OFFICIAL_RESULT_TIMING_POLICY),
        (NOW, NOW, None, NOW, OFFICIAL_RESULT_TIMING_POLICY),
        (NOW, NOW + timedelta(seconds=1), NOW, NOW, OFFICIAL_RESULT_TIMING_POLICY),
        (NOW, NOW, NOW + timedelta(seconds=1), NOW, OFFICIAL_RESULT_TIMING_POLICY),
        (
            NOW,
            NOW,
            NOW + OFFICIAL_RESULT_MAX_LATENCY + timedelta(microseconds=1),
            NOW + OFFICIAL_RESULT_MAX_LATENCY + timedelta(microseconds=1),
            OFFICIAL_RESULT_TIMING_POLICY,
        ),
        (NOW, NOW, NOW, NOW, "official-result-timing-v0"),
    ],
)
def test_official_result_timeline_rejects_missing_reversed_excessive_or_wrong_policy(
    published, observed, attempted, trusted, policy
):
    assert not _official_result_timeline_valid(published, observed, attempted, trusted, policy)


def test_odds_cohort_is_fully_validated_before_authority_mutation():
    mutations = []

    class Store:
        def advance_race(self, *args):
            mutations.append(("advance", args))

    class Repository:
        def record_odds_attempt(self, observation):
            mutations.append(("record", observation))

    class Artifacts:
        def verify(self, checksum):
            return checksum

    due = NOW
    jump = due + timedelta(seconds=30)
    payload_checksum = "sha256:" + "1" * 64
    mapping_checksum = "sha256:" + "2" * 64

    def race(number, status):
        return (
            RaceId(f"race_{number:032x}"),
            {
                "odds_attempts": [
                    {
                        "operation_id": f"op_{number:032x}",
                        "source": "official-odds",
                        "scheduled_due_at": due.isoformat(),
                        "attempted_at": (due + timedelta(seconds=2)).isoformat(),
                        "timing_policy": ADAPTIVE_ODDS_TIMING_POLICY,
                        "status": status,
                        "artifact_checksum": payload_checksum,
                        "runner_mapping_checksum": mapping_checksum,
                        "error": None,
                    }
                ]
            },
            {"scheduled_jump": jump.isoformat()},
        )

    adapter = object.__new__(ImmutableInputRuntimeAdapter)
    adapter._store = Store()
    adapter._repository = Repository()
    adapter._artifacts = Artifacts()
    adapter._race_inputs = lambda command: (race(1, "succeeded"), race(2, "not-canonical"))

    with pytest.raises(ServiceUnavailable):
        adapter._odds(SimpleNamespace(operation_id=operation(100)), NOW)
    assert mutations == []


def test_result_cohort_is_fully_validated_before_authority_mutation(monkeypatch):
    mutations = []

    class Authority:
        def open_results(self, *args):
            mutations.append(("open", args))

        def record_result_attempt(self, *args, **kwargs):
            mutations.append(("record", args, kwargs))
            return "collected"

    class Artifacts:
        def __init__(self, content):
            self.content = content

        def read(self, checksum):
            return self.content[str(checksum)]

    def result(number, observed_at):
        checksum = ArtifactChecksum(f"sha256:{number:064x}")
        source = f"official-results-{number}"
        attempted_at = NOW + timedelta(minutes=1)
        outcome = {
            "source": source,
            "published_at": NOW.isoformat(),
            "provenance": {"observed_at": observed_at.isoformat()},
            "order": [1, 2],
        }
        return (
            RaceId(f"race_{number:032x}"),
            {
                "result": {
                    "open_operation_id": f"op_{number:032x}",
                    "operation_id": f"op_{number + 10:032x}",
                    "attempt_id": f"attempt-{number}",
                    "attempted_at": attempted_at.isoformat(),
                    "timing_policy": OFFICIAL_RESULT_TIMING_POLICY,
                    "deadline": (NOW + timedelta(minutes=10)).isoformat(),
                    "max_attempts": 3,
                    "source": source,
                    "source_checksum": str(checksum),
                }
            },
            {},
            checksum,
            canonical(outcome),
        )

    first = result(1, NOW + timedelta(seconds=30))
    second = result(2, NOW + timedelta(minutes=2))
    adapter = object.__new__(ImmutableInputRuntimeAdapter)
    adapter._store = object()
    adapter._artifacts = Artifacts({str(first[3]): first[4], str(second[3]): second[4]})
    adapter._race_inputs = lambda command: (
        (first[0], first[1], first[2]),
        (second[0], second[1], second[2]),
    )
    monkeypatch.setattr(runtime_adapters, "ForecastingAuthority", lambda store: Authority())

    with pytest.raises(ServiceUnavailable):
        adapter._results(SimpleNamespace(), NOW + timedelta(minutes=3))
    assert mutations == []


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


def _runtime_fixture(
    tmp_path,
    *,
    result_blind=False,
    runtime_mode=None,
    release_authority=None,
    unregistered_role=None,
    release_bundle_versions=None,
    first_cohort_role=None,
):
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
    result_published_at = result_at - timedelta(seconds=1)
    result_observed_at = result_at - timedelta(milliseconds=500)
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
                "observed_at": result_observed_at.isoformat(),
            },
            "published_at": result_published_at.isoformat(),
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
                        "scheduled_due_at": discovery.isoformat(),
                        "attempted_at": discovery.isoformat(),
                        "timing_policy": "adaptive-odds-timing-v1",
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
                    "timing_policy": OFFICIAL_RESULT_TIMING_POLICY,
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
    if result_blind:
        if runtime_mode is not None:
            raise ValueError("result_blind and runtime_mode are mutually exclusive")
        runtime_mode = "result-blind-observation-v1"
    if runtime_mode is not None:
        cycle["mode"] = runtime_mode
    if runtime_mode == "result-blind-observation-v1":
        del cycle["races"][0]["result"]
        del cycle["races"][0]["training_example"]
    if unregistered_role is not None:
        replacement = {
            "bundle_id": f"unregistered-{unregistered_role}",
            "bundle_checksum": "sha256:" + "f" * 64,
        }
        member = next(
            item
            for item in cycle["forecast_cohort"]["members"]
            if item["role"] == unregistered_role
        )
        member.update(replacement)
        if unregistered_role == "champion":
            cycle["champion"].update(replacement)
    if first_cohort_role is not None:
        cycle["forecast_cohort"]["members"].sort(
            key=lambda member: member["role"] != first_cohort_role
        )
    runtime_document = {
        "schema_version": "phase7-runtime-input-v1",
        "release_id": "runtime-candidate",
        "cycles": [cycle],
    }
    runtime_artifact = artifacts.put(canonical(runtime_document), media_type="application/json")
    source_root = Path(__file__).resolve().parents[2]
    source_commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    bundle_versions = (
        (ORDERED_FINISH_CONTRACT,)
        if release_bundle_versions is None
        else tuple(release_bundle_versions)
    )
    configuration = ReleaseConfiguration(
        "phase7-config-v1",
        str(source_root),
        str(artifacts.root),
        str(store.path),
        ("official", "official-card", "official-form", "market", "official-results"),
        "adaptive-odds-v1",
        "phase6-promotion-v1",
        bundle_versions,
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
            source_commit,
            config_checksum,
            29,
            "canonical-artifacts-v1",
            "phase6-promotion-v1",
            bundle_versions,
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
    release_authority = release_authority or (
        "observation" if runtime_mode == "result-blind-observation-v1" else "active"
    )
    if release_authority == "observation":
        authority.authorize_observation(
            operation(305),
            candidate_release_id="runtime-candidate",
            actor="fixture",
            reason="exercise checked-in adapter",
            at=NOW - timedelta(days=1),
        )
    elif release_authority == "active":
        with store._operation(operation(306), "fixture_activate_release", {}) as (db, _):
            db.execute(
                "UPDATE phase7_release_pointer SET release_id=?,"
                "authority='race_collection_service',changed_at=?,operation_id=? "
                "WHERE singleton=1",
                (
                    "runtime-candidate",
                    (NOW - timedelta(days=1)).isoformat(),
                    str(operation(306)),
                ),
            )
    else:
        raise ValueError("unsupported fixture release authority")
    config_path = tmp_path / "release.json"
    config_path.write_bytes(canonical(configuration.document()))
    return store, artifacts, config_path, cycle, result_content


def test_result_blind_observation_stops_after_receipt_five_without_result_work(
    tmp_path, monkeypatch
):
    store, _, config_path, cycle, _ = _runtime_fixture(tmp_path, result_blind=True)
    monkeypatch.setattr(
        "race_collection.service.verify_release_source_identity",
        lambda *_args, **_kwargs: None,
    )
    trusted_now = datetime.now(timezone.utc)

    def load(path, *, owner, token, lease_ttl):
        composition = compose(
            path,
            owner=owner,
            token=token,
            lease_ttl=lease_ttl,
        )
        composition.clock = lambda: trusted_now
        return composition

    arguments = [
        "--config",
        str(config_path),
        "--once",
        "--owner",
        "result-blind-observer",
    ]
    assert main(arguments, composition_loader=load, token_factory=lambda: "blind-token") == 0
    assert main(arguments, composition_loader=load, token_factory=lambda: "blind-replay") == 0

    with store._connect() as db:
        progress = db.execute(
            "SELECT phase_ordinal,phase_name FROM phase7_scheduler_progress "
            "WHERE racing_day_id=? ORDER BY phase_ordinal",
            (cycle["racing_day_id"],),
        ).fetchall()
        receipts = db.execute(
            "SELECT phase_name FROM phase7_application_command_receipts "
            "WHERE racing_day_id=? ORDER BY committed_at",
            (cycle["racing_day_id"],),
        ).fetchall()
        assert [tuple(row) for row in progress] == [
            (1, "discover_programme"),
            (2, "collect_cards_and_form"),
            (3, "collect_adaptive_odds"),
            (4, "close_and_seal"),
            (5, "deferred_prediction"),
        ]
        assert [row[0] for row in receipts] == [row[1] for row in progress]
        assert db.execute("SELECT count(*) FROM result_attempts").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM training_examples").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM canonical_training_examples").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM phase7_reconciliation").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM phase7_day_training_requests").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM phase6_evaluation_evidence").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM phase6_trusted_evaluations").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM phase6_promotion_records").fetchone()[0] == 0
        assert (
            db.execute(
                "SELECT count(*) FROM phase6_runs WHERE run_kind IN ('training','promotion')"
            ).fetchone()[0]
            == 0
        )
        assert (
            db.execute(
                "SELECT count(*) FROM phase7_day_command_plan WHERE racing_day_id=?",
                (cycle["racing_day_id"],),
            ).fetchone()[0]
            == 9
        )
    assert "result" not in cycle["races"][0]
    assert "training_example" not in cycle["races"][0]


@pytest.mark.parametrize("runtime_mode", [None, "complete-v1"])
def test_observation_authority_rejects_missing_or_complete_runtime_mode(
    tmp_path, monkeypatch, runtime_mode
):
    store, _, config_path, _, _ = _runtime_fixture(
        tmp_path,
        runtime_mode=runtime_mode,
        release_authority="observation",
    )
    monkeypatch.setattr(
        "race_collection.service.verify_release_source_identity",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(ServiceUnavailable, match="composition failed") as rejected:
        compose(
            config_path,
            owner="observation-mode-boundary",
            token="observation-mode-boundary",
            lease_ttl=timedelta(seconds=30),
        )

    assert isinstance(rejected.value.__cause__, ServiceUnavailable)
    assert "explicit result-blind" in str(rejected.value.__cause__)
    with store._connect() as db:
        assert db.execute("SELECT count(*) FROM phase7_day_command_plan").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM phase7_scheduler_progress").fetchone()[0] == 0


def test_observation_authority_rejects_unknown_runtime_mode(tmp_path, monkeypatch):
    store, _, config_path, _, _ = _runtime_fixture(
        tmp_path,
        runtime_mode="unknown-v1",
        release_authority="observation",
    )
    monkeypatch.setattr(
        "race_collection.service.verify_release_source_identity",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(ServiceUnavailable, match="composition failed") as rejected:
        compose(
            config_path,
            owner="unknown-mode-boundary",
            token="unknown-mode-boundary",
            lease_ttl=timedelta(seconds=30),
        )

    assert isinstance(rejected.value.__cause__, ServiceUnavailable)
    assert "unsupported" in str(rejected.value.__cause__)
    with store._connect() as db:
        assert db.execute("SELECT count(*) FROM phase7_day_command_plan").fetchone()[0] == 0


def test_observation_authority_rejects_terminal_phase_after_deferred_prediction(
    tmp_path, monkeypatch
):
    store, _, config_path, cycle_document, _ = _runtime_fixture(tmp_path, result_blind=True)
    monkeypatch.setattr(
        "race_collection.service.verify_release_source_identity",
        lambda *_args, **_kwargs: None,
    )
    composition = compose(
        config_path,
        owner="terminal-boundary",
        token="terminal-boundary",
        lease_ttl=timedelta(seconds=30),
    )
    cycle = composition.adapter.next_cycle(now=datetime.now(timezone.utc))
    assert cycle is not None
    assert [command.phase for command in cycle.commands] == list(RaceCollectionService.ORDER[:5])
    complete_cycle = RacingDayCycle(
        cycle.racing_day_id,
        cycle.planned_commands,
        cycle.plan_operation_id,
        tuple(OperationId(value) for value in cycle_document["advancement_operation_ids"]),
        cycle.at,
    )

    with pytest.raises(OperationalRejected, match="observation authority"):
        composition.run_cycle(complete_cycle)
    with pytest.raises(ServiceUnavailable, match="conflicts with prior binding"):
        composition.adapter.bind_release_authority("active")

    with store._connect() as db:
        assert db.execute("SELECT count(*) FROM phase7_day_command_plan").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM phase7_scheduler_progress").fetchone()[0] == 0
    composition.close()


def test_full_authority_preserves_explicit_complete_cycle(tmp_path, monkeypatch):
    _, _, config_path, _, _ = _runtime_fixture(
        tmp_path,
        runtime_mode="complete-v1",
        release_authority="active",
    )
    monkeypatch.setattr(
        "race_collection.service.verify_release_source_identity",
        lambda *_args, **_kwargs: None,
    )
    composition = compose(
        config_path,
        owner="complete-cycle",
        token="complete-cycle",
        lease_ttl=timedelta(seconds=30),
    )
    cycle = composition.adapter.next_cycle(now=datetime.now(timezone.utc))

    assert composition.mode == "active"
    assert cycle is not None
    assert cycle.terminal_phase == "request_training"
    assert [command.phase for command in cycle.commands] == list(RaceCollectionService.ORDER)
    with pytest.raises(ValueError, match="exact ordered Racing Day plan"):
        replace(cycle, plan_commands=())
    composition.close()


@pytest.mark.parametrize("conflicting_mode", [None, "unknown"])
def test_composed_release_identity_with_missing_or_unknown_authority_mode_fails_closed(
    tmp_path, monkeypatch, conflicting_mode
):
    store, _, config_path, _, _ = _runtime_fixture(
        tmp_path,
        runtime_mode="complete-v1",
        release_authority="active",
    )
    monkeypatch.setattr(
        "race_collection.service.verify_release_source_identity",
        lambda *_args, **_kwargs: None,
    )
    composition = compose(
        config_path,
        owner="authority-mode-shape",
        token="authority-mode-shape",
        lease_ttl=timedelta(seconds=30),
    )
    composition.mode = conflicting_mode

    with pytest.raises(OperationalRejected, match="release authority mode"):
        composition._revalidate_release_mode()

    with store._connect() as db:
        assert db.execute("SELECT count(*) FROM phase7_day_command_plan").fetchone()[0] == 0
    composition.close()


@pytest.mark.parametrize("role", ["champion", "challenger"])
def test_adapter_rejects_unregistered_forecast_cohort_before_receipt_one(
    tmp_path, monkeypatch, role
):
    store, _, config_path, _, _ = _runtime_fixture(tmp_path, unregistered_role=role)
    monkeypatch.setattr(
        "race_collection.service.verify_release_source_identity",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(ServiceUnavailable, match="composition failed") as rejected:
        compose(
            config_path,
            owner="registration-preflight",
            token="registration-preflight",
            lease_ttl=timedelta(seconds=30),
        )
    assert isinstance(rejected.value.__cause__, ServiceUnavailable)
    assert "registered" in str(rejected.value.__cause__)
    with store._connect() as db:
        assert (
            db.execute("SELECT count(*) FROM phase7_application_command_receipts").fetchone()[0]
            == 0
        )


@pytest.mark.parametrize("role", ["champion", "challenger"])
def test_adapter_rejects_registered_cohort_contract_excluded_by_release_before_receipt_one(
    tmp_path, monkeypatch, role
):
    store, _, config_path, _, _ = _runtime_fixture(
        tmp_path,
        release_bundle_versions=(SUPPORTED_FORECAST_CONTRACT,),
        first_cohort_role=role,
    )
    monkeypatch.setattr(
        "race_collection.service.verify_release_source_identity",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(ServiceUnavailable, match="composition failed") as rejected:
        compose(
            config_path,
            owner="release-contract-preflight",
            token="release-contract-preflight",
            lease_ttl=timedelta(seconds=30),
        )
    assert isinstance(rejected.value.__cause__, ServiceUnavailable)
    assert f"runtime {role}" in str(rejected.value.__cause__)
    assert "configured release" in str(rejected.value.__cause__)
    with store._connect() as db:
        assert (
            db.execute("SELECT count(*) FROM phase7_application_command_receipts").fetchone()[0]
            == 0
        )


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
