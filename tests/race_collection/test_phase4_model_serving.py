import json
import platform
import sqlite3
import sys
import types
from dataclasses import replace
from datetime import datetime, timezone

import pytest
import numpy as np

from race_collection.artifacts import LocalArtifactStore
from race_collection.domain import ArtifactChecksum, OperationId, RaceId, RacingDayId
from race_collection.features import FeatureQuarantine, derive_features
from race_collection.forecast_service import (
    CanonicalDeferredPredictor,
    CanonicalForecastService,
    ForecastRequest,
    ForecastUnavailable,
    canonical_endpoint,
    legacy_prediction_adapter,
)
from race_collection.forecasting import (
    ForecastingAuthority,
    LegacyBundle,
    ModelRelease,
    PredictionRequest,
)
from race_collection.model_bundle import (
    BundleComponent,
    BundleUnavailable,
    CanonicalBundle,
    ChampionLoader,
    ModelBundleAuthority,
    PredictionProvenance,
    ServingAssignment,
    legacy_incumbent_conversion_status,
)
from race_collection.operations import ConflictingOperation, SQLiteOperationsStore


NOW = datetime(2026, 7, 23, 1, 2, 3, tzinfo=timezone.utc)


class SyntheticCalibratedClassifier:
    def predict_proba(self, rows):
        return np.asarray([[1.0 - min(0.9, row[0] / 10), min(0.9, row[0] / 10)] for row in rows])


def operation(number):
    return OperationId(f"op_{number:032x}")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@pytest.fixture
def canonical_setup(tmp_path):
    store = SQLiteOperationsStore(tmp_path / "operations.sqlite3")
    store.migrate()
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    schema = {
        "bundle_id": "bundle-fixture-1",
        "contract_version": "sealed-race-features-v1",
        "evidence_schema_version": "race-evidence-v1",
        "normalization_version": "normalizer-v1",
        "fields": [
            {"name": "speed", "source_field": "runner_features", "semantics": "identity-critical"},
            {"name": "form", "source_field": "runner_features", "semantics": "forecast-required"},
            {"name": "days_since_run", "source_field": "runner_features", "semantics": "optional"},
            {
                "name": "novice",
                "source_field": "runner_features",
                "semantics": "inapplicable",
                "encoded_value": -1,
            },
        ],
    }
    values = {
        "model": b"synthetic-calibrated-classifier-v1",
        "feature_schema": canonical(schema),
        "missingness_policy": canonical(
            {
                "bundle_id": "bundle-fixture-1",
                "feature_contract_version": "sealed-race-features-v1",
                "imputation": {"days_since_run": 7.0},
            }
        ),
        "training_configuration": canonical(
            {
                "model_id": "model-fixture-1",
                "feature_contract_version": "sealed-race-features-v1",
                "forecast_contract_version": "runner-win-probability-v1",
                "algorithm": "synthetic-fixture",
            }
        ),
        "dependency_manifest": canonical({"model_id": "model-fixture-1", "packages": {}}),
        "training_corpus": canonical(
            {
                "model_id": "model-fixture-1",
                "training_example_ids": ["example-1"],
                "corpus_id": "corpus-1",
            }
        ),
        "calibration": canonical(
            {
                "model_id": "model-fixture-1",
                "forecast_contract_version": "runner-win-probability-v1",
                "method": "fixture",
                "status": "calibrated",
            }
        ),
        "evaluation": canonical(
            {
                "model_id": "model-fixture-1",
                "forecast_contract_version": "runner-win-probability-v1",
                "population": "fixture",
                "log_loss": 0.2,
            }
        ),
        "runtime_requirements": canonical(
            {
                "model_id": "model-fixture-1",
                "python_implementation": platform.python_implementation(),
                "python_major_minor": f"{platform.python_version_tuple()[0]}.{platform.python_version_tuple()[1]}",
            }
        ),
    }
    components = []
    for kind, content in values.items():
        artifact = artifacts.put(content, media_type="application/octet-stream")
        components.append(BundleComponent(f"{kind}.json", kind, artifact.checksum, len(content)))
    placeholder = ArtifactChecksum("sha256:" + "0" * 64)
    provisional = CanonicalBundle(
        "bundle-fixture-1",
        "model-fixture-1",
        "canonical",
        placeholder,
        "sealed-race-features-v1",
        "runner-win-probability-v1",
        tuple(components),
        "2026-07-20",
    )
    manifest = artifacts.put(
        canonical(provisional.manifest()), media_type="application/vnd.canonical-model-bundle+json"
    )
    bundle = CanonicalBundle(
        provisional.bundle_id,
        provisional.model_id,
        provisional.origin,
        manifest.checksum,
        provisional.feature_contract_version,
        provisional.forecast_contract_version,
        provisional.components,
        provisional.trained_through,
    )
    authority = ModelBundleAuthority(store)
    authority.register(operation(1), bundle, NOW)
    assignment = ServingAssignment(
        "assignment-fixture-1",
        bundle.bundle_id,
        bundle.bundle_checksum,
        NOW.isoformat(),
        "2026-07-23",
        "promotion-fixture-1",
    )
    authority.register_assignment(operation(2), assignment, NOW)
    authority.bootstrap_champion(operation(3), assignment, NOW)
    loader = ChampionLoader(
        store, artifacts, deserializer=lambda _: SyntheticCalibratedClassifier()
    )
    return store, artifacts, bundle, loader, schema, values, assignment


def evidence_bytes():
    return canonical(
        {
            "schema_version": "race-evidence-v1",
            "normalization_version": "normalizer-v1",
            "race_id": "race_fixture",
            "fields": {
                "runner_set": ["dog-a", "dog-b"],
                "runner_identity": {"dog-a": "authoritative", "dog-b": "authoritative"},
                "runner_features": {
                    "dog-a": {
                        "speed": 8,
                        "form": 3,
                        "days_since_run": {"missing": True},
                        "novice": {"inapplicable": True},
                    },
                    "dog-b": {
                        "speed": 2,
                        "form": 6,
                        "days_since_run": 4,
                        "novice": {"inapplicable": True},
                    },
                },
            },
            "field_provenance": [],
            "freeze": {
                "at": NOW.isoformat(),
                "authority": "actual_jump",
                "odds_checksum": "sha256:" + "c" * 64,
            },
        }
    )


def test_pointer_bundle_replay_conflict_and_sql_append_only_relations(canonical_setup):
    store, _, bundle, *_, assignment = canonical_setup
    with store._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM model_bundles").fetchone()[0] == 0
    authority = ModelBundleAuthority(store)
    assert authority.register(operation(1), bundle, NOW) is False
    assert authority.register_assignment(operation(2), assignment, NOW) is False
    assert authority.bootstrap_champion(operation(3), assignment, NOW) is False
    with pytest.raises(ConflictingOperation):
        authority.bootstrap_champion(operation(3), assignment, NOW.replace(second=4))
    with store._connect() as db:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("UPDATE canonical_model_bundles SET model_id='forged'")
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO champion_pointer VALUES(2,?,?,?,?,?)",
                (
                    assignment.assignment_id,
                    bundle.bundle_id,
                    str(bundle.bundle_checksum),
                    NOW.isoformat(),
                    str(operation(99)),
                ),
            )


def test_legacy_origin_requires_exact_separate_phase3_binding(canonical_setup):
    store, artifacts, canonical_bundle, _, _, values, _ = canonical_setup
    converted_components = []
    for component in canonical_bundle.components:
        content = values[component.kind]
        if component.kind in {"feature_schema", "missingness_policy"}:
            document = json.loads(content)
            document["bundle_id"] = "converted-legacy-1"
            content = canonical(document)
        artifact = artifacts.put(content, media_type="application/octet-stream")
        converted_components.append(
            BundleComponent(component.name, component.kind, artifact.checksum, len(content))
        )
    provisional = CanonicalBundle(
        "converted-legacy-1",
        canonical_bundle.model_id,
        "legacy-origin",
        ArtifactChecksum("sha256:" + "0" * 64),
        canonical_bundle.feature_contract_version,
        canonical_bundle.forecast_contract_version,
        tuple(converted_components),
        canonical_bundle.trained_through,
        "phase3-legacy-1",
    )
    manifest = artifacts.put(canonical(provisional.manifest()), media_type="application/json")
    converted = CanonicalBundle(
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
    authority = ModelBundleAuthority(store)
    with pytest.raises(sqlite3.IntegrityError):
        authority.register(operation(50), converted, NOW)
    model = converted.component("model")
    phase3 = ForecastingAuthority(store)
    legacy = LegacyBundle(
        "phase3-legacy-1",
        converted.model_id,
        model.checksum,
        model.byte_size,
        converted.component("feature_schema").checksum,
        None,
        "raw_registry_model",
        {"source_proven": True},
    )
    phase3.register_bundle(
        operation(51),
        legacy,
        NOW,
    )
    assert authority.register(operation(52), converted, NOW)
    day = RacingDayId("day_" + "d" * 32)
    with store._connect() as db:
        db.execute(
            "INSERT INTO racing_days VALUES(?,?,?,?,NULL)",
            (str(day), "2026-07-23", "Australia/Melbourne", NOW.isoformat()),
        )
    release = ModelRelease("release-legacy-1", legacy.bundle_id, "policy-v1", {"fixture": True})
    phase3.register_release(operation(53), release, NOW)
    phase3.pin_day(operation(54), day, release, NOW)
    assignment = ServingAssignment(
        "assignment-legacy-1",
        converted.bundle_id,
        converted.bundle_checksum,
        NOW.isoformat(),
        "2026-07-23",
        "bootstrap-record-legacy-1",
    )
    authority.register_assignment(operation(55), assignment, NOW)
    authority.bind_day_assignment(operation(56), str(day), assignment, NOW)
    with pytest.raises(sqlite3.IntegrityError):
        authority.bind_day_assignment(operation(57), str(day), assignment, NOW)
    loader = ChampionLoader(
        store, artifacts, deserializer=lambda _: SyntheticCalibratedClassifier()
    )
    request = PredictionRequest(
        RaceId("race_" + "a" * 32),
        day,
        1,
        ArtifactChecksum("sha256:" + "b" * 64),
        legacy,
        release,
        release.policy_id,
    )
    pinned = loader.load_day_pin(request)
    assert pinned.bundle.bundle_id == converted.bundle_id
    assert pinned.bundle.bundle_checksum == converted.bundle_checksum
    for forged in (
        replace(request, racing_day_id=RacingDayId("day_" + "e" * 32)),
        replace(
            request, release=ModelRelease("other-release", legacy.bundle_id, release.policy_id, {})
        ),
        replace(
            request,
            release=ModelRelease(release.release_id, legacy.bundle_id, "other-policy", {}),
            policy_id="other-policy",
        ),
        replace(
            request,
            bundle=replace(legacy, artifact_checksum=ArtifactChecksum("sha256:" + "e" * 64)),
        ),
        replace(request, bundle=replace(legacy, artifact_size=legacy.artifact_size + 1)),
    ):
        with pytest.raises(BundleUnavailable, match="Racing Day pin"):
            loader.load_day_pin(forged)


def test_loader_verifies_everything_before_deserialization(canonical_setup):
    store, artifacts, bundle, loader, *_ = canonical_setup
    calls = []
    loader = ChampionLoader(store, artifacts, deserializer=lambda content: calls.append(content))
    with store._connect() as db:
        db.execute("DROP TRIGGER canonical_bundle_components_append_only_update")
        db.execute(
            "UPDATE canonical_bundle_components SET byte_size=byte_size+1 WHERE component_kind='evaluation'"
        )
    with pytest.raises(BundleUnavailable, match="manifest disagrees"):
        loader.load()
    assert calls == []


def test_registered_bundle_cannot_serve_without_complete_assignment(canonical_setup):
    store, _, _, loader, *_, assignment = canonical_setup
    forged = replace(assignment, promotion_record_id="forged-record")
    with pytest.raises(BundleUnavailable):
        loader._load_exact(forged)
    with store._connect() as db:
        db.execute("PRAGMA foreign_keys=OFF")
        db.execute("DROP TRIGGER champion_pointer_append_only_delete")
        db.execute("DROP TRIGGER canonical_serving_assignments_append_only_delete")
        db.execute("DELETE FROM champion_pointer")
        db.execute("DELETE FROM canonical_serving_assignments")
    with pytest.raises(BundleUnavailable):
        loader.load()


@pytest.mark.parametrize(
    "failure", ["missing_pointer", "manifest", "contract", "dependency", "runtime", "component"]
)
def test_loader_fail_closed_without_environment_registry_or_fallback(
    canonical_setup, monkeypatch, failure
):
    store, artifacts, bundle, loader, _, values, _ = canonical_setup
    monkeypatch.setenv("V4_MODEL_PATH", "/should/not/be/read")
    with store._connect() as db:
        if failure == "missing_pointer":
            db.execute("DROP TRIGGER champion_pointer_append_only_delete")
            db.execute("DELETE FROM champion_pointer")
        elif failure == "manifest":
            artifacts.path_for(bundle.bundle_checksum).write_bytes(b"corrupt")
        elif failure == "contract":
            db.execute("DROP TRIGGER canonical_model_bundles_append_only_update")
            db.execute("UPDATE canonical_model_bundles SET feature_contract_version='future-v9'")
        else:
            kind = failure if failure in {"dependency", "runtime"} else "evaluation"
            component_kind = {
                "dependency": "dependency_manifest",
                "runtime": "runtime_requirements",
            }.get(kind, kind)
            bad = (
                canonical({"packages": {"definitely-not-installed": "0"}})
                if failure == "dependency"
                else (
                    canonical({"python_implementation": "Other", "python_major_minor": "0.0"})
                    if failure == "runtime"
                    else b"corrupt"
                )
            )
            artifact = artifacts.put(bad, media_type="application/json")
            db.execute("DROP TRIGGER canonical_bundle_components_append_only_update")
            db.execute(
                "UPDATE canonical_bundle_components SET artifact_checksum=?,byte_size=? WHERE component_kind=?",
                (str(artifact.checksum), len(bad), component_kind),
            )
    with pytest.raises(Exception):
        loader.load()


def test_pure_deterministic_features_input_immutability_and_quarantine(canonical_setup):
    _, artifacts, bundle, _, _, values, _ = canonical_setup
    evidence = evidence_bytes()
    evidence_copy = bytes(evidence)
    checksum = artifacts.put(evidence, media_type="application/json").checksum
    kwargs = dict(
        expected_evidence_checksum=checksum,
        schema_bytes=values["feature_schema"],
        expected_schema_checksum=bundle.component("feature_schema").checksum,
        missingness_policy_bytes=values["missingness_policy"],
        expected_missingness_checksum=bundle.component("missingness_policy").checksum,
    )
    first = derive_features(evidence, **kwargs)
    second = derive_features(evidence, **kwargs)
    assert first == second and evidence == evidence_copy
    assert first.report.explicit_missing == {"dog-a": ("days_since_run",), "dog-b": ()}
    for mutation in (
        lambda value: value.update(schema_version="wrong"),
        lambda value: value["fields"]["runner_identity"].__setitem__("dog-a", "ambiguous"),
        lambda value: value["fields"]["runner_features"]["dog-a"].pop("speed"),
    ):
        value = json.loads(evidence)
        mutation(value)
        content = canonical(value)
        with pytest.raises(FeatureQuarantine):
            derive_features(
                content,
                **{**kwargs, "expected_evidence_checksum": LocalArtifactStore.checksum(content)},
            )


def test_parallel_top_level_evidence_envelope_is_rejected(canonical_setup):
    _, _, bundle, _, _, values, _ = canonical_setup
    envelope = json.loads(evidence_bytes())
    envelope["runners"] = envelope["fields"].pop("runner_set")
    content = canonical(envelope)
    with pytest.raises(FeatureQuarantine):
        derive_features(
            content,
            expected_evidence_checksum=LocalArtifactStore.checksum(content),
            schema_bytes=values["feature_schema"],
            expected_schema_checksum=bundle.component("feature_schema").checksum,
            missingness_policy_bytes=values["missingness_policy"],
            expected_missingness_checksum=bundle.component("missingness_policy").checksum,
        )


def test_service_endpoint_and_legacy_adapter_are_exact_and_provenanced(canonical_setup):
    _, artifacts, bundle, loader, *_ = canonical_setup
    evidence = artifacts.put(evidence_bytes(), media_type="application/json")
    service = CanonicalForecastService(loader, artifacts, clock=lambda: NOW)
    request = ForecastRequest(evidence.checksum)
    result = service.forecast(request)
    assert result["predictions"] == [
        {"dog_id": "dog-a", "win_probability": 0.8, "rank": 1},
        {"dog_id": "dog-b", "win_probability": 0.2, "rank": 2},
    ]
    assert set(result["provenance"]) == set(PredictionProvenance.__dataclass_fields__)
    payload = {
        "evidence_checksum": str(evidence.checksum),
        "sp": {"dog-b": 1.2},
        "gpt_rerank": True,
    }
    assert canonical_endpoint(service, payload) == legacy_prediction_adapter(service, payload)
    assert canonical_endpoint(service, payload)[0] == result


def test_runner_win_contract_uses_predict_proba_and_never_latent_strengths(canonical_setup):
    store, artifacts, _, _, *_ = canonical_setup

    class BothInterfaces:
        def predict_proba(self, rows):
            return [[0.2, 0.8], [0.8, 0.2]]

        def latent_strengths(self, rows):
            raise AssertionError("runner-win contract must not call latent_strengths")

    loader = ChampionLoader(store, artifacts, deserializer=lambda _: BothInterfaces())
    evidence = artifacts.put(evidence_bytes(), media_type="application/json")
    result = CanonicalForecastService(loader, artifacts, clock=lambda: NOW).forecast(
        ForecastRequest(evidence.checksum)
    )
    assert [row["win_probability"] for row in result["predictions"]] == [0.8, 0.2]
    assert "exacta_probabilities" not in result


def test_runner_win_loader_rejects_latent_only_model(canonical_setup):
    store, artifacts, _, _, *_ = canonical_setup

    class LatentOnly:
        def latent_strengths(self, rows):
            return [0.0 for _ in rows]

    with pytest.raises(BundleUnavailable, match="predict_proba"):
        ChampionLoader(store, artifacts, deserializer=lambda _: LatentOnly()).load()


def test_deferred_artifact_authenticates_phase3_commit_time(canonical_setup):
    _, artifacts, _, loader, *_ = canonical_setup
    evidence = artifacts.put(evidence_bytes(), media_type="application/json")
    service = CanonicalForecastService(loader, artifacts, clock=lambda: NOW)
    result = service.forecast(ForecastRequest(evidence.checksum))
    artifact = artifacts.put(canonical(result), media_type="application/json")
    predictor = CanonicalDeferredPredictor(service, artifacts, clock=lambda: NOW)
    predictor.authenticate(artifact.checksum, NOW)
    with pytest.raises(ForecastUnavailable, match="computation time disagrees"):
        predictor.authenticate(artifact.checksum, NOW.replace(second=4))


@pytest.mark.parametrize("frozen_at", [None, "not-a-time", "2026-07-23T01:02:03"])
def test_sealed_evidence_owns_aware_freeze_timestamp(canonical_setup, frozen_at):
    _, artifacts, _, loader, *_ = canonical_setup
    evidence = json.loads(evidence_bytes())
    if frozen_at is None:
        evidence["freeze"].pop("at")
    else:
        evidence["freeze"]["at"] = frozen_at
    artifact = artifacts.put(canonical(evidence), media_type="application/json")
    service = CanonicalForecastService(loader, artifacts, clock=lambda: NOW)
    with pytest.raises(ForecastUnavailable):
        service.forecast(ForecastRequest(artifact.checksum))


def test_deferred_requires_durable_seal_identity_before_scoring(canonical_setup):
    _, artifacts, _, loader, *_ = canonical_setup
    evidence = artifacts.put(evidence_bytes(), media_type="application/json")
    service = CanonicalForecastService(loader, artifacts, clock=lambda: NOW)
    with pytest.raises(ForecastUnavailable, match="seal relation"):
        service.forecast(ForecastRequest(evidence.checksum, 1, "race_fixture"))


def test_deferred_success_authenticates_durable_seal_before_scoring(canonical_setup, monkeypatch):
    _, artifacts, _, loader, *_ = canonical_setup
    evidence = artifacts.put(evidence_bytes(), media_type="application/json")
    calls = []
    monkeypatch.setattr(
        loader,
        "authenticate_seal",
        lambda **kwargs: calls.append(kwargs) or NOW,
    )
    service = CanonicalForecastService(loader, artifacts, clock=lambda: NOW)
    result = service.forecast(ForecastRequest(evidence.checksum, 7, "race_fixture"))
    assert calls[0]["seal_id"] == 7
    assert datetime.fromisoformat(result["provenance"]["evidence_frozen_at"]) == NOW


@pytest.mark.parametrize("mismatch", ["race", "checksum", "frozen_at"])
def test_deferred_seal_mismatches_fail_before_model_load(canonical_setup, monkeypatch, mismatch):
    _, artifacts, _, loader, *_ = canonical_setup
    evidence = artifacts.put(evidence_bytes(), media_type="application/json")
    loaded = []
    monkeypatch.setattr(loader, "load", lambda: loaded.append(True))
    if mismatch == "race":
        request = ForecastRequest(evidence.checksum, 7, "other-race")
    elif mismatch == "checksum":
        request = ForecastRequest(ArtifactChecksum("sha256:" + "f" * 64), 7, "race_fixture")
    else:
        monkeypatch.setattr(loader, "authenticate_seal", lambda **_: NOW.replace(second=4))
        request = ForecastRequest(evidence.checksum, 7, "race_fixture")
    with pytest.raises(ForecastUnavailable):
        CanonicalForecastService(loader, artifacts, clock=lambda: NOW).forecast(request)
    assert loaded == []


@pytest.mark.parametrize("field", list(PredictionProvenance.__dataclass_fields__))
def test_every_missing_or_unknown_provenance_field_fails(field):
    values = {
        "champion_model_id": "m",
        "artifact_checksum": "sha256:" + "a" * 64,
        "trained_through": "2026-07-20",
        "promotion_approved_at": NOW.isoformat(),
        "promotion_effective_from_racing_day": "2026-07-23",
        "promotion_record_id": "p",
        "prediction_computed_at": NOW.isoformat(),
        "evidence_frozen_at": NOW.isoformat(),
    }
    values[field] = "unknown"
    with pytest.raises((ValueError, TypeError)):
        PredictionProvenance(**values)


def test_legacy_conversion_is_honest_and_on_demand_unavailable(tmp_path):
    status = legacy_incumbent_conversion_status()
    assert status["classification"] == "sklearn.calibration.CalibratedClassifierCV"
    assert (
        status["status"] == "quarantined" and "trained_through" in status["missing_mandatory_facts"]
    )
    store = SQLiteOperationsStore(tmp_path / "db")
    store.migrate()
    service = CanonicalForecastService(
        ChampionLoader(store, LocalArtifactStore(tmp_path / "a"), deserializer=lambda _: None),
        LocalArtifactStore(tmp_path / "a"),
    )
    with pytest.raises(ForecastUnavailable):
        service.forecast(ForecastRequest(ArtifactChecksum("sha256:" + "a" * 64)))


@pytest.mark.parametrize("target", ["bundle", "bundle_tamper", "component", "pointer"])
def test_operation_replay_authenticates_durable_result(canonical_setup, target):
    store, _, bundle, *_, assignment = canonical_setup
    authority = ModelBundleAuthority(store)
    with store._connect() as db:
        if target == "bundle":
            db.execute("PRAGMA foreign_keys=OFF")
            db.execute("DROP TRIGGER canonical_model_bundles_append_only_delete")
            db.execute("DELETE FROM canonical_model_bundles")
        elif target == "bundle_tamper":
            db.execute("DROP TRIGGER canonical_model_bundles_append_only_update")
            db.execute("UPDATE canonical_model_bundles SET trained_through='2026-07-19'")
        elif target == "component":
            db.execute("DROP TRIGGER canonical_bundle_components_append_only_delete")
            db.execute("DELETE FROM canonical_bundle_components WHERE component_kind='evaluation'")
        else:
            db.execute("DROP TRIGGER champion_pointer_append_only_delete")
            db.execute("DELETE FROM champion_pointer")
    with pytest.raises(Exception, match="replay lacks exact durable"):
        if target == "pointer":
            authority.bootstrap_champion(operation(3), assignment, NOW)
        else:
            authority.register(operation(1), bundle, NOW)


@pytest.mark.parametrize(
    "case",
    ["duplicate", "semantics", "nan_imputation", "undeclared_imputation"],
)
def test_feature_contract_rejects_invalid_schema_and_imputation(canonical_setup, case):
    _, _, _, _, schema, values, _ = canonical_setup
    changed = json.loads(json.dumps(schema))
    policy = json.loads(values["missingness_policy"])
    if case == "duplicate":
        changed["fields"].append(dict(changed["fields"][0]))
    elif case == "semantics":
        changed["fields"][0]["semantics"] = "mystery"
    elif case == "nan_imputation":
        policy["imputation"]["days_since_run"] = float("nan")
    else:
        policy["imputation"]["undeclared"] = 0
    schema_bytes = canonical(changed)
    policy_bytes = json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    evidence = evidence_bytes()
    with pytest.raises(FeatureQuarantine):
        derive_features(
            evidence,
            expected_evidence_checksum=LocalArtifactStore.checksum(evidence),
            schema_bytes=schema_bytes,
            expected_schema_checksum=LocalArtifactStore.checksum(schema_bytes),
            missingness_policy_bytes=policy_bytes,
            expected_missingness_checksum=LocalArtifactStore.checksum(policy_bytes),
        )


@pytest.mark.parametrize("encoded", [None, True, float("nan"), float("inf")])
def test_inapplicable_encoding_must_be_declared_finite(canonical_setup, encoded):
    _, _, _, _, schema, values, _ = canonical_setup
    changed = json.loads(json.dumps(schema))
    next(item for item in changed["fields"] if item["semantics"] == "inapplicable")[
        "encoded_value"
    ] = encoded
    schema_bytes = json.dumps(changed, sort_keys=True, separators=(",", ":")).encode()
    evidence = evidence_bytes()
    with pytest.raises(FeatureQuarantine):
        derive_features(
            evidence,
            expected_evidence_checksum=LocalArtifactStore.checksum(evidence),
            schema_bytes=schema_bytes,
            expected_schema_checksum=LocalArtifactStore.checksum(schema_bytes),
            missingness_policy_bytes=values["missingness_policy"],
            expected_missingness_checksum=LocalArtifactStore.checksum(values["missingness_policy"]),
        )


@pytest.mark.parametrize("case", ["nonfinite", "inapplicable", "required"])
def test_feature_semantics_reject_invalid_values(canonical_setup, case):
    _, _, bundle, _, _, values, _ = canonical_setup
    evidence = json.loads(evidence_bytes())
    features = evidence["fields"]
    if case == "nonfinite":
        features["runner_features"]["dog-a"]["speed"] = float("inf")
    elif case == "inapplicable":
        features["runner_features"]["dog-a"]["novice"] = 1
    else:
        features["runner_features"]["dog-a"].pop("form")
    content = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(FeatureQuarantine):
        derive_features(
            content,
            expected_evidence_checksum=LocalArtifactStore.checksum(content),
            schema_bytes=values["feature_schema"],
            expected_schema_checksum=bundle.component("feature_schema").checksum,
            missingness_policy_bytes=values["missingness_policy"],
            expected_missingness_checksum=bundle.component("missingness_policy").checksum,
        )


@pytest.mark.parametrize(
    "probabilities",
    [
        np.asarray([[0.0, float("nan")], [0.8, 0.2]]),
        np.asarray([[float("nan"), 0.2], [0.8, 0.2]]),
        np.asarray([[float("inf"), 0.2], [0.8, 0.2]]),
        np.asarray([[1.2, 0.2], [0.8, 0.2]]),
        np.asarray([["bad", 0.2], [0.8, 0.2]]),
        np.asarray([[0.7, 0.2], [0.8, 0.2]]),
        np.asarray([[0.2], [0.8]]),
        np.asarray([0.2, 0.8]),
        "bad",
    ],
)
def test_service_rejects_nonfinite_and_malformed_predict_proba(canonical_setup, probabilities):
    _, artifacts, _, loader, *_ = canonical_setup

    class BadModel:
        def predict_proba(self, rows):
            return probabilities

    bad_loader = ChampionLoader(loader.store, artifacts, deserializer=lambda _: BadModel())
    service = CanonicalForecastService(bad_loader, artifacts, clock=lambda: NOW)
    evidence = artifacts.put(evidence_bytes(), media_type="application/json")
    with pytest.raises(ForecastUnavailable):
        service.forecast(ForecastRequest(evidence.checksum))


def test_real_flask_canonical_route_and_every_sealed_evidence_adapter(canonical_setup, monkeypatch):
    monkeypatch.setitem(sys.modules, "seaborn", types.ModuleType("seaborn"))
    import app as flask_application

    _, artifacts, _, loader, *_ = canonical_setup
    evidence = artifacts.put(evidence_bytes(), media_type="application/json")
    service = CanonicalForecastService(loader, artifacts, clock=lambda: NOW)
    flask_application.configure_canonical_forecast_service(service)
    client = flask_application.app.test_client()
    payload = {
        "evidence_checksum": str(evidence.checksum),
        "sp": {"dog-b": 1.01},
        "gpt_rerank": True,
    }
    routes = sorted(flask_application._SEALED_EVIDENCE_FORECAST_ROUTES)
    responses = [client.post(route, json=payload) for route in routes]
    assert all(response.status_code == 200 for response in responses)
    assert all(response.get_json() == responses[0].get_json() for response in responses)
    provenance = responses[0].get_json()["provenance"]
    assert provenance["prediction_computed_at"] == NOW.isoformat(timespec="microseconds")
    assert provenance["evidence_frozen_at"] == NOW.isoformat(timespec="microseconds")
    flask_application.configure_canonical_forecast_service(None)
    unavailable = client.post("/api/canonical/forecast", json=payload)
    assert unavailable.status_code == 503
    assert unavailable.get_json()["status"] == "unavailable"
    flask_application.configure_canonical_forecast_service(service)
    forged = dict(payload, prediction_computed_at="1900-01-01T00:00:00+00:00")
    rejected = client.post("/api/canonical/forecast", json=forged)
    assert rejected.status_code == 400
