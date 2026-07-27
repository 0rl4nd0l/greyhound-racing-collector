import copy
import hashlib
import json

import pytest

from race_collection.source_admission import (
    SourceAdmissionRejected,
    admit_historical_source,
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def checksum(content):
    return "sha256:" + hashlib.sha256(content).hexdigest()


def normalized_manifest_checksum(manifest):
    normalized = {**manifest, "races": sorted(manifest["races"], key=lambda race: race["race_id"])}
    return checksum(canonical(normalized))


def source_package(*, origin="synthetic-validation-fixture-v1", race_ids=("race-a", "race-b")):
    schema = canonical(
        {
            "bundle_id": "bundle-native-phase7-1",
            "contract_version": "sealed-race-features-v1",
            "evidence_schema_version": "race-evidence-v1",
            "normalization_version": "normalizer-v1",
            "fields": [
                {
                    "name": "speed",
                    "source_field": "runner_features",
                    "semantics": "forecast-required",
                },
                {
                    "name": "days_since_run",
                    "source_field": "runner_features",
                    "semantics": "optional",
                },
            ],
        }
    )
    missingness = canonical(
        {
            "bundle_id": "bundle-native-phase7-1",
            "feature_contract_version": "sealed-race-features-v1",
            "imputation": {"days_since_run": 7.0},
        }
    )
    artifacts = {checksum(schema): schema, checksum(missingness): missingness}
    races = []
    for index, race_id in enumerate(race_ids):
        runners = [f"dog-{index}-a", f"dog-{index}-b"]
        feature_observed_at = f"2026-07-{index + 1:02d}T09:00:00+10:00"
        scheduled_jump_at = f"2026-07-{index + 1:02d}T10:00:00+10:00"
        result_published_at = f"2026-07-{index + 1:02d}T10:31:00+10:00"
        result_observed_at = f"2026-07-{index + 1:02d}T10:32:00+10:00"
        source = canonical(
            {
                "schema_version": "race-evidence-v1",
                "normalization_version": "normalizer-v1",
                "race_id": race_id,
                "historical_capture": {
                    "source": "immutable-form-archive",
                    "source_record_id": f"form-{race_id}",
                    "observed_at": feature_observed_at,
                    "scheduled_jump_at": scheduled_jump_at,
                    "identity_authority": "source-native",
                    "reconstructed": False,
                },
                "fields": {
                    "runner_set": runners,
                    "runner_identity": {runner: "authoritative" for runner in runners},
                    "runner_features": {
                        runners[0]: {"speed": 8 + index, "days_since_run": {"missing": True}},
                        runners[1]: {"speed": 5 + index, "days_since_run": 3},
                    },
                },
            }
        )
        result = canonical(
            {
                "schema_version": "official-historical-result-v1",
                "race_id": race_id,
                "official": True,
                "order": list(reversed(runners)),
                "published_at": result_published_at,
                "exclusions": [],
                "provenance": {
                    "source": "official-results-archive",
                    "source_record_id": f"result-{race_id}",
                    "observed_at": result_observed_at,
                    "identity_authority": "source-native",
                    "reconstructed": False,
                },
            }
        )
        matrix = canonical(
            {
                "runner_ids": runners,
                "columns": ["speed", "days_since_run"],
                "rows": [[8.0 + index, 7.0], [5.0 + index, 3.0]],
            }
        )
        training_example_id = f"historical-{race_id}"
        artifact = canonical(
            {
                "schema_version": "historical-training-example-v1",
                "origin": origin,
                "forward_sealed": False,
                "promotion_evidence_eligible": False,
                "training_example_id": training_example_id,
                "race_id": race_id,
                "racing_date": f"2026-07-{index + 1:02d}",
                "source_checksum": checksum(source),
                "official_result_checksum": checksum(result),
                "feature_matrix_checksum": checksum(matrix),
                "runner_ids": runners,
                "official_order": list(reversed(runners)),
                "feature_observed_at": feature_observed_at,
                "scheduled_jump_at": scheduled_jump_at,
                "result_published_at": result_published_at,
                "result_observed_at": result_observed_at,
            }
        )
        for content in (source, result, matrix, artifact):
            artifacts[checksum(content)] = content
        races.append(
            {
                "training_example_id": training_example_id,
                "race_id": race_id,
                "racing_date": f"2026-07-{index + 1:02d}",
                "source_checksum": checksum(source),
                "official_result_checksum": checksum(result),
                "feature_matrix_checksum": checksum(matrix),
                "artifact_checksum": checksum(artifact),
                "runner_ids": runners,
                "feature_observed_at": feature_observed_at,
                "scheduled_jump_at": scheduled_jump_at,
                "result_published_at": result_published_at,
                "result_observed_at": result_observed_at,
            }
        )
    manifest = {
        "schema_version": "historical-source-manifest-v1",
        "corpus_origin": origin,
        "target_bundle_id": "bundle-native-phase7-1",
        "feature_schema_checksum": checksum(schema),
        "missingness_policy_checksum": checksum(missingness),
        "races": races,
    }
    envelope = {
        "schema_version": "historical-source-package-v1",
        "manifest_checksum": normalized_manifest_checksum(manifest),
        "manifest": manifest,
    }
    return envelope, artifacts


def replace_artifact(envelope, artifacts, race_index, field, mutate):
    old_checksum = envelope["manifest"]["races"][race_index][field]
    document = json.loads(artifacts.pop(old_checksum))
    mutate(document)
    content = canonical(document)
    new_checksum = checksum(content)
    artifacts[new_checksum] = content
    envelope["manifest"]["races"][race_index][field] = new_checksum
    envelope["manifest_checksum"] = normalized_manifest_checksum(envelope["manifest"])


def test_synthetic_package_validates_only_and_reordering_is_byte_identical():
    envelope, artifacts = source_package()
    first = admit_historical_source(canonical(envelope), artifacts=artifacts)
    reordered = copy.deepcopy(envelope)
    reordered["manifest"]["races"].reverse()
    second = admit_historical_source(
        canonical(reordered), artifacts=dict(reversed(artifacts.items()))
    )
    assert first == second
    admitted = json.loads(first)
    assert admitted["admission_decision"] == "VALIDATION_ONLY"
    assert admitted["production_readiness"] is False
    assert admitted["forward_sealed"] is False
    assert admitted["race_ids"] == ["race-a", "race-b"]
    assert admitted["races"][0]["runner_ids"] == ["dog-0-a", "dog-0-b"]


def test_legacy_origin_is_truthful_and_never_forward_sealed():
    envelope, artifacts = source_package(
        origin="legacy-historical-bootstrap-v1", race_ids=("race-a",)
    )
    admitted = json.loads(admit_historical_source(canonical(envelope), artifacts=artifacts))
    assert admitted["corpus_origin"] == "legacy-historical-bootstrap-v1"
    assert admitted["admission_decision"] == "TRAINING_ADMISSIBLE"
    assert admitted["forward_sealed"] is False
    assert admitted["promotion_evidence_eligible"] is False
    assert admitted["production_readiness"] is False
    envelope["manifest"]["corpus_origin"] = "forward-sealed"
    envelope["manifest_checksum"] = normalized_manifest_checksum(envelope["manifest"])
    with pytest.raises(SourceAdmissionRejected, match="corpus origin"):
        admit_historical_source(canonical(envelope), artifacts=artifacts)


def test_identity_mismatch_and_duplicate_normalized_runner_fail_closed():
    envelope, artifacts = source_package(race_ids=("race-a",))
    mismatch = copy.deepcopy(envelope)
    mismatch["manifest"]["races"][0]["runner_ids"][0] = "dog-0-0"
    mismatch["manifest_checksum"] = normalized_manifest_checksum(mismatch["manifest"])
    with pytest.raises(SourceAdmissionRejected, match="runner identities disagree"):
        admit_historical_source(canonical(mismatch), artifacts=artifacts)

    duplicate, duplicate_artifacts = source_package(race_ids=("race-a",))
    replace_artifact(
        duplicate,
        duplicate_artifacts,
        0,
        "source_checksum",
        lambda value: (
            value["fields"].__setitem__("runner_set", ["Dog A", "dog a"]),
            value["fields"].__setitem__(
                "runner_identity", {"Dog A": "authoritative", "dog a": "authoritative"}
            ),
            value["fields"].__setitem__(
                "runner_features",
                {
                    "Dog A": {"speed": 8, "days_since_run": {"missing": True}},
                    "dog a": {"speed": 5, "days_since_run": 3},
                },
            ),
        ),
    )
    duplicate["manifest"]["races"][0]["runner_ids"] = ["Dog A", "dog a"]
    duplicate["manifest_checksum"] = normalized_manifest_checksum(duplicate["manifest"])
    with pytest.raises(SourceAdmissionRejected, match="normalized runner identity"):
        admit_historical_source(canonical(duplicate), artifacts=duplicate_artifacts)


def test_result_leakage_and_post_result_feature_timestamp_fail_closed():
    envelope, artifacts = source_package(race_ids=("race-a",))
    replace_artifact(
        envelope,
        artifacts,
        0,
        "source_checksum",
        lambda value: value["fields"]["runner_features"]["dog-0-a"].__setitem__("result_order", 1),
    )
    with pytest.raises(SourceAdmissionRejected, match="post-result feature"):
        admit_historical_source(canonical(envelope), artifacts=artifacts)

    late, late_artifacts = source_package(race_ids=("race-a",))
    late["manifest"]["races"][0]["feature_observed_at"] = "2026-07-01T10:32:00+10:00"
    late["manifest_checksum"] = normalized_manifest_checksum(late["manifest"])
    with pytest.raises(SourceAdmissionRejected, match="temporal order"):
        admit_historical_source(canonical(late), artifacts=late_artifacts)


def test_missingness_policy_must_exactly_cover_optional_features():
    envelope, artifacts = source_package(race_ids=("race-a",))
    old_checksum = envelope["manifest"]["missingness_policy_checksum"]
    policy = json.loads(artifacts.pop(old_checksum))
    policy["imputation"] = {}
    content = canonical(policy)
    new_checksum = checksum(content)
    artifacts[new_checksum] = content
    envelope["manifest"]["missingness_policy_checksum"] = new_checksum
    envelope["manifest_checksum"] = normalized_manifest_checksum(envelope["manifest"])
    with pytest.raises(SourceAdmissionRejected, match="imputation"):
        admit_historical_source(canonical(envelope), artifacts=artifacts)


@pytest.mark.parametrize(
    "field,value",
    [
        ("feature_observed_at", "2026-07-01T09:00:00"),
        ("scheduled_jump_at", "not-a-timestamp"),
        ("result_published_at", 7),
    ],
)
def test_malformed_or_naive_timestamps_fail_closed(field, value):
    envelope, artifacts = source_package(race_ids=("race-a",))
    envelope["manifest"]["races"][0][field] = value
    envelope["manifest_checksum"] = normalized_manifest_checksum(envelope["manifest"])
    with pytest.raises(SourceAdmissionRejected, match="timestamp"):
        admit_historical_source(canonical(envelope), artifacts=artifacts)


def test_unsupported_feature_width_and_unverifiable_hash_fail_closed():
    envelope, artifacts = source_package(race_ids=("race-a",))
    replace_artifact(
        envelope,
        artifacts,
        0,
        "feature_matrix_checksum",
        lambda value: value["rows"][0].pop(),
    )
    with pytest.raises(SourceAdmissionRejected, match="feature matrix"):
        admit_historical_source(canonical(envelope), artifacts=artifacts)

    bad_hash, bad_hash_artifacts = source_package(race_ids=("race-a",))
    bad_hash_artifacts[bad_hash["manifest"]["races"][0]["source_checksum"]] += b" "
    with pytest.raises(SourceAdmissionRejected, match="checksum"):
        admit_historical_source(canonical(bad_hash), artifacts=bad_hash_artifacts)


def test_manifest_hash_and_closed_artifact_inventory_are_required():
    envelope, artifacts = source_package(race_ids=("race-a",))
    envelope["manifest_checksum"] = "sha256:" + "0" * 64
    with pytest.raises(SourceAdmissionRejected, match="manifest checksum"):
        admit_historical_source(canonical(envelope), artifacts=artifacts)

    envelope, artifacts = source_package(race_ids=("race-a",))
    artifacts["sha256:" + "f" * 64] = b"undeclared"
    with pytest.raises(SourceAdmissionRejected, match="artifact inventory"):
        admit_historical_source(canonical(envelope), artifacts=artifacts)
