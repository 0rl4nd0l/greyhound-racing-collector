import copy
import hashlib
import json
from datetime import datetime

import pytest

from race_collection.domain import ArtifactChecksum
from race_collection.forward_sealed_corpus import (
    ForwardCorpusRejected,
    ForwardSealedCorpus,
    canonical_json,
)
from race_collection.source_admission import (
    SourceAdmissionRejected,
    admit_historical_source,
)


def _corpus(path):
    return ForwardSealedCorpus(
        path,
        clock=lambda: datetime.fromisoformat("2026-07-29T09:45:00+10:00"),
    )


def fixture(index=1):
    day = "2026-07-29"
    race_id = f"race-{index}"
    runners = [
        {"source_native_runner_id": f"dog-{index}-a", "name": f"Dog {index} A"},
        {"source_native_runner_id": f"dog-{index}-b", "name": f"Dog {index} B"},
    ]
    raw_source_bytes = f"<html>source-{index}</html>".encode()
    raw_source_checksum = "sha256:" + hashlib.sha256(raw_source_bytes).hexdigest()
    schema = canonical_json(
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
    missingness = canonical_json(
        {
            "bundle_id": "bundle-native-phase7-1",
            "feature_contract_version": "sealed-race-features-v1",
            "imputation": {"days_since_run": 7.0},
        }
    )
    fields = {
        "runner_set": [row["source_native_runner_id"] for row in runners],
        "runner_identity": {row["source_native_runner_id"]: "authoritative" for row in runners},
        "runner_features": {
            runners[0]["source_native_runner_id"]: {
                "speed": 8 + index,
                "days_since_run": {"missing": True},
            },
            runners[1]["source_native_runner_id"]: {
                "speed": 5 + index,
                "days_since_run": 3,
            },
        },
    }
    evidence = canonical_json(
        {
            "schema_version": "race-evidence-v1",
            "normalization_version": "normalizer-v1",
            "race_id": race_id,
            "fields": fields,
            "field_provenance": [
                {
                    "field": field,
                    "authority": "source_card",
                    "critical": field in {"runner_set", "runner_identity"},
                    "value": fields[field],
                    "source": "thedogs-race-card",
                    "artifact_checksum": raw_source_checksum,
                }
                for field in ("runner_set", "runner_identity", "runner_features")
            ],
            "freeze": {
                "at": f"{day}T09:30:00+10:00",
                "authority": "scheduled_minus_buffer",
                "odds_checksum": "sha256:" + "a" * 64,
            },
        }
    )
    prepjump = {
        "race_id": race_id,
        "racing_date": day,
        "raw_source_bytes": raw_source_bytes,
        "sealed_evidence_bytes": evidence,
        "feature_schema_bytes": schema,
        "missingness_policy_bytes": missingness,
        "source_name": "thedogs-race-card",
        "canonical_source_url": f"https://www.thedogs.com.au/racing/venue/{day}/{index}",
        "source_native_race_id": f"thedogs-{day}-{index}",
        "runners": runners,
        "meeting_metadata": {
            "source_native_meeting_id": f"meeting-{day}",
            "venue": "Sandown",
        },
        "race_metadata": {
            "race_number": index,
            "distance_metres": 515,
        },
        "source_observed_at": f"{day}T09:00:00+10:00",
        "feature_frozen_at": f"{day}T09:30:00+10:00",
        "scheduled_jump_at": f"{day}T10:00:00+10:00",
    }
    result = {
        "race_id": race_id,
        "raw_result_bytes": f"<html>official-result-{index}</html>".encode(),
        "source_name": "thedogs-official",
        "canonical_source_url": (f"https://www.thedogs.com.au/racing/venue/{day}/{index}/results"),
        "source_native_race_id": f"thedogs-{day}-{index}",
        "runners": runners,
        "official_order": [
            runners[1]["source_native_runner_id"],
            runners[0]["source_native_runner_id"],
        ],
        "result_observed_at": f"{day}T10:32:00+10:00",
        "result_published_at": f"{day}T10:31:00+10:00",
        "publication_timestamp_status": "source-declared",
    }
    return prepjump, result


def close(corpus, index=1):
    prepjump, result = fixture(index)
    corpus.capture_prejump(**prepjump)
    corpus.capture_result(**result)
    return corpus.build_package()


def test_synthetic_end_to_end_closes_and_passes_source_admission(tmp_path):
    corpus = _corpus(tmp_path)
    package = close(corpus)

    admitted = json.loads(
        admit_historical_source(package.package_bytes, artifacts=package.artifacts)
    )
    assert admitted["admission_decision"] == "TRAINING_ADMISSIBLE"
    assert admitted["corpus_origin"] == "forward-sealed-corpus-v1"
    assert admitted["forward_sealed"] is True
    assert admitted["promotion_evidence_eligible"] is False
    assert admitted["production_readiness"] is False
    assert corpus.status()["races"] == [
        {
            "race_id": "race-1",
            "state": "CLOSED",
            "result_observation_count": 0,
        }
    ]


def test_prepjump_stage_rejects_late_capture_and_result_leakage(tmp_path):
    prepjump, _ = fixture()
    prepjump["feature_frozen_at"] = prepjump["scheduled_jump_at"]
    with pytest.raises(ForwardCorpusRejected, match="pre-jump"):
        _corpus(tmp_path / "late").capture_prejump(**prepjump)

    leaking, _ = fixture()
    evidence = json.loads(leaking["sealed_evidence_bytes"])
    evidence["fields"]["runner_features"]["dog-1-a"]["finish_position"] = 1
    leaking["sealed_evidence_bytes"] = canonical_json(evidence)
    with pytest.raises(ForwardCorpusRejected, match="result-derived"):
        _corpus(tmp_path / "leak").capture_prejump(**leaking)


def test_first_receipt_must_be_published_before_jump_but_exact_retry_remains_idempotent(
    tmp_path,
):
    prepjump, _ = fixture()
    late = ForwardSealedCorpus(
        tmp_path / "late",
        clock=lambda: datetime.fromisoformat("2026-07-29T10:00:00+10:00"),
    )
    with pytest.raises(ForwardCorpusRejected, match="prospectively"):
        late.capture_prejump(**prepjump)
    assert late.status()["race_count"] == 0

    on_time = _corpus(tmp_path / "retry")
    receipt = on_time.capture_prejump(**prepjump)
    after_jump = ForwardSealedCorpus(
        tmp_path / "retry",
        clock=lambda: datetime.fromisoformat("2026-07-29T10:30:00+10:00"),
    )
    assert after_jump.capture_prejump(**prepjump) == receipt


def test_prepjump_requires_exact_bundle_and_source_binding_contracts(tmp_path):
    prepjump, _ = fixture()
    policy = json.loads(prepjump["missingness_policy_bytes"])
    policy["bundle_id"] = "other-bundle"
    prepjump["missingness_policy_bytes"] = canonical_json(policy)
    with pytest.raises(ForwardCorpusRejected, match="feature schema"):
        _corpus(tmp_path / "policy").capture_prejump(**prepjump)

    unbound, _ = fixture()
    evidence = json.loads(unbound["sealed_evidence_bytes"])
    evidence["field_provenance"][0]["artifact_checksum"] = "sha256:" + "b" * 64
    unbound["sealed_evidence_bytes"] = canonical_json(evidence)
    with pytest.raises(ForwardCorpusRejected, match="not bound"):
        _corpus(tmp_path / "binding").capture_prejump(**unbound)

    wrong_value, _ = fixture()
    evidence = json.loads(wrong_value["sealed_evidence_bytes"])
    evidence["field_provenance"][0]["value"] = []
    wrong_value["sealed_evidence_bytes"] = canonical_json(evidence)
    with pytest.raises(ForwardCorpusRejected, match="not bound"):
        _corpus(tmp_path / "value").capture_prejump(**wrong_value)


def test_result_requires_prejump_and_strict_after_jump_order(tmp_path):
    prepjump, result = fixture()
    corpus = _corpus(tmp_path)
    with pytest.raises(ForwardCorpusRejected, match="pre-jump evidence"):
        corpus.capture_result(**result)
    corpus.capture_prejump(**prepjump)
    result["result_observed_at"] = prepjump["scheduled_jump_at"]
    with pytest.raises(ForwardCorpusRejected, match="after jump"):
        corpus.capture_result(**result)


@pytest.mark.parametrize(
    "field,value",
    [
        ("result_observed_at", "2026-07-01T10:32:00"),
        ("result_published_at", "unknown"),
        ("result_published_at", 7),
    ],
)
def test_unknown_or_naive_result_timestamps_fail_closed(tmp_path, field, value):
    prepjump, result = fixture()
    corpus = _corpus(tmp_path)
    corpus.capture_prejump(**prepjump)
    result[field] = value
    with pytest.raises(ForwardCorpusRejected, match="timestamp"):
        corpus.capture_result(**result)


def test_identity_runner_and_raw_hash_drift_fail_closed(tmp_path):
    prepjump, result = fixture()
    corpus = _corpus(tmp_path)
    corpus.capture_prejump(**prepjump)

    mismatch = copy.deepcopy(result)
    mismatch["source_native_race_id"] = "other-race"
    with pytest.raises(ForwardCorpusRejected, match="race identity mismatch"):
        corpus.capture_result(**mismatch)

    drift = copy.deepcopy(result)
    drift["runners"][0]["source_native_runner_id"] = "other-runner"
    with pytest.raises(ForwardCorpusRejected, match="runner drift"):
        corpus.capture_result(**drift)

    name_drift = copy.deepcopy(result)
    name_drift["runners"][0]["name"] = "Different Dog"
    with pytest.raises(ForwardCorpusRejected, match="name drift"):
        corpus.capture_result(**name_drift)

    blocked = copy.deepcopy(result)
    blocked["result_published_at"] = None
    blocked["publication_timestamp_status"] = "not-exposed-by-source"
    assert (
        corpus.capture_result(**blocked)["closure_decision"]
        == "BLOCKED_RESULT_PUBLICATION_TIMESTAMP"
    )
    changed = copy.deepcopy(result)
    changed["raw_result_bytes"] = b"different official bytes"
    with pytest.raises(ForwardCorpusRejected, match="hash drift"):
        corpus.capture_result(**changed)

    provenance_drift = copy.deepcopy(result)
    provenance_drift["canonical_source_url"] += "?different=true"
    with pytest.raises(ForwardCorpusRejected, match="provenance drift"):
        corpus.capture_result(**provenance_drift)


def test_thedogs_missing_publication_timestamp_is_retained_but_never_closed(tmp_path):
    prepjump, result = fixture()
    corpus = _corpus(tmp_path)
    corpus.capture_prejump(**prepjump)
    result["result_published_at"] = None
    result["publication_timestamp_status"] = "not-exposed-by-source"

    blocked = corpus.capture_result(**result)

    assert blocked["result_observed_at"].endswith("+10:00")
    assert blocked["result_published_at"] is None
    assert (
        corpus.artifacts.read(ArtifactChecksum(blocked["raw_result_checksum"]))
        == result["raw_result_bytes"]
    )
    assert corpus.status()["races"][0]["state"] == "BLOCKED_RESULT_PUBLICATION_TIMESTAMP"
    with pytest.raises(ForwardCorpusRejected, match="no closed races"):
        corpus.build_package()


def test_result_publication_timestamp_must_be_source_declared_and_after_jump(tmp_path):
    prepjump, result = fixture()
    corpus = _corpus(tmp_path)
    corpus.capture_prejump(**prepjump)

    result["result_published_at"] = "2026-07-01T09:59:59+10:00"
    with pytest.raises(ForwardCorpusRejected, match="temporal order"):
        corpus.capture_result(**result)

    result["result_published_at"] = None
    result["publication_timestamp_status"] = "unknown"
    with pytest.raises(ForwardCorpusRejected, match="unsupported"):
        corpus.capture_result(**result)


def test_blocked_result_closes_only_after_same_bytes_gain_source_timestamp(tmp_path):
    prepjump, result = fixture()
    corpus = _corpus(tmp_path)
    corpus.capture_prejump(**prepjump)
    blocked = copy.deepcopy(result)
    blocked["result_published_at"] = None
    blocked["publication_timestamp_status"] = "not-exposed-by-source"
    blocked_receipt = corpus.capture_result(**blocked)

    closure = corpus.capture_result(**result)

    assert closure["schema_version"] == "forward-race-closure-v1"
    assert corpus.status()["races"][0]["state"] == "CLOSED"
    package_race = json.loads(corpus.build_package().package_bytes)["manifest"]["races"][0]
    assert package_race["raw_result_checksum"] == blocked_receipt["raw_result_checksum"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_observed_at", "2026-07-01T09:00:00"),
        ("feature_frozen_at", "unknown"),
        ("scheduled_jump_at", 7),
    ],
)
def test_unknown_or_naive_prejump_timestamps_fail_closed(tmp_path, field, value):
    prepjump, _ = fixture()
    prepjump[field] = value
    with pytest.raises(ForwardCorpusRejected, match="timestamp"):
        _corpus(tmp_path).capture_prejump(**prepjump)


def test_exact_replay_is_idempotent_and_conflicting_duplicate_closure_fails(tmp_path):
    prepjump, result = fixture()
    corpus = _corpus(tmp_path)
    first_prejump = corpus.capture_prejump(**prepjump)
    assert corpus.capture_prejump(**prepjump) == first_prejump
    first_closure = corpus.capture_result(**result)
    assert corpus.capture_result(**result) == first_closure

    conflict = copy.deepcopy(result)
    conflict["official_order"].reverse()
    with pytest.raises(ForwardCorpusRejected, match="append-only receipt conflict"):
        corpus.capture_result(**conflict)


def test_closed_result_rejects_raw_hash_drift_and_later_blocked_observation(tmp_path):
    prepjump, result = fixture()
    corpus = _corpus(tmp_path)
    corpus.capture_prejump(**prepjump)
    corpus.capture_result(**result)

    drift = copy.deepcopy(result)
    drift["raw_result_bytes"] = b"different official bytes"
    with pytest.raises(ForwardCorpusRejected, match="hash drift"):
        corpus.capture_result(**drift)

    blocked = copy.deepcopy(result)
    blocked["result_published_at"] = None
    blocked["publication_timestamp_status"] = "not-exposed-by-source"
    with pytest.raises(ForwardCorpusRejected, match="closed result"):
        corpus.capture_result(**blocked)


def test_partial_write_recovery_reuses_objects_and_completes_receipt(tmp_path, monkeypatch):
    prepjump, _ = fixture()
    corpus = _corpus(tmp_path)
    original = ForwardSealedCorpus._publish_once
    failed = False

    def fail_receipt(path, content):
        nonlocal failed
        if path.name == "prejump.json" and not failed:
            failed = True
            raise OSError("simulated receipt publication crash")
        return original(path, content)

    monkeypatch.setattr(ForwardSealedCorpus, "_publish_once", staticmethod(fail_receipt))
    with pytest.raises(OSError, match="simulated"):
        corpus.capture_prejump(**prepjump)
    monkeypatch.setattr(ForwardSealedCorpus, "_publish_once", staticmethod(original))

    receipt = corpus.capture_prejump(**prepjump)

    assert corpus.status()["races"][0]["state"] == "PREJUMP_CAPTURED"
    assert corpus.artifacts.read(ArtifactChecksum(receipt["raw_source_checksum"]))


def test_result_receipt_crash_recovers_to_same_closure(tmp_path, monkeypatch):
    prepjump, result = fixture()
    corpus = _corpus(tmp_path)
    corpus.capture_prejump(**prepjump)
    original = ForwardSealedCorpus._close
    failed = False

    def fail_close(self, pre, result_receipt):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("simulated close crash")
        return original(self, pre, result_receipt)

    monkeypatch.setattr(ForwardSealedCorpus, "_close", fail_close)
    with pytest.raises(OSError, match="simulated"):
        corpus.capture_result(**result)
    monkeypatch.setattr(ForwardSealedCorpus, "_close", original)

    closure = corpus.capture_result(**result)

    assert closure["schema_version"] == "forward-race-closure-v1"
    assert corpus.status()["races"][0]["state"] == "CLOSED"


def test_race_and_input_reordering_produce_identical_package_bytes(tmp_path):
    first = _corpus(tmp_path / "first")
    second = _corpus(tmp_path / "second")
    for index in (1, 2):
        prepjump, result = fixture(index)
        first.capture_prejump(**prepjump)
        first.capture_result(**result)
    for index in (2, 1):
        prepjump, result = fixture(index)
        prepjump["runners"].reverse()
        result["runners"].reverse()
        second.capture_prejump(**prepjump)
        second.capture_result(**result)

    assert first.build_package().package_bytes == second.build_package().package_bytes


def test_shared_meeting_source_and_result_bytes_remain_content_addressed(tmp_path):
    first_prejump, first_result = fixture(1)
    second_prejump, second_result = fixture(2)
    second_prejump["raw_source_bytes"] = first_prejump["raw_source_bytes"]
    shared_source_checksum = (
        "sha256:" + hashlib.sha256(first_prejump["raw_source_bytes"]).hexdigest()
    )
    second_evidence = json.loads(second_prejump["sealed_evidence_bytes"])
    for item in second_evidence["field_provenance"]:
        item["artifact_checksum"] = shared_source_checksum
    second_prejump["sealed_evidence_bytes"] = canonical_json(second_evidence)
    second_prejump["canonical_source_url"] = first_prejump["canonical_source_url"]
    second_result["raw_result_bytes"] = first_result["raw_result_bytes"]
    second_result["canonical_source_url"] = first_result["canonical_source_url"]

    corpus = _corpus(tmp_path)
    for prepjump, result in (
        (first_prejump, first_result),
        (second_prejump, second_result),
    ):
        corpus.capture_prejump(**prepjump)
        corpus.capture_result(**result)

    package = corpus.build_package()
    admitted = json.loads(
        admit_historical_source(package.package_bytes, artifacts=package.artifacts)
    )
    assert admitted["race_ids"] == ["race-1", "race-2"]


def test_source_admission_detects_missing_raw_bytes_and_hash_drift(tmp_path):
    package = close(_corpus(tmp_path))
    missing = dict(package.artifacts)
    raw_checksum = json.loads(package.package_bytes)["manifest"]["races"][0]["raw_source_checksum"]
    missing.pop(raw_checksum)
    with pytest.raises(SourceAdmissionRejected, match="inventory"):
        admit_historical_source(package.package_bytes, artifacts=missing)

    drift = dict(package.artifacts)
    drift[raw_checksum] += b"changed"
    with pytest.raises(SourceAdmissionRejected, match="checksum"):
        admit_historical_source(package.package_bytes, artifacts=drift)


def test_status_detects_stored_artifact_hash_drift(tmp_path):
    corpus = _corpus(tmp_path)
    prepjump, _ = fixture()
    receipt = corpus.capture_prejump(**prepjump)
    raw_checksum = ArtifactChecksum(receipt["raw_source_checksum"])
    corpus.artifacts.path_for(raw_checksum).write_bytes(b"drift")

    with pytest.raises(ForwardCorpusRejected, match="hash drift"):
        corpus.status()
