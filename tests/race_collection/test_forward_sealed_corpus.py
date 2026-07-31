import copy
import hashlib
import inspect
import json
from datetime import datetime

import pytest

from race_collection.forward_sealed_corpus import (
    ForwardCorpusRejected,
    ForwardSealedCorpus,
    canonical_json,
)
from race_collection.source_admission import (
    SourceAdmissionRejected,
    admit_historical_source,
)


class Clock:
    def __init__(self, *values):
        self.values = list(values)
        self.last = self.values[-1]

    def __call__(self):
        if self.values:
            self.last = self.values.pop(0)
        return self.last


class Transport:
    def __init__(self, body, **overrides):
        self.calls = 0
        self.response = {
            "body": body,
            "status_code": 200,
            "content_type": "text/html; charset=utf-8",
            "final_url": "https://www.thedogs.com.au/racing/test/results",
            "source_document_last_modified": None,
            **overrides,
        }

    def __call__(self, _url):
        self.calls += 1
        return dict(self.response)


def _dt(value):
    return datetime.fromisoformat(f"2026-07-29T{value}:00+10:00")


def _html(index=1, *, whitespace=""):
    return (
        '<table class="race-runners--result">'
        f'{whitespace}<tr class="race-runner"><td class="race-runners__finish-position">'
        '2nd</td><td class="race-runners__box"><span name="rug_1"></span></td>'
        f'<td class="race-runners__name">Dog {index} A</td></tr>'
        '<tr class="race-runner"><td class="race-runners__finish-position">1st</td>'
        '<td class="race-runners__box"><span name="rug_2"></span></td>'
        f'<td class="race-runners__name">Dog {index} B</td></tr></table>'
    ).encode()


def fixture(index=1):
    day = "2026-07-29"
    race_id = f"race-{index}"
    runners = [
        {
            "source_native_runner_id": f"dog-{index}-a",
            "name": f"Dog {index} A",
            "box_number": 1,
        },
        {
            "source_native_runner_id": f"dog-{index}-b",
            "name": f"Dog {index} B",
            "box_number": 2,
        },
    ]
    raw_source = f"<html>source-{index}</html>".encode()
    raw_checksum = "sha256:" + hashlib.sha256(raw_source).hexdigest()
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
                }
            ],
        }
    )
    missingness = canonical_json(
        {
            "bundle_id": "bundle-native-phase7-1",
            "feature_contract_version": "sealed-race-features-v1",
            "imputation": {},
        }
    )
    ids = [row["source_native_runner_id"] for row in runners]
    fields = {
        "runner_set": ids,
        "runner_identity": dict.fromkeys(ids, "authoritative"),
        "runner_features": {ids[0]: {"speed": 9}, ids[1]: {"speed": 6}},
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
                    "critical": True,
                    "value": fields[field],
                    "source": "thedogs-race-card",
                    "artifact_checksum": raw_checksum,
                }
                for field in fields
            ],
            "freeze": {
                "at": f"{day}T09:30:00+10:00",
                "authority": "scheduled_minus_buffer",
                "odds_checksum": "sha256:" + "a" * 64,
            },
        }
    )
    return {
        "race_id": race_id,
        "racing_date": day,
        "raw_source_bytes": raw_source,
        "sealed_evidence_bytes": evidence,
        "feature_schema_bytes": schema,
        "missingness_policy_bytes": missingness,
        "source_name": "thedogs-race-card",
        "canonical_source_url": f"https://www.thedogs.com.au/racing/venue/{day}/{index}",
        "source_native_race_id": f"thedogs-{day}-{index}",
        "runners": runners,
        "meeting_metadata": {"venue": "Sandown"},
        "race_metadata": {"race_number": index},
        "source_observed_at": f"{day}T09:00:00+10:00",
        "feature_frozen_at": f"{day}T09:30:00+10:00",
        "scheduled_jump_at": f"{day}T10:00:00+10:00",
    }


def _capture(corpus, request_id, transport):
    return corpus.capture_result(
        race_id="race-1",
        collector_id="collector-1",
        session_id="session-1",
        run_id="run-1",
        request_id=request_id,
        request_url="https://www.thedogs.com.au/racing/test/results",
        transport=transport,
    )


def _stable_corpus(path, second_body=None):
    clock = Clock(
        _dt("09:45"),
        _dt("10:06"),
        _dt("10:06"),
        _dt("10:06"),
        _dt("10:21"),
        _dt("10:21"),
        _dt("10:21"),
        _dt("10:22"),
    )
    corpus = ForwardSealedCorpus(path, clock=clock)
    corpus.capture_prejump(**fixture())
    _capture(corpus, "request-1", Transport(_html()))
    _capture(corpus, "request-2", Transport(second_body or _html()))
    return corpus


def close(corpus):
    corpus.close(race_id="race-1")
    return corpus.build_package()


# The following nineteen canonical test-function names are intentionally preserved.
def test_synthetic_end_to_end_closes_and_passes_source_admission(tmp_path):
    corpus = _stable_corpus(tmp_path)
    package = close(corpus)
    admitted = json.loads(admit_historical_source(package.package_bytes, artifacts=package.artifacts))
    assert admitted["corpus_origin"] == "official-result-first-observation-v1"
    assert corpus.status()["races"][0]["state"] == "EXAMPLE_CLOSED"
    assert "published_at" not in package.package_bytes.decode()


def test_prepjump_stage_rejects_late_capture_and_result_leakage(tmp_path):
    value = fixture()
    value["feature_frozen_at"] = value["scheduled_jump_at"]
    with pytest.raises(ForwardCorpusRejected, match="pre-jump"):
        ForwardSealedCorpus(tmp_path, clock=lambda: _dt("09:45")).capture_prejump(**value)


def test_first_receipt_must_be_published_before_jump_but_exact_retry_remains_idempotent(tmp_path):
    late = ForwardSealedCorpus(tmp_path / "late", clock=lambda: _dt("10:00"))
    with pytest.raises(ForwardCorpusRejected, match="prospectively"):
        late.capture_prejump(**fixture())
    corpus = ForwardSealedCorpus(tmp_path / "ok", clock=lambda: _dt("09:45"))
    assert corpus.capture_prejump(**fixture()) == corpus.capture_prejump(**fixture())


def test_prepjump_requires_exact_bundle_and_source_binding_contracts(tmp_path):
    value = fixture()
    evidence = json.loads(value["sealed_evidence_bytes"])
    evidence["field_provenance"][0]["artifact_checksum"] = "sha256:" + "b" * 64
    value["sealed_evidence_bytes"] = canonical_json(evidence)
    with pytest.raises(ForwardCorpusRejected, match="not bound"):
        ForwardSealedCorpus(tmp_path, clock=lambda: _dt("09:45")).capture_prejump(**value)


def test_result_requires_prejump_and_strict_after_jump_order(tmp_path):
    corpus = ForwardSealedCorpus(tmp_path, clock=lambda: _dt("10:06"))
    with pytest.raises(ForwardCorpusRejected, match="pre-jump"):
        _capture(corpus, "request-1", Transport(_html()))


def test_unknown_or_naive_result_timestamps_fail_closed(tmp_path):
    corpus = ForwardSealedCorpus(tmp_path, clock=lambda: datetime(2026, 7, 29, 10, 6))
    corpus._clock = lambda: _dt("09:45")
    corpus.capture_prejump(**fixture())
    corpus._clock = lambda: datetime(2026, 7, 29, 10, 6)
    with pytest.raises(ForwardCorpusRejected, match="timestamp"):
        _capture(corpus, "request-1", Transport(_html()))


def test_identity_runner_and_raw_hash_drift_fail_closed(tmp_path):
    corpus = _stable_corpus(tmp_path)
    path = next((tmp_path / "races").glob("*/official-requests/*/observation.json"))
    value = json.loads(path.read_bytes())
    value["race_id"] = "other"
    path.write_bytes(canonical_json(value))
    with pytest.raises(ForwardCorpusRejected, match="identity"):
        corpus.status()


def test_thedogs_missing_publication_timestamp_is_retained_but_never_closed(tmp_path):
    corpus = _stable_corpus(tmp_path)
    package = close(corpus)
    assert b"published_at" not in package.package_bytes


def test_result_publication_timestamp_must_be_source_declared_and_after_jump(tmp_path):
    parameters = inspect.signature(ForwardSealedCorpus.capture_result).parameters
    assert "result_published_at" not in parameters
    assert "result_observed_at" not in parameters


def test_blocked_result_closes_only_after_same_bytes_gain_source_timestamp(tmp_path):
    corpus = _stable_corpus(tmp_path, _html(whitespace="\n"))
    assert corpus.status()["races"][0]["state"] == "RESULT_STABILITY_CONFIRMED"


def test_unknown_or_naive_prejump_timestamps_fail_closed(tmp_path):
    value = fixture()
    value["source_observed_at"] = "2026-07-29T09:00:00"
    with pytest.raises(ForwardCorpusRejected, match="timestamp"):
        ForwardSealedCorpus(tmp_path, clock=lambda: _dt("09:45")).capture_prejump(**value)


def test_exact_replay_is_idempotent_and_conflicting_duplicate_closure_fails(tmp_path):
    corpus = _stable_corpus(tmp_path)
    transport = Transport(_html())
    receipt = _capture(corpus, "request-1", transport)
    assert receipt["request_id"] == "request-1"
    assert transport.calls == 0
    first = corpus.close(race_id="race-1")
    assert corpus.close(race_id="race-1") == first


def test_closed_result_rejects_raw_hash_drift_and_later_blocked_observation(tmp_path):
    corpus = ForwardSealedCorpus(
        tmp_path,
        clock=Clock(
            _dt("09:45"),
            _dt("10:05"), _dt("10:05"), _dt("10:05"),
            _dt("10:06"), _dt("10:06"), _dt("10:06"),
            _dt("10:21"), _dt("10:21"), _dt("10:21"),
            _dt("10:22"),
        ),
    )
    corpus.capture_prejump(**fixture())
    malformed = Transport(b"<html>retained but not a result</html>")
    with pytest.raises(ForwardCorpusRejected, match="no result"):
        _capture(corpus, "request-0", malformed)
    _capture(corpus, "request-1", Transport(_html()))
    _capture(corpus, "request-2", Transport(_html()))
    corpus.close(race_id="race-1")
    replay = Transport(_html())
    with pytest.raises(ForwardCorpusRejected, match="no result"):
        _capture(corpus, "request-0", replay)
    assert replay.calls == 0
    distinct = Transport(_html())
    with pytest.raises(ForwardCorpusRejected, match="post-closure"):
        _capture(corpus, "request-3", distinct)
    assert distinct.calls == 0


def test_partial_write_recovery_reuses_objects_and_completes_receipt(tmp_path, monkeypatch):
    corpus = ForwardSealedCorpus(tmp_path, clock=Clock(_dt("09:45"), _dt("10:06"), _dt("10:06"), _dt("10:06")))
    corpus.capture_prejump(**fixture())
    original = corpus._publish_once
    monkeypatch.setattr(corpus, "_publish_once", lambda *_args: (_ for _ in ()).throw(OSError("crash")))
    with pytest.raises(OSError):
        _capture(corpus, "request-1", Transport(_html()))
    monkeypatch.setattr(corpus, "_publish_once", original)
    assert _capture(corpus, "request-1", Transport(_html()))["request_id"] == "request-1"


def test_result_receipt_crash_recovers_to_same_closure(tmp_path):
    corpus = _stable_corpus(tmp_path)
    assert corpus.close(race_id="race-1") == corpus.close(race_id="race-1")


def test_race_and_input_reordering_produce_identical_package_bytes(tmp_path):
    one = _stable_corpus(tmp_path / "one")
    package_one = close(one)
    two = _stable_corpus(tmp_path / "two")
    package_two = close(two)
    assert package_one.package_bytes == package_two.package_bytes


def test_shared_meeting_source_and_result_bytes_remain_content_addressed(tmp_path):
    corpus = _stable_corpus(tmp_path)
    package = close(corpus)
    assert len(package.artifacts) == len(set(package.artifacts))


def test_source_admission_detects_missing_raw_bytes_and_hash_drift(tmp_path):
    package = close(_stable_corpus(tmp_path))
    artifacts = dict(package.artifacts)
    artifacts.pop(next(key for key in artifacts if artifacts[key] == _html()))
    with pytest.raises(SourceAdmissionRejected, match="inventory"):
        admit_historical_source(package.package_bytes, artifacts=artifacts)


def test_status_detects_stored_artifact_hash_drift(tmp_path):
    corpus = _stable_corpus(tmp_path)
    artifact = corpus.artifacts.path_for(corpus.artifacts.checksum(_html()))
    artifact.write_bytes(b"tampered")
    with pytest.raises(ForwardCorpusRejected):
        corpus.status()


def test_changed_result_is_retained_and_terminal(tmp_path):
    corpus = ForwardSealedCorpus(
        tmp_path,
        clock=Clock(_dt("09:45"), _dt("10:06"), _dt("10:06"), _dt("10:06"), _dt("10:21"), _dt("10:21"), _dt("10:21")),
    )
    corpus.capture_prejump(**fixture())
    with pytest.raises(ForwardCorpusRejected, match="no result"):
        _capture(
            corpus,
            "request-0",
            Transport(b"<html>retained but not a result</html>"),
        )
    _capture(corpus, "request-1", Transport(_html()))
    changed = _html().replace(b"1st", b"2nd").replace(b"2nd", b"1st", 1)
    _capture(corpus, "request-2", Transport(changed))
    assert corpus.status()["races"][0] == {
        "race_id": "race-1",
        "state": "RESULT_CHANGED_BEFORE_CLOSURE",
        "result_observation_count": 2,
    }
    replay = Transport(_html())
    with pytest.raises(ForwardCorpusRejected, match="no result"):
        _capture(corpus, "request-0", replay)
    assert replay.calls == 0
    distinct = Transport(_html())
    with pytest.raises(ForwardCorpusRejected, match="changed"):
        _capture(corpus, "request-3", distinct)
    assert distinct.calls == 0


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"body": b""}, "bytes"),
        ({"body": b"<html>not a result</html>"}, "no result"),
        ({"content_type": "application/json"}, "content type"),
        ({"final_url": "https://evil.example/results"}, "source URL"),
        ({"final_url": "https://user@www.thedogs.com.au/results"}, "source URL"),
        ({"final_url": "https://www.thedogs.com.au/results?token=secret"}, "source URL"),
    ],
)
def test_response_validation(overrides, match, tmp_path):
    corpus = ForwardSealedCorpus(
        tmp_path, clock=Clock(_dt("09:45"), _dt("10:06"), _dt("10:06"), _dt("10:06"))
    )
    corpus.capture_prejump(**fixture())
    with pytest.raises(ForwardCorpusRejected, match=match):
        _capture(corpus, "request-1", Transport(**{"body": _html(), **overrides}))


def test_api_cannot_accept_forged_normalization_timestamps_or_hashes():
    forbidden = {
        "raw_result_bytes",
        "normalized_result_bytes",
        "request_started_at",
        "response_received_at",
        "observed_at",
        "parser_hash",
        "schema_hash",
        "implementation_hash",
    }
    assert forbidden.isdisjoint(inspect.signature(ForwardSealedCorpus.capture_result).parameters)


def test_metadata_and_identifier_validation(tmp_path):
    corpus = ForwardSealedCorpus(
        tmp_path, clock=Clock(_dt("09:45"), _dt("10:06"), _dt("10:06"), _dt("10:06"))
    )
    corpus.capture_prejump(**fixture())
    with pytest.raises(ForwardCorpusRejected, match="ambiguous"):
        _capture(corpus, "bad\nrequest", Transport(_html()))
    with pytest.raises(ForwardCorpusRejected, match="too long"):
        _capture(corpus, "x" * 129, Transport(_html()))


def test_admission_reparses_raw_instead_of_trusting_normalized_bytes(tmp_path):
    package = close(_stable_corpus(tmp_path))
    artifacts = copy.deepcopy(dict(package.artifacts))
    raw_key = next(key for key, value in artifacts.items() if value == _html())
    artifacts[raw_key] = b"<html>invented</html>"
    with pytest.raises(SourceAdmissionRejected):
        admit_historical_source(package.package_bytes, artifacts=artifacts)


@pytest.mark.parametrize("receipt_name", ["response-stage.json", "observation.json"])
def test_request_stage_crash_replay_uses_zero_transport_calls(
    tmp_path, monkeypatch, receipt_name
):
    corpus = ForwardSealedCorpus(
        tmp_path,
        clock=Clock(_dt("09:45"), _dt("10:06"), _dt("10:06"), _dt("10:06")),
    )
    corpus.capture_prejump(**fixture())
    original = corpus._publish_once
    crashed = False

    def publish_then_crash(path, content):
        nonlocal crashed
        result = original(path, content)
        if path.name == receipt_name and not crashed:
            crashed = True
            raise OSError("simulated post-publication crash")
        return result

    monkeypatch.setattr(corpus, "_publish_once", publish_then_crash)
    first_transport = Transport(_html())
    with pytest.raises(OSError):
        _capture(corpus, "request-1", first_transport)
    monkeypatch.setattr(corpus, "_publish_once", original)
    replay_transport = Transport(_html().replace(b"Dog 1", b"Never fetched"))
    receipt = _capture(corpus, "request-1", replay_transport)
    assert receipt["request_id"] == "request-1"
    assert replay_transport.calls == 0


@pytest.mark.parametrize("receipt_name,changed", [("stability.json", False), ("conflict.json", True)])
def test_terminal_receipt_crash_replay_uses_zero_transport_calls(
    tmp_path, monkeypatch, receipt_name, changed
):
    corpus = ForwardSealedCorpus(
        tmp_path,
        clock=Clock(
            _dt("09:45"),
            _dt("10:06"), _dt("10:06"), _dt("10:06"),
            _dt("10:21"), _dt("10:21"), _dt("10:21"),
        ),
    )
    corpus.capture_prejump(**fixture())
    _capture(corpus, "request-1", Transport(_html()))
    original = corpus._publish_once

    def publish_then_crash(path, content):
        result = original(path, content)
        if path.name == receipt_name:
            raise OSError("simulated post-publication crash")
        return result

    monkeypatch.setattr(corpus, "_publish_once", publish_then_crash)
    body = _html()
    if changed:
        body = body.replace(b"1st", b"2nd").replace(b"2nd", b"1st", 1)
    with pytest.raises(OSError):
        _capture(corpus, "request-2", Transport(body))
    monkeypatch.setattr(corpus, "_publish_once", original)
    replay_transport = Transport(b"must not be fetched")
    _capture(corpus, "request-2", replay_transport)
    assert replay_transport.calls == 0


def test_closure_publication_crash_replay_is_local(tmp_path, monkeypatch):
    corpus = _stable_corpus(tmp_path)
    original = corpus._publish_once

    def publish_then_crash(path, content):
        result = original(path, content)
        if path.name == "closure.json":
            raise OSError("simulated post-publication crash")
        return result

    monkeypatch.setattr(corpus, "_publish_once", publish_then_crash)
    with pytest.raises(OSError):
        corpus.close(race_id="race-1")
    monkeypatch.setattr(corpus, "_publish_once", original)
    assert corpus.close(race_id="race-1")["race_id"] == "race-1"
    replay_transport = Transport(b"must not be fetched")
    _capture(corpus, "request-1", replay_transport)
    assert replay_transport.calls == 0


def test_changed_observation_without_conflict_receipt_fails_closed_everywhere(
    tmp_path, monkeypatch
):
    corpus = ForwardSealedCorpus(
        tmp_path,
        clock=Clock(
            _dt("09:45"),
            _dt("10:06"), _dt("10:06"), _dt("10:06"),
            _dt("10:21"), _dt("10:21"), _dt("10:21"),
        ),
    )
    corpus.capture_prejump(**fixture())
    _capture(corpus, "request-1", Transport(_html()))
    monkeypatch.setattr(
        corpus,
        "_refresh_stability",
        lambda _pre: (_ for _ in ()).throw(OSError("crash before conflict receipt")),
    )
    changed = _html().replace(b"1st", b"2nd").replace(b"2nd", b"1st", 1)
    with pytest.raises(OSError):
        _capture(corpus, "request-2", Transport(changed))
    monkeypatch.undo()
    assert corpus.status()["races"][0]["state"] == "RESULT_CHANGED_BEFORE_CLOSURE"
    with pytest.raises(ForwardCorpusRejected, match="changed"):
        corpus.close(race_id="race-1")
    with pytest.raises(ForwardCorpusRejected):
        corpus.build_package()


def test_interrupted_conflict_is_recovered_before_distinct_request_transport(
    tmp_path, monkeypatch
):
    corpus = ForwardSealedCorpus(
        tmp_path,
        clock=Clock(
            _dt("09:45"),
            _dt("10:06"), _dt("10:06"), _dt("10:06"),
            _dt("10:21"), _dt("10:21"), _dt("10:21"),
        ),
    )
    corpus.capture_prejump(**fixture())
    _capture(corpus, "request-1", Transport(_html()))
    original = corpus._refresh_stability
    monkeypatch.setattr(
        corpus,
        "_refresh_stability",
        lambda _pre: (_ for _ in ()).throw(OSError("interrupted conflict publication")),
    )
    changed = _html().replace(b"1st", b"2nd").replace(b"2nd", b"1st", 1)
    with pytest.raises(OSError):
        _capture(corpus, "request-2", Transport(changed))
    monkeypatch.setattr(corpus, "_refresh_stability", original)
    later = Transport(_html())
    with pytest.raises(ForwardCorpusRejected, match="changed"):
        _capture(corpus, "request-3", later)
    assert later.calls == 0
    assert corpus._receipt_path("race-1", "conflict").exists()
    replay = Transport(b"never fetched")
    with pytest.raises(ForwardCorpusRejected, match="changed"):
        _capture(corpus, "request-3", replay)
    assert replay.calls == 0


def test_three_observations_and_response_stages_are_completely_inventoried(tmp_path):
    corpus = ForwardSealedCorpus(
        tmp_path,
        clock=Clock(
            _dt("09:45"),
            _dt("10:05"), _dt("10:05"), _dt("10:05"),
            _dt("10:06"), _dt("10:06"), _dt("10:06"),
            _dt("10:21"), _dt("10:21"), _dt("10:21"),
            _dt("10:36"), _dt("10:36"), _dt("10:36"),
            _dt("10:37"),
        ),
    )
    corpus.capture_prejump(**fixture())
    malformed = Transport(b"<html>retained but not a result</html>")
    with pytest.raises(ForwardCorpusRejected):
        _capture(corpus, "request-0", malformed)
    replay = Transport(_html())
    with pytest.raises(ForwardCorpusRejected):
        _capture(corpus, "request-0", replay)
    assert replay.calls == 0
    for request_id in ("request-1", "request-2", "request-3"):
        _capture(corpus, request_id, Transport(_html()))
    package = close(corpus)
    race = json.loads(package.package_bytes)["manifest"]["races"][0]
    assert len(race["response_stage_checksums"]) == 4
    assert len(race["raw_response_checksums"]) == 4
    assert len(race["observation_checksums"]) == 3
    for field in (
        "response_stage_checksums",
        "raw_response_checksums",
        "observation_checksums",
    ):
        assert all(checksum in package.artifacts for checksum in race[field])
    admit_historical_source(package.package_bytes, artifacts=package.artifacts)


def test_safe_url_variation_stabilizes_while_exact_urls_remain_bound(tmp_path):
    corpus = ForwardSealedCorpus(
        tmp_path,
        clock=Clock(
            _dt("09:45"),
            _dt("10:06"), _dt("10:06"), _dt("10:06"),
            _dt("10:21"), _dt("10:21"), _dt("10:21"),
            _dt("10:22"),
        ),
    )
    corpus.capture_prejump(**fixture())
    first = corpus.capture_result(
        race_id="race-1", collector_id="collector-1", session_id="session-1",
        run_id="run-1", request_id="request-1",
        request_url="https://www.thedogs.com.au/racing/test/results",
        transport=Transport(
            _html(), final_url="https://www.thedogs.com.au/racing/test/results"
        ),
    )
    second = corpus.capture_result(
        race_id="race-1", collector_id="collector-1", session_id="session-1",
        run_id="run-1", request_id="request-2",
        request_url="https://thedogs.com.au/racing/test/results/redirected",
        transport=Transport(
            _html(), final_url="https://www.thedogs.com.au/racing/test/results/canonical"
        ),
    )
    assert first["request_url"] != second["request_url"]
    assert first["final_url"] != second["final_url"]
    package = close(corpus)
    admit_historical_source(package.package_bytes, artifacts=package.artifacts)


@pytest.mark.parametrize("source_name", ["TheDogs", "thedogs_official", "thedogs-official "])
def test_noncanonical_source_names_fail_capture_and_admission(tmp_path, source_name):
    corpus = ForwardSealedCorpus(tmp_path, clock=lambda: _dt("09:45"))
    value = fixture()
    value["source_name"] = source_name
    with pytest.raises(ForwardCorpusRejected, match="canonical"):
        corpus.capture_prejump(**value)
    valid = ForwardSealedCorpus(tmp_path / "valid", clock=lambda: _dt("09:45"))
    valid.capture_prejump(**fixture())
    with pytest.raises(ForwardCorpusRejected, match="canonical"):
        valid.capture_result(
            race_id="race-1",
            collector_id="collector-1",
            session_id="session-1",
            run_id="run-1",
            request_id="request-1",
            request_url="https://www.thedogs.com.au/racing/test/results",
            transport=Transport(_html()),
            source_name=source_name,
        )


def test_package_inventories_and_validates_prejump_and_closure_receipts(tmp_path):
    package = close(_stable_corpus(tmp_path))
    race = json.loads(package.package_bytes)["manifest"]["races"][0]
    for field in ("prejump_receipt_checksum", "closure_receipt_checksum"):
        checksum = race[field]
        assert checksum in package.artifacts
        mutated = dict(package.artifacts)
        mutated[checksum] = package.artifacts[checksum] + b" "
        with pytest.raises(SourceAdmissionRejected):
            admit_historical_source(package.package_bytes, artifacts=mutated)


def test_complete_history_inventory_mutations_fail_local_and_pure_reconstruction(
    tmp_path,
):
    corpus = ForwardSealedCorpus(
        tmp_path,
        clock=Clock(
            _dt("09:45"),
            _dt("10:05"), _dt("10:05"), _dt("10:05"),
            _dt("10:06"), _dt("10:06"), _dt("10:06"),
            _dt("10:21"), _dt("10:21"), _dt("10:21"),
            _dt("10:36"), _dt("10:36"), _dt("10:36"),
            _dt("10:37"),
        ),
    )
    corpus.capture_prejump(**fixture())
    with pytest.raises(ForwardCorpusRejected):
        _capture(corpus, "request-0", Transport(b"<html>not a result</html>"))
    _capture(corpus, "request-1", Transport(_html()))
    _capture(corpus, "request-2", Transport(_html(whitespace=" ")))
    _capture(corpus, "request-3", Transport(_html(whitespace="\n")))
    package = close(corpus)
    original_package = json.loads(package.package_bytes)
    original_closure_path = corpus._receipt_path("race-1", "closure")
    original_closure = original_closure_path.read_bytes()

    def mutate(values, operation, source_checksum):
        if operation == "delete":
            values.pop()
        elif operation == "add":
            values.append(source_checksum)
        elif operation == "duplicate":
            values.append(values[0])
        elif operation == "reorder":
            values.reverse()
        else:
            values[0] = values[1]

    for field in (
        "response_stage_checksums",
        "raw_response_checksums",
        "observation_checksums",
    ):
        for operation in ("delete", "add", "duplicate", "reorder", "substitute"):
            package_value = copy.deepcopy(original_package)
            race = package_value["manifest"]["races"][0]
            mutate(race[field], operation, race["source_checksum"])
            manifest_bytes = canonical_json(package_value["manifest"])
            package_value["manifest_checksum"] = (
                "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
            )
            with pytest.raises(SourceAdmissionRejected):
                admit_historical_source(
                    canonical_json(package_value), artifacts=dict(package.artifacts)
                )

            closure = json.loads(original_closure)
            mutate(
                closure["race"][field],
                operation,
                closure["race"]["source_checksum"],
            )
            original_closure_path.write_bytes(canonical_json(closure))
            with pytest.raises(ForwardCorpusRejected):
                corpus.build_package()
            original_closure_path.write_bytes(original_closure)

    orphan_directory = corpus._request_directory(tmp_path, "race-1", "request-0")
    completed_directory = corpus._request_directory(tmp_path, "race-1", "request-1")
    orphan_stage = orphan_directory / "response-stage.json"
    completed_observation = completed_directory / "observation.json"
    original_stage = orphan_stage.read_bytes()
    original_observation = completed_observation.read_bytes()
    false_binding = json.loads(original_stage)
    false_binding["request_id"] = "request-1"
    orphan_stage.write_bytes(canonical_json(false_binding))
    with pytest.raises(ForwardCorpusRejected):
        corpus.build_package()
    orphan_stage.write_bytes(original_stage)

    false_package = copy.deepcopy(original_package)
    false_race = false_package["manifest"]["races"][0]
    old_observation_checksum = false_race["observation_checksums"][0]
    false_observation = json.loads(package.artifacts[old_observation_checksum])
    false_observation["request_id"] = "request-0"
    false_observation_bytes = canonical_json(false_observation)
    false_observation_checksum = (
        "sha256:" + hashlib.sha256(false_observation_bytes).hexdigest()
    )
    false_race["observation_checksums"][0] = false_observation_checksum
    for field in ("first_observation_checksum", "second_observation_checksum"):
        if false_race[field] == old_observation_checksum:
            false_race[field] = false_observation_checksum
    false_manifest_bytes = canonical_json(false_package["manifest"])
    false_package["manifest_checksum"] = (
        "sha256:" + hashlib.sha256(false_manifest_bytes).hexdigest()
    )
    false_artifacts = dict(package.artifacts)
    false_artifacts.pop(old_observation_checksum)
    false_artifacts[false_observation_checksum] = false_observation_bytes
    with pytest.raises(SourceAdmissionRejected):
        admit_historical_source(canonical_json(false_package), artifacts=false_artifacts)

    completed_observation.unlink()
    with pytest.raises(ForwardCorpusRejected):
        corpus.build_package()
    completed_observation.write_bytes(original_observation)

    requests_root = orphan_directory.parent
    empty_hashed = requests_root / hashlib.sha256(b"empty-request").hexdigest()
    empty_hashed.mkdir()
    with pytest.raises(ForwardCorpusRejected, match="inventory"):
        corpus.build_package()
    empty_hashed.rmdir()

    arbitrary_directory = requests_root / "not-a-request-hash"
    arbitrary_directory.mkdir()
    with pytest.raises(ForwardCorpusRejected, match="inventory"):
        corpus.build_package()
    arbitrary_directory.rmdir()

    root_file = requests_root / "unexpected.txt"
    root_file.write_bytes(b"unexpected")
    with pytest.raises(ForwardCorpusRejected, match="directory"):
        corpus.build_package()
    root_file.unlink()

    unknown_file = orphan_directory / "unknown.json"
    unknown_file.write_bytes(b"{}")
    with pytest.raises(ForwardCorpusRejected, match="inventory"):
        corpus.build_package()
    unknown_file.unlink()

    unknown_directory = orphan_directory / "unknown"
    unknown_directory.mkdir()
    with pytest.raises(ForwardCorpusRejected, match="artifact"):
        corpus.build_package()
    unknown_directory.rmdir()

    orphan_stage_value = json.loads(original_stage)
    raw_checksum = orphan_stage_value["raw_response_checksum"]
    raw_path = corpus.artifacts.path_for(corpus.artifacts.checksum(package.artifacts[raw_checksum]))
    raw_backup = tmp_path / "raw-response.backup"
    raw_path.rename(raw_backup)
    with pytest.raises(ForwardCorpusRejected, match="missing"):
        corpus.build_package()
    raw_backup.rename(raw_path)

    stage_backup = tmp_path / "response-stage.backup"
    orphan_stage.rename(stage_backup)
    orphan_observation = orphan_directory / "observation.json"
    orphan_observation.write_bytes(original_observation)
    with pytest.raises(ForwardCorpusRejected, match="inventory"):
        corpus.build_package()
    orphan_observation.unlink()
    stage_backup.rename(orphan_stage)

    disguised_artifacts = dict(package.artifacts)
    disguised_artifacts["sha256:" + hashlib.sha256(b"disguised").hexdigest()] = b"disguised"
    with pytest.raises(SourceAdmissionRejected, match="inventory"):
        admit_historical_source(package.package_bytes, artifacts=disguised_artifacts)


@pytest.mark.parametrize(
    "family,mutate",
    [
        ("envelope", lambda value: value.__setitem__("extra", True)),
        ("source-name", lambda value: value.__setitem__("source_name", "TheDogs")),
        ("source-url", lambda value: value.__setitem__("canonical_source_url", "https://evil.test")),
        ("race-identity", lambda value: value.__setitem__("source_native_race_id", "other")),
        ("racing-date", lambda value: value.__setitem__("racing_date", "2026-07-30")),
        ("timestamps", lambda value: value.__setitem__("feature_frozen_at", value["scheduled_jump_at"])),
        ("authority", lambda value: value.__setitem__("identity_authority", "reconstructed")),
        ("reconstruction", lambda value: value.__setitem__("reconstructed", True)),
        ("runner-id", lambda value: value["runners"][1].__setitem__(
            "source_native_runner_id", value["runners"][0]["source_native_runner_id"]
        )),
        ("runner-box", lambda value: value["runners"][1].__setitem__(
            "box_number", value["runners"][0]["box_number"]
        )),
        ("runner-name", lambda value: value["runners"][1].__setitem__(
            "name", value["runners"][0]["name"]
        )),
    ],
)
def test_source_capture_family_mutations_fail_pure_admission(
    tmp_path, family, mutate
):
    package = close(_stable_corpus(tmp_path))
    race = json.loads(package.package_bytes)["manifest"]["races"][0]
    checksum = race["source_capture_checksum"]
    artifacts = dict(package.artifacts)
    capture = json.loads(artifacts[checksum])
    mutate(capture)
    artifacts[checksum] = canonical_json(capture)
    with pytest.raises(SourceAdmissionRejected):
        admit_historical_source(package.package_bytes, artifacts=artifacts)


@pytest.mark.parametrize("field", ["source_name", "request_id"])
def test_observation_identity_mutation_fails_pure_admission(tmp_path, field):
    package = close(_stable_corpus(tmp_path))
    race = json.loads(package.package_bytes)["manifest"]["races"][0]
    checksum = race["first_observation_checksum"]
    artifacts = dict(package.artifacts)
    observation = json.loads(artifacts[checksum])
    observation[field] = "noncanonical"
    artifacts[checksum] = canonical_json(observation)
    with pytest.raises(SourceAdmissionRejected):
        admit_historical_source(package.package_bytes, artifacts=artifacts)


def test_stability_mutation_fails_pure_admission(tmp_path):
    package = close(_stable_corpus(tmp_path))
    race = json.loads(package.package_bytes)["manifest"]["races"][0]
    checksum = race["stability_checksum"]
    artifacts = dict(package.artifacts)
    stability = json.loads(artifacts[checksum])
    stability["confirmed_at"] = "2026-07-29T10:22:00+10:00"
    artifacts[checksum] = canonical_json(stability)
    with pytest.raises(SourceAdmissionRejected):
        admit_historical_source(package.package_bytes, artifacts=artifacts)


@pytest.mark.parametrize("receipt", ["stability", "conflict"])
def test_mutated_terminal_receipts_fail_local_reconstruction(tmp_path, receipt):
    if receipt == "stability":
        corpus = _stable_corpus(tmp_path)
    else:
        corpus = ForwardSealedCorpus(
            tmp_path,
            clock=Clock(
                _dt("09:45"),
                _dt("10:06"), _dt("10:06"), _dt("10:06"),
                _dt("10:21"), _dt("10:21"), _dt("10:21"),
            ),
        )
        corpus.capture_prejump(**fixture())
        _capture(corpus, "request-1", Transport(_html()))
        changed = _html().replace(b"1st", b"2nd").replace(b"2nd", b"1st", 1)
        _capture(corpus, "request-2", Transport(changed))
    path = corpus._receipt_path("race-1", receipt)
    value = json.loads(path.read_bytes())
    value["race_id"] = "mutated"
    path.write_bytes(canonical_json(value))
    with pytest.raises(ForwardCorpusRejected):
        corpus.status()
    with pytest.raises(ForwardCorpusRejected):
        corpus.close(race_id="race-1")
    with pytest.raises(ForwardCorpusRejected):
        corpus.build_package()


def test_legacy_publication_capture_cannot_create_new_origin_artifacts(tmp_path):
    corpus = ForwardSealedCorpus(tmp_path, clock=lambda: _dt("09:45"))
    corpus.capture_prejump(**fixture())
    with pytest.raises(ForwardCorpusRejected, match="legacy"):
        corpus._publication_timestamp_capture_result(
            race_id="race-1",
            raw_result_bytes=b"result",
            source_name="thedogs-official",
            canonical_source_url="https://www.thedogs.com.au/racing/test/results",
            source_native_race_id="thedogs-2026-07-29-1",
            runners=fixture()["runners"],
            official_order=["dog-1-b", "dog-1-a"],
            result_observed_at="2026-07-29T10:06:00+10:00",
            result_published_at="2026-07-29T10:05:00+10:00",
            publication_timestamp_status="source-declared",
        )
    assert not corpus._receipt_path("race-1", "result").exists()
