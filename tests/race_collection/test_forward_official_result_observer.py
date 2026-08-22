import hashlib
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import pytest

from race_collection.domain import ArtifactChecksum
from race_collection.forward_sealed_corpus import ForwardCorpusRejected, ForwardSealedCorpus
from scripts import observe_forward_official_results as observer


def _accepted_helpers():
    path = Path(__file__).with_name("test_forward_sealed_corpus.py")
    spec = importlib.util.spec_from_file_location("_accepted_t1_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


T1 = _accepted_helpers()
_fixture_html = T1._html


def _result_html(index=1, *, whitespace=""):
    body = _fixture_html(index, whitespace=whitespace)
    for suffix in ("a", "b"):
        name = f"Dog {index} {suffix.upper()}".encode()
        link = (
            f'<a href="/dogs/dog-{index}-{suffix}/dog-{index}-{suffix}">'.encode()
            + name
            + b"</a>"
        )
        body = body.replace(name, link)
    return body


T1._html = _result_html


def _dt(value):
    return datetime.fromisoformat(f"2026-07-29T{value}:00+10:00")


class Clock:
    def __init__(self, *values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


class Response:
    def __init__(self, body, url, *, headers=None, status=200):
        self.content = body
        self.url = url
        self.status_code = status
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8"}
        self.raw = Raw(body)
        self.closed = False

    def close(self):
        self.closed = True


class Raw:
    def __init__(self, body):
        self.body = body
        self.decode_content = True

    def read(self, amount, *, decode_content):
        assert self.decode_content is False
        assert decode_content is False
        return self.body[:amount]


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.closed = False

    def get(self, url, headers, timeout, *, allow_redirects, stream):
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "timeout": timeout,
                "allow_redirects": allow_redirects,
                "stream": stream,
            }
        )
        response = next(self.responses)
        if response.url is None:
            response.url = url
        return response

    def close(self):
        self.closed = True


def _seed(root, *, race_id="race-1", url=None):
    value = T1.fixture()
    value["race_id"] = race_id
    value["sealed_evidence_bytes"] = value["sealed_evidence_bytes"].replace(
        b'"race_id":"race-1"', f'"race_id":"{race_id}"'.encode()
    )
    value["canonical_source_url"] = url or (
        "https://www.thedogs.com.au/racing/venue/2026-07-29/1/race-name"
    )
    ForwardSealedCorpus(root, clock=lambda: _dt("09:45")).capture_prejump(**value)


def _run(root, cycle, times, responses, **observer_kwargs):
    session = Session(responses)
    report = observer.observe_once(
        corpus_root=root,
        cycle_id=cycle,
        clock=Clock(*times),
        session_factory=lambda: session,
        **observer_kwargs,
    )
    return report, session


def test_first_then_second_identical_closes_and_packages(tmp_path):
    _seed(tmp_path)
    url = (
        "https://www.thedogs.com.au/racing/venue/2026-07-29/1/race-name?trial=false"
    )
    first, session = _run(
        tmp_path,
        "cycle-1",
        [_dt("10:06")] * 4,
        [
            Response(
                T1._html(),
                url,
                headers={
                    "Content-Type": "text/html; charset=utf-8",
                    "Last-Modified": "Wed, 29 Jul 2026 00:05:00 GMT",
                    "Date": "Wed, 29 Jul 2026 00:06:00 GMT",
                },
            )
        ],
    )
    assert first["status"] == "COMPLETED"
    assert first["races"][0]["after_state"] == "RESULT_FIRST_OBSERVED"
    assert first["attempted_race_ids"] == ["race-1"]
    assert session.closed
    observation = next((tmp_path / "races").glob("*/official-requests/*/observation.json"))
    recorded = json.loads(observation.read_text())
    assert recorded["source_document_last_modified"].startswith("Wed")
    assert "Date" not in recorded and "published_at" not in recorded
    assert ForwardSealedCorpus(tmp_path).artifacts.read(
        ArtifactChecksum(recorded["raw_response_checksum"])
    ) == T1._html()

    second, _ = _run(
        tmp_path,
        "cycle-2",
        [_dt("10:21")] * 5,
        [Response(T1._html(), url)],
    )
    assert second["status"] == "COMPLETED"
    assert second["races"][0]["after_state"] == "EXAMPLE_CLOSED"
    assert second["counts"]["closed"] == 1
    assert second["package_hashes"][0]["package_checksum"]
    assert ForwardSealedCorpus(tmp_path).status()["races"][0]["state"] == "EXAMPLE_CLOSED"
    terminal, session = _run(tmp_path, "cycle-3", [], [])
    assert terminal["races"][0]["decision"] == "SKIPPED_TERMINAL"
    assert session.calls == []


def test_changed_second_is_terminal_and_retains_receipts(tmp_path):
    _seed(tmp_path)
    url = (
        "https://www.thedogs.com.au/racing/venue/2026-07-29/1/race-name?trial=false"
    )
    _run(tmp_path, "one", [_dt("10:06")] * 4, [Response(T1._html(), url)])
    changed = T1._html().replace(b"1st", b"2nd").replace(b"2nd", b"1st", 1)
    report, _ = _run(tmp_path, "two", [_dt("10:21")] * 4, [Response(changed, url)])
    assert report["races"][0]["after_state"] == "RESULT_CHANGED_BEFORE_CLOSURE"
    assert report["package_hashes"] == []
    assert len(list((tmp_path / "races").glob("*/official-requests/*/observation.json"))) == 2


def test_pre_boundary_skip_malformed_retention_and_terminal_skip(tmp_path):
    _seed(tmp_path)
    skipped, session = _run(tmp_path, "early", [_dt("10:04")], [])
    assert skipped["races"][0]["decision"] == "SKIPPED_PRE_BOUNDARY"
    assert session.calls == []

    url = (
        "https://www.thedogs.com.au/racing/venue/2026-07-29/1/race-name?trial=false"
    )
    malformed, _ = _run(
        tmp_path,
        "malformed",
        [_dt("10:06")] * 4,
        [Response(b"<html>not a result</html>", url)],
    )
    assert malformed["status"] == "COMPLETED_WITH_REJECTIONS"
    assert malformed["source_rejection_count"] == 1
    assert malformed["source_rejected_race_ids"] == ["race-1"]
    assert malformed["races"][0]["decision"] == "SOURCE_REJECTED"
    assert malformed["races"][0]["error"] is None
    assert malformed["races"][0]["source_rejection"] == (
        "ForwardCorpusRejected: official result HTML contains no result rows"
    )
    assert malformed["races"][0]["raw_response_hash"]
    assert malformed["races"][0]["semantic_fingerprint"] is None
    assert malformed["source_rejection_deferrals"] == []
    assert len(list((tmp_path / "races").glob("*/official-requests/*/response-stage.json"))) == 1


def test_partial_official_order_is_source_rejected_without_failing_observer(tmp_path):
    _seed(tmp_path)
    url = "https://www.thedogs.com.au/racing/venue/2026-07-29/1/race-name?trial=false"
    partial = T1._html().replace(b"2nd", b"-")

    report, _ = _run(
        tmp_path,
        "partial-order",
        [_dt("10:06")] * 4,
        [Response(partial, url)],
    )

    assert report["status"] == "COMPLETED_WITH_REJECTIONS"
    assert report["source_rejected_race_ids"] == ["race-1"]
    assert report["races"][0]["after_state"] == "RESULT_PENDING"
    assert report["races"][0]["decision"] == "SOURCE_REJECTED"
    assert report["races"][0]["error"] is None
    assert report["races"][0]["source_rejection"] == (
        "ForwardCorpusRejected: official finish/status combination is inconsistent"
    )


def test_duplicate_result_box_is_not_eligible_for_semantic_deferral(tmp_path):
    _seed(tmp_path)
    url = "https://www.thedogs.com.au/racing/venue/2026-07-29/1/race-name?trial=false"
    duplicate_box = T1._html().replace(b"rug_2", b"rug_1")

    report, _ = _run(
        tmp_path,
        "duplicate-result-box",
        [_dt("10:06")] * 4,
        [Response(duplicate_box, url)],
    )

    race = report["races"][0]
    assert race["decision"] == "SOURCE_REJECTED"
    assert race["source_rejection"] == (
        "ForwardCorpusRejected: official result runner box/rug identity mismatch"
    )
    assert race["semantic_fingerprint"] is None
    assert report["source_rejection_deferrals"] == []


def test_missing_result_box_is_not_eligible_for_semantic_deferral(tmp_path):
    _seed(tmp_path)
    url = "https://www.thedogs.com.au/racing/venue/2026-07-29/1/race-name?trial=false"
    missing_box = T1._html().replace(b'<tr class="race-runner">', b"<tr>", 1)

    report, _ = _run(
        tmp_path,
        "missing-result-box",
        [_dt("10:06")] * 4,
        [Response(missing_box, url)],
    )

    race = report["races"][0]
    assert race["decision"] == "SOURCE_REJECTED"
    assert race["source_rejection"] == (
        "ForwardCorpusRejected: official result runner box/rug identity mismatch"
    )
    assert race["semantic_fingerprint"] is None
    assert report["source_rejection_deferrals"] == []


def test_unknown_result_box_is_not_eligible_for_semantic_deferral(tmp_path):
    _seed(tmp_path)
    url = "https://www.thedogs.com.au/racing/venue/2026-07-29/1/race-name?trial=false"
    unknown_box = T1._html().replace(b"rug_2", b"rug_3")

    report, _ = _run(
        tmp_path,
        "unknown-result-box",
        [_dt("10:06")] * 4,
        [Response(unknown_box, url)],
    )

    race = report["races"][0]
    assert race["decision"] == "SOURCE_REJECTED"
    assert race["source_rejection"] == (
        "ForwardCorpusRejected: official result runner box/rug identity mismatch"
    )
    assert race["semantic_fingerprint"] is None
    assert report["source_rejection_deferrals"] == []


def test_identical_rejection_is_deferred_and_changed_bytes_can_close(tmp_path):
    _seed(tmp_path)
    url = "https://www.thedogs.com.au/racing/venue/2026-07-29/1/race-name?trial=false"
    partial = T1._html().replace(b"2nd", b"-")

    first, _ = _run(
        tmp_path,
        "rejected-first",
        [_dt("10:06")] * 4,
        [Response(partial, url)],
    )

    response_hash = first["races"][0]["raw_response_hash"]
    assert first["races"][0]["normalization_attempted"] is True
    deferral = first["races"][0]["rejection_deferral"]
    assert deferral["schema_version"] == observer.REJECTION_DEFERRAL_SCHEMA
    assert deferral["race_id"] == "race-1"
    assert deferral["source_native_race_id"] == "thedogs-2026-07-29-1"
    assert deferral["retained_request_id"] == first["races"][0]["request_id"]
    assert deferral["retained_raw_response_hash"] == response_hash
    assert deferral["semantic_fingerprint"] == first["races"][0][
        "semantic_fingerprint"
    ]
    assert deferral["reason"] == "official finish/status combination is inconsistent"
    assert deferral["rejected_at"] == "2026-07-29T10:06:00.000000+10:00"
    assert deferral["next_eligible_at"] == "2026-07-29T11:06:00.000000+10:00"
    assert deferral["deferral_decision"] == "SOURCE_REJECTED"
    assert deferral["pending_state"] == "RESULT_PENDING"
    assert first["source_rejection_deferrals"] == [
        first["races"][0]["rejection_deferral"]
    ]

    identical, session = _run(
        tmp_path,
        "rejected-identical",
        [_dt("10:21")],
        [Response(partial, url)],
        previous_rejection_deferrals=first["source_rejection_deferrals"],
    )

    assert identical["status"] == "COMPLETED_WITH_REJECTIONS"
    assert identical["races"][0]["decision"] == "SOURCE_REJECTION_DEFERRED"
    assert identical["races"][0]["after_state"] == "RESULT_PENDING"
    assert identical["races"][0]["normalization_attempted"] is False
    assert identical["races"][0]["raw_response_hash"] == response_hash
    assert identical["races"][0]["rejection_deferral"] == deferral | {
        "deferral_decision": "SOURCE_REJECTION_DEFERRED"
    }
    assert identical["races"][0]["deferral_reason"] == (
        "identical_result_semantics_before_next_eligibility"
    )
    assert len(session.calls) == 1
    assert len(list((tmp_path / "races").glob("*/official-requests/*/response-stage.json"))) == 1

    changed, _ = _run(
        tmp_path,
        "rejected-changed",
        [_dt("10:22")] * 4,
        [Response(T1._html(), url)],
        previous_rejection_deferrals=identical["source_rejection_deferrals"],
    )
    assert changed["status"] == "COMPLETED"
    assert changed["races"][0]["after_state"] == "RESULT_FIRST_OBSERVED"
    assert changed["source_rejection_deferrals"] == []

    closed, _ = _run(
        tmp_path,
        "valid-second",
        [_dt("10:38")] * 5,
        [Response(T1._html(), url)],
        previous_rejection_deferrals=changed["source_rejection_deferrals"],
    )
    assert closed["status"] == "COMPLETED"
    assert closed["races"][0]["after_state"] == "EXAMPLE_CLOSED"


def test_csrf_only_change_uses_semantic_deferral_without_conflating_raw_hash(tmp_path):
    _seed(tmp_path)
    url = "https://www.thedogs.com.au/racing/venue/2026-07-29/1/race-name?trial=false"
    partial = T1._html().replace(b"2nd", b"-")
    first_body = b'<meta name="csrf-token" content="first">' + partial
    second_body = b'<meta name="csrf-token" content="second">' + partial

    first, _ = _run(
        tmp_path,
        "semantic-first",
        [_dt("10:06")] * 4,
        [Response(first_body, url)],
    )
    deferred, _ = _run(
        tmp_path,
        "semantic-second",
        [_dt("10:21")],
        [Response(second_body, url)],
        previous_rejection_deferrals=first["source_rejection_deferrals"],
    )

    first_race = first["races"][0]
    deferred_race = deferred["races"][0]
    assert first_race["raw_response_hash"] != deferred_race["raw_response_hash"]
    assert first_race["semantic_fingerprint"] == deferred_race["semantic_fingerprint"]
    assert first_race["fingerprint_algorithm_version"] == (
        "thedogs-official-result-semantic-projection-sha256-v1"
    )
    assert deferred_race["decision"] == "SOURCE_REJECTION_DEFERRED"
    assert deferred_race["normalization_attempted"] is False
    assert deferred_race["rejection_deferral"]["retained_raw_response_hash"] == (
        first_race["raw_response_hash"]
    )
    stages = list((tmp_path / "races").glob("*/official-requests/*/response-stage.json"))
    assert len(stages) == 1
    retained = json.loads(stages[0].read_text())
    assert ForwardSealedCorpus(tmp_path).artifacts.read(
        ArtifactChecksum(retained["raw_response_checksum"])
    ) == first_body


@pytest.mark.parametrize(
    "changed_body",
    [
        T1._html(),
        T1._html().replace(b"2nd", b"DNF"),
        T1._html().replace(b"2nd", b"-").replace(b">-<", b">VOID<"),
        T1._html().replace(b"2nd", b"-").replace(b"Dog 1 A", b"Dog 1 A csrf_deadbeef"),
        T1._html().replace(b"2nd", b"-").replace(b"rug_1", b"rug_3"),
    ],
    ids=[
        "finish",
        "recognized-status",
        "unrecognized-status",
        "runner-identity-token-text",
        "box-rug",
    ],
)
def test_material_result_change_bypasses_semantic_deferral(tmp_path, changed_body):
    _seed(tmp_path)
    url = "https://www.thedogs.com.au/racing/venue/2026-07-29/1/race-name?trial=false"
    partial = T1._html().replace(b"2nd", b"-")
    first, _ = _run(
        tmp_path,
        "material-first",
        [_dt("10:06")] * 4,
        [Response(partial, url)],
    )

    changed, _ = _run(
        tmp_path,
        "material-changed",
        [_dt("10:21")] * 4,
        [Response(changed_body, url)],
        previous_rejection_deferrals=first["source_rejection_deferrals"],
    )

    assert changed["races"][0]["semantic_fingerprint"] != first["races"][0][
        "semantic_fingerprint"
    ]
    assert changed["races"][0]["decision"] != "SOURCE_REJECTION_DEFERRED"
    assert changed["races"][0]["normalization_attempted"] is True


def test_current_response_native_runner_identity_change_bypasses_deferral(tmp_path):
    _seed(tmp_path)
    url = "https://www.thedogs.com.au/racing/venue/2026-07-29/1/race-name?trial=false"
    partial = T1._html().replace(b"2nd", b"-")
    first, _ = _run(
        tmp_path,
        "native-response-first",
        [_dt("10:06")] * 4,
        [Response(partial, url)],
    )
    changed_native_id = partial.replace(
        b'/dogs/dog-1-a/dog-1-a', b'/dogs/dog-other/dog-1-a'
    )

    changed, _ = _run(
        tmp_path,
        "native-response-changed",
        [_dt("10:21")] * 4,
        [Response(changed_native_id, url)],
        previous_rejection_deferrals=first["source_rejection_deferrals"],
    )

    assert changed["races"][0]["semantic_fingerprint"] != first["races"][0][
        "semantic_fingerprint"
    ]
    assert changed["races"][0]["decision"] != "SOURCE_REJECTION_DEFERRED"
    assert changed["races"][0]["normalization_attempted"] is True


def test_duplicate_current_response_native_runner_identity_cannot_defer(tmp_path):
    _seed(tmp_path)
    url = "https://www.thedogs.com.au/racing/venue/2026-07-29/1/race-name?trial=false"
    duplicate_native_id = (
        T1._html()
        .replace(b"2nd", b"-")
        .replace(b'/dogs/dog-1-b/dog-1-b', b'/dogs/dog-1-a/dog-1-b')
    )

    first, _ = _run(
        tmp_path,
        "duplicate-native-first",
        [_dt("10:06")] * 4,
        [Response(duplicate_native_id, url)],
    )
    repeated, _ = _run(
        tmp_path,
        "duplicate-native-repeated",
        [_dt("10:21")] * 4,
        [Response(duplicate_native_id, url)],
        previous_rejection_deferrals=first["source_rejection_deferrals"],
    )

    assert first["races"][0]["semantic_fingerprint"] is None
    assert first["source_rejection_deferrals"] == []
    assert repeated["races"][0]["decision"] == "SOURCE_REJECTED"
    assert repeated["races"][0]["normalization_attempted"] is True


@pytest.mark.parametrize(
    "identity_mutation",
    [
        lambda body: body.replace(
            b'<a href="/dogs/dog-1-a/dog-1-a">Dog 1 A</a>', b"Dog 1 A"
        ),
        lambda body: body.replace(
            b'<a href="/dogs/dog-1-a/dog-1-a">',
            b'<blackbook-dog data-dog-id="dog-conflict"></blackbook-dog>'
            b'<a href="/dogs/dog-1-a/dog-1-a">',
        ),
    ],
    ids=["missing", "conflicting-within-row"],
)
def test_incomplete_current_response_native_runner_identity_cannot_defer(
    tmp_path, identity_mutation
):
    _seed(tmp_path)
    url = "https://www.thedogs.com.au/racing/venue/2026-07-29/1/race-name?trial=false"
    incomplete_identity = identity_mutation(T1._html().replace(b"2nd", b"-"))

    report, _ = _run(
        tmp_path,
        "incomplete-native-identity",
        [_dt("10:06")] * 4,
        [Response(incomplete_identity, url)],
    )

    assert report["races"][0]["decision"] == "SOURCE_REJECTED"
    assert report["races"][0]["semantic_fingerprint"] is None
    assert report["source_rejection_deferrals"] == []


def test_semantic_deferral_still_verifies_retained_exact_raw_bytes(tmp_path):
    _seed(tmp_path)
    url = "https://www.thedogs.com.au/racing/venue/2026-07-29/1/race-name?trial=false"
    partial = T1._html().replace(b"2nd", b"-")
    first, _ = _run(
        tmp_path,
        "retained-semantic-first",
        [_dt("10:06")] * 4,
        [Response(partial, url)],
    )
    deferral = dict(first["source_rejection_deferrals"][0])
    deferral["retained_raw_response_hash"] = deferral["semantic_fingerprint"]

    repeated, _ = _run(
        tmp_path,
        "retained-semantic-second",
        [_dt("10:21")],
        [Response(partial, url)],
        previous_rejection_deferrals=[deferral],
    )

    assert repeated["status"] == "FAILED"
    assert repeated["races"] == []
    assert "retained rejection raw-byte hash drift" in repeated["error"]


def test_race_and_native_runner_identity_are_part_of_semantic_fingerprint(tmp_path):
    url = "https://www.thedogs.com.au/racing/venue/2026-07-29/1/race-name?trial=false"
    partial = T1._html().replace(b"2nd", b"-")
    _seed(tmp_path / "base")
    base, _ = _run(
        tmp_path / "base", "identity-base", [_dt("10:06")] * 4, [Response(partial, url)]
    )

    _seed(tmp_path / "other-race", race_id="race-other")
    other_race, _ = _run(
        tmp_path / "other-race",
        "identity-race",
        [_dt("10:06")] * 4,
        [Response(partial, url)],
    )

    native_value = T1.fixture(2)
    native_value["race_id"] = "race-1"
    native_value["sealed_evidence_bytes"] = native_value["sealed_evidence_bytes"].replace(
        b'"race_id":"race-2"', b'"race_id":"race-1"'
    )
    native_value["canonical_source_url"] = url.removesuffix("?trial=false")
    native_value["source_native_race_id"] = "thedogs-2026-07-29-1"
    ForwardSealedCorpus(tmp_path / "other-native", clock=lambda: _dt("09:45")).capture_prejump(
        **native_value
    )
    native_partial = T1._html(2).replace(b"2nd", b"-")
    other_native, _ = _run(
        tmp_path / "other-native",
        "identity-native",
        [_dt("10:06")] * 4,
        [Response(native_partial, url)],
    )

    fingerprint = base["races"][0]["semantic_fingerprint"]
    assert other_race["races"][0]["semantic_fingerprint"] != fingerprint
    assert other_native["races"][0]["semantic_fingerprint"] != fingerprint


def test_known_legacy_deferral_is_reprocessed_once_into_versioned_state(tmp_path):
    _seed(tmp_path)
    url = "https://www.thedogs.com.au/racing/venue/2026-07-29/1/race-name?trial=false"
    partial = T1._html().replace(b"2nd", b"-")
    legacy = {
        "race_id": "race-1",
        "response_hash": hashlib.sha256(partial).hexdigest(),
        "reason": "official finish/status combination is inconsistent",
        "rejected_at": "2026-07-29T10:06:00.000000+10:00",
        "next_eligible_at": "2026-07-29T11:06:00.000000+10:00",
    }

    transitioned, _ = _run(
        tmp_path,
        "legacy-transition",
        [_dt("10:21")] * 4,
        [Response(partial, url)],
        previous_rejection_deferrals=[legacy],
    )

    race = transitioned["races"][0]
    assert race["decision"] == "SOURCE_REJECTED"
    assert race["normalization_attempted"] is True
    assert race["rejection_deferral"]["schema_version"] == (
        observer.REJECTION_DEFERRAL_SCHEMA
    )
    assert race["rejection_deferral"]["rejected_at"] == (
        "2026-07-29T10:21:00.000000+10:00"
    )
    assert race["rejection_deferral"]["next_eligible_at"] == (
        "2026-07-29T11:21:00.000000+10:00"
    )


def test_unprojectable_response_consumes_prior_deferral_instead_of_carrying_it(tmp_path):
    _seed(tmp_path)
    url = "https://www.thedogs.com.au/racing/venue/2026-07-29/1/race-name?trial=false"
    partial = T1._html().replace(b"2nd", b"-")
    first, _ = _run(
        tmp_path,
        "unprojectable-first",
        [_dt("10:06")] * 4,
        [Response(partial, url)],
    )

    unprojectable, _ = _run(
        tmp_path,
        "unprojectable-second",
        [_dt("10:21")] * 4,
        [Response(b"<html>not a result</html>", url)],
        previous_rejection_deferrals=first["source_rejection_deferrals"],
    )

    assert unprojectable["races"][0]["decision"] == "SOURCE_REJECTED"
    assert unprojectable["races"][0]["semantic_fingerprint"] is None
    assert unprojectable["source_rejection_deferrals"] == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "unknown-deferral-v99"),
        ("source_native_race_id", "wrong-source-race"),
        ("pending_state", "RESULT_FIRST_OBSERVED"),
        ("deferral_decision", "UNKNOWN"),
        ("fingerprint_algorithm_version", "unknown-fingerprint-v99"),
        ("next_eligible_at", "9999-07-29T11:06:00.000000+10:00"),
    ],
)
def test_versioned_deferral_corruption_and_inconsistency_are_fatal(
    tmp_path, field, value
):
    _seed(tmp_path)
    url = "https://www.thedogs.com.au/racing/venue/2026-07-29/1/race-name?trial=false"
    partial = T1._html().replace(b"2nd", b"-")
    first, _ = _run(
        tmp_path,
        "versioned-corruption-first",
        [_dt("10:06")] * 4,
        [Response(partial, url)],
    )
    corrupted = dict(first["source_rejection_deferrals"][0])
    corrupted[field] = value

    report, session = _run(
        tmp_path,
        "versioned-corruption-second",
        [],
        [],
        previous_rejection_deferrals=[corrupted],
    )

    assert report["status"] == "FAILED"
    assert "source rejection deferral" in report["error"]
    assert session.calls == []


@pytest.mark.parametrize(
    "deferrals",
    [
        [{"race_id": "race-1"}],
        [
            {
                "race_id": "race-1",
                "response_hash": "a" * 64,
                "reason": "official result is unresolved",
                "rejected_at": "2026-07-29T10:06:00.000000+10:00",
                "next_eligible_at": "9999-07-29T11:06:00.000000+10:00",
            }
        ],
    ],
)
def test_rejection_deferral_metadata_corruption_fails_closed(tmp_path, deferrals):
    _seed(tmp_path)

    report, session = _run(
        tmp_path,
        "corrupt-deferral",
        [],
        [],
        previous_rejection_deferrals=deferrals,
    )

    assert report["status"] == "FAILED"
    assert report["error"] == (
        "ForwardCorpusRejected: source rejection deferral metadata is invalid"
    )
    assert session.calls == []


@pytest.mark.parametrize(
    ("status", "expected_exit"),
    [
        ("COMPLETED", 0),
        ("COMPLETED_WITH_REJECTIONS", 0),
        ("COMPLETED_WITH_ERRORS", 2),
        ("FAILED", 2),
    ],
)
def test_observer_cli_exit_preserves_continuable_and_fatal_statuses(
    monkeypatch, status, expected_exit
):
    monkeypatch.setattr(
        observer,
        "parse_args",
        lambda argv=None: observer.argparse.Namespace(
            corpus_root=Path("/corpus"), cycle_id="cli-cycle", timeout_seconds=30.0
        ),
    )
    monkeypatch.setattr(
        observer,
        "observe_once",
        lambda **kwargs: {"status": status},
    )

    assert observer.main([]) == expected_exit


def test_identical_rejection_is_reprocessed_after_bounded_deferral(tmp_path):
    _seed(tmp_path)
    url = "https://www.thedogs.com.au/racing/venue/2026-07-29/1/race-name?trial=false"
    partial = T1._html().replace(b"2nd", b"-")
    first, _ = _run(
        tmp_path,
        "bounded-first",
        [_dt("10:06")] * 4,
        [Response(partial, url)],
    )

    retried, session = _run(
        tmp_path,
        "bounded-retry",
        [_dt("11:07")] * 4,
        [Response(partial, url)],
        previous_rejection_deferrals=first["source_rejection_deferrals"],
    )

    assert retried["status"] == "COMPLETED_WITH_REJECTIONS"
    assert retried["races"][0]["decision"] == "SOURCE_REJECTED"
    assert retried["races"][0]["normalization_attempted"] is True
    assert retried["races"][0]["rejection_deferral"]["next_eligible_at"] == (
        "2026-07-29T12:07:00.000000+10:00"
    )
    assert len(session.calls) == 1
    assert len(list((tmp_path / "races").glob("*/official-requests/*/response-stage.json"))) == 2


def test_deferred_rejection_does_not_block_unrelated_race(tmp_path):
    first_url = "https://www.thedogs.com.au/racing/venue/2026-07-29/1/race-one"
    second_url = "https://www.thedogs.com.au/racing/venue/2026-07-29/2/race-two"
    _seed(tmp_path, race_id="race-1", url=first_url)
    _seed(tmp_path, race_id="race-2", url=second_url)
    partial = T1._html().replace(b"2nd", b"-")
    first, _ = _run(
        tmp_path,
        "multi-first",
        [_dt("10:06")] * 8,
        [
            Response(partial, first_url + "?trial=false"),
            Response(T1._html(), second_url + "?trial=false"),
        ],
    )
    assert first["source_rejected_race_ids"] == ["race-1"]
    assert first["races"][1]["after_state"] == "RESULT_FIRST_OBSERVED"

    second, session = _run(
        tmp_path,
        "multi-second",
        [_dt("10:22")] * 8,
        [
            Response(partial, first_url + "?trial=false"),
            Response(T1._html(), second_url + "?trial=false"),
        ],
        previous_rejection_deferrals=first["source_rejection_deferrals"],
    )

    assert second["status"] == "COMPLETED_WITH_REJECTIONS"
    assert second["races"][0]["decision"] == "SOURCE_REJECTION_DEFERRED"
    assert second["races"][1]["after_state"] == "EXAMPLE_CLOSED"
    assert len(session.calls) == 2
    assert len(list((tmp_path / "races").glob("*/official-requests/*/response-stage.json"))) == 3


def test_operational_race_failure_remains_an_error(tmp_path):
    _seed(tmp_path)

    report, _ = _run(tmp_path, "transport-failed", [_dt("10:06")] * 2, [])

    assert report["status"] == "COMPLETED_WITH_ERRORS"
    assert report["source_rejection_count"] == 0
    assert report["source_rejected_race_ids"] == []
    assert report["races"][0]["decision"] == "ERROR"
    assert report["races"][0]["source_rejection"] is None
    assert report["races"][0]["error"] == "StopIteration: "


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("identity", "rejected response-stage identity drift"),
        ("raw_hash", "rejected response-stage raw-byte hash drift"),
    ],
)
def test_retained_rejection_stage_drift_remains_an_error(
    tmp_path, mutation, expected_error
):
    _seed(tmp_path)
    url = "https://www.thedogs.com.au/racing/venue/2026-07-29/1/race-name?trial=false"
    malformed = b"<html>not a result</html>"
    first, _ = _run(
        tmp_path,
        "retained-stage-drift",
        [_dt("10:06")] * 4,
        [Response(malformed, url)],
    )
    assert first["races"][0]["decision"] == "SOURCE_REJECTED"
    stage_path = next(
        (tmp_path / "races").glob("*/official-requests/*/response-stage.json")
    )
    stage = json.loads(stage_path.read_text())
    if mutation == "identity":
        stage["collector_id"] = "different-collector"
    else:
        replacement = ForwardSealedCorpus(tmp_path).artifacts.put(
            b"<html>different rejected response</html>",
            media_type="application/octet-stream",
        )
        stage["raw_response_checksum"] = str(replacement.checksum)
    stage_path.write_bytes(observer.canonical_json(stage))

    report, _ = _run(
        tmp_path,
        "retained-stage-drift",
        [_dt("10:21")],
        [Response(malformed, url)],
    )

    assert report["status"] == "COMPLETED_WITH_ERRORS"
    assert report["source_rejection_count"] == 0
    assert report["races"][0]["decision"] == "ERROR"
    assert report["races"][0]["source_rejection"] is None
    assert report["races"][0]["error"] == f"ForwardCorpusRejected: {expected_error}"


def test_source_rejection_cannot_mask_missing_persistence_receipt(tmp_path):
    _seed(tmp_path)
    url = "https://www.thedogs.com.au/racing/venue/2026-07-29/1/race-name?trial=false"
    partial = T1._html().replace(b"2nd", b"-")

    class NonPersistingCorpus(ForwardSealedCorpus):
        def capture_result(self, **kwargs):
            raise ForwardCorpusRejected("synthetic persistence rejection")

    report, _ = _run(
        tmp_path,
        "source-and-persistence-rejected",
        [_dt("10:06")],
        [Response(partial, url)],
        corpus_factory=NonPersistingCorpus,
    )

    assert report["status"] == "COMPLETED_WITH_ERRORS"
    assert report["source_rejection_count"] == 0
    assert report["races"][0]["decision"] == "ERROR"
    assert report["races"][0]["source_rejection"] is None
    assert report["races"][0]["error"] == (
        "ForwardCorpusRejected: rejected response-stage receipt was not persisted"
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://www.thedogs.com.au/racing/x",
        "https://evil.test/racing/x",
        "https://www.thedogs.com.au/racing/x?legacy=1",
        "https://www.thedogs.com.au/racing/x/results",
    ],
)
def test_url_derivation_rejects_unsafe_or_ambiguous_identity(url):
    with pytest.raises(ValueError):
        observer.canonical_result_url(url)


def test_url_derivation_uses_established_non_trial_result_route():
    race_url = "https://www.thedogs.com.au/racing/venue/2026-07-29/1/race-name"
    assert observer.canonical_result_url(race_url) == race_url + "?trial=false"


def test_final_url_rejection_empty_response_and_lock_contention(tmp_path):
    _seed(tmp_path)
    expected = (
        "https://www.thedogs.com.au/racing/venue/2026-07-29/1/race-name?trial=false"
    )
    redirected, _ = _run(
        tmp_path,
        "redirected",
        [_dt("10:06")] * 2,
        [Response(T1._html(), expected + "/other")],
    )
    assert redirected["status"] == "COMPLETED_WITH_REJECTIONS"
    empty, _ = _run(
        tmp_path,
        "empty",
        [_dt("10:06")] * 4,
        [Response(b"", expected)],
    )
    assert empty["status"] == "COMPLETED_WITH_REJECTIONS"

    lock = tmp_path / "forward-sealed-corpus.lock"
    lock.write_text('{"owner":"other"}')
    session = Session([])
    busy = observer.observe_once(
        corpus_root=tmp_path,
        cycle_id="busy",
        clock=lambda: _dt("10:06"),
        session_factory=lambda: session,
    )
    assert busy["status"] == "LOCK_BUSY"
    assert session.calls == []
    assert lock.read_text() == '{"owner":"other"}'


def test_ids_and_report_are_deterministic_and_legacy_material_is_excluded(tmp_path):
    _seed(tmp_path)
    (tmp_path / "races" / "legacy").mkdir()
    (tmp_path / "races" / "legacy" / "legacy.json").write_text("{}")
    url = (
        "https://www.thedogs.com.au/racing/venue/2026-07-29/1/race-name?trial=false"
    )
    report, _ = _run(
        tmp_path,
        "deterministic",
        [_dt("10:06")] * 4,
        [Response(T1._html(), url)],
    )
    assert report["counts"]["excluded"] == 0
    assert len(report["attempted_race_ids"]) == len(
        {row["request_id"] for row in report["races"] if row["request_id"]}
    )
    assert report["session_id"] == observer._identity("session", "deterministic")
    assert report["run_id"] == observer._identity("run", "deterministic")


def test_redirect_is_not_followed_and_response_is_closed(tmp_path):
    _seed(tmp_path)
    url = (
        "https://www.thedogs.com.au/racing/venue/2026-07-29/1/race-name?trial=false"
    )
    response = Response(b"redirect", url, status=302, headers={"Location": url + "/other"})
    report, session = _run(tmp_path, "redirect", [_dt("10:06")] * 2, [response])
    assert report["status"] == "COMPLETED_WITH_REJECTIONS"
    assert len(session.calls) == 1
    assert session.calls[0]["headers"] == {
        **observer.THEDOGS_PUBLIC_HEADERS,
        "Accept-Encoding": "identity",
    }
    assert session.calls[0]["headers"] is not observer.OFFICIAL_RESULT_REQUEST_HEADERS
    assert session.calls[0]["allow_redirects"] is False
    assert session.calls[0]["stream"] is True
    assert response.closed


def test_wire_exact_body_encoding_and_bound_are_enforced(tmp_path, monkeypatch):
    _seed(tmp_path)
    url = (
        "https://www.thedogs.com.au/racing/venue/2026-07-29/1/race-name?trial=false"
    )
    encoded = Response(T1._html(), url, headers={"Content-Encoding": "gzip"})
    report, _ = _run(tmp_path, "encoded", [_dt("10:06")] * 2, [encoded])
    assert report["status"] == "COMPLETED_WITH_REJECTIONS"
    assert "unsupported Content-Encoding" in report["races"][0]["source_rejection"]
    assert encoded.closed
    repeated = Response(T1._html(), url, headers={"Content-Encoding": "gzip"})
    deferred, _ = _run(
        tmp_path,
        "encoded-repeated",
        [_dt("10:21")],
        [repeated],
        previous_rejection_deferrals=report["source_rejection_deferrals"],
    )
    assert deferred["races"][0]["decision"] == "SOURCE_REJECTED"
    assert deferred["races"][0]["normalization_attempted"] is False
    assert deferred["source_rejection_deferrals"] == []
    assert not list((tmp_path / "races").glob("*/official-requests/*/response-stage.json"))

    monkeypatch.setattr(observer, "MAX_RESPONSE_BYTES", 8)
    oversized = Response(b"123456789", url)
    report, _ = _run(tmp_path, "oversized", [_dt("10:06")] * 2, [oversized])
    assert report["status"] == "COMPLETED_WITH_ERRORS"
    assert report["races"][0]["source_rejection"] is None
    assert "maximum byte size" in report["races"][0]["error"]
    assert oversized.closed


def test_lock_release_never_unlinks_replacement(tmp_path):
    lock = tmp_path / "forward-sealed-corpus.lock"
    owned = observer._acquire_lock(lock, "first")
    displaced = tmp_path / "displaced.lock"
    lock.rename(displaced)
    replacement = b'{"owner":"successor"}'
    lock.write_bytes(replacement)
    assert observer._release_lock(lock, owned) is False
    assert lock.read_bytes() == replacement
    assert displaced.exists()
