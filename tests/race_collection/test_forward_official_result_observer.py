import importlib.util
import json
from datetime import datetime
from pathlib import Path

import pytest

from race_collection.domain import ArtifactChecksum
from race_collection.forward_sealed_corpus import ForwardSealedCorpus
from scripts import observe_forward_official_results as observer


def _accepted_helpers():
    path = Path(__file__).with_name("test_forward_sealed_corpus.py")
    spec = importlib.util.spec_from_file_location("_accepted_t1_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


T1 = _accepted_helpers()


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

    def get(self, url, timeout, *, allow_redirects, stream):
        self.calls.append(
            {
                "url": url,
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
    if url is not None:
        value["canonical_source_url"] = url
    ForwardSealedCorpus(root, clock=lambda: _dt("09:45")).capture_prejump(**value)


def _run(root, cycle, times, responses):
    session = Session(responses)
    report = observer.observe_once(
        corpus_root=root,
        cycle_id=cycle,
        clock=Clock(*times),
        session_factory=lambda: session,
    )
    return report, session


def test_first_then_second_identical_closes_and_packages(tmp_path):
    _seed(tmp_path)
    url = "https://www.thedogs.com.au/racing/venue/2026-07-29/1/results"
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
    url = "https://www.thedogs.com.au/racing/venue/2026-07-29/1/results"
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

    url = "https://www.thedogs.com.au/racing/venue/2026-07-29/1/results"
    malformed, _ = _run(
        tmp_path,
        "malformed",
        [_dt("10:06")] * 4,
        [Response(b"<html>not a result</html>", url)],
    )
    assert malformed["status"] == "COMPLETED_WITH_ERRORS"
    assert malformed["races"][0]["raw_response_hash"]
    assert len(list((tmp_path / "races").glob("*/official-requests/*/response-stage.json"))) == 1


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


def test_final_url_rejection_empty_response_and_lock_contention(tmp_path):
    _seed(tmp_path)
    expected = "https://www.thedogs.com.au/racing/venue/2026-07-29/1/results"
    redirected, _ = _run(
        tmp_path,
        "redirected",
        [_dt("10:06")] * 2,
        [Response(T1._html(), expected + "/other")],
    )
    assert redirected["status"] == "COMPLETED_WITH_ERRORS"
    empty, _ = _run(
        tmp_path,
        "empty",
        [_dt("10:06")] * 4,
        [Response(b"", expected)],
    )
    assert empty["status"] == "COMPLETED_WITH_ERRORS"

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
    url = "https://www.thedogs.com.au/racing/venue/2026-07-29/1/results"
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
    url = "https://www.thedogs.com.au/racing/venue/2026-07-29/1/results"
    response = Response(b"redirect", url, status=302, headers={"Location": url + "/other"})
    report, session = _run(tmp_path, "redirect", [_dt("10:06")] * 2, [response])
    assert report["status"] == "COMPLETED_WITH_ERRORS"
    assert len(session.calls) == 1
    assert session.calls[0]["allow_redirects"] is False
    assert session.calls[0]["stream"] is True
    assert response.closed


def test_wire_exact_body_encoding_and_bound_are_enforced(tmp_path, monkeypatch):
    _seed(tmp_path)
    url = "https://www.thedogs.com.au/racing/venue/2026-07-29/1/results"
    encoded = Response(T1._html(), url, headers={"Content-Encoding": "gzip"})
    report, _ = _run(tmp_path, "encoded", [_dt("10:06")] * 2, [encoded])
    assert report["status"] == "COMPLETED_WITH_ERRORS"
    assert "unsupported Content-Encoding" in report["races"][0]["error"]
    assert encoded.closed

    monkeypatch.setattr(observer, "MAX_RESPONSE_BYTES", 8)
    oversized = Response(b"123456789", url)
    report, _ = _run(tmp_path, "oversized", [_dt("10:06")] * 2, [oversized])
    assert report["status"] == "COMPLETED_WITH_ERRORS"
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
