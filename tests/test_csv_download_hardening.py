import io
import os
import sys
import types
import tempfile
import json

import pytest


@pytest.fixture(autouse=True)
def _isolate_upcoming_dir(monkeypatch):
    # Use a temp directory for UPCOMING_RACES_DIR in these tests
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("UPCOMING_RACES_DIR", tmpdir)
        yield tmpdir


def test_normalize_race_url_strips_expert_form_and_fragments():
    from upcoming_race_browser import UpcomingRaceBrowser

    br = UpcomingRaceBrowser()
    base, expert = br._normalize_race_url(
        "https://www.thedogs.com.au/racing/grafton/2025-09-02/5/expert-form?foo=1#frag"
    )
    assert base.endswith("/racing/grafton/2025-09-02/5")
    assert expert.endswith("/racing/grafton/2025-09-02/5/expert-form")

    # Duplicated expert-form segments are collapsed
    base2, expert2 = br._normalize_race_url(
        "https://www.thedogs.com.au/racing/angle-park/2025-09-02/1/expert-form/expert-form/"
    )
    assert base2.endswith("/racing/angle-park/2025-09-02/1")
    assert expert2.endswith("/racing/angle-park/2025-09-02/1/expert-form")


def test_download_rejects_html_masquerading_as_csv(monkeypatch, _isolate_upcoming_dir):
    """download_race_csv should NOT save files when the body is HTML."""
    from upcoming_race_browser import UpcomingRaceBrowser

    br = UpcomingRaceBrowser()

    # Patch link finder to return direct CSV content which is actually HTML
    def _fake_find_csv_download_link(soup, race_url):
        return {"type": "direct_csv", "data": "<html><body>Not CSV</body></html>"}

    monkeypatch.setattr(br, "find_csv_download_link", _fake_find_csv_download_link)

    # Stub network GET for the race page to avoid external calls
    class _Resp:
        def __init__(self, status_code=200, text="", content=b""):
            self.status_code = status_code
            self.text = text
            self.content = content
            self.headers = {}
        def close(self):
            pass
    
    fake_session = types.SimpleNamespace(
        get=lambda url, timeout=30: _Resp(200, content=b"<html><title>Race 5</title></html>")
    )
    monkeypatch.setattr(br, "session", fake_session)

    res = br.download_race_csv("https://www.thedogs.com.au/racing/grafton/2025-09-02/5")
    assert not res.get("success"), f"Expected failure, got: {res}"
    # Ensure nothing was written (no filepath in result)
    assert "filepath" not in res


def test_download_accepts_valid_csv_and_writes_file(monkeypatch, _isolate_upcoming_dir):
    from upcoming_race_browser import UpcomingRaceBrowser

    br = UpcomingRaceBrowser()

    # Minimal plausible CSV header + one row
    csv_content = "\n".join(
        [
            "Dog Name,Box",
            "1. Alpha,1",
            "2. Bravo,2",
            "3. Charlie,3",
            "4. Delta,4",
        ]
    )

    def _fake_find_csv_download_link(soup, race_url):
        return {"type": "direct_csv", "data": csv_content}

    monkeypatch.setattr(br, "find_csv_download_link", _fake_find_csv_download_link)

    # Stub network GET for the race page to avoid external calls
    class _Resp:
        def __init__(self, status_code=200, text="", content=b""):
            self.status_code = status_code
            self.text = text
            self.content = content
            self.headers = {}
        def close(self):
            pass

    fake_session = types.SimpleNamespace(
        get=lambda url, timeout=30: _Resp(200, content=b"<html><title>Race 5</title></html>")
    )
    monkeypatch.setattr(br, "session", fake_session)

    res = br.download_race_csv("https://www.thedogs.com.au/racing/grafton/2025-09-02/5")
    assert res.get("success"), f"Expected success, got: {res}"
    fp = res.get("filepath")
    assert fp and os.path.exists(fp)
    with open(fp, "r", encoding="utf-8") as f:
        data = f.read()
    assert "Dog Name,Box" in data
    assert res["runner_completeness"]["status"] == "COMPLETE"
    assert os.path.exists(f"{fp}.metadata.json")


def test_download_rejects_partial_runner_set(monkeypatch, _isolate_upcoming_dir):
    from upcoming_race_browser import UpcomingRaceBrowser

    br = UpcomingRaceBrowser()
    csv_content = "Dog Name,Box\n2. Shima Lexie,2\n4. Sekiro,4\n"

    def _fake_find_csv_download_link(soup, race_url):
        return {"type": "direct_csv", "data": csv_content}

    monkeypatch.setattr(br, "find_csv_download_link", _fake_find_csv_download_link)

    class _Resp:
        def __init__(self, status_code=200, text="", content=b""):
            self.status_code = status_code
            self.text = text
            self.content = content
            self.headers = {}

        def close(self):
            pass

    fake_session = types.SimpleNamespace(
        get=lambda url, timeout=30: _Resp(200, content=b"<html><title>Race 5</title></html>")
    )
    monkeypatch.setattr(br, "session", fake_session)

    res = br.download_race_csv("https://www.thedogs.com.au/racing/grafton/2025-09-02/5")

    assert res.get("success") is not True
    assert res["error"] == "Incomplete runner set in downloaded CSV"
    assert res["runner_completeness"]["status"] == "INCOMPLETE"
    assert os.path.exists(res["quarantine_path"])


@pytest.fixture
def flask_client():
    # Importing app initializes the Flask app
    from app import app as flask_app

    flask_app.config.update({"TESTING": True})
    with flask_app.test_client() as client:
        yield client


def test_api_download_quarantines_html_file(monkeypatch, flask_client, _isolate_upcoming_dir):
    """
    The /api/download_upcoming_race endpoint should 502 if the saved file contains HTML
    and the guardian flags it for quarantine.
    """
    # Create a fake browser that pretends to download a file by writing HTML
    class FakeBrowser:
        def __init__(self, *a, **kw):
            pass

        def download_race_csv(self, race_url):
            # Write an HTML file into upcoming dir
            upcoming = os.environ.get("UPCOMING_RACES_DIR")
            fn = "Race 5 - GRAFTON - 2025-09-02.csv"
            fp = os.path.join(upcoming, fn)
            with open(fp, "w", encoding="utf-8") as f:
                f.write("<html><body>oops</body></html>")
            return {"success": True, "filename": fn, "filepath": fp}

    # Ensure the endpoint imports our fake browser implementation at call time
    fake_upcoming = types.ModuleType("upcoming_race_browser")
    fake_upcoming.UpcomingRaceBrowser = FakeBrowser
    sys.modules["upcoming_race_browser"] = fake_upcoming

    # Patch the guardian at import site to always quarantine
    # Create a proper fake module so `from utils.file_integrity_guardian import FileIntegrityGuardian` resolves correctly
    if "utils" not in sys.modules:
        sys.modules["utils"] = types.ModuleType("utils")
    fake_mod = types.ModuleType("utils.file_integrity_guardian")

    class _VR:
        def __init__(self):
            self.should_quarantine = True
            self.issues = ["CSV file contains HTML content"]

    class _Guardian:
        def validate_file(self, path):
            return _VR()

    fake_mod.FileIntegrityGuardian = _Guardian

    sys.modules["utils.file_integrity_guardian"] = fake_mod

    resp = flask_client.post(
        "/api/download_upcoming_race",
        data=json.dumps({"race_url": "https://www.thedogs.com.au/racing/grafton/2025-09-02/5"}),
        content_type="application/json",
    )
    assert resp.status_code == 502
    data = resp.get_json()
    assert data and data.get("error") == "Upstream returned HTML instead of CSV"


def test_api_download_success_passthrough(monkeypatch, flask_client, _isolate_upcoming_dir):
    """Happy path: guardian allows file, endpoint returns success JSON."""
    # Fake browser writes a valid CSV file
    class FakeBrowserOK:
        def __init__(self, *a, **kw):
            pass

        def download_race_csv(self, race_url):
            upcoming = os.environ.get("UPCOMING_RACES_DIR")
            fn = "Race 2 - GRAFTON - 2025-09-02.csv"
            fp = os.path.join(upcoming, fn)
            with open(fp, "w", encoding="utf-8") as f:
                f.write("Dog Name,Box\nRunner,1\n")
            return {"success": True, "filename": fn, "filepath": fp}

    # Ensure the endpoint imports our fake browser implementation at call time
    fake_upcoming_ok = types.ModuleType("upcoming_race_browser")
    fake_upcoming_ok.UpcomingRaceBrowser = FakeBrowserOK
    sys.modules["upcoming_race_browser"] = fake_upcoming_ok

    # Patch guardian to allow
    if "utils" not in sys.modules:
        sys.modules["utils"] = types.ModuleType("utils")
    fake_mod_ok = types.ModuleType("utils.file_integrity_guardian")

    class _VR2:
        def __init__(self):
            self.should_quarantine = False
            self.issues = []

    class _GuardianOK:
        def validate_file(self, path):
            return _VR2()

    fake_mod_ok.FileIntegrityGuardian = _GuardianOK
    sys.modules["utils.file_integrity_guardian"] = fake_mod_ok

    resp = flask_client.post(
        "/api/download_upcoming_race",
        data=json.dumps({"race_url": "https://www.thedogs.com.au/racing/grafton/2025-09-02/2"}),
        content_type="application/json",
    )
    assert resp.status_code == 200, resp.data
    payload = resp.get_json()
    assert payload and payload.get("success") is True
    assert payload.get("filename", "").endswith(".csv")
