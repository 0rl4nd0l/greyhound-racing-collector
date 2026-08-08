from __future__ import annotations

import base64
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.predictor.manual_independent_capture import (
    PROTECTED_PATH_KEYS,
    canonical_sha256,
)
from src.predictor.manual_independent_capture_executor import (
    execute_manual_capture_fixture,
)
from src.predictor.manual_independent_capture_sealer import (
    build_sealing_identity,
    expectations_from_execution,
    seal_manual_capture,
)
from src.predictor.manual_live_capture_child import (
    CONTENT_TYPE,
    LiveCaptureRejected,
    _install_navigation_guard,
    capture_from_page,
)

NOW = datetime(2026, 8, 5, 1, 0, 0, tzinfo=timezone.utc)
SOURCE_COMMIT = "e4f3699986237aad265b34e77d06d536f6046ee4"
SOURCE_TREE = "ac9cdde82d3a0ede953e36c9a87d9afd216c5826"


class _Response:
    def __init__(self, status: int = 200):
        self.status = status


class _Locator:
    def __init__(self, *, rows=None, count: int = 0):
        self.rows = rows
        self._count = count

    def count(self) -> int:
        return self._count

    def evaluate_all(self, _script: str):
        if self.rows is None:
            raise AssertionError("unexpected evaluation")
        return self.rows


class _Page:
    def __init__(
        self,
        rows,
        *,
        final_url: str,
        status: int = 200,
        challenge: bool = False,
        outcome: bool = False,
        title: str = "Race",
    ):
        self.rows = rows
        self.url = final_url
        self.status = status
        self.challenge = challenge
        self.outcome = outcome
        self.page_title = title
        self.goto_urls: list[str] = []

    def goto(self, url: str, **_kwargs):
        self.goto_urls.append(url)
        return _Response(self.status)

    def title(self) -> str:
        return self.page_title

    def locator(self, selector: str):
        if selector == "tr.race-runner":
            return _Locator(rows=self.rows)
        if selector in {"input[type='password']", "[data-captcha]", "iframe[src*='challenge']", ".cf-challenge", "#challenge-running"}:
            return _Locator(count=int(self.challenge))
        if selector in {"[data-result]", "[data-outcome]", "[data-finishing-position]", ".race-result", ".race-results", ".race-result__winner"}:
            return _Locator(count=int(self.outcome))
        raise AssertionError(f"unexpected selector: {selector}")


class _TimeoutPage(_Page):
    def goto(self, _url: str, **_kwargs):
        self.goto_urls.append(_url)
        raise TimeoutError("mock timeout")


class _Request:
    def __init__(self, url: str, navigation: bool):
        self.url = url
        self.navigation = navigation

    def is_navigation_request(self) -> bool:
        return self.navigation


class _Route:
    def __init__(self, url: str, navigation: bool):
        self.request = _Request(url, navigation)
        self.action = None

    def abort(self):
        self.action = "abort"

    def continue_(self):
        self.action = "continue"


class _RoutePage:
    def __init__(self):
        self.guard = None

    def route(self, _pattern, guard):
        self.guard = guard


def _url() -> str:
    return "https://www.thedogs.com.au/racing/richmond/2026-08-05/1/race-name"


def _race(scheduled: datetime | None = None) -> dict:
    scheduled = scheduled or NOW + timedelta(hours=1)
    return {
        "url": _url(),
        "race_id": "Race 1 - RICH - 2026-08-05",
        "race_date": "2026-08-05",
        "venue": "RICH",
        "venue_slug": "richmond",
        "race_number": 1,
        "scheduled_start": scheduled.isoformat(),
    }


def _rows() -> list[dict]:
    return [
        {
            "box_number": 1,
            "display_name": "Fast Dog",
            "source_native_runner_id": "native-1",
            "win_odds": "2.50",
        },
        {
            "box_number": 2,
            "display_name": "Quick Dog",
            "source_native_runner_id": "native-2",
            "win_odds": "3.10",
        },
    ]


def _readiness_rows() -> list[dict]:
    return [
        {
            "runner_id": "Race 1 - RICH - 2026-08-05|box:1|dog:FASTDOG",
            "box_number": 1,
            "dog_name": "Fast Dog",
            "source_native_runner_id": "native-1",
        },
        {
            "runner_id": "Race 1 - RICH - 2026-08-05|box:2|dog:QUICKDOG",
            "box_number": 2,
            "dog_name": "Quick Dog",
            "source_native_runner_id": "native-2",
        },
    ]


def _config(tmp_path: Path) -> dict:
    operations = tmp_path / "operations"
    operations.mkdir()
    manual = operations / "manual-independent-capture-v1"
    return {
        "schema_version": "manual_independent_capture_config_v1",
        "contract_version": "manual-independent-capture-v1",
        "safety": {
            "research_only": True,
            "canonical": False,
            "phase7_excluded": True,
            "phase7_eligible": False,
            "phase7_exclusion_reason": "manual_research_only_noncanonical",
        },
        "authority_profile": "manual_independent_capture_research_only_v1",
        "paths": {
            "operations_root": str(operations),
            "manual_root": str(manual),
            "browser_profile": str(manual / "browser-profile"),
            "runs_root": str(manual / "runs"),
            "manual_lock": str(manual / "manual-capture.lock"),
        },
        "timing": {
            "minimum_prejump_margin_seconds": 120,
            "hard_timeout_seconds": 5,
            "cancellation_grace_seconds": 1,
        },
        "attempt_policy": {
            "max_concurrent_manual_runs": 1,
            "max_capture_attempts": 1,
            "retries_allowed": False,
            "replay_allowed": False,
        },
    }


def _forbidden(tmp_path: Path) -> dict[str, str]:
    root = tmp_path / "protected"
    root.mkdir()
    return {name: str(root / name) for name in PROTECTED_PATH_KEYS}


def test_exact_url_live_fixture_record_is_strict_and_single_attempt():
    page = _Page(_rows(), final_url=_url())
    record = capture_from_page(
        page,
        exact_url=_url(),
        race_id=_race()["race_id"],
        race_identity_sha256=canonical_sha256(_race()),
        now=lambda: NOW,
    )
    assert page.goto_urls == [_url()]
    assert record["schema_version"] == "manual_independent_capture_child_live_v1"
    assert record["source"]["content_type"] == CONTENT_TYPE
    assert json.loads(base64.b64decode(record["source"]["bytes_base64"])) == {
        "runners": [
            {"box_number": 1, "display_name": "Fast Dog", "decimal_odds": 2.5},
            {"box_number": 2, "display_name": "Quick Dog", "decimal_odds": 3.1},
        ]
    }


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"final_url": "https://www.thedogs.com.au/racing/richmond/2026-08-05/2/other"}, "IDENTITY_MISMATCH"),
        ({"status": 503}, "SOURCE_ATTEMPT_FAILED"),
        ({"challenge": True}, "SOURCE_CHALLENGE"),
        ({"outcome": True}, "OUTCOME_MATERIAL_FORBIDDEN"),
        ({"rows": [{"box_number": 1, "display_name": "Fast Dog", "source_native_runner_id": None, "win_odds": "2.00", "unexpected": True}]}, "SOURCE_MALFORMED"),
        ({"rows": [{"box_number": 1, "display_name": "Fast Dog", "source_native_runner_id": None, "win_odds": ""}]}, "ODDS_INVALID"),
        ({"rows": [{"box_number": 1, "display_name": "Fast Dog", "source_native_runner_id": None, "win_odds": "1.00"}]}, "ODDS_INVALID"),
        ({"rows": [{"box_number": 2, "display_name": "Fast Dog", "source_native_runner_id": None, "win_odds": "2.00"}, {"box_number": 2, "display_name": "Quick Dog", "source_native_runner_id": None, "win_odds": "3.00"}]}, "RUNNER_SET_MISMATCH"),
    ],
)
def test_live_child_failures_are_terminal_without_fallback(kwargs, code):
    values = {"rows": _rows(), "final_url": _url()}
    values.update(kwargs)
    page = _Page(**values)
    with pytest.raises(LiveCaptureRejected, match=code):
        capture_from_page(
            page,
            exact_url=_url(),
            race_id=_race()["race_id"],
            race_identity_sha256=canonical_sha256(_race()),
            now=lambda: NOW,
        )
    assert page.goto_urls == [_url()]


def test_live_child_timeout_is_one_terminal_attempt():
    page = _TimeoutPage(_rows(), final_url=_url())
    with pytest.raises(LiveCaptureRejected, match="SOURCE_ATTEMPT_FAILED"):
        capture_from_page(
            page,
            exact_url=_url(),
            race_id=_race()["race_id"],
            race_identity_sha256=canonical_sha256(_race()),
            now=lambda: NOW,
        )
    assert page.goto_urls == [_url()]


def test_navigation_guard_blocks_redirects_and_outcome_subresources():
    page = _RoutePage()
    _install_navigation_guard(page, _url())
    exact = _Route(_url(), True)
    redirect = _Route("https://www.thedogs.com.au/results/richmond", True)
    outcome_api = _Route("https://www.thedogs.com.au/api/outcome/1", False)
    asset = _Route("https://www.thedogs.com.au/assets/race.css", False)
    for route in (exact, redirect, outcome_api, asset):
        page.guard(route)
    assert [route.action for route in (exact, redirect, outcome_api, asset)] == [
        "continue",
        "abort",
        "abort",
        "continue",
    ]


def test_live_output_passes_unchanged_ghu051_parent_runner_binding_and_ghu052_sealing(tmp_path: Path):
    race = _race()
    readiness = _readiness_rows()
    config = _config(tmp_path)
    forbidden = _forbidden(tmp_path)
    record = capture_from_page(
        _Page(_rows(), final_url=_url()),
        exact_url=_url(),
        race_id=race["race_id"],
        race_identity_sha256=canonical_sha256(race),
        now=lambda: NOW,
    )
    record["race_identity_sha256"] = canonical_sha256(race)
    raw = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    result = execute_manual_capture_fixture(
        config=config,
        forbidden_paths=forbidden,
        requested_race_url=_url(),
        selected_race=race,
        expected_runner_set=readiness,
        model_bytes=b"fixture-model",
        source_commit=SOURCE_COMMIT,
        source_tree=SOURCE_TREE,
        fixture_child_command=lambda _launch: [
            sys.executable,
            "-c",
            "import sys; sys.stdout.buffer.write(" + repr(raw) + ")",
        ],
        now=lambda: NOW,
    )
    assert result.artifact["terminal"] == {"status": "CAPTURE_READY", "failure_code": None}
    assert result.cleanup.confirmed is True
    assert result.cleanup.process_group_absent is True
    assert result.artifact["attempt"] == {"attempt_count": 1, "source_attempt_count": 1}
    assert result.source_response.content_type == CONTENT_TYPE
    expected = expectations_from_execution(result)
    sealed = seal_manual_capture(
        result,
            config=config,
            forbidden_paths=forbidden,
        expected=expected,
        identity=build_sealing_identity(
            repo_root=Path(__file__).parents[1],
            source_commit=SOURCE_COMMIT,
            source_tree=SOURCE_TREE,
        ),
        repo_root=Path(__file__).parents[1],
    )
    assert sealed.bundle["source"]["odds_parser"] == "manual_live_json_odds_v1"
    assert sealed.bundle["safety"] == {
        "research_only": True,
        "canonical": False,
        "phase7_excluded": True,
        "phase7_eligible": False,
        "phase7_exclusion_reason": "manual_research_only_noncanonical",
    }
