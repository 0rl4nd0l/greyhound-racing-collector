"""Offline publication interleavings; no provider, database or result reads."""

import hashlib
import json
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from race_collection.live_phase_checkpoint import atomic_json, native_publication_lock


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("network forbidden in publication fixtures")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)


@pytest.fixture
def native_fixture(tmp_path):
    from race_collection.freshness_rehearsal import native_observation
    from scripts import shadow_autopilot_daemon as daemon
    from src.operator_ui.live_adapters import InstalledUnits
    from tests.operator_ui.test_live_adapters import actual_payloads

    now = datetime(2026, 9, 24, 6, tzinfo=timezone.utc)
    values = actual_payloads(now, include_models=False)
    for lane in ("full", "odds"):
        values[lane + "_state"]["last_output_dir" if lane == "full" else "output_dir"] = str(tmp_path / lane)
        values[lane + "_report"]["output_dir"] = str(tmp_path / lane)
    for key in ("odds_report", "odds_state"):
        values[key]["autopilot_output_dir"] = str(tmp_path / "autopilot")
    paths = {}
    for key in ("full_report", "full_state", "odds_report", "odds_state", "odds_refresh"):
        path = tmp_path / ("autopilot/odds_capture_refresh_report.json" if key == "odds_refresh" else key + ".json")
        atomic_json(path, values[key])
        paths[key] = path
    raw = dict(
        full_service=daemon.service_file_text(repo_path=tmp_path, timeout_seconds=600).encode(),
        odds_service=daemon.odds_capture_service_file_text(repo_path=tmp_path, timeout_seconds=600).encode(),
        full_timer=daemon.timer_file_text().encode(),
        odds_timer=daemon.odds_capture_timer_file_text(live_freshness=True).encode(),
    )
    hashes = {key: hashlib.sha256(value).hexdigest() for key, value in raw.items()}
    units = InstalledUnits(
        **raw, **{key + "_sha256": value for key, value in hashes.items()},
        observed_at=now, working_directory=str(tmp_path),
        full_unit_name="shadow-autopilot.service", odds_unit_name="shadow-autopilot-odds-capture.service",
        full_active_state="inactive", full_sub_state="dead", full_exec_main_pid=0,
        odds_active_state="inactive", odds_sub_state="dead", odds_exec_main_pid=0,
    )

    def observe():
        return native_observation(
            now=now, paths=paths, units=units, evidence_root=tmp_path,
            index_path=tmp_path / "absent-index.json", output=tmp_path / "measurement",
            authority={"commit": "b" * 40, "tree": "c" * 40,
                       "source_root": str(tmp_path), "unit_sha256": hashes},
        )

    return values, paths, now, observe


def shared_contention_event(monkeypatch):
    """Signal an actual flock conflict instead of assuming thread scheduling."""
    import race_collection.live_phase_checkpoint as checkpoint

    blocked = threading.Event()
    original = checkpoint.fcntl.flock

    def observed(descriptor, operation):
        try:
            return original(descriptor, operation)
        except BlockingIOError:
            if operation & checkpoint.fcntl.LOCK_SH:
                blocked.set()
            raise

    monkeypatch.setattr(checkpoint.fcntl, "flock", observed)
    return blocked


def test_sample_waits_between_report_and_state_replacement(tmp_path, monkeypatch, native_fixture):
    from scripts import run_freshness_rehearsal as run

    values, paths, now, observe = native_fixture
    assert observe()["collector_status"] == "AVAILABLE/FRESH"
    entered = threading.Event()
    blocked = shared_contention_event(monkeypatch)

    def sample_locked(*args):
        entered.set()
        result = observe()
        result.update(read_start=now.isoformat(), read_end=now.isoformat(),
                      monotonic_start=time.monotonic(), monotonic_end=time.monotonic())
        return result

    monkeypatch.setattr(run, "_sample_status", lambda control: None)
    monkeypatch.setattr(run, "_sample_locked", sample_locked)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with native_publication_lock(tmp_path, exclusive=True):
            values["odds_report"]["run_id"] = "odds-next"
            atomic_json(paths["odds_report"], values["odds_report"])
            assert observe()["collector_status"] != "AVAILABLE/FRESH"
            reader = pool.submit(run._sample_once, {"evidence_root": str(tmp_path)}, tmp_path, None)
            assert blocked.wait(5), "reader never reached the publication lock"
            assert not entered.is_set()
            values["odds_state"]["run_id"] = "odds-next"
            atomic_json(paths["odds_state"], values["odds_state"])
        result = reader.result(timeout=5)
    assert result["collector_status"] == "AVAILABLE/FRESH"
    assert result["lanes"][1]["run_id"] == "odds-next"
    assert result["publication_lock_wait_seconds"] > 0


def test_index_publication_holds_lock_through_lifecycle_and_event(tmp_path, monkeypatch):
    from scripts import shadow_autopilot_v1 as autopilot

    runtime = tmp_path / "shadow_autopilot_daemon_runtime"
    output = tmp_path / "refresh"
    output.mkdir()
    paused, release = threading.Event(), threading.Event()
    blocked = shared_contention_event(monkeypatch)
    publication = {"status": "PUBLISHED", "packet_sha256": "a" * 64,
                   "source_generated_at": "2026-09-24T06:00:00+00:00"}

    def publish(**kwargs):
        atomic_json(runtime / "index.json", {"generation": 2})
        paused.set()
        assert release.wait(5), "test did not release publication"
        return publication

    def lifecycle(**kwargs):
        atomic_json(runtime / "lifecycle.json", {"generation": 2})

    def read():
        with native_publication_lock(tmp_path, exclusive=False):
            return (json.loads((runtime / "index.json").read_bytes()),
                    json.loads((runtime / "lifecycle.json").read_bytes()),
                    json.loads((runtime / "live-publication-events/000000.json").read_bytes()))

    monkeypatch.setattr(autopilot, "publish_current_race_index", publish)
    monkeypatch.setattr(autopilot, "publish_current_race_index_lifecycle", lifecycle)
    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(autopilot.publish_current_race_index_after_refresh,
            state_path=runtime / "odds_capture_state.json", evidence_root=tmp_path,
            output_dir=output, run_id="synthetic", source_refresh_report_path=output / "source.json",
            enforce_monotonic=True)
        try:
            assert paused.wait(5)
            reader = pool.submit(read)
            assert blocked.wait(5), "index reader was not excluded during publication"
        finally:
            release.set()
        assert writer.result(timeout=5) == publication
        index, state, event = reader.result(timeout=5)
    assert index == state == {"generation": 2}
    assert event["packet_sha256"] == publication["packet_sha256"]


def test_shared_readers_timeout_and_exception_release(tmp_path):
    with ThreadPoolExecutor(max_workers=1) as pool:
        def acquire(exclusive):
            with native_publication_lock(tmp_path, exclusive=exclusive, timeout_seconds=.05):
                return "acquired"

        with native_publication_lock(tmp_path, exclusive=False):
            assert pool.submit(acquire, False).result(timeout=5) == "acquired"
            with pytest.raises(TimeoutError, match="native_publication_lock_timeout"):
                pool.submit(acquire, True).result(timeout=5)
        with pytest.raises(RuntimeError, match="publisher_failed"):
            with native_publication_lock(tmp_path, exclusive=True):
                raise RuntimeError("publisher_failed")
        assert pool.submit(acquire, True).result(timeout=5) == "acquired"


@pytest.mark.parametrize("invalid", ["corrupt", "future"])
def test_publication_lock_does_not_make_invalid_evidence_valid(tmp_path, native_fixture, invalid):
    values, paths, now, observe = native_fixture
    with native_publication_lock(tmp_path, exclusive=True):
        if invalid == "corrupt":
            paths["odds_report"].write_text("not JSON")
        else:
            values["odds_report"]["generated_at"] = (now + timedelta(seconds=1)).isoformat()
            atomic_json(paths["odds_report"], values["odds_report"])
    with native_publication_lock(tmp_path, exclusive=False):
        if invalid == "corrupt":
            with pytest.raises(json.JSONDecodeError):
                observe()
        else:
            assert observe()["collector_status"] == "INVALID/INTEGRITY_FAILED"


def test_service_probes_do_not_hold_publication_lock(tmp_path, monkeypatch):
    from scripts import run_freshness_rehearsal as run

    def probe(control):
        # An independent writer can publish even while systemd is being queried.
        with native_publication_lock(tmp_path, exclusive=True, timeout_seconds=.05):
            atomic_json(tmp_path / "probe-publication.json", {"generation": 1})
        return ("snapshot",)

    def sample(plan, output, snapshot):
        assert snapshot == ("snapshot",)
        return {"monotonic_end": time.monotonic(), "read_end": run.now().isoformat()}

    monkeypatch.setattr(run, "_sample_status", probe)
    monkeypatch.setattr(run, "_sample_locked", sample)
    result = run._sample_once({"evidence_root": str(tmp_path)}, tmp_path, None)
    assert result["status_probe_seconds"] >= 0
    assert result["monotonic_end"] >= result["monotonic_start"]
