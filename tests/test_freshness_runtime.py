import sys
from pathlib import Path

import pytest


def test_packaging_rejects_runtime_missing_native_readiness_dependency(tmp_path):
    from scripts.check_freshness_runtime import probe_runtime

    package = tmp_path / "race_collection"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "freshness_rehearsal.py").write_text(
        "import missing_readiness_dependency_for_offline_test\n"
    )
    with pytest.raises(ValueError, match="runtime_dependency_probe_failed"):
        probe_runtime(python=sys.executable, source_root=tmp_path)


def test_native_runtime_probe_pins_environment_and_rejects_changed_identity():
    from scripts.check_freshness_runtime import probe_runtime, verify_runtime
    from race_collection.live_freshness_contract import digest

    root = Path(__file__).resolve().parents[1]
    identity = probe_runtime(python=sys.executable, source_root=root)
    assert identity["prefix"] == sys.prefix
    assert identity["startup_observation"] == {
        "index_status": "UNAVAILABLE/DATA_MISSING",
        "collector_status": "UNAVAILABLE/DATA_MISSING",
        "authority_status": "AVAILABLE/FRESH",
    }
    assert "race_collection.freshness_rehearsal" in identity["modules"]
    assert any(row["name"].lower() == "flask" for row in identity["distributions"])
    plan = {"python": sys.executable, "source_root": str(root), "runtime_sha256": digest(identity)}
    assert verify_runtime(plan) == identity
    with pytest.raises(ValueError, match="runtime_environment_changed"):
        verify_runtime({**plan, "runtime_sha256": "0" * 64})


def test_executor_checks_runtime_before_touching_services(tmp_path, monkeypatch):
    import hashlib
    from datetime import datetime, timedelta, timezone
    from scripts import run_freshness_rehearsal as run
    from race_collection.live_freshness_contract import create_once, digest

    now = datetime(2026, 9, 23, 1, 45, tzinfo=timezone.utc)
    plan = {
        "python": sys.executable,
        "python_sha256": hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
        "source_root": str(tmp_path),
        "source_identity_sha256": "a" * 64,
        "admission_starts_at": (now - timedelta(minutes=15)).isoformat(),
        "starts_at": (now + timedelta(minutes=15)).isoformat(),
        "runtime_sha256": "b" * 64,
    }
    create_once(tmp_path / "plan.json", plan)
    monkeypatch.setattr(run, "now", lambda: now)
    monkeypatch.setattr(run, "verify_source_package", lambda *args: {})

    def incompatible(plan):
        raise ValueError("runtime_environment_changed")

    monkeypatch.setattr(run, "verify_runtime", incompatible)
    monkeypatch.setattr(run, "SystemdControl", lambda: pytest.fail("must not touch services"))
    with pytest.raises(ValueError, match="runtime_environment_changed"):
        run.execute(tmp_path / "plan.json", digest(plan), "synthetic-offline")
    assert (tmp_path / "started.json").exists()
