#!/usr/bin/env python3
"""Import the finite collector/readiness runtime offline and pin its environment.

No browser, collector or model is constructed. The native reader is exercised
against synthetic missing startup files. All writes stay in a disposable
directory; network, databases and retained data are denied before application imports.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def probe_runtime(*, python, source_root):
    with tempfile.TemporaryDirectory(prefix="freshness-runtime-") as scratch:
        result = subprocess.run(
            [str(python), "-B", str(Path(__file__).resolve()), "--source-root", str(source_root)],
            cwd=scratch,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            text=True,
            capture_output=True,
            timeout=30,
        )
    if result.returncode:
        raise ValueError("runtime_dependency_probe_failed: " + result.stderr[-2000:])
    return json.loads(result.stdout.splitlines()[-1])


def verify_runtime(plan):
    from race_collection.live_freshness_contract import digest

    identity = probe_runtime(python=plan["python"], source_root=plan["source_root"])
    if digest(identity) != plan["runtime_sha256"]:
        raise ValueError("runtime_environment_changed")
    return identity


def check_startup_observation(source_root, scratch):
    from datetime import datetime, timezone
    from race_collection.freshness_rehearsal import native_observation
    from scripts import shadow_autopilot_daemon as daemon
    from src.operator_ui.live_adapters import InstalledUnits

    now = datetime.now(timezone.utc)
    evidence = scratch / "startup-evidence"
    evidence.mkdir()
    common = {
        "repo_path": source_root,
        "timeout_seconds": 600,
        "live_freshness": True,
        "live_freshness_profile": "bounded80-v1",
        "live_freshness_contract": scratch / "absent-contract.json",
    }
    raw = {
        "full_service": daemon.service_file_text(**common).encode(),
        "odds_service": daemon.odds_capture_service_file_text(**common).encode(),
        "full_timer": daemon.timer_file_text().encode(),
        "odds_timer": daemon.odds_capture_timer_file_text(live_freshness=True).encode(),
    }
    hashes = {key: hashlib.sha256(value).hexdigest() for key, value in raw.items()}
    units = InstalledUnits(
        **raw,
        **{key + "_sha256": value for key, value in hashes.items()},
        observed_at=now,
        working_directory=str(source_root),
        full_unit_name="shadow-autopilot.service",
        odds_unit_name="shadow-autopilot-odds-capture.service",
        full_active_state="inactive",
        full_sub_state="dead",
        full_exec_main_pid=0,
        odds_active_state="inactive",
        odds_sub_state="dead",
        odds_exec_main_pid=0,
    )
    result = native_observation(
        now=now,
        paths={
            key: evidence / (key + ".json")
            for key in ("full_report", "full_state", "odds_report", "odds_state", "odds_refresh")
        },
        units=units,
        evidence_root=evidence,
        index_path=evidence / "absent-index.json",
        output=scratch / "startup-observation",
        authority={
            "commit": "b" * 40,
            "tree": "c" * 40,
            "source_root": str(source_root),
            "unit_sha256": hashes,
        },
    )
    statuses = {
        key: result[key] for key in ("index_status", "collector_status", "authority_status")
    }
    if statuses != {
        "index_status": "UNAVAILABLE/DATA_MISSING",
        "collector_status": "UNAVAILABLE/DATA_MISSING",
        "authority_status": "AVAILABLE/FRESH",
    }:
        raise ValueError("native_startup_observation_failed")
    return statuses


def runtime_identity(source_root):
    source_root = Path(source_root).resolve(strict=True)
    scratch = Path.cwd().resolve()

    def guard(event, args):
        if event in {"socket.connect", "socket.getaddrinfo", "sqlite3.connect", "subprocess.Popen"}:
            raise RuntimeError("runtime_probe_external_access_denied")
        if event == "open" and isinstance(args[0], (str, bytes)):
            path = Path(os.fsdecode(args[0])).absolute()
            if "artifacts" in path.parts or path.suffix in {".db", ".sqlite", ".sqlite3"}:
                raise RuntimeError("runtime_probe_retained_data_denied")
            flags = args[2] or 0
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT) and not path.is_relative_to(scratch):
                raise RuntimeError("runtime_probe_write_outside_scratch")

    sys.addaudithook(guard)
    sys.path.insert(0, str(source_root))
    modules = {}
    for name in (
        "race_collection.freshness_rehearsal",
        "scripts.shadow_autopilot_daemon",
        "scripts.refresh_prejump_upcoming",
        "scripts.autonomous_live_odds_capture",
        "upcoming_race_browser",
        "bs4",
        "playwright.sync_api",
        "selenium.webdriver",
    ):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        modules[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    from race_collection.live_execution import (
        installed_browser_binaries,
        require_profile_dependencies,
    )

    require_profile_dependencies()
    browser_binaries = installed_browser_binaries()
    distributions = sorted(
        (
            {
                "name": distribution.metadata["Name"],
                "version": distribution.version,
                "record_sha256": hashlib.sha256(
                    (distribution.read_text("RECORD") or "").encode()
                ).hexdigest(),
            }
            for distribution in importlib.metadata.distributions()
        ),
        key=lambda row: (row["name"], row["version"], row["record_sha256"]),
    )
    return {
        "schema_version": "freshness_runtime_identity_v1",
        "executable": sys.executable,
        "prefix": sys.prefix,
        "version": sys.version,
        "modules": modules,
        "browser_binaries": browser_binaries,
        "distributions": distributions,
        "startup_observation": check_startup_observation(source_root, scratch),
        "scope": "Imports, distribution records and synthetic native startup observation; no acquisition or browser launch",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(runtime_identity(args.source_root), sort_keys=True))
