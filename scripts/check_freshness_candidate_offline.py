#!/usr/bin/env python3
"""Focused operational checks with network and retained-data reads denied.

Pass normal pytest node IDs/options. No top-level suite/conftest discovery.
Spawned multiprocessing tests re-enter this module and retain the audit guard.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def guard(event, args):
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise RuntimeError("offline test: network denied")
    if event in {"open", "sqlite3.connect"} and args and isinstance(args[0], (str, bytes)):
        path = os.fsdecode(args[0])
        if not path.startswith("/tmp/pytest-") and (
            "/tests/fixtures/thedogs" in path
            or "/artifacts/" in path
            or path.endswith("greyhound_racing_data.db")
        ):
            raise RuntimeError("offline test: retained data denied")


sys.addaudithook(guard)
if __name__ == "__main__":
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    import pytest

    raise SystemExit(pytest.main(["--noconftest", "-p", "no:cacheprovider", *sys.argv[1:]]))
