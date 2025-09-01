#!/usr/bin/env python3
"""
Run any command under DB guard (pre-backup + post-validation + optional optimization).

Usage:
  python scripts/run_with_db_guard.py --db greyhound_racing_data.db --label myop -- <command> [args...]

Environment:
  DB_GUARD_OPTIMIZE=optimize|analyze|vacuum  # optional post-op DB optimization
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

# Ensure we can import the db_guard module
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.db_guard import db_guard  # type: ignore


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a command under DB guard")
    ap.add_argument("--db", required=False, default=os.getenv("GREYHOUND_DB_PATH") or os.getenv("DATABASE_PATH") or "greyhound_racing_data.db")
    ap.add_argument("--label", required=False, default="guarded")
    ap.add_argument("--", dest="cmdsep", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to run (after --)")
    ns = ap.parse_args()
    # If user didn't put --, we still accept command in ns.cmd
    return ns


def main() -> int:
    ns = parse_args()
    cmd = ns.cmd
    if not cmd:
        print("[guard-run] ERROR: No command provided. Usage: run_with_db_guard.py --db <db> --label <label> -- <cmd>")
        return 2

    # If the first arg is '--', strip it
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("[guard-run] ERROR: No command after --")
        return 2

    # Use db_guard around the subprocess
    with db_guard(db_path=ns.db, label=ns.label):
        print(f"[guard-run] Running: {' '.join(shlex.quote(c) for c in cmd)}")
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"[guard-run] Command failed with exit code {proc.returncode}")
            return proc.returncode
    print("[guard-run] Completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

