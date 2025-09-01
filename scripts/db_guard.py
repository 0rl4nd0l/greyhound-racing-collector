#!/usr/bin/env python3
"""
DB Guard utilities to enforce archive-first, validate-after patterns around DB writers.

Usage in a script that writes to the DB:

from scripts.db_guard import db_guard

with db_guard(db_path="greyhound_racing_data.db", label="ingest_new_races") as guard:
    # ... perform writes ...
    # Optionally register expected row deltas for reporting:
    guard.expect_table_growth("race_metadata", min_delta=0)
    guard.expect_table_growth("dog_race_data", min_delta=0)

The guard will:
- Create a timestamped backup using sqlite online backup
- Run PRAGMA integrity_check afterwards
- Emit a small row count delta report
- Exit non-zero if integrity_check fails
"""
from __future__ import annotations

import contextlib
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[1]


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _resolve_db_path(db_path: str | None) -> Path:
    if db_path:
        return Path(db_path)
    env = (
        os.getenv("GREYHOUND_DB_PATH")
        or os.getenv("DATABASE_PATH")
        or "greyhound_racing_data.db"
    )
    return Path(env)


def backup_db(db_path: Path, label: str) -> Path:
    ts = _now_ts()
    out_dir = REPO_ROOT / "archive" / "db_backups" / f"{ts}_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    backup_file = out_dir / "pre_op.sqlite"

    # Use sqlite online backup if available
    try:
        import subprocess

        subprocess.run(
            ["sqlite3", str(db_path), ".backup", str(backup_file)], check=False
        )
    except Exception:
        pass

    if not backup_file.exists() or backup_file.stat().st_size == 0:
        # Fallback copy
        import shutil

        shutil.copyfile(db_path, backup_file)

    return backup_file


def integrity_check(db_path: Path) -> bool:
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute("PRAGMA integrity_check;").fetchall()
        ok = len(rows) == 1 and rows[0][0] == "ok"
        return ok


def row_counts(db_path: Path, tables: list[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        for t in tables:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {t}")
                counts[t] = int(cur.fetchone()[0])
            except Exception:
                counts[t] = -1
    return counts


def _maybe_optimize(db_path: Path, mode_env: str | None = None) -> None:
    """Optionally run SQLite optimization post-write.
    Controlled via DB_GUARD_OPTIMIZE env var (optimize|analyze|vacuum).
    - optimize: PRAGMA optimize;
    - analyze: ANALYZE; PRAGMA optimize;
    - vacuum: VACUUM; ANALYZE; PRAGMA optimize;
    """
    mode = (mode_env or os.getenv("DB_GUARD_OPTIMIZE", "")).strip().lower()
    if mode not in ("optimize", "analyze", "vacuum"):
        return
    try:
        with sqlite3.connect(str(db_path)) as conn:
            cur = conn.cursor()
            if mode == "vacuum":
                print("[db_guard] VACUUM (opt-in) ...")
                cur.execute("VACUUM")
            if mode in ("analyze", "vacuum"):
                print("[db_guard] ANALYZE (opt-in) ...")
                cur.execute("ANALYZE")
            print("[db_guard] PRAGMA optimize (opt-in) ...")
            cur.execute("PRAGMA analysis_limit=400")
            try:
                cur.execute("PRAGMA optimize")
            except Exception:
                pass
    except Exception as e:
        print(f"[db_guard] optimize skipped: {e}")


@dataclass
class GuardState:
    db_path: Path
    label: str
    tables: list[str] = field(
        default_factory=lambda: [
            "race_metadata",
            "dog_race_data",
            "dogs",
            "prediction_history",
            "live_odds",
            "value_bets",
        ]
    )
    before_counts: Dict[str, int] = field(default_factory=dict)
    after_counts: Dict[str, int] = field(default_factory=dict)
    min_growth: Dict[str, int] = field(default_factory=dict)

    def expect_table_growth(self, table: str, min_delta: int = 0) -> None:
        self.min_growth[table] = min_delta


@contextlib.contextmanager
def db_guard(db_path: str | None = None, label: str = "op"):
    db_file = _resolve_db_path(db_path)
    gs = GuardState(db_path=db_file, label=label)

    # Pre-op snapshot and backup
    gs.before_counts = row_counts(gs.db_path, gs.tables)
    bfile = backup_db(gs.db_path, gs.label)
    print(f"[db_guard] Pre-op backup: {bfile}")

    try:
        yield gs
    finally:
        # Post-op integrity and counts
        ok = integrity_check(gs.db_path)
        gs.after_counts = row_counts(gs.db_path, gs.tables)

        print("[db_guard] Table deltas:")
        for t in gs.tables:
            before = gs.before_counts.get(t, -1)
            after = gs.after_counts.get(t, -1)
            if before >= 0 and after >= 0:
                delta = after - before
                need = gs.min_growth.get(t, None)
                extra = f" (min {need})" if need is not None else ""
                print(f"  - {t}: {before} -> {after} (delta {delta}){extra}")

        if not ok:
            raise SystemExit("[db_guard] ERROR: integrity_check failed after operation")

        # Optional optimization (opt-in via DB_GUARD_OPTIMIZE)
        _maybe_optimize(gs.db_path)
