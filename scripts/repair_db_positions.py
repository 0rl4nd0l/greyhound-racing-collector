#!/usr/bin/env python3
"""
DB Repair Utility: normalize names, compact finish positions, and reconcile field_size.

What it does (safely, with backup):
- Backup the target SQLite DB to archive/db_repairs/<TS>/pre_fix.sqlite
- Normalize dog_race_data.dog_clean_name and race_metadata.winner_name by removing leading
  "N. " prefixes (e.g., "5. Mary Poppins" -> "Mary Poppins").
- Compact finish positions per race when it is safe to do so:
  - Only for races with c>=3, exactly one winner (finish_position=1), and no duplicates (udist==c)
  - Re-map sorted unique positions to a compact 1..K sequence, preserving the winner at 1
  - Skip races with duplicate positions or missing winners
- Recompute race_metadata.field_size from actual dog_race_data row counts
- Emit a JSON repair report under debug_artifacts/v4/repair_report.json

Usage:
  python scripts/repair_db_positions.py [--db /path/to.db] [--apply]

Defaults:
  --db defaults to ./greyhound_racing_data.db
  Without --apply, runs in dry-run mode (no writes), printing a summary

Notes:
  - This script intentionally does not alter races with duplicate positions or no clear winner
  - It does not attempt to infer winners or break ties; that requires source-of-truth scraping
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Route writable connections via staging/default
try:
    from scripts.db_utils import open_sqlite_writable
except Exception:
    def open_sqlite_writable(db_path: str | None = None):
        import os as _os, sqlite3 as _sqlite3
        path = str(db_path)
        return _sqlite3.connect(path)

RE_PREFIX = re.compile(r"^\s*\d+\.\s*")


def strip_prefix(name: str | None) -> str | None:
    if name is None:
        return None
    s = str(name)
    return RE_PREFIX.sub("", s).strip()


def backup_db(db_path: Path) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    archive_dir = Path("archive") / "db_repairs" / ts
    archive_dir.mkdir(parents=True, exist_ok=True)
    dest = archive_dir / "pre_fix.sqlite"
    try:
        # Attempt online backup via sqlite3 CLI
        import subprocess

        subprocess.run(["sqlite3", str(db_path), ".backup", str(dest)], check=False)
        if dest.exists() and dest.stat().st_size > 0:
            return dest
    except Exception:
        pass
    # Fallback: file copy
    shutil.copyfile(db_path, dest)
    return dest


def get_race_stats(conn: sqlite3.Connection) -> List[Tuple[str, int, int, int, int, int]]:
    sql = (
        "SELECT race_id, COUNT(*) as c, "
        "COUNT(DISTINCT finish_position) as udist, "
        "COALESCE(MIN(finish_position),0) as minp, "
        "COALESCE(MAX(finish_position),0) as maxp, "
        "SUM(CASE WHEN finish_position=1 THEN 1 ELSE 0 END) as winners "
        "FROM dog_race_data WHERE race_id IS NOT NULL "
        "GROUP BY race_id"
    )
    cur = conn.execute(sql)
    return list(cur.fetchall())


def plan_compactions(stats: List[Tuple[str, int, int, int, int, int]]) -> List[str]:
    """Return race_ids that can be safely compacted.
    Criteria: c>=3, winners==1, no duplicates (udist==c), and (minp!=1 or maxp>c)
    """
    candidates: List[str] = []
    for race_id, c, udist, minp, maxp, winners in stats:
        if c >= 3 and winners == 1 and udist == c and (minp != 1 or maxp > c):
            candidates.append(race_id)
    return candidates


def compute_rank_mapping(positions: List[int]) -> Dict[int, int]:
    """Map sorted unique positions to a compact 1..K rank sequence."""
    uniq_sorted = sorted(set(positions))
    return {p: i + 1 for i, p in enumerate(uniq_sorted)}


def normalize_names(conn: sqlite3.Connection, apply: bool) -> Dict[str, int]:
    report = {"dog_clean_name_updated": 0, "winner_name_updated": 0}

    # dog_race_data.dog_clean_name
    try:
        cur = conn.execute("SELECT id, dog_clean_name FROM dog_race_data")
        rows = cur.fetchall()
        for _id, name in rows:
            new = strip_prefix(name)
            if new is not None and new != name:
                report["dog_clean_name_updated"] += 1
                if apply:
                    conn.execute(
                        "UPDATE dog_race_data SET dog_clean_name=? WHERE id=?",
                        (new, _id),
                    )
    except Exception:
        # Table shape may differ; best-effort
        pass

    # race_metadata.winner_name
    try:
        cur = conn.execute("SELECT id, winner_name FROM race_metadata")
        rows = cur.fetchall()
        for _id, name in rows:
            new = strip_prefix(name)
            if new is not None and new != name:
                report["winner_name_updated"] += 1
                if apply:
                    conn.execute(
                        "UPDATE race_metadata SET winner_name=? WHERE id=?",
                        (new, _id),
                    )
    except Exception:
        pass

    return report


def compact_race(conn: sqlite3.Connection, race_id: str, apply: bool) -> Dict[str, int]:
    """Compact finish positions for a single race_id."""
    rep = {"rows_updated": 0}
    cur = conn.execute(
        "SELECT id, finish_position FROM dog_race_data WHERE race_id=? ORDER BY finish_position",
        (race_id,),
    )
    rows = cur.fetchall()
    positions = [r[1] for r in rows if r[1] is not None]
    if not positions:
        return rep
    mapping = compute_rank_mapping(positions)

    for _id, pos in rows:
        if pos is None:
            continue
        new_pos = mapping.get(pos, pos)
        if new_pos != pos:
            rep["rows_updated"] += 1
            if apply:
                conn.execute(
                    "UPDATE dog_race_data SET finish_position=? WHERE id=?",
                    (new_pos, _id),
                )
    return rep


def reconcile_field_sizes(conn: sqlite3.Connection, apply: bool) -> int:
    updated = 0
    try:
        cur = conn.execute("SELECT race_id FROM race_metadata")
        race_ids = [r[0] for r in cur.fetchall() if r[0] is not None]
        for rid in race_ids:
            cnt = conn.execute(
                "SELECT COUNT(*) FROM dog_race_data WHERE race_id=?", (rid,)
            ).fetchone()[0]
            # Only update when different to avoid write churn
            try:
                existing = conn.execute(
                    "SELECT field_size FROM race_metadata WHERE race_id=?",
                    (rid,),
                ).fetchone()
                existing_val = existing[0] if existing else None
            except Exception:
                existing_val = None
            if existing_val != cnt:
                updated += 1
                if apply:
                    conn.execute(
                        "UPDATE race_metadata SET field_size=? WHERE race_id=?",
                        (cnt, rid),
                    )
    except Exception:
        pass
    return updated


def main() -> int:
    ap = argparse.ArgumentParser(description="Repair DB positions and names")
    ap.add_argument("--db", default="greyhound_racing_data.db", help="SQLite DB path")
    ap.add_argument(
        "--apply", action="store_true", help="Apply changes (default: dry-run)"
    )
    args = ap.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    if not db_path.exists():
        print(f"❌ DB not found: {db_path}")
        return 2

    # Backup before any writes
    backup_path = None
    if args.apply:
        try:
            backup_path = backup_db(db_path)
            print(f"💾 Backup created at: {backup_path}")
        except Exception as e:
            print(f"⚠️ Backup failed (continuing cautiously): {e}")

    conn = open_sqlite_writable(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        # Analyze current stats
        stats = get_race_stats(conn)
        candidates = plan_compactions(stats)

        # Normalize names
        name_report = normalize_names(conn, apply=args.apply)

        # Compact races
        compaction_results: Dict[str, Dict[str, int]] = {}
        for rid in candidates:
            res = compact_race(conn, rid, apply=args.apply)
            if res.get("rows_updated", 0) > 0:
                compaction_results[rid] = res

        # Reconcile field_size from actual counts
        fs_updated = reconcile_field_sizes(conn, apply=args.apply)

        if args.apply:
            conn.commit()
        else:
            conn.rollback()

        # Build report
        report = {
            "db_path": str(db_path),
            "applied": bool(args.apply),
            "backup_path": str(backup_path) if backup_path else None,
            "races_total": len(stats),
            "compaction_candidates": len(candidates),
            "compactions_performed": len(compaction_results),
            "name_normalization": name_report,
            "field_size_updates": fs_updated,
            "sample_compactions": dict(list(compaction_results.items())[:10]),
        }

        # Save report
        out_dir = Path("debug_artifacts") / "v4"
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "repair_report.json").open("w") as f:
            json.dump(report, f, indent=2)

        print("✅ Repair planning/execution complete.")
        print(json.dumps(report, indent=2))
        return 0
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        print(f"❌ Error: {e}")
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
