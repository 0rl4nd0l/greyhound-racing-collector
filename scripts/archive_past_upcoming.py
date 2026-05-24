#!/usr/bin/env python3
"""
Archive past-date upcoming race CSVs.

- Scans UPCOMING_RACES_DIR for CSVs matching: "Race {num} - {VENUE} - {YYYY-MM-DD}.csv" (hyphen or en dash accepted)
- Moves files with date < today to ARCHIVE_DIR/past_races (preserving filename)
- Logs discovery summary, moves, and skips with reasons
- Supports dry-run via --dry-run flag or DRY_RUN=1 environment variable

Environment:
- UPCOMING_RACES_DIR: directory of live upcoming CSVs (default: ./upcoming_races or ${DATA_DIR}/upcoming_races)
- PAST_RACES_ARCHIVE_DIR: archive destination (default: ./archive/past_races)
- DRY_RUN: if "1", perform a dry run

Usage:
  python scripts/archive_past_upcoming.py [--dir PATH] [--archive PATH] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import re
import shutil

# Structured logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.logging_config import get_component_logger  # type: ignore
from config.paths import ARCHIVE_DIR, UPCOMING_RACES_DIR

log = get_component_logger()

API_PATTERN = re.compile(
    r"^Race\s+(\d{1,2})\s*[–-]\s*([A-Z_]+)\s*[–-]\s*(\d{4}-\d{2}-\d{2})\.csv$",
    re.IGNORECASE,
)


def parse_filename(path: Path) -> Tuple[Optional[int], Optional[str], Optional[date]]:
    if path.suffix.lower() != ".csv":
        return None, None, None
    m = API_PATTERN.match(path.name)
    if not m:
        return None, None, None
    try:
        race_no = int(m.group(1))
    except Exception:
        race_no = None
    venue = m.group(2).upper() if m.group(2) else None
    try:
        race_date = datetime.strptime(m.group(3), "%Y-%m-%d").date()
    except Exception:
        race_date = None
    return race_no, venue, race_date


def move_file(src: Path, dest_dir: Path, dry_run: bool) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dry_run:
        log.info(
            "DRY RUN: would move file",
            action="archive_upcoming_move",
            outcome="dry_run",
            details={"src": str(src), "dest": str(dest)},
            component="qa",
        )
        return
    try:
        shutil.move(str(src), str(dest))
        log.info(
            "Archived past-date upcoming CSV",
            action="archive_upcoming_move",
            outcome="moved",
            details={"src": str(src), "dest": str(dest)},
            component="qa",
        )
    except Exception as e:
        log.error(
            f"Failed to move file: {e}",
            action="archive_upcoming_move",
            outcome="error",
            details={"src": str(src), "dest": str(dest), "error": str(e)},
            component="qa",
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Archive past-date upcoming CSVs")
    parser.add_argument(
        "--dir",
        dest="directory",
        default=None,
        help="Directory to scan (defaults to UPCOMING_RACES_DIR)",
    )
    parser.add_argument(
        "--archive",
        dest="archive",
        default=None,
        help="Archive directory (defaults to PAST_RACES_ARCHIVE_DIR or ./archive/past_races)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Do not move files; just log actions"
    )
    args = parser.parse_args(argv)

    upcoming_dir = Path(
        args.directory
        or os.environ.get("UPCOMING_RACES_DIR")
        or str(UPCOMING_RACES_DIR)
    ).resolve()
    archive_dir = Path(
        args.archive
        or os.environ.get("PAST_RACES_ARCHIVE_DIR")
        or str(ARCHIVE_DIR / "past_races")
    ).resolve()
    dry_run = args.dry_run or (os.environ.get("DRY_RUN", "0") == "1")

    if not upcoming_dir.exists():
        log.warning(
            "Upcoming directory missing",
            action="archive_upcoming_scan",
            outcome="skipped",
            details={"directory": str(upcoming_dir)},
            component="qa",
        )
        print(
            f"INFO: Upcoming directory not found at {upcoming_dir}. Nothing to archive."
        )
        return 0

    entries = sorted([p for p in upcoming_dir.iterdir() if not p.name.startswith(".")])
    files = [p for p in entries if p.is_file() or p.is_symlink()]

    skipped: Dict[str, str] = {}
    candidates: List[Path] = []
    today = date.today()

    for p in entries:
        if not (p.is_file() or p.is_symlink()):
            skipped[p.name] = "not a regular file or symlink"
            continue
        if p.suffix.lower() != ".csv":
            skipped[p.name] = "invalid extension (only .csv allowed)"
            continue
        race_no, venue, race_date = parse_filename(p)
        if race_date is None:
            skipped[p.name] = "filename does not match expected pattern"
            continue
        if race_date < today:
            candidates.append(p)
        else:
            skipped[p.name] = f"not past-date (race_date={race_date.isoformat()})"

    log.info(
        "Upcoming housekeeping discovery",
        action="archive_upcoming_scan",
        details={
            "directory": str(upcoming_dir),
            "archive_dir": str(archive_dir),
            "found_count": len(files),
            "found_names": [p.name for p in files],
            "skipped_count": len(skipped),
            "skipped": skipped,
            "candidates_count": len(candidates),
            "candidates": [p.name for p in candidates],
            "dry_run": dry_run,
        },
        component="qa",
    )

    moved = 0
    for p in candidates:
        move_file(p, archive_dir, dry_run)
        moved += 1

    outcome = "dry_run" if dry_run else "archived"
    log.info(
        "Upcoming housekeeping completed",
        action="archive_upcoming",
        outcome=outcome,
        details={
            "total_candidates": len(candidates),
            "moved": moved,
            "archive_dir": str(archive_dir),
        },
        component="qa",
    )
    print(
        f"INFO: Housekeeping completed: {moved}/{len(candidates)} files {'would be moved' if dry_run else 'moved'} to {archive_dir}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
