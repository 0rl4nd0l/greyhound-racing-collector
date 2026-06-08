#!/usr/bin/env python3
"""
Normalize and sanitize upcoming race CSV files for prediction ingestion.

Actions (non-destructive):
- Archive-first policy:
  • Move non-CSV files found in UPCOMING_DIR to archive/upcoming_races/YYYY/MM/
  • Move CSV files with date in filename that are in the past or today (<= today) to archive/YYYY/MM/
- Normalize delimiter:
  • For remaining CSVs, convert comma-delimited files to pipe-delimited ("|")
  • Leave already pipe-delimited files as-is; unknown structures are skipped
- Emit a JSON summary to the path provided via --summary

Notes:
- Filename format expectation is documented in docs/FORM_GUIDE_SPEC.md
  (e.g., "Race {num} - {VENUE} - YYYY-MM-DD.csv"). This script does not rename
  files; it only archives or normalizes delimiters.
- The winner/outcome fields are not stripped here to avoid data loss; upstream
  validation must ensure upcoming files contain only pre-race fields.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
# Filename contract: "Race {num} - {VENUE} - YYYY-MM-DD.csv" (hyphen or en dash)
FILENAME_PATTERN = re.compile(
    r"^Race\s+(\d{1,2})\s*[–-]\s*([A-Z0-9_]+(?:-[A-Z0-9_]+)*)\s*[–-]\s*(\d{4}-\d{2}-\d{2})\.csv$",
    re.IGNORECASE,
)


@dataclass
class FileAction:
    name: str
    reason: str
    dest: Optional[str] = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanitize upcoming race files")
    p.add_argument("--dir", dest="directory", default="./upcoming_races_temp", help="Directory to scan")
    p.add_argument("--archive", dest="archive_root", default="archive/upcoming_races", help="Archive root dir")
    p.add_argument(
        "--summary", dest="summary_path", default="reports/validation/upcoming_sanitization_summary.json", help="Where to write JSON summary"
    )
    p.add_argument("--strict-future", action="store_true", help="Archive files dated today or earlier (<= today)")
    return p.parse_args()


def find_date_in_name(name: str) -> Optional[date]:
    m = DATE_RE.search(name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d").date()
    except Exception:
        return None


def safe_move(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not dst.exists():
        shutil.move(str(src), str(dst))
        return dst
    stem = dst.stem
    suffix = dst.suffix
    i = 1
    while True:
        cand = dst_dir / f"{stem} ({i}){suffix}"
        if not cand.exists():
            shutil.move(str(src), str(cand))
            return cand
        i += 1


def convert_to_pipe_delimited(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        # Short-circuit if clearly pipe-delimited
        first_line = text.splitlines(True)[:1]
        if first_line and "|" in first_line[0]:
            return False
        if first_line and "," not in first_line[0]:
            # Unknown; do not risk corrupting structure
            return False
        # Use csv.Sniffer for delimiter when possible
        sample = text[:8192]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",|\t;")
        except Exception:
            class _D:  # fallback to comma
                delimiter = ","
            dialect = _D()
        # Parse and rewrite with '|'
        rows: List[List[str]] = []
        for row in csv.reader(text.splitlines(), dialect):
            rows.append(row)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="") as out:
            writer = csv.writer(out, delimiter="|")
            for row in rows:
                writer.writerow(row)
        tmp.replace(path)
        return True
    except Exception:
        return False


def main() -> int:
    args = parse_args()
    upcoming_dir = Path(args.directory)
    archive_root = Path(args.archive_root)
    summary_path = Path(args.summary_path)
    strict_future = bool(args.strict_future)

    upcoming_dir.mkdir(parents=True, exist_ok=True)

    archived: List[FileAction] = []
    normalized: List[FileAction] = []
    untouched: List[FileAction] = []

    today = date.today()

    for entry in sorted(upcoming_dir.iterdir()):
        if entry.name.startswith("."):
            continue
        if not entry.is_file():
            continue

        lower = entry.name.lower()
        if not lower.endswith(".csv"):
            # Non-CSV -> archive/YYYY/MM
            d = find_date_in_name(entry.name) or today
            dst_dir = archive_root / f"{d.year:04d}" / f"{d.month:02d}"
            dest = safe_move(entry, dst_dir)
            archived.append(FileAction(entry.name, "non-csv", str(dest)))
            continue

        # CSV: enforce filename contract
        if not FILENAME_PATTERN.match(entry.name):
            d = find_date_in_name(entry.name) or today
            dst_dir = archive_root / f"{d.year:04d}" / f"{d.month:02d}"
            dest = safe_move(entry, dst_dir)
            archived.append(FileAction(entry.name, "invalid-filename", str(dest)))
            continue

        # CSV: date in name check
        d = find_date_in_name(entry.name)
        if strict_future and d is not None and d <= today:
            dst_dir = archive_root / f"{d.year:04d}" / f"{d.month:02d}"
            dest = safe_move(entry, dst_dir)
            archived.append(FileAction(entry.name, "past-or-today", str(dest)))
            continue

        # Normalize delimiter to '|'
        changed = convert_to_pipe_delimited(entry)
        if changed:
            normalized.append(FileAction(entry.name, "converted-to-pipe"))
        else:
            # Either already pipe-delimited or unknown structure (left untouched)
            reason = "already-pipe"
            try:
                with entry.open("r", encoding="utf-8", errors="replace") as f:
                    header = f.readline()
                    if "|" not in header:
                        reason = "unknown-delimiter"
            except Exception:
                reason = "read-error"
            untouched.append(FileAction(entry.name, reason))

    # Write summary JSON
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "directory": str(upcoming_dir),
        "strict_future": strict_future,
        "archived_count": len(archived),
        "normalized_count": len(normalized),
        "untouched_count": len(untouched),
        "archived": [asdict(a) for a in archived],
        "normalized": [asdict(a) for a in normalized],
        "untouched": [asdict(a) for a in untouched],
        "timestamp": datetime.now().isoformat(),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps({k: summary[k] for k in ("archived_count", "normalized_count", "untouched_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
