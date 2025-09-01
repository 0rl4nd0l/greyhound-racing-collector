#!/usr/bin/env python3
"""
Validate upcoming race CSV files.

Checks performed:
- Directory scan of UPCOMING_RACES_DIR (default: ./upcoming_races)
- Filenames must match: "Race {num} - {VENUE} - {YYYY-MM-DD}.csv" (hyphen or en dash accepted)
- File extension must be .csv
- CSV must be pipe-delimited ("|")
- Required columns: Dog Name (or dog_name); BOX is recommended
- Optionally sanity-check that the date in filename is today or in the future

Exit codes:
- 0: All validations passed (or no files found)
- 1: One or more files failed validation

Environment variables:
- UPCOMING_RACES_DIR: override directory to scan
- VALIDATE_STRICT_FUTURE: if set to "1", require date strictly > today (not today)

Usage:
  python scripts/validate_upcoming_races.py [--dir PATH] [--strict-future]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Structured logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.logging_config import get_component_logger  # type: ignore

log = get_component_logger()

API_PATTERN = re.compile(
    r"^Race\s+(\d{1,2})\s*[–-]\s*([A-Z_]+)\s*[–-]\s*(\d{4}-\d{2}-\d{2})\.csv$",
    re.IGNORECASE,
)

# Acceptable header synonyms (case-insensitive)
REQUIRED_HEADERS = [
    {"dog name", "dog_name", "dog"},  # at least one required
]
RECOMMENDED_HEADERS = [
    {"box", "box_number", "box no", "box_no"},
]


def normalize_header(h: str) -> str:
    return h.strip().lower()


def parse_headers(line: str, path: Path) -> List[str]:
    # Require pipe delimiter
    if "|" not in line:
        raise ValueError(
            f"{path.name}: Expected pipe-delimited headers ('|'), got: {line.strip()[:120]}"
        )
    headers = [normalize_header(h) for h in line.strip().split("|")]
    # Basic sanity: no empty headers
    if any(h == "" for h in headers):
        raise ValueError(f"{path.name}: Empty header name detected")
    return headers


def validate_required_columns(headers: List[str], path: Path) -> List[str]:
    problems: List[str] = []
    header_set = set(headers)

    # Required: at least one of each group must be present
    for group in REQUIRED_HEADERS:
        if not any(opt in header_set for opt in group):
            problems.append(
                f"{path.name}: Missing required column; expected one of: {sorted(group)}"
            )

    # Recommended: warn if missing
    for group in RECOMMENDED_HEADERS:
        if not any(opt in header_set for opt in group):
            problems.append(
                f"{path.name}: WARNING: Missing recommended column; expected one of: {sorted(group)}"
            )

    return problems


def iter_csv_rows(path: Path):
    # Stream file; check that each non-empty row has the same number of fields as headers
    with path.open("r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline()
        if not header_line:
            raise ValueError(f"{path.name}: File is empty; missing headers")
        headers = parse_headers(header_line, path)
        header_cols = len(headers)

        # Validate required/recommended headers
        header_problems = validate_required_columns(headers, path)

        line_no = 1
        for line in f:
            line_no += 1
            stripped = line.strip("\n")
            if not stripped:
                # allow blank lines
                continue
            parts = stripped.split("|")
            if len(parts) != header_cols:
                raise ValueError(
                    f"{path.name}: Row {line_no} has {len(parts)} columns; expected {header_cols}"
                )
        return header_problems


def validate_filename(
    path: Path,
) -> Tuple[Optional[int], Optional[str], Optional[date], List[str]]:
    problems: List[str] = []

    if path.suffix.lower() != ".csv":
        problems.append(f"{path.name}: Invalid extension; expected .csv")

    m = API_PATTERN.match(path.name)
    if not m:
        problems.append(
            f"{path.name}: Filename must match 'Race {{num}} - {{VENUE}} - YYYY-MM-DD.csv' (hyphen or en dash accepted)"
        )
        return None, None, None, problems

    try:
        race_no = int(m.group(1))
    except Exception:
        race_no = None
        problems.append(f"{path.name}: Invalid race number in filename")

    venue = m.group(2).upper() if m.group(2) else None
    try:
        race_date = datetime.strptime(m.group(3), "%Y-%m-%d").date()
    except Exception:
        race_date = None
        problems.append(f"{path.name}: Invalid date in filename; expected YYYY-MM-DD")

    return race_no, venue, race_date, problems


def validate_future_date(
    race_date: Optional[date], path: Path, strict_future: bool
) -> List[str]:
    problems: List[str] = []
    if race_date is None:
        return problems
    today = date.today()
    if strict_future:
        if race_date <= today:
            problems.append(
                f"{path.name}: Date should be strictly in the future (>{today.isoformat()}), got {race_date.isoformat()}"
            )
    else:
        if race_date < today:
            problems.append(
                f"{path.name}: Date should be today or in the future (≥{today.isoformat()}), got {race_date.isoformat()}"
            )
    return problems


def find_upcoming_dir(cli_dir: Optional[str]) -> Path:
    env_dir = os.environ.get("UPCOMING_RACES_DIR")
    base = Path(cli_dir or env_dir or "./upcoming_races").resolve()
    return base


def validate_file(path: Path, strict_future: bool) -> List[str]:
    problems: List[str] = []

    # Resolve symlinks but keep name for messaging
    try:
        actual = path.resolve()
    except Exception:
        actual = path

    # Filename checks
    race_no, venue, race_date, fn_problems = validate_filename(
        path,
    )
    problems.extend(fn_problems)

    # Date sanity
    problems.extend(validate_future_date(race_date, path, strict_future))

    # CSV structural checks
    try:
        header_problems = iter_csv_rows(actual)
        problems.extend(header_problems)
    except Exception as e:
        problems.append(str(e))

    return problems


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate upcoming race CSV files")
    parser.add_argument(
        "--dir",
        dest="directory",
        default=None,
        help="Directory to scan (defaults to env UPCOMING_RACES_DIR or ./upcoming_races)",
    )
    parser.add_argument(
        "--strict-future",
        action="store_true",
        help="Require dates strictly greater than today",
    )
    args = parser.parse_args(argv)

    strict_env = os.environ.get("VALIDATE_STRICT_FUTURE", "0") == "1"
    strict_future = args.strict_future or strict_env

    upcoming_dir = find_upcoming_dir(args.directory)
    if not upcoming_dir.exists():
        msg = (
            f"No upcoming races directory found at {upcoming_dir}. Skipping validation."
        )
        print(f"INFO: {msg}")
        log.info(
            msg,
            action="validate_upcoming_scan",
            details={"directory": str(upcoming_dir)},
            component="qa",
        )
        return 0

    entries = sorted([p for p in upcoming_dir.iterdir() if not p.name.startswith(".")])
    # Filter to files and symlinks only
    csv_paths = [p for p in entries if p.is_file() or p.is_symlink()]

    # Log discovery summary
    skipped: Dict[str, str] = {}
    for p in entries:
        if not (p.is_file() or p.is_symlink()):
            skipped[p.name] = "not a regular file or symlink"
        elif p.suffix.lower() != ".csv":
            skipped[p.name] = "invalid extension (only .csv allowed)"

    log.info(
        "Upcoming files discovery",
        action="validate_upcoming_scan",
        details={
            "directory": str(upcoming_dir),
            "found_count": len(csv_paths),
            "found_names": [p.name for p in csv_paths],
            "skipped_count": len(skipped),
            "skipped": skipped,
        },
        component="qa",
    )

    if not csv_paths:
        msg = f"No files found in {upcoming_dir}. Nothing to validate."
        print(f"INFO: {msg}")
        return 0

    all_problems: Dict[str, List[str]] = {}
    for p in csv_paths:
        if p.suffix.lower() != ".csv":
            # Fail early on non-csv (already accounted in skipped but keep guard)
            reason = f"{p.name}: Invalid extension; only .csv files are allowed in upcoming_races"
            all_problems.setdefault(p.name, []).append(reason)
            continue
        probs = validate_file(p, strict_future)
        if probs:
            all_problems[p.name] = probs

    if all_problems:
        print("Validation failed for the following files:")
        total_issues = 0
        for name, issues in all_problems.items():
            for issue in issues:
                print(f"- {issue}")
                total_issues += 1
        summary = {
            "files_with_issues": len(all_problems),
            "total_problems": total_issues,
        }
        log.warning(
            "Upcoming CSV validation failed",
            action="validate_upcoming",
            outcome="failed",
            details={**summary, "problems": all_problems},
            component="qa",
        )
        print(
            f"SUMMARY: {summary['files_with_issues']} files with issues; {summary['total_problems']} total problems."
        )
        return 1

    print(
        f"SUCCESS: Validated {len(csv_paths)} files in {upcoming_dir}. All checks passed."
    )
    log.info(
        "Upcoming CSV validation passed",
        action="validate_upcoming",
        outcome="success",
        details={"validated_count": len(csv_paths), "directory": str(upcoming_dir)},
        component="qa",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
