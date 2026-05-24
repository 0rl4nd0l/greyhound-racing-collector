#!/usr/bin/env python3
"""
Create symlink aliases for upcoming race CSVs to match the API's expected filename pattern.

API parser in app.py expects: "Race {number} - {VENUE} - {YYYY-MM-DD}.csv" (hyphens acceptable for dashes).
This script scans ./upcoming_races (and known archives), extracts metadata, and creates symlinks
in ./upcoming_races to that exact pattern, preserving originals. Writes a migration note.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
UPCOMING_DIR = REPO_ROOT / "upcoming_races"
MIGRATIONS_DIR = REPO_ROOT / "migrations"
ARCHIVE_DIRS = [
    REPO_ROOT / "archive",
    REPO_ROOT / "archive" / "corrupt_or_legacy_race_files",
    REPO_ROOT / "archive" / "corrupt_historical_race_data",
]

API_PATTERN = re.compile(
    r"^Race\s+(\d{1,2})\s*[–-]\s*([A-Z_]+)\s*[–-]\s*(\d{4}-\d{2}-\d{2})\.csv$",
    re.IGNORECASE,
)

# General extractors
RE_NUM = re.compile(r"(?:^|[ _-])(?P<num>\d{1,2})(?:[ _-]|\.)")
RE_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")
RE_VENUE = re.compile(r"([A-Z_]{2,5})")


def ensure_dirs():
    UPCOMING_DIR.mkdir(exist_ok=True)
    MIGRATIONS_DIR.mkdir(exist_ok=True)


def is_api_named(name: str) -> bool:
    return API_PATTERN.match(name) is not None


def extract_meta(name: str):
    # Try API pattern first
    m = API_PATTERN.match(name)
    if m:
        return {
            "race": int(m.group(1)),
            "venue": m.group(2).upper(),
            "date": m.group(3),
        }
    # Fallback: flexible extraction
    num = None
    m_num = RE_NUM.search(name)
    if m_num:
        try:
            num = int(m_num.group("num"))
        except Exception:
            num = None
    date = None
    m_date = RE_DATE.search(name)
    if m_date:
        date = m_date.group(1)
    venue = None
    m_venue = RE_VENUE.search(name)
    if m_venue:
        venue = m_venue.group(1).upper()
    return {"race": num, "venue": venue, "date": date}


def build_api_name(meta: dict) -> str:
    race = meta.get("race") or 1
    try:
        race = int(race)
    except Exception:
        race = 1
    venue = (meta.get("venue") or "UNKNOWN").upper()
    date = meta.get("date") or "1970-01-01"
    # Validate date
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except Exception:
        date = "1970-01-01"
    return f"Race {race} - {venue} - {date}.csv"


def discover_sources():
    sources = []
    search_dirs = [UPCOMING_DIR] + [d for d in ARCHIVE_DIRS if d.exists()]
    for base in search_dirs:
        if not base.exists():
            continue
        for p in base.rglob("*.csv"):
            if p.name.startswith("."):
                continue
            sources.append(p)
    return sources


def create_symlink(src: Path, new_name: str, plan: list):
    dest = UPCOMING_DIR / new_name
    if dest.exists():
        # If already correct symlink, skip
        if dest.is_symlink():
            try:
                if dest.resolve() == src.resolve():
                    return
            except Exception:
                pass
        # Avoid overwrite: suffix with __n
        stem, ext = os.path.splitext(new_name)
        i = 2
        while (UPCOMING_DIR / f"{stem}__{i}{ext}").exists():
            i += 1
        dest = UPCOMING_DIR / f"{stem}__{i}{ext}"
    rel_src = os.path.relpath(src, start=UPCOMING_DIR)
    dest.symlink_to(rel_src)
    plan.append({"old": str(src), "new": str(dest)})


def main():
    ensure_dirs()
    plan = []
    for src in discover_sources():
        meta = extract_meta(src.name)
        if not meta.get("race") or not meta.get("venue") or not meta.get("date"):
            continue
        new_name = build_api_name(meta)
        if is_api_named(src.name) and src.parent.resolve() == UPCOMING_DIR.resolve():
            # Already in desired pattern inside upcoming dir
            continue
        create_symlink(src, new_name, plan)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    note_json = MIGRATIONS_DIR / f"{ts}_upcoming_api_name_aliases.json"
    with open(note_json, "w") as f:
        json.dump(
            {"generated_at": ts, "mappings": plan, "policy": "symlink_only"},
            f,
            indent=2,
        )
    note_md = MIGRATIONS_DIR / f"{ts}_upcoming_api_name_aliases.md"
    with open(note_md, "w") as f:
        f.write("# Upcoming Race API Filename Aliases\n\n")
        if plan:
            f.write("Old path -\u003e new symlink (API pattern)\n\n")
            for m in plan:
                f.write(f"- {m['old']} -\u003e {m['new']}\n")
        else:
            f.write("No aliases created (files already matched API pattern).\n")

    print(
        json.dumps(
            {"created_aliases": len(plan), "notes": [str(note_json), str(note_md)]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
