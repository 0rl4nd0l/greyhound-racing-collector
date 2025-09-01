#!/usr/bin/env python3
"""
Create API-pattern aliases only for files in ./upcoming_races that already contain
clear venue/date/race in their name. Skip UNKNOWN/ambiguous files.
Pattern created: "Race {num} - {VENUE} - {YYYY-MM-DD}.csv"
"""
import os
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UPCOMING = ROOT / "upcoming_races"

API_NAME = lambda n, v, d: f"Race {n} - {v} - {d}.csv"
RE_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")
RE_NUM = re.compile(r"(?:^|[_\s-])(\d{1,2})(?:[_\s-]|\.)")
RE_VENUE_UPCOMING = re.compile(
    r"Upcoming_([A-Z_]{2,5})_(\d{4}-\d{2}-\d{2})_(\d{1,2})", re.IGNORECASE
)
RE_API = re.compile(
    r"^Race\s+(\d{1,2})\s*[–-]\s*([A-Z_]+)\s*[–-]\s*(\d{4}-\d{2}-\d{2})\.csv$",
    re.IGNORECASE,
)


def extract_from_name(name: str):
    m = RE_API.match(name)
    if m:
        return int(m.group(1)), m.group(2).upper(), m.group(3)
    m = RE_VENUE_UPCOMING.search(name)
    if m and m.group(1) != "UNKNOWN":
        return int(m.group(3)), m.group(1).upper(), m.group(2)
    # Flexible fallback
    mdate = RE_DATE.search(name)
    mnum = RE_NUM.search(name)
    if mdate and mnum:
        date = mdate.group(1)
        num = int(mnum.group(1))
        # Try to spot a venue token
        for tok in name.replace("-", " ").replace("_", " ").split():
            if tok.isupper() and 2 <= len(tok) <= 5 and tok != "UNKNOWN":
                return num, tok, date
    return None


def main():
    created = 0
    if not UPCOMING.exists():
        print("{}")
        return
    for p in UPCOMING.iterdir():
        if p.name.startswith("."):
            continue
        if p.is_symlink():
            # don't alias symlinks
            continue
        if not p.suffix.lower() == ".csv":
            continue
        meta = extract_from_name(p.name)
        if not meta:
            continue
        num, venue, date = meta
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except Exception:
            continue
        new_name = API_NAME(num, venue, date)
        dest = UPCOMING / new_name
        if dest.exists():
            continue
        rel_src = os.path.relpath(p, start=UPCOMING)
        dest.symlink_to(rel_src)
        created += 1
    print({"created": created})


if __name__ == "__main__":
    main()
