#!/usr/bin/env python3
"""
Staging writer for CSV-embedded dog histories.

Parses a single race CSV (form guide or enriched race CSV) and produces
normalized staging records for race_metadata and dog_race_data.

This module focuses on:
- Robust metadata extraction (race_date, venue, race_number) from CSV or filename
- Venue normalization (replace slashes with underscores)
- ISO date normalization (YYYY-MM-DD)
- Mapping common dog-level fields (dog_name, box, plc, time, first sectional, margin, weight, trainer, starting price)
- Preserving the full CSV row as JSON for auditability

It does not write to the database by itself; see scripts/ingest_csv_history.py
for creating staging tables and performing upserts into canonical tables.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# --- Helpers shared within this module ---

CANONICAL_DATE_FORMAT = "%Y-%m-%d"


@dataclass
class RaceMeta:
    race_date: str  # YYYY-MM-DD
    venue: str  # venue code (normalized)
    race_number: int
    race_name: Optional[str] = None
    grade: Optional[str] = None
    distance: Optional[str] = None

    @property
    def race_id(self) -> str:
        # Canonical race_id: VENUE_YYYY-MM-DD_RN
        return f"{self.venue}_{self.race_date}_{self.race_number}"


# Dialect sniffing similar to ingestion.ingest_race_csv
class _DefaultDialect(csv.Dialect):
    delimiter = ","
    quotechar = '"'
    escapechar = None
    doublequote = True
    skipinitialspace = False
    lineterminator = "\n"
    quoting = csv.QUOTE_MINIMAL


def sniff_dialect_and_headers(csv_path: Path) -> Tuple[csv.Dialect, List[str]]:
    with csv_path.open("r", encoding="utf-8", errors="replace") as f:
        sample = f.read(8192)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",|\t;")
        except Exception:
            dialect = _DefaultDialect()
    with csv_path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, dialect)
        headers = next(reader, [])
    return dialect, [h.strip() for h in headers]


def parse_date(value: str) -> Optional[str]:
    value = (value or "").strip()
    # Normalize common separators used in filenames
    norm = value.replace("_", " ")
    # Try a variety of common formats, including full month names (e.g., "03 July 2025")
    for fmt in (
        "%Y-%m-%d",  # ISO
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%Y/%m/%d",
        "%d %b %Y",  # 03 Jul 2025
        "%d %B %Y",  # 03 July 2025
        "%b %d, %Y",  # Jul 03, 2025
        "%B %d, %Y",  # July 03, 2025
        "%d_%b_%Y",  # 03_Jul_2025 (fallback for odd filenames)
        "%d_%B_%Y",  # 03_July_2025
    ):
        try:
            return datetime.strptime(norm, fmt).strftime(CANONICAL_DATE_FORMAT)
        except Exception:
            continue
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return value
    return None


def normalize_venue(value: str) -> str:
    v = (value or "").strip()
    # Replace slashes with underscores per ML system normalization (e.g., AP/K -> AP_K)
    v = v.replace("/", "_")
    # Uppercase venue codes; keep hyphens/underscores
    v = re.sub(r"[^A-Za-z0-9_\-]", "", v).upper()
    return v


# Fallback filename patterns
FILENAME_PATTERNS = [
    # Race 7 - MURR - 2025-08-24.csv (ISO date)
    re.compile(
        r"race[\s_]*(?P<race_number>\d+)[\s_]*[-_][\s_]*(?P<venue>[A-Za-z0-9_\-/]+)[\s_]*[-_][\s_]*(?P<race_date>\d{4}-\d{2}-\d{2})",
        re.IGNORECASE,
    ),
    # Race 7 - MURR - 03 July 2025.csv (human date with full month name)
    re.compile(
        r"race[\s_]*(?P<race_number>\d+)[\s_]*[-_][\s_]*(?P<venue>[A-Za-z0-9_\-/]+)[\s_]*[-_][\s_]*(?P<race_date>\d{1,2}[\s_]+[A-Za-z]{3,9}[\s_]+\d{4})",
        re.IGNORECASE,
    ),
    # 2025-08-24_MURR_R7.csv or 2025-08-24-murr-7.csv
    re.compile(
        r"(?P<race_date>\d{4}-\d{2}-\d{2})[_-](?P<venue>[A-Za-z0-9_\-/]+)[_-]R?(?P<race_number>\d+)",
        re.IGNORECASE,
    ),
]


def _first(row: Dict[str, str], keys: Iterable[str]) -> Optional[str]:
    lower_map = {k.lower(): k for k in row.keys()}
    for k in keys:
        lk = k.lower()
        if lk in lower_map:
            raw = row[lower_map[lk]]
            if raw is not None and str(raw).strip() != "":
                return str(raw).strip()
    return None


def extract_meta_from_csv(csv_path: Path, dialect: csv.Dialect) -> Optional[RaceMeta]:
    DATE_KEYS = ["race_date", "race date", "meeting_date", "date", "meeting date"]
    VENUE_KEYS = [
        "venue",
        "track",
        "venue_code",
        "venue code",
        "meeting_venue",
        "meeting venue",
    ]
    RACE_NO_KEYS = ["race_number", "race no", "race", "race_no", "race number"]
    GRADE_KEYS = ["grade"]
    DIST_KEYS = ["distance", "dist"]
    NAME_KEYS = ["race_name", "race name"]

    with csv_path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, dialect=dialect)
        for i, row in enumerate(reader):
            if i > 20:
                break
            date_raw = _first(row, DATE_KEYS)
            venue_raw = _first(row, VENUE_KEYS)
            rno_raw = _first(row, RACE_NO_KEYS)
            grade_raw = _first(row, GRADE_KEYS)
            dist_raw = _first(row, DIST_KEYS)
            name_raw = _first(row, NAME_KEYS)
            if date_raw and venue_raw and rno_raw:
                date = parse_date(date_raw)
                venue = normalize_venue(venue_raw)
                try:
                    race_number = int(re.sub(r"[^0-9]", "", rno_raw))
                except Exception:
                    continue
                if date:
                    return RaceMeta(
                        race_date=date,
                        venue=venue,
                        race_number=race_number,
                        race_name=name_raw,
                        grade=grade_raw,
                        distance=str(dist_raw) if dist_raw is not None else None,
                    )
    return None


def extract_meta_from_filename(p: Path) -> Optional[RaceMeta]:
    stem = p.stem
    for pat in FILENAME_PATTERNS:
        m = pat.search(stem)
        if m:
            date = parse_date(m.group("race_date"))
            venue = normalize_venue(m.group("venue"))
            try:
                rno = int(m.group("race_number"))
            except Exception:
                continue
            if date and venue and rno:
                return RaceMeta(race_date=date, venue=venue, race_number=rno)
    return None


# Dog row mapping
DOG_NAME_KEYS = ["Dog", "Dog Name", "dog_name", "dog name", "Name"]
BOX_KEYS = ["Box Number", "Box", "BOX", "box", "trap", "Trap"]
PLC_KEYS = ["PLC", "Position", "Finish", "finish_position", "placing"]
WGT_KEYS = ["Weight", "WGT", "weight"]
TIME_KEYS = ["Time", "Race Time", "RaceTime", "time", "individual_time"]
SEC1_KEYS = [
    "First Sectional",
    "1st Sectional",
    "first_sectional",
    "sectional_1st",
    "first split",
]
MARGIN_KEYS = ["Margin", "margin", "Beaten Margin", "beaten_margin"]
TRAINER_KEYS = ["Trainer", "trainer", "Trainer Name"]
SP_KEYS = [
    "Starting Price",
    "SP",
    "Odds",
    "Odds Decimal",
    "odds_decimal",
    "starting_price",
]


def parse_race_csv_for_staging(
    csv_path: str,
) -> Tuple[RaceMeta, List[Dict[str, object]]]:
    p = Path(csv_path).expanduser().resolve()
    dialect, headers = sniff_dialect_and_headers(p)

    meta = extract_meta_from_csv(p, dialect)
    if not meta:
        meta = extract_meta_from_filename(p)
    if not meta:
        raise ValueError(
            "Unable to extract race metadata (race_date, venue, race_number) from CSV or filename."
        )

    dogs: List[Dict[str, object]] = []
    with p.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, dialect=dialect)
        for row in reader:
            # Skip empty rows
            if not any((str(v).strip() if v is not None else "") for v in row.values()):
                continue

            def pick(keys: Iterable[str]) -> Optional[str]:
                return _first(row, keys)

            dog_name = pick(DOG_NAME_KEYS)
            if not dog_name:
                # Can't stage a dog row without a name
                continue
            dog_clean_name = str(dog_name).strip()

            # Box number
            box_raw = pick(BOX_KEYS)
            box_number: Optional[int] = None
            if box_raw is not None:
                try:
                    box_number = int(re.sub(r"[^0-9]", "", box_raw))
                except Exception:
                    box_number = None

            # Finish position
            plc_raw = pick(PLC_KEYS)
            finish_position: Optional[int] = None
            if plc_raw is not None and str(plc_raw).strip() != "":
                try:
                    finish_position = int(re.sub(r"[^0-9]", "", plc_raw))
                except Exception:
                    finish_position = None

            # Weight
            wgt_raw = pick(WGT_KEYS)
            weight: Optional[float] = None
            if wgt_raw is not None and str(wgt_raw).strip() != "":
                try:
                    weight = float(re.sub(r"[^0-9\.]+", "", str(wgt_raw)))
                except Exception:
                    weight = None

            # Times and margin
            indiv_time = pick(TIME_KEYS)
            sec1 = pick(SEC1_KEYS)
            margin_raw = pick(MARGIN_KEYS)
            margin: Optional[float] = None
            if margin_raw is not None and str(margin_raw).strip() != "":
                try:
                    margin = float(re.sub(r"[^0-9\.]+", "", str(margin_raw)))
                except Exception:
                    # keep as text if cannot parse
                    margin = None

            # Trainer and SP
            trainer = pick(TRAINER_KEYS)
            sp_raw = pick(SP_KEYS)
            starting_price: Optional[float] = None
            if sp_raw is not None and str(sp_raw).strip() != "":
                try:
                    starting_price = float(re.sub(r"[^0-9\.]+", "", str(sp_raw)))
                except Exception:
                    starting_price = None

            # Preserve raw row as JSON
            try:
                raw_row_json = json.dumps(row, ensure_ascii=False)
            except Exception:
                raw_row_json = None

            dogs.append(
                {
                    "race_id": meta.race_id,
                    "venue": meta.venue,
                    "race_number": meta.race_number,
                    "race_date": meta.race_date,
                    "dog_name": dog_name,
                    "dog_clean_name": dog_clean_name,
                    "box_number": box_number,
                    "finish_position": finish_position,
                    "weight": weight,
                    "starting_price": starting_price,
                    "individual_time": indiv_time,
                    "sectional_1st": sec1,
                    "margin": margin,
                    "trainer_name": trainer,
                    "data_source": "csv_stage",
                    "extraction_timestamp": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    "raw_row_json": raw_row_json,
                }
            )

    return meta, dogs


__all__ = [
    "RaceMeta",
    "parse_race_csv_for_staging",
]
