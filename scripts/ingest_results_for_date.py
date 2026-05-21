#!/usr/bin/env python3
"""
Official-first race results ingestion for upcoming-race CSVs.

Source order:
1. TheDogs official race page, when accessible and parseable.
2. Sportsbet results page top-four order as fallback.

Sportsbet only exposes top-four box order on the rendered results page, so
fallback rows are marked as partial_sportsbet_results rather than complete.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.race_lifecycle import RESULTED, UPCOMING_NOT_JUMPED, classify_race_record

TARGET_TABLE = "dog_race_data"
PARTIAL_SPORTSBET_RESULTS = "partial_sportsbet_results"
SPORTSBET_CATEGORY_TEMPLATE = (
    "https://www.sportsbet.com.au/results/{date}/racing/greyhound-racing-4"
)
THEDOGS_BASE = "https://www.thedogs.com.au"


class _SeleniumByFallback:
    TAG_NAME = "tag name"
    CSS_SELECTOR = "css selector"


VENUE_TO_THEDOGS_SLUG = {
    "AP_K": "angle-park",
    "APWE": "albion-park",
    "APTH": "albion-park",
    "BAL": "ballarat",
    "BEN": "bendigo",
    "CANN": "cannington",
    "CAS": "casino",
    "CASO": "casino",
    "DAPT": "dapto",
    "GRDN": "the-gardens",
    "GARD": "the-gardens",
    "HOB": "hobart",
    "HOBT": "hobart",
    "MAND": "mandurah",
    "MEA": "the-meadows",
    "MOUNT": "mount-gambier",
    "NOW": "nowra",
    "NOWRA": "nowra",
    "QOT": "ladbrokes-q-straight",
    "QST": "ladbrokes-q-straight",
    "Q1L": "ladbrokes-q1-lakeside",
    "Q2": "ladbrokes-q2-parklands",
    "RICH": "richmond",
    "RICH_S": "richmond-straight",
    "SAL": "sale",
    "SAN": "sandown",
    "SHEP": "shepparton",
    "WAR": "warrnambool",
    "WPK": "wentworth-park",
    "W_PK": "wentworth-park",
    "WRGL": "warragul",
    "WARR": "warragul",
}


def norm_name(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def clean_dog_name(raw: str) -> str:
    name = re.sub(r"^\s*\d{1,2}\s*[\.\):-]\s*", "", str(raw or "").strip())
    name = name.replace('"', "").replace("'", "").replace("`", "")
    return re.sub(r"\s+", " ", name).strip().title()


def code_from_race_id(race_id: str) -> Optional[str]:
    filename_match = re.match(
        r"^\s*Race\s+\d+\s+-\s*(.+?)\s+-\s*\d{4}-\d{2}-\d{2}\s*$",
        str(race_id or ""),
        re.IGNORECASE,
    )
    if filename_match:
        return filename_match.group(1).strip()

    canonical_match = re.match(
        r"^\s*(.+?)_\d{4}-\d{2}-\d{2}_\d+\s*$",
        str(race_id or ""),
    )
    if canonical_match:
        return canonical_match.group(1).strip()

    return None


def sportsbet_slug_from_url(url: str) -> Optional[str]:
    match = re.search(r"/greyhound-racing/australia-nz/([^/]+)/race-(\d+)", url or "")
    if not match:
        return None
    return match.group(1).lower()


def result_slug_from_url(url: str) -> Optional[str]:
    match = re.search(r"/results/\d{4}-\d{2}-\d{2}/racing/greyhound-racing-4/([^/?#]+)", url or "")
    if not match:
        return None
    return re.sub(r"-\d+$", "", match.group(1)).lower()


def parse_participants_from_csv(csv_path: Path) -> List[dict]:
    participants: List[dict] = []
    seen = set()
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_name = str(row.get("Dog Name") or "").strip()
            match = re.match(r"^\s*(\d{1,2})\s*[\.\):-]\s*(.+?)\s*$", raw_name)
            if not match:
                continue
            box_number = int(match.group(1))
            dog_name = clean_dog_name(raw_name)
            key = (box_number, norm_name(dog_name))
            if key in seen:
                continue
            seen.add(key)
            participants.append({"box_number": box_number, "dog_name": dog_name})
    return participants


def parse_sportsbet_result_text(text: str) -> Dict[int, dict]:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    parsed: Dict[int, dict] = {}

    for index in range(len(lines) - 2):
        if not re.match(r"^\d{1,2}:\d{2}$", lines[index]):
            continue

        race_match = re.match(r"^R(\d+)\b\s*(.*)$", lines[index + 1])
        if not race_match:
            continue

        boxes_line = lines[index + 2]
        if not re.match(r"^\d+(?:,\d+)+$", boxes_line):
            continue

        parsed[int(race_match.group(1))] = {
            "time": lines[index],
            "race_name": race_match.group(2).strip(),
            "boxes": [int(box) for box in boxes_line.split(",")],
        }

    return parsed


def _ordinal_to_position(value: str) -> Optional[int]:
    match = re.search(r"\b([1-9]|10)(?:st|nd|rd|th)\b", str(value or ""), re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def parse_thedogs_result_text(text: str, participants: List[dict]) -> Dict[int, int]:
    """Best-effort parser for rendered TheDogs result text.

    TheDogs markup changes over time, so this parser intentionally uses runner
    names/boxes from the local expert-form CSV and searches nearby rendered text
    for ordinal positions.
    """
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return {}

    by_name = {norm_name(p["dog_name"]): int(p["box_number"]) for p in participants}
    by_box = {int(p["box_number"]): norm_name(p["dog_name"]) for p in participants}
    positions: Dict[int, int] = {}

    for index, line in enumerate(lines):
        direct = re.match(
            r"^\s*([1-9]|10)(?:st|nd|rd|th)?\s+(\d{1,2})[\.\)]?\s+(.+?)\s*$",
            line,
            re.IGNORECASE,
        )
        if direct:
            position = int(direct.group(1))
            box_number = int(direct.group(2))
            if box_number in by_box and 1 <= position <= len(participants):
                positions.setdefault(box_number, position)
                continue

        position = _ordinal_to_position(line)
        if position is None:
            continue

        window_lines = [line]
        for next_line in lines[index + 1 : index + 8]:
            if _ordinal_to_position(next_line) is not None:
                break
            window_lines.append(next_line)
        window = " ".join(window_lines)
        for dog_key, box_number in by_name.items():
            if dog_key and dog_key in norm_name(window):
                positions.setdefault(box_number, position)
                break

    if positions:
        return positions

    # Fallback for compact rows that include dog names and ordinal in one line.
    for line in lines:
        position = _ordinal_to_position(line)
        if position is None:
            continue
        compact = norm_name(line)
        for dog_key, box_number in by_name.items():
            if dog_key and dog_key in compact:
                positions.setdefault(box_number, position)

    return positions


@dataclass
class RaceCandidate:
    race_id: str
    venue: str
    race_number: int
    race_date: str
    race_time: Optional[str]
    start_datetime: Optional[str]
    sportsbet_url: Optional[str]
    csv_path: Path
    participants: List[dict]
    lifecycle_status: str

    @property
    def venue_code(self) -> str:
        return code_from_race_id(self.race_id) or str(self.venue or "").strip()

    @property
    def sportsbet_slug(self) -> Optional[str]:
        return sportsbet_slug_from_url(self.sportsbet_url or "")

    @property
    def thedogs_slug(self) -> Optional[str]:
        key = self.venue_code.strip().upper().replace(" ", "_")
        if key in VENUE_TO_THEDOGS_SLUG:
            return VENUE_TO_THEDOGS_SLUG[key]
        venue = str(self.venue or "").strip().lower().replace("_", "-").replace(" ", "-")
        return venue or None


@dataclass
class SourceResult:
    source: str
    status: str
    source_url: Optional[str]
    positions_by_box: Dict[int, int]
    raw_order: List[int]
    race_name: Optional[str] = None
    error: Optional[str] = None

    @property
    def winner_box(self) -> Optional[int]:
        if not self.positions_by_box:
            return None
        return sorted(self.positions_by_box.items(), key=lambda item: item[1])[0][0]


class TheDogsResultFetcher:
    def __init__(self, driver, wait_seconds: float = 4.0, by=None):
        self.driver = driver
        self.wait_seconds = wait_seconds
        self.by = by or _SeleniumByFallback
        self.site_blocked_error: Optional[str] = None

    def fetch(self, candidate: RaceCandidate) -> SourceResult:
        if self.site_blocked_error:
            return SourceResult(
                source="thedogs_official",
                status="error",
                source_url=None,
                positions_by_box={},
                raw_order=[],
                error=self.site_blocked_error,
            )

        slug = candidate.thedogs_slug
        if not slug:
            return SourceResult(
                source="thedogs_official",
                status="error",
                source_url=None,
                positions_by_box={},
                raw_order=[],
                error="missing_thedogs_venue_slug",
            )

        urls = [
            f"{THEDOGS_BASE}/racing/{slug}/{candidate.race_date}/{candidate.race_number}",
            f"{THEDOGS_BASE}/racing/{slug}/{candidate.race_date}/{candidate.race_number}/results",
        ]
        last_error = None
        for url in urls:
            try:
                self.driver.get(url)
                time.sleep(self.wait_seconds)
                title = (self.driver.title or "").strip()
                text = self.driver.find_element(self.by.TAG_NAME, "body").text
                if title == "403 Forbidden" or text.strip() == "403 Forbidden":
                    last_error = "thedogs_403_forbidden"
                    self.site_blocked_error = last_error
                    break
                positions = parse_thedogs_result_text(text, candidate.participants)
                if positions:
                    ordered_boxes = [
                        box for box, _ in sorted(positions.items(), key=lambda item: item[1])
                    ]
                    return SourceResult(
                        source="thedogs_official",
                        status=RESULTED,
                        source_url=url,
                        positions_by_box=positions,
                        raw_order=ordered_boxes,
                    )
                last_error = "no_thedogs_positions_found"
            except Exception as exc:
                last_error = f"thedogs_error:{type(exc).__name__}"

        return SourceResult(
            source="thedogs_official",
            status="error",
            source_url=urls[0],
            positions_by_box={},
            raw_order=[],
            error=last_error or "thedogs_unknown_error",
        )


class SportsbetResultFetcher:
    def __init__(self, driver, target_date: str, wait_seconds: float = 3.0, by=None):
        self.driver = driver
        self.target_date = target_date
        self.wait_seconds = wait_seconds
        self.by = by or _SeleniumByFallback
        self.category_links: Optional[Dict[str, str]] = None
        self.page_cache: Dict[str, dict] = {}

    def _load_category_links(self) -> Dict[str, str]:
        if self.category_links is not None:
            return self.category_links

        category_url = SPORTSBET_CATEGORY_TEMPLATE.format(date=self.target_date)
        self.driver.get(category_url)
        time.sleep(self.wait_seconds + 1.0)

        links: Dict[str, str] = {}
        for anchor in self.driver.find_elements(self.by.CSS_SELECTOR, "a[href]"):
            href = anchor.get_attribute("href") or ""
            slug = result_slug_from_url(href)
            if slug:
                links.setdefault(slug, href)

        self.category_links = links
        return links

    def fetch(self, candidate: RaceCandidate) -> SourceResult:
        slug = candidate.sportsbet_slug
        if not slug:
            return SourceResult(
                source="sportsbet_results_top4",
                status="error",
                source_url=None,
                positions_by_box={},
                raw_order=[],
                error="missing_sportsbet_slug",
            )

        links = self._load_category_links()
        page_url = links.get(slug)
        if not page_url:
            return SourceResult(
                source="sportsbet_results_top4",
                status="error",
                source_url=None,
                positions_by_box={},
                raw_order=[],
                error=f"sportsbet_result_link_not_found:{slug}",
            )

        if slug not in self.page_cache:
            self.driver.get(page_url)
            time.sleep(self.wait_seconds)
            text = self.driver.find_element(self.by.TAG_NAME, "body").text
            self.page_cache[slug] = {
                "url": self.driver.current_url,
                "results": parse_sportsbet_result_text(text),
            }

        race_result = self.page_cache[slug]["results"].get(candidate.race_number)
        if not race_result:
            return SourceResult(
                source="sportsbet_results_top4",
                status="error",
                source_url=self.page_cache[slug]["url"],
                positions_by_box={},
                raw_order=[],
                error=f"sportsbet_race_result_not_found:R{candidate.race_number}",
            )

        boxes = race_result["boxes"]
        return SourceResult(
            source="sportsbet_results_top4",
            status=PARTIAL_SPORTSBET_RESULTS,
            source_url=self.page_cache[slug]["url"],
            positions_by_box={box: position for position, box in enumerate(boxes, start=1)},
            raw_order=boxes,
            race_name=race_result.get("race_name"),
        )


def resolve_csv_path(upcoming_dir: Path, row: sqlite3.Row) -> Optional[Path]:
    race_id = str(row["race_id"])
    exact = upcoming_dir / f"{race_id}.csv"
    if exact.exists():
        return exact

    code = code_from_race_id(race_id) or str(row["venue"] or "").strip()
    if code and row["race_number"]:
        candidate = upcoming_dir / f"Race {int(row['race_number'])} - {code} - {row['race_date']}.csv"
        if candidate.exists():
            return candidate

    return None


def _row_dict(row: sqlite3.Row) -> dict:
    return {key: row[key] for key in row.keys()}


def _lifecycle_record_for_row(row: sqlite3.Row) -> dict:
    data = _row_dict(row)
    if not str(data.get("race_time") or "").strip() and data.get("start_datetime"):
        start_text = str(data.get("start_datetime") or "")
        time_match = re.search(r"T(\d{2}:\d{2})", start_text)
        if time_match:
            data["race_time"] = time_match.group(1)
    return data


def jumped_or_already_resulted(row: sqlite3.Row, now: Optional[datetime] = None) -> tuple[bool, str]:
    lifecycle = classify_race_record(_lifecycle_record_for_row(row), now=now)
    return lifecycle.status != UPCOMING_NOT_JUMPED, lifecycle.status


def load_candidates(
    db_path: Path,
    target_date: str,
    upcoming_dir: Path,
    race_ids: Iterable[str],
    now: Optional[datetime] = None,
) -> tuple[List[RaceCandidate], List[dict]]:
    race_id_filter = [race_id for race_id in race_ids if race_id]
    params: List[object] = [target_date]
    where = ["race_date = ?"]
    if race_id_filter:
        placeholders = ",".join(["?"] * len(race_id_filter))
        where.append(f"race_id IN ({placeholders})")
        params.extend(race_id_filter)

    query = f"""
        SELECT race_id, venue, race_number, race_date, race_time, start_datetime,
               sportsbet_url, results_status, winner_name
        FROM race_metadata
        WHERE {' AND '.join(where)}
        ORDER BY venue, race_number, race_id
    """

    candidates: List[RaceCandidate] = []
    skipped: List[dict] = []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        for row in conn.execute(query, params).fetchall():
            eligible, lifecycle_status = jumped_or_already_resulted(row, now=now)
            if not eligible:
                skipped.append(
                    {
                        "race_id": row["race_id"],
                        "reason": f"race_not_jumped:{lifecycle_status}",
                    }
                )
                continue
            csv_path = resolve_csv_path(upcoming_dir, row)
            if not csv_path:
                skipped.append({"race_id": row["race_id"], "reason": "no_local_csv"})
                continue
            participants = parse_participants_from_csv(csv_path)
            if not participants:
                skipped.append(
                    {"race_id": row["race_id"], "reason": "no_participants_from_csv"}
                )
                continue
            candidates.append(
                RaceCandidate(
                    race_id=row["race_id"],
                    venue=row["venue"],
                    race_number=int(row["race_number"]),
                    race_date=row["race_date"],
                    race_time=row["race_time"],
                    start_datetime=row["start_datetime"],
                    sportsbet_url=row["sportsbet_url"],
                    csv_path=csv_path,
                    participants=participants,
                    lifecycle_status=lifecycle_status,
                )
            )
    finally:
        conn.close()

    return candidates, skipped


def winner_odds_for_box(conn: sqlite3.Connection, race_id: str, box_number: int) -> Optional[float]:
    try:
        row = conn.execute(
            """
            SELECT odds_decimal
            FROM live_odds
            WHERE race_id = ? AND market_type = 'win' AND CAST(box_number AS INTEGER) = ?
              AND (is_current = 1 OR is_current IS NULL)
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (race_id, box_number),
        ).fetchone()
        if row and row[0] is not None:
            return float(row[0])
    except Exception:
        return None
    return None


def write_result(
    conn: sqlite3.Connection,
    candidate: RaceCandidate,
    result: SourceResult,
    attempted_sources: List[SourceResult],
    dry_run: bool,
) -> dict:
    box_to_name = {
        int(participant["box_number"]): participant["dog_name"]
        for participant in candidate.participants
    }
    winner_box = result.winner_box
    winner_name = box_to_name.get(winner_box) if winner_box is not None else None
    winner_odds = (
        winner_odds_for_box(conn, candidate.race_id, winner_box)
        if winner_box is not None
        else None
    )
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    source_errors = [
        f"{attempt.source}:{attempt.error}"
        for attempt in attempted_sources
        if attempt.error
    ]
    note_parts = [
        f"result_source={result.source}",
        f"result_status={result.status}",
    ]
    if source_errors:
        note_parts.append("source_errors=" + "|".join(source_errors))
    if result.raw_order:
        note_parts.append("box_order=" + ",".join(str(box) for box in result.raw_order))
    data_quality_note = "; ".join(note_parts)

    if dry_run:
        return {
            "race_id": candidate.race_id,
            "status": result.status,
            "source": result.source,
            "winner_name": winner_name,
            "box_order": result.raw_order,
            "dry_run": True,
        }

    for participant in candidate.participants:
        box_number = int(participant["box_number"])
        dog_name = participant["dog_name"]
        finish_position = result.positions_by_box.get(box_number)
        existing = conn.execute(
            """
            SELECT id FROM dog_race_data
            WHERE race_id = ? AND CAST(box_number AS INTEGER) = ?
            LIMIT 1
            """,
            (candidate.race_id, box_number),
        ).fetchone()
        if existing:
            conn.execute(
                """
                UPDATE dog_race_data
                SET dog_name = ?, dog_clean_name = ?, finish_position = ?, placing = ?,
                    scraped_finish_position = ?, extraction_timestamp = ?, data_source = ?
                WHERE id = ?
                """,
                (
                    dog_name,
                    dog_name,
                    finish_position,
                    finish_position,
                    str(finish_position) if finish_position is not None else None,
                    now,
                    result.source,
                    existing[0],
                ),
            )
        else:
            conn.execute(
                f"""
                INSERT INTO {TARGET_TABLE} (
                    race_id, dog_name, dog_clean_name, box_number, finish_position,
                    placing, scraped_finish_position, extraction_timestamp, data_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.race_id,
                    dog_name,
                    dog_name,
                    box_number,
                    finish_position,
                    finish_position,
                    str(finish_position) if finish_position is not None else None,
                    now,
                    result.source,
                ),
            )

    conn.execute(
        """
        UPDATE race_metadata
        SET winner_name = ?,
            winner_odds = COALESCE(?, winner_odds),
            winner_source = ?,
            results_status = ?,
            scraping_attempts = COALESCE(scraping_attempts, 0) + 1,
            last_scraped_at = ?,
            extraction_timestamp = COALESCE(extraction_timestamp, ?),
            actual_field_size = COALESCE(actual_field_size, ?),
            field_size = COALESCE(field_size, ?),
            url = COALESCE(?, url),
            parse_confidence = ?,
            data_quality_note = ?
        WHERE race_id = ?
        """,
        (
            winner_name,
            winner_odds,
            result.source,
            result.status,
            now,
            now,
            len(candidate.participants),
            len(candidate.participants),
            result.source_url,
            1.0 if result.source == "thedogs_official" else 0.9,
            data_quality_note,
            candidate.race_id,
        ),
    )

    return {
        "race_id": candidate.race_id,
        "status": result.status,
        "source": result.source,
        "winner_name": winner_name,
        "box_order": result.raw_order,
    }


def backup_db(db_path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = (
        db_path.resolve().parent
        / "archive"
        / "db_backups"
        / f"{timestamp}_pre_results_ingest_official_first"
    )
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / "pre_op.sqlite"
    try:
        source_uri = f"{db_path.resolve().as_uri()}?mode=ro"
        with sqlite3.connect(source_uri, uri=True) as source:
            with sqlite3.connect(backup_path) as destination:
                source.backup(destination)
    except Exception:
        shutil.copy2(db_path, backup_path)
    return backup_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest race results with TheDogs official-first source order"
    )
    parser.add_argument("--db", default="greyhound_racing_data.db", help="SQLite DB path")
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Race date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--upcoming-dir",
        default="upcoming_races",
        help="Directory containing local upcoming expert-form CSVs",
    )
    parser.add_argument(
        "--race-id",
        action="append",
        default=[],
        help="Optional race_id filter. Can be repeated.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Parse but do not write")
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Show browser while fetching pages",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    db_path = Path(args.db)
    upcoming_dir = Path(args.upcoming_dir)

    if not db_path.exists():
        print(f"ERROR database not found: {db_path}", file=sys.stderr)
        return 2
    if not upcoming_dir.exists():
        print(f"ERROR upcoming directory not found: {upcoming_dir}", file=sys.stderr)
        return 2

    candidates, skipped = load_candidates(db_path, args.date, upcoming_dir, args.race_id)
    print(f"Candidates: {len(candidates)}")
    if skipped:
        print(f"Skipped before fetch: {len(skipped)}")
        for item in skipped:
            print(f"SKIPPED {item}")
    if not candidates:
        return 0

    backup_path = None
    if not args.dry_run:
        backup_path = backup_db(db_path)
        print(f"Backup: {backup_path}")

    from drivers import get_chrome_driver
    from selenium.webdriver.common.by import By

    driver = get_chrome_driver(headless=not args.no_headless)
    ingested: List[dict] = []
    failed: List[dict] = []
    try:
        thedogs = TheDogsResultFetcher(driver, by=By)
        sportsbet = SportsbetResultFetcher(driver, args.date, by=By)

        conn = sqlite3.connect(db_path)
        try:
            conn.execute("BEGIN")
            for candidate in candidates:
                attempts: List[SourceResult] = []
                official = thedogs.fetch(candidate)
                attempts.append(official)
                chosen = official if official.positions_by_box else None
                if chosen is None:
                    fallback = sportsbet.fetch(candidate)
                    attempts.append(fallback)
                    chosen = fallback if fallback.positions_by_box else None

                if chosen is None:
                    failed.append(
                        {
                            "race_id": candidate.race_id,
                            "errors": [a.error for a in attempts if a.error],
                        }
                    )
                    continue

                ingested.append(write_result(conn, candidate, chosen, attempts, args.dry_run))

            if args.dry_run:
                conn.rollback()
            else:
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    finally:
        driver.quit()

    print("\nINGEST SUMMARY")
    print(f"ingested={len(ingested)} failed={len(failed)} dry_run={args.dry_run}")
    for item in ingested:
        print(
            "INGESTED "
            f"{item['race_id']} source={item['source']} status={item['status']} "
            f"winner={item['winner_name']} boxes={item['box_order']}"
        )
    for item in failed:
        print(f"FAILED {item}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
