#!/usr/bin/env python3
"""Append strictly matched TheDogs results for sealed Sportsbet odds races.

The input identity is read from ``live_odds`` without modifying it.  Official
HTML is retained byte-for-byte in a content-addressed artifact.  Evidence is
only appended when date, venue slug, race number, and the complete (box, dog)
runner set agree.  Default operation is a report-only dry-run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ingest_results_for_date import (
    THEDOGS_PUBLIC_HEADERS,
    VENUE_TO_THEDOGS_SLUG,
    parse_thedogs_result_html_runner_rows,
)


OFFICIAL_SOURCE = "thedogs_official"
RACE_TABLE = "autonomous_official_result_evidence_races"
RUNNER_TABLE = "autonomous_official_result_evidence_runners"
RACE_ID_RE = re.compile(r"^Race\s+(\d+)\s+-\s+(.+?)\s+-\s+(\d{4}-\d{2}-\d{2})$")


class MatchRejected(ValueError):
    """Raised when preserved evidence cannot support one exact join."""


def canonical_name(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _one(values: Iterable[Any], reason: str) -> Any:
    unique = {value for value in values if value not in (None, "")}
    if len(unique) != 1:
        raise MatchRejected(reason)
    return next(iter(unique))


@dataclass(frozen=True)
class SealedRace:
    race_id: str
    race_date: str
    venue: str
    venue_slug: str
    race_number: int
    sportsbet_url: str
    runners: tuple[tuple[int, str], ...]
    odds_rows_sha256: str
    scheduled_start: datetime

    @property
    def official_url(self) -> str:
        return (
            f"https://www.thedogs.com.au/racing/{self.venue_slug}/"
            f"{self.race_date}/{self.race_number}/results?trial=false"
        )


def _venue_slug(venue: str, sportsbet_url: str, race_number: int) -> str:
    key = venue.strip().upper().replace(" ", "_")
    mapped = VENUE_TO_THEDOGS_SLUG.get(key)
    parsed = urlparse(sportsbet_url)
    sportsbet_match = re.search(
        r"^/greyhound-racing/australia-nz/([^/]+)/race-(\d+)(?:-|$)", parsed.path, re.I
    )
    if (
        parsed.scheme != "https"
        or parsed.hostname not in {"sportsbet.com.au", "www.sportsbet.com.au"}
        or not sportsbet_match
        or int(sportsbet_match.group(2)) != race_number
    ):
        raise MatchRejected("sportsbet_url_identity_mismatch")
    sportsbet_slug = sportsbet_match.group(1).lower()
    # Only an identical preserved venue/market slug is accepted without an
    # explicit mapping.  Fuzzy venue aliases are intentionally unsupported.
    venue_slug = re.sub(r"[^a-z0-9]+", "-", venue.lower()).strip("-")
    official_slug = mapped or venue_slug
    if official_slug == sportsbet_slug:
        return official_slug
    raise MatchRejected("venue_has_no_deterministic_thedogs_slug")


def seal_race_from_db(db_path: Path, race_id: str) -> SealedRace:
    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.execute("PRAGMA query_only=ON")
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT race_id, race_date, venue, race_number, source_url,
                   box_number, dog_name, dog_clean_name, odds_decimal,
                   capture_timestamp, capture_mode, race_time,
                   market_type, source, odds_level,
                   sportsbet_box_source
            FROM live_odds
            WHERE race_id = ? AND lower(COALESCE(market_type, '')) = 'win'
            ORDER BY capture_timestamp, box_number, id
            """,
            (race_id,),
        ).fetchall()
    if not rows:
        raise MatchRejected("missing_sportsbet_win_odds")
    if any(str(row["source"] or "").lower() != "sportsbet" for row in rows):
        raise MatchRejected("non_sportsbet_odds_source")
    if any(
        not re.fullmatch(r"autonomous_prejump_t\d+m", str(row["capture_mode"] or ""))
        for row in rows
    ):
        raise MatchRejected("non_prejump_odds_capture")
    race_date = str(_one((str(r["race_date"] or "")[:10] for r in rows), "ambiguous_race_date"))
    venue = str(_one((str(r["venue"] or "").strip() for r in rows), "ambiguous_venue"))
    race_number = int(_one((r["race_number"] for r in rows), "ambiguous_race_number"))
    sportsbet_url = str(
        _one((str(r["source_url"] or "").strip() for r in rows), "ambiguous_sportsbet_url")
    )
    race_time = str(_one((str(r["race_time"] or "").strip() for r in rows), "ambiguous_race_time"))
    match = RACE_ID_RE.match(race_id)
    if (
        not match
        or int(match.group(1)) != race_number
        or match.group(3) != race_date
        or match.group(2) != venue
    ):
        raise MatchRejected("race_id_components_disagree")
    if not all(
        r["capture_timestamp"] and r["odds_decimal"] and float(r["odds_decimal"]) > 1 for r in rows
    ):
        raise MatchRejected("incomplete_odds_provenance")
    try:
        start_clock = time.fromisoformat(race_time)
        captures = []
        for row in rows:
            captured = datetime.fromisoformat(str(row["capture_timestamp"]))
            if captured.tzinfo is None:
                raise ValueError
            captures.append(captured)
    except (TypeError, ValueError):
        raise MatchRejected("invalid_temporal_odds_provenance") from None
    for captured in captures:
        scheduled = datetime.combine(
            datetime.fromisoformat(race_date).date(), start_clock, tzinfo=captured.tzinfo
        )
        if captured >= scheduled:
            raise MatchRejected("odds_capture_not_before_scheduled_start")
    latest = max(captures)
    snapshot = [r for r, captured in zip(rows, captures) if captured == latest]
    runners: list[tuple[int, str]] = []
    for row in snapshot:
        box = int(row["box_number"] or 0)
        name = str(row["dog_name"] or row["dog_clean_name"] or "").strip()
        if (
            box <= 0
            or not name
            or str(row["odds_level"] or "dog").lower() not in {"", "dog", "runner"}
        ):
            raise MatchRejected("invalid_latest_odds_runner")
        runners.append((box, name))
    keys = [(box, canonical_name(name)) for box, name in runners]
    if (
        len(keys) < 2
        or len(keys) != len(set(keys))
        or len({b for b, _ in keys}) != len(keys)
        or len({n for _, n in keys}) != len(keys)
    ):
        raise MatchRejected("ambiguous_latest_odds_runner_set")
    sealed_rows = [dict(row) for row in rows]
    digest = sha256_bytes(
        json.dumps(sealed_rows, sort_keys=True, separators=(",", ":"), default=str).encode()
    )
    return SealedRace(
        race_id=race_id,
        race_date=race_date,
        venue=venue,
        venue_slug=_venue_slug(venue, sportsbet_url, race_number),
        race_number=race_number,
        sportsbet_url=sportsbet_url,
        runners=tuple(sorted(runners)),
        odds_rows_sha256=digest,
        scheduled_start=datetime.combine(
            datetime.fromisoformat(race_date).date(), start_clock, tzinfo=captures[0].tzinfo
        ),
    )


def validate_official_html(sealed: SealedRace, source_url: str, raw: bytes) -> list[dict[str, Any]]:
    parsed = urlparse(source_url)
    expected_path = f"/racing/{sealed.venue_slug}/{sealed.race_date}/{sealed.race_number}/results"
    if (
        parsed.scheme != "https"
        or parsed.hostname not in {"thedogs.com.au", "www.thedogs.com.au"}
        or parsed.path.rstrip("/") != expected_path
    ):
        raise MatchRejected("official_url_identity_mismatch")
    rows = parse_thedogs_result_html_runner_rows(raw.decode("utf-8", errors="strict"))
    if not rows:
        raise MatchRejected("official_result_missing")
    if any(row.get("finish_position") is None or row.get("status") for row in rows):
        raise MatchRejected("partial_or_terminal_official_result")
    official = [
        (int(r["box_number"]), str(r.get("dog_name") or "").strip(), int(r["finish_position"]))
        for r in rows
    ]
    keys = [(box, canonical_name(name)) for box, name, _ in official]
    sealed_keys = [(box, canonical_name(name)) for box, name in sealed.runners]
    positions = [position for _, _, position in official]
    if not all(name for _, name in keys) or len(keys) != len(set(keys)):
        raise MatchRejected("ambiguous_official_runner_set")
    if sorted(keys) != sorted(sealed_keys):
        raise MatchRejected("runner_set_mismatch")
    if sorted(positions) != list(range(1, len(official) + 1)):
        raise MatchRejected("incomplete_finish_positions")
    return [
        {"box_number": box, "dog_name": name, "finish_position": position}
        for box, name, position in sorted(official, key=lambda item: item[2])
    ]


def fetch_raw(url: str) -> tuple[str, bytes, str]:
    fetched_at = datetime.now(timezone.utc).isoformat()
    response = requests.get(url, headers=THEDOGS_PUBLIC_HEADERS, timeout=30, allow_redirects=False)
    if response.status_code != 200:
        raise MatchRejected(f"official_http_status:{response.status_code}")
    if response.url != url:
        raise MatchRejected("official_redirect_not_accepted")
    return fetched_at, response.content, response.url


def write_raw_artifact(output_dir: Path, raw: bytes) -> tuple[Path, str]:
    digest = sha256_bytes(raw)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"thedogs_official_{digest}.html"
    if path.exists() and path.read_bytes() != raw:
        raise MatchRejected("raw_artifact_hash_collision")
    if not path.exists():
        path.write_bytes(raw)
    if sha256_bytes(path.read_bytes()) != digest:
        raise MatchRejected("raw_artifact_hash_verification_failed")
    return path, digest


def _ensure_tables(conn: sqlite3.Connection) -> None:
    required = {RACE_TABLE, RUNNER_TABLE}
    present = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not required <= present:
        raise MatchRejected("official_evidence_tables_missing")


def _verified_existing_bundle(
    conn: sqlite3.Connection,
    sealed: SealedRace,
    runners: Sequence[Mapping[str, Any]],
    *,
    source_url: str,
    raw_sha256: str,
) -> bool:
    race_rows = conn.execute(
        f"""SELECT race_date, venue, race_number, source, source_url, status,
                   winner_name, winner_box, position_count, participant_count,
                   box_order_json, participant_source, captured_at,
                   source_artifact_dir, row_json
            FROM {RACE_TABLE} WHERE race_id = ?""",
        (sealed.race_id,),
    ).fetchall()
    runner_rows = conn.execute(
        f"""SELECT race_date, venue, race_number, source, source_url, box_number,
                   dog_name, finish_position, is_winner, captured_at,
                   source_artifact_dir, row_json
            FROM {RUNNER_TABLE} WHERE race_id = ?""",
        (sealed.race_id,),
    ).fetchall()
    if not race_rows and not runner_rows:
        return False
    if len(race_rows) != 1 or len(runner_rows) != len(runners):
        raise MatchRejected("conflicting_or_incomplete_existing_evidence")

    winner = next(r for r in runners if int(r["finish_position"]) == 1)
    expected_runner_signatures = {
        (int(r["box_number"]), canonical_name(r["dog_name"]), int(r["finish_position"]))
        for r in runners
    }
    race = race_rows[0]
    try:
        race_json = json.loads(race[-1])
        stored_fetched_at = datetime.fromisoformat(str(race_json.get("official_fetched_at")))
        if stored_fetched_at.tzinfo is None or stored_fetched_at <= sealed.scheduled_start:
            raise MatchRejected("conflicting_or_unverifiable_existing_evidence")
        stored_artifact = Path(str(race_json.get("raw_artifact_path") or ""))
        if (
            not stored_artifact.is_file()
            or sha256_bytes(stored_artifact.read_bytes()) != raw_sha256
            or race[12] != race_json.get("official_fetched_at")
            or race[13] != str(stored_artifact.parent)
        ):
            raise MatchRejected("conflicting_or_unverifiable_existing_evidence")
        stored_runner_rows = set()
        for row in runner_rows:
            row_json = json.loads(row[-1])
            if (
                row_json.get("identity_contract") != "sealed_sportsbet_box_name_v1"
                or row_json.get("raw_sha256") != raw_sha256
                or row_json.get("sealed_odds_rows_sha256") != sealed.odds_rows_sha256
                or row_json.get("official_source_url") != source_url
                or row_json.get("sportsbet_source_url") != sealed.sportsbet_url
                or row_json.get("race_id") != sealed.race_id
                or int(row_json.get("box_number")) != int(row[5])
                or canonical_name(row_json.get("dog_name")) != canonical_name(row[6])
                or int(row_json.get("finish_position")) != int(row[7])
                or row_json.get("official_fetched_at") != race_json.get("official_fetched_at")
                or row_json.get("raw_artifact_path") != str(stored_artifact)
                or row[9] != race_json.get("official_fetched_at")
                or row[10] != str(stored_artifact.parent)
            ):
                raise MatchRejected("conflicting_or_unverifiable_existing_evidence")
            stored_runner_rows.add(
                (
                    row[0],
                    row[1],
                    int(row[2]),
                    row[3],
                    row[4],
                    int(row[5]),
                    canonical_name(row[6]),
                    int(row[7]),
                    int(row[8]),
                )
            )
    except (TypeError, ValueError, json.JSONDecodeError, IndexError):
        raise MatchRejected("conflicting_or_unverifiable_existing_evidence") from None

    expected_race = (
        sealed.race_date,
        sealed.venue,
        sealed.race_number,
        OFFICIAL_SOURCE,
        source_url,
        "resulted",
        winner["dog_name"],
        winner["box_number"],
        len(runners),
        len(runners),
        json.dumps([int(r["box_number"]) for r in runners]),
        "sealed_sportsbet_latest_win_snapshot",
    )
    expected_runner_rows = {
        (
            sealed.race_date,
            sealed.venue,
            sealed.race_number,
            OFFICIAL_SOURCE,
            source_url,
            int(r["box_number"]),
            canonical_name(r["dog_name"]),
            int(r["finish_position"]),
            int(int(r["finish_position"]) == 1),
        )
        for r in runners
    }
    try:
        race_json_runners = {
            (int(r["box_number"]), canonical_name(r["dog_name"]), int(r["finish_position"]))
            for r in race_json.get("runners", [])
        }
    except (TypeError, ValueError, KeyError):
        raise MatchRejected("conflicting_or_unverifiable_existing_evidence") from None
    if (
        tuple(race[:12]) != expected_race
        or race_json.get("identity_contract") != "sealed_sportsbet_box_name_v1"
        or race_json.get("raw_sha256") != raw_sha256
        or race_json.get("sealed_odds_rows_sha256") != sealed.odds_rows_sha256
        or race_json.get("official_source_url") != source_url
        or race_json.get("sportsbet_source_url") != sealed.sportsbet_url
        or race_json.get("race_id") != sealed.race_id
        or race_json_runners != expected_runner_signatures
        or stored_runner_rows != expected_runner_rows
    ):
        raise MatchRejected("conflicting_or_unverifiable_existing_evidence")
    return True


def append_evidence(
    conn: sqlite3.Connection,
    sealed: SealedRace,
    runners: Sequence[Mapping[str, Any]],
    *,
    source_url: str,
    fetched_at: str,
    artifact_path: Path,
    raw_sha256: str,
) -> tuple[int, int]:
    _ensure_tables(conn)
    if _verified_existing_bundle(
        conn, sealed, runners, source_url=source_url, raw_sha256=raw_sha256
    ):
        return 0, 0
    provenance = {
        "identity_contract": "sealed_sportsbet_box_name_v1",
        "sportsbet_source_url": sealed.sportsbet_url,
        "sealed_odds_rows_sha256": sealed.odds_rows_sha256,
        "official_source_url": source_url,
        "official_fetched_at": fetched_at,
        "raw_artifact_path": str(artifact_path),
        "raw_sha256": raw_sha256,
    }
    winner = next(r for r in runners if int(r["finish_position"]) == 1)
    box_order = [int(r["box_number"]) for r in runners]
    race_row = {**provenance, "race_id": sealed.race_id, "runners": list(runners)}
    before = conn.total_changes
    conn.execute(
        f"""INSERT INTO {RACE_TABLE}
        (race_id,race_date,venue,race_number,race_time,start_datetime,source,source_url,status,winner_name,winner_box,position_count,participant_count,box_order_json,participant_source,captured_at,source_artifact_dir,row_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            sealed.race_id,
            sealed.race_date,
            sealed.venue,
            sealed.race_number,
            None,
            None,
            OFFICIAL_SOURCE,
            source_url,
            "resulted",
            winner["dog_name"],
            winner["box_number"],
            len(runners),
            len(runners),
            json.dumps(box_order),
            "sealed_sportsbet_latest_win_snapshot",
            fetched_at,
            str(artifact_path.parent),
            json.dumps(race_row, sort_keys=True),
        ),
    )
    race_inserted = conn.total_changes - before
    runner_inserted = 0
    for runner in runners:
        row_json = json.dumps(
            {**provenance, **dict(runner), "race_id": sealed.race_id}, sort_keys=True
        )
        before = conn.total_changes
        conn.execute(
            f"""INSERT INTO {RUNNER_TABLE}
            (race_id,race_date,venue,race_number,source,source_url,box_number,dog_name,finish_position,is_winner,captured_at,source_artifact_dir,row_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                sealed.race_id,
                sealed.race_date,
                sealed.venue,
                sealed.race_number,
                OFFICIAL_SOURCE,
                source_url,
                runner["box_number"],
                runner["dog_name"],
                runner["finish_position"],
                int(runner["finish_position"] == 1),
                fetched_at,
                str(artifact_path.parent),
                row_json,
            ),
        )
        runner_inserted += conn.total_changes - before
    if race_inserted != 1 or runner_inserted != len(runners):
        raise MatchRejected("partial_evidence_insert")
    return race_inserted, runner_inserted


def run(
    *,
    db_path: Path,
    race_id: str,
    output_dir: Path,
    execute: bool,
    raw_path: Path | None = None,
    source_url: str | None = None,
    raw_fetched_at: str | None = None,
) -> dict[str, Any]:
    sealed = seal_race_from_db(db_path, race_id)
    if raw_path:
        if not source_url or not raw_fetched_at:
            raise MatchRejected("raw_html_requires_source_url_and_fetched_at")
        try:
            parsed_fetched_at = datetime.fromisoformat(raw_fetched_at)
            if parsed_fetched_at.tzinfo is None:
                raise ValueError
        except (TypeError, ValueError):
            raise MatchRejected("invalid_official_fetched_at") from None
        raw = raw_path.read_bytes()
        fetched_at = parsed_fetched_at.isoformat()
        resolved_url = source_url
    else:
        if raw_fetched_at:
            raise MatchRejected("fetched_at_only_valid_with_raw_html")
        fetched_at, raw, resolved_url = fetch_raw(source_url or sealed.official_url)
        parsed_fetched_at = datetime.fromisoformat(fetched_at)
    if parsed_fetched_at <= sealed.scheduled_start:
        raise MatchRejected("official_result_fetched_before_scheduled_start")
    runners = validate_official_html(sealed, resolved_url, raw)
    artifact_path, raw_sha = write_raw_artifact(output_dir, raw)
    inserted = (0, 0)
    if execute:
        try:
            with sqlite3.connect(db_path) as conn:
                inserted = append_evidence(
                    conn,
                    sealed,
                    runners,
                    source_url=resolved_url,
                    fetched_at=fetched_at,
                    artifact_path=artifact_path,
                    raw_sha256=raw_sha,
                )
                conn.commit()
        except sqlite3.IntegrityError as exc:
            raise MatchRejected("conflicting_evidence_constraint") from exc
    return {
        "status": (
            "APPENDED"
            if any(inserted)
            else ("NOOP_ALREADY_PRESENT" if execute else "ACCEPTED_DRY_RUN")
        ),
        "race_id": sealed.race_id,
        "canonical_identity": [sealed.race_date, sealed.venue_slug, sealed.race_number],
        "runner_count": len(runners),
        "sportsbet_source_url": sealed.sportsbet_url,
        "sealed_odds_rows_sha256": sealed.odds_rows_sha256,
        "official_source_url": resolved_url,
        "official_fetched_at": fetched_at,
        "raw_artifact_path": str(artifact_path),
        "raw_sha256": raw_sha,
        "execute": execute,
        "inserted_race_rows": inserted[0],
        "inserted_runner_rows": inserted[1],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--race-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-html", type=Path)
    parser.add_argument("--source-url")
    parser.add_argument("--fetched-at")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        report = run(
            db_path=args.db,
            race_id=args.race_id,
            output_dir=args.output_dir,
            execute=args.execute,
            raw_path=args.raw_html,
            source_url=args.source_url,
            raw_fetched_at=args.fetched_at,
        )
        code = 0
    except MatchRejected as exc:
        report = {
            "status": "REJECTED",
            "race_id": args.race_id,
            "reason": str(exc),
            "execute": args.execute,
        }
        code = 2
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
