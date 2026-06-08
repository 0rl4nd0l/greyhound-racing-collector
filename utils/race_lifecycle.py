"""Race lifecycle classification for form-guide CSVs and live race records.

Directory names are not lifecycle evidence.  This module classifies a race from
target-race metadata, optional jump time, and explicit official-result evidence.
Embedded historical form-guide columns such as PLC/TIME/SP are not treated as
target-race results.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


UPCOMING_NOT_JUMPED = "upcoming_not_jumped"
JUMPED_PENDING_RESULTS = "jumped_pending_results"
RESULTED = "resulted"
STALE_FORM_GUIDE = "stale_form_guide"

LIVE_TARGET_STATUSES = {UPCOMING_NOT_JUMPED}


@dataclass(frozen=True)
class RaceLifecycle:
    status: str
    status_reason: str
    race_date: Optional[str] = None
    venue: Optional[str] = None
    race_number: Optional[int] = None
    jump_time: Optional[str] = None
    jump_datetime: Optional[str] = None
    has_official_result: bool = False
    result_evidence: Optional[str] = None
    source_path: Optional[str] = None

    @property
    def is_live_target(self) -> bool:
        return self.status in LIVE_TARGET_STATUSES

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["is_live_target"] = self.is_live_target
        return data


def _melbourne_tz():
    if ZoneInfo is None:
        return None
    try:
        return ZoneInfo("Australia/Melbourne")
    except Exception:
        return None


def melbourne_now() -> datetime:
    tz = _melbourne_tz()
    return datetime.now(tz) if tz else datetime.now()


def _parse_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "none", "null", "unknown", "tba", "tbd"}:
        return None
    raw = raw.replace("_", " ")
    for fmt in (
        "%Y-%m-%d",
        "%d %B %Y",
        "%d %b %Y",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%Y/%m/%d",
        "%Y%m%d",
    ):
        try:
            return datetime.strptime(raw, fmt).date()
        except Exception:
            continue
    return None


def _parse_time(value: Any) -> Optional[time]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "none", "null", "unknown", "tba", "tbd"}:
        return None
    cleaned = raw.upper().replace(".", ":")
    cleaned = re.sub(r"\s+", " ", cleaned)
    for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p", "%H%M"):
        try:
            return datetime.strptime(cleaned, fmt).time()
        except Exception:
            continue
    return None


def _combine_melbourne(race_day: date, jump: Optional[time]) -> Optional[datetime]:
    if jump is None:
        return None
    tz = _melbourne_tz()
    return datetime(
        race_day.year,
        race_day.month,
        race_day.day,
        jump.hour,
        jump.minute,
        tzinfo=tz,
    )


def _clean_venue(value: Any) -> Optional[str]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "none", "null", "unknown"}:
        return None
    return raw.replace("/", "_").upper()


def _clean_race_number(value: Any) -> Optional[int]:
    if value is None:
        return None
    match = re.search(r"\d+", str(value))
    if not match:
        return None
    try:
        return int(match.group(0))
    except Exception:
        return None


def _normalise_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


FILENAME_PATTERNS = (
    re.compile(
        r"race\s*(?P<race_number>\d+)\s*-\s*(?P<venue>.+?)\s*-\s*"
        r"(?P<race_date>\d{4}-\d{2}-\d{2}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})$",
        re.IGNORECASE,
    ),
    re.compile(
        r"race[_\s]*(?P<race_number>\d+)[_\s-]+(?P<venue>[A-Za-z0-9_/-]+)[_\s-]+"
        r"(?P<race_date>\d{4}-\d{2}-\d{2})$",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<venue>[A-Za-z0-9_/-]+)[_\s-]+race[_\s]*(?P<race_number>\d+)[_\s-]+"
        r"(?P<race_date>\d{4}-\d{2}-\d{2})$",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<race_date>\d{4}-\d{2}-\d{2})[_\s-]+(?P<venue>[A-Za-z0-9_/-]+)[_\s-]+R?"
        r"(?P<race_number>\d+)$",
        re.IGNORECASE,
    ),
)


def extract_target_metadata_from_filename(path_or_name: str | os.PathLike[str]) -> dict[str, Any]:
    stem = Path(path_or_name).stem
    stem = re.sub(r"^\d{8,14}_", "", stem)
    for pattern in FILENAME_PATTERNS:
        match = pattern.search(stem)
        if not match:
            continue
        race_day = _parse_date(match.group("race_date"))
        race_number = _clean_race_number(match.group("race_number"))
        venue = _clean_venue(match.group("venue"))
        if race_day and race_number is not None and venue:
            return {
                "race_date": race_day.isoformat(),
                "venue": venue,
                "race_number": race_number,
                "metadata_source": "filename",
            }
    return {}


def _read_sample_rows(csv_path: str | os.PathLike[str], max_rows: int = 25) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with open(csv_path, "r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            sample = handle.read(8192)
            handle.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",|\t;")
            except Exception:
                dialect = csv.excel
            reader = csv.DictReader(handle, dialect=dialect)
            headers = [h or "" for h in (reader.fieldnames or [])]
            rows: list[dict[str, str]] = []
            for idx, row in enumerate(reader):
                if idx >= max_rows:
                    break
                rows.append({str(k or ""): "" if v is None else str(v) for k, v in row.items()})
            return headers, rows
    except Exception:
        return [], []


RACE_DATE_KEYS = {"race_date", "race date", "meeting_date", "meeting date"}
VENUE_KEYS = {"venue", "track", "meeting_venue", "meeting venue", "venue_code"}
RACE_NO_KEYS = {"race_number", "race number", "race_no", "race no", "race"}
JUMP_TIME_KEYS = {
    "race_time",
    "race time",
    "jump_time",
    "jump time",
    "start_time",
    "start time",
    "scheduled_time",
    "scheduled time",
}


def _first_value(row: Mapping[str, Any], keys: Iterable[str]) -> Optional[Any]:
    lower_map = {_normalise_key(k): k for k in row.keys()}
    wanted = {_normalise_key(k) for k in keys}
    for key in wanted:
        original = lower_map.get(key)
        if original is None:
            continue
        value = row.get(original)
        if value is not None and str(value).strip():
            return value
    return None


def _extract_target_metadata_from_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    for row in rows:
        race_day = _parse_date(_first_value(row, RACE_DATE_KEYS))
        venue = _clean_venue(_first_value(row, VENUE_KEYS))
        race_number = _clean_race_number(_first_value(row, RACE_NO_KEYS))
        jump = _parse_time(_first_value(row, JUMP_TIME_KEYS))
        if race_day or venue or race_number is not None or jump:
            data: dict[str, Any] = {"metadata_source": "csv_race_headers"}
            if race_day:
                data["race_date"] = race_day.isoformat()
            if venue:
                data["venue"] = venue
            if race_number is not None:
                data["race_number"] = race_number
            if jump:
                data["jump_time"] = jump.strftime("%H:%M")
            return data
    return {}


def _extract_target_metadata_from_sidecar(
    csv_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Read target-race identity/timing from a CSV provenance sidecar."""

    sidecar = Path(f"{csv_path}.metadata.json")
    if not sidecar.exists():
        return {}
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, Mapping):
        return {}

    race_info = payload.get("race_info")
    if not isinstance(race_info, Mapping):
        race_info = {}

    def first(*keys: str) -> Any:
        for source in (race_info, payload):
            for key in keys:
                value = source.get(key)
                if value not in (None, ""):
                    return value
        return None

    race_day = _parse_date(first("race_date", "date", "meeting_date"))
    venue = _clean_venue(first("venue", "venue_code", "track"))
    race_number = _clean_race_number(first("race_number", "race_no", "number"))
    jump = _parse_time(
        first("jump_time", "race_time", "start_time", "scheduled_time")
    )

    data: dict[str, Any] = {}
    if race_day:
        data["race_date"] = race_day.isoformat()
    if venue:
        data["venue"] = venue
    if race_number is not None:
        data["race_number"] = race_number
    if jump:
        data["jump_time"] = jump.strftime("%H:%M")
    if data:
        data["metadata_source"] = "csv_sidecar"
    return data


RESULT_STATUS_KEYS = {"result_status", "results_status", "official_result_status"}
RESULT_COMPLETE_VALUES = {"complete", "completed", "final", "resulted", "official", "closed"}
WINNER_KEYS = {"winner_name", "winner", "actual_winner", "official_winner"}
OFFICIAL_POSITION_KEYS = {
    "official_finish_position",
    "official_position",
    "target_finish_position",
    "result_position",
}


def _non_empty_result_value(value: Any) -> bool:
    if value is None:
        return False
    raw = str(value).strip()
    return bool(raw) and raw.lower() not in {"nan", "none", "null", "unknown", "n/a", "na", "0"}


def _detect_explicit_result_in_rows(rows: list[Mapping[str, Any]]) -> Optional[str]:
    for row in rows:
        status = _first_value(row, RESULT_STATUS_KEYS)
        if status and str(status).strip().lower() in RESULT_COMPLETE_VALUES:
            return "csv_results_status"

        winner = _first_value(row, WINNER_KEYS)
        if _non_empty_result_value(winner):
            return "csv_winner_field"

        position = _first_value(row, OFFICIAL_POSITION_KEYS)
        if _non_empty_result_value(position):
            return "csv_official_position_field"

    return None


def _db_table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        cur = conn.execute(f"PRAGMA table_info({table})")
        return {str(row[1]) for row in cur.fetchall()}
    except Exception:
        return set()


def _detect_result_in_db(
    db_path: Optional[str],
    race_day: Optional[date],
    venue: Optional[str],
    race_number: Optional[int],
) -> Optional[str]:
    if not db_path or not race_day or race_number is None:
        return None
    if not os.path.exists(db_path):
        return None
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            columns = _db_table_columns(conn, "race_metadata")
            if not columns:
                return None

            select_cols = ["race_id"]
            for col in (
                "winner_name",
                "winner_odds",
                "winner_margin",
                "results_status",
                "result_status",
                "scraped_raw_result",
            ):
                if col in columns:
                    select_cols.append(col)

            race_number_col = "race_number" if "race_number" in columns else "race_no" if "race_no" in columns else None
            if not race_number_col or "race_date" not in columns:
                return None

            query = (
                f"SELECT {', '.join(select_cols)} FROM race_metadata "
                f"WHERE race_date = ? AND CAST({race_number_col} AS INTEGER) = ?"
            )
            params: list[Any] = [race_day.isoformat(), int(race_number)]
            rows = conn.execute(query, params).fetchall()
            if not rows:
                return None

            if venue and "venue" in columns:
                venue_upper = venue.upper().replace(" ", "").replace("_", "")

                def score(row: tuple[Any, ...]) -> int:
                    try:
                        race_id = str(row[0] or "").upper().replace(" ", "").replace("_", "")
                        return 1 if venue_upper and venue_upper in race_id else 0
                    except Exception:
                        return 0

                rows = sorted(rows, key=score, reverse=True)

            for row in rows:
                values = dict(zip(select_cols, row))
                for status_key in ("results_status", "result_status"):
                    status = values.get(status_key)
                    if status and str(status).strip().lower() in RESULT_COMPLETE_VALUES:
                        return f"db_{status_key}"
                if _non_empty_result_value(values.get("winner_name")):
                    return "db_winner_name"
                if _non_empty_result_value(values.get("scraped_raw_result")):
                    return "db_scraped_raw_result"
                if _non_empty_result_value(values.get("winner_margin")):
                    return "db_winner_margin"
    except Exception:
        return None
    return None


def _classify_from_metadata(
    *,
    race_day: Optional[date],
    venue: Optional[str],
    race_number: Optional[int],
    jump: Optional[time],
    has_result: bool,
    result_evidence: Optional[str],
    source_path: Optional[str],
    now: Optional[datetime],
    source_context: str,
) -> RaceLifecycle:
    now_dt = now or melbourne_now()
    if now_dt.tzinfo is None:
        tz = _melbourne_tz()
        if tz is not None:
            now_dt = now_dt.replace(tzinfo=tz)

    jump_dt = _combine_melbourne(race_day, jump) if race_day else None
    if jump_dt is not None and now_dt.tzinfo is not None and jump_dt.tzinfo is None:
        jump_dt = jump_dt.replace(tzinfo=now_dt.tzinfo)

    if has_result:
        return RaceLifecycle(
            status=RESULTED,
            status_reason=f"official_result_present:{result_evidence or 'unknown'}",
            race_date=race_day.isoformat() if race_day else None,
            venue=venue,
            race_number=race_number,
            jump_time=jump.strftime("%H:%M") if jump else None,
            jump_datetime=jump_dt.isoformat() if jump_dt else None,
            has_official_result=True,
            result_evidence=result_evidence,
            source_path=source_path,
        )

    if race_day is None:
        return RaceLifecycle(
            status=STALE_FORM_GUIDE,
            status_reason="missing_target_race_date",
            venue=venue,
            race_number=race_number,
            jump_time=jump.strftime("%H:%M") if jump else None,
            source_path=source_path,
        )

    today = now_dt.date()
    if jump_dt is not None:
        if jump_dt > now_dt:
            status = UPCOMING_NOT_JUMPED
            reason = "jump_time_after_now_no_result"
        else:
            status = JUMPED_PENDING_RESULTS
            reason = "jump_time_passed_no_official_result"
    elif race_day > today:
        status = UPCOMING_NOT_JUMPED
        reason = "future_race_date_no_result"
    elif race_day == today:
        if source_context == "live_record":
            status = UPCOMING_NOT_JUMPED
            reason = "live_source_today_no_result_no_jump_time"
        else:
            status = JUMPED_PENDING_RESULTS
            reason = "today_without_jump_time_not_live_safe"
    else:
        status = STALE_FORM_GUIDE
        reason = "past_race_date_no_official_result"

    return RaceLifecycle(
        status=status,
        status_reason=reason,
        race_date=race_day.isoformat(),
        venue=venue,
        race_number=race_number,
        jump_time=jump.strftime("%H:%M") if jump else None,
        jump_datetime=jump_dt.isoformat() if jump_dt else None,
        has_official_result=False,
        result_evidence=None,
        source_path=source_path,
    )


def classify_race_file(
    file_path: str | os.PathLike[str],
    *,
    now: Optional[datetime] = None,
    db_path: Optional[str] = None,
    source_context: str = "csv_file",
) -> RaceLifecycle:
    source_path = str(Path(file_path))
    filename_meta = extract_target_metadata_from_filename(source_path)
    headers, rows = _read_sample_rows(source_path)
    row_meta = _extract_target_metadata_from_rows(rows)
    sidecar_meta = _extract_target_metadata_from_sidecar(source_path)

    # Prefer filename target identity over embedded CSV DATE/TRACK history.
    meta = {**row_meta, **sidecar_meta, **filename_meta}
    race_day = _parse_date(meta.get("race_date"))
    venue = _clean_venue(meta.get("venue"))
    race_number = _clean_race_number(meta.get("race_number"))
    jump = _parse_time(meta.get("jump_time"))

    result_evidence = _detect_explicit_result_in_rows(rows) or _detect_result_in_db(
        db_path, race_day, venue, race_number
    )

    return _classify_from_metadata(
        race_day=race_day,
        venue=venue,
        race_number=race_number,
        jump=jump,
        has_result=bool(result_evidence),
        result_evidence=result_evidence,
        source_path=source_path,
        now=now,
        source_context=source_context,
    )


def classify_race_record(
    record: Mapping[str, Any],
    *,
    now: Optional[datetime] = None,
    source_context: str = "live_record",
) -> RaceLifecycle:
    race_day = _parse_date(
        record.get("race_date")
        or record.get("date")
        or record.get("meeting_date")
        or record.get("Race Date")
    )
    venue = _clean_venue(record.get("venue") or record.get("venue_name") or record.get("track"))
    race_number = _clean_race_number(record.get("race_number") or record.get("race_no") or record.get("number"))
    jump = _parse_time(
        record.get("race_time")
        or record.get("jump_time")
        or record.get("start_time")
        or record.get("scheduled_time")
    )

    result_evidence = None
    for key in RESULT_STATUS_KEYS:
        value = record.get(key)
        if value and str(value).strip().lower() in RESULT_COMPLETE_VALUES:
            result_evidence = f"record_{key}"
            break
    if result_evidence is None:
        for key in WINNER_KEYS:
            if _non_empty_result_value(record.get(key)):
                result_evidence = f"record_{key}"
                break

    return _classify_from_metadata(
        race_day=race_day,
        venue=venue,
        race_number=race_number,
        jump=jump,
        has_result=bool(result_evidence),
        result_evidence=result_evidence,
        source_path=None,
        now=now,
        source_context=source_context,
    )


def lifecycle_response_fields(lifecycle: RaceLifecycle) -> dict[str, Any]:
    data = lifecycle.to_dict()
    return {
        "lifecycle_status": data["status"],
        "lifecycle_status_reason": data["status_reason"],
        "is_live_prediction_target": data["is_live_target"],
        "lifecycle": data,
    }


def summarize_lifecycles(lifecycles: Iterable[RaceLifecycle]) -> dict[str, int]:
    counts = {
        UPCOMING_NOT_JUMPED: 0,
        JUMPED_PENDING_RESULTS: 0,
        RESULTED: 0,
        STALE_FORM_GUIDE: 0,
    }
    for lifecycle in lifecycles:
        counts[lifecycle.status] = counts.get(lifecycle.status, 0) + 1
    return counts
