#!/usr/bin/env python3
"""Build a report-only Betfair AU/NZ greyhound market comparison surface.

The script consumes frozen official monthly CSV files and the corrected canonical
Sportsbet WIN matrix.  It never writes a database, fits a model, or treats BSP
as a pre-jump feature.  Race identity requires date, an explicit venue alias,
race number, exact scheduled time, and a complete TAB/box set; names are only a
post-identity corroboration check.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "betfair_historical_surface_report_v1"
BETFAIR_ROW_SCHEMA_VERSION = "betfair_anz_greyhound_win_row_v1"
BETFAIR_SIDECAR_SCHEMA_VERSION = "betfair_anz_greyhound_win_source_sidecar_v1"
JOIN_SCHEMA_VERSION = "sportsbet_betfair_win_join_v1"
RACE_AUDIT_SCHEMA_VERSION = "sportsbet_betfair_race_join_audit_v1"
EXPECTED_SPORTSBET_SHA256 = (
    "eb1783d4cc07e6980463a097c97fdac9f5370b08f493ca15addf768aa0b014b6"
)

EXPECTED_COLUMNS = (
    "LOCAL_MEETING_DATE",
    "SCHEDULED_RACE_TIME",
    "ACTUAL_OFF_TIME",
    "TRACK",
    "STATE_CODE",
    "RACE_NO",
    "WIN_MARKET_ID",
    "WIN_MARKET_NAME",
    "PLACE_MARKET_ID",
    "RACING_TYPE",
    "DISTANCE",
    "RACE_TYPE",
    "SELECTION_ID",
    "TAB_NUMBER",
    "SELECTION_NAME",
    "WIN_RESULT",
    "WIN_BSP",
    "PLACE_RESULT",
    "PLACE_BSP",
    "WIN_BSP_VOLUME",
    "WIN_PREPLAY_MAX_PRICE_TAKEN",
    "WIN_PREPLAY_MIN_PRICE_TAKEN",
    "WIN_PREPLAY_LAST_PRICE_TAKEN",
    "WIN_PREPLAY_WEIGHTED_AVERAGE_PRICE_TAKEN",
    "WIN_PREPLAY_VOLUME",
    "WIN_INPLAY_MAX_PRICE_TAKEN",
    "WIN_INPLAY_MIN_PRICE_TAKEN",
    "WIN_LAST_PRICE_TAKEN",
    "WIN_INPLAY_WEIGHTED_AVERAGE_PRICE_TAKEN",
    "WIN_INPLAY_VOLUME",
    "PLACE_BSP_VOLUME",
    "PLACE_MAX_PRICE_TAKEN",
    "PLACE_MIN_PRICE_TAKEN",
    "PLACE_LAST_PRICE_TAKEN",
    "PLACE_WEIGHTED_AVERAGE_PRICE_TAKEN",
    "PLACE_PREPLAY_VOLUME",
    "BEST_AVAIL_BACK_AT_SCHEDULED_OFF",
    "BEST_AVAIL_LAY_AT_SCHEDULED_OFF",
    "BACK_MARKET_PERCENTAGE_AT_SCHEDULED_OFF",
    "LAY_MARKET_PERCENTAGE_AT_SCHEDULED_OFF",
)

# Explicit source-to-corpus aliases.  Multi-track families reflect existing
# canonical corpus collapsing; exact time and complete box identity must still
# leave one market or the join is ambiguous.
SPORTSBET_TO_BETFAIR_TRACKS: dict[str, frozenset[str]] = {
    "AP K": frozenset({"Angle Park"}),
    "BAL": frozenset({"Ballarat"}),
    "BEN": frozenset({"Bendigo"}),
    "BH": frozenset({"Broken Hill"}),
    "BULLI": frozenset({"Bulli"}),
    "CANN": frozenset({"Cannington"}),
    "CAPA": frozenset({"Capalaba"}),
    "CASO": frozenset({"Casino"}),
    "DARW": frozenset({"Darwin"}),
    "DUBBO": frozenset({"Dubbo"}),
    "GAWL": frozenset({"Gawler"}),
    "GEE": frozenset({"Geelong"}),
    "GOUL": frozenset({"Goulburn"}),
    "GRAF": frozenset({"Grafton"}),
    "GRDN": frozenset({"The Gardens"}),
    "GUNN": frozenset({"Gunnedah"}),
    "HEA": frozenset({"Healesville"}),
    "HOBT": frozenset({"Hobart"}),
    "HOR": frozenset({"Horsham"}),
    "MAND": frozenset({"Mandurah"}),
    "MEA": frozenset({"The Meadows"}),
    "MT_G": frozenset({"Mount Gambier"}),
    "MURR": frozenset({"Murray Bridge", "Murray Bridge Straight"}),
    "NOR": frozenset({"Northam"}),
    "NOWRA": frozenset({"Nowra"}),
    "QOT": frozenset({"Q Straight", "Q1 Lakeside", "Q2 Parklands"}),
    "RICH": frozenset({"Richmond", "Richmond Straight"}),
    "ROCK": frozenset({"Rockhampton"}),
    "SAL": frozenset({"Sale"}),
    "SAN": frozenset({"Sandown Park"}),
    "SHEP": frozenset({"Shepparton"}),
    "TAREE": frozenset({"Taree"}),
    "TEM": frozenset({"Temora"}),
    "TRA": frozenset({"Traralgon"}),
    "TWN": frozenset({"Townsville"}),
    "WAG": frozenset({"Wagga"}),
    "WAR": frozenset({"Warrnambool"}),
    "WPK": frozenset({"Wentworth Park"}),
    "WRGL": frozenset({"Warragul"}),
}

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME_RE = re.compile(r"^(\d{2}:\d{2}:\d{2})(?:\.\d{3})?$")
ID_RE = re.compile(r"^\d+$")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid_jsonl:{path}:{line_number}") from exc
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    with path.open("wb") as handle:
        for value in values:
            handle.write(canonical_json_bytes(value) + b"\n")


def parse_required_int(value: Any, field: str) -> int:
    text = str(value or "").strip()
    if not ID_RE.fullmatch(text):
        raise ValueError(f"invalid_integer:{field}:{text}")
    return int(text)


def parse_optional_price(value: Any, field: str) -> tuple[float | None, str]:
    text = str(value or "").strip()
    if not text:
        return None, "MISSING_BLANK"
    try:
        parsed = float(text)
    except ValueError as exc:
        raise ValueError(f"invalid_price:{field}:{text}") from exc
    if not math.isfinite(parsed):
        return None, "NONFINITE_LITERAL"
    if parsed <= 1.0:
        raise ValueError(f"invalid_price:{field}:{text}")
    return parsed, "PRESENT"


def parse_time(value: Any, field: str, *, optional: bool = False) -> tuple[str, str | None]:
    raw = str(value or "").strip()
    if not raw and optional:
        return raw, None
    match = TIME_RE.fullmatch(raw)
    if not match:
        raise ValueError(f"invalid_time:{field}:{raw}")
    return raw, match.group(1)


def normalized_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _win_projection(row: Mapping[str, Any]) -> dict[str, str]:
    """Fields that define one official WIN runner row, excluding PLACE data."""

    return {
        field: str(row.get(field) or "")
        for field in EXPECTED_COLUMNS
        if not field.startswith("PLACE_")
    }


def parse_betfair_sources(
    artifact_root: Path, manifest: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    parsed_rows: list[dict[str, Any]] = []
    source_checks: list[dict[str, Any]] = []
    projection_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    raw_row_count = 0

    for source in manifest.get("sources", []):
        path = artifact_root / str(source["raw_path"])
        receipt_path = artifact_root / str(source["receipt_path"])
        actual_hash = sha256_file(path)
        actual_receipt_hash = sha256_file(receipt_path)
        actual_size = path.stat().st_size
        if actual_hash != source["sha256"]:
            raise ValueError(f"raw_sha256_mismatch:{path}:{actual_hash}")
        if actual_receipt_hash != source["receipt_sha256"]:
            raise ValueError(f"receipt_sha256_mismatch:{receipt_path}")
        if actual_size != int(source["byte_size"]):
            raise ValueError(f"raw_byte_size_mismatch:{path}:{actual_size}")

        source_checks.append(
            {
                "byte_size": actual_size,
                "filename": source["filename"],
                "raw_path": source["raw_path"],
                "receipt_path": source["receipt_path"],
                "receipt_sha256": actual_receipt_hash,
                "retrieved_at_utc": source["retrieved_at_utc"],
                "sha256": actual_hash,
                "source_url": source["source_url"],
            }
        )

        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
                raise ValueError(f"unexpected_csv_schema:{path}:{reader.fieldnames}")
            for source_row_number, raw in enumerate(reader, 2):
                raw_row_count += 1
                if raw["RACING_TYPE"] != "Greyhounds":
                    raise ValueError(
                        f"unexpected_racing_type:{path}:{source_row_number}:"
                        f"{raw['RACING_TYPE']}"
                    )
                if raw["WIN_RESULT"] not in {"WINNER", "LOSER"}:
                    raise ValueError(
                        f"unexpected_win_result:{path}:{source_row_number}:"
                        f"{raw['WIN_RESULT']}"
                    )
                date = raw["LOCAL_MEETING_DATE"].strip()
                if not DATE_RE.fullmatch(date):
                    raise ValueError(
                        f"invalid_date:{path}:{source_row_number}:{date}"
                    )
                scheduled_raw, scheduled_clock = parse_time(
                    raw["SCHEDULED_RACE_TIME"], "SCHEDULED_RACE_TIME"
                )
                actual_raw, actual_clock = parse_time(
                    raw["ACTUAL_OFF_TIME"], "ACTUAL_OFF_TIME", optional=True
                )
                market_id = raw["WIN_MARKET_ID"].strip()
                selection_id = raw["SELECTION_ID"].strip()
                if not ID_RE.fullmatch(market_id) or not ID_RE.fullmatch(selection_id):
                    raise ValueError(
                        f"invalid_native_identity:{path}:{source_row_number}"
                    )
                projection = _win_projection(raw)
                projection_hash = sha256_bytes(canonical_json_bytes(projection))
                win_bsp, win_bsp_status = parse_optional_price(
                    raw["WIN_BSP"], "WIN_BSP"
                )
                scheduled_off_back, scheduled_off_back_status = parse_optional_price(
                    raw["BEST_AVAIL_BACK_AT_SCHEDULED_OFF"],
                    "BEST_AVAIL_BACK_AT_SCHEDULED_OFF",
                )
                row = {
                    "schema_version": BETFAIR_ROW_SCHEMA_VERSION,
                    "local_meeting_date": date,
                    "scheduled_race_time_raw": scheduled_raw,
                    "scheduled_race_clock": scheduled_clock,
                    "actual_off_time_raw": actual_raw,
                    "actual_off_clock": actual_clock,
                    "track": raw["TRACK"].strip(),
                    "state_code": raw["STATE_CODE"].strip(),
                    "race_number": parse_required_int(raw["RACE_NO"], "RACE_NO"),
                    "win_market_id": market_id,
                    "win_market_name": raw["WIN_MARKET_NAME"].strip(),
                    "place_market_id_raw": raw["PLACE_MARKET_ID"].strip(),
                    "racing_type": raw["RACING_TYPE"],
                    "distance_raw": raw["DISTANCE"].strip(),
                    "race_type_raw": raw["RACE_TYPE"].strip(),
                    "selection_id": selection_id,
                    "tab_number": parse_required_int(raw["TAB_NUMBER"], "TAB_NUMBER"),
                    "runner_name": raw["SELECTION_NAME"].strip(),
                    "win_result": raw["WIN_RESULT"],
                    "win_bsp_raw": raw["WIN_BSP"].strip(),
                    "win_bsp": win_bsp,
                    "win_bsp_status": win_bsp_status,
                    "scheduled_off_back_price_raw": raw[
                        "BEST_AVAIL_BACK_AT_SCHEDULED_OFF"
                    ].strip(),
                    "scheduled_off_back_price": scheduled_off_back,
                    "scheduled_off_back_price_status": scheduled_off_back_status,
                    "source_file": source["filename"],
                    "source_file_sha256": source["sha256"],
                    "source_row_number": source_row_number,
                    "win_projection_sha256": projection_hash,
                }
                projection_groups[(market_id, selection_id)].append(row)

    duplicate_runner_groups = 0
    duplicate_extra_rows = 0
    conflicting_win_projection_groups = 0
    sidecar_rows: list[dict[str, Any]] = []
    for native_key in sorted(projection_groups, key=lambda item: (int(item[0]), int(item[1]))):
        group = projection_groups[native_key]
        hashes = {row["win_projection_sha256"] for row in group}
        if len(group) > 1:
            duplicate_runner_groups += 1
            duplicate_extra_rows += len(group) - 1
        if len(hashes) != 1:
            conflicting_win_projection_groups += 1
            # Preserve every conflicting row; downstream race validation rejects
            # the duplicate native/box identity.
            selected = group
        else:
            selected = [group[0]]

        for row in selected:
            source_rows = sorted(
                {
                    (item["source_file"], item["source_row_number"])
                    for item in group
                    if item["win_projection_sha256"] == row["win_projection_sha256"]
                }
            )
            row = dict(row)
            row["duplicate_win_projection_count"] = len(source_rows)
            row["source_rows"] = [
                {"source_file": filename, "source_row_number": line}
                for filename, line in source_rows
            ]
            parsed_rows.append(row)
            sidecar_rows.append(
                {
                    "schema_version": BETFAIR_SIDECAR_SCHEMA_VERSION,
                    "win_market_id": row["win_market_id"],
                    "selection_id": row["selection_id"],
                    "tab_number": row["tab_number"],
                    "win_projection_sha256": row["win_projection_sha256"],
                    "source_file_sha256": row["source_file_sha256"],
                    "source_rows": row["source_rows"],
                }
            )

    parsed_rows.sort(
        key=lambda row: (
            row["local_meeting_date"],
            row["scheduled_race_clock"],
            row["track"],
            row["race_number"],
            int(row["win_market_id"]),
            row["tab_number"],
            int(row["selection_id"]),
        )
    )
    sidecar_rows.sort(
        key=lambda row: (int(row["win_market_id"]), row["tab_number"], int(row["selection_id"]))
    )
    counts = {
        "raw_csv_rows": raw_row_count,
        "canonical_win_projection_rows": len(parsed_rows),
        "duplicate_native_runner_groups": duplicate_runner_groups,
        "duplicate_extra_rows_collapsed": duplicate_extra_rows,
        "conflicting_win_projection_groups": conflicting_win_projection_groups,
        "missing_win_bsp_rows": sum(row["win_bsp"] is None for row in parsed_rows),
        "blank_win_bsp_rows": sum(
            row["win_bsp_status"] == "MISSING_BLANK" for row in parsed_rows
        ),
        "nonfinite_win_bsp_rows": sum(
            row["win_bsp_status"] == "NONFINITE_LITERAL" for row in parsed_rows
        ),
        "missing_scheduled_off_back_rows": sum(
            row["scheduled_off_back_price"] is None for row in parsed_rows
        ),
        "blank_scheduled_off_back_rows": sum(
            row["scheduled_off_back_price_status"] == "MISSING_BLANK"
            for row in parsed_rows
        ),
        "nonfinite_scheduled_off_back_rows": sum(
            row["scheduled_off_back_price_status"] == "NONFINITE_LITERAL"
            for row in parsed_rows
        ),
        "missing_actual_off_rows": sum(row["actual_off_clock"] is None for row in parsed_rows),
        "reserve_tab_9_or_10_rows": sum(row["tab_number"] in {9, 10} for row in parsed_rows),
    }
    return parsed_rows, sidecar_rows, {"source_checks": source_checks, "counts": counts}


def load_sportsbet_matrix(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actual_hash = sha256_file(path)
    if actual_hash != EXPECTED_SPORTSBET_SHA256:
        raise ValueError(f"sportsbet_sha256_mismatch:{actual_hash}")
    input_rows = read_jsonl(path)
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(input_rows):
        required = {
            "race_id",
            "race_date",
            "race_number",
            "venue",
            "jump_at",
            "box_number",
            "dog_name",
            "canonical_sportsbet_win_odds",
            "market_implied_probability",
            "label_is_winner",
        }
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"sportsbet_required_fields_missing:{index}:{missing}")
        row = dict(row)
        row["_matrix_row_index"] = index
        rows.append(row)
        grouped[str(row["race_id"])].append(row)

    for race_id, race_rows in grouped.items():
        boxes = [int(row["box_number"]) for row in race_rows]
        if len(boxes) != len(set(boxes)):
            raise ValueError(f"sportsbet_duplicate_box:{race_id}")
        metadata = {
            (
                str(row["race_date"]),
                int(row["race_number"]),
                str(row["venue"]),
                str(row["jump_at"]),
            )
            for row in race_rows
        }
        if len(metadata) != 1:
            raise ValueError(f"sportsbet_race_metadata_conflict:{race_id}")
        winner_count = sum(int(row["label_is_winner"]) == 1 for row in race_rows)
        if winner_count != 1:
            raise ValueError(f"sportsbet_winner_count_invalid:{race_id}:{winner_count}")
        probabilities: list[float] = []
        for row in race_rows:
            try:
                odds = float(row["canonical_sportsbet_win_odds"])
                probability = float(row["market_implied_probability"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"sportsbet_market_value_invalid:{race_id}") from exc
            if not math.isfinite(odds) or odds <= 1.0:
                raise ValueError(f"sportsbet_odds_invalid:{race_id}:{odds}")
            if not math.isfinite(probability) or probability <= 0.0:
                raise ValueError(
                    f"sportsbet_probability_invalid:{race_id}:{probability}"
                )
            probabilities.append(probability)
        probability_sum = sum(probabilities)
        if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"sportsbet_probability_sum_invalid:{race_id}:{probability_sum}"
            )

    dates = sorted({str(row["race_date"]) for row in rows})
    return rows, {
        "sha256": actual_hash,
        "runner_rows": len(rows),
        "races": len(grouped),
        "race_date_min": dates[0],
        "race_date_max": dates[-1],
        "distinct_race_dates": dates,
    }


def group_betfair_races(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, int, str], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["local_meeting_date"]),
            str(row["track"]),
            int(row["race_number"]),
            str(row["win_market_id"]),
        )
        grouped[key].append(row)
    return grouped


def _sportsbet_local_clock(jump_at: Any, race_date: str) -> str:
    parsed = datetime.fromisoformat(str(jump_at).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"sportsbet_jump_timezone_missing:{jump_at}")
    if parsed.date().isoformat() != race_date:
        raise ValueError(f"sportsbet_jump_date_conflict:{jump_at}:{race_date}")
    return parsed.strftime("%H:%M:%S")


def _candidate_summary(
    key: tuple[str, str, int, str], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "track": key[1],
        "win_market_id": key[3],
        "scheduled_race_clock": sorted(
            {str(row["scheduled_race_clock"]) for row in rows}
        ),
        "actual_off_clock": sorted(
            {row["actual_off_clock"] for row in rows if row["actual_off_clock"]}
        ),
        "tab_numbers": sorted(int(row["tab_number"]) for row in rows),
    }


def join_surfaces(
    sportsbet_rows: Sequence[Mapping[str, Any]],
    betfair_rows: Sequence[Mapping[str, Any]],
    available_months: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sportsbet_races: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in sportsbet_rows:
        sportsbet_races[str(row["race_id"])].append(row)
    betfair_races = group_betfair_races(betfair_rows)

    joined: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for race_id in sorted(sportsbet_races):
        sb_rows = sorted(sportsbet_races[race_id], key=lambda row: int(row["box_number"]))
        first = sb_rows[0]
        race_date = str(first["race_date"])
        venue = str(first["venue"])
        race_number = int(first["race_number"])
        sportsbet_clock = _sportsbet_local_clock(first["jump_at"], race_date)
        audit: dict[str, Any] = {
            "schema_version": RACE_AUDIT_SCHEMA_VERSION,
            "race_id": race_id,
            "race_date": race_date,
            "sportsbet_venue": venue,
            "race_number": race_number,
            "sportsbet_scheduled_clock": sportsbet_clock,
            "sportsbet_boxes": sorted(int(row["box_number"]) for row in sb_rows),
            "sportsbet_runner_count": len(sb_rows),
            "candidate_markets": [],
            "status": None,
            "exclusion_reason": None,
        }
        month = race_date[:7]
        if month not in available_months:
            audit["status"] = "EXCLUDED"
            audit["exclusion_reason"] = "BETFAIR_MONTHLY_FILE_UNAVAILABLE"
            audits.append(audit)
            continue
        tracks = SPORTSBET_TO_BETFAIR_TRACKS.get(venue)
        if tracks is None:
            audit["status"] = "EXCLUDED"
            audit["exclusion_reason"] = "SPORTSBET_VENUE_ALIAS_UNMAPPED"
            audits.append(audit)
            continue

        candidates = [
            (key, rows)
            for key, rows in betfair_races.items()
            if key[0] == race_date and key[1] in tracks and key[2] == race_number
        ]
        audit["candidate_markets"] = [
            _candidate_summary(key, rows) for key, rows in candidates
        ]
        if not candidates:
            audit["status"] = "EXCLUDED"
            audit["exclusion_reason"] = "RACE_METADATA_NO_CANDIDATE"
            audits.append(audit)
            continue
        scheduled_candidates = [
            (key, rows)
            for key, rows in candidates
            if {str(row["scheduled_race_clock"]) for row in rows}
            == {sportsbet_clock}
        ]
        if not scheduled_candidates:
            audit["status"] = "EXCLUDED"
            audit["exclusion_reason"] = "SCHEDULED_TIME_MISMATCH"
            audits.append(audit)
            continue
        if len(scheduled_candidates) != 1:
            audit["status"] = "EXCLUDED"
            audit["exclusion_reason"] = "AMBIGUOUS_RACE_CANDIDATE"
            audits.append(audit)
            continue

        key, bf_rows = scheduled_candidates[0]
        bf_boxes = [int(row["tab_number"]) for row in bf_rows]
        bf_selection_ids = [str(row["selection_id"]) for row in bf_rows]
        audit["selected_track"] = key[1]
        audit["selected_win_market_id"] = key[3]
        audit["betfair_tab_numbers"] = sorted(bf_boxes)
        audit["betfair_runner_count"] = len(bf_rows)
        if len(bf_boxes) != len(set(bf_boxes)) or len(bf_selection_ids) != len(
            set(bf_selection_ids)
        ):
            audit["status"] = "EXCLUDED"
            audit["exclusion_reason"] = "BETFAIR_RUNNER_IDENTITY_DUPLICATE"
            audits.append(audit)
            continue
        if set(bf_boxes) != {int(row["box_number"]) for row in sb_rows}:
            audit["status"] = "EXCLUDED"
            audit["exclusion_reason"] = "RUNNER_SET_MISMATCH_RESERVE_OR_SCRATCH"
            audit["betfair_reserve_tabs"] = sorted(
                box for box in bf_boxes if box in {9, 10}
            )
            audits.append(audit)
            continue

        actual_clocks = {row["actual_off_clock"] for row in bf_rows}
        if len(actual_clocks) != 1:
            audit["status"] = "EXCLUDED"
            audit["exclusion_reason"] = "PROVIDER_ACTUAL_OFF_TIME_CONFLICT"
            audit["provider_actual_off_clocks"] = sorted(
                "" if clock is None else str(clock) for clock in actual_clocks
            )
            audits.append(audit)
            continue

        bf_by_box = {int(row["tab_number"]): row for row in bf_rows}
        name_conflicts: list[int] = []
        result_conflicts: list[int] = []
        for sb_row in sb_rows:
            box = int(sb_row["box_number"])
            bf_row = bf_by_box[box]
            if normalized_name(sb_row["dog_name"]) != normalized_name(
                bf_row["runner_name"]
            ):
                name_conflicts.append(box)
            if (int(sb_row["label_is_winner"]) == 1) != (
                bf_row["win_result"] == "WINNER"
            ):
                result_conflicts.append(box)
        if name_conflicts:
            audit["status"] = "EXCLUDED"
            audit["exclusion_reason"] = "RUNNER_NAME_CORROBORATION_CONFLICT"
            audit["conflicting_boxes"] = name_conflicts
            audits.append(audit)
            continue
        if result_conflicts:
            audit["status"] = "EXCLUDED"
            audit["exclusion_reason"] = "RESULT_CORROBORATION_CONFLICT"
            audit["conflicting_boxes"] = result_conflicts
            audits.append(audit)
            continue

        actual_clock = next(iter(actual_clocks))
        scheduled_precedes_provider_actual = bool(
            actual_clock is not None and sportsbet_clock < str(actual_clock)
        )
        audit["status"] = "MATCHED"
        audit["exclusion_reason"] = None
        audit["actual_off_clock"] = actual_clock
        audit["scheduled_clock_precedes_provider_actual_off_clock"] = (
            scheduled_precedes_provider_actual
        )
        audits.append(audit)

        for sb_row in sb_rows:
            box = int(sb_row["box_number"])
            bf_row = bf_by_box[box]
            scheduled_price = bf_row["scheduled_off_back_price"]
            bsp = bf_row["win_bsp"]
            joined.append(
                {
                    "schema_version": JOIN_SCHEMA_VERSION,
                    "race_id": race_id,
                    "race_date": race_date,
                    "race_number": race_number,
                    "sportsbet_venue": venue,
                    "betfair_track": bf_row["track"],
                    "scheduled_race_time_raw": bf_row["scheduled_race_time_raw"],
                    "actual_off_time_raw": bf_row["actual_off_time_raw"],
                    "scheduled_clock_precedes_provider_actual_off_clock": (
                        scheduled_precedes_provider_actual
                    ),
                    "win_market_id": bf_row["win_market_id"],
                    "selection_id": bf_row["selection_id"],
                    "box_number": box,
                    "sportsbet_runner_name": sb_row["dog_name"],
                    "betfair_runner_name": bf_row["runner_name"],
                    "win_result": bf_row["win_result"],
                    "sportsbet_win_odds": float(
                        sb_row["canonical_sportsbet_win_odds"]
                    ),
                    "sportsbet_normalized_probability": float(
                        sb_row["market_implied_probability"]
                    ),
                    "betfair_scheduled_off_back_price_raw": bf_row[
                        "scheduled_off_back_price_raw"
                    ],
                    "betfair_scheduled_off_back_price": scheduled_price,
                    "betfair_scheduled_off_back_price_status": bf_row[
                        "scheduled_off_back_price_status"
                    ],
                    "betfair_bsp_raw": bf_row["win_bsp_raw"],
                    "betfair_bsp": bsp,
                    "betfair_bsp_status": bf_row["win_bsp_status"],
                    "betfair_scheduled_off_raw_implied_probability": (
                        None if scheduled_price is None else 1.0 / scheduled_price
                    ),
                    "betfair_bsp_raw_implied_probability": (
                        None if bsp is None else 1.0 / bsp
                    ),
                    "sportsbet_matrix_row_index": sb_row["_matrix_row_index"],
                    "sportsbet_matrix_sha256": EXPECTED_SPORTSBET_SHA256,
                    "betfair_source_file": bf_row["source_file"],
                    "betfair_source_file_sha256": bf_row["source_file_sha256"],
                    "betfair_source_rows": bf_row["source_rows"],
                    "betfair_win_projection_sha256": bf_row[
                        "win_projection_sha256"
                    ],
                }
            )
    joined.sort(key=lambda row: (row["race_date"], row["race_id"], row["box_number"]))
    audits.sort(key=lambda row: (row["race_date"], row["race_id"]))
    return joined, audits


def average_ranks(values: Sequence[float], *, reverse: bool = True) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda i: values[i], reverse=reverse)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and math.isclose(
            values[ordered[end]], values[ordered[index]], rel_tol=0.0, abs_tol=1e-15
        ):
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for position in ordered[index:end]:
            ranks[position] = average_rank
        index = end
    return ranks


def pearson(values_a: Sequence[float], values_b: Sequence[float]) -> float | None:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return None
    mean_a = mean(values_a)
    mean_b = mean(values_b)
    numerator = sum((a - mean_a) * (b - mean_b) for a, b in zip(values_a, values_b))
    denominator = math.sqrt(
        sum((a - mean_a) ** 2 for a in values_a)
        * sum((b - mean_b) ** 2 for b in values_b)
    )
    return None if denominator == 0.0 else numerator / denominator


def summarize_price_surface(
    joined_rows: Sequence[Mapping[str, Any]], price_field: str, label: str
) -> dict[str, Any]:
    races: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in joined_rows:
        races[str(row["race_id"])].append(row)
    complete_races = [
        rows for rows in races.values() if all(row[price_field] is not None for row in rows)
    ]
    missing_runner_rows = sum(row[price_field] is None for row in joined_rows)
    overrounds: list[float] = []
    absolute_differences: list[float] = []
    favourite_agreement = 0
    favourite_tie_races = 0
    rank_correlations: list[float] = []
    normalized_runner_rows = 0

    for rows in complete_races:
        sportsbet_probabilities = [
            float(row["sportsbet_normalized_probability"]) for row in rows
        ]
        raw = [1.0 / float(row[price_field]) for row in rows]
        overround = sum(raw)
        if overround <= 0.0:
            raise ValueError(f"nonpositive_overround:{label}:{rows[0]['race_id']}")
        probabilities = [value / overround for value in raw]
        overrounds.append(overround)
        normalized_runner_rows += len(rows)
        absolute_differences.extend(
            abs(sportsbet - betfair)
            for sportsbet, betfair in zip(sportsbet_probabilities, probabilities)
        )

        sportsbet_max = max(sportsbet_probabilities)
        betfair_max = max(probabilities)
        sportsbet_favourites = {
            index
            for index, value in enumerate(sportsbet_probabilities)
            if math.isclose(value, sportsbet_max, rel_tol=0.0, abs_tol=1e-15)
        }
        betfair_favourites = {
            index
            for index, value in enumerate(probabilities)
            if math.isclose(value, betfair_max, rel_tol=0.0, abs_tol=1e-15)
        }
        if len(sportsbet_favourites) != 1 or len(betfair_favourites) != 1:
            favourite_tie_races += 1
        elif sportsbet_favourites == betfair_favourites:
            favourite_agreement += 1

        correlation = pearson(
            average_ranks(sportsbet_probabilities), average_ranks(probabilities)
        )
        if correlation is not None:
            rank_correlations.append(correlation)

    non_tied = len(complete_races) - favourite_tie_races
    return {
        "surface": label,
        "complete_price_races": len(complete_races),
        "complete_price_runner_rows": normalized_runner_rows,
        "missing_price_races": len(races) - len(complete_races),
        "missing_price_runner_rows": missing_runner_rows,
        "mean_overround": mean(overrounds) if overrounds else None,
        "median_overround": median(overrounds) if overrounds else None,
        "mean_absolute_normalized_probability_difference_vs_sportsbet": (
            mean(absolute_differences) if absolute_differences else None
        ),
        "median_absolute_normalized_probability_difference_vs_sportsbet": (
            median(absolute_differences) if absolute_differences else None
        ),
        "favourite_tie_races_excluded_from_agreement_rate": favourite_tie_races,
        "favourite_agreement_races": favourite_agreement,
        "favourite_agreement_rate_non_tied": (
            favourite_agreement / non_tied if non_tied else None
        ),
        "mean_spearman_rank_correlation": (
            mean(rank_correlations) if rank_correlations else None
        ),
    }


def _surface_verdict(
    *,
    sportsbet_races: int,
    sportsbet_runner_rows: int,
    matched_races: int,
    matched_runner_rows: int,
    missing_scheduled_rows: int,
    reason_counts: Mapping[str, int],
) -> tuple[str, str, list[str]]:
    material_conflict_reasons = {
        "AMBIGUOUS_RACE_CANDIDATE",
        "BETFAIR_RUNNER_IDENTITY_DUPLICATE",
        "PROVIDER_ACTUAL_OFF_TIME_CONFLICT",
        "RESULT_CORROBORATION_CONFLICT",
        "RUNNER_NAME_CORROBORATION_CONFLICT",
    }
    material_conflicts = {
        reason: int(reason_counts.get(reason, 0))
        for reason in sorted(material_conflict_reasons)
        if reason_counts.get(reason, 0)
    }
    coverage = (
        f"The strict join matched {matched_races}/{sportsbet_races} Sportsbet races "
        f"and {matched_runner_rows}/{sportsbet_runner_rows} runner rows."
    )

    if material_conflicts:
        return (
            "NOT_READY",
            "BETFAIR_HISTORICAL_SURFACE_NOT_READY",
            [
                coverage,
                "Material identity or provider-timing conflicts make the surface "
                f"untrustworthy: {json.dumps(material_conflicts, sort_keys=True)}.",
            ],
        )
    if matched_races == 0 or matched_runner_rows == 0:
        return (
            "NOT_READY",
            "BETFAIR_HISTORICAL_SURFACE_NOT_READY",
            [coverage, "No non-empty strict matched surface is available."],
        )

    complete_coverage = (
        matched_races == sportsbet_races
        and matched_runner_rows == sportsbet_runner_rows
        and not reason_counts
    )
    if complete_coverage and missing_scheduled_rows == 0:
        return (
            "READY",
            "BETFAIR_HISTORICAL_SURFACE_READY",
            [
                coverage,
                "Every matched runner has a finite Betfair scheduled-off back price, "
                "with no excluded or ambiguous races.",
            ],
        )

    reasons = [coverage]
    if reason_counts:
        reasons.append(
            "Excluded races by reason: "
            f"{json.dumps(dict(sorted(reason_counts.items())), sort_keys=True)}."
        )
    if missing_scheduled_rows:
        reasons.append(
            f"{missing_scheduled_rows} matched runner rows lack a usable Betfair "
            "scheduled-off back price."
        )
    if not reason_counts and not missing_scheduled_rows:
        reasons.append("The strict matched surface does not cover the full Sportsbet corpus.")
    return "PARTIAL", "BETFAIR_HISTORICAL_SURFACE_PARTIAL", reasons


def build_report(
    *,
    artifact_root: Path,
    manifest: Mapping[str, Any],
    betfair_meta: Mapping[str, Any],
    sportsbet_meta: Mapping[str, Any],
    joined: Sequence[Mapping[str, Any]],
    audits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reason_counts = Counter(
        str(row["exclusion_reason"])
        for row in audits
        if row["exclusion_reason"] is not None
    )
    matched_audits = [row for row in audits if row["status"] == "MATCHED"]
    audit_by_race = {str(row["race_id"]): row for row in audits}
    excluded_runner_counts = Counter()
    for race_id, audit in audit_by_race.items():
        if audit["exclusion_reason"] is not None:
            excluded_runner_counts[str(audit["exclusion_reason"])] += int(
                audit["sportsbet_runner_count"]
            )

    matched_race_ids = {str(row["race_id"]) for row in joined}
    missing_scheduled_rows = sum(
        row["betfair_scheduled_off_back_price"] is None for row in joined
    )
    missing_bsp_rows = sum(row["betfair_bsp"] is None for row in joined)
    joined_scheduled_status = Counter(
        str(row["betfair_scheduled_off_back_price_status"]) for row in joined
    )
    joined_bsp_status = Counter(str(row["betfair_bsp_status"]) for row in joined)
    matched_races_with_reserve_tabs = sum(
        any(int(tab) in {9, 10} for tab in row.get("betfair_tab_numbers", []))
        for row in matched_audits
    )
    mismatch_reserve_tab_races = sum(
        bool(row.get("betfair_reserve_tabs"))
        for row in audits
        if row["exclusion_reason"] == "RUNNER_SET_MISMATCH_RESERVE_OR_SCRATCH"
    )
    scheduled_before_provider_actual_races = sum(
        bool(row.get("scheduled_clock_precedes_provider_actual_off_clock"))
        for row in matched_audits
    )

    scheduled_diagnostics = summarize_price_surface(
        joined, "betfair_scheduled_off_back_price", "BETFAIR_SCHEDULED_OFF_BACK"
    )
    bsp_diagnostics = summarize_price_surface(joined, "betfair_bsp", "BETFAIR_BSP")

    verdict, terminal_state, verdict_reasons = _surface_verdict(
        sportsbet_races=int(sportsbet_meta["races"]),
        sportsbet_runner_rows=int(sportsbet_meta["runner_rows"]),
        matched_races=len(matched_race_ids),
        matched_runner_rows=len(joined),
        missing_scheduled_rows=missing_scheduled_rows,
        reason_counts=reason_counts,
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "analysis_as_of_utc": max(
            [source["retrieved_at_utc"] for source in manifest["sources"]]
            + [source["checked_at_utc"] for source in manifest["unavailable_sources"]]
        ),
        "verdict": verdict,
        "terminal_state": terminal_state,
        "verdict_reasons": verdict_reasons,
        "inputs": {
            "sportsbet": sportsbet_meta,
            "betfair": betfair_meta,
            "unavailable_sources": manifest["unavailable_sources"],
        },
        "identity_contract": {
            "race_key": "local meeting date + explicit venue alias + race number + exact scheduled clock",
            "runner_key": "complete one-to-one Sportsbet box_number == Betfair TAB_NUMBER",
            "runner_key_semantics": "Betfair TAB_NUMBER is documented as TAB/rug identity and is not asserted to be an effective physical box",
            "name_role": "corroboration only after metadata and box identity; never a candidate key",
            "result_role": "post-join corroboration only; never a candidate key or pre-jump feature",
            "time_role": "provider time-only strings are compared as same-date clocks without deriving a timezone-aware instant; ACTUAL_OFF_TIME construction is undocumented",
            "multi_track_aliases": {
                key: sorted(value)
                for key, value in SPORTSBET_TO_BETFAIR_TRACKS.items()
                if len(value) > 1
            },
        },
        "overlap": {
            "sportsbet_races": sportsbet_meta["races"],
            "sportsbet_runner_rows": sportsbet_meta["runner_rows"],
            "matched_races": len(matched_race_ids),
            "matched_runner_rows": len(joined),
            "matched_race_rate": len(matched_race_ids) / sportsbet_meta["races"],
            "matched_runner_rate": len(joined) / sportsbet_meta["runner_rows"],
            "excluded_races_by_reason": dict(sorted(reason_counts.items())),
            "excluded_runner_rows_by_reason": dict(sorted(excluded_runner_counts.items())),
            "ambiguous_races": reason_counts.get("AMBIGUOUS_RACE_CANDIDATE", 0),
            "runner_name_corroboration_conflicts": reason_counts.get(
                "RUNNER_NAME_CORROBORATION_CONFLICT", 0
            ),
            "result_corroboration_conflicts": reason_counts.get(
                "RESULT_CORROBORATION_CONFLICT", 0
            ),
            "matched_races_with_tab_9_or_10": matched_races_with_reserve_tabs,
            "runner_set_mismatch_races_with_tab_9_or_10": mismatch_reserve_tab_races,
            "matched_races_scheduled_clock_before_provider_actual_off_clock": scheduled_before_provider_actual_races,
            "matched_races_not_proven_scheduled_clock_before_provider_actual_off_clock": len(matched_audits)
            - scheduled_before_provider_actual_races,
            "matched_rows_missing_scheduled_off_back": missing_scheduled_rows,
            "matched_scheduled_off_back_status_counts": dict(
                sorted(joined_scheduled_status.items())
            ),
            "matched_rows_missing_bsp": missing_bsp_rows,
            "matched_bsp_status_counts": dict(sorted(joined_bsp_status.items())),
        },
        "descriptive_diagnostics": {
            "sportsbet_probability_semantics": "canonical normalized within-race WIN probability",
            "betfair_probability_semantics": "1/decimal price normalized within complete-price race",
            "scheduled_off": scheduled_diagnostics,
            "bsp": bsp_diagnostics,
            "claim_boundary": "descriptive agreement and coverage only; no candidate was fit and BSP is not a pre-jump feature",
        },
        "findings": {
            "BLOCKING": [
                "Do not treat BSP as a pre-jump feature; it remains a distinct at-start diagnostic.",
                "Do not join the 91 runner-set mismatches or five scheduled-time mismatches by runner name, price order, or neighbouring race.",
                "Do not claim complete overlap while the official August file remains unavailable.",
            ],
            "IMPORTANT": [
                "Only rows whose scheduled clock strictly precedes the provider ACTUAL_OFF_TIME clock are timing-eligible for a later experiment; that provider field is not asserted to be an official jump timestamp.",
                "TAB 9/10 rows are reported as reserve-number evidence; the source has no explicit scratch field, so scratches are not inferred.",
                "Official duplicate WIN projections are collapsed only when every preserved WIN field is identical; all source lines remain in the sidecar.",
                "The published literal WIN_BSP=inf is retained verbatim, classified NONFINITE_LITERAL, and excluded from probability diagnostics because first-party semantics were not found.",
            ],
            "OPTIONAL": [
                "Re-run the same frozen parser after the official August file is published; do not substitute another provider.",
                "Investigate runner-set mismatches only with native provider metadata and immutable receipts under separate authority.",
            ],
        },
        "claims": {
            "supported": [
                "The recorded official June and July files deterministically support a strict matched Betfair/Sportsbet comparison subset.",
                "Scheduled-off back, BSP, and actual-off fields remain separate with explicit missingness.",
                "Descriptive probability, overround, favourite, rank, and coverage diagnostics are reproducible from frozen inputs.",
                "A later paired experiment is technically supportable on a newly frozen, strict eligible population.",
            ],
            "unsupported": [
                "Predictive edge, profitability, ROI, EV, betting value, promotion, or deployment.",
                "Complete corrected-corpus Betfair coverage.",
                "Any identity repair based only on normalized runner name.",
                "Use of BSP, results, or actual-off information as a pre-jump feature.",
            ],
        },
        "next_experiment": {
            "status": "DEFINED_NOT_RUN_CONDITIONAL",
            "hypothesis": "A predeclared pool of corrected Sportsbet WIN probability and Betfair best-available back probability at scheduled off improves paired out-of-sample race log loss over corrected Sportsbet WIN alone.",
            "sportsbet_baseline": "Corrected canonical Sportsbet WIN probabilities reconstructed under the same evidence contract as matrix SHA-256 eb1783d4cc07e6980463a097c97fdac9f5370b08f493ca15addf768aa0b014b6.",
            "eligible_population": "AU greyhound WIN races with one unique Betfair market, exact date/venue/race/scheduled-time identity, identical complete displayed box/TAB runner sets, corroborating names/results, complete Sportsbet and Betfair scheduled-off prices, and scheduled clock strictly before the provider ACTUAL_OFF_TIME clock.",
            "temporal_windows": {
                "development_fit": "2026-06-10 through 2026-06-30 strict eligible rows only",
                "development_validation": "2026-07-01 through 2026-07-18 strict eligible rows only",
                "excluded_existing_gap": "2026-08-01 through 2026-08-02 is not reused for fitting, selection, or testing",
                "future_untouched_test": "2026-08-18 through 2026-09-30, frozen before labels or Betfair outcomes are inspected",
            },
            "candidate_protocol": "Fit only a predeclared one-parameter convex probability pool on the development-fit window; select the weight on development-validation; lock it before the future test. Betfair BSP is diagnostic only and never enters the candidate.",
            "leakage_controls": [
                "No BSP, WIN_RESULT, actual-off value, in-play price, volume, or post-jump field enters a feature.",
                "Require the scheduled clock to precede the provider ACTUAL_OFF_TIME clock and retain both raw strings; do not reinterpret the latter as an official jump timestamp.",
                "Freeze race eligibility before outcomes; no name-only repairs, race substitution, skips added after scoring, or August reuse.",
                "Normalize each market only within its exact accepted runner set.",
            ],
            "primary_metric": "paired multiclass race log loss: candidate minus Sportsbet baseline",
            "secondary_metric": "paired multiclass race Brier score: candidate minus Sportsbet baseline",
            "uncertainty_plan": "Predeclare fixed sample size before collection; report meeting-date-cluster paired bootstrap 95% confidence intervals for mean metric differences, with no interim peeking or optional stopping.",
            "strongest_permitted_claim": "On the declared untouched 2026-08-18 through 2026-09-30 eligible cohort, the locked two-market pool improved probabilistic accuracy versus corrected Sportsbet WIN if and only if the paired primary-metric confidence interval is wholly below zero; no profitability or broader-market claim follows.",
        },
    }
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    overlap = report["overlap"]
    lines = [
        "# Betfair historical market surface audit",
        "",
        f"Terminal state: `{report['terminal_state']}` (`{report['verdict']}`).",
        "",
        "## Inputs",
        "",
        f"- Corrected Sportsbet matrix: `{report['inputs']['sportsbet']['sha256']}` ",
        f"({report['inputs']['sportsbet']['races']} races / {report['inputs']['sportsbet']['runner_rows']} runners).",
    ]
    for source in report["inputs"]["betfair"]["source_checks"]:
        lines.append(
            f"- `{source['filename']}`: {source['byte_size']} bytes, SHA-256 "
            f"`{source['sha256']}`, retrieved `{source['retrieved_at_utc']}` from "
            f"{source['source_url']}."
        )
    lines.extend(
        [
            "",
            "## Overlap",
            "",
            f"- Matched: {overlap['matched_races']} races / {overlap['matched_runner_rows']} runners.",
            f"- Race coverage: {overlap['matched_race_rate']:.6%}; runner coverage: {overlap['matched_runner_rate']:.6%}.",
            f"- Excluded races: `{json.dumps(overlap['excluded_races_by_reason'], sort_keys=True)}`.",
            f"- Excluded runners: `{json.dumps(overlap['excluded_runner_rows_by_reason'], sort_keys=True)}`.",
            f"- Missing scheduled-off prices in matched rows: {overlap['matched_rows_missing_scheduled_off_back']}; missing BSP: {overlap['matched_rows_missing_bsp']}.",
            "",
            "## Verdict",
            "",
        ]
    )
    lines.extend(f"- {reason}" for reason in report["verdict_reasons"])
    lines.extend(
        [
            "",
            "The strict subset supports a later paired experiment after a fresh freeze, but this audit does not fit a candidate or claim predictive edge. BSP remains a distinct at-start diagnostic and is prohibited as a pre-jump feature.",
            "",
        ]
    )
    return "\n".join(lines)


def output_hashes(output_dir: Path, filenames: Sequence[str]) -> dict[str, str]:
    return {filename: sha256_file(output_dir / filename) for filename in filenames}


def git_head(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sportsbet-matrix",
        type=Path,
        default=Path(
            "artifacts/sportsbet_win_market_surface_audit_20260815_report_only/"
            "canonical_win_matrix.jsonl"
        ),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/betfair_historical_surface_20260817_report_only"),
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    artifact_root = (repo_root / args.artifact_root).resolve()
    matrix_path = (repo_root / args.sportsbet_matrix).resolve()
    manifest = read_json(artifact_root / "source_manifest.json")
    if manifest.get("schema_version") != "betfair_historical_source_manifest_v1":
        raise ValueError("source_manifest_schema_mismatch")

    sportsbet_rows, sportsbet_meta = load_sportsbet_matrix(matrix_path)
    betfair_rows, sidecar_rows, betfair_meta = parse_betfair_sources(
        artifact_root, manifest
    )
    available_months = {
        str(source["filename"])[len("ANZ_Greyhounds_") : -len(".csv")].replace(
            "_", "-"
        )
        for source in manifest["sources"]
    }
    joined, audits = join_surfaces(
        sportsbet_rows, betfair_rows, available_months
    )
    report = build_report(
        artifact_root=artifact_root,
        manifest=manifest,
        betfair_meta=betfair_meta,
        sportsbet_meta=sportsbet_meta,
        joined=joined,
        audits=audits,
    )
    report["code_provenance"] = {
        "repo_head": git_head(repo_root),
        "script": str(Path(__file__).resolve().relative_to(repo_root)),
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    report["output_schema"] = {
        "path": "report.schema.json",
        "sha256": sha256_file(artifact_root / "report.schema.json"),
    }

    write_jsonl(artifact_root / "canonical_betfair_win_rows.jsonl", betfair_rows)
    write_jsonl(artifact_root / "canonical_betfair_win_sidecar.jsonl", sidecar_rows)
    write_jsonl(artifact_root / "race_join_audit.jsonl", audits)
    write_jsonl(artifact_root / "sportsbet_betfair_joined_surface.jsonl", joined)
    write_json(artifact_root / "report.json", report)
    (artifact_root / "REPORT.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    hashed = output_hashes(
        artifact_root,
        [
            "REPORT.md",
            "canonical_betfair_win_rows.jsonl",
            "canonical_betfair_win_sidecar.jsonl",
            "race_join_audit.jsonl",
            "report.json",
            "report.schema.json",
            "source_manifest.json",
            "sportsbet_betfair_joined_surface.jsonl",
        ],
    )
    (artifact_root / "SHA256SUMS").write_text(
        "".join(f"{digest}  {filename}\n" for filename, digest in sorted(hashed.items())),
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
