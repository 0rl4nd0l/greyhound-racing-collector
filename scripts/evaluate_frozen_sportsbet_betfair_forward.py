#!/usr/bin/env python3
"""Seal and later score the frozen Sportsbet/Betfair forward consensus test.

The ``seal-population`` command reads label-free corrected Sportsbet WIN and
independently outcome-free Betfair projections. Official result-bearing
Betfair ANZ monthly files are rejected; this module never converts them.

The ``score`` command is intentionally separate. It requires a sealed
population plus an approved result projection and applies the already-frozen
95% Betfair / 5% Sportsbet scorer. Do not run either command on the prospective
window without separate authority.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import unquote, urlparse
from zoneinfo import ZoneInfo

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_sportsbet_betfair_consensus_freeze as frozen


FORWARD_START = date(2026, 8, 20)
FORWARD_END = date(2026, 9, 30)
MELBOURNE = ZoneInfo("Australia/Melbourne")
BETFAIR_WEIGHT = 0.95
SPORTSBET_WEIGHT = 0.05
EXPECTED_SCORER_SHA256 = "929a9d5ebb073d199e30f20b4a724eee1eb2d42699ab0db6333d34d1e22ff5a6"
EXPECTED_RULE_SHA256 = "3a12760aba2d84bbe6530337ea0d66ef0ce8ae79f402c66207a727efb155b739"
EXPECTED_PROTOCOL_SHA256 = "610baf8847afeef179a778b264e533be18da1ec6500dae59bc4599b5d454e0df"
EXPECTED_PREDECESSOR_ELIGIBILITY_SHA256 = (
    "729cfe4e487b80bc1cb888d8f65222d2ac75520d1d5d8a1af8d834922181f3f2"
)
BETFAIR_REQUIRED_COLUMNS = (
    "LOCAL_MEETING_DATE",
    "SCHEDULED_RACE_TIME",
    "TRACK",
    "RACE_NO",
    "WIN_MARKET_ID",
    "SELECTION_ID",
    "TAB_NUMBER",
    "SELECTION_NAME",
    "BEST_AVAIL_BACK_AT_SCHEDULED_OFF",
)
BETFAIR_FORBIDDEN_PREDICTORS = frozenset(
    {
        "ACTUAL_OFF_TIME",
        "WIN_RESULT",
        "WIN_BSP",
        "PLACE_RESULT",
        "PLACE_BSP",
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
    }
)
SPORTSBET_ROW_FIELDS = frozenset(
    {
        "schema_version",
        "race_date",
        "sportsbet_venue",
        "race_number",
        "scheduled_race_time_raw",
        "box_number",
        "runner_name",
        "sportsbet_normalized_probability",
        "sportsbet_source_sha256",
        "sportsbet_source_row_id",
    }
)
SPORTSBET_ROW_SCHEMA = "sportsbet_corrected_win_predictor_projection_v1"
RESULT_ROW_FIELDS = frozenset(
    {
        "schema_version",
        "race_date",
        "sportsbet_venue",
        "race_number",
        "scheduled_race_time_raw",
        "winner_box",
        "approved_result_source_sha256",
        "approved_result_source_row_id",
    }
)
RESULT_ROW_SCHEMA = "approved_greyhound_result_projection_v1"
TIME_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})(?:\.000)?$")
BETFAIR_FILE_RE = re.compile(r"^ANZ_Greyhounds_(\d{4})_(\d{2})\.csv$")
BETFAIR_SOURCE_RECEIPT_SCHEMA = "betfair_forward_source_manifest_receipt_v1"
POPULATION_APPROVAL_RECEIPT_SCHEMA = "sportsbet_betfair_population_approval_receipt_v1"
POPULATION_MANIFEST_SCHEMA = "sportsbet_betfair_forward_population_manifest_v1"
EXPECTED_BETFAIR_FILENAMES = {
    "ANZ_Greyhounds_2026_08.csv",
    "ANZ_Greyhounds_2026_09.csv",
}
EXPECTED_SPORTSBET_PREDICTOR_FILENAME = (
    "sportsbet_corrected_win_predictors_20260820_20260930.jsonl"
)
POPULATION_MANIFEST_FIELDS = {
    "schema_version",
    "terminal_state",
    "window",
    "frozen_hashes",
    "sportsbet_predictor_sha256",
    "sportsbet_completeness_receipt_sha256",
    "betfair_source_manifest_receipt_sha256",
    "betfair_sources",
    "betfair_source_hashes",
    "eligible_predictors_sha256",
    "race_audit_sha256",
    "candidate_races",
    "eligible_races",
    "eligible_runner_rows",
    "exclusions_by_reason",
    "predictor_fields",
    "BSP_as_predictor",
    "actual_off_as_predictor",
    "outcome_rows_inspected",
    "scored_races",
}
FORWARD_PREDICTOR_FIELDS = frozenset(
    {
        "schema_version",
        "race_date",
        "sportsbet_venue",
        "race_number",
        "scheduled_race_time_raw",
        "win_market_id",
        "box_number",
        "selection_id",
        "sportsbet_normalized_probability",
        "betfair_scheduled_off_back_price",
        "sportsbet_source_sha256",
        "sportsbet_source_row_id",
        "betfair_source_file",
        "betfair_source_url",
        "betfair_source_sha256",
        "betfair_source_row_number",
    }
)
RACE_AUDIT_FIELDS = frozenset(
    {
        "schema_version",
        "race_date",
        "sportsbet_venue",
        "race_number",
        "scheduled_race_time_raw",
        "eligible",
        "exclusion_reason",
        "candidate_betfair_market_count",
    }
)
if set(BETFAIR_REQUIRED_COLUMNS) & BETFAIR_FORBIDDEN_PREDICTORS:
    raise RuntimeError("Betfair predictor whitelist includes a forbidden field")

# Frozen from the development audit. A track alias only nominates candidate
# markets; race number, exact clock and the complete box/TAB set must also agree.
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


class ForwardContractError(RuntimeError):
    """Raised when the prospective population or scoring contract fails."""


@dataclass(frozen=True)
class BetfairRunner:
    race_date: str
    scheduled_clock: str
    track: str
    race_number: int
    win_market_id: str
    selection_id: str
    tab_number: int
    runner_name: str
    scheduled_off_back_price: float
    source_file: str
    source_url: str
    source_sha256: str
    source_row_number: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def normalized_clock(value: Any, field: str) -> str:
    match = TIME_RE.fullmatch(str(value or "").strip())
    if not match:
        raise ForwardContractError(f"invalid {field}")
    hour, minute, second = (int(match.group(index)) for index in range(1, 4))
    if hour > 23 or minute > 59 or second > 59:
        raise ForwardContractError(f"invalid {field}")
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ForwardContractError(f"invalid {field}")
    return value.strip()


def parse_positive_int(value: Any, field: str) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ForwardContractError(f"invalid {field}") from exc
    if parsed <= 0:
        raise ForwardContractError(f"invalid {field}")
    return parsed


def parse_probability(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ForwardContractError("invalid Sportsbet probability") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ForwardContractError("invalid Sportsbet probability")
    return parsed


def parse_price(value: Any) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ForwardContractError("invalid Betfair scheduled-off price") from exc
    if not math.isfinite(parsed) or parsed <= 1.0:
        raise ForwardContractError("invalid Betfair scheduled-off price")
    return parsed


def closed_json_object(path: Path, fields: set[str], label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ForwardContractError(f"invalid {label} JSON") from exc
    if not isinstance(value, dict) or set(value) != fields:
        raise ForwardContractError(f"invalid {label} fields")
    return value


def _validate_official_betfair_url(value: Any, filename: str) -> str:
    url = nonempty_string(value, "Betfair source URL")
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "promo.betfair.com"
        or Path(unquote(parsed.path)).name != filename
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ForwardContractError("invalid official Betfair source URL")
    return url


def _validate_betfair_predictor_header(header: Sequence[str]) -> None:
    if len(header) != len(set(header)):
        raise ForwardContractError("duplicate Betfair CSV header")
    forbidden = sorted(set(header) & BETFAIR_FORBIDDEN_PREDICTORS)
    if forbidden:
        raise ForwardContractError(
            f"result-bearing Betfair columns are quarantined: {forbidden}"
        )
    missing = [field for field in BETFAIR_REQUIRED_COLUMNS if field not in header]
    if missing:
        raise ForwardContractError(f"missing Betfair columns: {missing}")
    unexpected = sorted(set(header) - set(BETFAIR_REQUIRED_COLUMNS))
    if unexpected:
        raise ForwardContractError(
            f"unexpected Betfair predictor projection columns: {unexpected}"
        )


def _verify_betfair_predictor_header(path: Path) -> None:
    """Reject a result-bearing Betfair file before any data row or full hash read."""

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ForwardContractError("empty Betfair CSV") from exc
    _validate_betfair_predictor_header(header)


def verify_betfair_source_receipt(
    receipt_path: Path,
    betfair_paths: Sequence[Path],
) -> tuple[dict[str, dict[str, Any]], str]:
    receipt = closed_json_object(
        receipt_path,
        {
            "schema_version",
            "terminal_state",
            "window",
            "declared_complete_without_results",
            "labels_inspected",
            "results_inspected",
            "sources",
        },
        "Betfair source manifest receipt",
    )
    if (
        receipt["schema_version"] != BETFAIR_SOURCE_RECEIPT_SCHEMA
        or receipt["terminal_state"] != "BETFAIR_FORWARD_SOURCES_FROZEN_LABEL_BLIND"
        or receipt["window"]
        != {
            "start_date_inclusive": FORWARD_START.isoformat(),
            "end_date_inclusive": FORWARD_END.isoformat(),
        }
        or receipt["declared_complete_without_results"] is not True
        or receipt["labels_inspected"] is not False
        or receipt["results_inspected"] is not False
        or not isinstance(receipt["sources"], list)
    ):
        raise ForwardContractError("invalid label-blind Betfair source receipt")
    if len(betfair_paths) != 2 or len({path.resolve() for path in betfair_paths}) != 2:
        raise ForwardContractError("exactly two unique Betfair source paths are required")
    if {path.name for path in betfair_paths} != EXPECTED_BETFAIR_FILENAMES:
        raise ForwardContractError("exactly the official August and September Betfair files are required")

    sources: dict[str, dict[str, Any]] = {}
    source_fields = {"filename", "source_url", "byte_size", "sha256"}
    for item in receipt["sources"]:
        if not isinstance(item, dict) or set(item) != source_fields:
            raise ForwardContractError("invalid Betfair source receipt entry")
        filename = nonempty_string(item["filename"], "Betfair source filename")
        if filename in sources:
            raise ForwardContractError("duplicate Betfair source receipt filename")
        source_url = _validate_official_betfair_url(item["source_url"], filename)
        if not isinstance(item["byte_size"], int) or isinstance(item["byte_size"], bool):
            raise ForwardContractError("invalid Betfair source byte size")
        byte_size = parse_positive_int(item["byte_size"], "Betfair source byte size")
        source_sha256 = nonempty_string(item["sha256"], "Betfair source SHA-256")
        if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
            raise ForwardContractError("invalid Betfair source SHA-256")
        sources[filename] = {
            "filename": filename,
            "source_url": source_url,
            "byte_size": byte_size,
            "sha256": source_sha256,
        }
    if set(sources) != EXPECTED_BETFAIR_FILENAMES:
        raise ForwardContractError("Betfair source receipt does not contain the exact monthly files")
    for path in betfair_paths:
        source = sources[path.name]
        _verify_betfair_predictor_header(path)
        if path.stat().st_size != source["byte_size"] or sha256_file(path) != source["sha256"]:
            raise ForwardContractError(f"Betfair source receipt drift: {path.name}")
    return sources, sha256_file(receipt_path)


def verify_frozen_artifacts(artifact_dir: Path) -> dict[str, str]:
    expected = {
        "frozen_consensus_rule.json": EXPECTED_RULE_SHA256,
        "protocol.json": EXPECTED_PROTOCOL_SHA256,
    }
    actual = {name: sha256_file(artifact_dir / name) for name in expected}
    if actual != expected:
        raise ForwardContractError("frozen artifact hash mismatch")
    predecessor_eligibility_path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/sportsbet_betfair_consensus_freeze_20260817_report_only"
        / "future_eligibility_protocol.json"
    )
    if sha256_file(predecessor_eligibility_path) != EXPECTED_PREDECESSOR_ELIGIBILITY_SHA256:
        raise ForwardContractError("frozen predecessor eligibility hash mismatch")
    scorer_path = Path(frozen.__file__).resolve()
    if sha256_file(scorer_path) != EXPECTED_SCORER_SHA256:
        raise ForwardContractError("frozen scorer hash mismatch")
    rule = json.loads((artifact_dir / "frozen_consensus_rule.json").read_text(encoding="utf-8"))
    if (
        rule.get("frozen") is not True
        or rule.get("selected_betfair_weight") != BETFAIR_WEIGHT
        or not math.isclose(
            float(rule.get("selected_sportsbet_weight")),
            SPORTSBET_WEIGHT,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ForwardContractError("frozen 95/5 rule mismatch")
    return {
        **actual,
        "predecessor_future_eligibility_protocol.json": (
            EXPECTED_PREDECESSOR_ELIGIBILITY_SHA256
        ),
        "scorer": EXPECTED_SCORER_SHA256,
    }


def verify_completeness_receipt(receipt_path: Path, sportsbet_path: Path) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "sportsbet_predictor_filename",
        "sportsbet_predictor_sha256",
        "start_date_inclusive",
        "end_date_inclusive",
        "declared_complete_without_results",
        "labels_inspected",
        "results_inspected",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected:
        raise ForwardContractError("invalid completeness receipt fields")
    if sportsbet_path.name != EXPECTED_SPORTSBET_PREDICTOR_FILENAME:
        raise ForwardContractError("Sportsbet predictor path is not the frozen input name")
    required_values = {
        "schema_version": "sportsbet_forward_completeness_receipt_v1",
        "sportsbet_predictor_filename": EXPECTED_SPORTSBET_PREDICTOR_FILENAME,
        "sportsbet_predictor_sha256": sha256_file(sportsbet_path),
        "start_date_inclusive": FORWARD_START.isoformat(),
        "end_date_inclusive": FORWARD_END.isoformat(),
        "declared_complete_without_results": True,
        "labels_inspected": False,
        "results_inspected": False,
    }
    if receipt != required_values:
        raise ForwardContractError("completeness receipt does not prove a label-blind fixed window")
    return receipt


def load_sportsbet(
    path: Path,
    receipt_path: Path,
) -> dict[tuple[str, str, int, str], list[dict[str, Any]]]:
    verify_completeness_receipt(receipt_path, path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                raise ForwardContractError(f"blank JSONL row at {path}:{line_number}")
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ForwardContractError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict) or frozenset(row) != SPORTSBET_ROW_FIELDS:
                raise ForwardContractError(f"unexpected fields at {path}:{line_number}")
            if row["schema_version"] != SPORTSBET_ROW_SCHEMA:
                raise ForwardContractError(f"schema mismatch at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ForwardContractError(f"empty JSONL input: {path}")
    grouped: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    seen_source_ids: set[str] = set()
    source_hashes: set[str] = set()
    for row in rows:
        try:
            race_date = date.fromisoformat(nonempty_string(row["race_date"], "race_date"))
        except ValueError as exc:
            raise ForwardContractError("invalid Sportsbet race_date") from exc
        if not FORWARD_START <= race_date <= FORWARD_END:
            raise ForwardContractError("Sportsbet row outside frozen future window")
        venue = nonempty_string(row["sportsbet_venue"], "sportsbet_venue")
        if venue not in SPORTSBET_TO_BETFAIR_TRACKS:
            raise ForwardContractError(f"unfrozen Sportsbet venue alias: {venue}")
        race_number = parse_positive_int(row["race_number"], "race_number")
        clock = normalized_clock(row["scheduled_race_time_raw"], "Sportsbet scheduled clock")
        box = parse_positive_int(row["box_number"], "box_number")
        if box > 8:
            raise ForwardContractError("reserve or invalid Sportsbet box")
        row["race_number"] = race_number
        row["box_number"] = box
        row["sportsbet_normalized_probability"] = parse_probability(
            row["sportsbet_normalized_probability"]
        )
        runner_name = nonempty_string(row["runner_name"], "Sportsbet runner_name")
        if not frozen.normalized_name(runner_name):
            raise ForwardContractError("invalid normalized Sportsbet runner_name")
        row["runner_name"] = runner_name
        source_id = nonempty_string(row["sportsbet_source_row_id"], "Sportsbet source row identity")
        if source_id in seen_source_ids:
            raise ForwardContractError("duplicate Sportsbet source row identity")
        seen_source_ids.add(source_id)
        source_hash = str(row["sportsbet_source_sha256"])
        if not re.fullmatch(r"[0-9a-f]{64}", source_hash):
            raise ForwardContractError("invalid Sportsbet source SHA-256")
        source_hashes.add(source_hash)
        grouped[(race_date.isoformat(), venue, race_number, clock)].append(row)
    if len(source_hashes) != 1:
        raise ForwardContractError("Sportsbet projection must bind exactly one source hash")
    for key, race_rows in grouped.items():
        boxes = [int(row["box_number"]) for row in race_rows]
        if len(boxes) < 2 or len(boxes) != len(set(boxes)):
            raise ForwardContractError(f"invalid Sportsbet runner set: {key}")
        total = math.fsum(float(row["sportsbet_normalized_probability"]) for row in race_rows)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ForwardContractError(f"Sportsbet probabilities do not sum to one: {key}")
    return grouped


def _project_betfair_csv(path: Path, source: Mapping[str, Any]) -> list[BetfairRunner]:
    match = BETFAIR_FILE_RE.fullmatch(path.name)
    if not match or (int(match.group(1)), int(match.group(2))) not in {(2026, 8), (2026, 9)}:
        raise ForwardContractError("Betfair source filename is outside frozen August/September files")
    file_year, file_month = int(match.group(1)), int(match.group(2))
    source_sha256 = nonempty_string(source["sha256"], "Betfair source SHA-256")
    source_url = _validate_official_betfair_url(source["source_url"], path.name)
    projected: list[BetfairRunner] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ForwardContractError("empty Betfair CSV") from exc
        _validate_betfair_predictor_header(header)
        indexes = {field: header.index(field) for field in BETFAIR_REQUIRED_COLUMNS}
        for row_number, values in enumerate(reader, 2):
            if len(values) != len(header):
                raise ForwardContractError(f"Betfair row width mismatch at {path}:{row_number}")
            raw_date = values[indexes["LOCAL_MEETING_DATE"]].strip()
            try:
                race_date = date.fromisoformat(raw_date)
            except ValueError as exc:
                raise ForwardContractError(f"invalid Betfair race date at {path}:{row_number}") from exc
            if (race_date.year, race_date.month) != (file_year, file_month):
                raise ForwardContractError(f"Betfair source file month mismatch at {path}:{row_number}")
            if not FORWARD_START <= race_date <= FORWARD_END:
                continue
            track = nonempty_string(values[indexes["TRACK"]], "Betfair TRACK")
            win_market_id = nonempty_string(
                values[indexes["WIN_MARKET_ID"]],
                "Betfair WIN_MARKET_ID",
            )
            selection_id = nonempty_string(
                values[indexes["SELECTION_ID"]],
                "Betfair SELECTION_ID",
            )
            runner_name = nonempty_string(
                values[indexes["SELECTION_NAME"]],
                "Betfair SELECTION_NAME",
            )
            if not frozen.normalized_name(runner_name):
                raise ForwardContractError("invalid normalized Betfair runner name")
            projected.append(
                BetfairRunner(
                    race_date=race_date.isoformat(),
                    scheduled_clock=normalized_clock(
                        values[indexes["SCHEDULED_RACE_TIME"]], "Betfair scheduled clock"
                    ),
                    track=track,
                    race_number=parse_positive_int(values[indexes["RACE_NO"]], "RACE_NO"),
                    win_market_id=win_market_id,
                    selection_id=selection_id,
                    tab_number=parse_positive_int(values[indexes["TAB_NUMBER"]], "TAB_NUMBER"),
                    runner_name=runner_name,
                    scheduled_off_back_price=parse_price(
                        values[indexes["BEST_AVAIL_BACK_AT_SCHEDULED_OFF"]]
                    ),
                    source_file=path.name,
                    source_url=source_url,
                    source_sha256=source_sha256,
                    source_row_number=row_number,
                )
            )
    return projected


def load_betfair(
    paths: Sequence[Path],
    sources: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str, int, str, str], list[BetfairRunner]]:
    if (
        len(paths) != 2
        or len({path.resolve() for path in paths}) != 2
        or {path.name for path in paths} != EXPECTED_BETFAIR_FILENAMES
    ):
        raise ForwardContractError("exactly the official August and September Betfair files are required")
    grouped: dict[tuple[str, str, int, str, str], list[BetfairRunner]] = defaultdict(list)
    seen_runner_keys: set[tuple[str, str]] = set()
    market_identities: dict[str, tuple[str, str, int, str]] = {}
    for path in sorted(paths, key=lambda item: item.name):
        for runner in _project_betfair_csv(path, sources[path.name]):
            runner_key = (runner.win_market_id, runner.selection_id)
            if runner_key in seen_runner_keys:
                raise ForwardContractError("duplicate Betfair market/selection row")
            seen_runner_keys.add(runner_key)
            identity = (
                runner.race_date,
                runner.track,
                runner.race_number,
                runner.scheduled_clock,
            )
            previous_identity = market_identities.get(runner.win_market_id)
            if previous_identity is not None and previous_identity != identity:
                raise ForwardContractError("Betfair win_market_id maps to multiple races")
            market_identities[runner.win_market_id] = identity
            grouped[
                (
                    runner.race_date,
                    runner.track,
                    runner.race_number,
                    runner.scheduled_clock,
                    runner.win_market_id,
                )
            ].append(runner)
    return grouped


def seal_population(
    sportsbet: Mapping[tuple[str, str, int, str], Sequence[dict[str, Any]]],
    betfair: Mapping[tuple[str, str, int, str, str], Sequence[BetfairRunner]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictor_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    betfair_by_identity: dict[
        tuple[str, int, str],
        list[tuple[tuple[str, str, int, str, str], Sequence[BetfairRunner]]],
    ] = defaultdict(list)
    for key, rows in betfair.items():
        betfair_by_identity[(key[0], key[2], key[3])].append((key, rows))
    for identity in sorted(sportsbet):
        race_date, venue, race_number, clock = identity
        sportsbet_rows = sorted(sportsbet[identity], key=lambda row: int(row["box_number"]))
        candidates = [
            (key, rows)
            for key, rows in betfair_by_identity.get((race_date, race_number, clock), ())
            if key[1] in SPORTSBET_TO_BETFAIR_TRACKS[venue]
        ]
        reason = None
        if not candidates:
            reason = "NO_EXACT_BETFAIR_RACE_MATCH"
        elif len(candidates) > 1:
            reason = "AMBIGUOUS_BETFAIR_RACE_MATCH"
        else:
            _, betfair_rows_raw = candidates[0]
            betfair_rows = sorted(betfair_rows_raw, key=lambda row: row.tab_number)
            sportsbet_boxes = tuple(int(row["box_number"]) for row in sportsbet_rows)
            betfair_tabs = tuple(row.tab_number for row in betfair_rows)
            if any(tab > 8 for tab in betfair_tabs) or sportsbet_boxes != betfair_tabs:
                reason = "RUNNER_SET_MISMATCH_RESERVE_OR_SCRATCH"
            elif any(
                frozen.normalized_name(sb["runner_name"]) != frozen.normalized_name(bf.runner_name)
                for sb, bf in zip(sportsbet_rows, betfair_rows)
            ):
                reason = "RUNNER_NAME_CORROBORATION_MISMATCH"
        audit_rows.append(
            {
                "schema_version": "sportsbet_betfair_forward_race_audit_v1",
                "race_date": race_date,
                "sportsbet_venue": venue,
                "race_number": race_number,
                "scheduled_race_time_raw": clock,
                "eligible": reason is None,
                "exclusion_reason": reason,
                "candidate_betfair_market_count": len(candidates),
            }
        )
        if reason is not None:
            continue
        betfair_key, betfair_rows_raw = candidates[0]
        betfair_rows = sorted(betfair_rows_raw, key=lambda row: row.tab_number)
        for sb, bf in zip(sportsbet_rows, betfair_rows):
            predictor_rows.append(
                {
                    "schema_version": "sportsbet_betfair_forward_predictor_v1",
                    "race_date": race_date,
                    "sportsbet_venue": venue,
                    "race_number": race_number,
                    "scheduled_race_time_raw": clock,
                    "win_market_id": betfair_key[4],
                    "box_number": int(sb["box_number"]),
                    "selection_id": bf.selection_id,
                    "sportsbet_normalized_probability": float(
                        sb["sportsbet_normalized_probability"]
                    ),
                    "betfair_scheduled_off_back_price": bf.scheduled_off_back_price,
                    "sportsbet_source_sha256": str(sb["sportsbet_source_sha256"]),
                    "sportsbet_source_row_id": str(sb["sportsbet_source_row_id"]),
                    "betfair_source_file": bf.source_file,
                    "betfair_source_url": bf.source_url,
                    "betfair_source_sha256": bf.source_sha256,
                    "betfair_source_row_number": bf.source_row_number,
                }
            )
    if not predictor_rows:
        raise ForwardContractError("frozen future population has no eligible races")
    return predictor_rows, audit_rows


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_jsonl_exclusive(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("xb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row))
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _write_bytes_atomic_exclusive(path: Path, payload: bytes) -> None:
    if path.exists():
        raise ForwardContractError(f"output already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
        _fsync_directory(path.parent)
    except FileExistsError as exc:
        raise ForwardContractError(f"output already exists: {path}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def write_seal(
    output_dir: Path,
    predictor_rows: Sequence[dict[str, Any]],
    audit_rows: Sequence[dict[str, Any]],
    sportsbet_path: Path,
    betfair_paths: Sequence[Path],
    receipt_path: Path,
    betfair_source_receipt_path: Path,
    betfair_sources: Mapping[str, Mapping[str, Any]],
    frozen_hashes: Mapping[str, str],
) -> None:
    if output_dir.exists():
        raise ForwardContractError("population output directory already exists")
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    predictor_path = staging_dir / "eligible_predictors.jsonl"
    audit_path = staging_dir / "race_audit.jsonl"
    try:
        _write_jsonl_exclusive(predictor_path, predictor_rows)
        _write_jsonl_exclusive(audit_path, audit_rows)
        race_count = len(
            {
                (
                    row["race_date"],
                    row["sportsbet_venue"],
                    row["race_number"],
                    row["scheduled_race_time_raw"],
                    row["win_market_id"],
                )
                for row in predictor_rows
            }
        )
        exclusions = Counter(
            str(row["exclusion_reason"])
            for row in audit_rows
            if row["exclusion_reason"] is not None
        )
        manifest = {
            "schema_version": POPULATION_MANIFEST_SCHEMA,
            "terminal_state": "FORWARD_POPULATION_SEALED_UNSCORED",
            "window": {
                "start_date_inclusive": FORWARD_START.isoformat(),
                "end_date_inclusive": FORWARD_END.isoformat(),
            },
            "frozen_hashes": dict(frozen_hashes),
            "sportsbet_predictor_sha256": sha256_file(sportsbet_path),
            "sportsbet_completeness_receipt_sha256": sha256_file(receipt_path),
            "betfair_source_manifest_receipt_sha256": sha256_file(
                betfair_source_receipt_path
            ),
            "betfair_sources": [
                dict(betfair_sources[name]) for name in sorted(betfair_sources)
            ],
            "betfair_source_hashes": {
                path.name: sha256_file(path)
                for path in sorted(betfair_paths, key=lambda item: item.name)
            },
            "eligible_predictors_sha256": sha256_file(predictor_path),
            "race_audit_sha256": sha256_file(audit_path),
            "candidate_races": len(audit_rows),
            "eligible_races": race_count,
            "eligible_runner_rows": len(predictor_rows),
            "exclusions_by_reason": dict(sorted(exclusions.items())),
            "predictor_fields": [
                "sportsbet_normalized_probability",
                "betfair_scheduled_off_back_price",
            ],
            "BSP_as_predictor": False,
            "actual_off_as_predictor": False,
            "outcome_rows_inspected": 0,
            "scored_races": 0,
        }
        _write_bytes_exclusive(
            staging_dir / "population_manifest.json",
            canonical_json_bytes(manifest),
        )
        if output_dir.exists():
            raise ForwardContractError("population output directory already exists")
        os.rename(staging_dir, output_dir)
        _fsync_directory(output_dir.parent)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise


def verify_population_approval_receipt(
    population_dir: Path,
    approval_receipt_path: Path,
    frozen_hashes: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = population_dir / "population_manifest.json"
    manifest = closed_json_object(
        manifest_path,
        POPULATION_MANIFEST_FIELDS,
        "population manifest",
    )
    manifest_sha256 = sha256_file(manifest_path)
    approval = closed_json_object(
        approval_receipt_path,
        {
            "schema_version",
            "terminal_state",
            "external_approval",
            "population_manifest_sha256",
            "population_review_used_results",
            "approved_by",
            "approved_at_utc",
        },
        "population approval receipt",
    )
    if (
        approval["schema_version"] != POPULATION_APPROVAL_RECEIPT_SCHEMA
        or approval["terminal_state"] != "POPULATION_EXTERNALLY_APPROVED_FOR_ONE_SHOT_SCORE"
        or approval["external_approval"] is not True
        or approval["population_manifest_sha256"] != manifest_sha256
        or approval["population_review_used_results"] is not False
        or not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
            nonempty_string(approval["approved_at_utc"], "approval timestamp"),
        )
    ):
        raise ForwardContractError("invalid or drifting population approval receipt")
    nonempty_string(approval["approved_by"], "external approver identity")
    if (
        manifest["schema_version"] != POPULATION_MANIFEST_SCHEMA
        or manifest["terminal_state"] != "FORWARD_POPULATION_SEALED_UNSCORED"
        or manifest["window"]
        != {
            "start_date_inclusive": FORWARD_START.isoformat(),
            "end_date_inclusive": FORWARD_END.isoformat(),
        }
        or manifest["frozen_hashes"] != dict(frozen_hashes)
        or manifest["predictor_fields"]
        != ["sportsbet_normalized_probability", "betfair_scheduled_off_back_price"]
        or manifest["BSP_as_predictor"] is not False
        or manifest["actual_off_as_predictor"] is not False
        or manifest["outcome_rows_inspected"] != 0
        or manifest["scored_races"] != 0
    ):
        raise ForwardContractError("population manifest contract drift")
    predictor_path = population_dir / "eligible_predictors.jsonl"
    audit_path = population_dir / "race_audit.jsonl"
    if sha256_file(predictor_path) != manifest["eligible_predictors_sha256"]:
        raise ForwardContractError("sealed predictor hash mismatch")
    if sha256_file(audit_path) != manifest["race_audit_sha256"]:
        raise ForwardContractError("sealed race audit hash mismatch")
    if not isinstance(manifest["betfair_sources"], list) or len(manifest["betfair_sources"]) != 2:
        raise ForwardContractError("sealed Betfair source manifest is invalid")
    source_hashes: dict[str, str] = {}
    for source in manifest["betfair_sources"]:
        if not isinstance(source, dict) or set(source) != {
            "filename", "source_url", "byte_size", "sha256",
        }:
            raise ForwardContractError("sealed Betfair source entry is invalid")
        filename = nonempty_string(source["filename"], "sealed Betfair filename")
        _validate_official_betfair_url(source["source_url"], filename)
        parse_positive_int(source["byte_size"], "sealed Betfair byte size")
        source_hash = nonempty_string(source["sha256"], "sealed Betfair SHA-256")
        if (
            not re.fullmatch(r"[0-9a-f]{64}", source_hash)
            or filename in source_hashes
        ):
            raise ForwardContractError("sealed Betfair source entry is invalid")
        source_hashes[filename] = source_hash
    if set(source_hashes) != EXPECTED_BETFAIR_FILENAMES:
        raise ForwardContractError("sealed Betfair source filenames drifted")
    if manifest["betfair_source_hashes"] != source_hashes:
        raise ForwardContractError("sealed Betfair source hashes drifted")
    if any(
        not isinstance(manifest[field], int) or isinstance(manifest[field], bool)
        for field in ("candidate_races", "eligible_races", "eligible_runner_rows")
    ):
        raise ForwardContractError("invalid population manifest counts")
    if manifest["candidate_races"] < 1 or manifest["eligible_races"] < 1:
        raise ForwardContractError("empty approved population manifest")
    audit_rows: list[dict[str, Any]] = []
    with audit_path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                raise ForwardContractError(f"blank JSONL row at {audit_path}:{line_number}")
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ForwardContractError(
                    f"invalid JSONL at {audit_path}:{line_number}"
                ) from exc
            if not isinstance(row, dict) or frozenset(row) != RACE_AUDIT_FIELDS:
                raise ForwardContractError(f"unexpected fields at {audit_path}:{line_number}")
            if row["schema_version"] != "sportsbet_betfair_forward_race_audit_v1":
                raise ForwardContractError(f"schema mismatch at {audit_path}:{line_number}")
            audit_rows.append(row)
    if not audit_rows:
        raise ForwardContractError(f"empty JSONL input: {audit_path}")
    exclusions = Counter(
        str(row["exclusion_reason"])
        for row in audit_rows
        if row["exclusion_reason"] is not None
    )
    if (
        len(audit_rows) != manifest["candidate_races"]
        or sum(row["eligible"] is True for row in audit_rows) != manifest["eligible_races"]
        or dict(sorted(exclusions.items())) != manifest["exclusions_by_reason"]
    ):
        raise ForwardContractError("population audit counts drifted")
    return manifest, {
        "population_manifest_sha256": manifest_sha256,
        "population_approval_receipt_sha256": sha256_file(approval_receipt_path),
        "approved_by": approval["approved_by"],
        "approved_at_utc": approval["approved_at_utc"],
    }


def authorize_and_load_sealed_races_for_score(
    population_dir: Path,
    results_path: Path,
    approval_receipt_path: Path,
    frozen_hashes: Mapping[str, str],
) -> tuple[
    list[frozen.Race],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    Path,
    Path,
]:
    """Apply every gate and consume the attempt before opening result bytes."""

    enforce_score_date()
    manifest, approval_provenance = verify_population_approval_receipt(
        population_dir,
        approval_receipt_path,
        frozen_hashes,
    )
    output_path, consumed_marker_path = consume_score_once(
        population_dir,
        approval_provenance,
    )
    if manifest.get("terminal_state") != "FORWARD_POPULATION_SEALED_UNSCORED":
        raise ForwardContractError("population is not sealed and unscored")
    predictor_path = population_dir / "eligible_predictors.jsonl"
    if sha256_file(predictor_path) != manifest.get("eligible_predictors_sha256"):
        raise ForwardContractError("sealed predictor hash mismatch")
    predictor_rows: list[dict[str, Any]] = []
    with predictor_path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                raise ForwardContractError(f"blank JSONL row at {predictor_path}:{line_number}")
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ForwardContractError(
                    f"invalid JSONL at {predictor_path}:{line_number}"
                ) from exc
            if not isinstance(row, dict) or frozenset(row) != FORWARD_PREDICTOR_FIELDS:
                raise ForwardContractError(
                    f"unexpected fields at {predictor_path}:{line_number}"
                )
            if row["schema_version"] != "sportsbet_betfair_forward_predictor_v1":
                raise ForwardContractError(f"schema mismatch at {predictor_path}:{line_number}")
            predictor_rows.append(row)
    if not predictor_rows:
        raise ForwardContractError(f"empty JSONL input: {predictor_path}")
    results: list[dict[str, Any]] = []
    with results_path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                raise ForwardContractError(
                    f"blank JSONL row at {results_path}:{line_number}"
                )
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ForwardContractError(
                    f"invalid JSONL at {results_path}:{line_number}"
                ) from exc
            if not isinstance(row, dict) or frozenset(row) != RESULT_ROW_FIELDS:
                raise ForwardContractError(
                    f"unexpected fields at {results_path}:{line_number}"
                )
            if row["schema_version"] != RESULT_ROW_SCHEMA:
                raise ForwardContractError(
                    f"schema mismatch at {results_path}:{line_number}"
                )
            results.append(row)
    if not results:
        raise ForwardContractError(f"empty JSONL input: {results_path}")
    result_by_key: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    result_source_hashes: set[str] = set()
    result_source_row_ids: set[str] = set()
    for row in results:
        try:
            result_date = date.fromisoformat(nonempty_string(row["race_date"], "result race_date"))
        except ValueError as exc:
            raise ForwardContractError("invalid approved result race_date") from exc
        if not FORWARD_START <= result_date <= FORWARD_END:
            raise ForwardContractError("approved result outside frozen future window")
        venue = nonempty_string(row["sportsbet_venue"], "result sportsbet_venue")
        if venue not in SPORTSBET_TO_BETFAIR_TRACKS:
            raise ForwardContractError("approved result has unfrozen venue alias")
        key = (
            result_date.isoformat(),
            venue,
            parse_positive_int(row["race_number"], "result race_number"),
            normalized_clock(row["scheduled_race_time_raw"], "result scheduled clock"),
        )
        if key in result_by_key:
            raise ForwardContractError("duplicate approved result race")
        winner_box = parse_positive_int(row["winner_box"], "winner_box")
        if winner_box > 8:
            raise ForwardContractError("invalid approved winner box")
        row["winner_box"] = winner_box
        source_hash = nonempty_string(
            row["approved_result_source_sha256"],
            "approved result source hash",
        )
        if not re.fullmatch(r"[0-9a-f]{64}", source_hash):
            raise ForwardContractError("invalid approved result source hash")
        source_row_id = nonempty_string(
            row["approved_result_source_row_id"],
            "approved result source row identity",
        )
        if source_row_id in result_source_row_ids:
            raise ForwardContractError("duplicate approved result source row identity")
        result_source_row_ids.add(source_row_id)
        result_source_hashes.add(source_hash)
        result_by_key[key] = row
    if len(result_source_hashes) != 1:
        raise ForwardContractError("approved results must bind exactly one source hash")

    grouped: dict[tuple[str, str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    result_identity_markets: dict[tuple[str, str, int, str], str] = {}
    sportsbet_source_row_ids: set[str] = set()
    sportsbet_source_hashes: set[str] = set()
    betfair_source_rows: set[tuple[str, int]] = set()
    manifest_sources = {source["filename"]: source for source in manifest["betfair_sources"]}
    for row in predictor_rows:
        try:
            predictor_date = date.fromisoformat(
                nonempty_string(row["race_date"], "predictor race_date")
            )
        except ValueError as exc:
            raise ForwardContractError("invalid predictor race_date") from exc
        if not FORWARD_START <= predictor_date <= FORWARD_END:
            raise ForwardContractError("predictor outside frozen future window")
        venue = nonempty_string(row["sportsbet_venue"], "predictor sportsbet_venue")
        if venue not in SPORTSBET_TO_BETFAIR_TRACKS:
            raise ForwardContractError("predictor has unfrozen venue alias")
        race_number = parse_positive_int(row["race_number"], "predictor race_number")
        clock = normalized_clock(row["scheduled_race_time_raw"], "predictor scheduled clock")
        market_id = nonempty_string(row["win_market_id"], "predictor win_market_id")
        box = parse_positive_int(row["box_number"], "predictor box_number")
        if box > 8:
            raise ForwardContractError("predictor has reserve or invalid box")
        selection_id = nonempty_string(row["selection_id"], "predictor selection_id")
        row["race_number"] = race_number
        row["box_number"] = box
        row["selection_id"] = selection_id
        row["sportsbet_normalized_probability"] = parse_probability(
            row["sportsbet_normalized_probability"]
        )
        row["betfair_scheduled_off_back_price"] = parse_price(
            row["betfair_scheduled_off_back_price"]
        )
        sportsbet_source_hash = nonempty_string(
            row["sportsbet_source_sha256"],
            "predictor Sportsbet source SHA-256",
        )
        betfair_source_hash = nonempty_string(
            row["betfair_source_sha256"],
            "predictor Betfair source SHA-256",
        )
        if not re.fullmatch(r"[0-9a-f]{64}", sportsbet_source_hash) or not re.fullmatch(
            r"[0-9a-f]{64}", betfair_source_hash
        ):
            raise ForwardContractError("invalid sealed predictor source hash")
        sportsbet_source_hashes.add(sportsbet_source_hash)
        sportsbet_row_id = nonempty_string(
            row["sportsbet_source_row_id"],
            "predictor Sportsbet source row identity",
        )
        if sportsbet_row_id in sportsbet_source_row_ids:
            raise ForwardContractError("duplicate predictor Sportsbet source row identity")
        sportsbet_source_row_ids.add(sportsbet_row_id)
        source_file = nonempty_string(row["betfair_source_file"], "predictor Betfair source file")
        frozen.validate_betfair_source_month(source_file, predictor_date)
        source_url = _validate_official_betfair_url(row["betfair_source_url"], source_file)
        if (
            source_file not in manifest_sources
            or manifest_sources[source_file]["source_url"] != source_url
            or manifest_sources[source_file]["sha256"] != betfair_source_hash
        ):
            raise ForwardContractError("sealed predictor Betfair source provenance drifted")
        source_row = (
            source_file,
            parse_positive_int(row["betfair_source_row_number"], "Betfair source row number"),
        )
        if source_row in betfair_source_rows:
            raise ForwardContractError("duplicate predictor Betfair source row")
        betfair_source_rows.add(source_row)
        result_identity = (predictor_date.isoformat(), venue, race_number, clock)
        previous_market = result_identity_markets.get(result_identity)
        if previous_market is not None and previous_market != market_id:
            raise ForwardContractError("sealed result identity maps to multiple markets")
        result_identity_markets[result_identity] = market_id
        grouped[(*result_identity, market_id)].append(row)
    race_result_keys = {(key[0], key[1], key[2], key[3]) for key in grouped}
    if len(sportsbet_source_hashes) != 1:
        raise ForwardContractError("sealed predictors must bind one Sportsbet source hash")
    if set(result_by_key) != race_result_keys:
        raise ForwardContractError("approved result population does not exactly equal sealed races")

    races: list[frozen.Race] = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda row: int(row["box_number"]))
        boxes = tuple(int(row["box_number"]) for row in rows)
        selection_ids = tuple(str(row["selection_id"]) for row in rows)
        if (
            len(rows) < 2
            or len(set(boxes)) != len(boxes)
            or len(set(selection_ids)) != len(selection_ids)
        ):
            raise ForwardContractError("invalid sealed runner set")
        if len({str(row["sportsbet_source_sha256"]) for row in rows}) != 1:
            raise ForwardContractError("inconsistent Sportsbet source hash within race")
        if len({str(row["betfair_source_file"]) for row in rows}) != 1 or len(
            {str(row["betfair_source_sha256"]) for row in rows}
        ) != 1:
            raise ForwardContractError("inconsistent Betfair source provenance within race")
        result = result_by_key[(key[0], key[1], key[2], key[3])]
        if int(result["winner_box"]) not in boxes:
            raise ForwardContractError("approved winner is outside sealed runner set")
        prices = tuple(float(row["betfair_scheduled_off_back_price"]) for row in rows)
        sportsbet_raw = tuple(float(row["sportsbet_normalized_probability"]) for row in rows)
        if not math.isclose(math.fsum(sportsbet_raw), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ForwardContractError("sealed Sportsbet probabilities do not sum to one")
        sportsbet = frozen.normalize(
            sportsbet_raw,
            "sealed Sportsbet probabilities",
        )
        betfair = frozen.normalize(tuple(1.0 / price for price in prices), "sealed Betfair prices")
        races.append(
            frozen.Race(
                race_id=f"{key[0]}|{key[1]}|{key[2]}|{key[3]}",
                race_date=key[0],
                venue=key[1],
                race_number=key[2],
                scheduled_race_time_raw=key[3],
                win_market_id=key[4],
                split="future",
                boxes=boxes,
                selection_ids=selection_ids,
                sportsbet_probabilities=sportsbet,
                betfair_probabilities=betfair,
                betfair_prices=prices,
                winner_index=boxes.index(int(result["winner_box"])),
                sportsbet_matrix_row_indices=tuple(range(len(rows))),
                betfair_source_file=str(rows[0]["betfair_source_file"]),
                betfair_source_file_sha256=str(rows[0]["betfair_source_sha256"]),
            )
        )
    if len(predictor_rows) != manifest["eligible_runner_rows"] or len(races) != manifest[
        "eligible_races"
    ]:
        raise ForwardContractError("sealed population counts drifted")
    return races, {
        "approved_results_projection_sha256": sha256_file(results_path),
        "approved_result_source_sha256": next(iter(result_source_hashes)),
        "approved_result_rows": len(results),
    }, manifest, approval_provenance, output_path, consumed_marker_path


def forward_report_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Frozen Sportsbet Betfair forward evaluation",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "terminal_state",
            "window",
            "frozen_rule",
            "inputs",
            "population",
            "metrics",
            "paired_deltas_consensus_minus_sportsbet",
            "meeting_date_cluster_bootstrap_consensus_minus_sportsbet",
            "confirmation_rule",
        ],
        "properties": {
            "schema_version": {"const": "sportsbet_betfair_forward_evaluation_v1"},
            "terminal_state": {"const": "FORWARD_CONSENSUS_TEST_SCORED_ONCE"},
            "window": {"type": "object"},
            "frozen_rule": {"type": "object"},
            "inputs": {"type": "object"},
            "population": {"type": "object"},
            "metrics": {"type": "object"},
            "paired_deltas_consensus_minus_sportsbet": {"type": "object"},
            "meeting_date_cluster_bootstrap_consensus_minus_sportsbet": {"type": "object"},
            "confirmation_rule": {"type": "object"},
        },
    }


def score_forward(
    races: Sequence[frozen.Race],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    sportsbet = frozen.evaluate(races, "sportsbet")
    consensus = frozen.evaluate(races, "consensus", BETFAIR_WEIGHT)
    deltas = {
        metric: consensus[metric] - sportsbet[metric]
        for metric in (
            "log_loss",
            "brier",
            "top1_accuracy",
            "top2_accuracy",
            "top3_accuracy",
            "mean_winner_rank",
        )
    }
    bootstrap = frozen.bootstrap_delta(
        races,
        "consensus",
        BETFAIR_WEIGHT,
        replicates=10000,
        seed=20260817,
    )
    delta_log_loss = deltas["log_loss"]
    upper_95 = bootstrap["intervals"]["log_loss"]["upper_95"]
    return {
        "schema_version": "sportsbet_betfair_forward_evaluation_v1",
        "terminal_state": "FORWARD_CONSENSUS_TEST_SCORED_ONCE",
        "window": {
            "start_date_inclusive": FORWARD_START.isoformat(),
            "end_date_inclusive": FORWARD_END.isoformat(),
        },
        "frozen_rule": {
            "selected_betfair_weight": BETFAIR_WEIGHT,
            "selected_sportsbet_weight": SPORTSBET_WEIGHT,
        },
        "inputs": {
            **dict(provenance),
            "report_schema_sha256": hashlib.sha256(
                canonical_json_bytes(forward_report_schema())
            ).hexdigest(),
        },
        "population": {
            "races": len(races),
            "runner_rows": sum(len(race.boxes) for race in races),
            "meeting_date_clusters": len({race.cluster_key for race in races}),
        },
        "metrics": {
            "sportsbet": sportsbet,
            "consensus": consensus,
        },
        "paired_deltas_consensus_minus_sportsbet": deltas,
        "meeting_date_cluster_bootstrap_consensus_minus_sportsbet": bootstrap,
        "confirmation_rule": {
            "delta_log_loss_below_zero": delta_log_loss < 0.0,
            "cluster_bootstrap_upper_95_below_zero": upper_95 < 0.0,
            "confirmed": delta_log_loss < 0.0 and upper_95 < 0.0,
        },
    }


def enforce_score_date(today: date | None = None) -> None:
    effective_date = datetime.now(MELBOURNE).date() if today is None else today
    if effective_date <= FORWARD_END:
        raise ForwardContractError("forward score is forbidden until after 2026-09-30")


def fixed_score_paths(population_dir: Path, manifest_sha256: str) -> tuple[Path, Path]:
    prefix = f"sportsbet_betfair_forward_{manifest_sha256}"
    output_path = population_dir.parent / f"{prefix}.evaluation.json"
    consumed_marker_path = population_dir.parent / f"{prefix}.score_consumed.json"
    return output_path, consumed_marker_path


def consume_score_once(
    population_dir: Path,
    approval_provenance: Mapping[str, Any],
) -> tuple[Path, Path]:
    output_path, consumed_marker_path = fixed_score_paths(
        population_dir,
        str(approval_provenance["population_manifest_sha256"]),
    )
    if output_path.exists() or consumed_marker_path.exists():
        raise ForwardContractError("forward score path has already been consumed")
    marker = {
        "schema_version": "sportsbet_betfair_forward_score_consumed_v1",
        "terminal_state": "FORWARD_SCORE_ATTEMPT_CONSUMED",
        "population_manifest_sha256": approval_provenance["population_manifest_sha256"],
        "population_approval_receipt_sha256": approval_provenance[
            "population_approval_receipt_sha256"
        ],
        "approved_results_opened_before_marker": False,
        "fixed_output_filename": output_path.name,
    }
    try:
        _write_bytes_exclusive(consumed_marker_path, canonical_json_bytes(marker))
    except FileExistsError as exc:
        raise ForwardContractError("forward score path has already been consumed") from exc
    return output_path, consumed_marker_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal = subparsers.add_parser("seal-population")
    seal.add_argument("--sportsbet-predictors", type=Path, required=True)
    seal.add_argument("--sportsbet-completeness-receipt", type=Path, required=True)
    seal.add_argument("--betfair-source-manifest-receipt", type=Path, required=True)
    seal.add_argument("--betfair-csv", type=Path, action="append", required=True)
    seal.add_argument("--frozen-artifact-dir", type=Path, required=True)
    seal.add_argument("--output-dir", type=Path, required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--population-dir", type=Path, required=True)
    score.add_argument("--approved-results", type=Path, required=True)
    score.add_argument("--population-approval-receipt", type=Path, required=True)
    score.add_argument("--frozen-artifact-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frozen_hashes = verify_frozen_artifacts(args.frozen_artifact_dir)
    if args.command == "seal-population":
        betfair_sources, _ = verify_betfair_source_receipt(
            args.betfair_source_manifest_receipt,
            args.betfair_csv,
        )
        sportsbet = load_sportsbet(
            args.sportsbet_predictors,
            args.sportsbet_completeness_receipt,
        )
        betfair = load_betfair(args.betfair_csv, betfair_sources)
        predictor_rows, audit_rows = seal_population(sportsbet, betfair)
        write_seal(
            args.output_dir,
            predictor_rows,
            audit_rows,
            args.sportsbet_predictors,
            args.betfair_csv,
            args.sportsbet_completeness_receipt,
            args.betfair_source_manifest_receipt,
            betfair_sources,
            frozen_hashes,
        )
        print(json.dumps({"terminal_state": "FORWARD_POPULATION_SEALED_UNSCORED"}))
        return 0
    (
        races,
        result_provenance,
        manifest,
        approval_provenance,
        output_path,
        consumed_marker_path,
    ) = authorize_and_load_sealed_races_for_score(
        args.population_dir,
        args.approved_results,
        args.population_approval_receipt,
        frozen_hashes,
    )
    provenance = {
        **approval_provenance,
        **result_provenance,
        "score_consumed_marker_sha256": sha256_file(consumed_marker_path),
        "eligible_predictors_sha256": manifest["eligible_predictors_sha256"],
        "race_audit_sha256": manifest["race_audit_sha256"],
        "frozen_hashes": dict(frozen_hashes),
    }
    report = score_forward(races, provenance)
    _write_bytes_atomic_exclusive(output_path, frozen.canonical_json_bytes(report))
    print(json.dumps({"terminal_state": report["terminal_state"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
