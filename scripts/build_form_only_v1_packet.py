#!/usr/bin/env python3
"""Build deterministic odds-free FORM_ONLY_V1 acquisition packets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import stat
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path, PurePosixPath
from statistics import mean
from typing import Any, Callable, Iterable, Iterator, Mapping
from zoneinfo import ZoneInfo


MELBOURNE = ZoneInfo("Australia/Melbourne")
HISTORY_CAP = 20
DEVELOPMENT_END = date(2026, 7, 9)
OUT_OF_TIME_START = date(2026, 7, 11)
OUT_OF_TIME_END = date(2026, 8, 9)
EXCLUDED_PUBLISHED_RACE_ID = "Race 9 - TEMORA - 2026-06-10"
DEVELOPMENT_SCOPE = "development"
OUT_OF_TIME_SCOPE = "out_of_time"

FORBIDDEN_FEATURE_TOKENS = {
    "actual_win", "finish_position", "open", "low", "high", "sp", "odds",
    "result", "winner", "dog_name", "dog_identity", "speed", "sectional",
    "time", "opponent", "prize", "weather", "trainer",
}
FORBIDDEN_ARTIFACT_FIELDS = {
    "dog_name", "dog_name_token", "dog_identity", "dog_identity_sha256",
    "source_runner_id", "runner_id",
}

TRAINER_ARTIFACT_ROLES = {
    "development_exclusions.csv": "TRAINER_SAFE_ELIGIBILITY_METADATA",
    "development_features.csv": "MODEL_INPUT_DATA",
    "development_manifest.json": "TRAINER_SAFE_SPLIT_METADATA",
    "development_races.csv": "MODEL_INPUT_DATA",
    "development_runners.csv": "MODEL_INPUT_DATA",
    "feature_contract.json": "TRAINER_SAFE_FEATURE_METADATA",
    "market_coverage.json": "TRAINER_SAFE_BOUNDARY_METADATA",
    "out_of_time_manifest.json": "TRAINER_SAFE_SPLIT_METADATA",
    "out_of_time_races.csv": "MODEL_INPUT_DATA",
    "out_of_time_runners.csv": "MODEL_INPUT_DATA",
}
TRAINER_ARTIFACT_NAMES = set(TRAINER_ARTIFACT_ROLES)
CONTROL_PLANE_ARTIFACT_NAMES = {
    "artifact-manifest.sha256", "trainer_input_manifest.json",
}
TRAINER_ROOT_NAME = "trainer"
CONTROL_PLANE_ROOT_NAME = "control_plane"
PACKET_DOMAIN_ROOT_NAMES = (
    TRAINER_ROOT_NAME,
    CONTROL_PLANE_ROOT_NAME,
    "sealed_validation",
    "non_authoritative_diagnostic",
)
SEALED_VALIDATION_ARTIFACT_NAMES = {
    "development_source_inventory.csv", "out_of_time_exclusions.csv",
    "out_of_time_source_inventory.csv", "development_runner_alignment.csv",
    "out_of_time_runner_alignment.csv",
}
SEALED_VALIDATION_ARTIFACT_ROLES = {
    "development_runner_alignment.csv": "SEALED_IDENTITY_ALIGNMENT",
    "development_source_inventory.csv": "SEALED_SOURCE_PROVENANCE",
    "out_of_time_exclusions.csv": "SEALED_SOURCE_EXCLUSION_PROVENANCE",
    "out_of_time_runner_alignment.csv": "SEALED_IDENTITY_ALIGNMENT",
    "out_of_time_source_inventory.csv": "SEALED_SOURCE_PROVENANCE",
    "sealed-validation-manifest.sha256": "SEALED_DOMAIN_INTEGRITY_SIGNATURE",
}
DIAGNOSTIC_ARTIFACT_NAMES = {
    "overlap_reconciliation.csv", "reconciliation_summary.json",
}
DIAGNOSTIC_ARTIFACT_ROLES = {
    "non-authoritative-diagnostic-manifest.sha256": "DIAGNOSTIC_DOMAIN_INTEGRITY_SIGNATURE",
    "overlap_reconciliation.csv": "NON_AUTHORITATIVE_RECONCILIATION",
    "reconciliation_summary.json": "NON_AUTHORITATIVE_DIAGNOSTIC_SUMMARY",
}
CONTROL_PLANE_ARTIFACT_ROLES = {
    "artifact-manifest.sha256": "CONTROL_INTEGRITY_SIGNATURE",
    "trainer_input_manifest.json": "CONTROL_DECLARATION_METADATA",
}
AUTHORITATIVE_DOMAIN_ROOT_NAMES = (
    TRAINER_ROOT_NAME,
    CONTROL_PLANE_ROOT_NAME,
    "sealed_validation",
)
DOMAIN_ARTIFACT_ROLES = {
    TRAINER_ROOT_NAME: TRAINER_ARTIFACT_ROLES,
    CONTROL_PLANE_ROOT_NAME: CONTROL_PLANE_ARTIFACT_ROLES,
    "sealed_validation": SEALED_VALIDATION_ARTIFACT_ROLES,
    "non_authoritative_diagnostic": DIAGNOSTIC_ARTIFACT_ROLES,
}

VENUE_ALIASES = {
    "AP/K": "AP_K", "ANGLE PARK": "AP_K", "AP_K": "AP_K",
    "BALLARAT": "BAL", "BAL": "BAL", "BENDIGO": "BEN", "BEN": "BEN",
    "BROKEN HILL": "BH", "BROKEN-HILL": "BH", "BH": "BH",
    "CANNINGTON": "CANN", "CANN": "CANN", "CAPALABA": "CAPA", "CAPA": "CAPA",
    "CASINO": "CASO", "CASO": "CASO", "DARWIN": "DARW", "DARW": "DARW",
    "DUBBO": "DUBBO", "GARDENS": "GRDN", "THE GARDENS": "GRDN", "GRDN": "GRDN",
    "GEELONG": "GEE", "GEE": "GEE", "GOULBURN": "GOUL", "GOUL": "GOUL",
    "GRAFTON": "GRAF", "GRAF": "GRAF", "GAWLER": "GAWL", "GAWL": "GAWL",
    "GOSFORD": "GOSF", "GOSF": "GOSF", "GUNNEDAH": "GUNN", "GUNN": "GUNN",
    "HEALESVILLE": "HEA", "HEA": "HEA", "HOBART": "HOBT", "HOBT": "HOBT",
    "HORSHAM": "HOR", "HOR": "HOR", "LAUNCESTON": "LAU", "LCTN": "LAU", "LAU": "LAU",
    "LADBROKES-Q-STRAIGHT": "QOT", "LADBROKES-Q1-LAKESIDE": "QOT",
    "LADBROKES-Q2-PARKLANDS": "QOT", "Q1L": "QOT", "QST": "QOT", "QOT": "QOT",
    "MAITLAND": "MAIT", "MAIT": "MAIT", "MANDURAH": "MAND", "MAND": "MAND",
    "MEADOWS": "MEA", "THE MEADOWS": "MEA", "MEA": "MEA",
    "MOUNT GAMBIER": "MOUNT", "MOUNT": "MOUNT", "MT_G": "MOUNT",
    "MURRAY BRIDGE": "MURR", "MURRAY-BRIDGE-STRAIGHT": "MURR", "MURR": "MURR",
    "NORTHAM": "NOR", "NOR": "NOR", "NOWRA": "NOWRA", "RICHMOND": "RICH",
    "RICHMOND-STRAIGHT": "RICH", "RICH": "RICH", "ROCKHAMPTON": "ROCK", "ROCK": "ROCK",
    "SALE": "SAL", "SAL": "SAL", "SANDOWN": "SAN", "SAN": "SAN",
    "SHEPPARTON": "SHEP", "SHEP": "SHEP", "TAREE": "TAREE", "TAR": "TAREE",
    "TEMORA": "TEM", "TEM": "TEM", "TOWNSVILLE": "TWN", "TWN": "TWN",
    "TRARALGON": "TRA", "TRA": "TRA", "WAGGA": "WAG", "WAG": "WAG",
    "WARRAGUL": "WRGL", "WRGL": "WRGL", "WARRNAMBOOL": "WAR", "WAR": "WAR",
    "WENTWORTH PARK": "WPK", "W_PK": "WPK", "WPK": "WPK",
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_path(path), "bytes": path.stat().st_size}


def retained_file_record(path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def verify_retained_file_record(
    path: Path,
    payload: bytes,
    *,
    expected_sha256: str,
    expected_bytes: int | str | None = None,
    require_expected_bytes: bool = False,
) -> dict[str, Any]:
    actual = retained_file_record(path, payload)
    if actual["sha256"] != expected_sha256:
        raise ValueError(f"source hash mismatch: {path}")
    if require_expected_bytes and expected_bytes in (None, ""):
        raise ValueError(f"source byte declaration missing: {path}")
    if expected_bytes not in (None, "") and actual["bytes"] != int(expected_bytes):
        raise ValueError(f"source byte mismatch: {path}")
    return actual


def verify_file_record(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int | str | None = None,
    require_expected_bytes: bool = False,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = file_record(path)
    if actual["sha256"] != expected_sha256:
        raise ValueError(f"source hash mismatch: {path}")
    if require_expected_bytes and expected_bytes in (None, ""):
        raise ValueError(f"source byte declaration missing: {path}")
    if expected_bytes not in (None, "") and actual["bytes"] != int(expected_bytes):
        raise ValueError(f"source byte mismatch: {path}")
    return actual


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_set_digest(
    records: Iterable[Mapping[str, Any]], *, reject_duplicate_declarations: bool = True
) -> str:
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        record_path = Path(str(record["path"]))
        if not record_path.is_absolute() or ".." in record_path.parts:
            raise ValueError(f"source declaration path must be absolute and traversal-free: {record_path}")
        normalized = {
            "role": str(record["role"]),
            "path": str(record_path),
            "sha256": str(record["sha256"]),
            "bytes": int(record["bytes"]),
        }
        key = (normalized["role"], normalized["path"])
        previous = unique.get(key)
        if previous is not None:
            if previous != normalized:
                raise ValueError(f"conflicting source declaration: {key}")
            if reject_duplicate_declarations:
                raise ValueError(f"duplicate source declaration: {key}")
        unique[key] = normalized
    return canonical_digest([unique[key] for key in sorted(unique)])


def load_reproducibility_contract(
    path: Path, *, include_diagnostic: bool = True
) -> dict[str, Any]:
    try:
        payload = json.loads(
            _read_regular_path_no_follow(
                path, label="reproducibility contract"
            ).decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed reproducibility contract: {path}") from exc
    if payload.get("schema_version") != "form_only_v1_reproducibility_v4":
        raise ValueError(f"unsupported reproducibility contract: {path}")
    if not isinstance(payload.get("trusted_inputs"), dict):
        raise ValueError(f"reproducibility contract has no trusted_inputs: {path}")
    if not isinstance(payload.get("expected_output"), dict):
        raise ValueError(f"reproducibility contract has no typed expected_output: {path}")
    trusted = payload["trusted_inputs"]
    required_trust_domains = {"development", "out_of_time_freeze"}
    if not required_trust_domains.issubset(trusted):
        raise ValueError("reproducibility trust domains are incomplete")
    if include_diagnostic and set(trusted) != required_trust_domains | {"diagnostic"}:
        raise ValueError("reproducibility trust domains are incomplete")
    development = trusted["development"]
    authoritative_files = {
        role: record
        for role, record in (development.get("files") or {}).items()
        if role != "tier_a_provenance"
    }
    payload["construction_contract_sha256"] = canonical_digest({
        "development": {
            "files": authoritative_files,
            "authoritative_source_record_count": development.get(
                "authoritative_source_record_count"
            ),
            "authoritative_source_set_sha256": development.get(
                "authoritative_source_set_sha256"
            ),
        },
        "out_of_time_freeze": trusted["out_of_time_freeze"],
    })
    if include_diagnostic:
        diagnostic = trusted.get("diagnostic")
        if not isinstance(diagnostic, dict):
            raise ValueError("reproducibility diagnostic trust domain is malformed")
        payload["diagnostic_contract_sha256"] = canonical_digest(diagnostic)
    return payload


def stable_json(output_domain: _OutputDomain, name: str, value: Any) -> None:
    output_domain.write_bytes(
        name,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def stable_csv(
    output_domain: _OutputDomain,
    name: str,
    fieldnames: list[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    output_domain.write_bytes(name, handle.getvalue().encode("utf-8"))


def parse_timestamp(value: str, *, require_timezone: bool = False) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        if require_timezone:
            raise ValueError(f"timestamp has no source timezone: {value}")
        parsed = parsed.replace(tzinfo=MELBOURNE)
    return parsed.astimezone(MELBOURNE)


def capture_timestamp(
    metadata: Mapping[str, Any], *, require_timezone: bool = False
) -> datetime:
    for key in ("metadata_captured_at", "created_at", "capture_timestamp", "captured_at"):
        if metadata.get(key):
            return parse_timestamp(str(metadata[key]), require_timezone=require_timezone)
    raise ValueError("card sidecar has no capture timestamp")


def sidecar_jump_timestamp(metadata: Mapping[str, Any], race_id: str) -> datetime:
    info = metadata.get("race_info") or {}
    if (
        info.get("race_time_mapping_status") != "exact_url_match"
        or info.get("race_time_source") != "canonical_race_url"
    ):
        raise ValueError(f"jump time lacks exact canonical source evidence: {race_id}")
    race_date = str(info.get("date") or race_id.rsplit(" - ", 1)[-1])
    race_time = str(info.get("race_time") or "")
    for fmt in ("%Y-%m-%d %I:%M %p", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(f"{race_date} {race_time}", fmt).replace(tzinfo=MELBOURNE)
        except ValueError:
            continue
    raise ValueError(f"cannot parse jump timestamp for {race_id}")


def dog_token(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def canonical_runner_id(race_id: str, box: Any, dog_name: Any) -> str:
    """Sealed alignment key. Never write this value to generated artifacts."""
    return f"{race_id}|box:{int(float(str(box)))}|dog:{dog_token(dog_name)}"


def row_id(race_id: str, box: Any, dog_name: Any = None, *, scope: str = "") -> str:
    """Trainer-safe race-scoped row key derived only from race identity and box."""
    del dog_name, scope
    payload = f"FORM_ONLY_V1|race_box|{race_id}|box:{int(float(str(box)))}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_venue(value: Any) -> str:
    text = re.sub(r"[_\s]+", " ", str(value or "").upper().strip())
    return VENUE_ALIASES.get(text, VENUE_ALIASES.get(text.replace(" ", "_"), text.replace(" ", "_")))


def canonical_grade(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip()).upper()
    if not text:
        return "__MISSING__"
    text = re.sub(r"^(TIER\s*3|BOTTOM\s*UP)\s*-\s*", "", text)
    text = re.sub(r"\s+(HEAT|FINAL)$", "", text)
    compact = re.sub(r"[^A-Z0-9+/]", "", text)
    aliases = {
        "M": "MAIDEN", "MAIDEN": "MAIDEN", "NOV": "NOVICE", "NOVICE": "NOVICE",
        "FFA": "FREE_FOR_ALL", "FREEFORALL": "FREE_FOR_ALL", "I": "INVITATION",
        "INV": "INVITATION", "INVITATION": "INVITATION", "INVITATIONAL": "INVITATION",
        "RW": "RESTRICTED_WIN", "R/W": "RESTRICTED_WIN", "RESTRICTED": "RESTRICTED_WIN",
        "RESTRICTEDWIN": "RESTRICTED_WIN", "OPEN": "OPEN", "MIXED": "MIXED",
        "NG": "NON_GRADED", "NONGRADED": "NON_GRADED", "SE": "SPECIAL_EVENT",
        "S/E": "SPECIAL_EVENT", "SPECIALEVENT": "SPECIAL_EVENT",
    }
    if compact in aliases:
        return aliases[compact]
    grade_match = re.fullmatch(r"(?:GRADE)?([1-8])(?:ST|ND|RD|TH)?(?:GRADE)?", compact)
    if grade_match:
        return f"GRADE_{grade_match.group(1)}"
    mixed_match = re.fullmatch(r"(?:(?:MIXED|GRADE))?([1-8](?:/[1-8]){1,2})", compact)
    if mixed_match:
        return "MIXED_" + mixed_match.group(1).replace("/", "_")
    return re.sub(r"[^A-Z0-9]+", "_", text).strip("_") or "__MISSING__"


def safe_float(value: Any) -> float | None:
    try:
        text = "" if value is None else str(value).strip()
        return float(text) if text else None
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    number = safe_float(value)
    return int(number) if number is not None else None


def fmt_number(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.8f}".rstrip("0").rstrip(".")


def parse_form_blocks_bytes(
    payload: bytes, *, source: str
) -> dict[str, list[dict[str, Any]]]:
    lines = payload.decode("utf-8", errors="replace").splitlines()
    if not lines:
        raise ValueError(f"empty form CSV: {source}")
    delimiter = "|" if lines[0].count("|") > lines[0].count(",") else ","
    blocks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    current = ""
    for raw in csv.DictReader(lines, delimiter=delimiter):
        name = str(raw.get("Dog Name") or "").strip().strip('"')
        if name:
            current = dog_token(re.sub(r"^\s*\d+\.\s*", "", name))
        if current:
            blocks[current].append(dict(raw))
    return blocks


def parse_form_blocks(path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = _read_regular_path_no_follow(path, label="pre-race form card")
    return parse_form_blocks_bytes(payload, source=str(path))


def canonical_roster(
    participants: Iterable[Mapping[str, Any]], *, box_key: str, name_key: str, source: str
) -> list[tuple[int, str]]:
    roster: list[tuple[int, str]] = []
    for participant in participants:
        box = safe_int(participant.get(box_key))
        token = dog_token(participant.get(name_key))
        if box is None or not 1 <= box <= 10 or not token:
            raise ValueError(f"invalid runner identity in {source}")
        roster.append((box, token))
    counts = Counter(roster)
    duplicate_pairs = sorted(pair for pair, count in counts.items() if count > 1)
    box_counts = Counter(box for box, _token in roster)
    token_counts = Counter(token for _box, token in roster)
    if duplicate_pairs or any(count > 1 for count in box_counts.values()) or any(
        count > 1 for count in token_counts.values()
    ):
        raise ValueError(f"duplicate or colliding runner identity in {source}")
    return sorted(roster)


def sidecar_roster(metadata: Mapping[str, Any], *, source: str) -> list[tuple[int, str]]:
    completeness = metadata.get("runner_completeness") or {}
    roster = canonical_roster(
        completeness.get("participants") or [],
        box_key="box_number",
        name_key="dog_name",
        source=source,
    )
    if safe_int(completeness.get("runner_count")) != len(roster):
        raise ValueError(f"sidecar runner_count disagrees with participants in {source}")
    return roster


def parse_card_target_roster_bytes(
    payload: bytes, *, source: str
) -> list[tuple[int, str]]:
    lines = payload.decode("utf-8", errors="replace").splitlines()
    if not lines:
        raise ValueError(f"empty form CSV: {source}")
    delimiter = "|" if lines[0].count("|") > lines[0].count(",") else ","
    participants: list[dict[str, Any]] = []
    for raw in csv.DictReader(lines, delimiter=delimiter):
        name = str(raw.get("Dog Name") or "").strip().strip('"')
        if not name:
            continue
        match = re.fullmatch(r"\s*(\d+)\.\s*(.+?)\s*", name)
        if not match:
            raise ValueError(f"target runner lacks verified box prefix in {source}: {name}")
        participants.append({"box": match.group(1), "name": match.group(2)})
    return canonical_roster(participants, box_key="box", name_key="name", source=source)


def parse_card_target_roster(path: Path) -> list[tuple[int, str]]:
    payload = _read_regular_path_no_follow(path, label="pre-race form card")
    return parse_card_target_roster_bytes(payload, source=str(path))


def verify_card_sidecar_roster(
    csv_path: Path,
    metadata: Mapping[str, Any],
    *,
    race_id: str,
    csv_bytes: bytes | None = None,
) -> list[tuple[int, str]]:
    card = (
        parse_card_target_roster(csv_path)
        if csv_bytes is None
        else parse_card_target_roster_bytes(csv_bytes, source=str(csv_path))
    )
    sidecar = sidecar_roster(metadata, source=f"sidecar:{race_id}")
    if Counter(card) != Counter(sidecar):
        raise ValueError(f"card and COMPLETE sidecar roster mismatch: {race_id}")
    return sidecar


def history_order_value(raw: Mapping[str, Any]) -> tuple[int, str] | None:
    for key in ("RACE_NUMBER", "RACE_NO", "EVENT_NUMBER", "RACE_ORDINAL"):
        value = safe_int(raw.get(key))
        if value is not None:
            return value, key
    for key in ("START_DATETIME", "RACE_TIMESTAMP", "RACE_DATETIME"):
        value = str(raw.get(key) or "").strip()
        if value:
            parsed = parse_timestamp(value)
            return int(parsed.timestamp() * 1_000_000), key
    return None


def accepted_history(
    raw_rows: Iterable[Mapping[str, Any]], target_date: date
) -> tuple[list[dict[str, Any]], list[tuple[str, Mapping[str, Any]]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[tuple[str, Mapping[str, Any]]] = []
    seen: set[tuple[Any, ...]] = set()
    for raw in raw_rows:
        try:
            history_date = date.fromisoformat(str(raw.get("DATE") or "").strip())
        except ValueError:
            rejected.append(("INVALID_HISTORY_DATE", raw))
            continue
        if history_date >= target_date:
            rejected.append(("TARGET_OR_POST_TARGET_HISTORY", raw))
            continue
        normalized = {
            "date": history_date,
            "venue": canonical_venue(raw.get("TRACK")),
            "distance": safe_int(raw.get("DIST")),
            "grade": canonical_grade(raw.get("G")),
            "finish": safe_int(raw.get("PLC")),
            "box": safe_int(raw.get("BOX")),
            "margin": safe_float(raw.get("MGN")),
            "order": history_order_value(raw),
        }
        key = (
            normalized["date"], normalized["venue"], normalized["distance"],
            normalized["grade"], normalized["finish"], normalized["box"], normalized["margin"],
            normalized["order"],
        )
        if key in seen:
            rejected.append(("NORMALIZED_DUPLICATE_HISTORY", raw))
            continue
        seen.add(key)
        accepted.append(normalized)

    by_date: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        by_date[row["date"]].append(row)
    for history_date, rows in by_date.items():
        if len(rows) == 1:
            rows[0]["order_value"] = 0
            continue
        if any(row["order"] is None for row in rows):
            raise ValueError(f"unprovable same-day history ordering: {history_date}")
        order_keys = [(row["order"][1], row["order"][0]) for row in rows]
        if len(set(order_keys)) != len(order_keys):
            raise ValueError(f"duplicate same-day history ordering key: {history_date}")
        if len({source for source, _value in order_keys}) != 1:
            raise ValueError(f"mixed same-day history ordering keys: {history_date}")
        for row in rows:
            row["order_value"] = row["order"][0]

    accepted.sort(key=lambda row: (row["date"], row["order_value"]), reverse=True)
    for row in accepted[HISTORY_CAP:]:
        rejected.append(("HISTORY_CAP_20", row))
    for row in accepted:
        row.pop("order", None)
        row.pop("order_value", None)
        row.pop("box", None)
    return accepted[:HISTORY_CAP], rejected


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finishes = [row["finish"] for row in rows if row["finish"] is not None]
    margins = [row["margin"] for row in rows if row["margin"] is not None]
    return {
        "start_count": len(rows),
        "finish_mean": mean(finishes) if finishes else None,
        "win_rate": mean([1.0 if value == 1 else 0.0 for value in finishes]) if finishes else None,
        "place_rate": mean([1.0 if value <= 3 else 0.0 for value in finishes]) if finishes else None,
        "margin_mean": mean(margins) if margins else None,
    }


def feature_row(
    race_id: str,
    target_date: date,
    target_venue: str,
    target_distance: int | None,
    target_grade: str,
    field_size: int,
    box: int,
    dog_name: str,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    recent3 = history[:3]
    recent5 = history[:5]
    career = aggregate(history)
    r3 = aggregate(recent3)
    r5 = aggregate(recent5)
    venue_rows = [row for row in history if row["venue"] == target_venue]
    distance_rows = [row for row in history if row["distance"] == target_distance and target_distance is not None]
    grade_rows = [row for row in history if row["grade"] == target_grade and target_grade != "__MISSING__"]
    context = {
        "same_venue": aggregate(venue_rows),
        "same_distance": aggregate(distance_rows),
        "same_grade": aggregate(grade_rows),
    }
    newest = history[0]["date"] if history else None
    result: dict[str, Any] = {
        "row_id": row_id(race_id, box, dog_name), "race_id": race_id, "box_number": box,
        "prior_start_count": career["start_count"],
        "days_since_last_start": (target_date - newest).days if newest else "",
        "recent_finish_mean_3": fmt_number(r3["finish_mean"]),
        "recent_finish_mean_5": fmt_number(r5["finish_mean"]),
        "recent_win_rate_5": fmt_number(r5["win_rate"]),
        "recent_place_rate_5": fmt_number(r5["place_rate"]),
        "recent_margin_mean_5": fmt_number(r5["margin_mean"]),
        "career_finish_mean": fmt_number(career["finish_mean"]),
        "career_win_rate": fmt_number(career["win_rate"]),
        "career_place_rate": fmt_number(career["place_rate"]),
        "career_margin_mean": fmt_number(career["margin_mean"]),
        "history_missing": int(not history), "recency_missing": int(newest is None),
        "finish_missing": int(career["finish_mean"] is None),
        "margin_missing": int(career["margin_mean"] is None),
        "target_venue": target_venue, "target_distance_m": target_distance or "",
        "target_grade": target_grade, "target_field_size": field_size,
    }
    for name, values in context.items():
        result[f"{name}_start_count"] = values["start_count"]
        result[f"{name}_finish_mean"] = fmt_number(values["finish_mean"])
        result[f"{name}_win_rate"] = fmt_number(values["win_rate"])
        result[f"{name}_place_rate"] = fmt_number(values["place_rate"])
        result[f"{name}_margin_mean"] = fmt_number(values["margin_mean"])
        result[f"{name}_missing"] = int(not values["start_count"])
    return result


def validate_sidecar(
    csv_path: Path, sidecar_path: Path, *, csv_bytes: bytes | None = None
) -> dict[str, Any]:
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"malformed metadata sidecar: {sidecar_path}")
    if metadata.get("metadata_is_leakage_safe") is not True:
        raise ValueError(f"unsafe metadata sidecar: {sidecar_path}")
    completeness = metadata.get("runner_completeness") or {}
    if not isinstance(completeness, dict):
        raise ValueError(f"malformed runner completeness: {sidecar_path}")
    if completeness.get("status") != "COMPLETE":
        raise ValueError(f"incomplete runner sidecar: {sidecar_path}")
    if csv_bytes is None:
        try:
            csv_bytes = _read_regular_path_no_follow(
                csv_path, label="pre-race form card"
            )
        except FileNotFoundError as exc:
            raise ValueError(f"missing source CSV: {csv_path}") from exc
    actual_sha = hashlib.sha256(csv_bytes).hexdigest()
    if actual_sha != metadata.get("content_sha256"):
        raise ValueError(f"source CSV hash mismatch: {csv_path}")
    if not isinstance(metadata.get("content_sha256"), str) or not isinstance(
        metadata.get("content_length"), int
    ):
        raise ValueError(f"malformed sidecar content identity: {sidecar_path}")
    if len(csv_bytes) != metadata.get("content_length"):
        raise ValueError(f"source CSV byte mismatch: {csv_path}")
    return metadata


def require_consistent(rows: list[Mapping[str, Any]], fields: Iterable[str], *, race_id: str) -> None:
    for field in fields:
        values = {str(row.get(field) or "") for row in rows}
        if len(values) != 1:
            raise ValueError(f"conflicting {field} declarations for {race_id}")


def load_development_sources(
    eligibility_dir: Path, training_dir: Path, reproducibility: Mapping[str, Any]
) -> dict[str, Any]:
    race_path = eligibility_dir / "historical_win_eligibility_races_v1.csv"
    runner_path = eligibility_dir / "historical_win_eligibility_runners_v1.csv"
    provenance_path = eligibility_dir / "historical_win_tier_a_race_provenance_v1.json"
    training_path = training_dir / "thedogs_training_rows_v1.csv"
    expected_roles = {
        "eligibility_races": race_path,
        "eligibility_runners": runner_path,
        "tier_a_provenance": provenance_path,
        "training_rows": training_path,
    }
    development_contract = (reproducibility.get("trusted_inputs") or {}).get("development") or {}
    expected_files = development_contract.get("files") or {}
    if set(expected_files) != set(expected_roles):
        raise ValueError("development reproducibility file roles are incomplete")
    top_input_records: list[dict[str, Any]] = []
    for role, path in expected_roles.items():
        expected = expected_files[role]
        if Path(expected["path"]).resolve() != path.resolve():
            raise ValueError(f"development input path mismatch for {role}")
        record = verify_file_record(
            path,
            expected_sha256=expected["sha256"],
            expected_bytes=expected.get("bytes"),
            require_expected_bytes=True,
        )
        top_input_records.append({"role": role, **record})
    top_input_by_role = {record["role"]: record for record in top_input_records}

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))["races"]
    tier_a_runners: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with runner_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["strongest_tier"] == "A":
                tier_a_runners[row["race_id"]].append({
                    "box": int(row["box_number"]), "dog_name": row["dog_name"],
                    "source_runner_id": row["runner_id"],
                })

    selected_published: set[str] = set()
    with race_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["used_for_training"] == "1":
                selected_published.add(row["race_id"])

    published_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with training_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["race_id"] not in selected_published:
                continue
            published_rows[row["race_id"]].append(row)

    published_consistency_fields = (
        "source_csv_path", "source_csv_sha256", "race_date", "race_timestamp_utc",
        "target_distance", "target_grade", "listed_participants_count", "active_field_size",
        "scratched_runner_count", "reserve_runner_count", "has_scratch_or_reserve",
    )
    for race_id, rows in published_rows.items():
        require_consistent(rows, published_consistency_fields, race_id=race_id)
        if any(str(row.get("runner_status") or "").lower() != "active" for row in rows):
            raise ValueError(f"published active roster contains non-active runner: {race_id}")

    if EXCLUDED_PUBLISHED_RACE_ID in published_rows:
        raise ValueError("published exclusion was incorrectly selected")

    candidate_ids = sorted(set(provenance).union(published_rows))
    candidate_runners: dict[str, list[dict[str, Any]]] = {}
    source_options: dict[str, list[dict[str, Any]]] = defaultdict(list)
    trusted_source_records: list[dict[str, Any]] = []
    semantic_records: list[dict[str, Any]] = []
    retained_card_bytes_by_path: dict[str, bytes] = {}

    def retained_card_bytes(path: Path) -> bytes:
        resolved = str(path.resolve())
        if resolved not in retained_card_bytes_by_path:
            retained_card_bytes_by_path[resolved] = _read_regular_path_no_follow(
                path, label="pre-race form card"
            )
        return retained_card_bytes_by_path[resolved]

    for race_id in candidate_ids:
        if race_id in tier_a_runners:
            candidate_runners[race_id] = sorted(tier_a_runners[race_id], key=lambda row: (row["box"], dog_token(row["dog_name"])))
            canonical_roster(
                candidate_runners[race_id], box_key="box", name_key="dog_name", source=f"tier-a:{race_id}"
            )
            item = provenance[race_id]
            csv_path = Path(item["source_csv_path"])
            sidecar_path = Path(item["sidecar_path"])
            csv_bytes = retained_card_bytes(csv_path)
            metadata = validate_sidecar(csv_path, sidecar_path, csv_bytes=csv_bytes)
            csv_record = verify_retained_file_record(
                csv_path,
                csv_bytes,
                expected_sha256=item["source_csv_sha256"],
            )
            sidecar_record = verify_file_record(sidecar_path, expected_sha256=item["sidecar_sha256"])
            canonical_jump = parse_timestamp(item["jump_timestamp"])
            semantic_records.append(validate_sidecar_semantics(
                race_id,
                csv_path,
                sidecar_path,
                metadata,
                expected_jump=canonical_jump,
                expected_roster=canonical_roster(
                    candidate_runners[race_id],
                    box_key="box",
                    name_key="dog_name",
                    source=f"tier-a:{race_id}",
                ),
                csv_bytes=csv_bytes,
            ))
            trusted_source_records.extend([
                {"role": "development_card", **csv_record},
                {"role": "development_sidecar", **sidecar_record},
            ])
            label_records: list[dict[str, Any]] = []
            for path_text, digest in zip(
                (item["official_race_artifact_path"], item["official_runner_artifact_path"]),
                (item["official_race_artifact_sha256"], item["official_runner_artifact_sha256"]),
                strict=True,
            ):
                record = verify_file_record(Path(path_text), expected_sha256=digest)
                label_records.append(record)
                trusted_source_records.append({"role": "development_label", **record})
            source_options[race_id].append({
                "source_class": "OFFICIAL_RACE_PAGE_TIER_A",
                "precedence": 0,
                "csv_path": csv_path,
                "csv_bytes": csv_bytes,
                "sidecar_path": sidecar_path,
                "csv_sha256": csv_record["sha256"],
                "sidecar_sha256": sidecar_record["sha256"],
                "capture": capture_timestamp(metadata),
                "jump": canonical_jump,
                "metadata": metadata,
                "label_provenance_class": "OFFICIAL_RACE_PAGE_TIER_A",
                "label_source_paths": [record["path"] for record in label_records],
                "label_source_sha256": [record["sha256"] for record in label_records],
                "label_urls": item.get("official_urls") or [],
                "active_roster_evidence": None,
            })
        if race_id in published_rows:
            first = published_rows[race_id][0]
            csv_path = Path(first["source_csv_path"])
            sidecar_path = Path(str(csv_path) + ".metadata.json")
            csv_bytes = retained_card_bytes(csv_path)
            metadata = validate_sidecar(csv_path, sidecar_path, csv_bytes=csv_bytes)
            csv_record = verify_retained_file_record(
                csv_path,
                csv_bytes,
                expected_sha256=first["source_csv_sha256"],
            )
            sidecar_record = file_record(sidecar_path)
            trusted_source_records.extend([
                {"role": "development_card", **csv_record},
                {"role": "development_sidecar", **sidecar_record},
                {
                    "role": "development_label",
                    **{key: top_input_by_role["training_rows"][key] for key in ("path", "sha256", "bytes")},
                },
            ])
            option = {
                "source_class": "THEDOGS_PUBLISHED_HISTORY_NOT_TIER_A",
                "precedence": 1,
                "csv_path": csv_path,
                "csv_bytes": csv_bytes,
                "sidecar_path": sidecar_path,
                "csv_sha256": csv_record["sha256"],
                "sidecar_sha256": sidecar_record["sha256"],
                "capture": capture_timestamp(metadata),
                "jump": parse_timestamp(first["race_timestamp_utc"]),
                "metadata": metadata,
                "label_provenance_class": "THEDOGS_PUBLISHED_HISTORY_NOT_TIER_A",
                "label_source_paths": [str(training_path)],
                "label_source_sha256": [sha256_path(training_path)],
                "label_urls": [first["odds_url"]] if first.get("odds_url") else [],
                "active_roster_evidence": {
                    "listed_participants_count": int(first["listed_participants_count"]),
                    "active_field_size": int(first["active_field_size"]),
                    "scratched_runner_count": int(first["scratched_runner_count"]),
                    "reserve_runner_count": int(first["reserve_runner_count"]),
                    "has_scratch_or_reserve": int(first["has_scratch_or_reserve"]),
                    "path": str(training_path.resolve()),
                    "sha256": top_input_by_role["training_rows"]["sha256"],
                },
            }
            source_options[race_id].append(option)
            published_runner_list = sorted([
                {"box": int(row["box_number"]), "dog_name": row["csv_dog_name"], "source_runner_id": row["runner_id"]}
                for row in published_rows[race_id]
            ], key=lambda row: (row["box"], dog_token(row["dog_name"])))
            canonical_roster(
                published_runner_list,
                box_key="box",
                name_key="dog_name",
                source=f"published-active:{race_id}",
            )
            semantic_records.append(validate_sidecar_semantics(
                race_id,
                csv_path,
                sidecar_path,
                metadata,
                expected_jump=parse_timestamp(first["race_timestamp_utc"]),
                expected_roster=canonical_roster(
                    published_runner_list,
                    box_key="box",
                    name_key="dog_name",
                    source=f"published-active:{race_id}",
                ),
                expected_url=first.get("odds_url"),
                enforce_jump_equality=False,
                allow_roster_superset=True,
                csv_bytes=csv_bytes,
            ))
            if race_id not in candidate_runners:
                candidate_runners[race_id] = published_runner_list
            else:
                left = [(row["box"], dog_token(row["dog_name"])) for row in candidate_runners[race_id]]
                right = [(row["box"], dog_token(row["dog_name"])) for row in published_runner_list]
                if left != right:
                    raise ValueError(f"overlap runner identity mismatch: {race_id}")

    source_digest = source_set_digest(
        trusted_source_records, reject_duplicate_declarations=False
    )
    if source_digest != development_contract.get("authoritative_source_set_sha256"):
        raise ValueError(f"development source-set binding mismatch: {source_digest}")
    source_record_count = len({(row["role"], row["path"]) for row in trusted_source_records})
    if source_record_count != int(development_contract.get("authoritative_source_record_count", -1)):
        raise ValueError(f"development source-set count mismatch: {source_record_count}")
    return {
        "candidate_ids": candidate_ids,
        "candidate_runners": candidate_runners,
        "source_options": source_options,
        "provenance": provenance,
        "published_rows": published_rows,
        "top_input_records": top_input_records,
        "trusted_source_records": trusted_source_records,
        "trusted_source_set_sha256": source_digest,
        "training_path": training_path,
        "construction_contract_sha256": reproducibility["construction_contract_sha256"],
        "semantic_trust_root_sha256": canonical_digest(sorted(
            semantic_records, key=lambda row: (row["race_id"], row["source_path"])
        )),
        "semantic_records": semantic_records,
        "retained_card_bytes_by_path": retained_card_bytes_by_path,
    }


def load_diagnostic_sources(
    authoritative: Mapping[str, Any], reproducibility: Mapping[str, Any]
) -> dict[str, Any]:
    """Load and bind shadow sources only for the optional diagnostic phase."""
    overlap_ids = sorted(
        set(authoritative["provenance"]).intersection(authoritative["published_rows"])
    )
    retained_by_path = authoritative.get("retained_card_bytes_by_path")
    if not isinstance(retained_by_path, Mapping):
        raise ValueError("diagnostic phase has no retained development card bytes")
    diagnostic_card_bytes_by_race: dict[str, bytes] = {}
    for race_id in overlap_ids:
        item = authoritative["provenance"][race_id]
        path = Path(item["source_csv_path"])
        payload = retained_by_path.get(str(path.resolve()))
        if not isinstance(payload, bytes):
            raise ValueError(f"diagnostic phase lacks retained card bytes: {race_id}")
        if hashlib.sha256(payload).hexdigest() != item["source_csv_sha256"]:
            raise ValueError(f"diagnostic retained card hash mismatch: {race_id}")
        diagnostic_card_bytes_by_race[race_id] = payload
    shadow_source_by_race: dict[str, dict[str, Any]] = {}
    diagnostic_source_records: list[dict[str, Any]] = []
    for race_id in overlap_ids:
        item = authoritative["provenance"][race_id]
        paths = item.get("feature_source_paths") or []
        digests = item.get("feature_source_sha256") or []
        if len(paths) != 1 or len(digests) != 1:
            raise ValueError(f"overlap race requires one justified shadow source: {race_id}")
        path = Path(paths[0])
        source_bytes = _read_regular_path_no_follow(
            path, label="shadow diagnostic source"
        )
        actual_hash = hashlib.sha256(source_bytes).hexdigest()
        if actual_hash != digests[0]:
            raise ValueError(f"shadow diagnostic source hash mismatch: {path}")
        record = {"path": str(path), "sha256": actual_hash, "bytes": len(source_bytes)}
        shadow_source_by_race[race_id] = record
        diagnostic_source_records.append({"role": "shadow_reconciliation_source", **record})

    diagnostic_contract = (reproducibility.get("trusted_inputs") or {}).get("diagnostic")
    if not isinstance(diagnostic_contract, dict):
        raise ValueError("reproducibility diagnostic trust domain is malformed")
    diagnostic_digest = source_set_digest(
        diagnostic_source_records, reject_duplicate_declarations=False
    )
    if diagnostic_digest != diagnostic_contract.get("source_set_sha256"):
        raise ValueError(f"diagnostic source-set binding mismatch: {diagnostic_digest}")
    diagnostic_count = len({
        (row["role"], row["path"]) for row in diagnostic_source_records
    })
    if diagnostic_count != int(diagnostic_contract.get("source_record_count", -1)):
        raise ValueError("diagnostic source-set count mismatch")
    return {
        **authoritative,
        "diagnostic_source_records": diagnostic_source_records,
        "diagnostic_source_set_sha256": diagnostic_digest,
        "shadow_source_by_race": shadow_source_by_race,
        "diagnostic_card_bytes_by_race": diagnostic_card_bytes_by_race,
        "diagnostic_contract_sha256": canonical_digest(diagnostic_contract),
    }


def _csv_rows_from_bytes(payload: bytes, *, label: str) -> list[dict[str, str]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"malformed UTF-8 in {label}") from exc
    return [dict(row) for row in csv.DictReader(io.StringIO(text))]


def _diagnostic_context_from_authoritative_packet(
    authoritative_payloads: Mapping[str, Mapping[str, bytes]],
    reproducibility: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct diagnostic inputs from the completed authoritative packet."""
    sealed = authoritative_payloads["sealed_validation"]
    source_rows = _csv_rows_from_bytes(
        sealed["development_source_inventory.csv"],
        label="sealed development source inventory",
    )
    source_classes: dict[str, set[str]] = defaultdict(set)
    tier_a_cards: dict[str, dict[str, str]] = {}
    published_cards: dict[str, dict[str, str]] = {}
    for row in source_rows:
        race_id = row.get("race_id") or ""
        source_class = row.get("source_class") or ""
        source_classes[race_id].add(source_class)
        if row.get("role") != "raw_pre_race_card":
            continue
        target = None
        label = ""
        if source_class == "OFFICIAL_RACE_PAGE_TIER_A":
            target = tier_a_cards
            label = "tier-A"
        elif source_class == "THEDOGS_PUBLISHED_HISTORY_NOT_TIER_A":
            target = published_cards
            label = "published-history"
        if target is not None:
            if race_id in target and target[race_id] != row:
                raise ValueError(f"ambiguous sealed {label} card declaration: {race_id}")
            target[race_id] = row
    overlap_ids = sorted(
        race_id
        for race_id, classes in source_classes.items()
        if {
            "OFFICIAL_RACE_PAGE_TIER_A",
            "THEDOGS_PUBLISHED_HISTORY_NOT_TIER_A",
        }.issubset(classes)
    )

    development_contract = (
        (reproducibility.get("trusted_inputs") or {}).get("development") or {}
    )
    development_files = development_contract.get("files") or {}
    training_declaration = development_files.get("training_rows")
    if not isinstance(training_declaration, dict):
        raise ValueError("diagnostic phase has no bound training-row declaration")
    training_path = Path(str(training_declaration.get("path") or ""))
    training_bytes = _read_regular_path_no_follow(
        training_path, label="diagnostic training rows"
    )
    if len(training_bytes) != int(training_declaration.get("bytes", -1)):
        raise ValueError("diagnostic training-row byte length mismatch")
    if hashlib.sha256(training_bytes).hexdigest() != training_declaration.get("sha256"):
        raise ValueError("diagnostic training-row sha256 mismatch")

    published_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    published_keys: set[tuple[str, int, str]] = set()
    for row in _csv_rows_from_bytes(training_bytes, label="diagnostic training rows"):
        race_id = row.get("race_id") or ""
        if race_id not in overlap_ids:
            continue
        box = int(row.get("box_number") or 0)
        dog_name = row.get("csv_dog_name") or ""
        key = (race_id, box, dog_token(dog_name))
        if box <= 0 or not dog_name or key in published_keys:
            raise ValueError(f"duplicate or malformed diagnostic training row: {key}")
        published_keys.add(key)
        published_card = published_cards.get(race_id)
        tier_a_card = tier_a_cards.get(race_id)
        if published_card is None or tier_a_card is None:
            raise ValueError(f"missing sealed overlap card declaration: {race_id}")
        if (
            row.get("source_csv_path") != published_card.get("path")
            or row.get("source_csv_sha256") != published_card.get("sha256")
            or published_card.get("sha256") != tier_a_card.get("sha256")
        ):
            raise ValueError(f"diagnostic overlap card binding mismatch: {race_id}")
        published_rows[race_id].append(dict(row))
    if set(published_rows) != set(overlap_ids):
        raise ValueError("bound training rows are missing diagnostic overlap races")

    provenance_declaration = development_files.get("tier_a_provenance")
    if not isinstance(provenance_declaration, dict):
        raise ValueError("diagnostic phase has no bound tier-A provenance declaration")
    provenance_path = Path(str(provenance_declaration.get("path") or ""))
    provenance_bytes = _read_regular_path_no_follow(
        provenance_path, label="diagnostic tier-A provenance"
    )
    if len(provenance_bytes) != int(provenance_declaration.get("bytes", -1)):
        raise ValueError("diagnostic tier-A provenance byte length mismatch")
    if hashlib.sha256(provenance_bytes).hexdigest() != provenance_declaration.get("sha256"):
        raise ValueError("diagnostic tier-A provenance sha256 mismatch")
    try:
        all_provenance = json.loads(provenance_bytes.decode("utf-8"))["races"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("malformed diagnostic tier-A provenance") from exc

    provenance: dict[str, dict[str, Any]] = {}
    retained_card_bytes_by_path: dict[str, bytes] = {}
    for race_id in overlap_ids:
        item = all_provenance.get(race_id)
        card = tier_a_cards[race_id]
        if not isinstance(item, dict):
            raise ValueError(f"diagnostic provenance missing overlap race: {race_id}")
        card_path = Path(card["path"])
        resolved_card_path = str(card_path.resolve())
        if resolved_card_path not in retained_card_bytes_by_path:
            card_bytes = _read_regular_path_no_follow(
                card_path, label="diagnostic pre-race form card"
            )
            verify_retained_file_record(
                card_path,
                card_bytes,
                expected_sha256=card["sha256"],
                expected_bytes=card["bytes"],
                require_expected_bytes=True,
            )
            retained_card_bytes_by_path[resolved_card_path] = card_bytes
        provenance[race_id] = {
            **item,
            "source_csv_path": card["path"],
            "source_csv_sha256": card["sha256"],
        }
    return load_diagnostic_sources(
        {
            "provenance": provenance,
            "published_rows": published_rows,
            "retained_card_bytes_by_path": retained_card_bytes_by_path,
        },
        reproducibility,
    )


def select_development_sources(loaded: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    selected: dict[str, dict[str, Any]] = {}
    excluded: dict[str, str] = {}
    for race_id in loaded["candidate_ids"]:
        eligible = [
            option for option in loaded["source_options"][race_id]
            if option["capture"] <= option["jump"] - timedelta(minutes=60)
        ]
        if not eligible:
            excluded[race_id] = "CARD_NOT_AVAILABLE_BY_T60"
            continue
        winning_precedence = min(option["precedence"] for option in eligible)
        precedence_winners = [
            option for option in eligible if option["precedence"] == winning_precedence
        ]
        winning_capture = max(option["capture"] for option in precedence_winners)
        time_winners = [
            option for option in precedence_winners if option["capture"] == winning_capture
        ]
        identities: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for option in time_winners:
            identity = canonical_digest({
                "card_sha256": option["csv_sha256"],
                "sidecar_sha256": option["sidecar_sha256"],
                "capture": option["capture"].isoformat(),
                "jump": option["jump"].isoformat(),
                "roster": canonical_roster(
                    (option["metadata"].get("runner_completeness") or {}).get("participants") or [],
                    box_key="box_number",
                    name_key="dog_name",
                    source=f"selection:{race_id}",
                ),
            })
            identities[identity].append(option)
        if len(identities) != 1:
            raise ValueError(f"ambiguous equal-precedence development sources: {race_id}")
        canonical_identity, aliases = next(iter(identities.items()))
        representative = min(aliases, key=lambda option: str(option["csv_path"].resolve()))
        representative["canonical_source_identity"] = canonical_identity
        representative["byte_identical_alias_count"] = len(aliases)
        selected[race_id] = representative
    return selected, excluded


def target_metadata(option: Mapping[str, Any], race_id: str) -> tuple[str, int | None, str, int]:
    metadata = option["metadata"]
    info = metadata.get("race_info") or {}
    venue = canonical_venue(info.get("venue") or race_id.split(" - ")[1])
    distance = safe_int(metadata.get("target_distance") or info.get("distance"))
    grade = canonical_grade(metadata.get("target_grade") or info.get("grade"))
    field_size = int((metadata.get("runner_completeness") or {}).get("runner_count") or 0)
    return venue, distance, grade, field_size


def reconcile_development_roster(
    option: Mapping[str, Any], runners: list[dict[str, Any]], race_id: str
) -> tuple[list[tuple[int, str]], list[dict[str, Any]]]:
    sidecar = verify_card_sidecar_roster(
        option["csv_path"],
        option["metadata"],
        race_id=race_id,
        csv_bytes=option.get("csv_bytes"),
    )
    active = canonical_roster(
        runners, box_key="box", name_key="dog_name", source=f"active-label-roster:{race_id}"
    )
    sidecar_counter = Counter(sidecar)
    active_counter = Counter(active)
    missing_from_sidecar = list((active_counter - sidecar_counter).elements())
    sidecar_only = list((sidecar_counter - active_counter).elements())
    if missing_from_sidecar:
        raise ValueError(f"active label runner absent from COMPLETE sidecar: {race_id}")
    if not sidecar_only:
        return active, []

    evidence = option.get("active_roster_evidence")
    if option["source_class"] != "THEDOGS_PUBLISHED_HISTORY_NOT_TIER_A" or not evidence:
        raise ValueError(f"sidecar-only runner has no immutable exclusion evidence: {race_id}")
    if (
        evidence["listed_participants_count"] != len(sidecar)
        or evidence["active_field_size"] != len(active)
        or evidence["listed_participants_count"] - evidence["active_field_size"] != len(sidecar_only)
        or evidence["has_scratch_or_reserve"] != 1
        or evidence["scratched_runner_count"] + evidence["reserve_runner_count"] < len(sidecar_only)
    ):
        raise ValueError(f"scratch/reserve evidence does not explain roster difference: {race_id}")
    exclusions = [
        {
            "entity_type": "runner",
            "entity_id": row_id(race_id, box, token, scope=DEVELOPMENT_SCOPE),
            "race_id": race_id,
            "reason": "HASH_BOUND_PUBLISHED_ACTIVE_ROSTER_EXCLUSION",
            "history_date": "",
        }
        for box, token in sorted(sidecar_only)
    ]
    return active, exclusions


def build_development_packet(
    loaded: Mapping[str, Any],
    output_domain: _OutputDomain,
    sealed_domain: _OutputDomain,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    selected, excluded = select_development_sources(loaded)
    candidate_runners = loaded["candidate_runners"]
    race_rows: list[dict[str, Any]] = []
    runner_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    leakage = Counter()

    for race_id in loaded["candidate_ids"]:
        if race_id not in selected:
            exclusion_rows.append({"entity_type": "race", "entity_id": race_id, "race_id": race_id, "reason": excluded[race_id]})
            for runner in candidate_runners[race_id]:
                exclusion_rows.append({
                    "entity_type": "runner",
                    "entity_id": row_id(race_id, runner["box"], runner["dog_name"], scope=DEVELOPMENT_SCOPE),
                    "race_id": race_id, "reason": excluded[race_id],
                })
        for option in loaded["source_options"][race_id]:
            status = "INCLUDED" if selected.get(race_id) is option else "CANDIDATE_NOT_SELECTED"
            lead = (option["jump"] - option["capture"]).total_seconds() / 60.0
            for role, path, digest in (
                ("raw_pre_race_card", option["csv_path"], option["csv_sha256"]),
                ("raw_pre_race_sidecar", option["sidecar_path"], option["sidecar_sha256"]),
            ):
                source_bytes = (
                    len(option["csv_bytes"])
                    if role == "raw_pre_race_card"
                    else path.stat().st_size
                )
                source_rows.append({
                    "race_id": race_id, "selection_status": status, "source_class": option["source_class"],
                    "role": role, "path": str(path.resolve()), "sha256": digest,
                    "bytes": source_bytes, "capture_timestamp": option["capture"].isoformat(),
                    "jump_timestamp": option["jump"].isoformat(), "lead_minutes": fmt_number(lead),
                })
            for path_text, digest in zip(
                option["label_source_paths"], option["label_source_sha256"], strict=True
            ):
                path = Path(path_text)
                source_rows.append({
                    "race_id": race_id, "selection_status": status, "source_class": option["source_class"],
                    "role": "label_provenance_only_not_opened_by_builder", "path": str(path.resolve()),
                    "sha256": digest, "bytes": path.stat().st_size, "capture_timestamp": "",
                    "jump_timestamp": option["jump"].isoformat(), "lead_minutes": "",
                })
    for race_id in sorted(selected):
        option = selected[race_id]
        race_date = date.fromisoformat(race_id.rsplit(" - ", 1)[-1])
        if race_date > DEVELOPMENT_END:
            raise ValueError(f"development race after cutoff: {race_id}")
        venue, distance, grade, _sidecar_field_size = target_metadata(option, race_id)
        runners = candidate_runners[race_id]
        _active_roster, roster_exclusions = reconcile_development_roster(option, runners, race_id)
        exclusion_rows.extend(roster_exclusions)
        field_size = len(runners)
        blocks = parse_form_blocks_bytes(
            option["csv_bytes"], source=str(option["csv_path"])
        )
        race_rows.append({
            "race_id": race_id, "race_date": race_date.isoformat(), "target_venue": venue,
            "race_number": int(re.search(r"Race (\d+)", race_id).group(1)),
            "target_distance_m": distance or "", "target_grade": grade, "field_size": field_size,
            "card_capture_timestamp": option["capture"].isoformat(), "jump_timestamp": option["jump"].isoformat(),
            "card_lead_minutes": fmt_number((option["jump"] - option["capture"]).total_seconds() / 60),
            "card_source_class": option["source_class"],
            "canonical_source_identity": option["canonical_source_identity"],
            "label_provenance_class": option["label_provenance_class"],
            "label_value_included": 0,
        })
        for runner in runners:
            token = dog_token(runner["dog_name"])
            if token not in blocks:
                raise ValueError(f"runner absent from raw card: {race_id} {token}")
            history, rejected = accepted_history(blocks[token], race_date)
            opaque = row_id(race_id, runner["box"], runner["dog_name"], scope=DEVELOPMENT_SCOPE)
            alignment_rows.append({
                "split": DEVELOPMENT_SCOPE,
                "race_id": race_id,
                "box_number": runner["box"],
                "row_id": opaque,
                "dog_name_token": token,
                "canonical_runner_id": canonical_runner_id(
                    race_id, runner["box"], runner["dog_name"]
                ),
            })
            runner_rows.append({
                "row_id": opaque, "race_id": race_id, "box_number": runner["box"],
                "label_provenance_class": option["label_provenance_class"],
                "label_value_included": 0,
            })
            feature_rows.append(feature_row(
                race_id, race_date, venue, distance, grade, field_size,
                runner["box"], runner["dog_name"], history,
            ))
            for reason, raw in rejected:
                if reason == "TARGET_OR_POST_TARGET_HISTORY":
                    leakage["rejected_target_or_post_target_history"] += 1
                exclusion_rows.append({
                    "entity_type": "history_row", "entity_id": opaque, "race_id": race_id,
                    "reason": reason, "history_date": str(raw.get("DATE") or raw.get("date") or ""),
                })

    race_rows.sort(key=lambda row: row["race_id"])
    runner_rows.sort(key=lambda row: (row["race_id"], row["box_number"], row["row_id"]))
    feature_rows.sort(key=lambda row: (row["race_id"], row["box_number"], row["row_id"]))
    exclusion_rows.sort(key=lambda row: (row["race_id"], row["entity_type"], row["entity_id"], row["reason"]))
    source_rows.sort(key=lambda row: (row["race_id"], row["source_class"], row["role"], row["path"]))

    stable_csv(output_domain, "development_races.csv", list(race_rows[0]), race_rows)
    stable_csv(output_domain, "development_runners.csv", list(runner_rows[0]), runner_rows)
    stable_csv(output_domain, "development_features.csv", list(feature_rows[0]), feature_rows)
    exclusion_fields = [
        "entity_type", "entity_id", "race_id", "reason", "history_date",
    ]
    stable_csv(output_domain, "development_exclusions.csv", exclusion_fields, exclusion_rows)
    stable_csv(
        sealed_domain,
        "development_source_inventory.csv",
        list(source_rows[0]),
        source_rows,
    )
    stable_csv(
        sealed_domain,
        "development_runner_alignment.csv",
        list(alignment_rows[0]),
        alignment_rows,
    )
    source_inventory_record = sealed_domain.file_record(
        "development_source_inventory.csv"
    )

    feature_columns = list(feature_rows[0])
    forbidden_columns = sorted(set(feature_columns).intersection(FORBIDDEN_FEATURE_TOKENS))
    if forbidden_columns:
        raise ValueError(f"forbidden feature columns: {forbidden_columns}")
    summary = {
        "candidate_race_count": len(loaded["candidate_ids"]),
        "candidate_runner_count": sum(len(candidate_runners[race_id]) for race_id in loaded["candidate_ids"]),
        "included_race_count": len(race_rows), "included_runner_count": len(runner_rows),
        "excluded_race_count": len(excluded),
        "excluded_runner_count": sum(len(candidate_runners[race_id]) for race_id in excluded),
        "sidecar_only_runner_exclusion_count": sum(
            row["reason"] == "HASH_BOUND_PUBLISHED_ACTIVE_ROSTER_EXCLUSION" for row in exclusion_rows
        ),
        "accepted_target_or_post_target_history_count": 0,
        "rejected_target_or_post_target_history_count": leakage["rejected_target_or_post_target_history"],
        "outcome_feature_count": 0, "market_feature_count": 0, "dog_identity_feature_count": 0,
        "feature_columns": feature_columns,
    }
    stable_json(output_domain, "development_manifest.json", {
        "schema_version": "form_only_v1_development_manifest_v2",
        "status": "ACQUISITION_ONLY_NO_MODEL_FIT",
        "development_end": DEVELOPMENT_END.isoformat(),
        "label_values_included": False,
        "card_requirement": "capture_timestamp <= canonical_jump_timestamp - 60 minutes",
        "source_precedence": ["OFFICIAL_RACE_PAGE_TIER_A", "THEDOGS_PUBLISHED_HISTORY_NOT_TIER_A"],
        "summary": summary,
        "construction_contract_sha256": loaded["construction_contract_sha256"],
        "authoritative_trust_root": {
            "record_count": len({
                (row["role"], row["path"]) for row in loaded["trusted_source_records"]
            }),
            "aggregate_sha256": loaded["trusted_source_set_sha256"],
            "semantic_sha256": loaded["semantic_trust_root_sha256"],
            "sealed_inventory_sha256": source_inventory_record["sha256"],
        },
    })
    return summary, selected


def load_shadow_feature_rows(
    sources: Mapping[str, Mapping[str, Any]]
) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for expected_race_id, record in sorted(sources.items()):
        path = Path(record["path"])
        source_bytes = _read_regular_path_no_follow(
            path, label="shadow diagnostic source"
        )
        if len(source_bytes) != int(record.get("bytes", -1)):
            raise ValueError(f"shadow diagnostic source byte mismatch: {path}")
        if hashlib.sha256(source_bytes).hexdigest() != str(record["sha256"]):
            raise ValueError(f"shadow diagnostic source hash mismatch: {path}")
        try:
            payload = json.loads(source_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"malformed shadow diagnostic source: {path}") from exc
        if not isinstance(payload, list):
            raise ValueError(f"unexpected shadow feature payload: {path}")
        matched = 0
        for row in payload:
            if row.get("race_id") != expected_race_id:
                continue
            token = dog_token(row.get("dog_name"))
            if not token:
                raise ValueError(f"shadow feature row has no runner identity: {path}")
            key = (expected_race_id, token)
            if key in rows:
                kind = "conflicting" if rows[key] != row else "duplicate"
                raise ValueError(f"{kind} shadow overlap key before insertion: {key}")
            rows[key] = {**row, "_bound_source_path": str(path)}
            matched += 1
        if not matched:
            raise ValueError(f"bound shadow source has no rows for {expected_race_id}: {path}")
    return rows


def mismatch_cause(
    family: str,
    *,
    differs: bool,
    shadow: Mapping[str, Any],
    shadow_value: Any,
    canonical_value: Any,
    history: list[dict[str, Any]],
    rejected: list[tuple[str, Mapping[str, Any]]],
    target_date: date,
) -> str:
    if not differs:
        return "MATCH"
    reason_counts = Counter(reason for reason, _row in rejected)
    if family == "history":
        if shadow_value == 0 and canonical_value > 0 and shadow.get("days_since_last_start") is None:
            return "SHADOW_EMPTY_HISTORY_WHILE_RAW_PRIOR_ROWS_EXIST"
        if shadow_value == canonical_value + reason_counts["NORMALIZED_DUPLICATE_HISTORY"]:
            return "SHADOW_COUNTED_NORMALIZED_DUPLICATES"
        if shadow_value == canonical_value + reason_counts["HISTORY_CAP_20"]:
            return "CANONICAL_VERIFIED_HISTORY_CAP_20"
        if shadow_value == canonical_value + reason_counts["TARGET_OR_POST_TARGET_HISTORY"]:
            return "SHADOW_INCLUDED_TARGET_OR_POST_TARGET_ROWS"
    elif family == "recency":
        if shadow_value is None and canonical_value is not None and safe_int(shadow.get("prior_start_count")) == 0:
            return "SHADOW_EMPTY_HISTORY_HAS_NO_RECENCY"
        if shadow_value is not None:
            older_recencies = {(target_date - row["date"]).days for row in history[1:]}
            if shadow_value in older_recencies:
                return "SHADOW_SELECTED_NONLATEST_COMPARED_RAW_ROW"
            rejected_dates = set()
            for reason, row in rejected:
                if reason != "TARGET_OR_POST_TARGET_HISTORY":
                    continue
                try:
                    rejected_dates.add((target_date - date.fromisoformat(str(row.get("DATE")))).days)
                except (TypeError, ValueError):
                    continue
            if shadow_value in rejected_dates:
                return "SHADOW_RECENCY_USED_TARGET_OR_POST_TARGET_ROW"
    elif family == "grade":
        if shadow.get("target_grade_missing") in (1, True) and canonical_value != "__MISSING__":
            return "SHADOW_TARGET_GRADE_MISSING_WHILE_CARD_GRADE_PRESENT"
    return f"UNEXPLAINED_{family.upper()}_DIFFERENCE"


def build_overlap_reconciliation(
    loaded: Mapping[str, Any], output_domain: _OutputDomain
) -> dict[str, Any]:
    overlap_ids = sorted(set(loaded["provenance"]).intersection(loaded["published_rows"]))
    shadow_rows = load_shadow_feature_rows(loaded["shadow_source_by_race"])
    expected_keys = {
        (race_id, dog_token(row["csv_dog_name"]))
        for race_id in overlap_ids
        for row in loaded["published_rows"][race_id]
    }
    if set(shadow_rows) != expected_keys:
        missing = len(expected_keys - set(shadow_rows))
        extra = len(set(shadow_rows) - expected_keys)
        raise ValueError(f"shadow overlap key-set mismatch: missing={missing} extra={extra}")
    rows: list[dict[str, Any]] = []
    causes = Counter()
    raw_identical_races = 0
    for race_id in overlap_ids:
        tier_a = loaded["provenance"][race_id]
        published = loaded["published_rows"][race_id]
        if tier_a["source_csv_sha256"] != published[0]["source_csv_sha256"]:
            raise ValueError(f"overlap raw card hash mismatch: {race_id}")
        raw_identical_races += 1
        blocks = parse_form_blocks_bytes(
            loaded["diagnostic_card_bytes_by_race"][race_id],
            source=str(tier_a["source_csv_path"]),
        )
        target_date = date.fromisoformat(published[0]["race_date"])
        for published_row in sorted(published, key=lambda row: (int(row["box_number"]), dog_token(row["csv_dog_name"]))):
            token = dog_token(published_row["csv_dog_name"])
            shadow = shadow_rows[(race_id, token)]
            history, rejected = accepted_history(blocks[token], target_date)
            canonical_history_count = len(history)
            canonical_recency = (target_date - history[0]["date"]).days if history else None
            canonical_target_grade = canonical_grade(published_row["target_grade"])
            shadow_history = safe_int(shadow.get("prior_start_count"))
            shadow_recency = safe_int(shadow.get("days_since_last_start"))
            shadow_grade_raw = str(shadow.get("target_grade_normalized") or "")
            shadow_grade = canonical_grade(shadow_grade_raw)
            history_diff = shadow_history != canonical_history_count
            recency_diff = shadow_recency != canonical_recency
            grade_diff = shadow_grade != canonical_target_grade
            history_cause = mismatch_cause(
                "history", differs=history_diff, shadow=shadow, shadow_value=shadow_history,
                canonical_value=canonical_history_count, history=history, rejected=rejected,
                target_date=target_date,
            )
            recency_cause = mismatch_cause(
                "recency", differs=recency_diff, shadow=shadow, shadow_value=shadow_recency,
                canonical_value=canonical_recency, history=history, rejected=rejected,
                target_date=target_date,
            )
            grade_cause = mismatch_cause(
                "grade", differs=grade_diff, shadow=shadow, shadow_value=shadow_grade,
                canonical_value=canonical_target_grade, history=history, rejected=rejected,
                target_date=target_date,
            )
            causes[history_cause] += 1
            causes[recency_cause] += 1
            causes[grade_cause] += 1
            unexplained = int(any(
                cause.startswith("UNEXPLAINED_")
                for cause in (history_cause, recency_cause, grade_cause)
            ))
            opaque = row_id(
                race_id, published_row["box_number"], published_row["csv_dog_name"],
                scope=DEVELOPMENT_SCOPE,
            )
            if safe_int(shadow.get("box_number")) != int(published_row["box_number"]):
                raise ValueError(f"shadow overlap box conflict: {race_id} {opaque}")
            rows.append({
                "row_id": opaque,
                "race_id": race_id, "box_number": int(published_row["box_number"]),
                "shadow_source_path": shadow["_bound_source_path"],
                "raw_card_sha256": tier_a["source_csv_sha256"], "raw_cards_byte_identical": 1,
                "shadow_prior_start_count": "" if shadow_history is None else shadow_history,
                "canonical_prior_start_count": canonical_history_count,
                "history_discrepancy": int(history_diff), "history_cause": history_cause,
                "shadow_days_since_last_start": "" if shadow_recency is None else shadow_recency,
                "canonical_days_since_last_start": "" if canonical_recency is None else canonical_recency,
                "recency_discrepancy": int(recency_diff), "recency_cause": recency_cause,
                "shadow_target_grade": shadow_grade, "published_target_grade": canonical_grade(published_row["target_grade"]),
                "canonical_target_grade": canonical_target_grade,
                "grade_discrepancy": int(grade_diff), "grade_cause": grade_cause,
                "unexplained_mismatch": unexplained,
            })
    rows.sort(key=lambda row: (row["race_id"], row["box_number"], row["row_id"]))
    stable_csv(output_domain, "overlap_reconciliation.csv", list(rows[0]), rows)
    summary = {
        "authority": "NON_AUTHORITATIVE_DIAGNOSTIC",
        "trainer_dependency": False,
        "overlap_race_count": len(overlap_ids), "overlap_runner_count": len(rows),
        "byte_identical_raw_card_race_count": raw_identical_races,
        "history_discrepancy_count": sum(row["history_discrepancy"] for row in rows),
        "recency_discrepancy_count": sum(row["recency_discrepancy"] for row in rows),
        "grade_discrepancy_count": sum(row["grade_discrepancy"] for row in rows),
        "unexplained_mismatch_count": sum(row["unexplained_mismatch"] for row in rows),
        "unexplained_race_count": len({
            row["race_id"] for row in rows if row["unexplained_mismatch"]
        }),
        "independent_review_baseline": {
            "unexplained_runner_rows": 504,
            "overlap_runner_rows": 530,
            "covered_overlap_races": 73,
            "total_overlap_races": 73,
            "note": "baseline used the numeric-zero-as-missing parser; recomputed counts above parse zero as a value",
        },
        "cause_counts": dict(sorted(causes.items())),
        "bound_shadow_source_count": len(loaded["shadow_source_by_race"]),
        "bound_shadow_source_set_sha256": source_set_digest(
            (
                {"role": "shadow_reconciliation_source", **record}
                for record in loaded["shadow_source_by_race"].values()
            ),
            reject_duplicate_declarations=False,
        ),
        "canonical_rule": "rebuild from byte-identical raw pre-race card; never select a legacy builder value",
    }
    stable_json(output_domain, "reconciliation_summary.json", summary)
    return summary


def parse_race_id_from_sidecar(path: Path) -> tuple[str, date] | None:
    match = re.fullmatch(r"(Race \d+ - .+ - (\d{4}-\d{2}-\d{2}))\.csv\.metadata\.json", path.name)
    if not match:
        return None
    return match.group(1), date.fromisoformat(match.group(2))


def validate_race_identity(race_id: str, sidecar_path: Path, metadata: Mapping[str, Any]) -> date:
    parsed = parse_race_id_from_sidecar(sidecar_path)
    if parsed is None or parsed[0] != race_id:
        raise ValueError(f"sidecar path race identity mismatch: {race_id}")
    match = re.fullmatch(r"Race (\d+) - (.+) - (\d{4}-\d{2}-\d{2})", race_id)
    if not match:
        raise ValueError(f"invalid race identity: {race_id}")
    info = metadata.get("race_info") or {}
    required = {"date", "venue", "race_number", "race_time", "url"}
    if not required.issubset(info) or any(info.get(key) in (None, "") for key in required):
        raise ValueError(f"sidecar race identity is incomplete: {race_id}")
    if date.fromisoformat(str(info["date"])) != parsed[1]:
        raise ValueError(f"sidecar race date mismatch: {race_id}")
    if canonical_venue(info["venue"]) != canonical_venue(match.group(2)):
        raise ValueError(f"sidecar race venue mismatch: {race_id}")
    declared_number = safe_int(info.get("race_number"))
    if declared_number != int(match.group(1)):
        raise ValueError(f"sidecar race number mismatch: {race_id}")
    return parsed[1]


def canonical_card_url(value: Any) -> str:
    url = str(value or "").strip().split("?", 1)[0].rstrip("/")
    for suffix in ("/odds", "/export-expert-form?sort_by=&sort_dir="):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    return url.rstrip("/")


def validate_sidecar_semantics(
    race_id: str,
    csv_path: Path,
    sidecar_path: Path,
    metadata: Mapping[str, Any],
    *,
    expected_jump: datetime,
    expected_roster: Iterable[tuple[int, str]],
    csv_bytes: bytes | None = None,
    expected_url: str | None = None,
    enforce_jump_equality: bool = True,
    allow_roster_superset: bool = False,
) -> dict[str, Any]:
    """Bind meaning independently from inventory hashes."""
    race_date = validate_race_identity(race_id, sidecar_path, metadata)
    actual_roster = verify_card_sidecar_roster(
        csv_path, metadata, race_id=race_id, csv_bytes=csv_bytes
    )
    expected_roster = list(expected_roster)
    actual_counter = Counter(actual_roster)
    expected_counter = Counter(expected_roster)
    roster_matches = (
        not (expected_counter - actual_counter)
        if allow_roster_superset
        else actual_counter == expected_counter
    )
    if not roster_matches:
        raise ValueError(f"sidecar roster disagrees with canonical acquisition evidence: {race_id}")
    jump = sidecar_jump_timestamp(metadata, race_id)
    if enforce_jump_equality and jump != expected_jump.astimezone(MELBOURNE):
        raise ValueError(f"sidecar jump timestamp disagrees with canonical evidence: {race_id}")
    info = metadata["race_info"]
    urls = {
        canonical_card_url(info["url"]),
        canonical_card_url(metadata.get("race_url")),
        canonical_card_url(metadata.get("metadata_source_url")),
    }
    if "" in urls or len(urls) != 1:
        raise ValueError(f"sidecar source URL identity is inconsistent: {race_id}")
    if expected_url and canonical_card_url(expected_url) not in urls:
        raise ValueError(f"sidecar source URL disagrees with canonical evidence: {race_id}")
    match = re.fullmatch(r"Race (\d+) - .+ - (\d{4}-\d{2}-\d{2})", race_id)
    url = next(iter(urls))
    if match is None or f"/{match.group(2)}/{int(match.group(1))}/" not in f"{url}/":
        raise ValueError(f"sidecar source URL has wrong race identity: {race_id}")
    return {
        "race_id": race_id,
        "race_date": race_date.isoformat(),
        "venue": canonical_venue(info["venue"]),
        "race_number": int(info["race_number"]),
        "capture_timestamp": capture_timestamp(metadata).isoformat(),
        "jump_timestamp": expected_jump.astimezone(MELBOURNE).isoformat(),
        "sidecar_jump_timestamp": jump.isoformat(),
        "source_path": str(csv_path.resolve()),
        "source_url": url,
        "roster_sha256": canonical_digest(actual_roster),
        "canonical_roster_sha256": canonical_digest(expected_roster),
    }


def out_of_time_path_allowed(path: Path) -> bool:
    lower = str(path).lower()
    if not any(marker in lower for marker in ("/eligible_inputs/", "/refreshed_upcoming/", "/upcoming_races/", "/upcoming/")):
        return False
    banned = ("result", "replay", "reconstruct", "repair", "backfill", "official_result", "post_jump")
    return not any(marker in lower for marker in banned)


def scan_out_of_time_sources(evidence_roots: Iterable[Path]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_race: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exclusions: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for root in evidence_roots:
        for dirpath, _dirnames, filenames in os.walk(root):
            for filename in filenames:
                if not filename.endswith(".csv.metadata.json"):
                    continue
                sidecar_path = Path(dirpath) / filename
                parsed = parse_race_id_from_sidecar(sidecar_path)
                if not parsed or not (OUT_OF_TIME_START <= parsed[1] <= OUT_OF_TIME_END):
                    continue
                race_id, race_date = parsed
                resolved = str(sidecar_path.resolve())
                if resolved in seen_paths:
                    raise ValueError(f"duplicate live discovery path: {resolved}")
                seen_paths.add(resolved)
                if not out_of_time_path_allowed(sidecar_path):
                    exclusions.append({
                        "race_id": race_id, "race_date": race_date.isoformat(), "source_path": resolved,
                        "source_sha256": "NOT_OPENED_PATH_REJECTED",
                        "source_bytes": sidecar_path.stat().st_size,
                        "reason": "RECONSTRUCTED_OR_NONCONTEMPORANEOUS_PATH_REJECTED",
                    })
                    continue
                csv_path = Path(str(sidecar_path)[:-14])
                try:
                    csv_bytes = _read_regular_path_no_follow(
                        csv_path, label="pre-race form card"
                    )
                    metadata = validate_sidecar(
                        csv_path, sidecar_path, csv_bytes=csv_bytes
                    )
                    validate_race_identity(race_id, sidecar_path, metadata)
                    capture = capture_timestamp(metadata, require_timezone=True)
                    jump = sidecar_jump_timestamp(metadata, race_id)
                    validate_sidecar_semantics(
                        race_id,
                        csv_path,
                        sidecar_path,
                        metadata,
                        expected_jump=jump,
                        expected_roster=sidecar_roster(
                            metadata, source=f"live-discovery:{race_id}"
                        ),
                        csv_bytes=csv_bytes,
                    )
                except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
                    exclusions.append({
                        "race_id": race_id, "race_date": race_date.isoformat(), "source_path": resolved,
                        "source_sha256": sha256_path(sidecar_path), "source_bytes": sidecar_path.stat().st_size,
                        "reason": "INVALID_PRE_RACE_SOURCE:" + type(exc).__name__,
                    })
                    continue
                if capture > jump - timedelta(minutes=60):
                    exclusions.append({
                        "race_id": race_id, "race_date": race_date.isoformat(), "source_path": resolved,
                        "source_sha256": sha256_path(sidecar_path), "source_bytes": sidecar_path.stat().st_size,
                        "reason": "NOT_AVAILABLE_BY_T60",
                    })
                    continue
                by_race[race_id].append({
                    "race_id": race_id, "race_date": race_date, "csv_path": csv_path,
                    "csv_bytes": csv_bytes, "sidecar_path": sidecar_path,
                    "metadata": metadata, "capture": capture, "jump": jump,
                    "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
                    "sidecar_sha256": sha256_path(sidecar_path),
                })
    selected: dict[str, dict[str, Any]] = {}
    for race_id, options in by_race.items():
        freshest = max(option["capture"] for option in options)
        winners = [option for option in options if option["capture"] == freshest]
        identities: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for option in winners:
            identities[(option["csv_sha256"], option["sidecar_sha256"])].append(option)
        if len(identities) != 1:
            raise ValueError(f"ambiguous same-time live sources: {race_id}")
        aliases = next(iter(identities.values()))
        selected[race_id] = min(
            aliases, key=lambda item: str(item["sidecar_path"].resolve())
        )
        for option in options:
            if option is selected[race_id]:
                continue
            exclusions.append({
                "race_id": race_id, "race_date": option["race_date"].isoformat(),
                "source_path": str(option["sidecar_path"].resolve()), "source_sha256": option["sidecar_sha256"],
                "source_bytes": option["sidecar_path"].stat().st_size,
                "reason": (
                    "BYTE_IDENTICAL_ALIAS_OF_CANONICAL_SOURCE"
                    if option in aliases
                    else "VALID_DUPLICATE_NOT_FRESHEST_BY_T60"
                ),
            })
    exclusions.sort(key=lambda row: (row["race_id"], row["reason"], row["source_path"]))
    return selected, exclusions


def load_frozen_out_of_time_sources(
    freeze_dir: Path, reproducibility: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    inventory_path = freeze_dir / "out_of_time_source_inventory.csv"
    exclusions_path = freeze_dir / "out_of_time_exclusions.csv"
    manifest_path = freeze_dir / "out_of_time_manifest.json"
    freeze_contract = (reproducibility.get("trusted_inputs") or {}).get("out_of_time_freeze") or {}
    if Path(freeze_contract.get("path", "")).resolve() != freeze_dir.resolve():
        raise ValueError("out-of-time freeze path is not the bound reviewer-accessible location")
    freeze_paths = {
        "source_inventory": inventory_path,
        "exclusions": exclusions_path,
        "manifest": manifest_path,
    }
    expected_files = freeze_contract.get("files") or {}
    if set(expected_files) != set(freeze_paths):
        raise ValueError("out-of-time freeze file roles are incomplete")
    freeze_records: list[dict[str, Any]] = []
    for role, path in freeze_paths.items():
        expected = expected_files[role]
        record = verify_file_record(
            path,
            expected_sha256=expected["sha256"],
            expected_bytes=expected.get("bytes"),
            require_expected_bytes=True,
        )
        freeze_records.append({"role": f"out_of_time_freeze_{role}", **record})
    if source_set_digest(freeze_records) != freeze_contract.get("aggregate_sha256"):
        raise ValueError("out-of-time freeze aggregate mismatch")

    frozen_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        frozen_manifest.get("status") != "OUTCOME_UNOPENED_OUT_OF_TIME"
        or frozen_manifest.get("outcomes_opened") is not False
        or frozen_manifest.get("window_start") != OUT_OF_TIME_START.isoformat()
        or frozen_manifest.get("window_end") != OUT_OF_TIME_END.isoformat()
    ):
        raise ValueError("out-of-time freeze manifest policy mismatch")
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    with inventory_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["role"] not in {"raw_pre_race_card", "raw_pre_race_sidecar"}:
                raise ValueError(f"unexpected frozen source role: {row['role']}")
            if row["role"] in grouped[row["race_id"]]:
                raise ValueError(f"duplicate frozen source role: {row['race_id']} {row['role']}")
            grouped[row["race_id"]][row["role"]] = row
    selected: dict[str, dict[str, Any]] = {}
    for race_id, roles in sorted(grouped.items()):
        if set(roles) != {"raw_pre_race_card", "raw_pre_race_sidecar"}:
            raise ValueError(f"incomplete frozen source roles: {race_id}")
        csv_row = roles["raw_pre_race_card"]
        sidecar_row = roles["raw_pre_race_sidecar"]
        if (
            csv_row.get("status") != "OUTCOME_UNOPENED_OUT_OF_TIME"
            or sidecar_row.get("status") != "OUTCOME_UNOPENED_OUT_OF_TIME"
        ):
            raise ValueError(f"frozen source status mismatch: {race_id}")
        csv_path = Path(csv_row["path"])
        sidecar_path = Path(sidecar_row["path"])
        if not out_of_time_path_allowed(sidecar_path):
            raise ValueError(f"frozen sidecar path is forbidden: {sidecar_path}")
        if csv_path.resolve() != Path(str(sidecar_path)[:-14]).resolve():
            raise ValueError(f"frozen card/sidecar path mismatch: {race_id}")
        csv_bytes = _read_regular_path_no_follow(
            csv_path, label="pre-race form card"
        )
        metadata = validate_sidecar(csv_path, sidecar_path, csv_bytes=csv_bytes)
        csv_record = verify_retained_file_record(
            csv_path,
            csv_bytes,
            expected_sha256=csv_row["sha256"],
            expected_bytes=csv_row.get("bytes"),
            require_expected_bytes=True,
        )
        sidecar_record = verify_file_record(
            sidecar_path,
            expected_sha256=sidecar_row["sha256"],
            expected_bytes=sidecar_row.get("bytes"),
            require_expected_bytes=True,
        )
        race_date = validate_race_identity(race_id, sidecar_path, metadata)
        if not OUT_OF_TIME_START <= race_date <= OUT_OF_TIME_END:
            raise ValueError(f"frozen race outside Jul 11-Aug 9: {race_id}")
        capture = capture_timestamp(metadata, require_timezone=True)
        jump = sidecar_jump_timestamp(metadata, race_id)
        if capture > jump - timedelta(minutes=60):
            raise ValueError(f"frozen source is not available by T60: {race_id}")
        if parse_timestamp(csv_row["capture_timestamp"], require_timezone=True) != capture:
            raise ValueError(f"frozen capture timestamp declaration mismatch: {race_id}")
        if parse_timestamp(csv_row["jump_timestamp"], require_timezone=True) != jump:
            raise ValueError(f"frozen jump timestamp declaration mismatch: {race_id}")
        if (
            parse_timestamp(sidecar_row["capture_timestamp"], require_timezone=True) != capture
            or parse_timestamp(sidecar_row["jump_timestamp"], require_timezone=True) != jump
        ):
            raise ValueError(f"frozen sidecar timestamp declaration mismatch: {race_id}")
        validate_sidecar_semantics(
            race_id,
            csv_path,
            sidecar_path,
            metadata,
            expected_jump=jump,
            expected_roster=sidecar_roster(metadata, source=f"freeze:{race_id}"),
            csv_bytes=csv_bytes,
        )
        selected[race_id] = {
            "race_id": race_id, "race_date": race_date,
            "csv_path": csv_path, "csv_bytes": csv_bytes,
            "sidecar_path": sidecar_path, "metadata": metadata,
            "capture": capture, "jump": jump,
            "csv_sha256": csv_record["sha256"], "sidecar_sha256": sidecar_record["sha256"],
        }
    with exclusions_path.open(encoding="utf-8", newline="") as handle:
        exclusions = list(csv.DictReader(handle))
    return selected, exclusions, {
        "freeze_records": freeze_records,
        "freeze_aggregate_sha256": freeze_contract["aggregate_sha256"],
        "frozen_manifest": frozen_manifest,
    }


def build_out_of_time_manifest(
    evidence_roots: list[Path],
    output_domain: _OutputDomain,
    reproducibility: Mapping[str, Any],
    freeze_dir: Path | None = None,
    sealed_domain: _OutputDomain | None = None,
) -> dict[str, Any]:
    if sealed_domain is None:
        raise ValueError("sealed output domain is required")
    freeze_binding: dict[str, Any] | None = None
    if freeze_dir is None:
        selected, exclusions = scan_out_of_time_sources(evidence_roots)
        discovery_mode = "LIVE_READ_ONLY_DISCOVERY_THEN_SEALED"
    else:
        selected, exclusions, freeze_binding = load_frozen_out_of_time_sources(
            freeze_dir, reproducibility
        )
        discovery_mode = "HASH_BOUND_FROZEN_DISCOVERY_INPUT"
    race_rows: list[dict[str, Any]] = []
    runner_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    for race_id in sorted(selected):
        option = selected[race_id]
        venue, distance, grade, field_size = target_metadata(option, race_id)
        participants = (option["metadata"].get("runner_completeness") or {}).get("participants") or []
        verified_roster = verify_card_sidecar_roster(
            option["csv_path"],
            option["metadata"],
            race_id=race_id,
            csv_bytes=option.get("csv_bytes"),
        )
        race_rows.append({
            "race_id": race_id, "race_date": option["race_date"].isoformat(), "target_venue": venue,
            "target_distance_m": distance or "", "target_grade": grade, "field_size": field_size,
            "card_capture_timestamp": option["capture"].isoformat(), "jump_timestamp": option["jump"].isoformat(),
            "card_lead_minutes": fmt_number((option["jump"] - option["capture"]).total_seconds() / 60),
            "status": "OUTCOME_UNOPENED_OUT_OF_TIME",
        })
        for participant in sorted(participants, key=lambda row: (int(row["box_number"]), dog_token(row["dog_name"]))):
            box = int(participant["box_number"])
            runner_rows.append({
                "row_id": row_id(race_id, box, participant["dog_name"], scope=OUT_OF_TIME_SCOPE),
                "race_id": race_id, "box_number": box,
                "status": "OUTCOME_UNOPENED_OUT_OF_TIME",
            })
            alignment_rows.append({
                "split": OUT_OF_TIME_SCOPE,
                "race_id": race_id,
                "box_number": box,
                "row_id": row_id(race_id, box),
                "dog_name_token": dog_token(participant["dog_name"]),
                "canonical_runner_id": canonical_runner_id(
                    race_id, box, participant["dog_name"]
                ),
            })
        if len(verified_roster) != len(participants):
            raise ValueError(f"out-of-time roster count mismatch: {race_id}")
        for role, path, digest in (
            ("raw_pre_race_card", option["csv_path"], option["csv_sha256"]),
            ("raw_pre_race_sidecar", option["sidecar_path"], option["sidecar_sha256"]),
        ):
            source_bytes = (
                len(option["csv_bytes"])
                if role == "raw_pre_race_card"
                else path.stat().st_size
            )
            source_rows.append({
                "race_id": race_id, "role": role, "path": str(path.resolve()), "sha256": digest,
                "bytes": source_bytes, "capture_timestamp": option["capture"].isoformat(),
                "jump_timestamp": option["jump"].isoformat(), "status": "OUTCOME_UNOPENED_OUT_OF_TIME",
            })
    stable_csv(
        output_domain,
        "out_of_time_races.csv",
        list(race_rows[0]) if race_rows else ["race_id"],
        race_rows,
    )
    stable_csv(
        output_domain,
        "out_of_time_runners.csv",
        list(runner_rows[0]) if runner_rows else ["row_id"],
        runner_rows,
    )
    stable_csv(
        sealed_domain,
        "out_of_time_source_inventory.csv",
        list(source_rows[0]) if source_rows else ["race_id"],
        source_rows,
    )
    stable_csv(
        sealed_domain,
        "out_of_time_runner_alignment.csv",
        list(alignment_rows[0]) if alignment_rows else ["race_id"],
        alignment_rows,
    )
    exclusion_fields = ["race_id", "race_date", "source_path", "source_sha256", "source_bytes", "reason"]
    stable_csv(
        sealed_domain,
        "out_of_time_exclusions.csv",
        exclusion_fields,
        exclusions,
    )
    summary = {
        "schema_version": "form_only_v1_out_of_time_manifest_v2",
        "status": "OUTCOME_UNOPENED_OUT_OF_TIME", "outcomes_opened": False,
        "window_start": OUT_OF_TIME_START.isoformat(), "window_end": OUT_OF_TIME_END.isoformat(),
        "included_race_count": len(race_rows), "included_runner_count": len(runner_rows),
        "excluded_source_count": len(exclusions),
        "source_roots": [str(path.resolve()) for path in evidence_roots] if freeze_dir is None else [],
        "selection_rule": "freshest leakage-safe complete contemporaneous raw card available by T60",
        "discovery_mode": discovery_mode,
        "source_display_timezone": str(MELBOURNE),
        "jump_time_evidence": "race_info exact_url_match canonical_race_url",
        "construction_contract_sha256": reproducibility["construction_contract_sha256"],
    }
    if freeze_binding is not None:
        frozen_manifest = freeze_binding["frozen_manifest"]
        if (
            len(race_rows) != int(frozen_manifest["included_race_count"])
            or len(runner_rows) != int(frozen_manifest["included_runner_count"])
        ):
            raise ValueError("frozen manifest counts do not match source-derived roster")
        summary["bound_freeze"] = {
            "aggregate_sha256": freeze_binding["freeze_aggregate_sha256"],
            "file_count": len(freeze_binding["freeze_records"]),
        }
    selected_source_records: list[dict[str, Any]] = []
    for option in selected.values():
        selected_source_records.extend([
            {
                "role": "out_of_time_card",
                **retained_file_record(option["csv_path"], option["csv_bytes"]),
            },
            {
                "role": "out_of_time_sidecar",
                **file_record(option["sidecar_path"]),
            },
        ])
    summary["selected_source_set_sha256"] = source_set_digest(
        selected_source_records
    )
    stable_json(output_domain, "out_of_time_manifest.json", summary)
    return summary


def write_feature_contract(output_domain: _OutputDomain) -> None:
    stable_json(output_domain, "feature_contract.json", {
        "schema_version": "form_only_v1_feature_contract_v1",
        "mode": "ACQUISITION_ONLY_NO_MODEL_FIT_OR_EVALUATION",
        "as_of": {
            "card": "capture_timestamp <= canonical_jump_timestamp - 60 minutes",
            "history": "history_date < target_date",
            "history_cap": HISTORY_CAP,
            "recent_windows": [3, 5],
            "source_precedence": ["OFFICIAL_RACE_PAGE_TIER_A", "THEDOGS_PUBLISHED_HISTORY_NOT_TIER_A"],
        },
        "core": [
            "prior_start_count", "days_since_last_start", "recent_finish_mean_3",
            "recent_finish_mean_5", "recent_win_rate_5", "recent_place_rate_5",
            "recent_margin_mean_5", "career_finish_mean", "career_win_rate",
            "career_place_rate", "career_margin_mean", "same_venue_*",
            "same_distance_*", "same_grade_*", "*_missing",
        ],
        "context": ["box_number", "target_venue", "target_distance_m", "target_grade", "target_field_size"],
        "context_vocab_policy_for_later_fit": {
            "fit_on_training_fold_only": True, "rare_minimum_count": 10,
            "rare_token": "__RARE__", "unknown_token": "__UNKNOWN__",
        },
        "non_feature_keys": ["row_id", "race_id"],
        "identity_policy": (
            "trainer row IDs derive only from race identity plus box; dog names, tokens, "
            "digests, alignment maps, source paths, and cross-race join keys exist only "
            "inside the separately manifested sealed-validation boundary"
        ),
        "deferred": ["speed", "times", "sectionals", "opponent_strength", "high_dimensional_interactions"],
        "forbidden": sorted(FORBIDDEN_FEATURE_TOKENS),
        "missingness": "blank numeric value plus explicit family missingness flag; no silent zero fill",
        "grade_aliases": "embedded canonical_grade function; unknown values retained as stable uppercase tokens",
        "venue_aliases_sha256": hashlib.sha256(
            json.dumps(VENUE_ALIASES, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    })


def write_market_coverage(output_domain: _OutputDomain) -> None:
    stable_json(output_domain, "market_coverage.json", {
        "schema_version": "form_only_v1_separate_market_coverage_v1",
        "status": "DATA_MISSING",
        "separate_from_form_input_eligibility": True,
        "market_fields_in_packet": False,
        "paired_race_counts": {"T-60": 2, "T-30": 402, "T-10": 497, "T-2": 501},
        "count_provenance": "OWNER_GOAL_2026_07_18_CARRY_FORWARD_EXPECTATIONS",
        "independent_frozen_cohort_manifest_bound": False,
        "blocker": (
            "No immutable market cohort manifest available to this acquisition-only lane "
            "binds the supplied timing counts and their pairing semantics."
        ),
        "T-60_status": "DATA_MISSING",
        "preserved_for_later_evaluation": ["T-30", "T-10", "T-2"],
        "not_a_form_packet_blocker": True,
    })


def validate_trainer_visible_artifacts(
    output_domain: Path | _OutputDomain,
) -> None:
    if isinstance(output_domain, Path):
        with _bound_existing_output_domain(
            output_domain, label="trainer output domain", writable=False
        ) as bound_domain:
            validate_trainer_visible_artifacts(bound_domain)
        return
    identity_scopes: dict[str, set[tuple[str, str]]] = defaultdict(set)
    required_nonempty_csv = {
        "development_features.csv", "development_races.csv", "development_runners.csv",
        "out_of_time_races.csv", "out_of_time_runners.csv",
    }
    for name in sorted(TRAINER_ARTIFACT_NAMES):
        payload_bytes = output_domain.read_bytes(name)
        if not payload_bytes:
            raise ValueError(f"missing or empty trainer artifact: {name}")
        text = payload_bytes.decode("utf-8")
        if "|dog:" in text:
            raise ValueError(f"sealed dog alignment key leaked into artifact: {name}")
        if name.endswith(".csv"):
            reader = csv.DictReader(io.StringIO(text, newline=""))
            forbidden = set(reader.fieldnames or []).intersection(FORBIDDEN_ARTIFACT_FIELDS)
            if forbidden:
                raise ValueError(f"identity-bearing fields in {name}: {sorted(forbidden)}")
            rows_seen = 0
            split = OUT_OF_TIME_SCOPE if name.startswith("out_of_time_") else DEVELOPMENT_SCOPE
            for row in reader:
                rows_seen += 1
                race_id = row.get("race_id") or ""
                for key in ("row_id", "entity_id"):
                    value = row.get(key) or ""
                    if re.fullmatch(r"[0-9a-f]{64}", value):
                        identity_scopes[value].add((split, race_id))
            if name in required_nonempty_csv and rows_seen == 0:
                raise ValueError(f"empty generated trainer CSV: {name}")
        elif name.endswith(".json"):
            payload = json.loads(text)
            stack = [payload]
            while stack:
                value = stack.pop()
                if isinstance(value, dict):
                    forbidden = set(value).intersection(FORBIDDEN_ARTIFACT_FIELDS)
                    if forbidden:
                        raise ValueError(f"identity-bearing fields in {name}: {sorted(forbidden)}")
                    stack.extend(value.values())
                elif isinstance(value, list):
                    stack.extend(value)
    crossing = {key: scopes for key, scopes in identity_scopes.items() if len(scopes) > 1}
    if crossing:
        raise ValueError("opaque runner ID links multiple races or splits")


def write_trainer_input_manifest(
    trainer_domain: _OutputDomain,
    control_domain: _OutputDomain,
    trainer_manifest: Mapping[str, Any],
) -> None:
    allowed = []
    manifest_rows = {row["path"]: row for row in trainer_manifest["files"]}
    if set(manifest_rows) != TRAINER_ARTIFACT_NAMES:
        raise ValueError("trainer artifact manifest does not match the declared read surface")
    for name in sorted(TRAINER_ARTIFACT_NAMES):
        payload = trainer_domain.read_bytes(name)
        if not payload:
            raise ValueError(f"missing or empty generated artifact: {name}")
        allowed.append({
            **manifest_rows[name],
            "type": "regular_file",
            "role": TRAINER_ARTIFACT_ROLES[name],
        })
    signature = control_domain.read_bytes("artifact-manifest.sha256")
    stable_json(control_domain, "trainer_input_manifest.json", {
        "schema_version": "form_only_v1_trainer_input_manifest_v2",
        "trust_domain": "TRAINER_VISIBLE_AUTHORITATIVE",
        "trainer_root": TRAINER_ROOT_NAME,
        "allowed_files": allowed,
        "declared_file_count": len(allowed),
        "artifact_manifest": {
            "path": "artifact-manifest.sha256",
            "type": "regular_file",
            "role": "CONTROL_INTEGRITY_SIGNATURE",
            "sha256": hashlib.sha256(signature).hexdigest(),
            "bytes": len(signature),
            "trainer_aggregate_sha256": trainer_manifest["aggregate_sha256"],
        },
        "row_identity": "sha256(FORM_ONLY_V1|race_box|race_id|box_number)",
        "sealed_validation_bundle": "NOT_TRAINER_READABLE",
        "non_authoritative_diagnostic_bundle": "NOT_TRAINER_INPUT",
        "forbidden_roots": [
            CONTROL_PLANE_ROOT_NAME, "sealed_validation", "non_authoritative_diagnostic",
        ],
    })


def write_artifact_manifest(
    output_domain: Path | _OutputDomain,
    names: set[str],
    *,
    filename: str = "artifact-manifest.sha256",
    manifest_domain: _OutputDomain | None = None,
) -> dict[str, Any]:
    if isinstance(output_domain, Path):
        with _bound_existing_output_domain(
            output_domain, label="artifact manifest domain", writable=True
        ) as bound_domain:
            return write_artifact_manifest(
                bound_domain,
                names,
                filename=filename,
                manifest_domain=bound_domain,
            )
    manifest = artifact_manifest_records(output_domain, names)
    destination = manifest_domain or output_domain
    destination.write_bytes(filename, manifest["text"].encode("utf-8"))
    return {key: value for key, value in manifest.items() if key != "text"}


def artifact_manifest_records(
    output_domain: Path | _OutputDomain, names: set[str]
) -> dict[str, Any]:
    if isinstance(output_domain, Path):
        with _bound_existing_output_domain(
            output_domain, label="artifact record domain", writable=False
        ) as bound_domain:
            return artifact_manifest_records(bound_domain, names)
    rows = []
    for name in sorted(names):
        payload = output_domain.read_bytes(name)
        if not payload:
            raise ValueError(f"missing or empty generated artifact: {name}")
        rows.append({
            "path": name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        })
    text = "".join(f"{row['sha256']}  {row['path']}\n" for row in rows)
    return {
        "files": rows,
        "aggregate_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text": text,
    }


def _snapshot_bound_domains(
    domains: _PacketOutputScope,
    manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, bytes]]:
    return {
        domain: {
            str(row["path"]): domains[domain].read_bytes(str(row["path"]))
            for row in manifest["files"]
        }
        for domain, manifest in manifests.items()
    }


def _open_directory_no_follow(path: Path, *, label: str) -> int:
    """Open an absolute directory by walking every component without following links."""
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} path must be absolute and traversal-free: {path}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    current_fd = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            next_fd = _open_child_directory_no_follow(
                current_fd, component, label=f"{label} ancestor"
            )
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except Exception:
        os.close(current_fd)
        raise


def _open_child_directory_no_follow(parent_fd: int, name: str, *, label: str) -> int:
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"unsafe {label} component: {name!r}")
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ValueError(f"missing {label} directory: {name}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise ValueError(f"{label} path is a symlink or not a directory: {name}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        child_fd = os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise ValueError(f"cannot safely open {label} directory: {name}") from exc
    after = os.fstat(child_fd)
    if (
        (before.st_dev, before.st_ino, stat.S_IFMT(before.st_mode))
        != (after.st_dev, after.st_ino, stat.S_IFMT(after.st_mode))
    ):
        os.close(child_fd)
        raise ValueError(f"{label} path changed during open: {name}")
    return child_fd


def _read_regular_at(directory_fd: int, name: str, *, label: str) -> bytes:
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"unsafe {label} component: {name!r}")
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ValueError(f"missing {label}: {name}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} is not a regular file: {name}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        file_fd = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise ValueError(f"{label} is not a regular file: {name}") from exc
    try:
        metadata = os.fstat(file_fd)
        if (
            (before.st_dev, before.st_ino, stat.S_IFMT(before.st_mode))
            != (metadata.st_dev, metadata.st_ino, stat.S_IFMT(metadata.st_mode))
        ):
            raise ValueError(f"{label} path changed during open: {name}")
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(f"{label} is not a regular single-link file: {name}")
        chunks = []
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(file_fd)


def _directory_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_nlink,
    )


def _entry_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return _directory_identity(metadata)


_OutputEntryRecord = tuple[tuple[int, int, int, int], int, str]


def _output_component(name: str, *, label: str) -> str:
    if (
        not name
        or name in {".", ".."}
        or name.startswith(".")
        or "/" in name
        or "\\" in name
    ):
        raise ValueError(f"unsafe {label} component: {name!r}")
    return name


class _OutputDomain:
    """A validated output directory held open for its complete write phase."""

    def __init__(
        self,
        *,
        name: str,
        directory_fd: int,
        container_fd: int,
        container_verifier: Callable[[], None],
        writable: bool,
    ) -> None:
        self.name = _output_component(name, label="output domain")
        self.fd = directory_fd
        self._container_fd = container_fd
        self._container_verifier = container_verifier
        self._writable = writable
        self._closed = False
        self._write_counter = 0
        self._owned_entries: dict[str, _OutputEntryRecord] = {}
        self._temporary_entries: dict[
            str, tuple[int, int, int, int] | None
        ] = {}
        metadata = os.fstat(self.fd)
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_nlink < 1:
            raise ValueError(f"output domain is not a live directory: {self.name}")
        self._identity = _directory_identity(metadata)
        self._baseline_entries = {
            entry_name: self._entry_record(entry_name, label="baseline output artifact")
            for entry_name in os.listdir(self.fd)
        }
        self.assert_bound()

    def _assert_open(self) -> None:
        if self._closed:
            raise ValueError(f"output domain descriptor is closed: {self.name}")

    def _assert_descriptor_identity(self) -> None:
        self._assert_open()
        try:
            metadata = os.fstat(self.fd)
        except OSError as exc:
            raise ValueError(f"output domain descriptor is invalid: {self.name}") from exc
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_nlink < 1
            or _directory_identity(metadata) != self._identity
        ):
            raise ValueError(f"output domain descriptor identity changed: {self.name}")

    def _entry_record(self, name: str, *, label: str) -> _OutputEntryRecord:
        try:
            before = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise ValueError(f"{label} disappeared: {self.name}/{name}") from exc
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
        ):
            raise ValueError(
                f"{label} is not a regular single-link file: {self.name}/{name}"
            )
        identity = _entry_identity(before)
        payload = _read_regular_at(self.fd, name, label=label)
        try:
            after = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise ValueError(f"{label} disappeared: {self.name}/{name}") from exc
        if _entry_identity(after) != identity:
            raise ValueError(f"{label} identity changed: {self.name}/{name}")
        return identity, len(payload), hashlib.sha256(payload).hexdigest()

    def _assert_record(
        self,
        name: str,
        expected: _OutputEntryRecord,
        *,
        label: str,
    ) -> None:
        actual = self._entry_record(name, label=label)
        if actual[0] != expected[0]:
            raise ValueError(f"{label} identity changed: {self.name}/{name}")
        if actual[1:] != expected[1:]:
            raise ValueError(f"{label} content changed: {self.name}/{name}")

    def _assert_exact_entries(self) -> None:
        actual = set(os.listdir(self.fd))
        expected = (
            set(self._baseline_entries)
            | set(self._owned_entries)
            | set(self._temporary_entries)
        )
        if actual != expected:
            raise ValueError(
                f"output domain surface changed: {self.name}: "
                f"unexpected={sorted(actual - expected)} "
                f"missing={sorted(expected - actual)}"
            )

    def _assert_tracked_entries(self) -> None:
        for name, expected in self._baseline_entries.items():
            self._assert_record(
                name, expected, label="baseline output artifact"
            )
        for name, expected in self._owned_entries.items():
            self._assert_record(name, expected, label="output artifact")
        for name, expected in self._temporary_entries.items():
            if expected is None:
                raise ValueError(
                    f"staged output identity was not established: {self.name}/{name}"
                )
            try:
                metadata = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
            except FileNotFoundError as exc:
                raise ValueError(
                    f"staged output disappeared: {self.name}/{name}"
                ) from exc
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or _entry_identity(metadata) != expected
            ):
                raise ValueError(
                    f"staged output identity changed: {self.name}/{name}"
                )

    def assert_bound(self) -> None:
        self._assert_open()
        self._container_verifier()
        self._assert_descriptor_identity()
        try:
            entry = os.stat(
                self.name,
                dir_fd=self._container_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError as exc:
            raise ValueError(f"output domain entry disappeared: {self.name}") from exc
        if (
            stat.S_ISLNK(entry.st_mode)
            or not stat.S_ISDIR(entry.st_mode)
            or _directory_identity(entry) != self._identity
        ):
            raise ValueError(f"output domain entry identity changed: {self.name}")
        self._assert_exact_entries()
        self._assert_tracked_entries()

    def list_names(self) -> set[str]:
        self.assert_bound()
        names = set(os.listdir(self.fd))
        self.assert_bound()
        return names

    def read_bytes(self, name: str) -> bytes:
        name = _output_component(name, label="output artifact")
        self.assert_bound()
        payload = _read_regular_at(
            self.fd, name, label=f"{self.name} output artifact"
        )
        self.assert_bound()
        return payload

    def file_record(self, name: str) -> dict[str, Any]:
        payload = self.read_bytes(name)
        return {
            "path": name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }

    def _remove_owned_entry(
        self,
        name: str,
        identity: tuple[int, int, int, int],
        *,
        directory_fd: int | None = None,
    ) -> None:
        target_fd = self.fd if directory_fd is None else directory_fd
        try:
            metadata = os.stat(name, dir_fd=target_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        actual = _entry_identity(metadata)
        if actual[:3] != identity[:3]:
            return
        try:
            os.unlink(name, dir_fd=target_fd)
        except FileNotFoundError:
            return

    def rollback(self, *, directory_fd: int | None = None) -> None:
        target_fd = self.fd if directory_fd is None else directory_fd
        if directory_fd is None:
            self._assert_open()
        for name, identity in reversed(tuple(self._temporary_entries.items())):
            if identity is not None:
                self._remove_owned_entry(name, identity, directory_fd=target_fd)
            self._temporary_entries.pop(name, None)
        for name, record in reversed(tuple(self._owned_entries.items())):
            self._remove_owned_entry(name, record[0], directory_fd=target_fd)
            self._owned_entries.pop(name, None)
        try:
            os.fsync(target_fd)
        except OSError:
            pass

    def write_bytes(self, name: str, payload: bytes) -> None:
        if not self._writable:
            raise ValueError(f"output domain is read-only: {self.name}")
        name = _output_component(name, label="output artifact")
        if not payload:
            raise ValueError(f"refusing to write empty output artifact: {self.name}/{name}")
        self.assert_bound()
        try:
            os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ValueError(f"output artifact already exists: {self.name}/{name}")

        self._write_counter += 1
        temporary_name = (
            f".{name}.tmp-{os.getpid()}-{self._write_counter}"
        )
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        file_fd: int | None = None
        temporary_identity: tuple[int, int, int, int] | None = None
        try:
            file_fd = os.open(
                temporary_name, flags, 0o600, dir_fd=self.fd
            )
            self._temporary_entries[temporary_name] = None
            temporary_metadata = os.fstat(file_fd)
            if (
                not stat.S_ISREG(temporary_metadata.st_mode)
                or temporary_metadata.st_nlink != 1
            ):
                raise ValueError(
                    f"staged output is not a regular single-link file: "
                    f"{self.name}/{name}"
                )
            temporary_identity = _entry_identity(temporary_metadata)
            self._temporary_entries[temporary_name] = temporary_identity
            offset = 0
            while offset < len(payload):
                written = os.write(file_fd, payload[offset:])
                if written <= 0:
                    raise OSError("short write while staging output artifact")
                offset += written
            os.fsync(file_fd)
            completed = os.fstat(file_fd)
            if (
                _entry_identity(completed) != temporary_identity
                or completed.st_size != len(payload)
                or not stat.S_ISREG(completed.st_mode)
                or completed.st_nlink != 1
            ):
                raise ValueError(f"staged output identity changed: {self.name}/{name}")

            self.assert_bound()
            try:
                os.stat(name, dir_fd=self.fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise ValueError(f"output artifact appeared during write: {self.name}/{name}")
            os.replace(
                temporary_name,
                name,
                src_dir_fd=self.fd,
                dst_dir_fd=self.fd,
            )
            payload_record = (
                temporary_identity,
                len(payload),
                hashlib.sha256(payload).hexdigest(),
            )
            self._owned_entries[name] = payload_record
            self._temporary_entries.pop(temporary_name, None)
            final_metadata = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
            final_identity = _entry_identity(final_metadata)
            if (
                final_identity != temporary_identity
                or not stat.S_ISREG(final_metadata.st_mode)
                or final_metadata.st_nlink != 1
            ):
                raise ValueError(f"published output identity changed: {self.name}/{name}")
            os.fsync(self.fd)
            if self.read_bytes(name) != payload:
                raise ValueError(f"published output verification failed: {self.name}/{name}")
            self.assert_bound()
        except BaseException as original_error:
            if file_fd is not None:
                try:
                    if temporary_identity is None:
                        try:
                            temporary_identity = _entry_identity(os.fstat(file_fd))
                        except OSError:
                            temporary_identity = None
                    os.close(file_fd)
                except BaseException as cleanup_error:
                    if hasattr(original_error, "add_note"):
                        original_error.add_note(
                            f"staged descriptor cleanup error: {cleanup_error}"
                        )
                file_fd = None
            record = self._owned_entries.pop(name, None)
            try:
                if temporary_identity is not None:
                    self._remove_owned_entry(name, temporary_identity)
                if record is not None and record[0] != temporary_identity:
                    self._remove_owned_entry(name, record[0])
                staged_identity = self._temporary_entries.pop(temporary_name, None)
                if staged_identity is None:
                    staged_identity = temporary_identity
                if staged_identity is not None:
                    self._remove_owned_entry(temporary_name, staged_identity)
            except BaseException as cleanup_error:
                if hasattr(original_error, "add_note"):
                    original_error.add_note(
                        f"staged pathname cleanup error: {cleanup_error}"
                    )
            raise
        finally:
            if file_fd is not None:
                os.close(file_fd)

    def verify_for_close(self) -> None:
        self.assert_bound()
        if self._temporary_entries:
            raise ValueError(f"staged output files remain in domain: {self.name}")

    def _close_descriptor(self) -> None:
        if self._closed:
            return
        try:
            os.close(self.fd)
        finally:
            self._closed = True

    def close(self) -> None:
        if self._closed:
            return
        self.verify_for_close()
        self._close_descriptor()


class _PacketOutputScope:
    """Own the packet root and all bound output-domain descriptors."""

    def __init__(
        self,
        *,
        parent_fd: int,
        packet_fd: int,
        packet_name: str,
        root_names: tuple[str, ...],
        created_root: bool,
        created_domains: set[str],
    ) -> None:
        self._parent_fd = parent_fd
        self._packet_fd = packet_fd
        self._packet_name = packet_name
        self._root_names = root_names
        self._created_root = created_root
        self._created_domains = created_domains
        self._domains: dict[str, _OutputDomain] = {}
        self._closed = False
        parent_metadata = os.fstat(parent_fd)
        packet_metadata = os.fstat(packet_fd)
        if not stat.S_ISDIR(parent_metadata.st_mode):
            raise ValueError("packet root parent is not a directory")
        if not stat.S_ISDIR(packet_metadata.st_mode) or packet_metadata.st_nlink < 1:
            raise ValueError("packet root is not a live directory")
        self._parent_identity = (
            parent_metadata.st_dev,
            parent_metadata.st_ino,
            stat.S_IFMT(parent_metadata.st_mode),
        )
        self._packet_identity = _directory_identity(packet_metadata)

    def _assert_packet_bound(self) -> None:
        if self._closed:
            raise ValueError("packet output scope is closed")
        parent = os.fstat(self._parent_fd)
        parent_identity = (
            parent.st_dev,
            parent.st_ino,
            stat.S_IFMT(parent.st_mode),
        )
        if parent_identity != self._parent_identity or not stat.S_ISDIR(parent.st_mode):
            raise ValueError("packet root parent descriptor identity changed")
        packet = os.fstat(self._packet_fd)
        if (
            not stat.S_ISDIR(packet.st_mode)
            or packet.st_nlink < 1
            or _directory_identity(packet) != self._packet_identity
        ):
            raise ValueError("packet root descriptor identity changed")
        try:
            entry = os.stat(
                self._packet_name,
                dir_fd=self._parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError as exc:
            raise ValueError("packet root entry disappeared") from exc
        if (
            stat.S_ISLNK(entry.st_mode)
            or not stat.S_ISDIR(entry.st_mode)
            or _directory_identity(entry) != self._packet_identity
        ):
            raise ValueError("packet root entry identity changed")
        actual_names = set(os.listdir(self._packet_fd))
        if actual_names != set(self._root_names):
            raise ValueError(
                "packet root surface changed during write phase: "
                f"unexpected={sorted(actual_names - set(self._root_names))} "
                f"missing={sorted(set(self._root_names) - actual_names)}"
            )

    def add_domain(self, name: str, directory_fd: int, *, writable: bool) -> None:
        self._domains[name] = _OutputDomain(
            name=name,
            directory_fd=directory_fd,
            container_fd=self._packet_fd,
            container_verifier=self._assert_packet_bound,
            writable=writable,
        )

    def __getitem__(self, name: str) -> _OutputDomain:
        return self._domains[name]

    @property
    def packet_fd(self) -> int:
        return self._packet_fd

    def verify_for_close(self) -> None:
        self._assert_packet_bound()
        for domain in self._domains.values():
            domain.verify_for_close()
        identities = {
            (domain._identity[0], domain._identity[1])
            for domain in self._domains.values()
        }
        if len(identities) != len(self._domains):
            raise ValueError("packet output domains alias the same directory object")

    def rollback(self, *, directory_fds: Mapping[str, int] | None = None) -> None:
        errors: list[BaseException] = []
        for name, domain in reversed(tuple(self._domains.items())):
            try:
                domain.rollback(
                    directory_fd=None
                    if directory_fds is None
                    else directory_fds.get(name)
                )
            except BaseException as caught:
                errors.append(caught)
        if errors:
            first = errors[0]
            for extra in errors[1:]:
                if hasattr(first, "add_note"):
                    first.add_note(f"additional output rollback error: {extra}")
            raise first

    def _remove_created_directories(self, *, packet_fd: int | None = None) -> None:
        target_fd = self._packet_fd if packet_fd is None else packet_fd
        for name in reversed(self._root_names):
            if name not in self._created_domains:
                continue
            domain = self._domains.get(name)
            if domain is None:
                continue
            try:
                entry = os.stat(
                    name, dir_fd=target_fd, follow_symlinks=False
                )
                if _directory_identity(entry) != domain._identity:
                    continue
                os.rmdir(name, dir_fd=target_fd)
            except OSError:
                continue

    def __enter__(self) -> _PacketOutputScope:
        try:
            self.verify_for_close()
        except BaseException as caught:
            try:
                self.__exit__(type(caught), caught, caught.__traceback__)
            except BaseException as cleanup_error:
                if hasattr(caught, "add_note"):
                    caught.add_note(f"pre-enter cleanup error: {cleanup_error}")
            raise
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> bool:
        validation_error: BaseException | None = exc
        cleanup_domains: dict[str, int] = {}
        cleanup_packet_fd: int | None = None
        cleanup_parent_fd: int | None = None
        if validation_error is None:
            try:
                self.verify_for_close()
            except BaseException as caught:
                validation_error = caught
        if validation_error is not None:
            try:
                self.rollback()
            except BaseException as cleanup_error:
                validation_error.add_note(f"output rollback error: {cleanup_error}")
        else:
            try:
                cleanup_parent_fd = os.dup(self._parent_fd)
                cleanup_packet_fd = os.dup(self._packet_fd)
                for name, domain in self._domains.items():
                    cleanup_domains[name] = os.dup(domain.fd)
            except BaseException as caught:
                validation_error = caught
                try:
                    self.rollback()
                except BaseException as cleanup_error:
                    validation_error.add_note(
                        f"output rollback error: {cleanup_error}"
                    )

        if validation_error is None:
            for domain in reversed(tuple(self._domains.values())):
                try:
                    domain.close()
                except BaseException as close_error:
                    validation_error = close_error
                    break
            if validation_error is not None and cleanup_domains:
                try:
                    self.rollback(directory_fds=cleanup_domains)
                except BaseException as cleanup_error:
                    validation_error.add_note(
                        f"output rollback error: {cleanup_error}"
                    )
        if validation_error is not None and cleanup_packet_fd is not None:
            self._remove_created_directories(packet_fd=cleanup_packet_fd)
        elif validation_error is not None:
            self._remove_created_directories()

        for domain in self._domains.values():
            if not domain._closed:
                try:
                    domain._close_descriptor()
                except BaseException as close_error:
                    if validation_error is None:
                        validation_error = close_error
                    else:
                        validation_error.add_note(
                            f"output descriptor close error: {close_error}"
                        )

        try:
            os.close(self._packet_fd)
        except OSError as close_error:
            if validation_error is None:
                validation_error = close_error
        if validation_error is not None and cleanup_domains:
            try:
                self.rollback(directory_fds=cleanup_domains)
            except BaseException as cleanup_error:
                validation_error.add_note(f"output rollback error: {cleanup_error}")
            if cleanup_packet_fd is not None:
                self._remove_created_directories(packet_fd=cleanup_packet_fd)
        if (
            validation_error is not None
            and self._created_root
        ):
            try:
                entry = os.stat(
                    self._packet_name,
                    dir_fd=self._parent_fd,
                    follow_symlinks=False,
                )
                if _directory_identity(entry)[:3] == self._packet_identity[:3]:
                    os.rmdir(self._packet_name, dir_fd=self._parent_fd)
            except OSError:
                pass
        try:
            os.close(self._parent_fd)
        except OSError as close_error:
            if validation_error is None:
                validation_error = close_error
        if validation_error is not None and cleanup_domains:
            try:
                self.rollback(directory_fds=cleanup_domains)
            except BaseException as cleanup_error:
                validation_error.add_note(f"output rollback error: {cleanup_error}")
            if cleanup_packet_fd is not None:
                self._remove_created_directories(packet_fd=cleanup_packet_fd)
            if self._created_root and cleanup_parent_fd is not None:
                try:
                    entry = os.stat(
                        self._packet_name,
                        dir_fd=cleanup_parent_fd,
                        follow_symlinks=False,
                    )
                    if _directory_identity(entry)[:3] == self._packet_identity[:3]:
                        os.rmdir(self._packet_name, dir_fd=cleanup_parent_fd)
                except OSError:
                    pass
        for cleanup_fd in set(cleanup_domains.values()):
            try:
                os.close(cleanup_fd)
            except OSError:
                pass
        for cleanup_fd in (cleanup_packet_fd, cleanup_parent_fd):
            if cleanup_fd is not None:
                try:
                    os.close(cleanup_fd)
                except OSError:
                    pass
        self._closed = True
        if exc is None and validation_error is not None:
            raise validation_error
        return False


@contextmanager
def _bound_existing_output_domain(
    path: Path, *, label: str, writable: bool
) -> Iterator[_OutputDomain]:
    if not path.is_absolute() or ".." in path.parts or path.name in {"", ".", ".."}:
        raise ValueError(f"{label} path must be absolute and traversal-free: {path}")
    parent_fd = _open_directory_no_follow(path.parent, label=f"{label} parent")
    directory_fd: int | None = None
    domain: _OutputDomain | None = None
    parent_metadata = os.fstat(parent_fd)
    parent_identity = (
        parent_metadata.st_dev,
        parent_metadata.st_ino,
        stat.S_IFMT(parent_metadata.st_mode),
    )

    def verify_parent() -> None:
        current = os.fstat(parent_fd)
        current_identity = (
            current.st_dev,
            current.st_ino,
            stat.S_IFMT(current.st_mode),
        )
        if current_identity != parent_identity:
            raise ValueError(f"{label} parent descriptor identity changed")

    try:
        directory_fd = _open_child_directory_no_follow(
            parent_fd, path.name, label=label
        )
        domain = _OutputDomain(
            name=path.name,
            directory_fd=directory_fd,
            container_fd=parent_fd,
            container_verifier=verify_parent,
            writable=writable,
        )
        try:
            yield domain
            domain.verify_for_close()
        except BaseException:
            if writable:
                domain.rollback()
            raise
        finally:
            domain.close()
            directory_fd = None
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
        os.close(parent_fd)


def _read_regular_path_no_follow(path: Path, *, label: str) -> bytes:
    if not path.is_absolute() or ".." in path.parts or path.name in {"", ".", ".."}:
        raise ValueError(f"{label} path must be absolute and traversal-free: {path}")
    parent_fd = _open_directory_no_follow(path.parent, label=f"{label} parent")
    try:
        return _read_regular_at(parent_fd, path.name, label=label)
    finally:
        os.close(parent_fd)


def _domain_declarations(
    expected_output: Mapping[str, Any], domain: str
) -> dict[str, Mapping[str, Any]]:
    expected_domain = ((expected_output.get("domains") or {}).get(domain) or {})
    artifacts = expected_domain.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError(f"reproducibility contract has no physical declarations for {domain}")
    roles = DOMAIN_ARTIFACT_ROLES[domain]
    declared: dict[str, Mapping[str, Any]] = {}
    for declaration in artifacts:
        if not isinstance(declaration, dict):
            raise ValueError(f"malformed {domain} artifact declaration")
        name = _safe_declaration_name(declaration.get("path"))
        if name in declared:
            raise ValueError(f"duplicate {domain} artifact declaration: {name}")
        if declaration.get("type") != "regular_file":
            raise ValueError(f"{domain} type declaration mismatch: {name}")
        if declaration.get("role") != roles.get(name):
            raise ValueError(f"{domain} role declaration mismatch: {name}")
        expected_bytes = declaration.get("bytes")
        if type(expected_bytes) is not int or expected_bytes < 1:
            raise ValueError(f"invalid {domain} byte length declaration: {name}")
        expected_hash = declaration.get("sha256")
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ValueError(f"invalid {domain} sha256 declaration: {name}")
        declared[name] = declaration
    if set(declared) != set(roles):
        raise ValueError(
            f"declared {domain} physical surface mismatch: "
            f"unexpected={sorted(set(declared) - set(roles))} "
            f"missing={sorted(set(roles) - set(declared))}"
        )
    if expected_domain.get("declared_file_count") != len(declared):
        raise ValueError(f"declared {domain} file count mismatch")
    declared_hashes = {
        name: str(declaration["sha256"])
        for name, declaration in sorted(declared.items())
    }
    if expected_domain.get("artifact_files") != declared_hashes:
        raise ValueError(f"conflicting {domain} artifact hash declarations")
    if expected_domain.get("aggregate_sha256") != expected_domain.get(
        "physical_aggregate_sha256"
    ):
        raise ValueError(f"conflicting {domain} aggregate declarations")
    return declared


def _verify_declared_domain(
    directory_fd: int,
    domain: str,
    expected_output: Mapping[str, Any],
) -> dict[str, bytes]:
    declarations = _domain_declarations(expected_output, domain)
    actual_names = set(os.listdir(directory_fd))
    if actual_names != set(declarations):
        raise ValueError(
            f"{domain} physical surface mismatch: "
            f"unexpected={sorted(actual_names - set(declarations))} "
            f"missing={sorted(set(declarations) - actual_names)}"
        )
    payloads: dict[str, bytes] = {}
    rows: list[str] = []
    for name in sorted(declarations):
        declaration = declarations[name]
        payload = _read_regular_at(directory_fd, name, label=f"{domain} artifact")
        if len(payload) != declaration["bytes"]:
            raise ValueError(f"{domain} artifact byte length mismatch: {name}")
        actual_hash = hashlib.sha256(payload).hexdigest()
        if actual_hash != declaration["sha256"]:
            raise ValueError(f"{domain} artifact sha256 mismatch: {name}")
        rows.append(f"{actual_hash}  {name}\n")
        payloads[name] = payload
    domain_signature = {
        "sealed_validation": (
            "sealed-validation-manifest.sha256",
            SEALED_VALIDATION_ARTIFACT_NAMES,
        ),
        "non_authoritative_diagnostic": (
            "non-authoritative-diagnostic-manifest.sha256",
            DIAGNOSTIC_ARTIFACT_NAMES,
        ),
    }.get(domain)
    if domain_signature is not None:
        signature_name, signed_names = domain_signature
        expected_signature = "".join(
            f"{hashlib.sha256(payloads[name]).hexdigest()}  {name}\n"
            for name in sorted(signed_names)
        ).encode("utf-8")
        if payloads[signature_name] != expected_signature:
            raise ValueError(f"{domain} signature content mismatch")
    physical_aggregate = hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()
    expected_domain = ((expected_output.get("domains") or {}).get(domain) or {})
    if physical_aggregate != expected_domain.get("physical_aggregate_sha256"):
        raise ValueError(f"{domain} physical aggregate sha256 mismatch")
    return payloads


def _verify_declared_packet_domains(
    packet_root: Path,
    expected_output: Mapping[str, Any],
    domain_names: tuple[str, ...],
    *,
    enumerate_packet_root: bool,
) -> dict[str, dict[str, bytes]]:
    packet_fd = _open_directory_no_follow(packet_root, label="packet root")
    domain_payloads: dict[str, dict[str, bytes]] = {}
    try:
        if enumerate_packet_root:
            actual_domains = set(os.listdir(packet_fd))
            if actual_domains != set(domain_names):
                raise ValueError(
                    "packet root surface mismatch: "
                    f"unexpected={sorted(actual_domains - set(domain_names))} "
                    f"missing={sorted(set(domain_names) - actual_domains)}"
                )
        for domain in domain_names:
            domain_fd = _open_child_directory_no_follow(
                packet_fd, domain, label=f"packet domain {domain}"
            )
            try:
                domain_payloads[domain] = _verify_declared_domain(
                    domain_fd, domain, expected_output
                )
            finally:
                os.close(domain_fd)
    finally:
        os.close(packet_fd)
    return domain_payloads


def _safe_declaration_name(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"unsafe trainer declaration path: {value!r}")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or len(parsed.parts) != 1 or parsed.name != value or value.startswith("."):
        raise ValueError(f"unsafe trainer declaration path: {value!r}")
    return value


def _verified_trainer_read_surface_from_fds(
    domain_fds: Mapping[str, int],
) -> tuple[dict[str, bytes], dict[str, bytes]]:
    control_fd = domain_fds[CONTROL_PLANE_ROOT_NAME]
    trainer_fd = domain_fds[TRAINER_ROOT_NAME]
    sealed_fd = domain_fds["sealed_validation"]
    sealed_names = set(os.listdir(sealed_fd))
    expected_sealed_names = set(SEALED_VALIDATION_ARTIFACT_ROLES)
    if sealed_names != expected_sealed_names:
        raise ValueError(
            "sealed_validation physical surface mismatch: "
            f"unexpected={sorted(sealed_names - expected_sealed_names)} "
            f"missing={sorted(expected_sealed_names - sealed_names)}"
        )
    sealed_payloads = {
        name: _read_regular_at(
            sealed_fd, name, label="sealed_validation artifact"
        )
        for name in sorted(expected_sealed_names)
    }
    sealed_signature = sealed_payloads["sealed-validation-manifest.sha256"]
    expected_sealed_signature = "".join(
        f"{hashlib.sha256(sealed_payloads[name]).hexdigest()}  {name}\n"
        for name in sorted(SEALED_VALIDATION_ARTIFACT_NAMES)
    ).encode("utf-8")
    if sealed_signature != expected_sealed_signature:
        raise ValueError("sealed_validation signature content mismatch")
    control_names = set(os.listdir(control_fd))
    if control_names != CONTROL_PLANE_ARTIFACT_NAMES:
        raise ValueError(
            "control-plane surface mismatch: "
            f"unexpected={sorted(control_names - CONTROL_PLANE_ARTIFACT_NAMES)} "
            f"missing={sorted(CONTROL_PLANE_ARTIFACT_NAMES - control_names)}"
        )
    manifest_bytes = _read_regular_at(
        control_fd,
        "trainer_input_manifest.json",
        label="control-plane manifest",
    )
    signature_bytes = _read_regular_at(
        control_fd, "artifact-manifest.sha256", label="trainer signature"
    )
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("malformed trainer input manifest") from exc
    if manifest.get("schema_version") != "form_only_v1_trainer_input_manifest_v2":
        raise ValueError("unexpected trainer input manifest schema")
    if manifest.get("trainer_root") != TRAINER_ROOT_NAME:
        raise ValueError("trainer root declaration mismatch")
    declarations = manifest.get("allowed_files")
    if not isinstance(declarations, list):
        raise ValueError("trainer declarations must be a list")
    declared: dict[str, Mapping[str, Any]] = {}
    for declaration in declarations:
        if not isinstance(declaration, dict):
            raise ValueError("malformed trainer declaration")
        name = _safe_declaration_name(declaration.get("path"))
        if name in declared:
            raise ValueError(f"duplicate trainer declaration: {name}")
        declared[name] = declaration
    expected_names = set(TRAINER_ARTIFACT_ROLES)
    if set(declared) != expected_names:
        raise ValueError(
            "declared trainer read surface mismatch: "
            f"unexpected={sorted(set(declared) - expected_names)} "
            f"missing={sorted(expected_names - set(declared))}"
        )
    if manifest.get("declared_file_count") != len(declared):
        raise ValueError("declared trainer file count mismatch")
    actual_names = set(os.listdir(trainer_fd))
    if actual_names != set(declared):
        raise ValueError(
            "trainer read surface mismatch: "
            f"unexpected={sorted(actual_names - set(declared))} "
            f"missing={sorted(set(declared) - actual_names)}"
        )
    payloads: dict[str, bytes] = {}
    signature_rows = []
    for name in sorted(declared):
        declaration = declared[name]
        if declaration.get("type") != "regular_file":
            raise ValueError(f"trainer type declaration mismatch: {name}")
        if declaration.get("role") != TRAINER_ARTIFACT_ROLES[name]:
            raise ValueError(f"trainer role declaration mismatch: {name}")
        payload = _read_regular_at(trainer_fd, name, label="trainer artifact")
        expected_bytes = declaration.get("bytes")
        if type(expected_bytes) is not int or expected_bytes < 1:
            raise ValueError(f"invalid trainer byte length declaration: {name}")
        if len(payload) != expected_bytes:
            raise ValueError(f"trainer artifact byte length mismatch: {name}")
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if actual_sha256 != declaration.get("sha256"):
            raise ValueError(f"trainer artifact sha256 mismatch: {name}")
        signature_rows.append(f"{actual_sha256}  {name}\n")
        payloads[name] = payload
    expected_signature = "".join(signature_rows).encode("utf-8")
    if signature_bytes != expected_signature:
        raise ValueError("trainer artifact signature content mismatch")
    signature = manifest.get("artifact_manifest")
    if not isinstance(signature, dict):
        raise ValueError("missing trainer artifact signature declaration")
    if signature.get("path") != "artifact-manifest.sha256":
        raise ValueError("trainer artifact signature path mismatch")
    if (
        signature.get("type") != "regular_file"
        or signature.get("role") != "CONTROL_INTEGRITY_SIGNATURE"
    ):
        raise ValueError("trainer artifact signature type or role mismatch")
    if signature.get("bytes") != len(signature_bytes):
        raise ValueError("trainer artifact signature byte length mismatch")
    if signature.get("sha256") != hashlib.sha256(signature_bytes).hexdigest():
        raise ValueError("trainer artifact signature sha256 mismatch")
    aggregate = hashlib.sha256(signature_bytes).hexdigest()
    if signature.get("trainer_aggregate_sha256") != aggregate:
        raise ValueError("trainer aggregate sha256 mismatch")
    return payloads, {
        "trainer_input_manifest.json": manifest_bytes,
        "artifact-manifest.sha256": signature_bytes,
    }


def _verified_trainer_read_surface(
    packet_root: Path,
) -> tuple[dict[str, bytes], dict[str, bytes]]:
    packet_fd = _open_directory_no_follow(packet_root, label="packet root")
    domain_fds: dict[str, int] = {}
    try:
        # The authoritative loader deliberately does not enumerate or open the
        # optional diagnostic domain. Complete four-domain validation is a
        # separate phase.
        for name in AUTHORITATIVE_DOMAIN_ROOT_NAMES:
            domain_fds[name] = _open_child_directory_no_follow(
                packet_fd, name, label=f"packet domain {name}"
            )
        return _verified_trainer_read_surface_from_fds(domain_fds)
    finally:
        for directory_fd in reversed(tuple(domain_fds.values())):
            os.close(directory_fd)
        os.close(packet_fd)


def validate_trainer_read_surface(packet_root: Path) -> None:
    """Validate the complete trainer surface without exposing trainer bytes."""
    _verified_trainer_read_surface(packet_root)


def load_verified_trainer_inputs(
    packet_root: Path, reproducibility_contract_path: Path
) -> dict[str, bytes]:
    """Load trainer bytes only after the Git-tracked control trust root is verified."""
    contract = load_reproducibility_contract(
        reproducibility_contract_path, include_diagnostic=False
    )
    expected_control = (
        ((contract.get("expected_output") or {}).get("domains") or {}).get("control_plane")
        or {}
    )
    expected_files = expected_control.get("artifact_files")
    if not isinstance(expected_files, dict) or set(expected_files) != CONTROL_PLANE_ARTIFACT_NAMES:
        raise ValueError("reproducibility contract has no exact control-plane trust root")
    payloads, control_payloads = _verified_trainer_read_surface(packet_root)
    declared_payloads = _verify_declared_packet_domains(
        packet_root,
        contract.get("expected_output") or {},
        AUTHORITATIVE_DOMAIN_ROOT_NAMES,
        enumerate_packet_root=False,
    )
    if (
        declared_payloads[TRAINER_ROOT_NAME] != payloads
        or declared_payloads[CONTROL_PLANE_ROOT_NAME] != control_payloads
    ):
        raise ValueError("authoritative packet changed during verified load")
    actual_files = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in sorted(control_payloads.items())
    }
    if actual_files != expected_files:
        raise ValueError("control-plane artifact hash mismatch")
    control_text = "".join(
        f"{actual_files[name]}  {name}\n" for name in sorted(actual_files)
    )
    if hashlib.sha256(control_text.encode("utf-8")).hexdigest() != expected_control.get("aggregate_sha256"):
        raise ValueError("control-plane aggregate hash mismatch")
    return payloads


def validate_complete_packet(
    packet_root: Path, reproducibility_contract_path: Path
) -> None:
    """Validate exact declared physical sets in all four packet domains."""
    contract = load_reproducibility_contract(
        reproducibility_contract_path, include_diagnostic=True
    )
    expected_domains = ((contract.get("expected_output") or {}).get("domains") or {})
    if set(expected_domains) != set(PACKET_DOMAIN_ROOT_NAMES):
        raise ValueError("reproducibility contract domain set is not exact")
    _verify_declared_packet_domains(
        packet_root,
        contract.get("expected_output") or {},
        PACKET_DOMAIN_ROOT_NAMES,
        enumerate_packet_root=True,
    )
    _verified_trainer_read_surface(packet_root)


def verify_expected_output(
    summary: Mapping[str, Any],
    manifests: Mapping[str, Mapping[str, Any]],
    expected: Mapping[str, Any],
    *,
    include_diagnostic: bool = True,
) -> None:
    if not expected:
        raise ValueError("reproducibility contract has no expected_output")
    authoritative_counts = {
        "candidate_races": summary["development"]["candidate_race_count"],
        "candidate_runners": summary["development"]["candidate_runner_count"],
        "included_races": summary["development"]["included_race_count"],
        "included_runners": summary["development"]["included_runner_count"],
        "sidecar_only_exclusions": summary["development"]["sidecar_only_runner_exclusion_count"],
        "out_of_time_races": summary["out_of_time"]["included_race_count"],
        "out_of_time_runners": summary["out_of_time"]["included_runner_count"],
    }
    if authoritative_counts != expected.get("authoritative_counts"):
        raise ValueError(f"expected authoritative count mismatch: {authoritative_counts}")
    if include_diagnostic:
        diagnostic_counts = {
            "overlap_races": summary["reconciliation"]["overlap_race_count"],
            "overlap_runners": summary["reconciliation"]["overlap_runner_count"],
            "history_differences": summary["reconciliation"]["history_discrepancy_count"],
            "recency_differences": summary["reconciliation"]["recency_discrepancy_count"],
            "grade_differences": summary["reconciliation"]["grade_discrepancy_count"],
            "unexplained_differences": summary["reconciliation"]["unexplained_mismatch_count"],
        }
        if diagnostic_counts != expected.get("diagnostic_counts"):
            raise ValueError(f"expected diagnostic count mismatch: {diagnostic_counts}")
    for domain, manifest in manifests.items():
        expected_domain = (expected.get("domains") or {}).get(domain) or {}
        actual_files = {row["path"]: row["sha256"] for row in manifest["files"]}
        if actual_files != expected_domain.get("artifact_files"):
            raise ValueError(f"expected {domain} artifact hash mismatch")
        if manifest["aggregate_sha256"] != expected_domain.get("aggregate_sha256"):
            raise ValueError(f"expected {domain} aggregate hash mismatch")


def _prepare_empty_packet_output(
    output_dir: Path,
    domain_names: tuple[str, ...] = AUTHORITATIVE_DOMAIN_ROOT_NAMES,
) -> _PacketOutputScope:
    """Bind empty authoritative domains for their complete write lifetime."""
    return _open_packet_output_scope(
        output_dir,
        root_names=domain_names,
        required_existing=(),
        creatable=set(domain_names),
        empty_domains=set(domain_names),
        writable_domains=set(domain_names),
        allow_create_root=True,
    )


def _open_packet_output_scope(
    output_dir: Path,
    *,
    root_names: tuple[str, ...],
    required_existing: tuple[str, ...],
    creatable: set[str],
    empty_domains: set[str],
    writable_domains: set[str],
    allow_create_root: bool,
) -> _PacketOutputScope:
    if not output_dir.is_absolute() or ".." in output_dir.parts:
        raise ValueError(f"packet root must be absolute and traversal-free: {output_dir}")
    packet_name = _output_component(output_dir.name, label="packet root")
    parent_fd = _open_directory_no_follow(
        output_dir.parent, label="packet root parent"
    )
    packet_fd: int | None = None
    created_root = False
    created_domains: set[str] = set()
    domain_fds: dict[str, int] = {}
    scope: _PacketOutputScope | None = None
    try:
        try:
            os.stat(packet_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            if not allow_create_root:
                raise ValueError(f"missing packet root: {output_dir}")
            try:
                os.mkdir(packet_name, mode=0o700, dir_fd=parent_fd)
            except OSError as mkdir_exc:
                raise ValueError(
                    f"cannot safely create packet root: {output_dir}"
                ) from mkdir_exc
            created_root = True
        packet_fd = _open_child_directory_no_follow(
            parent_fd, packet_name, label="packet root"
        )
        entries = set(os.listdir(packet_fd))
        unexpected = entries - set(root_names)
        if unexpected:
            raise ValueError(
                f"packet root has unexpected pre-existing entries: {sorted(unexpected)}"
            )
        missing_required = set(required_existing) - entries
        if missing_required:
            raise ValueError(
                "packet root is missing required authoritative domains: "
                f"{sorted(missing_required)}"
            )
        for name in root_names:
            if name not in entries:
                if name not in creatable:
                    raise ValueError(f"packet root is missing required domain: {name}")
                try:
                    os.mkdir(name, mode=0o700, dir_fd=packet_fd)
                except OSError as exc:
                    raise ValueError(f"cannot safely create packet domain: {name}") from exc
                created_domains.add(name)
            domain_fds[name] = _open_child_directory_no_follow(
                packet_fd, name, label=f"packet domain {name}"
            )
            if name in empty_domains:
                domain_entries = os.listdir(domain_fds[name])
                if domain_entries:
                    raise ValueError(
                        f"packet domain has unexpected pre-existing entries: {name}: "
                        f"{sorted(domain_entries)}"
                    )
        scope = _PacketOutputScope(
            parent_fd=parent_fd,
            packet_fd=packet_fd,
            packet_name=packet_name,
            root_names=root_names,
            created_root=created_root,
            created_domains=created_domains,
        )
        added: set[str] = set()
        try:
            for name in root_names:
                scope.add_domain(
                    name,
                    domain_fds[name],
                    writable=name in writable_domains,
                )
                added.add(name)
            scope.verify_for_close()
        except BaseException as exc:
            for name, directory_fd in domain_fds.items():
                if name not in added:
                    os.close(directory_fd)
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise
        return scope
    except BaseException:
        if scope is None:
            for directory_fd in reversed(tuple(domain_fds.values())):
                try:
                    os.close(directory_fd)
                except OSError:
                    pass
            if packet_fd is not None:
                for name in reversed(root_names):
                    if name in created_domains:
                        try:
                            os.rmdir(name, dir_fd=packet_fd)
                        except OSError:
                            pass
                os.close(packet_fd)
                if created_root:
                    try:
                        os.rmdir(packet_name, dir_fd=parent_fd)
                    except OSError:
                        pass
            os.close(parent_fd)
        raise


def _prepare_empty_diagnostic_output(packet_root: Path) -> _PacketOutputScope:
    """Independently bind all domains before the first diagnostic byte."""
    diagnostic_name = "non_authoritative_diagnostic"
    return _open_packet_output_scope(
        packet_root,
        root_names=PACKET_DOMAIN_ROOT_NAMES,
        required_existing=AUTHORITATIVE_DOMAIN_ROOT_NAMES,
        creatable={diagnostic_name},
        empty_domains={diagnostic_name},
        writable_domains={diagnostic_name},
        allow_create_root=False,
    )


def _build_authoritative_packet(
    eligibility_dir: Path,
    training_dir: Path,
    evidence_roots: list[Path],
    output_dir: Path,
    out_of_time_freeze_dir: Path | None,
    reproducibility_contract_path: Path,
    *,
    enforce_expected_output: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    reproducibility = load_reproducibility_contract(
        reproducibility_contract_path, include_diagnostic=False
    )
    with _prepare_empty_packet_output(output_dir) as domains:
        trainer_domain = domains[TRAINER_ROOT_NAME]
        control_domain = domains[CONTROL_PLANE_ROOT_NAME]
        sealed_domain = domains["sealed_validation"]
        loaded = load_development_sources(
            eligibility_dir, training_dir, reproducibility
        )
        development_summary, _selected = build_development_packet(
            loaded, trainer_domain, sealed_domain
        )
        out_of_time_summary = build_out_of_time_manifest(
            evidence_roots,
            trainer_domain,
            reproducibility,
            out_of_time_freeze_dir,
            sealed_domain,
        )
        write_feature_contract(trainer_domain)
        write_market_coverage(trainer_domain)
        validate_trainer_visible_artifacts(trainer_domain)
        trainer_manifest = write_artifact_manifest(
            trainer_domain,
            TRAINER_ARTIFACT_NAMES,
            manifest_domain=control_domain,
        )
        write_trainer_input_manifest(
            trainer_domain, control_domain, trainer_manifest
        )
        sealed_payload_manifest = write_artifact_manifest(
            sealed_domain,
            SEALED_VALIDATION_ARTIFACT_NAMES,
            filename="sealed-validation-manifest.sha256",
        )
        sealed_manifest = artifact_manifest_records(
            sealed_domain, set(SEALED_VALIDATION_ARTIFACT_ROLES)
        )
        sealed_manifest.pop("text")
        sealed_manifest["payload_aggregate_sha256"] = sealed_payload_manifest[
            "aggregate_sha256"
        ]
        _verified_trainer_read_surface_from_fds(
            {
                name: domains[name].fd
                for name in AUTHORITATIVE_DOMAIN_ROOT_NAMES
            }
        )
        control_manifest = artifact_manifest_records(
            control_domain, CONTROL_PLANE_ARTIFACT_NAMES
        )
        control_manifest.pop("text")
        manifests = {
            "trainer": trainer_manifest,
            "control_plane": control_manifest,
            "sealed_validation": sealed_manifest,
        }
        summary = {
            "phase": "AUTHORITATIVE",
            "development": development_summary,
            "out_of_time": out_of_time_summary,
            "artifact_manifest": trainer_manifest,
            "domain_manifests": manifests,
        }
        if enforce_expected_output:
            expected_output = reproducibility.get("expected_output") or {}
            verify_expected_output(
                summary,
                manifests,
                expected_output,
                include_diagnostic=False,
            )
            for name in AUTHORITATIVE_DOMAIN_ROOT_NAMES:
                _verify_declared_domain(
                    domains[name].fd, name, expected_output
                )
        return summary, loaded, reproducibility


def build_authoritative_packet(
    eligibility_dir: Path,
    training_dir: Path,
    evidence_roots: list[Path],
    output_dir: Path,
    out_of_time_freeze_dir: Path | None,
    reproducibility_contract_path: Path,
    *,
    enforce_expected_output: bool = True,
) -> dict[str, Any]:
    """Build the authoritative packet without touching diagnostic inputs or outputs."""
    summary, _loaded, _reproducibility = _build_authoritative_packet(
        eligibility_dir,
        training_dir,
        evidence_roots,
        output_dir,
        out_of_time_freeze_dir,
        reproducibility_contract_path,
        enforce_expected_output=enforce_expected_output,
    )
    return summary


def build_all(
    eligibility_dir: Path,
    training_dir: Path,
    evidence_roots: list[Path],
    output_dir: Path,
    out_of_time_freeze_dir: Path | None,
    reproducibility_contract_path: Path,
    *,
    enforce_expected_output: bool = True,
) -> dict[str, Any]:
    authoritative, loaded, reproducibility = _build_authoritative_packet(
        eligibility_dir,
        training_dir,
        evidence_roots,
        output_dir,
        out_of_time_freeze_dir,
        reproducibility_contract_path,
        enforce_expected_output=enforce_expected_output,
    )
    with _prepare_empty_diagnostic_output(output_dir) as domains:
        authoritative_before = _snapshot_bound_domains(
            domains, authoritative["domain_manifests"]
        )
        diagnostic_loaded = load_diagnostic_sources(loaded, reproducibility)
        trainer_manifest = authoritative["artifact_manifest"]
        trainer_before_diagnostics = trainer_manifest["aggregate_sha256"]
        diagnostic_domain = domains["non_authoritative_diagnostic"]
        reconciliation_summary = build_overlap_reconciliation(
            diagnostic_loaded, diagnostic_domain
        )
        diagnostic_payload_manifest = write_artifact_manifest(
            diagnostic_domain,
            DIAGNOSTIC_ARTIFACT_NAMES,
            filename="non-authoritative-diagnostic-manifest.sha256",
        )
        diagnostic_manifest = artifact_manifest_records(
            diagnostic_domain, set(DIAGNOSTIC_ARTIFACT_ROLES)
        )
        diagnostic_manifest.pop("text")
        diagnostic_manifest["payload_aggregate_sha256"] = (
            diagnostic_payload_manifest["aggregate_sha256"]
        )
        authoritative_after = _snapshot_bound_domains(
            domains, authoritative["domain_manifests"]
        )
        if authoritative_before != authoritative_after:
            raise ValueError("diagnostic construction changed authoritative packet bytes")
        trainer_after_diagnostics = artifact_manifest_records(
            domains[TRAINER_ROOT_NAME], TRAINER_ARTIFACT_NAMES
        )["aggregate_sha256"]
        if trainer_before_diagnostics != trainer_after_diagnostics:
            raise ValueError("diagnostic construction changed trainer artifacts")
        _verified_trainer_read_surface_from_fds(
            {
                name: domains[name].fd
                for name in AUTHORITATIVE_DOMAIN_ROOT_NAMES
            }
        )
        manifests = {
            **authoritative["domain_manifests"],
            "non_authoritative_diagnostic": diagnostic_manifest,
        }
        summary = {
            **authoritative,
            "phase": "AUTHORITATIVE_PLUS_OPTIONAL_DIAGNOSTIC",
            "reconciliation": reconciliation_summary,
            "domain_manifests": manifests,
            "diagnostic_isolation": {
                "trainer_aggregate_before": trainer_before_diagnostics,
                "trainer_aggregate_after": trainer_after_diagnostics,
                "byte_identical": True,
            },
        }
        if enforce_expected_output:
            expected_output = reproducibility.get("expected_output") or {}
            verify_expected_output(
                summary, manifests, expected_output, include_diagnostic=True
            )
            for name in PACKET_DOMAIN_ROOT_NAMES:
                _verify_declared_domain(
                    domains[name].fd, name, expected_output
                )
        return summary


def build_optional_diagnostics(
    packet_root: Path,
    reproducibility_contract_path: Path,
    *,
    enforce_expected_output: bool = True,
) -> dict[str, Any]:
    """Build optional diagnostics from a verified authoritative packet read-only."""
    reproducibility = load_reproducibility_contract(
        reproducibility_contract_path, include_diagnostic=True
    )
    expected_output = reproducibility.get("expected_output") or {}
    with _prepare_empty_diagnostic_output(packet_root) as domains:
        authoritative_before = {
            domain: _verify_declared_domain(
                domains[domain].fd, domain, expected_output
            )
            for domain in AUTHORITATIVE_DOMAIN_ROOT_NAMES
        }
        loaded = _diagnostic_context_from_authoritative_packet(
            authoritative_before, reproducibility
        )
        diagnostic_domain = domains["non_authoritative_diagnostic"]
        reconciliation = build_overlap_reconciliation(loaded, diagnostic_domain)
        diagnostic_payload_manifest = write_artifact_manifest(
            diagnostic_domain,
            DIAGNOSTIC_ARTIFACT_NAMES,
            filename="non-authoritative-diagnostic-manifest.sha256",
        )
        diagnostic_manifest = artifact_manifest_records(
            diagnostic_domain, set(DIAGNOSTIC_ARTIFACT_ROLES)
        )
        diagnostic_manifest.pop("text")
        diagnostic_manifest["payload_aggregate_sha256"] = (
            diagnostic_payload_manifest["aggregate_sha256"]
        )
        authoritative_after = {
            domain: _verify_declared_domain(
                domains[domain].fd, domain, expected_output
            )
            for domain in AUTHORITATIVE_DOMAIN_ROOT_NAMES
        }
        before_hashes = {
            domain: {
                name: hashlib.sha256(payload).hexdigest()
                for name, payload in sorted(files.items())
            }
            for domain, files in authoritative_before.items()
        }
        after_hashes = {
            domain: {
                name: hashlib.sha256(payload).hexdigest()
                for name, payload in sorted(files.items())
            }
            for domain, files in authoritative_after.items()
        }
        if before_hashes != after_hashes:
            raise ValueError(
                "diagnostic construction changed authoritative packet bytes"
            )
        if enforce_expected_output:
            actual_counts = {
                "overlap_races": reconciliation["overlap_race_count"],
                "overlap_runners": reconciliation["overlap_runner_count"],
                "history_differences": reconciliation["history_discrepancy_count"],
                "recency_differences": reconciliation["recency_discrepancy_count"],
                "grade_differences": reconciliation["grade_discrepancy_count"],
                "unexplained_differences": reconciliation[
                    "unexplained_mismatch_count"
                ],
            }
            if actual_counts != expected_output.get("diagnostic_counts"):
                raise ValueError(f"expected diagnostic count mismatch: {actual_counts}")
            expected_domain = (
                (expected_output.get("domains") or {}).get(
                    "non_authoritative_diagnostic"
                )
                or {}
            )
            actual_files = {
                row["path"]: row["sha256"]
                for row in diagnostic_manifest["files"]
            }
            if actual_files != expected_domain.get("artifact_files"):
                raise ValueError(
                    "expected non_authoritative_diagnostic artifact hash mismatch"
                )
            if diagnostic_manifest["aggregate_sha256"] != expected_domain.get(
                "aggregate_sha256"
            ):
                raise ValueError(
                    "expected non_authoritative_diagnostic aggregate hash mismatch"
                )
        for domain in PACKET_DOMAIN_ROOT_NAMES:
            _verify_declared_domain(
                domains[domain].fd, domain, expected_output
            )
        _verified_trainer_read_surface_from_fds(
            {
                name: domains[name].fd
                for name in AUTHORITATIVE_DOMAIN_ROOT_NAMES
            }
        )
        return {
            "phase": "NON_AUTHORITATIVE_DIAGNOSTIC",
            "authority": "NON_AUTHORITATIVE_DIAGNOSTIC",
            "reconciliation": reconciliation,
            "domain_manifest": diagnostic_manifest,
            "authoritative_hashes_before": before_hashes,
            "authoritative_hashes_after": after_hashes,
            "authoritative_bytes_identical": True,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("authoritative", "diagnostic", "all"),
        default="authoritative",
    )
    parser.add_argument("--eligibility-dir", type=Path)
    parser.add_argument("--training-dir", type=Path)
    parser.add_argument("--evidence-root", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--out-of-time-freeze-dir", type=Path)
    parser.add_argument("--reproducibility-contract", type=Path, required=True)
    args = parser.parse_args()
    if args.phase in {"authoritative", "all"}:
        missing = [
            name
            for name in (
                "eligibility_dir",
                "training_dir",
                "out_of_time_freeze_dir",
            )
            if getattr(args, name) is None
        ]
        if missing:
            parser.error(
                "authoritative construction requires "
                + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            )
    return args


def main() -> int:
    args = parse_args()
    if args.phase == "authoritative":
        summary = build_authoritative_packet(
            args.eligibility_dir,
            args.training_dir,
            args.evidence_root,
            args.output_dir,
            args.out_of_time_freeze_dir,
            args.reproducibility_contract,
        )
    elif args.phase == "diagnostic":
        summary = build_optional_diagnostics(
            args.output_dir, args.reproducibility_contract
        )
    else:
        summary = build_all(
            args.eligibility_dir,
            args.training_dir,
            args.evidence_root,
            args.output_dir,
            args.out_of_time_freeze_dir,
            args.reproducibility_contract,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
