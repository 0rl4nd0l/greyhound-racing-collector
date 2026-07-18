#!/usr/bin/env python3
"""Build deterministic odds-free FORM_ONLY_V1 acquisition packets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping
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

GENERATED_ARTIFACT_NAMES = {
    "development_exclusions.csv", "development_features.csv", "development_manifest.json",
    "development_races.csv", "development_runners.csv", "development_source_inventory.csv",
    "feature_contract.json", "market_coverage.json", "out_of_time_exclusions.csv",
    "out_of_time_manifest.json", "out_of_time_races.csv", "out_of_time_runners.csv",
    "out_of_time_source_inventory.csv", "overlap_reconciliation.csv", "reconciliation_summary.json",
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


def verify_file_record(
    path: Path, *, expected_sha256: str, expected_bytes: int | str | None = None
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = file_record(path)
    if actual["sha256"] != expected_sha256:
        raise ValueError(f"source hash mismatch: {path}")
    if expected_bytes not in (None, "") and actual["bytes"] != int(expected_bytes):
        raise ValueError(f"source byte mismatch: {path}")
    return actual


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_set_digest(records: Iterable[Mapping[str, Any]]) -> str:
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        normalized = {
            "role": str(record["role"]),
            "path": str(Path(str(record["path"])).resolve()),
            "sha256": str(record["sha256"]),
            "bytes": int(record["bytes"]),
        }
        key = (normalized["role"], normalized["path"])
        previous = unique.get(key)
        if previous is not None and previous != normalized:
            raise ValueError(f"conflicting source declaration: {key}")
        unique[key] = normalized
    return canonical_digest([unique[key] for key in sorted(unique)])


def load_reproducibility_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "form_only_v1_reproducibility_v1":
        raise ValueError(f"unsupported reproducibility contract: {path}")
    if not isinstance(payload.get("trusted_inputs"), dict):
        raise ValueError(f"reproducibility contract has no trusted_inputs: {path}")
    payload["construction_contract_sha256"] = canonical_digest(payload["trusted_inputs"])
    return payload


def stable_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_csv(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


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


def row_id(race_id: str, box: Any, dog_name: Any, *, scope: str = DEVELOPMENT_SCOPE) -> str:
    payload = f"FORM_ONLY_V1|{scope}|{canonical_runner_id(race_id, box, dog_name)}".encode("utf-8")
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
        text = str(value or "").strip()
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


def parse_form_blocks(path: Path) -> dict[str, list[dict[str, Any]]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise ValueError(f"empty form CSV: {path}")
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


def parse_card_target_roster(path: Path) -> list[tuple[int, str]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise ValueError(f"empty form CSV: {path}")
    delimiter = "|" if lines[0].count("|") > lines[0].count(",") else ","
    participants: list[dict[str, Any]] = []
    for raw in csv.DictReader(lines, delimiter=delimiter):
        name = str(raw.get("Dog Name") or "").strip().strip('"')
        if not name:
            continue
        match = re.fullmatch(r"\s*(\d+)\.\s*(.+?)\s*", name)
        if not match:
            raise ValueError(f"target runner lacks verified box prefix in {path}: {name}")
        participants.append({"box": match.group(1), "name": match.group(2)})
    return canonical_roster(participants, box_key="box", name_key="name", source=str(path))


def verify_card_sidecar_roster(
    csv_path: Path, metadata: Mapping[str, Any], *, race_id: str
) -> list[tuple[int, str]]:
    card = parse_card_target_roster(csv_path)
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


def validate_sidecar(csv_path: Path, sidecar_path: Path) -> dict[str, Any]:
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if metadata.get("metadata_is_leakage_safe") is not True:
        raise ValueError(f"unsafe metadata sidecar: {sidecar_path}")
    completeness = metadata.get("runner_completeness") or {}
    if completeness.get("status") != "COMPLETE":
        raise ValueError(f"incomplete runner sidecar: {sidecar_path}")
    if not csv_path.is_file():
        raise ValueError(f"missing source CSV: {csv_path}")
    actual_sha = sha256_path(csv_path)
    if actual_sha != metadata.get("content_sha256"):
        raise ValueError(f"source CSV hash mismatch: {csv_path}")
    if csv_path.stat().st_size != metadata.get("content_length"):
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
            path, expected_sha256=expected["sha256"], expected_bytes=expected.get("bytes")
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
    for race_id in candidate_ids:
        if race_id in tier_a_runners:
            candidate_runners[race_id] = sorted(tier_a_runners[race_id], key=lambda row: (row["box"], dog_token(row["dog_name"])))
            canonical_roster(
                candidate_runners[race_id], box_key="box", name_key="dog_name", source=f"tier-a:{race_id}"
            )
            item = provenance[race_id]
            csv_path = Path(item["source_csv_path"])
            sidecar_path = Path(item["sidecar_path"])
            metadata = validate_sidecar(csv_path, sidecar_path)
            csv_record = verify_file_record(csv_path, expected_sha256=item["source_csv_sha256"])
            sidecar_record = verify_file_record(sidecar_path, expected_sha256=item["sidecar_sha256"])
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
                "sidecar_path": sidecar_path,
                "csv_sha256": csv_record["sha256"],
                "sidecar_sha256": sidecar_record["sha256"],
                "capture": capture_timestamp(metadata),
                "jump": parse_timestamp(item["jump_timestamp"]),
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
            metadata = validate_sidecar(csv_path, sidecar_path)
            csv_record = verify_file_record(csv_path, expected_sha256=first["source_csv_sha256"])
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
            if race_id not in candidate_runners:
                candidate_runners[race_id] = published_runner_list
            else:
                left = [(row["box"], dog_token(row["dog_name"])) for row in candidate_runners[race_id]]
                right = [(row["box"], dog_token(row["dog_name"])) for row in published_runner_list]
                if left != right:
                    raise ValueError(f"overlap runner identity mismatch: {race_id}")

    overlap_ids = sorted(set(provenance).intersection(published_rows))
    shadow_source_by_race: dict[str, dict[str, Any]] = {}
    for race_id in overlap_ids:
        paths = provenance[race_id].get("feature_source_paths") or []
        digests = provenance[race_id].get("feature_source_sha256") or []
        if len(paths) != 1 or len(digests) != 1:
            raise ValueError(f"overlap race requires one justified shadow source: {race_id}")
        record = verify_file_record(Path(paths[0]), expected_sha256=digests[0])
        shadow_source_by_race[race_id] = record
        trusted_source_records.append({"role": "shadow_reconciliation_source", **record})

    source_digest = source_set_digest(trusted_source_records)
    if source_digest != development_contract.get("source_set_sha256"):
        raise ValueError(f"development source-set binding mismatch: {source_digest}")
    source_record_count = len({(row["role"], row["path"]) for row in trusted_source_records})
    if source_record_count != int(development_contract.get("source_record_count", -1)):
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
        "shadow_source_by_race": shadow_source_by_race,
        "training_path": training_path,
        "construction_contract_sha256": reproducibility["construction_contract_sha256"],
    }


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
        eligible.sort(key=lambda option: (option["precedence"], str(option["csv_path"])))
        selected[race_id] = eligible[0]
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
    sidecar = verify_card_sidecar_roster(option["csv_path"], option["metadata"], race_id=race_id)
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
            "evidence_path": evidence["path"],
            "evidence_sha256": evidence["sha256"],
        }
        for box, token in sorted(sidecar_only)
    ]
    return active, exclusions


def build_development_packet(
    loaded: Mapping[str, Any], output_dir: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    selected, excluded = select_development_sources(loaded)
    candidate_runners = loaded["candidate_runners"]
    race_rows: list[dict[str, Any]] = []
    runner_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
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
                source_rows.append({
                    "race_id": race_id, "selection_status": status, "source_class": option["source_class"],
                    "role": role, "path": str(path.resolve()), "sha256": digest,
                    "bytes": path.stat().st_size, "capture_timestamp": option["capture"].isoformat(),
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
    for race_id, record in sorted(loaded["shadow_source_by_race"].items()):
        source_rows.append({
            "race_id": race_id,
            "selection_status": "RECONCILIATION_ONLY",
            "source_class": "LEGACY_SHADOW_RECONCILIATION_ONLY",
            "role": "shadow_reconciliation_source",
            "path": record["path"],
            "sha256": record["sha256"],
            "bytes": record["bytes"],
            "capture_timestamp": "",
            "jump_timestamp": "",
            "lead_minutes": "",
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
        blocks = parse_form_blocks(option["csv_path"])
        label_paths = option["label_source_paths"]
        label_hashes = option["label_source_sha256"]
        race_rows.append({
            "race_id": race_id, "race_date": race_date.isoformat(), "target_venue": venue,
            "race_number": int(re.search(r"Race (\d+)", race_id).group(1)),
            "target_distance_m": distance or "", "target_grade": grade, "field_size": field_size,
            "card_capture_timestamp": option["capture"].isoformat(), "jump_timestamp": option["jump"].isoformat(),
            "card_lead_minutes": fmt_number((option["jump"] - option["capture"]).total_seconds() / 60),
            "card_source_class": option["source_class"], "card_source_path": str(option["csv_path"].resolve()),
            "card_source_sha256": option["csv_sha256"], "card_source_bytes": option["csv_path"].stat().st_size,
            "card_sidecar_path": str(option["sidecar_path"].resolve()), "card_sidecar_sha256": option["sidecar_sha256"],
            "card_sidecar_bytes": option["sidecar_path"].stat().st_size,
            "label_provenance_class": option["label_provenance_class"],
            "label_source_paths": "|".join(label_paths), "label_source_sha256": "|".join(label_hashes),
            "label_urls": "|".join(option["label_urls"]), "label_value_included": 0,
        })
        for runner in runners:
            token = dog_token(runner["dog_name"])
            if token not in blocks:
                raise ValueError(f"runner absent from raw card: {race_id} {token}")
            history, rejected = accepted_history(blocks[token], race_date)
            opaque = row_id(race_id, runner["box"], runner["dog_name"], scope=DEVELOPMENT_SCOPE)
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

    stable_csv(output_dir / "development_races.csv", list(race_rows[0]), race_rows)
    stable_csv(output_dir / "development_runners.csv", list(runner_rows[0]), runner_rows)
    stable_csv(output_dir / "development_features.csv", list(feature_rows[0]), feature_rows)
    exclusion_fields = [
        "entity_type", "entity_id", "race_id", "reason", "history_date",
        "evidence_path", "evidence_sha256",
    ]
    stable_csv(output_dir / "development_exclusions.csv", exclusion_fields, exclusion_rows)
    stable_csv(output_dir / "development_source_inventory.csv", list(source_rows[0]), source_rows)
    source_inventory_record = file_record(output_dir / "development_source_inventory.csv")

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
    stable_json(output_dir / "development_manifest.json", {
        "schema_version": "form_only_v1_development_manifest_v2",
        "status": "ACQUISITION_ONLY_NO_MODEL_FIT",
        "development_end": DEVELOPMENT_END.isoformat(),
        "label_values_included": False,
        "card_requirement": "capture_timestamp <= canonical_jump_timestamp - 60 minutes",
        "source_precedence": ["OFFICIAL_RACE_PAGE_TIER_A", "THEDOGS_PUBLISHED_HISTORY_NOT_TIER_A"],
        "summary": summary,
        "construction_contract_sha256": loaded["construction_contract_sha256"],
        "bound_inputs": loaded["top_input_records"],
        "trusted_source_set": {
            "record_count": len({
                (row["role"], row["path"]) for row in loaded["trusted_source_records"]
            }),
            "aggregate_sha256": loaded["trusted_source_set_sha256"],
            "inventory_path": "development_source_inventory.csv",
            "inventory_sha256": source_inventory_record["sha256"],
            "inventory_bytes": source_inventory_record["bytes"],
        },
    })
    return summary, selected


def load_shadow_feature_rows(
    sources: Mapping[str, Mapping[str, Any]]
) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for expected_race_id, record in sorted(sources.items()):
        path = Path(record["path"])
        verify_file_record(
            path, expected_sha256=str(record["sha256"]), expected_bytes=record.get("bytes")
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
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
            rows[key] = {**row, "_bound_source_path": str(path.resolve())}
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


def build_overlap_reconciliation(loaded: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
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
        blocks = parse_form_blocks(Path(tier_a["source_csv_path"]))
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
    stable_csv(output_dir / "overlap_reconciliation.csv", list(rows[0]), rows)
    summary = {
        "overlap_race_count": len(overlap_ids), "overlap_runner_count": len(rows),
        "byte_identical_raw_card_race_count": raw_identical_races,
        "history_discrepancy_count": sum(row["history_discrepancy"] for row in rows),
        "recency_discrepancy_count": sum(row["recency_discrepancy"] for row in rows),
        "grade_discrepancy_count": sum(row["grade_discrepancy"] for row in rows),
        "unexplained_mismatch_count": sum(row["unexplained_mismatch"] for row in rows),
        "cause_counts": dict(sorted(causes.items())),
        "bound_shadow_source_count": len(loaded["shadow_source_by_race"]),
        "bound_shadow_source_set_sha256": source_set_digest(
            {"role": "shadow_reconciliation_source", **record}
            for record in loaded["shadow_source_by_race"].values()
        ),
        "canonical_rule": "rebuild from byte-identical raw pre-race card; never select a legacy builder value",
    }
    stable_json(output_dir / "reconciliation_summary.json", summary)
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
    if info.get("date") and date.fromisoformat(str(info["date"])) != parsed[1]:
        raise ValueError(f"sidecar race date mismatch: {race_id}")
    if info.get("venue") and canonical_venue(info["venue"]) != canonical_venue(match.group(2)):
        raise ValueError(f"sidecar race venue mismatch: {race_id}")
    declared_number = safe_int(info.get("race_number"))
    if declared_number is not None and declared_number != int(match.group(1)):
        raise ValueError(f"sidecar race number mismatch: {race_id}")
    return parsed[1]


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
                    continue
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
                    metadata = validate_sidecar(csv_path, sidecar_path)
                    validate_race_identity(race_id, sidecar_path, metadata)
                    verify_card_sidecar_roster(csv_path, metadata, race_id=race_id)
                    capture = capture_timestamp(metadata, require_timezone=True)
                    jump = sidecar_jump_timestamp(metadata, race_id)
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
                    "sidecar_path": sidecar_path, "metadata": metadata, "capture": capture, "jump": jump,
                    "csv_sha256": metadata["content_sha256"], "sidecar_sha256": sha256_path(sidecar_path),
                })
    selected: dict[str, dict[str, Any]] = {}
    for race_id, options in by_race.items():
        options.sort(key=lambda item: (item["capture"], str(item["sidecar_path"])), reverse=True)
        selected[race_id] = options[0]
        for option in options[1:]:
            exclusions.append({
                "race_id": race_id, "race_date": option["race_date"].isoformat(),
                "source_path": str(option["sidecar_path"].resolve()), "source_sha256": option["sidecar_sha256"],
                "source_bytes": option["sidecar_path"].stat().st_size, "reason": "VALID_DUPLICATE_NOT_FRESHEST_BY_T60",
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
            path, expected_sha256=expected["sha256"], expected_bytes=expected.get("bytes")
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
        metadata = validate_sidecar(csv_path, sidecar_path)
        csv_record = verify_file_record(
            csv_path, expected_sha256=csv_row["sha256"], expected_bytes=csv_row.get("bytes")
        )
        sidecar_record = verify_file_record(
            sidecar_path,
            expected_sha256=sidecar_row["sha256"],
            expected_bytes=sidecar_row.get("bytes"),
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
        verify_card_sidecar_roster(csv_path, metadata, race_id=race_id)
        selected[race_id] = {
            "race_id": race_id, "race_date": race_date,
            "csv_path": csv_path, "sidecar_path": sidecar_path, "metadata": metadata,
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
    output_dir: Path,
    reproducibility: Mapping[str, Any],
    freeze_dir: Path | None = None,
) -> dict[str, Any]:
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
    for race_id in sorted(selected):
        option = selected[race_id]
        venue, distance, grade, field_size = target_metadata(option, race_id)
        participants = (option["metadata"].get("runner_completeness") or {}).get("participants") or []
        verified_roster = verify_card_sidecar_roster(
            option["csv_path"], option["metadata"], race_id=race_id
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
        if len(verified_roster) != len(participants):
            raise ValueError(f"out-of-time roster count mismatch: {race_id}")
        for role, path, digest in (
            ("raw_pre_race_card", option["csv_path"], option["csv_sha256"]),
            ("raw_pre_race_sidecar", option["sidecar_path"], option["sidecar_sha256"]),
        ):
            source_rows.append({
                "race_id": race_id, "role": role, "path": str(path.resolve()), "sha256": digest,
                "bytes": path.stat().st_size, "capture_timestamp": option["capture"].isoformat(),
                "jump_timestamp": option["jump"].isoformat(), "status": "OUTCOME_UNOPENED_OUT_OF_TIME",
            })
    stable_csv(output_dir / "out_of_time_races.csv", list(race_rows[0]) if race_rows else ["race_id"], race_rows)
    stable_csv(output_dir / "out_of_time_runners.csv", list(runner_rows[0]) if runner_rows else ["row_id"], runner_rows)
    stable_csv(output_dir / "out_of_time_source_inventory.csv", list(source_rows[0]) if source_rows else ["race_id"], source_rows)
    exclusion_fields = ["race_id", "race_date", "source_path", "source_sha256", "source_bytes", "reason"]
    stable_csv(output_dir / "out_of_time_exclusions.csv", exclusion_fields, exclusions)
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
            "files": freeze_binding["freeze_records"],
        }
    summary["selected_source_set_sha256"] = source_set_digest(
        {
            "role": role,
            **file_record(path),
        }
        for option in selected.values()
        for role, path in (
            ("out_of_time_card", option["csv_path"]),
            ("out_of_time_sidecar", option["sidecar_path"]),
        )
    )
    stable_json(output_dir / "out_of_time_manifest.json", summary)
    return summary


def write_feature_contract(output_dir: Path) -> None:
    stable_json(output_dir / "feature_contract.json", {
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
            "dog tokens exist only inside the sealed card/sidecar/label alignment boundary; "
            "emitted row IDs are SHA-256 opaque and domain-separated by split and race"
        ),
        "deferred": ["speed", "times", "sectionals", "opponent_strength", "high_dimensional_interactions"],
        "forbidden": sorted(FORBIDDEN_FEATURE_TOKENS),
        "missingness": "blank numeric value plus explicit family missingness flag; no silent zero fill",
        "grade_aliases": "embedded canonical_grade function; unknown values retained as stable uppercase tokens",
        "venue_aliases_sha256": hashlib.sha256(
            json.dumps(VENUE_ALIASES, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    })


def write_market_coverage(output_dir: Path) -> None:
    stable_json(output_dir / "market_coverage.json", {
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


def validate_trainer_visible_artifacts(output_dir: Path) -> None:
    identity_scopes: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for name in sorted(GENERATED_ARTIFACT_NAMES):
        path = output_dir / name
        text = path.read_text(encoding="utf-8")
        if "|dog:" in text:
            raise ValueError(f"sealed dog alignment key leaked into artifact: {name}")
        if path.suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                forbidden = set(reader.fieldnames or []).intersection(FORBIDDEN_ARTIFACT_FIELDS)
                if forbidden:
                    raise ValueError(f"identity-bearing fields in {name}: {sorted(forbidden)}")
                split = OUT_OF_TIME_SCOPE if name.startswith("out_of_time_") else DEVELOPMENT_SCOPE
                for row in reader:
                    race_id = row.get("race_id") or ""
                    for key in ("row_id", "entity_id"):
                        value = row.get(key) or ""
                        if re.fullmatch(r"[0-9a-f]{64}", value):
                            identity_scopes[value].add((split, race_id))
        elif path.suffix == ".json":
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


def write_artifact_manifest(output_dir: Path) -> dict[str, Any]:
    rows = []
    for name in sorted(GENERATED_ARTIFACT_NAMES):
        path = output_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing generated artifact: {path}")
        rows.append({"path": path.name, "sha256": sha256_path(path), "bytes": path.stat().st_size})
    text = "".join(f"{row['sha256']}  {row['path']}\n" for row in rows)
    (output_dir / "artifact-manifest.sha256").write_text(text, encoding="utf-8")
    return {"files": rows, "aggregate_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def verify_expected_output(
    summary: Mapping[str, Any], artifact_manifest: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    if not expected:
        raise ValueError("reproducibility contract has no expected_output")
    actual_counts = {
        "candidate_races": summary["development"]["candidate_race_count"],
        "candidate_runners": summary["development"]["candidate_runner_count"],
        "included_races": summary["development"]["included_race_count"],
        "included_runners": summary["development"]["included_runner_count"],
        "sidecar_only_exclusions": summary["development"]["sidecar_only_runner_exclusion_count"],
        "overlap_races": summary["reconciliation"]["overlap_race_count"],
        "overlap_runners": summary["reconciliation"]["overlap_runner_count"],
        "history_differences": summary["reconciliation"]["history_discrepancy_count"],
        "recency_differences": summary["reconciliation"]["recency_discrepancy_count"],
        "grade_differences": summary["reconciliation"]["grade_discrepancy_count"],
        "unexplained_differences": summary["reconciliation"]["unexplained_mismatch_count"],
        "out_of_time_races": summary["out_of_time"]["included_race_count"],
        "out_of_time_runners": summary["out_of_time"]["included_runner_count"],
    }
    if actual_counts != expected.get("counts"):
        raise ValueError(f"expected output count mismatch: {actual_counts}")
    actual_files = {row["path"]: row["sha256"] for row in artifact_manifest["files"]}
    if actual_files != expected.get("artifact_files"):
        raise ValueError("expected output artifact hash mismatch")
    if artifact_manifest["aggregate_sha256"] != expected.get("artifact_manifest_sha256"):
        raise ValueError("expected output aggregate hash mismatch")


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
    output_dir.mkdir(parents=True, exist_ok=True)
    reproducibility = load_reproducibility_contract(reproducibility_contract_path)
    loaded = load_development_sources(eligibility_dir, training_dir, reproducibility)
    development_summary, _selected = build_development_packet(loaded, output_dir)
    reconciliation_summary = build_overlap_reconciliation(loaded, output_dir)
    out_of_time_summary = build_out_of_time_manifest(
        evidence_roots, output_dir, reproducibility, out_of_time_freeze_dir
    )
    write_feature_contract(output_dir)
    write_market_coverage(output_dir)
    validate_trainer_visible_artifacts(output_dir)
    artifact_manifest = write_artifact_manifest(output_dir)
    summary = {
        "development": development_summary,
        "reconciliation": reconciliation_summary,
        "out_of_time": out_of_time_summary,
        "artifact_manifest": artifact_manifest,
    }
    if enforce_expected_output:
        verify_expected_output(summary, artifact_manifest, reproducibility.get("expected_output") or {})
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eligibility-dir", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--out-of-time-freeze-dir", type=Path, required=True)
    parser.add_argument("--reproducibility-contract", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_all(
        args.eligibility_dir, args.training_dir, args.evidence_root, args.output_dir,
        args.out_of_time_freeze_dir, args.reproducibility_contract,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
