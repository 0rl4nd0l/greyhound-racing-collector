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

FORBIDDEN_FEATURE_TOKENS = {
    "actual_win", "finish_position", "open", "low", "high", "sp", "odds",
    "result", "winner", "dog_name", "dog_identity", "speed", "sectional",
    "time", "opponent", "prize", "weather", "trainer",
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


def file_record(path: Path, *, declared_sha256: str | None = None) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": declared_sha256 or sha256_path(path),
        "bytes": path.stat().st_size,
    }


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


def parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=MELBOURNE)
    return parsed.astimezone(MELBOURNE)


def capture_timestamp(metadata: Mapping[str, Any]) -> datetime:
    for key in ("metadata_captured_at", "created_at", "capture_timestamp", "captured_at"):
        if metadata.get(key):
            return parse_timestamp(str(metadata[key]))
    raise ValueError("card sidecar has no capture timestamp")


def sidecar_jump_timestamp(metadata: Mapping[str, Any], race_id: str) -> datetime:
    info = metadata.get("race_info") or {}
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
    return f"{race_id}|box:{int(float(str(box)))}|dog:{dog_token(dog_name)}"


def row_id(race_id: str, box: Any, dog_name: Any) -> str:
    payload = canonical_runner_id(race_id, box, dog_name).encode("utf-8")
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
    mixed_match = re.fullmatch(r"(?:MIXED)?([1-8](?:/[1-8]){1,2})", compact)
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
        key = (
            history_date.isoformat(), raw.get("TRACK"), raw.get("DIST"), raw.get("G"),
            raw.get("PLC"), raw.get("BOX"), raw.get("MGN"),
        )
        if key in seen:
            rejected.append(("EXACT_DUPLICATE_HISTORY", raw))
            continue
        seen.add(key)
        accepted.append({
            "date": history_date,
            "venue": canonical_venue(raw.get("TRACK")),
            "distance": safe_int(raw.get("DIST")),
            "grade": canonical_grade(raw.get("G")),
            "finish": safe_int(raw.get("PLC")),
            "margin": safe_float(raw.get("MGN")),
        })
    accepted.sort(key=lambda row: (row["date"], row["venue"], row["distance"] or -1), reverse=True)
    for row in accepted[HISTORY_CAP:]:
        rejected.append(("HISTORY_CAP_20", row))
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


def load_development_sources(eligibility_dir: Path, training_dir: Path) -> dict[str, Any]:
    race_path = eligibility_dir / "historical_win_eligibility_races_v1.csv"
    runner_path = eligibility_dir / "historical_win_eligibility_runners_v1.csv"
    provenance_path = eligibility_dir / "historical_win_tier_a_race_provenance_v1.json"
    manifest_path = eligibility_dir / "historical_win_eligibility_manifest_v1.json"
    training_path = training_dir / "thedogs_training_rows_v1.csv"
    required = [race_path, runner_path, provenance_path, manifest_path, training_path]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

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

    if EXCLUDED_PUBLISHED_RACE_ID in published_rows:
        raise ValueError("published exclusion was incorrectly selected")

    candidate_ids = sorted(set(provenance).union(published_rows))
    candidate_runners: dict[str, list[dict[str, Any]]] = {}
    source_options: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for race_id in candidate_ids:
        if race_id in tier_a_runners:
            candidate_runners[race_id] = sorted(tier_a_runners[race_id], key=lambda row: (row["box"], dog_token(row["dog_name"])))
            item = provenance[race_id]
            csv_path = Path(item["source_csv_path"])
            sidecar_path = Path(item["sidecar_path"])
            metadata = validate_sidecar(csv_path, sidecar_path)
            source_options[race_id].append({
                "source_class": "OFFICIAL_RACE_PAGE_TIER_A",
                "precedence": 0,
                "csv_path": csv_path,
                "sidecar_path": sidecar_path,
                "csv_sha256": item["source_csv_sha256"],
                "sidecar_sha256": item["sidecar_sha256"],
                "capture": capture_timestamp(metadata),
                "jump": parse_timestamp(item["jump_timestamp"]),
                "metadata": metadata,
                "label_provenance_class": "OFFICIAL_RACE_PAGE_TIER_A",
                "label_source_paths": [item["official_race_artifact_path"], item["official_runner_artifact_path"]],
                "label_source_sha256": [item["official_race_artifact_sha256"], item["official_runner_artifact_sha256"]],
                "label_urls": item.get("official_urls") or [],
            })
        if race_id in published_rows:
            first = published_rows[race_id][0]
            csv_path = Path(first["source_csv_path"])
            sidecar_path = Path(str(csv_path) + ".metadata.json")
            metadata = validate_sidecar(csv_path, sidecar_path)
            option = {
                "source_class": "THEDOGS_PUBLISHED_HISTORY_NOT_TIER_A",
                "precedence": 1,
                "csv_path": csv_path,
                "sidecar_path": sidecar_path,
                "csv_sha256": first["source_csv_sha256"],
                "sidecar_sha256": sha256_path(sidecar_path),
                "capture": capture_timestamp(metadata),
                "jump": parse_timestamp(first["race_timestamp_utc"]),
                "metadata": metadata,
                "label_provenance_class": "THEDOGS_PUBLISHED_HISTORY_NOT_TIER_A",
                "label_source_paths": [str(training_path)],
                "label_source_sha256": [sha256_path(training_path)],
                "label_urls": [first["odds_url"]] if first.get("odds_url") else [],
            }
            source_options[race_id].append(option)
            published_runner_list = sorted([
                {"box": int(row["box_number"]), "dog_name": row["csv_dog_name"], "source_runner_id": row["runner_id"]}
                for row in published_rows[race_id]
            ], key=lambda row: (row["box"], dog_token(row["dog_name"])))
            if race_id not in candidate_runners:
                candidate_runners[race_id] = published_runner_list
            else:
                left = [(row["box"], dog_token(row["dog_name"])) for row in candidate_runners[race_id]]
                right = [(row["box"], dog_token(row["dog_name"])) for row in published_runner_list]
                if left != right:
                    raise ValueError(f"overlap runner identity mismatch: {race_id}")

    return {
        "candidate_ids": candidate_ids,
        "candidate_runners": candidate_runners,
        "source_options": source_options,
        "provenance": provenance,
        "published_rows": published_rows,
        "input_files": required,
        "training_path": training_path,
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
                    "entity_type": "runner", "entity_id": canonical_runner_id(race_id, runner["box"], runner["dog_name"]),
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

    for race_id in sorted(selected):
        option = selected[race_id]
        race_date = date.fromisoformat(race_id.rsplit(" - ", 1)[-1])
        if race_date > DEVELOPMENT_END:
            raise ValueError(f"development race after cutoff: {race_id}")
        venue, distance, grade, sidecar_field_size = target_metadata(option, race_id)
        runners = candidate_runners[race_id]
        field_size = len(runners)
        blocks = parse_form_blocks(option["csv_path"])
        if sidecar_field_size < field_size:
            raise ValueError(f"sidecar field smaller than frozen runners: {race_id}")
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
            rid = canonical_runner_id(race_id, runner["box"], runner["dog_name"])
            opaque = row_id(race_id, runner["box"], runner["dog_name"])
            runner_rows.append({
                "row_id": opaque, "runner_id": rid, "race_id": race_id, "box_number": runner["box"],
                "dog_identity_sha256": hashlib.sha256(token.encode()).hexdigest(),
                "source_runner_id": runner["source_runner_id"], "label_provenance_class": option["label_provenance_class"],
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
    runner_rows.sort(key=lambda row: (row["race_id"], row["box_number"], row["runner_id"]))
    feature_rows.sort(key=lambda row: (row["race_id"], row["box_number"], row["row_id"]))
    exclusion_rows.sort(key=lambda row: (row["race_id"], row["entity_type"], row["entity_id"], row["reason"]))
    source_rows.sort(key=lambda row: (row["race_id"], row["source_class"], row["role"], row["path"]))

    stable_csv(output_dir / "development_races.csv", list(race_rows[0]), race_rows)
    stable_csv(output_dir / "development_runners.csv", list(runner_rows[0]), runner_rows)
    stable_csv(output_dir / "development_features.csv", list(feature_rows[0]), feature_rows)
    exclusion_fields = ["entity_type", "entity_id", "race_id", "reason", "history_date"]
    stable_csv(output_dir / "development_exclusions.csv", exclusion_fields, exclusion_rows)
    stable_csv(output_dir / "development_source_inventory.csv", list(source_rows[0]), source_rows)

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
        "accepted_target_or_post_target_history_count": 0,
        "rejected_target_or_post_target_history_count": leakage["rejected_target_or_post_target_history"],
        "outcome_feature_count": 0, "market_feature_count": 0, "dog_identity_feature_count": 0,
        "feature_columns": feature_columns,
    }
    if summary["candidate_race_count"] != 1267 or summary["candidate_runner_count"] != 8914:
        raise ValueError(f"candidate freeze mismatch: {summary}")
    if summary["included_race_count"] != 917 or summary["included_runner_count"] != 6456:
        raise ValueError(f"T60 freeze mismatch: {summary}")
    stable_json(output_dir / "development_manifest.json", {
        "schema_version": "form_only_v1_development_manifest_v1",
        "status": "ACQUISITION_ONLY_NO_MODEL_FIT",
        "development_end": DEVELOPMENT_END.isoformat(),
        "label_values_included": False,
        "card_requirement": "capture_timestamp <= canonical_jump_timestamp - 60 minutes",
        "source_precedence": ["OFFICIAL_RACE_PAGE_TIER_A", "THEDOGS_PUBLISHED_HISTORY_NOT_TIER_A"],
        "summary": summary,
        "bound_inputs": [file_record(path) for path in loaded["input_files"]],
    })
    return summary, selected


def load_shadow_feature_rows(paths: Iterable[str]) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_path in sorted(set(paths)):
        payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"unexpected shadow feature payload: {raw_path}")
        for row in payload:
            rows[(row["race_id"], dog_token(row["dog_name"]))] = row
    return rows


def build_overlap_reconciliation(loaded: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    overlap_ids = sorted(set(loaded["provenance"]).intersection(loaded["published_rows"]))
    feature_paths = [
        path for race_id in overlap_ids
        for path in loaded["provenance"][race_id]["feature_source_paths"]
    ]
    shadow_rows = load_shadow_feature_rows(feature_paths)
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
            history, _ = accepted_history(blocks[token], target_date)
            canonical_history_count = len(history)
            canonical_recency = (target_date - history[0]["date"]).days if history else None
            canonical_target_grade = canonical_grade(published_row["target_grade"])
            shadow_history = safe_int(shadow.get("prior_start_count"))
            shadow_recency = safe_int(shadow.get("days_since_last_start"))
            shadow_grade = str(shadow.get("target_grade_normalized") or "")
            history_diff = shadow_history != canonical_history_count
            recency_diff = shadow_recency != canonical_recency
            grade_diff = shadow_grade != published_row["target_grade"]
            if not history_diff:
                history_cause = "MATCH"
            elif (shadow_history or 0) == 0 and canonical_history_count > 0:
                history_cause = "SHADOW_GLOBAL_HISTORY_LOOKUP_MISS"
            else:
                history_cause = "SHADOW_GLOBAL_HISTORY_COUNT_DIFFERENCE"
            if not recency_diff:
                recency_cause = "MATCH"
            elif shadow_recency is None and canonical_recency is not None:
                recency_cause = "SHADOW_GLOBAL_RECENCY_LOOKUP_MISS"
            else:
                recency_cause = "SHADOW_GLOBAL_LATEST_DATE_DIFFERENCE"
            grade_cause = "MATCH" if not grade_diff else "BUILDER_TARGET_GRADE_NORMALIZATION_DIFFERENCE"
            causes[history_cause] += 1
            causes[recency_cause] += 1
            causes[grade_cause] += 1
            rows.append({
                "runner_id": canonical_runner_id(race_id, published_row["box_number"], published_row["csv_dog_name"]),
                "race_id": race_id, "box_number": int(published_row["box_number"]),
                "raw_card_sha256": tier_a["source_csv_sha256"], "raw_cards_byte_identical": 1,
                "shadow_prior_start_count": "" if shadow_history is None else shadow_history,
                "canonical_prior_start_count": canonical_history_count,
                "history_discrepancy": int(history_diff), "history_cause": history_cause,
                "shadow_days_since_last_start": "" if shadow_recency is None else shadow_recency,
                "canonical_days_since_last_start": "" if canonical_recency is None else canonical_recency,
                "recency_discrepancy": int(recency_diff), "recency_cause": recency_cause,
                "shadow_target_grade": shadow_grade, "published_target_grade": published_row["target_grade"],
                "canonical_target_grade": canonical_target_grade,
                "grade_discrepancy": int(grade_diff), "grade_cause": grade_cause,
                "unexplained_mismatch": 0,
            })
    rows.sort(key=lambda row: (row["race_id"], row["box_number"], row["runner_id"]))
    stable_csv(output_dir / "overlap_reconciliation.csv", list(rows[0]), rows)
    summary = {
        "overlap_race_count": len(overlap_ids), "overlap_runner_count": len(rows),
        "byte_identical_raw_card_race_count": raw_identical_races,
        "history_discrepancy_count": sum(row["history_discrepancy"] for row in rows),
        "recency_discrepancy_count": sum(row["recency_discrepancy"] for row in rows),
        "grade_discrepancy_count": sum(row["grade_discrepancy"] for row in rows),
        "unexplained_mismatch_count": sum(row["unexplained_mismatch"] for row in rows),
        "cause_counts": dict(sorted(causes.items())),
        "canonical_rule": "rebuild from byte-identical raw pre-race card; never select a legacy builder value",
    }
    expected = (len(rows), summary["history_discrepancy_count"], summary["recency_discrepancy_count"], summary["grade_discrepancy_count"])
    if expected != (530, 486, 527, 219) or summary["unexplained_mismatch_count"]:
        raise ValueError(f"overlap reconciliation mismatch: {summary}")
    stable_json(output_dir / "reconciliation_summary.json", summary)
    return summary


def parse_race_id_from_sidecar(path: Path) -> tuple[str, date] | None:
    match = re.fullmatch(r"(Race \d+ - .+ - (\d{4}-\d{2}-\d{2}))\.csv\.metadata\.json", path.name)
    if not match:
        return None
    return match.group(1), date.fromisoformat(match.group(2))


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
                    capture = capture_timestamp(metadata)
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


def load_frozen_out_of_time_sources(freeze_dir: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    inventory_path = freeze_dir / "out_of_time_source_inventory.csv"
    exclusions_path = freeze_dir / "out_of_time_exclusions.csv"
    if not inventory_path.is_file() or not exclusions_path.is_file():
        raise FileNotFoundError(f"incomplete out-of-time freeze: {freeze_dir}")
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    with inventory_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["race_id"]][row["role"]] = row
    selected: dict[str, dict[str, Any]] = {}
    for race_id, roles in sorted(grouped.items()):
        csv_row = roles["raw_pre_race_card"]
        sidecar_row = roles["raw_pre_race_sidecar"]
        csv_path = Path(csv_row["path"])
        sidecar_path = Path(sidecar_row["path"])
        metadata = validate_sidecar(csv_path, sidecar_path)
        if sha256_path(sidecar_path) != sidecar_row["sha256"]:
            raise ValueError(f"frozen sidecar hash mismatch: {sidecar_path}")
        if metadata["content_sha256"] != csv_row["sha256"]:
            raise ValueError(f"frozen card hash mismatch: {csv_path}")
        selected[race_id] = {
            "race_id": race_id, "race_date": date.fromisoformat(race_id.rsplit(" - ", 1)[-1]),
            "csv_path": csv_path, "sidecar_path": sidecar_path, "metadata": metadata,
            "capture": parse_timestamp(csv_row["capture_timestamp"]),
            "jump": parse_timestamp(csv_row["jump_timestamp"]),
            "csv_sha256": csv_row["sha256"], "sidecar_sha256": sidecar_row["sha256"],
        }
    with exclusions_path.open(encoding="utf-8", newline="") as handle:
        exclusions = list(csv.DictReader(handle))
    return selected, exclusions


def build_out_of_time_manifest(
    evidence_roots: list[Path], output_dir: Path, freeze_dir: Path | None = None
) -> dict[str, Any]:
    if freeze_dir is None:
        selected, exclusions = scan_out_of_time_sources(evidence_roots)
        discovery_mode = "LIVE_READ_ONLY_DISCOVERY_THEN_SEALED"
    else:
        selected, exclusions = load_frozen_out_of_time_sources(freeze_dir)
        discovery_mode = "HASH_BOUND_FROZEN_DISCOVERY_INPUT"
    race_rows: list[dict[str, Any]] = []
    runner_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for race_id in sorted(selected):
        option = selected[race_id]
        venue, distance, grade, field_size = target_metadata(option, race_id)
        participants = (option["metadata"].get("runner_completeness") or {}).get("participants") or []
        race_rows.append({
            "race_id": race_id, "race_date": option["race_date"].isoformat(), "target_venue": venue,
            "target_distance_m": distance or "", "target_grade": grade, "field_size": field_size,
            "card_capture_timestamp": option["capture"].isoformat(), "jump_timestamp": option["jump"].isoformat(),
            "card_lead_minutes": fmt_number((option["jump"] - option["capture"]).total_seconds() / 60),
            "status": "OUTCOME_UNOPENED_OUT_OF_TIME",
        })
        for participant in sorted(participants, key=lambda row: (int(row["box_number"]), dog_token(row["dog_name"]))):
            box = int(participant["box_number"])
            token = dog_token(participant["dog_name"])
            runner_rows.append({
                "row_id": row_id(race_id, box, participant["dog_name"]), "race_id": race_id,
                "box_number": box, "dog_identity_sha256": hashlib.sha256(token.encode()).hexdigest(),
                "status": "OUTCOME_UNOPENED_OUT_OF_TIME",
            })
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
        "schema_version": "form_only_v1_out_of_time_manifest_v1",
        "status": "OUTCOME_UNOPENED_OUT_OF_TIME", "outcomes_opened": False,
        "window_start": OUT_OF_TIME_START.isoformat(), "window_end": OUT_OF_TIME_END.isoformat(),
        "included_race_count": len(race_rows), "included_runner_count": len(runner_rows),
        "excluded_source_count": len(exclusions), "source_roots": [str(path.resolve()) for path in evidence_roots],
        "selection_rule": "freshest leakage-safe complete contemporaneous raw card available by T60",
        "discovery_mode": discovery_mode,
    }
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
        "identity_policy": "dog identity is used only to bind raw history and construct an opaque row key",
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


def build_all(
    eligibility_dir: Path,
    training_dir: Path,
    evidence_roots: list[Path],
    output_dir: Path,
    out_of_time_freeze_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_development_sources(eligibility_dir, training_dir)
    development_summary, _selected = build_development_packet(loaded, output_dir)
    reconciliation_summary = build_overlap_reconciliation(loaded, output_dir)
    out_of_time_summary = build_out_of_time_manifest(evidence_roots, output_dir, out_of_time_freeze_dir)
    write_feature_contract(output_dir)
    write_market_coverage(output_dir)
    artifact_manifest = write_artifact_manifest(output_dir)
    return {
        "development": development_summary,
        "reconciliation": reconciliation_summary,
        "out_of_time": out_of_time_summary,
        "artifact_manifest": artifact_manifest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eligibility-dir", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--out-of-time-freeze-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_all(
        args.eligibility_dir, args.training_dir, args.evidence_root, args.output_dir,
        args.out_of_time_freeze_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
