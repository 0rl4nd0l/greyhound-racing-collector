#!/usr/bin/env python3
"""
Official-first race results ingestion for upcoming-race CSVs.

Default operator mode should be --dry-run. Database label writes require either
--write-labels-approved or APPROVE_RESULT_LABEL_WRITE=true.

Source order:
1. TheDogs official race page, when accessible and parseable.
2. Sportsbet results page top-four order as fallback.

Sportsbet only exposes top-four box order on the rendered results page, so
fallback rows are marked as partial_sportsbet_results rather than complete.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional
from urllib.parse import urljoin, urlparse

import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.race_lifecycle import RESULTED, UPCOMING_NOT_JUMPED, classify_race_record
from utils.runner_completeness import (
    MIN_COMPLETE_RUNNERS,
    RunnerRow,
    analyze_csv_runner_completeness,
    analyze_runner_rows,
    participants_from_runner_rows,
)

try:
    from accuracy_program.snapshots import assert_no_result_fields
except Exception:  # pragma: no cover - keeps the ingestion CLI usable in partial envs
    assert_no_result_fields = None

TARGET_TABLE = "dog_race_data"
PARTIAL_SPORTSBET_RESULTS = "partial_sportsbet_results"
SPORTSBET_CATEGORY_TEMPLATE = (
    "https://www.sportsbet.com.au/results/{date}/racing/greyhound-racing-4"
)
THEDOGS_BASE = "https://www.thedogs.com.au"
THEDOGS_PUBLIC_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def env_flag_enabled(name: str) -> bool:
    return str(os.environ.get(name) or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
        "approved",
    }


def result_label_write_approved(args: argparse.Namespace) -> dict:
    cli_approved = bool(getattr(args, "write_labels_approved", False))
    env_approved = env_flag_enabled("APPROVE_RESULT_LABEL_WRITE")
    approved = cli_approved or env_approved
    return {
        "approved": approved,
        "status": "approved" if approved else "not_approved",
        "sources": {
            "cli_write_labels_approved": cli_approved,
            "env_APPROVE_RESULT_LABEL_WRITE": env_approved,
        },
        "required_for": "official_result_label_writes",
    }


class _SeleniumByFallback:
    TAG_NAME = "tag name"
    CSS_SELECTOR = "css selector"


class _StatelessPublicHttpClient:
    def get(self, url: str, **kwargs):
        kwargs.pop("cookies", None)
        with requests.Session() as session:
            session.trust_env = False
            session.cookies.clear()
            return session.get(url, cookies={}, **kwargs)


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
    "DARW": "darwin",
    "GAWL": "gawler",
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


def thedogs_slug_from_race_url(url: str) -> Optional[str]:
    try:
        parts = [part for part in urlparse(str(url or "")).path.split("/") if part]
    except Exception:
        return None
    try:
        racing_index = parts.index("racing")
    except ValueError:
        return None
    if len(parts) <= racing_index + 1:
        return None
    slug = parts[racing_index + 1].strip().lower()
    return slug or None


def thedogs_result_urls_from_race_url(url: str) -> List[str]:
    parsed = urlparse(str(url or ""))
    if not parsed.scheme or not parsed.netloc:
        return []
    base = parsed._replace(query="", fragment="").geturl().rstrip("/")
    if not base:
        return []
    if base.endswith("/results"):
        return [f"{base}?trial=false", base]
    return [f"{base}/results?trial=false", f"{base}/results", f"{base}?trial=false", base]


def parse_participants_from_csv(csv_path: Path) -> List[dict]:
    return participants_from_runner_rows(
        [
            RunnerRow(
                box_number=int(participant["box_number"]),
                dog_name=str(participant["dog_name"]),
            )
            for participant in analyze_csv_runner_completeness(csv_path).participants
        ]
    )


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


def _strict_ordinal_to_position(value: str) -> Optional[int]:
    match = re.match(r"^\s*([1-9]|10)(?:st|nd|rd|th)\s*$", str(value or ""), re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _rug_box_from_markup(markup: str) -> Optional[int]:
    match = re.search(r"\bname=[\"']rug_(\d{1,2})[\"']", str(markup or ""), re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _clean_official_runner_name(value: str) -> Optional[str]:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = re.sub(r"^\s*\d{1,2}\s*[\.\):-]\s*", "", text)
    text = re.sub(r"\s+\d{1,2}\.\d{2}\s+T:\s+.*$", "", text)
    text = re.sub(r"\s+T:\s+.*$", "", text)
    return text or None


def _result_identity_name(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


PROMOTED_RESERVE_RESULT_BOXES = {9, 10}
PROMOTED_RESERVE_NON_NAME_SUFFIXES = frozenset({"NBT"})


def _reserve_from_box(value: object) -> Optional[int]:
    match = re.search(r"\(\s*from\s+box\s+(\d{1,2})\s*\)\s*$", str(value or ""), re.I)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _clean_promoted_reserve_name(value: object) -> Optional[str]:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = re.sub(
        r"\s+\d{1,2}\.\d{2}\s*(?=\(\s*from\s+box\s+\d{1,2}\s*\)\s*$)",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"\s*\(\s*from\s+box\s+\d{1,2}\s*\)\s*$", "", text, flags=re.I)
    for suffix in PROMOTED_RESERVE_NON_NAME_SUFFIXES:
        text = re.sub(rf"\s+{re.escape(suffix)}\s*$", "", text, flags=re.I)
    return _clean_official_runner_name(text)


def remap_promoted_reserve_runner_rows(
    official_rows: List[dict],
    participants: List[dict],
) -> dict:
    """Map promoted reserve result rows back to verified frozen boxes.

    TheDogs can report a promoted reserve under rug 9/10 while including text
    such as "(from box 8)". Remapping is allowed only when that target box is in
    the frozen participants and the cleaned official name exactly matches.
    """

    participant_by_box: Dict[int, str] = {}
    for participant in participants or []:
        try:
            box = int(participant.get("box_number"))
        except Exception:
            continue
        name = str(participant.get("dog_name") or "").strip()
        if box and name:
            participant_by_box[box] = name

    participant_boxes = set(participant_by_box)
    remap_by_original: Dict[int, dict] = {}
    target_counts: Dict[int, int] = {}
    rejected: List[dict] = []
    for row in official_rows or []:
        try:
            original_box = int(row.get("box_number") or 0)
        except Exception:
            continue
        target_box = _reserve_from_box(row.get("dog_name"))
        if (
            not original_box
            or target_box is None
            or target_box not in participant_boxes
            or original_box not in PROMOTED_RESERVE_RESULT_BOXES
            or original_box in participant_boxes
            or row.get("finish_position") is None
        ):
            continue
        cleaned_name = _clean_promoted_reserve_name(row.get("dog_name"))
        expected_name = participant_by_box.get(target_box)
        if _result_identity_name(cleaned_name) != _result_identity_name(expected_name):
            rejected.append(
                {
                    "original_box_number": original_box,
                    "target_box_number": target_box,
                    "official_dog_name": row.get("dog_name"),
                    "cleaned_official_dog_name": cleaned_name,
                    "expected_dog_name": expected_name,
                    "reason": "promoted_reserve_name_mismatch",
                }
            )
            continue
        target_counts[target_box] = target_counts.get(target_box, 0) + 1
        remap_by_original[original_box] = {
            "original_box_number": original_box,
            "target_box_number": target_box,
            "official_dog_name": row.get("dog_name"),
            "cleaned_official_dog_name": cleaned_name,
            "expected_dog_name": expected_name,
            "source": "thedogs_result_from_box_note",
        }

    ambiguous_targets = {box for box, count in target_counts.items() if count > 1}
    for original_box, remap in list(remap_by_original.items()):
        if remap["target_box_number"] in ambiguous_targets:
            rejected.append({**remap, "reason": "duplicate_promoted_reserve_target_box"})
            remap_by_original.pop(original_box, None)

    promoted_target_boxes = {item["target_box_number"] for item in remap_by_original.values()}
    remapped_rows: List[dict] = []
    ignored_terminal_rows: List[dict] = []
    for row in official_rows or []:
        try:
            box = int(row.get("box_number") or 0)
        except Exception:
            remapped_rows.append(dict(row))
            continue
        status = str(row.get("status") or "").upper()
        if (
            box in promoted_target_boxes
            and status in {"SCR", "L/SCR", "LSCR"}
            and row.get("finish_position") is None
        ):
            ignored_terminal_rows.append(
                {
                    "box_number": box,
                    "status": status,
                    "dog_name": row.get("dog_name"),
                    "reason": "replaced_by_promoted_reserve_from_box_note",
                }
            )
            continue
        remap = remap_by_original.get(box)
        if remap:
            updated = dict(row)
            updated["original_box_number"] = box
            updated["box_number"] = remap["target_box_number"]
            updated["dog_name"] = remap["cleaned_official_dog_name"]
            updated["reserve_box_remap_source"] = remap["source"]
            remapped_rows.append(updated)
        else:
            remapped_rows.append(dict(row))

    return {
        "rows": remapped_rows,
        "remappings": list(remap_by_original.values()),
        "ignored_terminal_status_rows": ignored_terminal_rows,
        "rejected_remappings": rejected,
    }


def _terminal_status_from_text(value: str) -> Optional[str]:
    status = re.sub(r"\s+", " ", str(value or "").strip().upper())
    if status in {"FELL", "SCR", "L/SCR", "LSCR", "DNF", "DISQ"}:
        return status
    return None


def rendered_text_from_html(markup: str) -> str:
    try:
        from bs4 import BeautifulSoup

        return BeautifulSoup(markup or "", "html.parser").get_text("\n", strip=True)
    except Exception:
        cleaned = re.sub(
            r"<(script|style)\b[^>]*>.*?</\1>",
            "\n",
            str(markup or ""),
            flags=re.IGNORECASE | re.DOTALL,
        )
        cleaned = re.sub(r"<br\s*/?>", "\n", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"</(?:p|div|tr|li|td|th|h[1-6])>", "\n", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        cleaned = html.unescape(cleaned)
        return "\n".join(line.strip() for line in cleaned.splitlines() if line.strip())


def parse_thedogs_result_html(markup: str) -> Dict[int, int]:
    """Parse official TheDogs result rows by rug box from the result table.

    This deliberately does not filter to local participants. Unknown official
    boxes must stay visible so participant-alignment validation can reject the
    race instead of silently treating a later local runner as the winner.
    """
    if not str(markup or "").strip():
        return {}

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(markup or "", "html.parser")
        positions: Dict[int, int] = {}
        for row in soup.select("table.race-runners--result tr.race-runner"):
            position_cell = row.select_one("td.race-runners__finish-position")
            box_cell = row.select_one("td.race-runners__box")
            if position_cell is None or box_cell is None:
                continue

            position = _strict_ordinal_to_position(position_cell.get_text(" ", strip=True))
            if position is None:
                continue

            box_number = None
            rug = box_cell.find(attrs={"name": re.compile(r"^rug_\d{1,2}$")})
            if rug is not None:
                box_number = _rug_box_from_markup(str(rug))
            if box_number is None:
                box_number = _rug_box_from_markup(str(box_cell))
            if box_number is None:
                continue

            positions.setdefault(box_number, position)
        return positions
    except Exception:
        positions: Dict[int, int] = {}
        row_pattern = re.compile(
            r"<tr\b(?=[^>]*\brace-runner\b)[^>]*>(?P<row>.*?)</tr>",
            re.IGNORECASE | re.DOTALL,
        )
        position_pattern = re.compile(
            r"<td\b(?=[^>]*\brace-runners__finish-position\b)[^>]*>(?P<value>.*?)</td>",
            re.IGNORECASE | re.DOTALL,
        )
        box_pattern = re.compile(
            r"<td\b(?=[^>]*\brace-runners__box\b)[^>]*>(?P<value>.*?)</td>",
            re.IGNORECASE | re.DOTALL,
        )
        for row_match in row_pattern.finditer(str(markup or "")):
            row_markup = row_match.group("row")
            position_match = position_pattern.search(row_markup)
            box_match = box_pattern.search(row_markup)
            if not position_match or not box_match:
                continue
            position_text = rendered_text_from_html(position_match.group("value"))
            position = _strict_ordinal_to_position(position_text)
            box_number = _rug_box_from_markup(box_match.group("value"))
            if position is None or box_number is None:
                continue
            positions.setdefault(box_number, position)
        return positions


def parse_thedogs_result_html_runner_rows(markup: str) -> List[dict]:
    """Parse official result rows with box, finish/status, and dog name.

    This is a diagnostic companion to parse_thedogs_result_html. It does not
    replace the existing position parser because label gates still rely on the
    stricter box-to-position map.
    """
    if not str(markup or "").strip():
        return []

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(markup or "", "html.parser")
        rows: List[dict] = []
        for row in soup.select("table.race-runners--result tr.race-runner"):
            position_cell = row.select_one("td.race-runners__finish-position")
            box_cell = row.select_one("td.race-runners__box")
            name_cell = row.select_one("td.race-runners__name")
            if position_cell is None or box_cell is None:
                continue

            position_text = position_cell.get_text(" ", strip=True)
            position = _strict_ordinal_to_position(position_text)
            status = _terminal_status_from_text(position_text)

            box_number = None
            rug = box_cell.find(attrs={"name": re.compile(r"^rug_\d{1,2}$")})
            if rug is not None:
                box_number = _rug_box_from_markup(str(rug))
            if box_number is None:
                box_number = _rug_box_from_markup(str(box_cell))
            if box_number is None:
                continue

            dog_name = (
                _clean_official_runner_name(name_cell.get_text(" ", strip=True))
                if name_cell is not None
                else None
            )
            rows.append(
                {
                    "box_number": box_number,
                    "finish_position": position,
                    "dog_name": dog_name,
                    "status": status,
                }
            )
        return rows
    except Exception:
        rows = []
        row_pattern = re.compile(
            r"<tr\b(?=[^>]*\brace-runner\b)[^>]*>(?P<row>.*?)</tr>",
            re.IGNORECASE | re.DOTALL,
        )
        position_pattern = re.compile(
            r"<td\b(?=[^>]*\brace-runners__finish-position\b)[^>]*>(?P<value>.*?)</td>",
            re.IGNORECASE | re.DOTALL,
        )
        box_pattern = re.compile(
            r"<td\b(?=[^>]*\brace-runners__box\b)[^>]*>(?P<value>.*?)</td>",
            re.IGNORECASE | re.DOTALL,
        )
        name_pattern = re.compile(
            r"<td\b(?=[^>]*\brace-runners__name\b)[^>]*>(?P<value>.*?)</td>",
            re.IGNORECASE | re.DOTALL,
        )
        for row_match in row_pattern.finditer(str(markup or "")):
            row_markup = row_match.group("row")
            position_match = position_pattern.search(row_markup)
            box_match = box_pattern.search(row_markup)
            if not position_match or not box_match:
                continue
            position_text = rendered_text_from_html(position_match.group("value"))
            position = _strict_ordinal_to_position(position_text)
            status = _terminal_status_from_text(position_text)
            box_number = _rug_box_from_markup(box_match.group("value"))
            if box_number is None:
                continue
            name_match = name_pattern.search(row_markup)
            dog_name = (
                _clean_official_runner_name(rendered_text_from_html(name_match.group("value")))
                if name_match
                else None
            )
            rows.append(
                {
                    "box_number": box_number,
                    "finish_position": position,
                    "dog_name": dog_name,
                    "status": status,
                }
            )
        return rows


def parse_thedogs_result_html_terminal_statuses(markup: str) -> Dict[int, str]:
    if not str(markup or "").strip():
        return {}

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(markup or "", "html.parser")
        statuses: Dict[int, str] = {}
        for row in soup.select("table.race-runners--result tr.race-runner"):
            position_cell = row.select_one("td.race-runners__finish-position")
            box_cell = row.select_one("td.race-runners__box")
            if position_cell is None or box_cell is None:
                continue

            status = _terminal_status_from_text(position_cell.get_text(" ", strip=True))
            if status is None:
                continue

            box_number = None
            rug = box_cell.find(attrs={"name": re.compile(r"^rug_\d{1,2}$")})
            if rug is not None:
                box_number = _rug_box_from_markup(str(rug))
            if box_number is None:
                box_number = _rug_box_from_markup(str(box_cell))
            if box_number is None:
                continue

            statuses.setdefault(box_number, status)
        return statuses
    except Exception:
        statuses: Dict[int, str] = {}
        row_pattern = re.compile(
            r"<tr\b(?=[^>]*\brace-runner\b)[^>]*>(?P<row>.*?)</tr>",
            re.IGNORECASE | re.DOTALL,
        )
        position_pattern = re.compile(
            r"<td\b(?=[^>]*\brace-runners__finish-position\b)[^>]*>(?P<value>.*?)</td>",
            re.IGNORECASE | re.DOTALL,
        )
        box_pattern = re.compile(
            r"<td\b(?=[^>]*\brace-runners__box\b)[^>]*>(?P<value>.*?)</td>",
            re.IGNORECASE | re.DOTALL,
        )
        for row_match in row_pattern.finditer(str(markup or "")):
            row_markup = row_match.group("row")
            position_match = position_pattern.search(row_markup)
            box_match = box_pattern.search(row_markup)
            if not position_match or not box_match:
                continue
            position_text = rendered_text_from_html(position_match.group("value"))
            status = _terminal_status_from_text(position_text)
            box_number = _rug_box_from_markup(box_match.group("value"))
            if status is None or box_number is None:
                continue
            statuses.setdefault(box_number, status)
        return statuses


def thedogs_result_rows_present(markup: str) -> bool:
    if not str(markup or "").strip():
        return False
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(markup or "", "html.parser")
        return bool(soup.select("table.race-runners--result tr.race-runner"))
    except Exception:
        return bool(
            re.search(
                r"<tr\b(?=[^>]*\brace-runner\b)",
                str(markup or ""),
                re.IGNORECASE | re.DOTALL,
            )
        )


def response_is_forbidden(status_code: Optional[int], title: str, text: str) -> bool:
    return (
        status_code == 403
        or (title or "").strip() == "403 Forbidden"
        or (text or "").strip() == "403 Forbidden"
    )


def terminal_public_http_error(error: Optional[str]) -> bool:
    return bool(
        error == "thedogs_403_forbidden"
        or re.match(r"^thedogs_http_4\d\d$", str(error or ""))
    )


def title_from_html(markup: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", str(markup or ""), re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return html.unescape(re.sub(r"\s+", " ", match.group(1))).strip()


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
    participant_source: str = "csv"
    csv_participants: Optional[List[dict]] = None
    runner_completeness: Optional[dict] = None
    canonical_thedogs_url: Optional[str] = None

    @property
    def venue_code(self) -> str:
        return code_from_race_id(self.race_id) or str(self.venue or "").strip()

    @property
    def sportsbet_slug(self) -> Optional[str]:
        return sportsbet_slug_from_url(self.sportsbet_url or "") or self.thedogs_slug

    @property
    def thedogs_slug(self) -> Optional[str]:
        canonical_slug = thedogs_slug_from_race_url(self.canonical_thedogs_url or "")
        if canonical_slug:
            return canonical_slug
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
    terminal_status_by_box: Optional[Dict[int, str]] = None
    reserve_box_remappings: Optional[List[dict]] = None
    ignored_terminal_status_rows: Optional[List[dict]] = None
    rejected_reserve_box_remappings: Optional[List[dict]] = None

    @property
    def winner_box(self) -> Optional[int]:
        if not self.positions_by_box:
            return None
        return sorted(self.positions_by_box.items(), key=lambda item: item[1])[0][0]


def _source_result_diagnostic(result: SourceResult) -> dict:
    return {
        "source": result.source,
        "status": result.status,
        "source_url": result.source_url,
        "error": result.error,
        "raw_order": list(result.raw_order),
        "terminal_statuses": [
            {
                "box_number": int(box),
                "status": str(status),
            }
            for box, status in sorted((result.terminal_status_by_box or {}).items())
        ],
        "reserve_box_remappings": list(result.reserve_box_remappings or []),
        "ignored_terminal_status_rows": list(result.ignored_terminal_status_rows or []),
        "rejected_reserve_box_remappings": list(result.rejected_reserve_box_remappings or []),
        "positions": [
            {
                "box_number": int(box),
                "finish_position": int(position),
            }
            for box, position in sorted(
                result.positions_by_box.items(),
                key=lambda item: (item[1], item[0]),
            )
        ],
    }


def result_validation_error(candidate: RaceCandidate, result: SourceResult) -> Optional[str]:
    if not result.positions_by_box:
        return result.error or "no_result_positions"

    participant_boxes = {
        int(participant["box_number"])
        for participant in candidate.participants
        if participant.get("box_number") is not None
    }
    result_boxes = {int(box) for box in result.positions_by_box}
    unknown_boxes = sorted(result_boxes - participant_boxes)
    if unknown_boxes:
        reason = (
            "result_boxes_not_in_frozen_participants"
            if candidate.participant_source == "snapshot"
            else "result_boxes_not_in_participants"
        )
        return reason + ":" + ",".join(
            str(box) for box in unknown_boxes
        )
    finish_positions = [
        int(position)
        for position in result.positions_by_box.values()
        if position is not None
    ]
    if 1 not in finish_positions:
        return "missing_first_place_result"
    if len(finish_positions) != len(set(finish_positions)):
        return "duplicate_finish_positions"
    if result.winner_box is None:
        return "missing_winner_box"
    return None


class TheDogsResultFetcher:
    def __init__(self, driver, wait_seconds: float = 4.0, by=None, http_session=None):
        self.driver = driver
        self.wait_seconds = wait_seconds
        self.by = by or _SeleniumByFallback
        self.http_session = http_session
        self._meeting_url_cache: Dict[tuple, List[str]] = {}

    def _result_urls(self, candidate: RaceCandidate) -> List[str]:
        slug = candidate.thedogs_slug
        if not slug:
            return []

        urls: List[str] = []
        urls.extend(thedogs_result_urls_from_race_url(candidate.canonical_thedogs_url or ""))
        urls.extend(self._discover_meeting_race_urls(candidate))
        urls.extend(
            [
                f"{THEDOGS_BASE}/racing/{slug}/{candidate.race_date}/{candidate.race_number}/results?trial=false",
                f"{THEDOGS_BASE}/racing/{slug}/{candidate.race_date}/{candidate.race_number}/results",
                f"{THEDOGS_BASE}/racing/{slug}/{candidate.race_date}/{candidate.race_number}",
            ]
        )

        deduped: List[str] = []
        seen = set()
        for url in urls:
            if url and url not in seen:
                deduped.append(url)
                seen.add(url)
        return deduped

    def _discover_meeting_race_urls(self, candidate: RaceCandidate) -> List[str]:
        if self.http_session is None:
            return []

        slug = candidate.thedogs_slug
        if not slug:
            return []

        cache_key = (slug, candidate.race_date, int(candidate.race_number))
        if cache_key in self._meeting_url_cache:
            return self._meeting_url_cache[cache_key]

        meeting_url = f"{THEDOGS_BASE}/racing/{slug}/{candidate.race_date}?trial=false"
        urls: List[str] = []
        try:
            response = self.http_session.get(
                meeting_url,
                headers=THEDOGS_PUBLIC_HEADERS,
                timeout=20,
                allow_redirects=True,
            )
            text = getattr(response, "text", "") or ""
            if response_is_forbidden(
                getattr(response, "status_code", None),
                title_from_html(text),
                rendered_text_from_html(text),
            ):
                self._meeting_url_cache[cache_key] = []
                return []

            pattern = re.compile(
                rf"""href=["'](?P<href>/racing/{re.escape(slug)}/{re.escape(candidate.race_date)}/{int(candidate.race_number)}/[^"']+)["']""",
                re.IGNORECASE,
            )
            for match in pattern.finditer(text):
                href = html.unescape(match.group("href"))
                if "/expert-form" in href or "/results/" in href:
                    continue
                urls.append(urljoin(THEDOGS_BASE, href))
        except Exception:
            urls = []

        self._meeting_url_cache[cache_key] = urls
        return urls

    def _result_from_text(
        self,
        candidate: RaceCandidate,
        source_url: str,
        text: str,
    ) -> Optional[SourceResult]:
        positions = parse_thedogs_result_text(text, candidate.participants)
        if not positions:
            return None
        ordered_boxes = [
            box for box, _ in sorted(positions.items(), key=lambda item: item[1])
        ]
        return SourceResult(
            source="thedogs_official",
            status=RESULTED,
            source_url=source_url,
            positions_by_box=positions,
            raw_order=ordered_boxes,
        )

    def _result_from_html(
        self,
        candidate: RaceCandidate,
        source_url: str,
        markup: str,
    ) -> Optional[SourceResult]:
        runner_rows = parse_thedogs_result_html_runner_rows(markup)
        reserve_remap = remap_promoted_reserve_runner_rows(
            runner_rows,
            candidate.participants,
        )
        remapped_rows = reserve_remap["rows"]
        positions = {
            int(row["box_number"]): int(row["finish_position"])
            for row in remapped_rows
            if row.get("box_number") is not None and row.get("finish_position") is not None
        }
        terminal_statuses = {
            int(row["box_number"]): str(row["status"])
            for row in remapped_rows
            if row.get("box_number") is not None and row.get("status")
        }
        if not runner_rows:
            positions = parse_thedogs_result_html(markup)
            terminal_statuses = parse_thedogs_result_html_terminal_statuses(markup)
        if positions:
            ordered_boxes = [
                box for box, _ in sorted(positions.items(), key=lambda item: item[1])
            ]
            return SourceResult(
                source="thedogs_official",
                status=RESULTED,
                source_url=source_url,
                positions_by_box=positions,
                raw_order=ordered_boxes,
                terminal_status_by_box=terminal_statuses,
                reserve_box_remappings=reserve_remap["remappings"],
                ignored_terminal_status_rows=reserve_remap["ignored_terminal_status_rows"],
                rejected_reserve_box_remappings=reserve_remap["rejected_remappings"],
            )
        if thedogs_result_rows_present(markup):
            return SourceResult(
                source="thedogs_official",
                status="error",
                source_url=source_url,
                positions_by_box={},
                raw_order=[],
                error="thedogs_result_table_without_strict_positions",
            )
        return self._result_from_text(
            candidate,
            source_url,
            rendered_text_from_html(markup),
        )

    def _fetch_via_http(self, candidate: RaceCandidate, urls: List[str]) -> Optional[SourceResult]:
        if self.http_session is None:
            return None

        last_error = None
        for url in urls:
            try:
                response = self.http_session.get(
                    url,
                    headers=THEDOGS_PUBLIC_HEADERS,
                    timeout=20,
                    allow_redirects=True,
                )
                markup = getattr(response, "text", "") or ""
                text = rendered_text_from_html(markup)
                status_code = getattr(response, "status_code", None)
                if response_is_forbidden(status_code, title_from_html(markup), text):
                    last_error = "thedogs_403_forbidden"
                    break
                if status_code and status_code >= 400:
                    last_error = f"thedogs_http_{status_code}"
                    continue

                result = self._result_from_html(candidate, getattr(response, "url", url), markup)
                if result:
                    return result
                last_error = "no_thedogs_positions_found"
            except Exception as exc:
                last_error = f"thedogs_http_error:{type(exc).__name__}"

        if last_error:
            return SourceResult(
                source="thedogs_official",
                status="error",
                source_url=urls[0] if urls else None,
                positions_by_box={},
                raw_order=[],
                error=last_error,
            )
        return None

    def fetch(self, candidate: RaceCandidate) -> SourceResult:
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

        urls = self._result_urls(candidate)
        http_result = self._fetch_via_http(candidate, urls)
        if http_result and http_result.positions_by_box:
            return http_result
        if http_result and terminal_public_http_error(http_result.error):
            return http_result

        last_error = None
        if http_result and http_result.error:
            last_error = http_result.error
        for url in urls:
            try:
                self.driver.get(url)
                time.sleep(self.wait_seconds)
                title = (self.driver.title or "").strip()
                text = self.driver.find_element(self.by.TAG_NAME, "body").text
                if response_is_forbidden(None, title, text):
                    last_error = "thedogs_403_forbidden"
                    break
                page_source = getattr(self.driver, "page_source", None)
                result = (
                    self._result_from_html(candidate, url, page_source)
                    if page_source
                    else self._result_from_text(candidate, url, text)
                )
                if result:
                    return result
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


def resolve_csv_path_from_identity(
    upcoming_dir: Path,
    *,
    race_id: str,
    venue: str,
    race_number: int,
    race_date: str,
    source_file_path: Optional[str] = None,
) -> Optional[Path]:
    if source_file_path:
        source_path = Path(source_file_path)
        if source_path.exists():
            return source_path
        basename_match = upcoming_dir / source_path.name
        if basename_match.exists():
            return basename_match

    exact = upcoming_dir / f"{race_id}.csv"
    if exact.exists():
        return exact

    code = code_from_race_id(race_id) or str(venue or "").strip()
    if code and race_number:
        candidate = upcoming_dir / f"Race {int(race_number)} - {code} - {race_date}.csv"
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


def jumped_or_already_resulted(
    row: sqlite3.Row,
    now: Optional[datetime] = None,
    *,
    source_context: str = "live_record",
) -> tuple[bool, str]:
    lifecycle = classify_race_record(
        _lifecycle_record_for_row(row),
        now=now,
        source_context=source_context,
    )
    return lifecycle.status != UPCOMING_NOT_JUMPED, lifecycle.status


def _snapshot_files(snapshot_dir: Path, target_date: str) -> List[Path]:
    date_dir = snapshot_dir / target_date
    if not date_dir.exists():
        return []
    return sorted(path for path in date_dir.glob("*/*.json") if path.is_file())


def _snapshot_identity(snapshot: dict) -> Optional[dict]:
    if assert_no_result_fields is None:
        raise ValueError("snapshot_result_guard_unavailable")
    assert_no_result_fields(snapshot)

    if snapshot.get("is_pre_jump_snapshot") is not True:
        return None
    if snapshot.get("snapshot_state") != "pre_jump_feature_freeze":
        return None
    readiness = snapshot.get("snapshot_readiness")
    if not isinstance(readiness, dict) or readiness.get("status") != "READY":
        return {
            "_skip_reason": "snapshot_not_ready_for_result_labels",
            "race_id": str(snapshot.get("race_id") or ""),
            "snapshot_readiness": readiness if isinstance(readiness, dict) else None,
        }

    race_id = str(snapshot.get("race_id") or "").strip()
    race_date = str(snapshot.get("race_date") or "").strip()
    venue = str(snapshot.get("venue") or "").strip()
    race_number = snapshot.get("race_number")
    if not race_id or not race_date or not venue or race_number in (None, ""):
        return None

    try:
        race_number_int = int(race_number)
    except (TypeError, ValueError):
        return None

    frozen_participants = []
    for participant in snapshot.get("frozen_participants") or []:
        if not isinstance(participant, dict):
            continue
        try:
            box = int(participant.get("box_number"))
        except (TypeError, ValueError):
            continue
        dog_name = clean_dog_name(participant.get("dog_name"))
        if dog_name:
            frozen_participants.append({"box_number": box, "dog_name": dog_name})

    if not frozen_participants:
        for row in snapshot.get("predictions") or []:
            if not isinstance(row, dict):
                continue
            try:
                box = int(row.get("box_number"))
            except (TypeError, ValueError):
                continue
            dog_name = clean_dog_name(row.get("dog_name") or row.get("dog_clean_name"))
            if dog_name:
                frozen_participants.append({"box_number": box, "dog_name": dog_name})

    runner_report = analyze_runner_rows(
        [
            RunnerRow(
                box_number=int(participant["box_number"]),
                dog_name=str(participant["dog_name"]),
            )
            for participant in frozen_participants
        ],
        source=f"snapshot:{race_id}",
        min_complete_runners=MIN_COMPLETE_RUNNERS,
    ).as_dict()
    source_report = snapshot.get("source_runner_completeness")
    if isinstance(source_report, dict) and source_report.get("status") == "INCOMPLETE":
        return {
            "_skip_reason": "snapshot_incomplete_runner_set",
            "race_id": race_id,
            "runner_completeness": source_report,
        }
    if runner_report.get("status") != "COMPLETE":
        return {
            "_skip_reason": "snapshot_incomplete_runner_set",
            "race_id": race_id,
            "runner_completeness": runner_report,
        }

    start_datetime = str(snapshot.get("jump_datetime") or "").strip() or None
    race_time = str(snapshot.get("jump_time") or "").strip() or None
    return {
        "race_id": race_id,
        "venue": venue,
        "race_number": race_number_int,
        "race_date": race_date,
        "race_time": race_time,
        "start_datetime": start_datetime,
        "sportsbet_url": None,
        "canonical_thedogs_url": (
            str(
                snapshot.get("canonical_race_url")
                or snapshot.get("final_runner_set_source_url")
                or ""
            ).strip()
            or None
        ),
        "results_status": None,
        "winner_name": None,
        "source_file_path": str(snapshot.get("source_file_path") or "").strip() or None,
        "participants": frozen_participants,
        "runner_completeness": runner_report,
        "participant_source": "snapshot",
    }


def load_snapshot_candidate_rows(
    snapshot_dir: Path,
    target_date: str,
    upcoming_dir: Path,
    race_ids: Iterable[str],
) -> tuple[List[dict], List[dict]]:
    race_id_filter = {race_id for race_id in race_ids if race_id}
    skipped: List[dict] = []
    latest_by_race_id: Dict[str, dict] = {}

    for path in _snapshot_files(snapshot_dir, target_date):
        try:
            snapshot = json.loads(path.read_text(encoding="utf-8"))
            identity = _snapshot_identity(snapshot)
        except Exception as exc:
            skipped.append(
                {
                    "race_id": str(path),
                    "reason": f"snapshot_unreadable_or_not_result_free:{type(exc).__name__}",
                }
            )
            continue

        if identity is None:
            skipped.append({"race_id": str(path), "reason": "not_frozen_pre_jump_snapshot"})
            continue
        if identity.get("_skip_reason"):
            skipped.append(
                {
                    "race_id": identity.get("race_id") or str(path),
                    "reason": str(identity["_skip_reason"]),
                    "runner_completeness": identity.get("runner_completeness"),
                    "snapshot_readiness": identity.get("snapshot_readiness"),
                }
            )
            continue
        if race_id_filter and identity["race_id"] not in race_id_filter:
            continue

        identity["_snapshot_sort_key"] = str(
            snapshot.get("feature_freeze_timestamp")
            or snapshot.get("prediction_timestamp")
            or ""
        )
        existing = latest_by_race_id.get(identity["race_id"])
        if existing and existing.get("_snapshot_sort_key", "") >= identity["_snapshot_sort_key"]:
            continue
        latest_by_race_id[identity["race_id"]] = identity

    rows: List[dict] = []
    for identity in latest_by_race_id.values():

        csv_path = resolve_csv_path_from_identity(
            upcoming_dir,
            race_id=identity["race_id"],
            venue=identity["venue"],
            race_number=identity["race_number"],
            race_date=identity["race_date"],
            source_file_path=identity.get("source_file_path"),
        )
        if not csv_path:
            skipped.append({"race_id": identity["race_id"], "reason": "snapshot_csv_missing"})
            continue

        identity["csv_path"] = csv_path
        identity.pop("_snapshot_sort_key", None)
        rows.append(identity)

    return rows, skipped


def load_candidates(
    db_path: Path,
    target_date: str,
    upcoming_dir: Path,
    race_ids: Iterable[str],
    now: Optional[datetime] = None,
    snapshot_dir: Optional[Path] = None,
    require_ready_snapshot: bool = False,
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
    candidate_race_ids: set[str] = set()
    metadata_by_race_id: Dict[str, dict] = {}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        for row in conn.execute(query, params).fetchall():
            metadata_by_race_id[str(row["race_id"])] = _row_dict(row)
            if require_ready_snapshot:
                continue
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
            runner_completeness = analyze_csv_runner_completeness(csv_path).as_dict()
            if runner_completeness.get("status") != "COMPLETE":
                skipped.append(
                    {
                        "race_id": row["race_id"],
                        "reason": "incomplete_runner_set",
                        "runner_completeness": runner_completeness,
                    }
                )
                continue
            participants = parse_participants_from_csv(csv_path)
            if not participants:
                skipped.append(
                    {"race_id": row["race_id"], "reason": "no_participants_from_csv"}
                )
                continue
            candidate_race_ids.add(str(row["race_id"]))
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
                    participant_source="csv",
                    csv_participants=participants,
                    runner_completeness=runner_completeness,
                )
            )
    finally:
        conn.close()

    if snapshot_dir and snapshot_dir.exists():
        snapshot_rows, snapshot_skipped = load_snapshot_candidate_rows(
            snapshot_dir, target_date, upcoming_dir, race_id_filter
        )
        skipped.extend(snapshot_skipped)
        for row in snapshot_rows:
            if row["race_id"] in candidate_race_ids:
                continue
            eligible, lifecycle_status = jumped_or_already_resulted(
                row,
                now=now,
                source_context="csv_file",
            )
            if not eligible:
                skipped.append(
                    {
                        "race_id": row["race_id"],
                        "reason": f"race_not_jumped:{lifecycle_status}",
                    }
                )
                continue
            frozen_participants = list(row.get("participants") or [])
            csv_runner_completeness = analyze_csv_runner_completeness(row["csv_path"]).as_dict()
            csv_participants = parse_participants_from_csv(row["csv_path"])
            if not frozen_participants:
                skipped.append(
                    {"race_id": row["race_id"], "reason": "no_participants_from_csv"}
                )
                continue
            frozen_boxes = {int(participant["box_number"]) for participant in frozen_participants}
            csv_boxes = {int(participant["box_number"]) for participant in csv_participants}
            if csv_boxes and csv_boxes != frozen_boxes:
                skipped.append(
                    {
                        "race_id": row["race_id"],
                        "reason": "snapshot_csv_participant_mismatch",
                        "snapshot_boxes": sorted(frozen_boxes),
                        "csv_boxes": sorted(csv_boxes),
                    }
                )
                continue
            metadata_row = metadata_by_race_id.get(str(row["race_id"])) or {}
            sportsbet_url = row.get("sportsbet_url") or metadata_row.get("sportsbet_url")
            if isinstance(sportsbet_url, str):
                sportsbet_url = sportsbet_url.strip() or None
            canonical_thedogs_url = row.get("canonical_thedogs_url")
            if isinstance(canonical_thedogs_url, str):
                canonical_thedogs_url = canonical_thedogs_url.strip() or None
            candidate_race_ids.add(str(row["race_id"]))
            candidates.append(
                RaceCandidate(
                    race_id=row["race_id"],
                    venue=row["venue"],
                    race_number=int(row["race_number"]),
                    race_date=row["race_date"],
                    race_time=row["race_time"],
                    start_datetime=row["start_datetime"],
                    sportsbet_url=sportsbet_url,
                    csv_path=row["csv_path"],
                    participants=frozen_participants,
                    lifecycle_status=lifecycle_status,
                    participant_source="snapshot",
                    csv_participants=csv_participants,
                    runner_completeness=row.get("runner_completeness") or csv_runner_completeness,
                    canonical_thedogs_url=canonical_thedogs_url,
                )
            )

    if require_ready_snapshot:
        skipped_race_ids = {str(item.get("race_id")) for item in skipped}
        for race_id in sorted(metadata_by_race_id):
            if race_id in candidate_race_ids or race_id in skipped_race_ids:
                continue
            skipped.append(
                {
                    "race_id": race_id,
                    "reason": "ready_prejump_snapshot_required",
                }
            )
        if snapshot_dir is None or not snapshot_dir.exists():
            skipped.append(
                {
                    "race_id": "__snapshot_dir__",
                    "reason": "ready_prejump_snapshot_required_but_snapshot_dir_missing",
                    "snapshot_dir": str(snapshot_dir) if snapshot_dir else None,
                }
            )

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


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}
    except Exception:
        return set()


def ensure_race_metadata_row(conn: sqlite3.Connection, candidate: RaceCandidate) -> bool:
    existing = conn.execute(
        "SELECT 1 FROM race_metadata WHERE race_id = ? LIMIT 1",
        (candidate.race_id,),
    ).fetchone()
    if existing:
        return False

    columns = _table_columns(conn, "race_metadata")
    values = {
        "race_id": candidate.race_id,
        "venue": candidate.venue,
        "race_number": candidate.race_number,
        "race_date": candidate.race_date,
        "race_time": candidate.race_time,
        "start_datetime": candidate.start_datetime,
        "sportsbet_url": candidate.sportsbet_url,
        "results_status": "pending",
        "field_size": len(candidate.participants),
        "actual_field_size": len(candidate.participants),
        "data_source": "frozen_snapshot" if candidate.participant_source == "snapshot" else "upcoming_csv",
        "extraction_timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    insert_columns = [column for column in values if column in columns]
    if not insert_columns:
        raise RuntimeError("race_metadata_schema_missing_insertable_columns")
    placeholders = ", ".join(["?"] * len(insert_columns))
    conn.execute(
        f"INSERT INTO race_metadata ({', '.join(insert_columns)}) VALUES ({placeholders})",
        tuple(values[column] for column in insert_columns),
    )
    return True


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
    terminal_statuses = {
        str(int(box)): str(status)
        for box, status in sorted((result.terminal_status_by_box or {}).items())
    }
    if terminal_statuses:
        note_parts.append(
            "terminal_statuses="
            + ",".join(
                f"{box}:{status}" for box, status in terminal_statuses.items()
            )
        )
    data_quality_note = "; ".join(note_parts)

    if dry_run:
        summary = {
            "race_id": candidate.race_id,
            "status": result.status,
            "source": result.source,
            "winner_name": winner_name,
            "box_order": result.raw_order,
            "dry_run": True,
        }
        if terminal_statuses:
            summary["terminal_statuses"] = terminal_statuses
        return summary

    metadata_seeded = ensure_race_metadata_row(conn, candidate)

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

    metadata_update = conn.execute(
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
    if metadata_update.rowcount == 0:
        raise RuntimeError(f"race_metadata_update_failed:{candidate.race_id}")

    return {
        "race_id": candidate.race_id,
        "status": result.status,
        "source": result.source,
        "winner_name": winner_name,
        "box_order": result.raw_order,
        "metadata_seeded": metadata_seeded,
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


def _normalise_report_path(path: Path) -> str:
    return str(path.expanduser().resolve())


def _race_id_scope(race_ids: Iterable[str]) -> List[str]:
    return sorted({str(race_id) for race_id in race_ids if race_id})


def _report_scope(
    *,
    args: argparse.Namespace,
    db_path: Path,
    upcoming_dir: Path,
    snapshot_dir: Optional[Path],
) -> dict:
    return {
        "db_path": _normalise_report_path(db_path),
        "date": args.date,
        "upcoming_dir": _normalise_report_path(upcoming_dir),
        "snapshot_dir": _normalise_report_path(snapshot_dir)
        if snapshot_dir is not None
        else None,
        "race_ids": _race_id_scope(getattr(args, "race_id", []) or []),
        "require_ready_snapshot": bool(getattr(args, "require_ready_snapshot", False)),
    }


def _candidate_race_ids(candidates: Iterable[RaceCandidate]) -> List[str]:
    return sorted({candidate.race_id for candidate in candidates})


def _candidate_boxes(candidate: RaceCandidate) -> set[int]:
    boxes: set[int] = set()
    for participant in candidate.participants:
        try:
            boxes.add(int(participant.get("box_number")))
        except Exception:
            continue
    return boxes


def _result_boxes(item: Mapping[str, object]) -> set[int]:
    boxes: set[int] = set()
    for value in item.get("box_order") or []:
        try:
            boxes.add(int(value))
        except Exception:
            continue
    return boxes


def _terminal_statuses(item: Mapping[str, object]) -> Dict[int, str]:
    statuses: Dict[int, str] = {}
    raw = item.get("terminal_statuses") or {}
    if isinstance(raw, Mapping):
        iterable = raw.items()
    elif isinstance(raw, list):
        pairs = []
        for entry in raw:
            if not isinstance(entry, Mapping):
                continue
            pairs.append((entry.get("box_number"), entry.get("status")))
        iterable = pairs
    else:
        return statuses
    for box, status in iterable:
        try:
            box_number = int(box)
        except Exception:
            continue
        status_text = str(status or "").strip()
        if status_text:
            statuses[box_number] = status_text
    return statuses


def _label_write_blockers(
    ingested: Iterable[dict],
    candidates: Iterable[RaceCandidate],
) -> List[dict]:
    blockers: List[dict] = []
    candidates_by_race_id = {candidate.race_id: candidate for candidate in candidates}
    for item in ingested:
        source = item.get("source")
        status = item.get("status")
        race_id = item.get("race_id")
        if source == "thedogs_official" and status == RESULTED:
            candidate = candidates_by_race_id.get(str(race_id))
            expected_boxes = _candidate_boxes(candidate) if candidate else set()
            observed_boxes = _result_boxes(item)
            terminal_statuses = _terminal_statuses(item)
            expected_terminal_statuses = {
                box: status
                for box, status in terminal_statuses.items()
                if expected_boxes and box in expected_boxes
            }
            accounted_boxes = observed_boxes | set(expected_terminal_statuses)
            if (
                expected_terminal_statuses
                and expected_boxes
                and accounted_boxes == expected_boxes
                and not (observed_boxes - expected_boxes)
            ):
                blockers.append(
                    {
                        "race_id": race_id,
                        "source": source,
                        "status": status,
                        "reason": "label_write_requires_terminal_status_support",
                        "terminal_statuses": {
                            str(box): terminal_status
                            for box, terminal_status in sorted(
                                expected_terminal_statuses.items()
                            )
                        },
                    }
                )
                continue
            if expected_boxes and observed_boxes == expected_boxes:
                continue
            blockers.append(
                {
                    "race_id": race_id,
                    "source": source,
                    "status": status,
                    "reason": "label_write_requires_complete_official_result_positions",
                    "expected_box_count": len(expected_boxes),
                    "result_box_count": len(observed_boxes),
                    "missing_result_boxes": sorted(expected_boxes - observed_boxes),
                    "unexpected_result_boxes": sorted(observed_boxes - expected_boxes),
                }
            )
            continue
        blockers.append(
            {
                "race_id": race_id,
                "source": source,
                "status": status,
                "reason": "label_write_requires_complete_official_result",
            }
        )
    return blockers


def _build_report(
    *,
    args: argparse.Namespace,
    db_path: Path,
    upcoming_dir: Path,
    snapshot_dir: Optional[Path],
    write_approval: dict,
    candidates: List[RaceCandidate],
    skipped: List[dict],
    ingested: List[dict],
    failed: List[dict],
    backup_path: Optional[Path],
    dry_run_report_gate: Optional[dict] = None,
) -> dict:
    candidate_ids = _candidate_race_ids(candidates)
    status = "SUCCESS"
    if not candidates:
        status = "DATA_MISSING"
    if failed:
        status = "FAILED"
    label_write_blockers = _label_write_blockers(ingested, candidates)
    clean_for_label_write = bool(
        args.dry_run
        and candidates
        and not failed
        and len(ingested) == len(candidates)
        and not label_write_blockers
    )
    report = {
        "schema_version": "official_result_ingest_report_v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": status,
        "dry_run": bool(args.dry_run),
        "scope": _report_scope(
            args=args,
            db_path=db_path,
            upcoming_dir=upcoming_dir,
            snapshot_dir=snapshot_dir,
        ),
        "candidate_count": len(candidates),
        "candidate_race_ids": candidate_ids,
        "skipped_count": len(skipped),
        "skipped": skipped,
        "ingested_count": len(ingested),
        "ingested": ingested,
        "failed_count": len(failed),
        "failed": failed,
        "label_write_blockers": label_write_blockers,
        "backup_path": str(backup_path) if backup_path else None,
        "result_label_write_approval": write_approval,
        "dry_run_report_gate": dry_run_report_gate,
        "clean_for_label_write": clean_for_label_write,
    }
    return report


def _write_output_report(report: dict, output_path: Optional[str]) -> None:
    if not output_path:
        return
    path = Path(output_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_clean_dry_run_report(
    *,
    report_path: Optional[str],
    args: argparse.Namespace,
    db_path: Path,
    upcoming_dir: Path,
    snapshot_dir: Optional[Path],
    candidate_race_ids: Iterable[str],
) -> dict:
    gate = {
        "approved": False,
        "status": "not_approved",
        "report_path": report_path,
        "required_for": "official_result_label_writes",
    }
    if not report_path:
        gate["reason"] = "missing_approved_dry_run_report"
        return gate

    path = Path(report_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        gate["reason"] = "approved_dry_run_report_not_found"
        gate["resolved_report_path"] = str(path)
        return gate

    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        gate["reason"] = f"approved_dry_run_report_unreadable:{type(exc).__name__}"
        gate["resolved_report_path"] = str(path)
        return gate

    expected_scope = _report_scope(
        args=args,
        db_path=db_path,
        upcoming_dir=upcoming_dir,
        snapshot_dir=snapshot_dir,
    )
    observed_scope = report.get("scope") if isinstance(report, dict) else None
    expected_candidate_ids = _race_id_scope(candidate_race_ids)
    observed_candidate_ids = _race_id_scope(
        report.get("candidate_race_ids") if isinstance(report, dict) else []
    )
    failures: List[str] = []
    if not isinstance(report, dict):
        failures.append("report_root_not_object")
    else:
        if report.get("schema_version") != "official_result_ingest_report_v1":
            failures.append("schema_version_mismatch")
        if report.get("dry_run") is not True:
            failures.append("report_is_not_dry_run")
        if report.get("status") != "SUCCESS":
            failures.append("report_status_not_success")
        if report.get("clean_for_label_write") is not True:
            failures.append("report_not_clean_for_label_write")
        if observed_scope != expected_scope:
            failures.append("report_scope_mismatch")
        if observed_candidate_ids != expected_candidate_ids:
            failures.append("candidate_race_ids_mismatch")
        if int(report.get("failed_count") or 0) != 0:
            failures.append("dry_run_failed_count_nonzero")
        if int(report.get("candidate_count") or 0) <= 0:
            failures.append("dry_run_candidate_count_zero")
        if int(report.get("ingested_count") or 0) != int(report.get("candidate_count") or 0):
            failures.append("dry_run_ingested_count_mismatch")

    gate.update(
        {
            "resolved_report_path": str(path),
            "expected_scope": expected_scope,
            "observed_scope": observed_scope,
            "expected_candidate_race_ids": expected_candidate_ids,
            "observed_candidate_race_ids": observed_candidate_ids,
        }
    )
    if failures:
        gate["reason"] = ",".join(failures)
        return gate

    gate["approved"] = True
    gate["status"] = "approved"
    return gate


def _skip_reason_counts(skipped: Iterable[Mapping[str, object]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in skipped:
        reason = str(item.get("reason") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def build_label_write_readiness_report(
    *,
    args: argparse.Namespace,
    db_path: Path,
    upcoming_dir: Path,
    snapshot_dir: Optional[Path],
    write_approval: dict,
    candidates: List[RaceCandidate],
    skipped: List[dict],
    dry_run_report_gate: dict,
) -> dict:
    planned_command = [
        str(REPO_ROOT / ".venv/bin/python"),
        "scripts/ingest_results_for_date.py",
        "--db",
        str(db_path),
        "--date",
        args.date,
        "--upcoming-dir",
        str(upcoming_dir),
    ]
    if snapshot_dir is not None:
        planned_command.extend(["--snapshot-dir", str(snapshot_dir)])
    if getattr(args, "require_ready_snapshot", False):
        planned_command.append("--require-ready-snapshot")
    if args.approved_dry_run_report:
        planned_command.extend(["--approved-dry-run-report", args.approved_dry_run_report])
    for race_id in getattr(args, "race_id", []) or []:
        planned_command.extend(["--race-id", race_id])
    planned_command.append("--write-labels-approved")
    if getattr(args, "output", None):
        planned_output = Path(args.output)
        planned_command.extend(
            [
                "--output",
                str(planned_output.with_name("result_label_write_report_if_approved.json")),
            ]
        )

    ready = bool(dry_run_report_gate.get("approved") and candidates)
    skipped_by_reason = _skip_reason_counts(skipped)
    return {
        "schema_version": "result_label_write_readiness_validation_v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "READY_FOR_EXPLICIT_APPROVAL" if ready else "NOT_READY",
        "scope": _report_scope(
            args=args,
            db_path=db_path,
            upcoming_dir=upcoming_dir,
            snapshot_dir=snapshot_dir,
        ),
        "candidate_count_loaded_for_write_scope": len(candidates),
        "candidate_race_ids_loaded_for_write_scope": _candidate_race_ids(candidates),
        "skipped_count_before_write_scope_validation": len(skipped),
        "skipped_before_write_scope_validation_by_reason": skipped_by_reason,
        "skipped_before_write_scope_validation": skipped,
        "dry_run_report_gate": dry_run_report_gate,
        "result_label_write_approval": write_approval,
        "approval_required": True,
        "required_cli_flag": "--write-labels-approved",
        "required_env_var": "APPROVE_RESULT_LABEL_WRITE",
        "planned_command_if_approved": planned_command,
        "write_performed": False,
    }


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
        "--snapshot-dir",
        default="artifacts/prediction_snapshots",
        help="Directory containing frozen pre-jump prediction snapshots",
    )
    parser.add_argument(
        "--race-id",
        action="append",
        default=[],
        help="Optional race_id filter. Can be repeated.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Parse but do not write")
    parser.add_argument(
        "--output",
        help="Optional JSON report path for dry-run/write evidence",
    )
    parser.add_argument(
        "--write-labels-approved",
        action="store_true",
        help=(
            "Approval gate for writing official result labels. Without this flag "
            "or APPROVE_RESULT_LABEL_WRITE=true, non-dry-run execution exits "
            "before fetching or mutating the database."
        ),
    )
    parser.add_argument(
        "--approved-dry-run-report",
        help=(
            "Required for non-dry-run label writes. Must point to a clean "
            "official_result_ingest_report_v1 dry-run report for the same "
            "date, DB, upcoming directory, snapshot directory, and race filter."
        ),
    )
    parser.add_argument(
        "--validate-label-write-readiness",
        action="store_true",
        help=(
            "Read-only validation for a planned result label write. Validates "
            "--approved-dry-run-report against the exact date, DB, upcoming "
            "directory, snapshot directory, and race filter, writes the "
            "readiness report to --output, then exits without fetching pages "
            "or mutating the database."
        ),
    )
    parser.add_argument(
        "--require-ready-snapshot",
        action="store_true",
        help=(
            "Only ingest labels for races backed by a result-free persisted "
            "prediction_snapshot_v1 artifact with snapshot_readiness.status READY."
        ),
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Show browser while fetching pages",
    )
    return parser


def optional_browser_driver(headless: bool):
    try:
        from selenium.webdriver.common.by import By
    except Exception as exc:  # noqa: BLE001 - CLI records dependency state in reports.
        return None, _SeleniumByFallback, f"browser_unavailable:{type(exc).__name__}"

    try:
        from drivers import get_chrome_driver

        return get_chrome_driver(headless=headless), By, None
    except Exception as exc:  # noqa: BLE001 - keep official HTTP dry-runs usable.
        return None, By, f"browser_unavailable:{type(exc).__name__}"


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    db_path = Path(args.db)
    upcoming_dir = Path(args.upcoming_dir)
    snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else None

    if not db_path.exists():
        print(f"ERROR database not found: {db_path}", file=sys.stderr)
        return 2
    if not upcoming_dir.exists():
        print(f"ERROR upcoming directory not found: {upcoming_dir}", file=sys.stderr)
        return 2
    write_approval = result_label_write_approved(args)
    if (
        not args.dry_run
        and not args.validate_label_write_readiness
        and not write_approval["approved"]
    ):
        print(
            "ERROR result label writes require --write-labels-approved or "
            "APPROVE_RESULT_LABEL_WRITE=true; rerun with --dry-run for fetch-only "
            "official-first validation.",
            file=sys.stderr,
        )
        print(json.dumps({"result_label_write_approval": write_approval}, sort_keys=True))
        return 2

    candidates, skipped = load_candidates(
        db_path,
        args.date,
        upcoming_dir,
        args.race_id,
        snapshot_dir=snapshot_dir,
        require_ready_snapshot=bool(args.require_ready_snapshot),
    )
    print(f"Candidates: {len(candidates)}")
    if skipped:
        print(f"Skipped before fetch: {len(skipped)}")
        for item in skipped:
            print(f"SKIPPED {item}")
    if args.validate_label_write_readiness:
        dry_run_report_gate = validate_clean_dry_run_report(
            report_path=args.approved_dry_run_report,
            args=args,
            db_path=db_path,
            upcoming_dir=upcoming_dir,
            snapshot_dir=snapshot_dir,
            candidate_race_ids=_candidate_race_ids(candidates),
        )
        report = build_label_write_readiness_report(
            args=args,
            db_path=db_path,
            upcoming_dir=upcoming_dir,
            snapshot_dir=snapshot_dir,
            write_approval=write_approval,
            candidates=candidates,
            skipped=skipped,
            dry_run_report_gate=dry_run_report_gate,
        )
        _write_output_report(report, args.output)
        print(json.dumps({"label_write_readiness": report}, sort_keys=True))
        return 0 if report["status"] == "READY_FOR_EXPLICIT_APPROVAL" else 2
    dry_run_report_gate = None
    if not args.dry_run:
        dry_run_report_gate = validate_clean_dry_run_report(
            report_path=args.approved_dry_run_report,
            args=args,
            db_path=db_path,
            upcoming_dir=upcoming_dir,
            snapshot_dir=snapshot_dir,
            candidate_race_ids=_candidate_race_ids(candidates),
        )
        if not dry_run_report_gate["approved"] and candidates:
            print(
                "ERROR result label writes require a clean prior --dry-run report "
                "via --approved-dry-run-report.",
                file=sys.stderr,
            )
            print(json.dumps({"dry_run_report_gate": dry_run_report_gate}, sort_keys=True))
            report = _build_report(
                args=args,
                db_path=db_path,
                upcoming_dir=upcoming_dir,
                snapshot_dir=snapshot_dir,
                write_approval=write_approval,
                candidates=candidates,
                skipped=skipped,
                ingested=[],
                failed=[],
                backup_path=None,
                dry_run_report_gate=dry_run_report_gate,
            )
            _write_output_report(report, args.output)
            return 2
    if not candidates:
        report = _build_report(
            args=args,
            db_path=db_path,
            upcoming_dir=upcoming_dir,
            snapshot_dir=snapshot_dir,
            write_approval=write_approval,
            candidates=candidates,
            skipped=skipped,
            ingested=[],
            failed=[],
            backup_path=None,
            dry_run_report_gate=dry_run_report_gate,
        )
        _write_output_report(report, args.output)
        return 0

    backup_path = None
    if not args.dry_run:
        backup_path = backup_db(db_path)
        print(f"Backup: {backup_path}")

    driver, By, browser_error = optional_browser_driver(headless=not args.no_headless)
    if browser_error:
        print(f"Browser fallback unavailable: {browser_error}", file=sys.stderr)
    ingested: List[dict] = []
    failed: List[dict] = []
    try:
        thedogs = TheDogsResultFetcher(
            driver,
            by=By,
            http_session=_StatelessPublicHttpClient(),
        )
        sportsbet = SportsbetResultFetcher(driver, args.date, by=By) if driver else None

        conn = sqlite3.connect(db_path)
        try:
            conn.execute("BEGIN")
            for candidate in candidates:
                attempts: List[SourceResult] = []
                official = thedogs.fetch(candidate)
                attempts.append(official)
                official_error = result_validation_error(candidate, official)
                if official_error and not official.error:
                    official.error = official_error
                chosen = official if official_error is None else None
                if chosen is None:
                    fallback = (
                        sportsbet.fetch(candidate)
                        if sportsbet is not None
                        else SourceResult(
                            source="sportsbet_results_top4",
                            status="error",
                            source_url=None,
                            positions_by_box={},
                            raw_order=[],
                            error=browser_error or "browser_unavailable",
                        )
                    )
                    attempts.append(fallback)
                    fallback_error = result_validation_error(candidate, fallback)
                    if fallback_error and not fallback.error:
                        fallback.error = fallback_error
                    chosen = fallback if fallback_error is None else None

                if chosen is None:
                    failed.append(
                        {
                            "race_id": candidate.race_id,
                            "errors": [a.error for a in attempts if a.error],
                            "attempts": [
                                _source_result_diagnostic(attempt)
                                for attempt in attempts
                            ],
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
        if driver is not None:
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

    report = _build_report(
        args=args,
        db_path=db_path,
        upcoming_dir=upcoming_dir,
        snapshot_dir=snapshot_dir,
        write_approval=write_approval,
        candidates=candidates,
        skipped=skipped,
        ingested=ingested,
        failed=failed,
        backup_path=backup_path,
        dry_run_report_gate=dry_run_report_gate,
    )
    _write_output_report(report, args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
