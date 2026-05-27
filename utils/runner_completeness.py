"""Runner-set completeness checks for live form-guide evidence.

The expert-form CSV format stores the target race field as the rows whose
``Dog Name`` value starts with a box number, while following rows contain that
dog's historical starts. These helpers deliberately count only target runners.
"""

from __future__ import annotations

import csv
import io
import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

try:
    import bs4
except Exception:
    bs4 = None


MIN_COMPLETE_RUNNERS = 4
PRIMARY_RUNNER_RE = re.compile(r"^\s*(\d{1,2})\s*[\.\):-]\s*(.+?)\s*$")
POST_RESULT_URL_MARKERS = ("result", "results", "dividend", "payout")
CSV_DELIMITERS = ",|;\t"


def normalise_runner_name(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def clean_runner_name(raw: Any) -> str:
    name = re.sub(r"^\s*\d{1,2}\s*[\.\):-]\s*", "", str(raw or "").strip())
    name = name.replace('"', "").replace("'", "").replace("`", "")
    return re.sub(r"\s+", " ", name).strip().title()


@dataclass(frozen=True)
class RunnerRow:
    box_number: int
    dog_name: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunnerCompleteness:
    schema_version: str
    status: str
    source: str | None
    runner_count: int
    min_complete_runners: int
    boxes: list[int]
    dog_names: list[str]
    participants: list[dict[str, Any]]
    duplicate_boxes: list[int]
    duplicate_dog_names: list[str]
    invalid_runner_rows: int
    reasons: list[str]

    @property
    def is_complete(self) -> bool:
        return self.status == "COMPLETE"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _header_value(row: Mapping[str, Any], names: Iterable[str]) -> Any:
    lookup = {str(key).strip().lower(): key for key in row.keys()}
    for name in names:
        key = lookup.get(name.lower())
        if key is not None:
            return row.get(key)
    return None


def detect_csv_delimiter(content: str) -> str:
    """Detect the form-guide delimiter without accepting mixed structures."""

    sample = str(content or "")[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=CSV_DELIMITERS)
        return dialect.delimiter
    except Exception:
        first_line = next((line for line in str(content or "").splitlines() if line.strip()), "")
        counts = {delimiter: first_line.count(delimiter) for delimiter in CSV_DELIMITERS}
        delimiter, count = max(counts.items(), key=lambda item: item[1])
        return delimiter if count > 0 else ","


def parse_runner_rows_from_text(content: str) -> tuple[list[RunnerRow], int]:
    """Return target-race runner rows and invalid target-like row count."""

    delimiter = detect_csv_delimiter(content)
    reader = csv.DictReader(io.StringIO(content), delimiter=delimiter)
    prefixed: list[RunnerRow] = []
    fallback: list[RunnerRow] = []
    invalid = 0

    for row in reader:
        raw_name = str(_header_value(row, ("Dog Name", "dog_name", "runner", "name")) or "").strip()
        if not raw_name or raw_name == '""':
            continue

        match = PRIMARY_RUNNER_RE.match(raw_name)
        if match:
            prefixed.append(
                RunnerRow(
                    box_number=int(match.group(1)),
                    dog_name=clean_runner_name(raw_name),
                )
            )
            continue

        box_value = _header_value(row, ("Box", "BOX", "box_number", "box"))
        if box_value not in (None, ""):
            try:
                fallback.append(
                    RunnerRow(
                        box_number=int(str(box_value).strip()),
                        dog_name=clean_runner_name(raw_name),
                    )
                )
            except (TypeError, ValueError):
                invalid += 1

    return (prefixed or fallback), invalid


def parse_runner_rows_from_csv(path: str | Path) -> list[RunnerRow]:
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    rows, _invalid = parse_runner_rows_from_text(text)
    return rows


def _duplicates(values: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    dupes: set[Any] = set()
    for value in values:
        if value in seen:
            dupes.add(value)
        seen.add(value)
    return sorted(dupes)


def analyze_runner_rows(
    rows: Iterable[RunnerRow],
    *,
    source: str | None = None,
    invalid_runner_rows: int = 0,
    min_complete_runners: int = MIN_COMPLETE_RUNNERS,
) -> RunnerCompleteness:
    unique: dict[tuple[int, str], RunnerRow] = {}
    for row in rows:
        unique.setdefault((row.box_number, normalise_runner_name(row.dog_name)), row)

    deduped = list(unique.values())
    boxes = [row.box_number for row in deduped]
    dog_names = [row.dog_name for row in deduped]
    duplicate_boxes = _duplicates(boxes)
    duplicate_dog_names = [
        name
        for name in _duplicates(normalise_runner_name(name) for name in dog_names)
        if name
    ]

    reasons: list[str] = []
    if not deduped:
        reasons.append("no_target_runner_rows")
    if len(deduped) < min_complete_runners:
        reasons.append(f"runner_count_below_min:{len(deduped)}<{min_complete_runners}")
    if duplicate_boxes:
        reasons.append(
            "duplicate_box_numbers:" + ",".join(str(value) for value in duplicate_boxes)
        )
    if duplicate_dog_names:
        reasons.append("duplicate_dog_names:" + ",".join(duplicate_dog_names))
    if invalid_runner_rows:
        reasons.append(f"invalid_runner_rows:{invalid_runner_rows}")

    return RunnerCompleteness(
        schema_version="runner_completeness_v1",
        status="COMPLETE" if not reasons else "INCOMPLETE",
        source=source,
        runner_count=len(deduped),
        min_complete_runners=min_complete_runners,
        boxes=sorted(boxes),
        dog_names=dog_names,
        participants=[row.as_dict() for row in deduped],
        duplicate_boxes=duplicate_boxes,
        duplicate_dog_names=duplicate_dog_names,
        invalid_runner_rows=invalid_runner_rows,
        reasons=reasons,
    )


def analyze_csv_runner_completeness(
    path: str | Path,
    *,
    min_complete_runners: int = MIN_COMPLETE_RUNNERS,
) -> RunnerCompleteness:
    path_obj = Path(path)
    try:
        text = path_obj.read_text(encoding="utf-8-sig", errors="replace")
    except Exception as exc:
        return RunnerCompleteness(
            schema_version="runner_completeness_v1",
            status="INCOMPLETE",
            source=str(path_obj),
            runner_count=0,
            min_complete_runners=min_complete_runners,
            boxes=[],
            dog_names=[],
            participants=[],
            duplicate_boxes=[],
            duplicate_dog_names=[],
            invalid_runner_rows=0,
            reasons=[f"csv_unreadable:{type(exc).__name__}"],
        )
    return analyze_csv_text_runner_completeness(
        text,
        source=str(path_obj),
        min_complete_runners=min_complete_runners,
    )


def analyze_csv_text_runner_completeness(
    content: str,
    *,
    source: str | None = None,
    min_complete_runners: int = MIN_COMPLETE_RUNNERS,
) -> RunnerCompleteness:
    rows, invalid = parse_runner_rows_from_text(content)
    return analyze_runner_rows(
        rows,
        source=source,
        invalid_runner_rows=invalid,
        min_complete_runners=min_complete_runners,
    )


def participants_from_runner_rows(rows: Iterable[RunnerRow]) -> list[dict[str, Any]]:
    participants: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for row in rows:
        key = (row.box_number, normalise_runner_name(row.dog_name))
        if key in seen:
            continue
        seen.add(key)
        participants.append({"box_number": row.box_number, "dog_name": row.dog_name})
    return participants


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(str(value).strip())
    except Exception:
        return None


def _int_list(values: Iterable[Any]) -> list[int]:
    output: list[int] = []
    for value in values or []:
        parsed = _safe_int(value)
        if parsed is not None:
            output.append(parsed)
    return sorted(set(output))


def _source_participants(source_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_participants = source_report.get("participants")
    participants: list[dict[str, Any]] = []
    if isinstance(raw_participants, list):
        for row in raw_participants:
            if not isinstance(row, Mapping):
                continue
            box = _safe_int(row.get("box_number") or row.get("box"))
            name = clean_runner_name(row.get("dog_name") or row.get("name") or "")
            if box is None or not name:
                continue
            participants.append({"box_number": box, "dog_name": name})
    if participants:
        return participants

    boxes = list(source_report.get("boxes") or [])
    names = list(source_report.get("dog_names") or [])
    for idx, box_value in enumerate(boxes):
        box = _safe_int(box_value)
        name = clean_runner_name(names[idx]) if idx < len(names) else ""
        if box is None:
            continue
        participants.append({"box_number": box, "dog_name": name})
    return participants


def _runner_row_box(row: Any) -> int | None:
    try:
        box_cell = row.select_one(".race-runners__box")
        if box_cell is not None:
            sprite = box_cell.select_one("sprite-svg[name]")
            if sprite is not None:
                match = re.search(r"rug_(\d{1,2})\b", str(sprite.get("name") or ""), re.I)
                if match:
                    return int(match.group(1))
            text_match = re.search(r"\b(\d{1,2})\b", box_cell.get_text(" ", strip=True))
            if text_match:
                return int(text_match.group(1))
        class_text = " ".join(row.get("class", [])) if isinstance(row.get("class"), list) else ""
        class_match = re.search(r"(?:rug|box)[_-](\d{1,2})\b", class_text, re.I)
        if class_match:
            return int(class_match.group(1))
    except Exception:
        return None
    return None


def _runner_row_name(row: Any) -> str:
    try:
        name_el = row.select_one(".race-runners__name__dog")
        if name_el is not None:
            # TheDogs nests time and reserve-box notes inside spans. Remove them
            # before reading the dog name text.
            clone = bs4.BeautifulSoup(str(name_el), "html.parser")
            clone_name = clone.select_one(".race-runners__name__dog") or clone
            for span in clone_name.select("span"):
                span.extract()
            return clean_runner_name(clone_name.get_text(" ", strip=True))
        name_cell = row.select_one(".race-runners__name")
        if name_cell is not None:
            text = name_cell.get_text(" ", strip=True)
            text = re.split(r"\bT:\s*", text, maxsplit=1)[0]
            text = re.sub(r"\b\d{2}\.\d{2}\b", "", text)
            text = re.sub(r"\(into box \d+\)", "", text, flags=re.I)
            return clean_runner_name(text)
    except Exception:
        return ""
    return ""


def _runner_row_into_box(row: Any) -> int | None:
    try:
        text = row.get_text(" ", strip=True)
    except Exception:
        return None
    match = re.search(r"\(\s*into\s+box\s+(\d{1,2})\s*\)", text, re.I)
    return int(match.group(1)) if match else None


def _runner_row_is_scratched(row: Any) -> bool:
    try:
        classes = row.get("class", [])
        if isinstance(classes, list) and any("scratch" in str(value).lower() for value in classes):
            return True
        odds_cell = row.select_one(".race-runners__odds")
        if odds_cell is not None and odds_cell.get_text(" ", strip=True).upper() == "SCR":
            return True
        return bool(re.search(r"\bSCR(?:ATCHED)?\b", row.get_text(" ", strip=True), re.I))
    except Exception:
        return False


def _extract_race_number_from_url(source_url: str | None) -> int | None:
    try:
        parts = [part for part in urlparse(str(source_url or "")).path.split("/") if part]
        racing_idx = parts.index("racing")
        return _safe_int(parts[racing_idx + 3])
    except Exception:
        return None


def _extract_race_number_from_page(soup: Any) -> int | None:
    try:
        headers = soup.select(".race-header")
        if len(headers) == 1:
            for element in headers[0].select(".race-box__number"):
                match = re.search(r"\bR?(\d{1,2})\b", element.get_text(" ", strip=True), re.I)
                if match:
                    return int(match.group(1))
    except Exception:
        return None
    return None


def _looks_post_result_url(source_url: str | None) -> bool:
    url = str(source_url or "").lower()
    if not url:
        return False
    try:
        parsed = urlparse(url)
        text = " ".join(part for part in (parsed.path, parsed.query, parsed.fragment) if part)
    except Exception:
        text = url
    tokens = {token for token in re.split(r"[^a-z0-9]+", text) if token}
    return any(marker in tokens for marker in POST_RESULT_URL_MARKERS)


def _canonical_unavailable(
    *,
    source_url: str | None,
    reason: str,
    extraction_timestamp: str | None = None,
    http_status: int | None = None,
) -> dict[str, Any]:
    report = {
        "schema_version": "canonical_pre_race_runner_set_v1",
        "canonical_runner_set_status": "unavailable",
        "final_runner_source": "canonical_pre_race_page",
        "final_runner_source_url": source_url,
        "final_runner_boxes": [],
        "final_runner_names": [],
        "final_runner_participants": [],
        "scratched_boxes": [],
        "scratched_participants": [],
        "reserve_boxes": [],
        "vacant_boxes": [],
        "race_number": _extract_race_number_from_url(source_url),
        "extraction_timestamp": extraction_timestamp or _iso_utc_now(),
        "unavailable_reason": reason,
    }
    if http_status is not None:
        report["http_status"] = http_status
    return report


def extract_canonical_runner_set_from_html(
    html: str,
    *,
    source_url: str | None = None,
    expected_race_number: int | None = None,
    extraction_timestamp: str | None = None,
) -> dict[str, Any]:
    """Extract final active runners from a canonical TheDogs pre-race page.

    This parser intentionally treats box 9/10 rows as reserves unless the page
    explicitly says the reserve moved into a real box, for example "(into box 4)".
    """

    timestamp = extraction_timestamp or _iso_utc_now()
    if _looks_post_result_url(source_url):
        return _canonical_unavailable(
            source_url=source_url,
            reason="post_result_url_refused",
            extraction_timestamp=timestamp,
        )
    if not html or not str(html).strip():
        return _canonical_unavailable(
            source_url=source_url,
            reason="empty_canonical_page",
            extraction_timestamp=timestamp,
        )
    if bs4 is None:
        return _canonical_unavailable(
            source_url=source_url,
            reason="beautifulsoup_unavailable",
            extraction_timestamp=timestamp,
        )

    soup = bs4.BeautifulSoup(html, "html.parser")
    expected = expected_race_number or _extract_race_number_from_url(source_url)
    page_race_number = _extract_race_number_from_page(soup)
    rows = soup.select("tr.race-runner")
    if not rows:
        return _canonical_unavailable(
            source_url=source_url,
            reason="no_race_runner_rows",
            extraction_timestamp=timestamp,
        )

    active: list[dict[str, Any]] = []
    scratched: list[dict[str, Any]] = []
    reserve_boxes: set[int] = set()
    vacant_boxes: set[int] = set()
    ambiguous_reasons: list[str] = []

    for row in rows:
        original_box = _runner_row_box(row)
        dog_name = _runner_row_name(row)
        into_box = _runner_row_into_box(row)
        if original_box is not None and original_box > 8:
            reserve_boxes.add(original_box)

        name_key = normalise_runner_name(dog_name)
        if original_box is not None and (not name_key or name_key in {"VACANT", "SCRATCHED"}):
            vacant_boxes.add(original_box)
            continue

        if original_box is None or not dog_name:
            ambiguous_reasons.append("runner_row_missing_box_or_name")
            continue

        participant = {"box_number": original_box, "dog_name": dog_name}
        if _runner_row_is_scratched(row):
            scratched.append(participant)
            continue

        if original_box > 8 and into_box is None:
            continue

        final_box = into_box or original_box
        active_participant = {"box_number": final_box, "dog_name": dog_name}
        if into_box is not None and into_box != original_box:
            active_participant["original_box_number"] = original_box
        active.append(active_participant)

    active_boxes = [row["box_number"] for row in active]
    duplicate_active_boxes = _duplicates(active_boxes)
    if duplicate_active_boxes:
        ambiguous_reasons.append(
            "duplicate_canonical_active_boxes:"
            + ",".join(str(value) for value in duplicate_active_boxes)
        )
    if expected is not None and page_race_number is not None and expected != page_race_number:
        ambiguous_reasons.append(
            f"race_number_mismatch:{page_race_number}!={expected}"
        )
    if not active:
        ambiguous_reasons.append("no_active_runners")

    status = "ambiguous" if ambiguous_reasons else "available"
    active_sorted = sorted(active, key=lambda item: (item["box_number"], item["dog_name"]))
    scratched_sorted = sorted(scratched, key=lambda item: (item["box_number"], item["dog_name"]))
    return {
        "schema_version": "canonical_pre_race_runner_set_v1",
        "canonical_runner_set_status": status,
        "final_runner_source": "canonical_pre_race_page",
        "final_runner_source_url": source_url,
        "final_runner_boxes": sorted(set(active_boxes)),
        "final_runner_names": [row["dog_name"] for row in active_sorted],
        "final_runner_participants": active_sorted,
        "scratched_boxes": _int_list(row["box_number"] for row in scratched_sorted),
        "scratched_participants": scratched_sorted,
        "reserve_boxes": sorted(reserve_boxes),
        "vacant_boxes": sorted(vacant_boxes),
        "race_number": page_race_number or expected,
        "expected_race_number": expected,
        "extraction_timestamp": timestamp,
        "ambiguous_reasons": ambiguous_reasons,
    }


def fetch_canonical_runner_set(
    source_url: str | None,
    *,
    session: Any = None,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Fetch and extract a canonical pre-race runner set.

    The caller is expected to pass a race URL captured before jump, not a
    results/dividends URL. Post-result-looking URLs are refused before request.
    """

    if not source_url:
        return _canonical_unavailable(source_url=source_url, reason="missing_canonical_url")
    if _looks_post_result_url(source_url):
        return _canonical_unavailable(source_url=source_url, reason="post_result_url_refused")
    try:
        if session is None:
            from utils.http_client import get_shared_session

            session = get_shared_session()
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.thedogs.com.au/racing",
        }
        try:
            session.headers.update(headers)
            response = session.get(source_url, timeout=timeout)
        except TypeError:
            response = session.get(source_url, timeout=timeout, headers=headers)
        status_code = getattr(response, "status_code", None)
        if status_code != 200:
            return _canonical_unavailable(
                source_url=source_url,
                reason=f"http_status:{status_code}",
                http_status=_safe_int(status_code),
            )
        return extract_canonical_runner_set_from_html(
            getattr(response, "text", "") or "",
            source_url=source_url,
        )
    except Exception as exc:
        return _canonical_unavailable(
            source_url=source_url,
            reason=f"fetch_error:{type(exc).__name__}",
        )


def verify_final_runner_set(
    source_report: Mapping[str, Any] | RunnerCompleteness | None,
    canonical_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(source_report, RunnerCompleteness):
        source_data = source_report.as_dict()
    else:
        source_data = dict(source_report or {})
    canonical = dict(canonical_report or {})

    source_participants = _source_participants(source_data)
    explicit_active_boxes = source_data.get("active_boxes")
    if explicit_active_boxes is None:
        source_active_boxes = _int_list(row["box_number"] for row in source_participants)
    else:
        source_active_boxes = _int_list(explicit_active_boxes)
    source_reserve_boxes = _int_list(
        source_data.get("reserve_boxes")
        or [box for box in source_active_boxes if box > 8]
    )
    source_scratch_boxes = _int_list(source_data.get("scratched_boxes") or [])

    canonical_status = str(canonical.get("canonical_runner_set_status") or "unavailable")
    canonical_active_boxes = _int_list(canonical.get("final_runner_boxes") or [])
    canonical_scratch_boxes = _int_list(canonical.get("scratched_boxes") or [])
    canonical_reserve_boxes = _int_list(canonical.get("reserve_boxes") or [])
    source_by_box = {
        row["box_number"]: row["dog_name"]
        for row in source_participants
        if row.get("box_number") in source_active_boxes
    }
    canonical_participants = [
        row
        for row in (canonical.get("final_runner_participants") or [])
        if isinstance(row, Mapping)
    ]
    canonical_by_box = {
        _safe_int(row.get("box_number")): clean_runner_name(row.get("dog_name") or "")
        for row in canonical_participants
        if _safe_int(row.get("box_number")) is not None
    }

    mismatch_reasons: list[str] = []
    status = "verified"
    if canonical_status in {"unavailable", "ambiguous"}:
        status = canonical_status
        reason = canonical.get("unavailable_reason") or ",".join(
            str(value) for value in canonical.get("ambiguous_reasons") or []
        )
        mismatch_reasons.append(reason or canonical_status)
    else:
        extra_source = sorted(set(source_active_boxes) - set(canonical_active_boxes))
        missing_source = sorted(set(canonical_active_boxes) - set(source_active_boxes))
        if extra_source:
            mismatch_reasons.append(
                "source_extra_active_boxes:" + ",".join(map(str, extra_source))
            )
        if missing_source:
            mismatch_reasons.append(
                "source_missing_active_boxes:" + ",".join(map(str, missing_source))
            )
        for box in sorted(set(source_active_boxes) & set(canonical_active_boxes)):
            source_name = source_by_box.get(box)
            canonical_name = canonical_by_box.get(box)
            if source_name and canonical_name and (
                normalise_runner_name(source_name) != normalise_runner_name(canonical_name)
            ):
                mismatch_reasons.append(
                    f"box_{box}_name_mismatch:{source_name}!={canonical_name}"
                )
        if mismatch_reasons:
            status = "mismatch"

    return {
        "schema_version": "final_runner_set_verification_v1",
        "final_runner_set_status": status,
        "final_runner_source": "canonical_pre_race_page",
        "final_runner_set_source": "canonical_pre_race_page",
        "final_runner_source_url": canonical.get("final_runner_source_url"),
        "final_runner_set_source_url": canonical.get("final_runner_source_url"),
        "final_runner_boxes": canonical_active_boxes,
        "final_runner_names": list(canonical.get("final_runner_names") or []),
        "final_runner_participants": list(canonical.get("final_runner_participants") or []),
        "scratched_boxes": canonical_scratch_boxes,
        "reserve_boxes": canonical_reserve_boxes,
        "vacant_boxes": _int_list(canonical.get("vacant_boxes") or []),
        "canonical_active_boxes": canonical_active_boxes,
        "source_active_boxes": source_active_boxes,
        "canonical_scratch_boxes": canonical_scratch_boxes,
        "source_scratch_boxes": source_scratch_boxes,
        "canonical_reserve_boxes": canonical_reserve_boxes,
        "source_reserve_boxes": source_reserve_boxes,
        "source_participants": source_participants,
        "canonical_runner_set_status": canonical_status,
        "canonical_evidence": canonical,
        "mismatch_reason": ";".join(mismatch_reasons) if mismatch_reasons else None,
        "extraction_timestamp": canonical.get("extraction_timestamp") or _iso_utc_now(),
    }


def canonical_race_url_from_sidecar(csv_path: str | Path) -> str | None:
    sidecar_path = Path(f"{csv_path}.metadata.json")
    if not sidecar_path.exists():
        return None
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    race_info = payload.get("race_info") if isinstance(payload.get("race_info"), Mapping) else {}
    for value in (
        payload.get("race_url"),
        race_info.get("url"),
        payload.get("metadata_source_url"),
    ):
        if value not in (None, ""):
            return str(value)
    return None


def analyze_prediction_runner_match(
    predictions: Iterable[Mapping[str, Any]],
    source_report: Mapping[str, Any] | RunnerCompleteness | None,
) -> dict[str, Any]:
    if isinstance(source_report, RunnerCompleteness):
        source_data = source_report.as_dict()
    else:
        source_data = dict(source_report or {})

    prediction_boxes: list[int] = []
    for row in predictions:
        box = row.get("box_number")
        try:
            if box is not None:
                prediction_boxes.append(int(box))
        except (TypeError, ValueError):
            continue

    source_boxes = [int(box) for box in source_data.get("boxes") or []]
    reasons: list[str] = []
    if source_boxes:
        missing = sorted(set(source_boxes) - set(prediction_boxes))
        extra = sorted(set(prediction_boxes) - set(source_boxes))
        if missing:
            reasons.append("prediction_missing_source_boxes:" + ",".join(map(str, missing)))
        if extra:
            reasons.append("prediction_extra_boxes:" + ",".join(map(str, extra)))
        if len(prediction_boxes) != len(source_boxes):
            reasons.append(
                f"prediction_runner_count_mismatch:{len(prediction_boxes)}!={len(source_boxes)}"
            )

    duplicate_prediction_boxes = _duplicates(prediction_boxes)
    if duplicate_prediction_boxes:
        reasons.append(
            "duplicate_prediction_boxes:"
            + ",".join(str(value) for value in duplicate_prediction_boxes)
        )

    return {
        "schema_version": "prediction_runner_match_v1",
        "status": "MATCHED" if not reasons else "MISMATCH",
        "source_runner_count": len(source_boxes) if source_boxes else None,
        "prediction_runner_count": len(prediction_boxes),
        "source_boxes": sorted(source_boxes),
        "prediction_boxes": sorted(prediction_boxes),
        "reasons": reasons,
    }


def quarantine_existing_file(path: str | Path, *, reason: str) -> Path:
    path_obj = Path(path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine_dir = path_obj.parent / "quarantine"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    target = quarantine_dir / f"{timestamp}_{path_obj.stem}_{reason}{path_obj.suffix}"
    shutil.move(str(path_obj), str(target))
    return target


def quarantine_csv_content(
    content: str,
    destination_dir: str | Path,
    filename: str,
    *,
    reason: str,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine_dir = Path(destination_dir) / "quarantine"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    target = quarantine_dir / f"{timestamp}_{Path(filename).stem}_{reason}.csv"
    target.write_text(content, encoding="utf-8", newline="")
    return target
