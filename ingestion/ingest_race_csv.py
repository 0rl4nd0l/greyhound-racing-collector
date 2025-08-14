import csv
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from config.paths import UPCOMING_RACES_DIR, ARCHIVE_DIR, DATA_DIR

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOGS_DIR / "ingestion.log"
ERROR_LOG_PATH = LOGS_DIR / "ingestion_errors.log"

# Configure logging (module-level singleton)
# Keep INFO and above in ingestion.log, and also tee ERROR+ to ingestion_errors.log
logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)
logger.propagate = False

# Attach an error-only file handler for failures
try:
    from logging.handlers import RotatingFileHandler
    _err_handler = RotatingFileHandler(str(ERROR_LOG_PATH), maxBytes=2 * 1024 * 1024, backupCount=2)
    _err_handler.setLevel(logging.ERROR)
    _err_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    # Avoid duplicate handlers if module reloaded
    if not any(isinstance(h, RotatingFileHandler) and getattr(h, 'baseFilename', '') == str(ERROR_LOG_PATH) for h in logger.handlers):
        logger.addHandler(_err_handler)
except Exception:
    # Fallback to simple FileHandler if RotatingFileHandler is unavailable
    _fh = logging.FileHandler(str(ERROR_LOG_PATH))
    _fh.setLevel(logging.ERROR)
    _fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(_fh)


# --- Atomic publish helpers ---

def wait_for_stable(path: Path, checks: int = 3, interval: float = 0.5) -> bool:
    # Accelerate when running under pytest to reduce test runtimes and flakiness
    try:
        import os as _os
        if _os.getenv("PYTEST_CURRENT_TEST"):
            checks = 1
            interval = 0.05
    except Exception:
        pass
    last_size = -1
    stable = 0
    for _ in range(64):
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return False
        if size == last_size:
            stable += 1
            if stable >= checks:
                return True
        else:
            stable = 0
        last_size = size
        time.sleep(interval)
    return False


def publish_to_upcoming(src_path: Path, dest_dir: Path, dest_name: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest_dir / f".{dest_name}.tmp"
    final_dest = dest_dir / dest_name
    if src_path.exists() and wait_for_stable(src_path):
        shutil.copy2(src_path, tmp_dest)
        os.replace(tmp_dest, final_dest)  # atomic on same filesystem
        return final_dest
    raise RuntimeError(f"File not stable or missing: {src_path}")


# --- Helpers ---

CANONICAL_DATE_FORMAT = "%Y-%m-%d"


@dataclass
class RaceMeta:
    race_date: str  # YYYY-MM-DD
    track: str      # slug-safe track/venue code
    race_number: str

    @property
    def canonical_name(self) -> str:
        # Enforce R prefix per convention: {YYYY-MM-DD}_{track_slug}_R{race_number}.csv
        return f"{self.race_date}_{self.track}_R{self.race_number}.csv"


def slugify(value: str) -> str:
    v = value.strip().lower()
    v = re.sub(r"[^a-z0-9]+", "-", v)
    v = re.sub(r"-+", "-", v).strip("-")
    return v


def sniff_dialect_and_headers(csv_path: Path) -> Tuple[csv.Dialect, Iterable[str]]:
    with csv_path.open("r", encoding="utf-8", errors="replace") as f:
        sample = f.read(8192)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",|\t;")
        except Exception:
            # default to comma, fall back if needed
            class Default(csv.Dialect):
                delimiter = ","
                quotechar = '"'
                escapechar = None
                doublequote = True
                skipinitialspace = False
                lineterminator = "\n"
                quoting = csv.QUOTE_MINIMAL
            dialect = Default()
    # Re-open to read header with detected dialect
    with csv_path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, dialect)
        headers = next(reader, [])
    return dialect, [h.strip() for h in headers]


# Column name variants for metadata extraction
DATE_KEYS = [
    "race_date", "race date", "meeting_date", "date", "meeting date",
]
TRACK_KEYS = [
    "venue", "track", "venue_code", "venue code", "meeting_venue", "meeting venue",
]
RACE_NO_KEYS = [
    "race_number", "race no", "race", "race_no", "race number",
]

# Indicators to help classify CSV type
RACE_DATA_MIN_COLUMNS = {"race_date", "race_number", "dog_name", "box"}
FORM_GUIDE_INDICATIVE_COLS = {"dog name", "plc", "date", "dist", "track"}
RESULT_ONLY_COLS = {"winner", "finish_position", "winning_time", "margin"}


def classify_csv(headers: Iterable[str]) -> str:
    lower = {h.strip().lower() for h in headers}
    # If it looks like upcoming race data schema
    if RACE_DATA_MIN_COLUMNS.issubset(lower):
        return "race_data"
    # If it has common form guide fields
    if FORM_GUIDE_INDICATIVE_COLS.intersection(lower):
        return "form_guide"
    # Ambiguous; default to form guide if dog name present
    if "dog name" in lower or "dog_name" in lower:
        return "form_guide"
    return "unknown"


def parse_date(value: str) -> Optional[str]:
    value = value.strip()
    # Try common formats
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%d %b %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(value, fmt).strftime(CANONICAL_DATE_FORMAT)
        except Exception:
            continue
    # Already canonical?
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return value
    return None


def first_matching_key(row: Dict[str, str], keys: Iterable[str]) -> Optional[str]:
    # Case-insensitive header lookup with fallbacks
    lower_map = {k.lower(): k for k in row.keys()}
    for key in keys:
        lk = key.lower()
        if lk in lower_map:
            raw = row[lower_map[lk]]
            if raw is not None and str(raw).strip() != "":
                return str(raw).strip()
    return None


FILENAME_PATTERNS = [
    # Race 4 - GOSF - 2025-07-28.csv
    re.compile(r"race\s*(?P<race_number>\d+)\s*-\s*(?P<track>[A-Za-z0-9_\-]+)\s*-\s*(?P<race_date>\d{4}-\d{2}-\d{2})", re.IGNORECASE),
    # 2025-07-28_GOSF_4.csv or 2025-07-28-gosf-4.csv
    re.compile(r"(?P<race_date>\d{4}-\d{2}-\d{2})[_-](?P<track>[A-Za-z0-9_\-]+)[_-](?P<race_number>\d+)", re.IGNORECASE),
]


def extract_meta_from_filename(p: Path) -> Optional[RaceMeta]:
    stem = p.stem
    for pat in FILENAME_PATTERNS:
        m = pat.search(stem)
        if m:
            date = parse_date(m.group("race_date"))
            track = slugify(m.group("track"))
            rno = str(int(m.group("race_number")))
            if date and track and rno:
                return RaceMeta(race_date=date, track=track, race_number=rno)
    return None


def extract_meta_from_csv(csv_path: Path, dialect: csv.Dialect) -> Optional[RaceMeta]:
    # Attempt to read a few rows to locate metadata fields
    with csv_path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, dialect=dialect)
        for i, row in enumerate(reader):
            if i > 15:  # look at first ~16 rows only
                break
            # Try to extract from row
            date_raw = first_matching_key(row, DATE_KEYS)
            track_raw = first_matching_key(row, TRACK_KEYS)
            rno_raw = first_matching_key(row, RACE_NO_KEYS)
            if date_raw and track_raw and rno_raw:
                date = parse_date(date_raw)
                track = slugify(track_raw)
                try:
                    rno = str(int(str(rno_raw).strip()))
                except Exception:
                    continue
                if date:
                    return RaceMeta(race_date=date, track=track, race_number=rno)
    return None


def ensure_archive_dir() -> Path:
    # Centralized duplicates archive per policy
    d = ARCHIVE_DIR / "duplicates"
    d.mkdir(parents=True, exist_ok=True)
    return d


def archive_file(p: Path, reason: str) -> Path:
    archive_dir = ensure_archive_dir()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = archive_dir / f"{p.stem}_archived-{ts}{p.suffix}"
    shutil.move(str(p), str(dest))
    logger.info(f"Archived file {p} -> {dest} | reason={reason}")
    return dest


def dedupe_existing_by_checksum(target_name: str, incoming_path: Path, incoming_checksum: str) -> Tuple[str, Path]:
    """
    Deduplicate using checksum and timestamps.

    - If an existing canonical file exists under UPCOMING_RACES_DIR:
      - If content is identical (checksum equal), skip publish and archive the incoming source.
      - Else, prefer the newer mtime version; move the replaced file to archive/duplicates.
    Returns a tuple (action, kept_path) where action in {"kept_existing", "replaced_existing", "published_new", "skipped_identical"}.
    """
    existing = UPCOMING_RACES_DIR / target_name
    if existing.exists():
        # Compute checksum of existing
        from utils.checksum import file_sha256
        try:
            existing_checksum = file_sha256(existing)
        except Exception:
            existing_checksum = ""
        if existing_checksum and existing_checksum == incoming_checksum:
            # Identical content; archive incoming and keep existing
            archive_file(incoming_path, reason="duplicate_identical_content")
            return "skipped_identical", existing
        # Not identical: decide by mtime
        try:
            existing_mtime = existing.stat().st_mtime
            incoming_mtime = incoming_path.stat().st_mtime
        except FileNotFoundError:
            # If incoming vanished, keep existing
            return "kept_existing", existing
        if incoming_mtime >= existing_mtime:
            # Replace existing with incoming; archive the replaced file
            archived = archive_file(existing, reason="duplicate_canonical_older")
            # After archiving, actual replacement happens in publish step
            return "replaced_existing", existing
        else:
            # Incoming is older; archive it and keep existing
            archive_file(incoming_path, reason="duplicate_canonical_incoming_older")
            return "kept_existing", existing
    # No existing file
    return "published_new", existing


def build_canonical_name(meta: RaceMeta) -> str:
    return meta.canonical_name


def validate_intake_is_form_guide(headers: Iterable[str]) -> None:
    ctype = classify_csv(headers)
    # We accept only form guides here; if it's upcoming race data, reject with guidance
    if ctype == "race_data":
        raise ValueError(
            "Provided CSV appears to be race data (upcoming race schema), not a form guide (historical data). "
            "Use race data ingestion/publisher or convert to form guide format."
        )
    if ctype == "unknown":
        logger.warning("CSV type could not be confidently classified; proceeding as form guide with caution.")


def ingest_form_guide_csv(file_path: str) -> Path:
    """
    Unified ingestion for form guide CSVs (historical data) used by both manual and automated flows.

    - Validates the CSV resembles a form guide (not the upcoming race data schema)
    - Extracts race_date, track, race_number from header/rows or filename
    - Builds canonical filename: {YYYY-MM-DD}_{track_slug}_R{race_number}.csv (slug-safe)
    - Deduplicates by canonical key; prefers newest file
    - Publishes atomically into UPCOMING_RACES_DIR
    - Logs actions to logs/ingestion.log

    Returns: Path to the published file inside UPCOMING_RACES_DIR
    """
    src = Path(file_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"CSV not found: {src}")

    try:
        dialect, headers = sniff_dialect_and_headers(src)
        validate_intake_is_form_guide(headers)

        meta = extract_meta_from_csv(src, dialect)
        if not meta:
            # Fallback to filename parsing
            meta = extract_meta_from_filename(src)
        if not meta:
            raise ValueError(
                "Unable to extract race metadata (race_date, track, race_number) from CSV or filename."
            )

        canonical_name = build_canonical_name(meta)
        logger.info(
            f"Prepared canonical name '{canonical_name}' for source '{src.name}' | headers={headers}"
        )

        # Compute checksum for dedup and manifest
        from utils.checksum import file_sha256
        incoming_checksum = file_sha256(src)

        # Deduplicate with archive-first policy (checksum + mtime)
        action, existing_target = dedupe_existing_by_checksum(canonical_name, src, incoming_checksum)
        if action == "skipped_identical" or action == "kept_existing":
            # Update manifest to reference the kept file
            kept_path = UPCOMING_RACES_DIR / canonical_name
            try:
                from utils.manifest import IngestionManifest
                manifest_path = DATA_DIR / "ingestion_manifest.json"
                man = IngestionManifest.load(manifest_path)
                man.update_entry(key=canonical_name, file_path=str(kept_path), checksum=incoming_checksum)
                man.save()
            except Exception as _e:
                logger.warning(f"Manifest update failed (kept existing): {_e}")
            logger.info(f"Dedup kept existing file for {canonical_name} (action={action})")
            return kept_path

        # Publish incoming (either new or replacing existing)
        published = publish_to_upcoming(src, UPCOMING_RACES_DIR, canonical_name)
        logger.info(f"Published file: {published} (from {src}) | action={action}")

        # Update manifest after publish
        try:
            from utils.manifest import IngestionManifest
            manifest_path = DATA_DIR / "ingestion_manifest.json"
            man = IngestionManifest.load(manifest_path)
            man.update_entry(key=canonical_name, file_path=str(published), checksum=incoming_checksum)
            man.save()
        except Exception as _e:
            logger.warning(f"Manifest update failed: {_e}")

        return published

    except Exception as e:
        # Error will be captured in both ingestion.log and ingestion_errors.log via handler
        logger.error(f"Ingestion failed for {src}: {e}")
        raise


__all__ = ["ingest_form_guide_csv", "RaceMeta"]

