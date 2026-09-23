"""Read only operational attempt accounting under an already owned collector lock.

Never reads form/source bodies, results, model artifacts or research datasets.
The caller freezes the authoritative root inventory; missing/unreadable roots or
unclassifiable attempted identities stop admission. Output contains identities,
consumption and hashes only, not odds or embedded acquisition payloads.
"""

import hashlib
import json
import sqlite3
from pathlib import Path
from urllib.parse import quote

from race_collection.live_freshness_contract import digest, reconcile_projections


DOMAINS = (
    "scheduled_progress",
    "scheduled_reports",
    "manual_claims",
    "manual_attempts",
    "phase_checkpoints",
    "prior_rehearsals",
    "live_odds",
)
PATTERNS = {
    "scheduled_progress": ("**/autonomous_live_odds_capture_attempts.progress.jsonl",),
    "scheduled_reports": (
        "**/autonomous_live_odds_capture_attempts.jsonl",
        "**/autonomous_live_odds_capture_report.json",
    ),
    "phase_checkpoints": ("**/phase-checkpoint.json", "**/*.live-phase-checkpoint.json"),
    "prior_rehearsals": (
        "**/capture-reservation.json",
        "**/phase-checkpoint.json",
        "**/autonomous_live_odds_capture_attempts*.jsonl",
    ),
}
NON_ATTEMPTS = {
    "PLANNED_NOT_EXECUTED",
    "SKIPPED_NOT_READY",
    "BLOCKED_AUTO_SCRAPE_NOT_APPROVED",
    "BLOCKED_TIME_GATE_BEFORE_FETCH",
}


def reconcile(*, roots, db_path, source_date, lock_path, owner_run_id):
    owner = json.loads(Path(lock_path).read_bytes())
    if owner.get("run_id") != owner_run_id or not owner.get("pid"):
        raise ValueError("reconciliation_requires_owned_lock")
    if set(roots) != set(DOMAINS) - {"live_odds"}:
        raise ValueError("reconciliation_roots_incomplete")
    projections = {}
    for domain, paths in roots.items():
        if not paths:
            raise ValueError("reconciliation_roots_missing")
        records, inventory = [], []
        for raw_root in paths:
            root = Path(raw_root).resolve(strict=True)
            if not root.is_dir():
                raise ValueError("reconciliation_root_invalid")
            if domain.startswith("manual_"):
                kind = "claims" if domain == "manual_claims" else "attempts"
                # A missing initialized protocol directory is not empty accounting.
                directory = root / kind
                if not directory.is_dir() or not (root / "requests").is_dir():
                    raise ValueError("manual_accounting_incomplete")
                files = sorted(directory.glob("*.json"))
            else:
                files = sorted({p for pattern in PATTERNS[domain] for p in root.glob(pattern)})
            inventory.append(
                {"root": str(root), "files": [str(p.relative_to(root)) for p in files]}
            )
            for path in files:
                if path.is_symlink() or path.stat().st_size > 32 * 1024 * 1024:
                    raise ValueError("attempt_record_unbounded")
                raw = path.read_bytes()
                inventory.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
                if domain.startswith("manual_"):
                    # Claimed/started manual work conservatively consumes all windows.
                    record = json.loads(raw)
                    request = root / "requests" / f"{record['request_id']}.json"
                    request_raw = request.read_bytes()
                    inventory.append(
                        {"path": str(request), "sha256": hashlib.sha256(request_raw).hexdigest()}
                    )
                    race = json.loads(request_raw)["race"]
                    if race["race_date"] == source_date:
                        records.extend(
                            {"race_id": race["race_id"], "capture_window_minutes": window}
                            for window in (60, 30, 10, 2)
                        )
                    continue
                rows = (
                    [json.loads(line) for line in raw.splitlines()]
                    if path.suffix == ".jsonl"
                    else [json.loads(raw)]
                )
                for payload in rows:
                    if payload.get("schema_version") == "freshness_capture_reservation_v1":
                        attempts = [payload["item"]]
                    elif "phases" in payload:
                        attempts = [
                            phase["inputs"]
                            for phase in payload["phases"]
                            if phase["kind"] == "capture"
                        ]
                    elif "attempts" in payload:
                        attempts = payload["attempts"]
                    elif payload.get("schema_version") == "autonomous_live_odds_capture_attempt_v1":
                        attempts = [payload]
                    else:
                        raise ValueError("unknown_attempt_record")
                    for row in attempts:
                        if row.get("status") in NON_ATTEMPTS:
                            continue
                        race_id = row.get("race_id")
                        if not isinstance(race_id, str):
                            raise ValueError("ambiguous_attempt_identity")
                        if source_date not in race_id:
                            continue
                        window = row.get("capture_window_minutes")
                        # Old checkpoints did not bind a window: exclude all four.
                        windows = (window,) if window in (60, 30, 10, 2) else (60, 30, 10, 2)
                        records.extend(
                            {"race_id": race_id, "capture_window_minutes": w} for w in windows
                        )
        projections[domain] = {
            "complete": True,
            "consumed": records,
            "inventory_sha256": digest(inventory),
            "inventory": inventory,
        }
    db_path = Path(db_path).resolve(strict=True)
    uri = "file:" + quote(str(db_path)) + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=3)
    try:
        connection.execute("PRAGMA query_only=ON")
        # No prices, runner history, labels or results are selected.
        rows = connection.execute(
            "SELECT DISTINCT race_id, capture_mode FROM live_odds WHERE race_id LIKE ?",
            ("%" + source_date,),
        ).fetchall()
    finally:
        connection.close()
    records = []
    for race_id, mode in rows:
        if isinstance(mode, str) and mode.startswith("autonomous_prejump_t") and mode.endswith("m"):
            window = int(mode[len("autonomous_prejump_t") : -1])
            windows = (window,) if window in (60, 30, 10, 2) else (60, 30, 10, 2)
        else:
            windows = (60, 30, 10, 2)
        records.extend({"race_id": race_id, "capture_window_minutes": w} for w in windows)
    projections["live_odds"] = {
        "complete": True,
        "consumed": records,
        "inventory_sha256": digest(rows),
    }
    if json.loads(Path(lock_path).read_bytes()).get("run_id") != owner_run_id:
        raise ValueError("reconciliation_lock_changed")
    result = reconcile_projections(projections)
    result["source_date"] = source_date
    result["root_inventory"] = roots
    result["projections"] = projections
    return result
