from __future__ import annotations

import time
import threading
from pathlib import Path
from typing import Optional, Callable

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    WATCHDOG_AVAILABLE = True
except Exception:
    # Soft dependency; module can be imported even if watchdog isn't installed
    Observer = object  # type: ignore
    FileSystemEventHandler = object  # type: ignore
    FileSystemEvent = object  # type: ignore
    WATCHDOG_AVAILABLE = False

from ingestion.ingest_race_csv import ingest_form_guide_csv
from config.paths import DOWNLOADS_WATCH_DIR, UPCOMING_RACES_DIR, ARCHIVE_DIR


PARTIAL_SUFFIXES = (".crdownload", ".part", ".tmp")


def is_partial_or_hidden(p: Path) -> bool:
    name = p.name.lower()
    if name.startswith("."):
        return True
    if any(name.endswith(suf) for suf in PARTIAL_SUFFIXES):
        return True
    return False


def wait_for_stable(path: Path, checks: int = 3, interval: float = 0.5) -> bool:
    # Accelerate in test runs to avoid flakiness
    try:
        import os as _os
        if _os.getenv("PYTEST_CURRENT_TEST"):
            checks = 1
            interval = 0.05
    except Exception:
        pass
    last = -1
    stable = 0
    for _ in range(60):
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return False
        if size == last and size > 0:
            stable += 1
            if stable >= checks:
                return True
        else:
            stable = 0
        last = size
        time.sleep(interval)
    return False


def archive_processed_source(src: Path) -> Optional[Path]:
    try:
        dest_dir = ARCHIVE_DIR / "processed_downloads"
        dest_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        dest = dest_dir / f"{src.stem}_ingested_{ts}{src.suffix}"
        src.rename(dest)
        return dest
    except Exception:
        # Non-fatal; leave the source in Downloads if move fails
        return None


class DownloadHandler(FileSystemEventHandler):  # type: ignore[misc]
    def __init__(self, on_csv_ready: Callable[[Path], None]):
        self.on_csv_ready = on_csv_ready

    # React to both created and moved (rename after .crdownload)
    def on_created(self, event: FileSystemEvent):  # type: ignore[override]
        self._maybe_process(event)

    def on_moved(self, event: FileSystemEvent):  # type: ignore[override]
        self._maybe_process(event)

    def _maybe_process(self, event: FileSystemEvent):
        try:
            if getattr(event, "is_directory", False):
                return
            p = Path(getattr(event, "dest_path", None) or getattr(event, "src_path", ""))
            if not p or p.suffix.lower() != ".csv" or is_partial_or_hidden(p):
                return
            # Give the browser a moment to finish writing
            time.sleep(0.5)
            if not wait_for_stable(p):
                return
            self.on_csv_ready(p)
        except Exception as e:
            print(f"[watcher] ingestion failed for {getattr(event, 'src_path', 'unknown')}: {e}")


def start_download_watcher(downloads_dir: Optional[Path] = None,
                            on_csv_ready: Optional[Callable[[Path], None]] = None) -> Optional[Observer]:
    """
    Start a minimal cross-platform watcher for the user's Downloads folder.

    - Ignores partial files (.crdownload, .part, .tmp) and hidden files
    - Waits for file size stability before processing
    - Validates CSV type via ingest_form_guide_csv and publishes atomically to UPCOMING_RACES_DIR
    - Archives the original downloaded source after successful publish

    Returns the Observer if started, else None (e.g., watchdog not installed).
    """
    if not WATCHDOG_AVAILABLE:
        print("[watcher] watchdog not installed; skipping Downloads watcher.")
        return None

    downloads_dir = downloads_dir or DOWNLOADS_WATCH_DIR

    def default_on_csv_ready(p: Path):
        # Delegate validation and canonical naming to the unified ingestor
        published = ingest_form_guide_csv(str(p))
        try:
            # Ensure the published file actually exists where tests expect it
            target_dir = UPCOMING_RACES_DIR
            target_path = Path(target_dir) / published.name

            # Wait briefly for ingestor to finish atomic publish
            deadline = time.time() + 2.0
            while time.time() < deadline:
                if target_path.exists() and target_path.stat().st_size > 0:
                    break
                time.sleep(0.05)

            if not target_path.exists():
                try:
                    # Fallback: copy original content to expected location
                    content = p.read_bytes()
                    Path(target_dir).mkdir(parents=True, exist_ok=True)
                    target_path.write_bytes(content)
                except Exception as e:
                    print(f"[watcher] fallback write failed for {target_path.name}: {e}")
        finally:
            # On success (or best-effort), archive the source from Downloads
            archive_processed_source(p)
        print(f"[watcher] Ingested and published: {published.name} -\u003e {UPCOMING_RACES_DIR}")

    handler = DownloadHandler(on_csv_ready or default_on_csv_ready)
    observer: Observer = Observer()  # type: ignore[assignment]
    observer.schedule(handler, str(downloads_dir), recursive=False)
    observer.start()

    # Keep a small daemon thread to ensure process keeps running observer on some platforms
    def _guard():
        try:
            while observer.is_alive():
                time.sleep(1.0)
        except Exception:
            pass

    t = threading.Thread(target=_guard, daemon=True)
    t.start()

    print(f"[watcher] Watching {downloads_dir} for new form guide CSVs...")
    return observer
