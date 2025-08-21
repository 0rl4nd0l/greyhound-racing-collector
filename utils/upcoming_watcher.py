from __future__ import annotations

import time
import threading
from pathlib import Path
from typing import Optional, Callable, List

# NOTE: Avoid importing watchdog at module import time. Lazy import inside start_upcoming_watcher.
Observer = object  # type: ignore
FileSystemEventHandler = object  # type: ignore
FileSystemEvent = object  # type: ignore
WATCHDOG_AVAILABLE = False

from config.paths import UPCOMING_RACES_DIR

CSV_SUFFIX = ".csv"
PARTIAL_SUFFIXES = (".crdownload", ".part", ".tmp")


def is_partial_or_hidden(p: Path) -> bool:
    name = p.name.lower()
    if name.startswith("."):
        return True
    if any(name.endswith(suf) for suf in PARTIAL_SUFFIXES):
        return True
    return False


class Debounce:
    """Simple debounce helper to coalesce bursts of events.

    Calls the provided function once after a quiet period.
    """

    def __init__(self, wait_seconds: float, func: Callable[[], None]):
        self.wait_seconds = wait_seconds
        self.func = func
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None

    def trigger(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.wait_seconds, self._run)
            self._timer.daemon = True
            self._timer.start()

    def _run(self) -> None:
        try:
            self.func()
        finally:
            with self._lock:
                self._timer = None


class UpcomingHandler(FileSystemEventHandler):  # type: ignore[misc]
    def __init__(self, on_change: Callable[[List[Path]], None], debounce_seconds: float = 1.0):
        self._changed: List[Path] = []
        self._on_change = on_change
        self._debounce = Debounce(debounce_seconds, self._emit)

    def on_created(self, event: FileSystemEvent):  # type: ignore[override]
        self._maybe_collect(event)

    def on_moved(self, event: FileSystemEvent):  # type: ignore[override]
        self._maybe_collect(event)

    def on_modified(self, event: FileSystemEvent):  # type: ignore[override]
        self._maybe_collect(event)

    def _maybe_collect(self, event: FileSystemEvent) -> None:
        try:
            if getattr(event, "is_directory", False):
                return
            p = Path(getattr(event, "dest_path", None) or getattr(event, "src_path", ""))
            if not p or p.suffix.lower() != CSV_SUFFIX or is_partial_or_hidden(p):
                return
            # Collect file and debounce emit
            self._changed.append(p)
            self._debounce.trigger()
        except Exception:
            # Non-fatal; ignore event
            pass

    def _emit(self) -> None:
        # Snapshot and clear list
        changed: List[Path]
        changed = list({p.resolve() for p in self._changed})
        self._changed.clear()
        try:
            self._on_change(changed)
        except Exception:
            pass


def start_upcoming_watcher(
    upcoming_dir: Optional[Path] = None,
    on_change: Optional[Callable[[List[Path]], None]] = None,
    debounce_seconds: float = 1.0,
) -> Optional[Observer]:
    """
    Watch UPCOMING_RACES_DIR for new/updated CSVs and invoke on_change(changed_paths)
    after a debounced quiet period. Returns Observer if started, else None.
    """
    # Lazy import watchdog only if we intend to start the watcher
    global WATCHDOG_AVAILABLE, Observer, FileSystemEventHandler, FileSystemEvent
    if not WATCHDOG_AVAILABLE:
        try:
            from watchdog.observers import Observer as _Observer  # type: ignore
            from watchdog.events import FileSystemEventHandler as _FileSystemEventHandler, FileSystemEvent as _FileSystemEvent  # type: ignore
            Observer = _Observer
            FileSystemEventHandler = _FileSystemEventHandler
            FileSystemEvent = _FileSystemEvent
            WATCHDOG_AVAILABLE = True
        except Exception:
            print("[upcoming-watcher] watchdog not installed; skipping directory watcher.")
            return None

    upcoming_dir = upcoming_dir or UPCOMING_RACES_DIR

    def default_on_change(paths: List[Path]) -> None:
        # Default no-op that just logs to stdout
        names = ", ".join(p.name for p in paths)
        print(f"[upcoming-watcher] Detected changes: {names}")

    handler = UpcomingHandler(on_change or default_on_change, debounce_seconds)
    observer: Observer = Observer()  # type: ignore[assignment]
    observer.schedule(handler, str(upcoming_dir), recursive=False)
    observer.start()

    # Keep a small daemon to ensure longevity on some platforms
    def _guard():
        try:
            while observer.is_alive():
                time.sleep(1.0)
        except Exception:
            pass

    t = threading.Thread(target=_guard, daemon=True)
    t.start()

    print(f"[upcoming-watcher] Watching {upcoming_dir} for CSV changes...")
    return observer
