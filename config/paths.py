import os
from pathlib import Path


def _bound_connected_dir(name: str) -> Path | None:
    if os.environ.get("OPERATOR_UI_R3_PROFILE") != "repository-v1":
        return None
    from src.operator_ui.deployment import bound_operator_ui_runtime_dir

    return bound_operator_ui_runtime_dir(name)


def get_dir(name: str, default: str) -> Path:
    """Resolve and ensure a directory from env var with fallback default.

    - Expands ~, resolves to absolute path
    - Creates the directory if it doesn't exist
    - Returns a Path object
    """
    bound = _bound_connected_dir(name)
    if bound is not None:
        return bound
    p = Path(os.getenv(name, default)).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


# Base data directory holding subfolders used by the app and pipelines
DATA_DIR: Path = get_dir("DATA_DIR", "./data")

# Canonical target for upcoming race CSVs used by the UI/API (race data)
# Default aligns with repo docs and WARP.md
UPCOMING_RACES_DIR: Path = get_dir(
    "UPCOMING_RACES_DIR", "./upcoming_races_temp"
)

# Race-day outcomes, weather, winners; separate from form guides (race data)
RACE_DATA_DIR: Path = get_dir("RACE_DATA_DIR", str(DATA_DIR / "race_data"))

# Archive directory for moving old or redundant files (follows archive-first policy)
ARCHIVE_DIR: Path = get_dir("ARCHIVE_DIR", "./archive")

# Optional OS Downloads watch directory (not created by default; just resolved)
DOWNLOADS_WATCH_DIR: Path = _bound_connected_dir("DOWNLOADS_WATCH_DIR") or (
    Path(os.getenv("DOWNLOADS_WATCH_DIR", str(Path.home() / "Downloads")))
    .expanduser()
    .resolve()
)
