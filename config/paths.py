import os
from pathlib import Path


def get_dir(name: str, default: str) -> Path:
    """Resolve and ensure a directory from env var with fallback default.

    - Expands ~, resolves to absolute path
    - Creates the directory if it doesn't exist
    - Returns a Path object
    """
    p = Path(os.getenv(name, default)).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


# Base data directory holding subfolders used by the app and pipelines
DATA_DIR: Path = get_dir("DATA_DIR", "./data")

# Canonical target for form guide CSVs used by predictions (historical data)
UPCOMING_RACES_DIR: Path = get_dir(
    "UPCOMING_RACES_DIR", str(DATA_DIR / "upcoming_races")
)

# Race-day outcomes, weather, winners; separate from form guides (race data)
RACE_DATA_DIR: Path = get_dir("RACE_DATA_DIR", str(DATA_DIR / "race_data"))

# Archive directory for moving old or redundant files (follows archive-first policy)
ARCHIVE_DIR: Path = get_dir("ARCHIVE_DIR", "./archive")

# Optional OS Downloads watch directory (not created by default; just resolved)
DOWNLOADS_WATCH_DIR: Path = (
    Path(os.getenv("DOWNLOADS_WATCH_DIR", str(Path.home() / "Downloads")))
    .expanduser()
    .resolve()
)
