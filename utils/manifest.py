import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional


@dataclass
class ManifestEntry:
    file_path: str
    checksum: str


class IngestionManifest:
    """
    A simple JSON-backed manifest mapping canonical key -e {file_path, checksum}.

    Stored at DATA_DIR/ingestion_manifest.json by convention.
    """

    def __init__(self, entries: Optional[Dict[str, ManifestEntry]] = None):
        self._entries: Dict[str, ManifestEntry] = entries or {}

    @staticmethod
    def load(path: Path) -> "IngestionManifest":
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                entries = {k: ManifestEntry(**v) for k, v in data.items()}
                return IngestionManifest(entries)
        except Exception:
            pass
        return IngestionManifest()

    def save(self, path: Optional[Path] = None) -> None:
        # Default to DATA_DIR/ingestion_manifest.json if not specified
        from config.paths import DATA_DIR
        path = path or (DATA_DIR / "ingestion_manifest.json")
        serializable = {k: asdict(v) for k, v in self._entries.items()}
        path.write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")

    def update_entry(self, key: str, file_path: str, checksum: str) -> None:
        self._entries[key] = ManifestEntry(file_path=file_path, checksum=checksum)

    def get(self, key: str) -> Optional[ManifestEntry]:
        return self._entries.get(key)

    def remove(self, key: str) -> None:
        if key in self._entries:
            del self._entries[key]

