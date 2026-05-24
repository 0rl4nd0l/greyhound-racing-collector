import json
from pathlib import Path

import pytest

# We import modules inside tests after setting env to ensure they pick up temp dirs

FORM_GUIDE_CSV_CONTENT = """Race Date,Venue,Race Number,Dog Name,Box Number,Trainer
2025-08-12,GOSF,4,Dog A,1,Trainer A
2025-08-12,GOSF,4,Dog B,2,Trainer B
"""

# This CSV lacks metadata headers; name must be parsed from filename
FILENAME_ONLY_CONTENT = """Dog Name,PLC,Date,Dist,Track
Dog X,1,2025-08-15,520m,GOSF
Dog Y,2,2025-08-15,520m,GOSF
"""


@pytest.fixture()
def temp_env_dirs(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    upcoming_dir = data_dir / "upcoming_races"
    archive_dir = tmp_path / "archive"
    data_dir.mkdir(parents=True, exist_ok=True)
    upcoming_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("UPCOMING_RACES_DIR", str(upcoming_dir))
    monkeypatch.setenv("ARCHIVE_DIR", str(archive_dir))

    # Reload config.paths to pick up env changes
    import importlib

    if "config.paths" in list(importlib.sys.modules.keys()):
        importlib.reload(importlib.import_module("config.paths"))

    yield {
        "DATA_DIR": data_dir,
        "UPCOMING_RACES_DIR": upcoming_dir,
        "ARCHIVE_DIR": archive_dir,
    }


def _fresh_ingestor():
    # Ensure fresh import of ingestor to read updated paths
    import importlib

    if "ingestion.ingest_race_csv" in list(importlib.sys.modules.keys()):
        importlib.reload(importlib.import_module("ingestion.ingest_race_csv"))
    from ingestion.ingest_race_csv import ingest_form_guide_csv

    return ingest_form_guide_csv


def test_classification_rejects_race_data_like_schema(
    tmp_path, monkeypatch, temp_env_dirs
):
    # A CSV that resembles upcoming race data (has dog_name + box exact keys)
    bad_content = "Dog_Name,box,race_date,race_number\nA,1,2025-08-12,3\n"
    src = tmp_path / "bad.csv"
    src.write_text(bad_content, encoding="utf-8")

    ingest_form_guide_csv = _fresh_ingestor()

    with pytest.raises(ValueError):
        ingest_form_guide_csv(str(src))


def test_canonical_naming_from_csv_headers(tmp_path, temp_env_dirs):
    # CSV has metadata headers; should build canonical name YYYY-MM-DD_track_RN.csv
    src = tmp_path / "random_upload.csv"
    src.write_text(FORM_GUIDE_CSV_CONTENT, encoding="utf-8")

    ingest_form_guide_csv = _fresh_ingestor()
    published_path = ingest_form_guide_csv(str(src))

    assert published_path.exists()
    # Expect canonical name
    assert published_path.name == "2025-08-12_gosf_R4.csv"

    # Ensure file lives in UPCOMING_RACES_DIR
    assert published_path.parent == temp_env_dirs["UPCOMING_RACES_DIR"]


def test_canonical_naming_from_filename_fallback(tmp_path, temp_env_dirs):
    # No metadata headers; rely on filename pattern: "Race 4 - GOSF - 2025-08-15.csv"
    src = tmp_path / "Race 4 - GOSF - 2025-08-15.csv"
    src.write_text(FILENAME_ONLY_CONTENT, encoding="utf-8")

    ingest_form_guide_csv = _fresh_ingestor()
    published_path = ingest_form_guide_csv(str(src))

    assert published_path.exists()
    assert published_path.name == "2025-08-15_gosf_R4.csv"


def test_dedup_identical_content_archives_incoming(tmp_path, temp_env_dirs):
    ingest_form_guide_csv = _fresh_ingestor()

    # First ingest
    src1 = tmp_path / "first.csv"
    src1.write_text(FORM_GUIDE_CSV_CONTENT, encoding="utf-8")
    published1 = ingest_form_guide_csv(str(src1))
    assert published1.exists()

    # Second ingest with identical content but different filename
    src2 = tmp_path / "second.csv"
    src2.write_text(FORM_GUIDE_CSV_CONTENT, encoding="utf-8")
    published2 = ingest_form_guide_csv(str(src2))

    # Should return kept existing canonical path (same as published1)
    assert published2 == published1

    # Incoming source should be archived
    archived_dups_dir = temp_env_dirs["ARCHIVE_DIR"] / "duplicates"
    archived_files = list(archived_dups_dir.glob("second_archived-*.csv"))
    assert len(archived_files) == 1


def test_atomic_publish_and_no_tmp_leftovers(tmp_path, temp_env_dirs):
    ingest_form_guide_csv = _fresh_ingestor()

    src = tmp_path / "upload.csv"
    src.write_text(FORM_GUIDE_CSV_CONTENT, encoding="utf-8")
    published = ingest_form_guide_csv(str(src))

    # Ensure published exists and there is no lingering .tmp in the directory
    assert published.exists()
    leftovers = list(published.parent.glob(".*.tmp"))
    assert leftovers == []


def test_manifest_updates_on_publish_and_dedup(tmp_path, temp_env_dirs):
    ingest_form_guide_csv = _fresh_ingestor()

    src = tmp_path / "manifest_a.csv"
    src.write_text(FORM_GUIDE_CSV_CONTENT, encoding="utf-8")
    published = ingest_form_guide_csv(str(src))

    # Manifest should exist and contain entry
    manifest_path = temp_env_dirs["DATA_DIR"] / "ingestion_manifest.json"
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    key = published.name
    assert key in data
    assert Path(data[key]["file_path"]).name == key
    assert len(data[key]["checksum"]) == 64

    # Ingest identical to trigger dedup and ensure manifest still good
    src_dup = tmp_path / "manifest_b.csv"
    src_dup.write_text(FORM_GUIDE_CSV_CONTENT, encoding="utf-8")
    ingest_form_guide_csv(str(src_dup))

    # Re-read manifest, ensure the same key/path remains
    data2 = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert key in data2
    assert Path(data2[key]["file_path"]).name == key
