import os
from pathlib import Path
import tempfile
import pytest

from csv_ingestion import FormGuideCsvIngestor, create_ingestor, FormGuideCsvIngestionError


def make_csv(content: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    Path(path).write_text(content, encoding="utf-8")
    return path


def test_prefix_derives_box_and_cleans_name_when_box_missing():
    # CSV without BOX column; dog_name contains numeric prefix
    csv_content = (
        "Dog Name,PLC,DATE,TRACK,DIST\n"
        "1. Alpha Dog,1,2025-08-01,SAN,515\n"
        ",2,2025-07-20,SAN,515\n"  # continuation row for Alpha Dog
        "2. Bravo Runner,3,2025-07-15,SAN,515\n"
    )
    p = make_csv(csv_content)
    try:
        ingestor = create_ingestor("moderate")
        processed, vr = ingestor.ingest_csv(p)
        # Expect two records (only rows with place and date retained)
        assert len(processed) == 3  # both rows for Alpha Dog + Bravo Runner
        # First record
        r1 = processed[0]
        assert r1["dog_name"] == "Alpha Dog"
        assert str(r1.get("box")) == "1"
        # Continuation row inherits same dog and box
        r2 = processed[1]
        assert r2["dog_name"] == "Alpha Dog"
        assert str(r2.get("box")) == "1"
        # Third record
        r3 = processed[2]
        assert r3["dog_name"] == "Bravo Runner"
        assert str(r3.get("box")) == "2"
    finally:
        os.unlink(p)


def test_prefix_respected_even_if_box_column_present():
    # CSV with explicit BOX, but ensure prefix does not overwrite explicit box
    csv_content = (
        "Dog Name,BOX,PLC,DATE,TRACK,DIST\n"
        "1. Alpha Dog,4,1,2025-08-01,SAN,515\n"  # BOX column says 4
    )
    p = make_csv(csv_content)
    try:
        ingestor = create_ingestor("moderate")
        processed, vr = ingestor.ingest_csv(p)
        assert len(processed) == 1
        r = processed[0]
        assert r["dog_name"] == "Alpha Dog"
        # Should keep explicit BOX value, not prefix-derived value
        assert str(r.get("box")) == "4"
    finally:
        os.unlink(p)

