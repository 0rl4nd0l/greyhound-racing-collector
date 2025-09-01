import os
import tempfile

import pytest

from src.parsers.csv_ingestion import CsvIngestion


# Fixtures for different types of CSV content
@pytest.fixture
def valid_csv_content():
    return """Header1,Header2,Header3\nValue1,Value2,Value3\nValue4,Value5,Value6"""


@pytest.fixture
def mixed_delimiter_content():
    return """Header1;Header2;Header3\nValue1;Value2;Value3"""


@pytest.fixture
def continuation_and_bom_content():
    return "\ufeffHeader1,Header2\nValue1,Value2\nValue3,Value4"


@pytest.fixture
def malformed_csv_content():
    return """Header1,Header2\nValue1,Value2,ExtraValue\nValue3,Value4"""


@pytest.fixture
def empty_csv_content():
    return ""


@pytest.fixture
def headers_only_csv_content():
    return "Header1,Header2,Header3"


@pytest.fixture
def csv_with_empty_rows():
    return """Header1,Header2,Header3\nValue1,Value2,Value3\n,,\nValue4,Value5,Value6"""


# Unit tests using fixtures


def test_valid_csv_preprocessing(valid_csv_content):
    """Test preprocessing of valid CSV content"""
    ingestion = CsvIngestion("some_path.csv")
    processed, delimiter = ingestion.pre_process(valid_csv_content)
    assert delimiter == ","
    assert "Header1" in processed
    assert "Value1" in processed


def test_mixed_delimiter_detection(mixed_delimiter_content):
    """Test delimiter detection with semicolons"""
    ingestion = CsvIngestion("some_path.csv")
    processed, delimiter = ingestion.pre_process(mixed_delimiter_content)
    assert delimiter == ";"
    assert "Value2" in processed


def test_bom_removal(continuation_and_bom_content):
    """Test BOM character removal"""
    ingestion = CsvIngestion("some_path.csv")
    processed, delimiter = ingestion.pre_process(continuation_and_bom_content)
    assert delimiter == ","
    assert processed.startswith("Header1")
    assert "\ufeff" not in processed


def test_csv_structure_validation():
    """Test CSV structure validation with temporary files"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("Header1,Header2\nValue1,Value2\nValue3,Value4")
        f.flush()

        try:
            ingestion = CsvIngestion(f.name)
            parsed_race, validation_report = ingestion.parse_csv()

            assert validation_report.is_valid
            assert len(parsed_race.headers) == 2
            assert len(parsed_race.records) == 2
            assert parsed_race.file_hash != ""
        finally:
            os.unlink(f.name)


def test_malformed_csv_quarantine():
    """Test that malformed CSV triggers quarantine"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("Header1,Header2\nValue1,Value2,ExtraValue\nValue3,Value4")
        f.flush()

        try:
            ingestion = CsvIngestion(f.name)
            parsed_race, validation_report = ingestion.parse_csv()

            assert not validation_report.is_valid
            assert len(validation_report.errors) > 0
            assert "Expected 2 columns, got 3" in validation_report.errors[0]
        finally:
            os.unlink(f.name)


def test_empty_csv_handling():
    """Test handling of empty CSV files"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("")
        f.flush()

        try:
            ingestion = CsvIngestion(f.name)
            parsed_race, validation_report = ingestion.parse_csv()

            assert not validation_report.is_valid
            assert "Empty CSV file" in validation_report.errors
        finally:
            os.unlink(f.name)


def test_headers_only_csv():
    """Test CSV with headers but no data"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("Header1,Header2,Header3")
        f.flush()

        try:
            ingestion = CsvIngestion(f.name)
            parsed_race, validation_report = ingestion.parse_csv()

            assert not validation_report.is_valid
            assert "CSV file has no data records" in validation_report.errors
        finally:
            os.unlink(f.name)


def test_csv_with_empty_rows_quarantine():
    """Test that CSV with too many empty cells triggers quarantine"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(
            "Header1,Header2,Header3\nValue1,Value2,Value3\n,,\nValue4,Value5,Value6"
        )
        f.flush()

        try:
            ingestion = CsvIngestion(f.name)
            parsed_race, validation_report = ingestion.parse_csv()

            assert not validation_report.is_valid
            assert any(
                "Empty record found" in error for error in validation_report.errors
            )
        finally:
            os.unlink(f.name)


def test_file_hash_generation():
    """Test that file hash is generated correctly"""
    content = "Header1,Header2\nValue1,Value2"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(content)
        f.flush()

        try:
            ingestion = CsvIngestion(f.name)
            parsed_race, validation_report = ingestion.parse_csv()

            assert len(parsed_race.file_hash) == 64  # SHA-256 hash length
            assert parsed_race.file_path == f.name
        finally:
            os.unlink(f.name)


def test_invisible_characters_detection():
    """Test detection of invisible characters"""
    content_with_invisible = (
        "Header1,Header2\nValue1\x00,Value2"  # Contains null character
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write(content_with_invisible)
        f.flush()

        try:
            ingestion = CsvIngestion(f.name)
            parsed_race, validation_report = ingestion.parse_csv()

            assert len(validation_report.warnings) > 0
            assert (
                "Invisible characters found and removed" in validation_report.warnings
            )
        finally:
            os.unlink(f.name)
