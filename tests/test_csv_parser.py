import pytest
import pandas as pd
from csv_ingestion import FormGuideCsvIngestor, FormGuideCsvIngestionError

import os

@pytest.fixture
def ingestor():
    return FormGuideCsvIngestor()


def test_perfect_file(ingestor):
    headers = ["Dog Name", "PLC", "BOX", "DIST", "DATE", "TRACK"]  # Raw CSV headers
    try:
        ingestor.validate_headers(headers)
    except ValueError:
        pytest.fail("Validation failed on perfect file")


def test_missing_header(ingestor):
    headers = ["Dog Name", "PLC", "BOX", "DATE", "TRACK"]  # Missing 'DIST'
    with pytest.raises(ValueError, match="Missing required column: DIST"):
        ingestor.validate_headers(headers)


def test_validate_csv_schema(ingestor):
    file_path = 'datasets/normal.csv'
    result = ingestor.validate_csv_schema(file_path)
    assert result.is_valid is True


def test_continuation_rows(ingestor):
    df = pd.DataFrame({
        "dog_name": ["Dog A", "", "Dog B", ""],  # Already mapped column names
        "place": [1, 1, 2, 2],
        "box": [1, 1, 2, 2],
        "distance": [500, 500, 500, 500],
        "date": ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02"],
        "track": ["TrackA", "TrackA", "TrackB", "TrackB"]
    })
    processed_data = ingestor.process_form_guide_format(df)
    assert processed_data[1]["dog_name"] == "Dog A"
    assert processed_data[3]["dog_name"] == "Dog B"


def test_process_form_guide_format_with_real_csv(ingestor):
    """Test that process_form_guide_format correctly handles continuation rows from real CSV."""
    # Test with the continuation CSV to ensure blank rows are properly collapsed
    file_path = 'datasets/continuation_rows.csv'
    processed_data, _ = ingestor.ingest_csv(file_path)
    
    # Should have exactly 8 records (3 for Speedy Sam, 2 for Fast Fido, 3 for Quick Quinn)
    assert len(processed_data) == 8
    
    # Check that continuation rows have the correct dog names
    assert processed_data[0]["dog_name"] == "Speedy Sam"
    assert processed_data[1]["dog_name"] == "Speedy Sam"
    assert processed_data[2]["dog_name"] == "Speedy Sam"
    assert processed_data[3]["dog_name"] == "Fast Fido"
    assert processed_data[4]["dog_name"] == "Fast Fido"
    assert processed_data[5]["dog_name"] == "Quick Quinn"
    assert processed_data[6]["dog_name"] == "Quick Quinn"
    assert processed_data[7]["dog_name"] == "Quick Quinn"
    
    # Verify place data is preserved
    assert processed_data[0]["place"] == "1"
    assert processed_data[1]["place"] == "2"
    assert processed_data[2]["place"] == "3"


def test_validate_real_race_csv(ingestor):
    """Test that validate_csv_schema works with a real race CSV file."""
    file_path = 'tmp_testing/Race 1 - AP_K - 01 July 2025.csv'
    if os.path.exists(file_path):
        result = ingestor.validate_csv_schema(file_path)
        assert result.is_valid is True
        # Check that it found the Dog Name column
        assert "Dog Name" in result.available_columns
    else:
        # Skip test if file doesn't exist
        print(f"Skipping test - file {file_path} not found")


def test_extra_unexpected_columns(ingestor):
    headers = ["Dog Name", "PLC", "BOX", "DIST", "DATE", "TRACK", "extra_column"]
    try:
        ingestor.validate_headers(headers)
    except ValueError:
        pytest.fail("Validation failed with extra unexpected columns")

