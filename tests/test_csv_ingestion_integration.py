import pytest

from src.parsers.csv_ingestion import CsvIngestion


# Integration test to ensure end-to-end functionality
@pytest.mark.integration
def test_csv_ingestion_end_to_end(tmp_path, sample_csv_content):
    # Setup test CSV file
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(sample_csv_content)

    # Initialize the ingestion
    ingestion = CsvIngestion(str(csv_file))

    # Parse the CSV
    parsed_race, validation_report = ingestion.parse_csv()

    # Assertions for data parsing
    assert not validation_report.errors

    # Assertions for database insertions can go here
    # Assuming functions to insert into DB
    # Assumed function names: insert_race_metadata, insert_dog_race_data

    # Example pseudo code for insertion and assertion
    # insert_race_metadata(parsed_race)
    # assert len(RaceMetadata.query.all()) == 1
    # assert RaceMetadata.query.first().venue == 'Example Venue'

    # Quarantine check for malformed entries
    malformed_content = "Header1,Header2\nBroken,Entry,Value"
    malformed_file = tmp_path / "malformed.csv"
    malformed_file.write_text(malformed_content)
    ingestion_malformed = CsvIngestion(str(malformed_file))

    # This should go to quarantine due to extra value
    _, quarantine_report = ingestion_malformed.parse_csv()
    assert quarantine_report.errors

    # Duplicate processing check
    duplicate_file = tmp_path / "duplicate.csv"
    duplicate_file.write_text(sample_csv_content)
    ingestion_duplicate = CsvIngestion(str(duplicate_file))

    # Run duplicate check
    # Example pseudo-code for duplicate check and assert
    # process_duplicate(duplicate_file)
    # assert len(ProcessedRaceFiles.query.filter_by(file_hash='hash_of_duplicate')) == 1
