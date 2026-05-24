from unittest.mock import patch

import pytest

from form_guide_csv_scraper import FormGuideCsvScraper


# Parametrize test cases for filename parsing
@pytest.mark.parametrize(
    "filename, expected_result",
    [
        # Standard format with human-readable date (as requested)
        ("Race 1 - AP_K - 24 July 2025.csv", ("2025-07-24", "AP_K", "1")),
        (
            "Race 6 - CAS - 31 July 2025.csv",
            ("2025-07-31", "CAS", "6"),
        ),  # CASINO -> CAS
        # Alternative compact pattern with ISO date format (as requested)
        ("form_Race_3_RICH_2025-08-05.csv", ("2025-08-05", "RICH", "3")),
        # Additional edge cases
        ("Race 12 - MEA - 1 January 2025.csv", ("2025-01-01", "MEA", "12")),
        ("prefix_Race_9_WOL_2025-12-25.csv", ("2025-12-25", "WOL", "9")),
    ],
)
def test_load_collected_races(filename, expected_result):
    with patch("os.listdir", return_value=[filename]), patch(
        "os.path.exists", return_value=True
    ):

        scraper = FormGuideCsvScraper()
        scraper.collected_races.clear()  # Clear existing races if any
        scraper.existing_files.clear()
        scraper.load_collected_races()

        # Assert the race_id is created correctly
        assert expected_result in scraper.collected_races
