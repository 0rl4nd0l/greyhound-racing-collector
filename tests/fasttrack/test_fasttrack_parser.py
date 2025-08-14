import os

import pytest
from bs4 import BeautifulSoup

# This assumes the parser functions are in src/collectors/fasttrack_scraper.py
# You might need to adjust the import path
from src.collectors.fasttrack_scraper import FastTrackScraper


# Helper function to load HTML samples
def load_html_sample(filename):
    # Correct the path to be relative to the 'tests' directory
    base_dir = os.path.dirname(
        os.path.dirname(__file__)
    )  # This should be the 'tests' directory
    sample_path = os.path.join(base_dir, "..", "samples", "fasttrack_raw", filename)
    with open(sample_path, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def scraper():
    """Pytest fixture to initialize the scraper."""
    return FastTrackScraper()


@pytest.fixture
def race_result_html():
    """Fixture to load the race result HTML sample."""
    return load_html_sample("race_result_1186391057.html")


@pytest.fixture
def dog_profile_html():
    """Fixture to load the dog profile HTML sample."""
    # Note: The provided dog profile sample is a 403 error page.
    # A more meaningful test would require a valid dog profile page.
    return load_html_sample("dog_890320106.html")


def test_parse_race_result_structure(scraper, race_result_html):
    """Test that the parsed race result has the expected top-level keys."""
    soup = BeautifulSoup(race_result_html, "html.parser")
    parsed_data = scraper._parse_race(soup)

    assert isinstance(parsed_data, dict)
    expected_keys = [
        "race_id",
        "race_name",
        "venue",
        "race_date",
        "distance",
        "grade",
        "track_condition",
        "results",
    ]
    for key in expected_keys:
        assert key in parsed_data


@pytest.mark.parametrize(
    "dog_result_index, expected_fields",
    [
        (
            0,
            {
                "dog_name": str,
                "box_number": int,
                "finish_position": (int, type(None)),
                "race_time": (float, type(None)),
            },
        ),
        (
            1,
            {
                "dog_name": str,
                "box_number": int,
                "finish_position": (int, type(None)),
                "race_time": (float, type(None)),
            },
        ),
        # Add more dogs to test if needed
    ],
)
def test_parse_race_dog_performance_fields(
    scraper, race_result_html, dog_result_index, expected_fields
):
    """Parametric test for completeness and type correctness of dog performance fields."""
    soup = BeautifulSoup(race_result_html, "html.parser")
    parsed_data = scraper._parse_race(soup)

    assert "results" in parsed_data
    dog_results = parsed_data["results"]
    assert len(dog_results) > dog_result_index

    dog_data = dog_results[dog_result_index]
    for field, expected_type in expected_fields.items():
        assert field in dog_data
        assert isinstance(
            dog_data[field], expected_type
        ), f"Field {field} has type {type(dog_data[field])} but expected {expected_type}"


def test_parse_dog_profile_structure(scraper, dog_profile_html):
    """Test that the parsed dog profile has the expected top-level keys."""
    soup = BeautifulSoup(dog_profile_html, "html.parser")
    parsed_data = scraper._parse_dog(soup)

    assert isinstance(parsed_data, dict)

    # Check if this is a 403 error page (as the current sample is)
    if "403" in dog_profile_html and "Forbidden" in dog_profile_html:
        # For 403 error pages, we expect an empty dict or minimal data
        # This is the correct behavior for error pages
        assert parsed_data == {} or len(parsed_data) == 0
    else:
        # For valid dog profile pages, we would expect these keys
        expected_keys = ["dog_id", "name"]
        for key in expected_keys:
            assert key in parsed_data
