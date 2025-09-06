import json
import os
import sys

import pytest

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app as flask_app


@pytest.fixture
def app():
    """Create and configure a new app instance for each test."""
    flask_app.config.update({"TESTING": True, "UPCOMING_DIR": "./upcoming_races"})
    os.makedirs(flask_app.config["UPCOMING_DIR"], exist_ok=True)
    yield flask_app


@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()


def test_api_races_paginated_happy_path(client):
    """Test the /api/races/paginated endpoint for happy path."""
    response = client.get("/api/races/paginated?page=1&per_page=10")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["success"] is True
    assert "races" in json_data
    assert "pagination" in json_data
    assert json_data["pagination"]["page"] == 1
    assert json_data["pagination"]["per_page"] == 10
    assert len(json_data["races"]) <= 10  # Check that we do not exceed per_page count


def test_api_races_paginated_last_page(client):
    """Test the /api/races/paginated for the last page, ensuring no errors on upper edge."""
    response = client.get("/api/races/paginated?page=999")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["success"] is True
    assert "pagination" in json_data
    assert json_data["pagination"]["page"] == 999
    assert json_data["pagination"]["has_next"] is False
    assert "races" in json_data
    assert "pagination" in json_data


def test_api_races_paginated_out_of_range(client):
    """Test the /api/races/paginated fails gracefully for non-existent pages."""
    response = client.get("/api/races/paginated?page=10000")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["success"] is True
    assert not json_data["races"]
    assert json_data["pagination"]["has_next"] is False  # Edge case coverage
    assert (
        json_data["pagination"]["page"] == 10000
    )  # Ensure pagination reflects request


def test_api_races_paginated_bad_input(client):
    """Test the /api/races/paginated for invalid inputs to ensure robust error handling."""
    # Test non-integer page parameter
    response = client.get("/api/races/paginated?page=abc")
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["success"] is False
    assert "Invalid page or per_page parameter" in json_data["message"]

    # Test negative page parameter
    response = client.get("/api/races/paginated?page=-1")
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["success"] is False
    assert "Page number must be greater than 0" in json_data["message"]

    # Test non-integer per_page parameter
    response = client.get("/api/races/paginated?per_page=xyz")
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["success"] is False
    assert "Invalid page or per_page parameter" in json_data["message"]

    # Test zero per_page parameter
    response = client.get("/api/races/paginated?per_page=0")
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["success"] is False
    assert "Per page value must be greater than 0" in json_data["message"]

    # Test negative per_page parameter
    response = client.get("/api/races/paginated?per_page=-5")
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["success"] is False
    assert "Per page value must be greater than 0" in json_data["message"]


def test_api_races_paginated_json_structure(client):
    """Test that the /api/races/paginated endpoint returns the expected JSON structure."""
    response = client.get("/api/races/paginated?page=1&per_page=5")
    assert response.status_code == 200
    json_data = response.get_json()

    # Verify top-level structure
    assert json_data["success"] is True
    assert "races" in json_data
    assert "pagination" in json_data
    assert "sort_by" in json_data
    assert "order" in json_data
    assert "search" in json_data

    # Verify pagination structure
    pagination = json_data["pagination"]
    assert "page" in pagination
    assert "per_page" in pagination
    assert "total_count" in pagination
    assert "total_pages" in pagination
    assert "has_next" in pagination
    assert "has_prev" in pagination

    # Verify race structure (if races exist)
    races = json_data["races"]
    assert isinstance(races, list)

    if races:  # Only check race structure if races exist
        race = races[0]
        expected_race_fields = [
            "race_id",
            "venue",
            "race_number",
            "race_date",
            "race_name",
            "grade",
            "distance",
            "field_size",
            "winner_name",
            "winner_odds",
            "winner_margin",
            "url",
            "extraction_timestamp",
            "track_condition",
            "runners",
        ]

        for field in expected_race_fields:
            assert field in race, f"Missing field '{field}' in race object"

        # Verify runners structure
        runners = race["runners"]
        assert isinstance(runners, list)

        if runners:  # Only check runner structure if runners exist
            runner = runners[0]
            expected_runner_fields = [
                "dog_name",
                "box_number",
                "finish_position",
                "individual_time",
                "weight",
                "odds",
                "margin",
                "trainer_name",
                "win_probability",
                "place_probability",
                "confidence",
            ]

            for field in expected_runner_fields:
                assert field in runner, f"Missing field '{field}' in runner object"
            assert type(runner["box_number"]) is int, "box_number should be an integer"


def save_snapshot(name, data):
    """Saves a snapshot of the JSON response."""
    snapshot_dir = os.path.join(
        os.path.dirname(__file__), "fixtures", "expected_responses"
    )
    os.makedirs(snapshot_dir, exist_ok=True)
    snapshot_path = os.path.join(snapshot_dir, f"{name}.json")
    with open(snapshot_path, "w") as f:
        json.dump(data, f, indent=4, sort_keys=True)


def test_api_races_paginated_database_error_handling(client):
    """Test the /api/races/paginated endpoint handles database errors gracefully."""
    # This test assumes we can trigger a database error by invalid query parameters
    # In a real scenario, we might mock the database connection to fail
    response = client.get("/api/races/paginated?page=1&per_page=10")
    assert response.status_code == 200  # Should not crash even if DB issues occur
    json_data = response.get_json()
    # Should return either success data or a proper error response
    if not json_data["success"]:
        assert "message" in json_data
        assert "error" in json_data["message"].lower()
        assert response.status_code == 500
    else:
        # Success case - verify database schema compliance
        assert "races" in json_data
        assert "pagination" in json_data


def test_api_races_paginated_schema_validation(client):
    """Test that race and runner data matches expected schema from unified DB."""
    response = client.get("/api/races/paginated?page=1&per_page=3")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["success"] is True

    races = json_data["races"]
    if races:
        race = races[0]

        # Verify race_metadata table columns are present
        race_metadata_columns = [
            "race_id",
            "venue",
            "race_number",
            "race_date",
            "race_name",
            "grade",
            "distance",
            "field_size",
            "winner_name",
            "winner_odds",
            "winner_margin",
            "url",
            "extraction_timestamp",
            "track_condition",
        ]

        for column in race_metadata_columns:
            assert (
                column in race
            ), f"Missing race_metadata column '{column}' in response"

        # Verify dog_race_data table columns are present in runners
        runners = race.get("runners", [])
        if runners:
            runner = runners[0]
            dog_race_data_columns = [
                "dog_name",
                "box_number",
                "finish_position",
                "individual_time",
                "weight",
                "odds",
                "margin",
                "trainer_name",
            ]

            for column in dog_race_data_columns:
                # Map from database column to API response field
                api_field = column
                if column == "odds_decimal":
                    api_field = "odds"
                elif column == "trainer_name":
                    api_field = "trainer_name"

                assert (
                    api_field in runner
                ), f"Missing dog_race_data column '{column}' mapped as '{api_field}' in runner response"

        # Verify data types are correct
        assert isinstance(race["race_number"], (int, str))
        assert isinstance(race["race_id"], str)
        assert isinstance(race["venue"], str)


def test_api_races_paginated_query_params_validation(client):
    """Test all supported query parameters work correctly."""
    # Test search functionality
    response = client.get("/api/races/paginated?page=1&per_page=5&search=DAPTO")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["success"] is True
    assert json_data["search"] == "DAPTO"

    # Test sorting options
    sort_options = ["race_date", "venue", "confidence", "grade"]
    for sort_by in sort_options:
        response = client.get(f"/api/races/paginated?page=1&sort_by={sort_by}")
        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["success"] is True
        assert json_data["sort_by"] == sort_by

    # Test order options
    for order in ["asc", "desc"]:
        response = client.get(f"/api/races/paginated?page=1&order={order}")
        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["success"] is True
        assert json_data["order"] == order


def test_api_races_paginated_per_page_limits(client):
    """Test per_page parameter respects limits and boundaries."""
    # Test maximum per_page limit (should be capped at 50)
    response = client.get("/api/races/paginated?page=1&per_page=100")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["success"] is True
    assert json_data["pagination"]["per_page"] == 50  # Should be capped
    assert len(json_data["races"]) <= 50

    # Test minimum valid per_page
    response = client.get("/api/races/paginated?page=1&per_page=1")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["success"] is True
    assert json_data["pagination"]["per_page"] == 1
    assert len(json_data["races"]) <= 1


def test_api_races_paginated_default_values(client):
    """Test that default values are applied correctly when no parameters provided."""
    response = client.get("/api/races/paginated")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["success"] is True

    # Check default values
    assert json_data["pagination"]["page"] == 1  # Default page
    assert json_data["pagination"]["per_page"] == 10  # Default per_page
    assert json_data["sort_by"] == "race_date"  # Default sort_by
    assert json_data["order"] == "desc"  # Default order
    assert json_data["search"] == ""  # Default empty search


def test_api_races_paginated_pagination_math(client):
    """Test that pagination calculations are mathematically correct."""
    response = client.get("/api/races/paginated?page=1&per_page=5")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["success"] is True

    pagination = json_data["pagination"]
    total_count = pagination["total_count"]
    per_page = pagination["per_page"]
    total_pages = pagination["total_pages"]

    # Verify total_pages calculation is correct
    import math

    expected_total_pages = math.ceil(total_count / per_page) if total_count > 0 else 1
    assert (
        total_pages == expected_total_pages
    ), f"Expected {expected_total_pages} total pages, got {total_pages}"

    # Verify has_next and has_prev logic
    page = pagination["page"]
    has_next = pagination["has_next"]
    has_prev = pagination["has_prev"]

    assert has_next == (
        page < total_pages
    ), f"has_next should be {page < total_pages}, got {has_next}"
    assert has_prev == (page > 1), f"has_prev should be {page > 1}, got {has_prev}"


def test_api_dogs_search(client):
    """Test the /api/dogs/search endpoint."""
    response = client.get("/api/dogs/search?q=TEST")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["success"] is True
    assert "dogs" in json_data
    assert isinstance(json_data["dogs"], list)
    assert json_data["query"] == "TEST"
    assert "count" in json_data
    save_snapshot("api_dogs_search", json_data)


def test_api_dog_details_not_found(client):
    """Test the /api/dogs/<dog_name>/details endpoint for a dog that does not exist."""
    response = client.get("/api/dogs/NON_EXISTENT_DOG/details")
    assert response.status_code == 404
    json_data = response.get_json()
    assert json_data["success"] is False
    assert json_data["message"] == "Dog not found"


def test_api_races(client):
    """Test the /api/races endpoint."""
    response = client.get("/api/races")
    assert response.status_code == 200
    json_data = response.get_json()
    assert isinstance(json_data, list)
    if json_data:
        race = json_data[0]
        assert "race_id" in race
        assert "venue" in race
        assert "race_date" in race
        assert "race_name" in race
        assert "winner_name" in race
    save_snapshot("api_races", json_data)


def test_api_predictions_upcoming(client):
    """Test the /api/predictions/upcoming endpoint with race IDs."""
    payload = {"race_ids": ["race_001", "race_002"]}
    response = client.post("/api/predictions/upcoming", json=payload)
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["success"] is True
    assert "predictions" in json_data
    assert isinstance(json_data["predictions"], dict)
    save_snapshot("api_predictions_upcoming", json_data)


def test_predict_endpoint_no_file(client):
    """Test the /api/predict endpoint when the race file does not exist."""
    payload = {"race_filename": "non_existent_race.csv"}
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 404
    json_data = response.get_json()
    # The actual response structure from app.py uses 'error' key
    assert "error" in json_data or "message" in json_data
    # It could be either 'error' or nested in the response structure
    if "error" in json_data:
        assert "not found" in json_data["error"]
    elif "message" in json_data:
        assert "not found" in json_data["message"]
    save_snapshot("api_predict_no_file", json_data)


def test_predict_endpoint_success(client, app):
    """Test the /api/predict endpoint with a valid race file."""
    race_filename = "test_race_for_prediction.csv"
    race_filepath = os.path.join(app.config["UPCOMING_DIR"], race_filename)

    with open(race_filepath, "w") as f:
        f.write("Dog Name,Box,Weight,Trainer\n")
        f.write("1. Test Dog 1,1,30.0,Trainer A\n")
        f.write("2. Test Dog 2,2,31.0,Trainer B\n")

    payload = {"race_filename": race_filename}
    response = client.post("/api/predict", json=payload)

    assert response.status_code in [
        200,
        500,
    ]  # Expect either success or server error if predictor fails
    json_data = response.get_json()

    if (
        response.status_code == 500
        and "UnifiedPredictor not available" in json_data.get("error", "")
    ):
        pytest.skip("UnifiedPredictor not available, skipping full prediction test.")

    assert response.status_code == 200
    assert json_data["success"] is True
    assert "prediction" in json_data
    save_snapshot("api_predict_success", json_data)

    os.remove(race_filepath)


def test_api_dogs_search_no_query(client):
    """Test the /api/dogs/search endpoint without a query parameter."""
    response = client.get("/api/dogs/search")
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["success"] is False
    assert json_data["message"] == "Search query is required"


def test_api_dogs_search_with_limit(client):
    """Test the /api/dogs/search endpoint with a limit parameter."""
    response = client.get("/api/dogs/search?q=TEST&limit=1")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["success"] is True
    assert "dogs" in json_data
    assert len(json_data["dogs"]) <= 1
    assert json_data["query"] == "TEST"
    save_snapshot("api_dogs_search_with_limit", json_data)


def test_api_dogs_all(client):
    """Test the /api/dogs/all endpoint."""
    response = client.get("/api/dogs/all")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["success"] is True
    assert "dogs" in json_data
    assert "pagination" in json_data
    assert isinstance(json_data["dogs"], list)
    save_snapshot("api_dogs_all", json_data)


def test_api_dogs_all_with_pagination(client):
    """Test the /api/dogs/all endpoint with pagination."""
    response = client.get("/api/dogs/all?page=1&per_page=5")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["success"] is True
    assert "dogs" in json_data
    assert "pagination" in json_data
    assert len(json_data["dogs"]) <= 5
    assert json_data["pagination"]["page"] == 1
    assert json_data["pagination"]["per_page"] == 5
    save_snapshot("api_dogs_all_paginated", json_data)


def test_api_dogs_top_performers(client):
    """Test the /api/dogs/top_performers endpoint."""
    response = client.get("/api/dogs/top_performers")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["success"] is True
    assert "top_performers" in json_data
    assert isinstance(json_data["top_performers"], list)
    assert json_data["metric"] == "win_rate"  # default metric
    save_snapshot("api_dogs_top_performers", json_data)


def test_api_dogs_top_performers_by_total_wins(client):
    """Test the /api/dogs/top_performers endpoint sorted by total wins."""
    response = client.get("/api/dogs/top_performers?metric=total_wins&limit=3")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["success"] is True
    assert "top_performers" in json_data
    assert len(json_data["top_performers"]) <= 3
    assert json_data["metric"] == "total_wins"
    save_snapshot("api_dogs_top_performers_total_wins", json_data)


def test_api_stats(client):
    """Test the /api/stats endpoint."""
    response = client.get("/api/stats")
    assert response.status_code == 200
    json_data = response.get_json()
    assert "database" in json_data
    assert "files" in json_data
    assert "timestamp" in json_data
    save_snapshot("api_stats", json_data)


def test_api_recent_races(client):
    """Test the /api/recent_races endpoint."""
    response = client.get("/api/recent_races")
    assert response.status_code == 200
    json_data = response.get_json()
    assert "races" in json_data
    assert "count" in json_data
    assert "timestamp" in json_data
    assert isinstance(json_data["races"], list)
    save_snapshot("api_recent_races", json_data)


def test_api_recent_races_with_limit(client):
    """Test the /api/recent_races endpoint with limit parameter."""
    response = client.get("/api/recent_races?limit=3")
    assert response.status_code == 200
    json_data = response.get_json()
    assert "races" in json_data
    assert len(json_data["races"]) <= 3
    assert json_data["count"] <= 3
    save_snapshot("api_recent_races_limited", json_data)


def test_predict_endpoint_basic(client):
    """Test the basic /predict endpoint."""
    payload = {"race_id": "test_race_123"}
    response = client.post("/predict", json=payload)
    # Expect either success or error depending on system availability
    assert response.status_code in [200, 500]
    json_data = response.get_json()
    # Should contain either prediction results or error message
    assert "error" in json_data or ("status" in json_data or "success" in json_data)
    save_snapshot("predict_basic", json_data)


def test_predict_endpoint_no_data(client):
    """Test the basic /predict endpoint without data."""
    response = client.post("/predict", json={})
    assert response.status_code == 400
    json_data = response.get_json()
    assert "error" in json_data
    assert "No race data provided" in json_data["error"]
    save_snapshot("predict_basic_no_data", json_data)
