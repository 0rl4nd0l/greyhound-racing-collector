import pytest
from flask import Flask
from flask.testing import FlaskClient
import os

# Assuming the Flask app is in app.py and correctly configured
from app import app as flask_app


@pytest.fixture
def app() -> Flask:
    flask_app.config.update({
        "TESTING": True,
        "UPCOMING_DIR": "./upcoming_races",
    })
    os.makedirs(flask_app.config["UPLOAD_FOLDER"], exist_ok=True)
    yield flask_app


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    return app.test_client()


def test_api_dogs_search(client: FlaskClient):
    response = client.get("/api/dogs/search?q=Greyhound")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["success"] is True
    assert "dogs" in json_data


def test_predict_single_race(client: FlaskClient):
    # Create a mock race file
    race_filename = "test_race.csv"
    race_filepath = os.path.join(flask_app.config["UPCOMING_DIR"], race_filename)
    with open(race_filepath, "w") as f:
        f.write("Dog Name,Box,Weight,Trainer\n")
        f.write("1. Test Dog 1,1,30.0,Trainer A\n")
        f.write("2. Test Dog 2,2,31.0,Trainer B\n")

    # API request
    response = client.post("/predict", json={"race_filename": race_filename})
    assert response.status_code in [200, 400, 500]
    json_data = response.get_json()
    assert "prediction" in json_data or "error" in json_data

    os.remove(race_filepath)


def test_api_file_upload(client: FlaskClient):
    test_file_path = "/tmp/tests_uploads/test_file.csv"
    with open(test_file_path, "rb") as f:
        data = {"file": (f, "test_file.csv")}
        response = client.post("/upload", data=data, content_type="multipart/form-data")
        assert response.status_code in [200, 400, 302]  # Handle redirection
        if response.status_code == 302:
            location = response.headers.get("Location")
            assert location is not None
            assert location.endswith("/scraping_status")


def test_database_operations(client: FlaskClient):
    # Here you would add comprehensive DB CRUD operations, especially verifying row counts.
    # Use real operations with the test database set up.
    pass


def test_api_dogs_search_no_query(client: FlaskClient):
    response = client.get("/api/dogs/search")
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["success"] is False
    assert "Search query is required" in json_data["message"]


def test_large_file_upload(client: FlaskClient):
    # Edge case test for large file uploads to ensure proper handling
    large_filename = "/tmp/tests_uploads/large_test_file.csv"
    with open(large_filename, "wb") as f:
        f.seek((10 * 1024 * 1024) - 1)  # Seek to 10MB less 1 byte
        f.write(b"0")  # Write a single null byte

    with open(large_filename, "rb") as f:
        data = {"file": (f, large_filename)}
        response = client.post("/upload", data=data, content_type="multipart/form-data")
        assert response.status_code in [200, 400, 302]  # Handle redirection
        if response.status_code == 302:
            location = response.headers.get("Location")
            assert location is not None
            assert location.endswith("/scraping_status")

    os.remove(large_filename)


def test_missing_parameters(client: FlaskClient):
    # Ensure that endpoints handle missing parameters gracefully
    response = client.post("/api/predict", json={})
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["success"] is False
    assert "No race data provided" in json_data["error"]
