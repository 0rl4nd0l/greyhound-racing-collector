import os
import tempfile
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def temp_csv_files():
    # Create temporary files
    temp_files = []
    try:
        for _ in range(2):
            fd, path = tempfile.mkstemp(suffix='.csv')
            with open(fd, 'w') as f:
                f.write("race_id,dog_name,odds\nexample_id,example_dog,5.0")
            temp_files.append(path)
        yield temp_files
    finally:
        for path in temp_files:
            os.remove(path)

def test_single_race_prediction_by_id(client):
    response = client.post('/api/predict_single_race_enhanced', json={"race_id": "example_id"})
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] is True

def test_single_race_prediction_by_filename(client):
    response = client.post('/api/predict_single_race_enhanced', json={"race_filename": "example.csv"})
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] is True

def test_single_race_prediction_missing_params(client):
    response = client.post('/api/predict_single_race_enhanced', json={})
    assert response.status_code == 400
    data = response.get_json()
    assert data['success'] is False

def test_all_upcoming_races_prediction(client, temp_csv_files):
    response = client.post('/api/predict_all_upcoming_races_enhanced')
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] is True
    assert data['total_races'] == 2
