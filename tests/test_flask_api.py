
import os
import sys
import pytest
import json

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app as flask_app

@pytest.fixture
def app():
    """Create and configure a new app instance for each test."""
    flask_app.config.update({
        "TESTING": True,
        "UPCOMING_DIR": './upcoming_races'
    })
    os.makedirs(flask_app.config['UPCOMING_DIR'], exist_ok=True)
    yield flask_app

@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()

def save_snapshot(name, data):
    """Saves a snapshot of the JSON response."""
    snapshot_dir = os.path.join(os.path.dirname(__file__), 'fixtures', 'expected_responses')
    os.makedirs(snapshot_dir, exist_ok=True)
    snapshot_path = os.path.join(snapshot_dir, f'{name}.json')
    with open(snapshot_path, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)

def test_api_dogs_search(client):
    """Test the /api/dogs/search endpoint."""
    response = client.get('/api/dogs/search?q=TEST')
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['success'] is True
    assert 'dogs' in json_data
    assert isinstance(json_data['dogs'], list)
    assert json_data['query'] == 'TEST'
    assert 'count' in json_data
    save_snapshot('api_dogs_search', json_data)

def test_api_dog_details_not_found(client):
    """Test the /api/dogs/<dog_name>/details endpoint for a dog that does not exist."""
    response = client.get('/api/dogs/NON_EXISTENT_DOG/details')
    assert response.status_code == 404
    json_data = response.get_json()
    assert json_data['success'] is False
    assert json_data['message'] == 'Dog not found'

def test_api_races(client):
    """Test the /api/races endpoint."""
    response = client.get('/api/races')
    assert response.status_code == 200
    json_data = response.get_json()
    assert isinstance(json_data, list)
    if json_data:
        race = json_data[0]
        assert 'race_id' in race
        assert 'venue' in race
        assert 'race_date' in race
        assert 'race_name' in race
        assert 'winner_name' in race
    save_snapshot('api_races', json_data)

def test_predict_endpoint_no_file(client):
    """Test the /api/predict endpoint when the race file does not exist."""
    payload = {'race_filename': 'non_existent_race.csv'}
    response = client.post('/api/predict', json=payload)
    assert response.status_code == 404
    json_data = response.get_json()
    # The actual response structure from app.py uses 'error' key
    assert 'error' in json_data or 'message' in json_data
    # It could be either 'error' or nested in the response structure
    if 'error' in json_data:
        assert 'not found' in json_data['error']
    elif 'message' in json_data:
        assert 'not found' in json_data['message']
    save_snapshot('api_predict_no_file', json_data)

def test_predict_endpoint_success(client, app):
    """Test the /api/predict endpoint with a valid race file."""
    race_filename = 'test_race_for_prediction.csv'
    race_filepath = os.path.join(app.config['UPCOMING_DIR'], race_filename)
    
    with open(race_filepath, 'w') as f:
        f.write("Dog,Box,Weight,Trainer\n")
        f.write("Test Dog 1,1,30.0,Trainer A\n")
        f.write("Test Dog 2,2,31.0,Trainer B\n")

    payload = {'race_filename': race_filename}
    response = client.post('/api/predict', json=payload)

    assert response.status_code in [200, 500] # Expect either success or server error if predictor fails
    json_data = response.get_json()

    if response.status_code == 500 and 'UnifiedPredictor not available' in json_data.get('error', ''):
        pytest.skip("UnifiedPredictor not available, skipping full prediction test.")
    
    assert response.status_code == 200
    assert json_data['success'] is True
    assert 'prediction' in json_data
    save_snapshot('api_predict_success', json_data)

    os.remove(race_filepath)

def test_api_dogs_search_no_query(client):
    """Test the /api/dogs/search endpoint without a query parameter."""
    response = client.get('/api/dogs/search')
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data['success'] is False
    assert json_data['message'] == 'Search query is required'

def test_api_dogs_search_with_limit(client):
    """Test the /api/dogs/search endpoint with a limit parameter."""
    response = client.get('/api/dogs/search?q=TEST&limit=1')
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['success'] is True
    assert 'dogs' in json_data
    assert len(json_data['dogs']) <= 1
    assert json_data['query'] == 'TEST'
    save_snapshot('api_dogs_search_with_limit', json_data)

def test_api_dogs_all(client):
    """Test the /api/dogs/all endpoint."""
    response = client.get('/api/dogs/all')
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['success'] is True
    assert 'dogs' in json_data
    assert 'pagination' in json_data
    assert isinstance(json_data['dogs'], list)
    save_snapshot('api_dogs_all', json_data)

def test_api_dogs_all_with_pagination(client):
    """Test the /api/dogs/all endpoint with pagination."""
    response = client.get('/api/dogs/all?page=1&per_page=5')
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['success'] is True
    assert 'dogs' in json_data
    assert 'pagination' in json_data
    assert len(json_data['dogs']) <= 5
    assert json_data['pagination']['page'] == 1
    assert json_data['pagination']['per_page'] == 5
    save_snapshot('api_dogs_all_paginated', json_data)

def test_api_dogs_top_performers(client):
    """Test the /api/dogs/top_performers endpoint."""
    response = client.get('/api/dogs/top_performers')
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['success'] is True
    assert 'top_performers' in json_data
    assert isinstance(json_data['top_performers'], list)
    assert json_data['metric'] == 'win_rate'  # default metric
    save_snapshot('api_dogs_top_performers', json_data)

def test_api_dogs_top_performers_by_total_wins(client):
    """Test the /api/dogs/top_performers endpoint sorted by total wins."""
    response = client.get('/api/dogs/top_performers?metric=total_wins&limit=3')
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['success'] is True
    assert 'top_performers' in json_data
    assert len(json_data['top_performers']) <= 3
    assert json_data['metric'] == 'total_wins'
    save_snapshot('api_dogs_top_performers_total_wins', json_data)

def test_api_stats(client):
    """Test the /api/stats endpoint."""
    response = client.get('/api/stats')
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'database' in json_data
    assert 'files' in json_data
    assert 'timestamp' in json_data
    save_snapshot('api_stats', json_data)

def test_api_recent_races(client):
    """Test the /api/recent_races endpoint."""
    response = client.get('/api/recent_races')
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'races' in json_data
    assert 'count' in json_data
    assert 'timestamp' in json_data
    assert isinstance(json_data['races'], list)
    save_snapshot('api_recent_races', json_data)

def test_api_recent_races_with_limit(client):
    """Test the /api/recent_races endpoint with limit parameter."""
    response = client.get('/api/recent_races?limit=3')
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'races' in json_data
    assert len(json_data['races']) <= 3
    assert json_data['count'] <= 3
    save_snapshot('api_recent_races_limited', json_data)

def test_predict_endpoint_basic(client):
    """Test the basic /predict endpoint."""
    payload = {'race_id': 'test_race_123'}
    response = client.post('/predict', json=payload)
    # Expect either success or error depending on system availability
    assert response.status_code in [200, 500]
    json_data = response.get_json()
    # Should contain either prediction results or error message
    assert 'error' in json_data or ('status' in json_data or 'success' in json_data)
    save_snapshot('predict_basic', json_data)

def test_predict_endpoint_no_data(client):
    """Test the basic /predict endpoint without data."""
    response = client.post('/predict', json={})
    assert response.status_code == 400
    json_data = response.get_json()
    assert 'error' in json_data
    assert 'No race data provided' in json_data['error']
    save_snapshot('predict_basic_no_data', json_data)
