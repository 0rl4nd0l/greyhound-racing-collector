import os
import tempfile
import pytest
from app import app, UPCOMING_DIR

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def test_race_file():
    """Create a test race file in the upcoming races directory"""
    # Ensure upcoming directory exists
    os.makedirs(UPCOMING_DIR, exist_ok=True)
    
    # Create test CSV file
    test_filename = "example.csv"
    test_filepath = os.path.join(UPCOMING_DIR, test_filename)
    
    # Sample race data with proper greyhound format
    race_data = """Dog Name,Box,Weight,Trainer
1. Test Dog,1,30.0,Test Trainer
2. Another Dog,2,31.5,Another Trainer
3. Third Dog,3,29.8,Third Trainer
"""
    
    with open(test_filepath, 'w') as f:
        f.write(race_data)
    
    try:
        yield test_filename
    finally:
        # Cleanup
        if os.path.exists(test_filepath):
            os.remove(test_filepath)

@pytest.fixture
def temp_csv_files():
    """Create temporary files for testing multiple race predictions"""
    # Ensure upcoming directory exists
    os.makedirs(UPCOMING_DIR, exist_ok=True)
    
    temp_files = []
    try:
        for i in range(2):
            filename = f"test_race_{i+1}.csv"
            filepath = os.path.join(UPCOMING_DIR, filename)
            
            race_data = f"""Dog Name,Box,Weight,Trainer
1. Test Dog {i+1},1,30.0,Test Trainer {i+1}
2. Another Dog {i+1},2,31.5,Another Trainer {i+1}
"""
            
            with open(filepath, 'w') as f:
                f.write(race_data)
            temp_files.append(filepath)
        
        yield temp_files
    finally:
        for filepath in temp_files:
            if os.path.exists(filepath):
                os.remove(filepath)

def test_single_race_prediction_by_id(client):
    response = client.post('/api/predict_single_race_enhanced', json={"race_id": "example_id"})
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] is True

def test_single_race_prediction_by_filename(client, test_race_file):
    response = client.post('/api/predict_single_race_enhanced', json={"race_filename": test_race_file})
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
