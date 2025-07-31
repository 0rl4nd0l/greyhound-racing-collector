import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from app import app

@pytest.fixture
def client():
    """Flask test client fixture"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def mock_gpt_enhancer():
    """Mock GPT enhancer for testing"""
    mock = MagicMock()
    mock.gpt_available = True
    mock.enhance_race_prediction.return_value = {
        'race_info': {'venue': 'Test', 'race_number': 1},
        'gpt_race_analysis': {'analysis_confidence': 0.8},
        'tokens_used': 100
    }
    return mock

# Test success path with mocked OpenAI
@patch('app.get_gpt_enhancer')
def test_gpt_enhance_race_success(mock_get_enhancer, mock_gpt_enhancer, client):
    """Test successful GPT race enhancement with mocked OpenAI"""
    mock_get_enhancer.return_value = mock_gpt_enhancer
    
    data = {
        'race_file_path': 'test_race.csv',
        'include_betting_strategy': True,
        'include_pattern_analysis': True
    }
    
    response = client.post('/api/gpt/enhance_race', json=data)
    assert response.status_code == 200
    assert response.json['success'] is True
    assert 'enhancement' in response.json
    assert response.json['tokens_used'] == 100

# Test missing file path returns 400
def test_gpt_enhance_race_missing_file_path(client):
    """Test GPT enhancement with missing file path returns 400"""
    data = {}  # Missing race_file_path
    
    response = client.post('/api/gpt/enhance_race', json=data)
    assert response.status_code == 400
    assert response.json['success'] is False
    assert 'Race file path is required' in response.json['message']

# Test OpenAI failure returns 500
@patch('app.get_gpt_enhancer')
def test_gpt_enhance_race_openai_failure(mock_get_enhancer, client):
    """Test GPT enhancement with OpenAI failure returns 500"""
    mock_get_enhancer.return_value = None  # GPT enhancer not available
    
    data = {
        'race_file_path': 'test_race.csv',
        'include_betting_strategy': True
    }
    
    response = client.post('/api/gpt/enhance_race', json=data)
    assert response.status_code == 500
    assert response.json['success'] is False
    assert 'GPT enhancement not available' in response.json['message']

# Test GPT status endpoint
@patch('os.getenv')
@patch('app.get_gpt_enhancer')
def test_gpt_status_endpoint(mock_get_enhancer, mock_getenv, mock_gpt_enhancer, client):
    """Test GPT status endpoint"""
    mock_getenv.return_value = 'test_api_key'
    mock_get_enhancer.return_value = mock_gpt_enhancer
    
    response = client.get('/api/gpt/status')
    assert response.status_code == 200
    assert response.json['success'] is True
    assert 'status' in response.json
    assert response.json['status']['api_key_configured'] is True

# Test multiple race enhancement
@patch('app.get_gpt_enhancer')
def test_gpt_enhance_multiple_races_missing_list(mock_get_enhancer, mock_gpt_enhancer, client):
    """Test multiple race enhancement with missing race files list"""
    mock_get_enhancer.return_value = mock_gpt_enhancer
    
    data = {}  # Missing race_files
    
    response = client.post('/api/gpt/enhance_multiple', json=data)
    assert response.status_code == 400
    assert response.json['success'] is False
    assert 'Race files list is required' in response.json['message']
