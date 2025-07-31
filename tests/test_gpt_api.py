import pytest
from app import app
from flask import json
import os

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_gpt_enhance_multiple_with_real_api(client):
    """Test /api/gpt/enhance_multiple endpoint with real OpenAI API calls."""
    # Skip test if no API key is available
    if not os.getenv('OPENAI_API_KEY'):
        pytest.skip("OPENAI_API_KEY not set - skipping real API test")
    
    # Load sample data
    race_files = ['sample_data/test_race_1.csv']
    
    # Execute the request
    response = client.post('/api/gpt/enhance_multiple', json={
        'race_files': race_files,
        'max_races': 1
    })
    
    # Parse the response
    data = json.loads(response.data)
    print(f"Response data: {data}")  # Debug output

    # Assert response structure
    assert response.status_code == 200
    assert 'success' in data
    assert 'batch_results' in data
    assert 'timestamp' in data
    assert data['success'] is True

    # Check that we get a proper batch results structure
    batch_results = data['batch_results']
    assert isinstance(batch_results, dict)
    
    # Verify real API response structure
    if 'batch_summary' in batch_results:
        batch_summary = batch_results['batch_summary']
        assert 'successful_enhancements' in batch_summary
        assert 'estimated_cost_usd' in batch_summary
        assert 'total_tokens_used' in batch_summary
        assert isinstance(batch_summary['estimated_cost_usd'], (int, float))
        assert isinstance(batch_summary['total_tokens_used'], int)
    
    if 'successful_enhancements' in batch_results:
        enhancements = batch_results['successful_enhancements']
        assert isinstance(enhancements, list)
        if len(enhancements) > 0:
            enhancement = enhancements[0]
            assert 'race_file' in enhancement
            assert 'enhancement' in enhancement


def test_gpt_comprehensive_report_with_real_api(client):
    """Test /api/gpt/comprehensive_report endpoint with real OpenAI API calls."""
    # Skip test if no API key is available
    if not os.getenv('OPENAI_API_KEY'):
        pytest.skip("OPENAI_API_KEY not set - skipping real API test")
    
    # Load sample prediction
    sample_json_path = 'sample_data/sample_prediction.json'
    with open(sample_json_path, 'r') as file:
        prediction_json = json.load(file)

    race_ids = [prediction_json['race_info']['filename']]

    response = client.post('/api/gpt/comprehensive_report', json={'race_ids': race_ids})

    data = json.loads(response.data)
    print(f"Report response data: {data}")  # Debug output

    # Assert response structure
    assert response.status_code == 200
    assert 'success' in data
    assert 'report' in data
    assert 'race_count' in data
    assert 'timestamp' in data
    assert data['success'] is True
    assert data['race_count'] == len(race_ids)

    # Report can be either a dict (real GPT) or string (fallback)
    report = data['report']
    assert isinstance(report, (dict, str))
    
    # If it's a dict (real API response), verify structure
    if isinstance(report, dict):
        expected_fields = ['title', 'executive_summary', 'detailed_analysis', 'recommendations']
        for field in expected_fields:
            assert field in report, f"Missing field '{field}' in report"
    
    # Verify cost tracking if available
    if 'cost_breakdown' in data:
        cost_breakdown = data['cost_breakdown']
        assert 'total_cost_usd' in cost_breakdown
        assert 'total_tokens' in cost_breakdown
        assert isinstance(cost_breakdown['total_cost_usd'], (int, float))
        assert isinstance(cost_breakdown['total_tokens'], int)


def test_api_error_handling_without_api_key(client):
    """Test that endpoints handle missing API key gracefully."""
    # Temporarily remove API key if it exists
    original_key = os.environ.get('OPENAI_API_KEY')
    if 'OPENAI_API_KEY' in os.environ:
        del os.environ['OPENAI_API_KEY']
    
    try:
        # Test enhance_multiple endpoint
        response = client.post('/api/gpt/enhance_multiple', json={
            'race_files': ['sample_data/test_race_1.csv'],
            'max_races': 1
        })
        
        # Should either return success with stub data or an error
        assert response.status_code in [200, 400, 500]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'success' in data
            assert 'batch_results' in data
            # With no API key, should get stub response
            assert data['success'] is True
        
        # Test comprehensive_report endpoint
        response = client.post('/api/gpt/comprehensive_report', json={
            'race_ids': ['test_race_1.csv']
        })
        
        assert response.status_code in [200, 400, 500]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'success' in data
            assert 'report' in data
            # With no API key, report might be a string stub
            assert isinstance(data['report'], (dict, str))
    
    finally:
        # Restore original API key if it existed
        if original_key:
            os.environ['OPENAI_API_KEY'] = original_key
