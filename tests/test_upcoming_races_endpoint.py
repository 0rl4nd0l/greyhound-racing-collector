"""
Unit tests for the upcoming_races endpoint functionality.

This test module validates that the upcoming_races endpoint:
1. Returns expected HTTP status codes
2. Returns properly formatted JSON responses
3. Handles various edge cases gracefully
4. Does not produce server errors or exceptions
"""

import json
import os
import pytest
import sys
from unittest.mock import patch, MagicMock

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app as flask_app


@pytest.fixture
def app():
    """Create and configure a new app instance for each test."""
    flask_app.config.update({
        "TESTING": True,
        "UPCOMING_DIR": "./upcoming_races",
        "DATABASE_URL": "sqlite:///:memory:",
        "REDIS_URL": "redis://localhost:6380/0"
    })
    
    # Ensure directories exist
    os.makedirs("./upcoming_races", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    yield flask_app


@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()


class TestUpcomingRacesEndpoint:
    """Test suite for the upcoming_races endpoint."""
    
    def test_upcoming_races_page_get_request(self, client):
        """Test GET request to /upcoming returns 200 status."""
        response = client.get("/upcoming")
        assert response.status_code == 200
        assert response.content_type.startswith('text/html')
    
    def test_upcoming_races_api_endpoint(self, client):
        """Test /api/upcoming_races endpoint returns valid JSON."""
        response = client.get("/api/upcoming_races")
        assert response.status_code == 200
        assert response.content_type == 'application/json'
        
        data = response.get_json()
        assert data is not None
        assert isinstance(data, dict)
        assert "success" in data
        assert "races" in data
        assert "count" in data
        assert "timestamp" in data
        
        # Validate data types
        assert isinstance(data["success"], bool)
        assert isinstance(data["races"], list)
        assert isinstance(data["count"], int)
        assert isinstance(data["timestamp"], str)
    
    def test_upcoming_races_csv_alias_endpoint(self, client):
        """Test /api/upcoming_races_csv endpoint (alias) works."""
        response = client.get("/api/upcoming_races_csv")
        assert response.status_code == 200
        assert response.content_type == 'application/json'
        
        data = response.get_json()
        assert data["success"] is True
    
    def test_upcoming_races_with_refresh_parameter(self, client):
        """Test /api/upcoming_races with refresh parameter."""
        response = client.get("/api/upcoming_races?refresh=true")
        assert response.status_code == 200
        
        data = response.get_json()
        assert data["success"] is True
        assert "from_cache" in data
        # When refresh=true, should not be from cache
        assert data["from_cache"] is False
    
    def test_upcoming_races_cache_behavior(self, client):
        """Test that caching works as expected."""
        # First request (should populate cache)
        response1 = client.get("/api/upcoming_races")
        assert response1.status_code == 200
        data1 = response1.get_json()
        
        # Second request (should use cache)
        response2 = client.get("/api/upcoming_races")
        assert response2.status_code == 200
        data2 = response2.get_json()
        
        # Both should be successful
        assert data1["success"] is True
        assert data2["success"] is True
        
        # Should have cache information
        assert "cache_expires_in_minutes" in data1
        assert "cache_expires_in_minutes" in data2
    
    def test_upcoming_races_empty_response_structure(self, client):
        """Test that endpoint returns correct structure even when no races exist."""
        response = client.get("/api/upcoming_races")
        assert response.status_code == 200
        
        data = response.get_json()
        
        # Even with no races, structure should be consistent
        required_fields = ["success", "races", "count", "timestamp", "from_cache", "cache_expires_in_minutes"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Races should be an empty list if none exist
        assert isinstance(data["races"], list)
        assert data["count"] == len(data["races"])
    
    @patch('app.load_upcoming_races')
    def test_upcoming_races_error_handling(self, mock_load_races, client):
        """Test that endpoint handles errors gracefully."""
        # Mock an exception in load_upcoming_races
        mock_load_races.side_effect = Exception("Test error")
        
        response = client.get("/api/upcoming_races")
        
        # Should return 500 status code on error
        assert response.status_code == 500
        
        data = response.get_json()
        assert data is not None
        assert "error" in data or "message" in data
    
    def test_upcoming_races_with_mock_data(self, client):
        """Test endpoint with mocked race data."""
        mock_race_data = [
            {
                "race_name": "Test Race 1",
                "venue": "Test Venue",
                "race_date": "2025-01-01",
                "race_time": "2:00 PM",
                "distance": "500m",
                "grade": "Grade 5",
                "race_number": "1",
                "filename": "test_race_1.csv",
                "race_id": "test_race_1"
            },
            {
                "race_name": "Test Race 2", 
                "venue": "Test Venue 2",
                "race_date": "2025-01-01",
                "race_time": "2:30 PM",
                "distance": "520m",
                "grade": "Grade 4",
                "race_number": "2", 
                "filename": "test_race_2.csv",
                "race_id": "test_race_2"
            }
        ]
        
        with patch('app.load_upcoming_races', return_value=mock_race_data):
            response = client.get("/api/upcoming_races")
            assert response.status_code == 200
            
            data = response.get_json()
            assert data["success"] is True
            assert len(data["races"]) == 2
            assert data["count"] == 2
            
            # Validate race data structure
            race = data["races"][0]
            expected_fields = ["race_name", "venue", "race_date", "race_time", "distance", "grade", "race_number", "filename", "race_id"]
            for field in expected_fields:
                assert field in race, f"Missing race field: {field}"
    
    def test_upcoming_races_performance(self, client):
        """Test that endpoint responds within reasonable time."""
        import time
        
        start_time = time.time()
        response = client.get("/api/upcoming_races")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Should respond within 5 seconds (generous for CI environment)
        assert response_time < 5.0, f"Response took too long: {response_time:.2f} seconds"
        assert response.status_code == 200
    
    def test_upcoming_races_page_template_rendering(self, client):
        """Test that the upcoming races page renders without template errors."""
        response = client.get("/upcoming")
        assert response.status_code == 200
        
        # Check that it's HTML content
        assert response.content_type.startswith('text/html')
        
        # Basic check that it's not an error page
        html_content = response.get_data(as_text=True)
        assert len(html_content) > 100  # Should have substantial content
        assert "Error" not in html_content or "error" not in html_content.lower()[:200]  # No immediate errors visible
    
    def test_concurrent_requests_upcoming_races(self, client):
        """Test that multiple concurrent requests don't cause issues."""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request():
            try:
                response = client.get("/api/upcoming_races")
                results.put(response.status_code)
            except Exception as e:
                results.put(f"Error: {e}")
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check all requests succeeded
        status_codes = []
        while not results.empty():
            status_codes.append(results.get())
        
        assert len(status_codes) == 5
        for status_code in status_codes:
            assert status_code == 200, f"Unexpected status code: {status_code}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
