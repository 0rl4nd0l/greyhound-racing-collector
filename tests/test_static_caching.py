"""
Tests for static file caching functionality in Flask app.

Tests that static files are served with appropriate cache headers
when the app is not in debug mode.
"""

import pytest
from app import app
from unittest.mock import patch


class TestStaticCaching:
    """Test suite for static file caching"""
    
    def setup_method(self):
        """Set up test client"""
        self.app = app
        self.client = self.app.test_client()
    
    def test_static_css_cache_control_production(self):
        """Test that CSS files have correct cache headers in production mode"""
        # Ensure app is not in debug mode for this test
        with patch.object(self.app, 'debug', False):
            response = self.client.get('/static/css/style.css')
            
            # Assert cache-control header is present and correct
            assert 'Cache-Control' in response.headers
            cache_control = response.headers['Cache-Control']
            
            # Should contain 'public' and 'max-age=31536000' (1 year)
            assert 'public' in cache_control
            assert 'max-age=31536000' in cache_control
    
    def test_static_css_no_cache_debug(self):
        """Test that CSS files don't have long cache headers in debug mode"""
        # Ensure app is in debug mode for this test
        with patch.object(self.app, 'debug', True):
            response = self.client.get('/static/css/style.css')
            
            # In debug mode, we should not have the long-term cache headers
            # The default Flask behavior should apply
            if 'Cache-Control' in response.headers:
                cache_control = response.headers['Cache-Control']
                # Should not have the 1-year max-age
                assert 'max-age=31536000' not in cache_control
    
    def test_static_js_cache_control_production(self):
        """Test that JS files have correct cache headers in production mode"""
        # Ensure app is not in debug mode for this test
        with patch.object(self.app, 'debug', False):
            response = self.client.get('/static/js/main.js')
            
            # Assert cache-control header is present and correct
            if response.status_code == 200:  # Only test if file exists
                assert 'Cache-Control' in response.headers
                cache_control = response.headers['Cache-Control']
                
                # Should contain 'public' and 'max-age=31536000' (1 year)
                assert 'public' in cache_control
                assert 'max-age=31536000' in cache_control
    
    def test_cache_control_value_is_one_year(self):
        """Test that the max-age value equals exactly one year in seconds"""
        # 1 year = 60 seconds * 60 minutes * 24 hours * 365 days = 31536000 seconds
        expected_max_age = 60 * 60 * 24 * 365
        assert expected_max_age == 31536000
        
        # Test with a static file in production mode
        with patch.object(self.app, 'debug', False):
            response = self.client.get('/static/css/style.css')
            
            if response.status_code == 200 and 'Cache-Control' in response.headers:
                cache_control = response.headers['Cache-Control']
                assert f'max-age={expected_max_age}' in cache_control
    
    def test_non_static_routes_not_affected(self):
        """Test that non-static routes are not affected by static caching config"""
        with patch.object(self.app, 'debug', False):
            response = self.client.get('/')
            
            # Home page should not have the same cache headers as static files
            if 'Cache-Control' in response.headers:
                cache_control = response.headers['Cache-Control']
                # Should not have the 1-year max-age for dynamic content
                assert 'max-age=31536000' not in cache_control
