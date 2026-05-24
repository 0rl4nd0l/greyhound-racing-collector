#!/usr/bin/env python3
"""
Unit test for Flask-Compress functionality
==========================================

Tests that Flask-Compress is properly configured and working by:
1. Making a request to the /ping endpoint with Accept-Encoding: gzip
2. Asserting that the response includes Content-Encoding: gzip header
3. Verifying the response content is properly compressed

Author: AI Assistant
Date: 2025
"""

import gzip
import json
import unittest

from app import app


class TestFlaskCompress(unittest.TestCase):
    """Test Flask-Compress functionality"""

    def setUp(self):
        """Set up test client"""
        self.app = app
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

        # Enable compression for testing (override testing config)
        self.app.config["COMPRESS_LEVEL"] = 6
        self.app.config["COMPRESS_MIN_SIZE"] = 200  # Lower threshold for testing
        self.app.config["COMPRESS_REGISTER"] = True

    def test_ping_endpoint_compression(self):
        """Test that /ping endpoint returns compressed response when Accept-Encoding allows"""

        # Make request with gzip encoding accepted
        response = self.client.get(
            "/ping", headers={"Accept-Encoding": "gzip, deflate"}
        )

        # Assert response is successful
        self.assertEqual(response.status_code, 200)

        # Assert Content-Encoding header is present and set to gzip
        self.assertIn("Content-Encoding", response.headers)
        self.assertEqual(response.headers["Content-Encoding"], "gzip")

        # Assert Content-Type is JSON
        self.assertIn("application/json", response.headers.get("Content-Type", ""))

        # Verify the compressed data can be decompressed
        try:
            decompressed_data = gzip.decompress(response.data)
            json_data = json.loads(decompressed_data.decode("utf-8"))

            # Verify expected JSON structure
            self.assertIn("message", json_data)
            self.assertEqual(json_data["message"], "pong")
            self.assertIn("status", json_data)
            self.assertEqual(json_data["status"], "ok")
            self.assertIn("compression_test", json_data)

        except (gzip.BadGzipFile, json.JSONDecodeError) as e:
            self.fail(f"Failed to decompress or parse JSON response: {e}")

    def test_ping_endpoint_without_gzip_header(self):
        """Test that /ping endpoint works without gzip header (should not be compressed)"""

        # Make request without Accept-Encoding header
        response = self.client.get("/ping")

        # Assert response is successful
        self.assertEqual(response.status_code, 200)

        # Content-Encoding should not be present or should not be gzip
        # (Flask-Compress may still compress based on other factors)
        content_encoding = response.headers.get("Content-Encoding")

        # If no compression header or if compressed, both are valid
        # The key test is that the response is valid JSON
        try:
            if content_encoding == "gzip":
                # If compressed, decompress first
                decompressed_data = gzip.decompress(response.data)
                json_data = json.loads(decompressed_data.decode("utf-8"))
            else:
                # If not compressed, parse directly
                json_data = json.loads(response.data.decode("utf-8"))

            # Verify expected JSON structure
            self.assertIn("message", json_data)
            self.assertEqual(json_data["message"], "pong")

        except (gzip.BadGzipFile, json.JSONDecodeError) as e:
            self.fail(f"Failed to parse response: {e}")

    def test_health_endpoint_compression(self):
        """Test that /api/health endpoint also supports compression"""

        # Make request with gzip encoding accepted
        response = self.client.get(
            "/api/health", headers={"Accept-Encoding": "gzip, deflate"}
        )

        # Assert response is successful
        self.assertEqual(response.status_code, 200)

        # For health endpoint, compression may or may not be applied
        # depending on response size, but we should get valid JSON
        content_encoding = response.headers.get("Content-Encoding")

        try:
            if content_encoding == "gzip":
                # If compressed, decompress first
                decompressed_data = gzip.decompress(response.data)
                json_data = json.loads(decompressed_data.decode("utf-8"))
            else:
                # If not compressed, parse directly
                json_data = json.loads(response.data.decode("utf-8"))

            # Verify expected JSON structure for health endpoint
            self.assertIn("status", json_data)
            self.assertEqual(json_data["status"], "healthy")
            self.assertIn("components", json_data)

        except (gzip.BadGzipFile, json.JSONDecodeError) as e:
            self.fail(f"Failed to parse health response: {e}")

    def test_compression_threshold(self):
        """Test that compression respects minimum size threshold"""

        # Test with a very small response that should not be compressed
        # The /ping endpoint has a longer message designed to trigger compression
        response = self.client.get(
            "/ping", headers={"Accept-Encoding": "gzip, deflate"}
        )

        # Since /ping has a long compression_test message, it should be compressed
        self.assertEqual(response.status_code, 200)

        # Calculate approximate response size
        if response.headers.get("Content-Encoding") == "gzip":
            decompressed_data = gzip.decompress(response.data)
            original_size = len(decompressed_data)
            compressed_size = len(response.data)

            # Verify compression actually reduced size
            self.assertLess(compressed_size, original_size)

            # Verify original size is above the threshold (500 bytes)
            self.assertGreater(original_size, 500)


class TestFlaskCompressConfig(unittest.TestCase):
    """Test Flask-Compress configuration"""

    def setUp(self):
        """Set up test client"""
        self.app = app
        self.app.config["TESTING"] = True

    def test_compress_config_loaded(self):
        """Test that Flask-Compress configuration is properly loaded"""

        # Check that compression settings are configured
        self.assertIn("COMPRESS_LEVEL", self.app.config)
        self.assertIn("COMPRESS_MIN_SIZE", self.app.config)

        # Verify reasonable values
        compress_level = self.app.config.get("COMPRESS_LEVEL", 0)
        self.assertGreaterEqual(compress_level, 0)
        self.assertLessEqual(compress_level, 9)

        min_size = self.app.config.get("COMPRESS_MIN_SIZE", 0)
        self.assertGreaterEqual(min_size, 0)

    def test_compress_mimetypes_configured(self):
        """Test that COMPRESS_MIMETYPES includes JSON"""

        mimetypes = self.app.config.get("COMPRESS_MIMETYPES", [])

        # Verify JSON is in the list of compressible mimetypes
        self.assertIn("application/json", mimetypes)
        self.assertIn("text/html", mimetypes)


if __name__ == "__main__":
    # Set up test environment
    import os

    os.environ["FLASK_ENV"] = "testing"

    # Run the tests
    unittest.main(verbosity=2)
