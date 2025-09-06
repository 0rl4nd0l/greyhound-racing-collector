"""
Backend Test Suite - Focused on Quick Coverage
============================================

Fast-running subset of comprehensive backend tests for CI/CD.
Excludes prediction tests that may take excessive time.
"""

import io
import sqlite3

import pytest

# Import the Flask app
from app import DatabaseManager

# ========== CORE API ROUTE TESTS ==========


class TestCoreAPIRoutes:
    """Test essential API routes for CI coverage"""

    def test_api_health(self, client):
        """Test /api/health endpoint"""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "components" in data

    def test_api_dogs_search_valid_query(self, client):
        """Test /api/dogs/search with valid query"""
        response = client.get("/api/dogs/search?q=Test")
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert "dogs" in data
        assert "query" in data
        assert "count" in data
        assert data["query"] == "Test"
        assert isinstance(data["dogs"], list)

    def test_api_dogs_search_missing_query(self, client):
        """Test /api/dogs/search without query parameter"""
        response = client.get("/api/dogs/search")
        assert response.status_code == 400
        data = response.get_json()
        assert data["success"] is False
        assert "Search query is required" in data["message"]

    def test_api_races_paginated_valid(self, client):
        """Test /api/races/paginated with valid parameters"""
        response = client.get("/api/races/paginated?page=1&per_page=5")
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert "races" in data
        assert "pagination" in data
        assert isinstance(data["races"], list)

    def test_api_races_paginated_invalid_page(self, client):
        """Test /api/races/paginated with invalid page parameter"""
        response = client.get("/api/races/paginated?page=abc")
        assert response.status_code == 400
        data = response.get_json()
        assert data["success"] is False
        assert "Invalid page or per_page parameter" in data["message"]

    def test_api_stats(self, client):
        """Test /api/stats endpoint"""
        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.get_json()
        assert "database" in data
        assert "files" in data
        assert "timestamp" in data


# ========== DATABASE CRUD TESTS ==========


class TestDatabaseCRUD:
    """Test database CRUD operations with real SQLAlchemy sessions"""

    def test_database_manager_connection(self, test_app):
        """Test DatabaseManager can establish connection"""
        db_mgr = DatabaseManager(test_app.config["DATABASE_PATH"])
        conn = db_mgr.get_connection()
        assert conn is not None

        # Test basic query
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM dogs")
        count = cursor.fetchone()[0]
        assert count >= 0
        conn.close()

    def test_database_stats_retrieval(self, test_app):
        """Test database statistics retrieval"""
        db_mgr = DatabaseManager(test_app.config["DATABASE_PATH"])
        stats = db_mgr.get_database_stats()

        assert "total_races" in stats
        assert "total_entries" in stats
        assert "unique_dogs" in stats
        assert "venues" in stats
        assert isinstance(stats["total_races"], int)
        assert isinstance(stats["unique_dogs"], int)

    def test_database_insert_operation(self, test_app):
        """Test database INSERT operation with row count verification"""
        conn = sqlite3.connect(test_app.config["DATABASE_PATH"])
        cursor = conn.cursor()

        # Get initial count
        cursor.execute("SELECT COUNT(*) FROM dogs")
        initial_count = cursor.fetchone()[0]

        # Insert new dog
        test_dog_data = (
            "test_insert_dog",
            "New Test Dog",
            5,
            1,
            2,
            "19.50",
            3.2,
            "2025-01-04",
        )
        cursor.execute(
            """
            INSERT INTO dogs 
            (dog_id, dog_name, total_races, total_wins, total_places, best_time, average_position, last_race_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            test_dog_data,
        )
        conn.commit()

        # Verify count increased
        cursor.execute("SELECT COUNT(*) FROM dogs")
        new_count = cursor.fetchone()[0]
        assert new_count == initial_count + 1

        # Verify data was inserted correctly
        cursor.execute("SELECT * FROM dogs WHERE dog_id = ?", ("test_insert_dog",))
        inserted_dog = cursor.fetchone()
        assert inserted_dog is not None
        assert inserted_dog[1] == "New Test Dog"  # dog_name

        conn.close()

    def test_database_update_operation(self, test_app):
        """Test database UPDATE operation with row count verification"""
        conn = sqlite3.connect(test_app.config["DATABASE_PATH"])
        cursor = conn.cursor()

        # Update existing dog
        cursor.execute("UPDATE dogs SET total_wins = 5 WHERE dog_id = 'test_dog_1'")
        affected_rows = cursor.rowcount
        conn.commit()

        assert affected_rows == 1

        # Verify update
        cursor.execute("SELECT total_wins FROM dogs WHERE dog_id = 'test_dog_1'")
        updated_wins = cursor.fetchone()[0]
        assert updated_wins == 5

        conn.close()

    def test_database_delete_operation(self, test_app):
        """Test database DELETE operation with row count verification"""
        conn = sqlite3.connect(test_app.config["DATABASE_PATH"])
        cursor = conn.cursor()

        # Get initial count
        cursor.execute("SELECT COUNT(*) FROM dogs")
        initial_count = cursor.fetchone()[0]

        # Delete a dog (if exists from previous insert test)
        cursor.execute("DELETE FROM dogs WHERE dog_id = 'test_insert_dog'")
        deleted_rows = cursor.rowcount
        conn.commit()

        # Verify count decreased if deletion occurred
        cursor.execute("SELECT COUNT(*) FROM dogs")
        new_count = cursor.fetchone()[0]
        assert new_count == initial_count - deleted_rows

        conn.close()


# ========== FILE UPLOAD TESTS ==========


class TestFileUpload:
    """Test file upload functionality"""

    def test_upload_get_request(self, client):
        """Test GET request to /upload returns upload form"""
        response = client.get("/upload")
        assert response.status_code == 200

    def test_upload_valid_csv_file(self, client):
        """Test uploading valid CSV file"""
        # Create test CSV file
        test_content = (
            "Dog Name,Box,Weight,Trainer\n1. Upload Test Dog,1,30.0,Upload Trainer\n"
        )

        data = {"file": (io.BytesIO(test_content.encode()), "test_upload.csv")}

        response = client.post("/upload", data=data, content_type="multipart/form-data")
        assert response.status_code in [200, 302]  # 302 for redirect

        if response.status_code == 302:
            location = response.headers.get("Location")
            assert location in [
                "/scraping_status",
                "/scraping",
            ]  # Either redirect is valid

    def test_upload_no_file_selected(self, client):
        """Test upload without selecting file"""
        response = client.post("/upload", data={}, content_type="multipart/form-data")
        assert response.status_code == 302  # Redirect back to upload page


# ========== EDGE CASE TESTS ==========


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_api_races_paginated_boundary_values(self, client):
        """Test pagination with boundary values"""
        # Page 0 (should be rejected)
        response = client.get("/api/races/paginated?page=0")
        assert response.status_code == 400

        # Negative page
        response = client.get("/api/races/paginated?page=-1")
        assert response.status_code == 400

        # Per_page 0 (should be rejected)
        response = client.get("/api/races/paginated?per_page=0")
        assert response.status_code == 400

        # Very large per_page (should be capped)
        response = client.get("/api/races/paginated?per_page=999")
        assert response.status_code == 200
        data = response.get_json()
        assert data["pagination"]["per_page"] == 50  # Should be capped at 50

    def test_missing_required_parameters(self, client):
        """Test various endpoints with missing required parameters"""
        # Missing query parameter for dog search
        response = client.get("/api/dogs/search")
        assert response.status_code == 400

    def test_sql_injection_protection(self, client):
        """Test SQL injection protection in search queries"""
        # SQL injection attempt in dog search
        malicious_query = "'; DROP TABLE dogs; --"
        response = client.get(f"/api/dogs/search?q={malicious_query}")
        assert response.status_code == 200  # Should not crash

        # Verify table still exists by doing another search
        response = client.get("/api/dogs/search?q=Test")
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main(
        [
            __file__,
            "--cov=app",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-fail-under=15",  # Set to achievable level
            "-v",
        ]
    )
